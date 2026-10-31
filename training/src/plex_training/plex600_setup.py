"""Read-only Plex 600M inspection and guarded tokenizer preparation CLI.

This module deliberately contains no optimizer, model instantiation, training
execution, automatic downloads, or inference. Only `tokenizer-train` can write,
and its create-only, source-verified pipeline refuses overwrites.
"""

from __future__ import annotations

import argparse
from array import array
from itertools import zip_longest
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from .plex600 import Plex600Config
from .tokenizer import PlexTokenizer, sha256_file, train_tokenizer, _records, _local_file
from tokenizers import Tokenizer

MODEL_FAMILY = "plex-600m-v1"
MODEL_CAPACITY = Plex600Config().vocab_size
MAX_INDEX_BYTES = 32 * 1024 * 1024
MAX_RECORD_TOKENS = 1_048_577
MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024


def _snapshot(path: Path, expected_hash: str, limit: int) -> bytes:
    """Hash the immutable bytes consumers will use, with a hard read bound."""
    if not path.is_file() or path.is_symlink():
        raise ValueError("Expected a regular snapshot input")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Input exceeds the snapshot byte limit")
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("Snapshot SHA mismatch")
    return raw


def _token_snapshot(path: Path, info: dict[str, Any]) -> bytes:
    size = info["tokens"] * 2
    if size > MAX_SNAPSHOT_BYTES:
        raise ValueError("Token split exceeds the 256 MiB snapshot limit")
    raw = _snapshot(path, info["sha256"], size)
    if len(raw) != size:
        raise ValueError("Token snapshot byte count mismatch")
    return raw


def inspect() -> dict[str, Any]:
    """Architecture facts without allocating the 600M model or touching CUDA."""
    config = Plex600Config()
    return {
        "modelFamily": MODEL_FAMILY,
        "initializedWeights": False,
        "trainingAuthorized": False,
        "optimizerUpdates": 0,
        "parameterCount": config.parameter_count(),
        "configuration": config.to_dict(),
        "gpuMemoryVerified": False,
    }


def _read_json(path: Path, *, limit_bytes: int = 1024 * 1024) -> Any:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > limit_bytes:
        raise ValueError("Expected a regular, bounded JSON file")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_verified_index(path: Path, expected_sha256: Any) -> list[Any]:
    """Hash exactly the bounded bytes that are parsed, without a second read."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Missing or invalid index SHA-256; legacy 600M bundle must be rebuilt")
    if not path.is_file() or path.is_symlink():
        raise ValueError("Expected a regular, bounded index file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_INDEX_BYTES + 1)
    if len(raw) > MAX_INDEX_BYTES:
        raise ValueError("Expected a regular, bounded index file")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Packed index SHA mismatch")
    rows = json.loads(raw.decode("utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Packed index must be a nonempty array")
    return rows


def _check_tokenizer_bundle(directory: Path) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """Check SHA, family, config, split lengths, indexes, token ID and EOS framing.

    Does not open training source text, construct the language model or update
    any weights. This is an artifact-format check, not a license/data audit.
    """
    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Tokenizer bundle must be a directory")
    settings = _read_json(root / "tokenizer-config.json")
    model_config = _read_json(root / "model-config.json")
    manifest = _read_json(root / "manifest.json")
    if not isinstance(settings, dict) or not isinstance(model_config, dict) or not isinstance(manifest, dict):
        raise ValueError("Tokenizer metadata must be JSON objects")
    if settings.get("modelFamily") != MODEL_FAMILY or model_config != Plex600Config().to_dict():
        raise ValueError("Tokenizer bundle is not configured for Plex 600M")
    if settings.get("modelVocabularyCapacity") != MODEL_CAPACITY:
        raise ValueError("Invalid Plex 600M vocabulary capacity")
    if sha256_file(root / "model-config.json") != manifest.get("modelConfigSha256"):
        raise ValueError("Model configuration SHA mismatch")
    if sha256_file(root / "source-dataset-manifest.json") != manifest.get("sourceDatasetManifestSha256"):
        raise ValueError("Source manifest SHA mismatch")
    if manifest.get("tokenizerSha256") != settings.get("tokenizerSha256"):
        raise ValueError("Tokenizer SHAs disagree")
    tokenizer = PlexTokenizer.load(root)
    vocab = tokenizer.vocabulary_size
    if not 260 <= vocab <= MODEL_CAPACITY or manifest.get("actualVocabularySize") != vocab:
        raise ValueError("Tokenizer vocabulary capacity mismatch")
    if manifest.get("modelVocabularyCapacity") != MODEL_CAPACITY:
        raise ValueError("Packed manifest is for another model")
    if manifest.get("storageDtype") != "uint16-le":
        raise ValueError("Expected uint16 little-endian packed tokens")

    verified: dict[str, Any] = {}
    verified_indices: dict[str, list[Any]] = {}
    for split in ("train", "validation"):
        info = manifest.get(split)
        if not isinstance(info, dict) or info.get("path") != split + ".tokens.u16le":
            raise ValueError("Unknown or missing packed split path")
        token_path = root / (split + ".tokens.u16le")
        index_path = root / (split + ".index.json")
        if not token_path.is_file() or token_path.is_symlink():
            raise ValueError("Packed token split missing or is a symlink")
        if not isinstance(info.get("sha256"), str) or sha256_file(token_path) != info["sha256"]:
            raise ValueError("Packed token SHA mismatch")
        rows = _read_verified_index(index_path, info.get("indexSha256"))
        if info.get("records") != len(rows) or type(info.get("tokenCount")) is not int:
            raise ValueError("Packed split record count mismatch")
        expected_start = 0
        record_ids: set[str] = set()
        with token_path.open("rb") as stream:
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"recordId", "startToken", "tokenCount"}:
                    raise ValueError("Invalid token index entry")
                start, count = row["startToken"], row["tokenCount"]
                if (not isinstance(row["recordId"], str) or not row["recordId"]
                        or type(start) is not int or start != expected_start
                        or type(count) is not int or not 1 <= count <= MAX_RECORD_TOKENS):
                    raise ValueError("Token index is not contiguous or is out of bounds")
                if row["recordId"] in record_ids:
                    raise ValueError("Duplicate record ID in packed index")
                record_ids.add(row["recordId"])
                packed = stream.read(count * 2)
                if len(packed) != count * 2:
                    raise ValueError("Packed token data ends before index")
                ids = array("H")
                ids.frombytes(packed)
                if sys.byteorder != "little":
                    ids.byteswap()
                if ids[-1] != 3 or any(token < 4 or token >= vocab for token in ids[:-1]):
                    raise ValueError("Packed row is missing EOS or contains invalid/control tokens")
                expected_start += count
            if stream.read(1):
                raise ValueError("Unindexed packed token bytes remain")
        if expected_start != info["tokenCount"] or token_path.stat().st_size != expected_start * 2:
            raise ValueError("Packed token count/byte size mismatch")
        verified[split] = {"records": len(rows), "tokens": expected_start,
                           "sha256": info["sha256"], "indexSha256": info["indexSha256"]}
        verified_indices[split] = rows

    return {
        "modelFamily": MODEL_FAMILY,
        "tokenizerSha256": settings["tokenizerSha256"],
        "actualVocabularySize": vocab,
        "modelVocabularyCapacity": MODEL_CAPACITY,
        "sourceDatasetManifestSha256": manifest["sourceDatasetManifestSha256"],
        "verifiedSplits": verified,
        "modelLoaded": False,
        "optimizerUpdates": 0,
        "trainingAuthorized": False,
    }, verified_indices


def check_tokenizer_bundle(directory: Path) -> dict[str, Any]:
    """Verify bundle integrity without proving its source mapping or rights.

    Index hashes are bound to the bundle manifest. A jointly modified manifest
    and index require independent source review and cannot be detected here.
    """
    report, _ = _check_tokenizer_bundle(directory)
    return report


def check_source_binding(directory: Path, dataset_dir: Path, *,
                         bundle_manifest_sha256: str, source_manifest_sha256: str) -> dict[str, Any]:
    """Read-only comparison against independently supplied manifest identities.

    Does not fit a tokenizer or infer rights/quality from a matching hash.
    Only the explicitly supplied train/validation JSONL files are opened.
    """
    root, dataset = directory.resolve(strict=True), dataset_dir.resolve(strict=True)
    manifest = json.loads(_snapshot(root / "manifest.json", bundle_manifest_sha256, 1024**2))
    source = json.loads(_snapshot(dataset / "manifest.json", source_manifest_sha256, 1024**2))
    if not isinstance(manifest, dict) or not isinstance(source, dict):
        raise ValueError("Expected object manifests")
    report, indices = _check_tokenizer_bundle(root)
    if report["sourceDatasetManifestSha256"] != source_manifest_sha256:
        raise ValueError("Source manifest identity differs from bundle")
    if source.get("schemaVersion") != 1 or source.get("pipelineVersion") not in (
            "p1-14.2", "p2-02.0", "plex600-m5.0", "plex600-owner-authored-pilot-v1"):
        raise ValueError("Unsupported source dataset manifest")
    if source.get("pipelineVersion") == "plex600-m5.0" and (
            source.get("modelFamily") != MODEL_FAMILY
            or source.get("status") != "tokenizer-input-only-not-training-authorization"
            or source.get("trainingAuthorized") is not False):
        raise ValueError("M5 source is not an approved tokenizer-input receipt")
    if source.get("pipelineVersion") == "plex600-owner-authored-pilot-v1":
        from .plex600_owner_pilot import verify as verify_owner_pilot
        if verify_owner_pilot(dataset)["status"] != "verified-owner-authored-tokenizer-ready":
            raise ValueError("Owner-authored source dataset failed independent verification")
    sources = {row["id"]: row for row in source["sources"]}
    tokenizer_raw = _snapshot(root / "tokenizer.json", manifest["tokenizerSha256"], MAX_SNAPSHOT_BYTES)
    tokenizer = PlexTokenizer(Tokenizer.from_str(tokenizer_raw.decode("utf-8")))
    checked = {}
    for split in ("train", "validation"):
        info = report["verifiedSplits"][split]
        pinned = manifest[split]
        if (info["sha256"] != pinned["sha256"] or info["indexSha256"] != pinned["indexSha256"]
                or info["records"] != pinned["records"] or info["tokens"] != pinned["tokenCount"]):
            raise ValueError("Bundle identity changed during source verification")
        raw = _token_snapshot(root / f"{split}.tokens.u16le", info)
        records = _records(_local_file(dataset, split + ".jsonl"),
                           source["summary"][split + "JsonlSha256"],
                           source["summary"][split + "Records"], sources,
                           family_stratified=source["pipelineVersion"] in ("p2-02.0", "plex600-m5.0", "plex600-owner-authored-pilot-v1"))
        for record, index in zip_longest(records, indices[split]):
            if record is None or index is None or record["recordId"] != index["recordId"]:
                raise ValueError("Packed record identity/order differs from source")
            ids = array("H", tokenizer.encode(record["text"]) + [3])
            if sys.byteorder != "little":
                ids.byteswap()
            start, size = index["startToken"] * 2, index["tokenCount"] * 2
            if ids.tobytes() != raw[start:start + size]:
                raise ValueError("Packed record tokens differ from source text")
        checked[split] = info["records"]
    return {"modelFamily": MODEL_FAMILY, "sourceBindingVerified": True,
            "bundleManifestSha256": bundle_manifest_sha256,
            "sourceManifestSha256": source_manifest_sha256, "verifiedRecords": checked,
            "modelLoaded": False, "optimizerUpdates": 0, "trainingAuthorized": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plex 600M safe model/tokenizer setup")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="Print exact size and config without model allocation")
    fit = commands.add_parser("tokenizer-train", help="Fit scratch BPE on verified training split only")
    fit.add_argument("--dataset-dir", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--vocab-size", type=int, default=MODEL_CAPACITY)
    fit.add_argument("--min-frequency", type=int, default=2)
    verify = commands.add_parser("tokenizer-check", help="Verify an existing 600M bundle without changes")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    binding = commands.add_parser("source-check", help="Compare packed records with explicitly supplied source")
    binding.add_argument("--bundle-dir", type=Path, required=True)
    binding.add_argument("--dataset-dir", type=Path, required=True)
    binding.add_argument("--bundle-manifest-sha256", required=True)
    binding.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect()
        elif args.command == "tokenizer-train":
            result = train_tokenizer(
                args.dataset_dir, args.output_dir, vocab_size=args.vocab_size,
                min_frequency=args.min_frequency, model_family=MODEL_FAMILY,
            )
            result = {"modelFamily": MODEL_FAMILY, "tokenizer": result,
                      "verifiedBundle": check_tokenizer_bundle(args.output_dir)}
        elif args.command == "source-check":
            result = check_source_binding(args.bundle_dir, args.dataset_dir,
                        bundle_manifest_sha256=args.bundle_manifest_sha256,
                        source_manifest_sha256=args.source_manifest_sha256)
        else:
            result = check_tokenizer_bundle(args.bundle_dir)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"Plex 600M setup refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
