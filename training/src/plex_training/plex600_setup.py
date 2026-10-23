"""Read-only Plex 600M inspection and guarded tokenizer preparation CLI.

This module deliberately contains no optimizer, model instantiation, training
execution, automatic downloads, or inference. Only `tokenizer-train` can write,
and its create-only, source-verified pipeline refuses overwrites.
"""

from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import sys
from typing import Any

from .plex600 import Plex600Config
from .tokenizer import PlexTokenizer, sha256_file, train_tokenizer

MODEL_FAMILY = "plex-600m-v1"
MODEL_CAPACITY = Plex600Config().vocab_size
MAX_INDEX_BYTES = 32 * 1024 * 1024
MAX_RECORD_TOKENS = 1_048_577


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


def check_tokenizer_bundle(directory: Path) -> dict[str, Any]:
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
        rows = _read_json(index_path, limit_bytes=MAX_INDEX_BYTES)
        if not isinstance(rows, list) or not rows:
            raise ValueError("Packed index must be a nonempty array")
        if info.get("records") != len(rows) or type(info.get("tokenCount")) is not int:
            raise ValueError("Packed split record count mismatch")
        expected_start = 0
        with token_path.open("rb") as stream:
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"recordId", "startToken", "tokenCount"}:
                    raise ValueError("Invalid token index entry")
                start, count = row["startToken"], row["tokenCount"]
                if (not isinstance(row["recordId"], str) or not row["recordId"]
                        or type(start) is not int or start != expected_start
                        or type(count) is not int or not 1 <= count <= MAX_RECORD_TOKENS):
                    raise ValueError("Token index is not contiguous or is out of bounds")
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
        verified[split] = {"records": len(rows), "tokens": expected_start, "sha256": info["sha256"]}

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
    }


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
        else:
            result = check_tokenizer_bundle(args.bundle_dir)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Plex 600M setup refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
