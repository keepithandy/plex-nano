"""Prepare and inspect the frozen P3 copy-fidelity train/validation packed bundle.

No model, optimizer, checkpoint, dev control, or final holdout is opened.
Default action is read-only preflight; --prepare publishes to a fresh directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import struct
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from plex_training.artifacts import enforce_storage_limit, path_within_root
from plex_training.completion import _completion_tokenizer
from plex_training.tokenizer import CODEC, sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARTIFACT_ROOT = Path.home() / "PlexArtifacts"
SOURCE = ARTIFACT_ROOT / "p3-02-training-bundle"
OUTPUT = ARTIFACT_ROOT / "file-edits/p3-copy-fidelity-v1-bundle"
SOURCE_BUNDLE_SHA = "0cd6df3a6d8eec1d7da8c7a02af067b15299393d8b0e1756a071be17b9aca354"
CANDIDATE_SHA = "9414d1da7672a386a920d1935723e9133d011c9c3a4029e5d5800f643835ca7a"
TOKENIZER_SHA = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
MAX_STORAGE = 200 * 1024**3
EXPECTED_TARGET_POSITIONS = {"train": 7300, "validation": 2420}
PARTS = ("jsonl", "tokens.u16le", "weights.u8", "index.json")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer-config.json", "model-config.json")
SPLITS = ("train", "validation")

SPEC = importlib.util.spec_from_file_location("copy_fidelity_prep_bundle", HERE / "prepare_copy_fidelity.py")
assert SPEC and SPEC.loader
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


def _serialized(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _regular_file(path: Path, maximum: int = 20 * 1024 * 1024) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError(f"Missing or invalid regular bundle file: {path.name}")
    return path.read_bytes()


def _tokenizer(source: Path) -> tuple[Any, dict[str, Any]]:
    tokenizer, identity = _completion_tokenizer(source)
    if (identity.get("tokenizerSha256") != TOKENIZER_SHA
            or identity.get("bundleManifestSha256") != SOURCE_BUNDLE_SHA
            or tokenizer.vocabulary_size != 16384):
        raise ValueError("Frozen P3-02 tokenizer/bundle identity changed")
    return tokenizer, identity


def _candidate() -> dict[str, Any]:
    raw = _regular_file(HERE / "copy-fidelity-candidate-v1.json", maximum=1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != CANDIDATE_SHA or raw != PREP.serialized():
        raise ValueError("Frozen copy-fidelity candidate changed")
    m = json.loads(raw)
    if (m["authorization"]["trainingAuthorized"] is not False
            or m["authorization"]["evaluationAuthorized"] is not False
            or m["model"]["startingCheckpointSha256"] != PREP.STEP100_SHA256):
        raise ValueError("Frozen candidate permission/checkpoint changed")
    return m


def _sampler(count: int, target_lengths: list[int], seed: int = 1337) -> dict[str, Any]:
    """Deterministic balanced proposal, not permission to use an optimizer."""
    if count != 288 or len(target_lengths) != count:
        raise ValueError("Expected 288 training examples")
    rng = random.Random(seed)
    draws = []
    while len(draws) < 1600:
        epoch = list(range(count))
        rng.shuffle(epoch)
        draws.extend(epoch[:1600 - len(draws)])
    if set(draws) != set(range(count)):
        raise ValueError("Sampler did not cover every training example")
    return {
        "kind": "seeded-epoch-shuffle-full-coverage-proposal-v1",
        "seed": seed,
        "draws": len(draws),
        "distinctRecords": len(set(draws)),
        "drawIndexSha256": hashlib.sha256(struct.pack("<" + "H" * len(draws), *draws)).hexdigest(),
        "expectedAnswerAndEosTargetPositions": sum(target_lengths[i] for i in draws),
        "optimizerUpdatesPerformed": 0,
        "samplingExecutionAuthorized": False,
    }


def pack(tokenizer: Any) -> tuple[dict[str, dict[str, bytes]], dict[str, Any]]:
    m = _candidate()
    result = {}
    details = {}
    target_lengths = []
    for split in SPLITS:
        tokens, weights, text, index = bytearray(), bytearray(), bytearray(), []
        for row in m["rows"]:
            if row["split"] != split:
                continue
            pids = tokenizer.encode(row["prompt"])
            yids = tokenizer.encode(row["expectedFile"])
            if (not pids or not yids
                    or tokenizer.decode(pids) != row["prompt"]
                    or tokenizer.decode(yids) != row["expectedFile"]
                    or any(type(i) is not int or not 4 <= i < 16384 for i in pids + yids)
                    or len(pids) + len(yids) + 1 > 512
                    or len(pids) + 193 > 512 or len(yids) + 1 > 192):
                raise ValueError(f"Tokenizer/supervision mismatch: {row['id']}")
            packed = pids + yids + [3]
            mask = bytes([0] * (len(pids) - 1) + [1] * (len(yids) + 1))
            index.append({
                "recordId": row["id"], "startToken": len(tokens) // 2,
                "tokenCount": len(packed), "startTarget": len(weights),
                "targetCount": len(mask), "promptTokens": len(pids),
                "answerTokensIncludingEos": len(yids) + 1,
            })
            text.extend(_serialized({
                "recordId": row["id"], "promptTokens": len(pids),
                "answerTokensIncludingEos": len(yids) + 1,
                "text": row["prompt"] + row["expectedFile"],
            }))
            tokens.extend(struct.pack("<" + "H" * len(packed), *packed))
            weights.extend(mask)
            if split == "train":
                target_lengths.append(sum(mask))
        if (len(index) != (288 if split == "train" else 96)
                or sum(weights) != EXPECTED_TARGET_POSITIONS[split]):
            # The target counts were *measured* by the owner using frozen
            # tokenizer: never silently replace them with a new dataset.
            raise ValueError(f"Observed frozen answer/EOS target counts changed: {split}")
        result[split] = {
            "jsonl": bytes(text), "tokens.u16le": bytes(tokens),
            "weights.u8": bytes(weights),
            "index.json": (json.dumps(index, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        }
        details[split] = {
            "records": len(index), "recordTokens": len(tokens) // 2,
            "answerAndEosTargetPositions": sum(weights),
            "maxRecordTokens": max(r["tokenCount"] for r in index),
            **{suffix + "Sha256": hashlib.sha256(raw).hexdigest()
               for suffix, raw in result[split].items()},
        }
    return result, {"splits": details, "sampler": _sampler(288, target_lengths)}


def _manifest(plan: dict[str, Any], source_identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": 1, "kind": "p3-copy-fidelity-answer-eos-bundle-v1",
        "candidateSha256": CANDIDATE_SHA,
        "codec": CODEC, "actualVocabularySize": 16384,
        "contextLength": 512, "tokenizerSha256": TOKENIZER_SHA,
        "sourceTokenizerBundleManifestSha256": SOURCE_BUNDLE_SHA,
        "modelVocabularyCapacity": 16384,
        "modelConfigSha256": source_identity["modelConfigSha256"],
        "splits": plan["splits"], "proposalSampler": plan["sampler"],
        "trainingAuthorized": False, "trainingPerformed": False,
        "evaluationAuthorized": False, "optimizerUpdatesPerformed": 0,
        "finalHoldoutOpened": False,
    }


def preflight(source_bundle: Path = SOURCE) -> dict[str, Any]:
    _candidate()
    tokenizer, identity = _tokenizer(source_bundle)
    _, plan = pack(tokenizer)
    return {"status": "copy-fidelity-bundle-read-only-preflight-passed",
            "candidateSha256": CANDIDATE_SHA, "tokenizerSha256": TOKENIZER_SHA,
            **plan, "trainingPerformed": False, "optimizerUpdates": 0}


def inspect(bundle_dir: Path) -> dict[str, Any]:
    if bundle_dir.is_symlink() or not bundle_dir.is_dir():
        raise ValueError("Packed bundle must be a normal directory")
    tokenizer, identity = _completion_tokenizer(bundle_dir)
    if identity["tokenizerSha256"] != TOKENIZER_SHA or tokenizer.vocabulary_size != 16384:
        raise ValueError("Packed bundle tokenizer identity changed")
    files, plan = pack(tokenizer)
    manifest = json.loads(_regular_file(bundle_dir / "manifest.json"))
    expected = _manifest(plan, identity)
    if manifest != expected:
        raise ValueError("Packed bundle manifest/permissions differ")
    expected_names = {"manifest.json", *TOKENIZER_FILES}
    for split in SPLITS:
        for suffix, raw in files[split].items():
            name = f"{split}.{suffix}"
            expected_names.add(name)
            if _regular_file(bundle_dir / name) != raw:
                raise ValueError(f"Packed bundle contents changed: {name}")
    if {p.name for p in bundle_dir.iterdir()} != expected_names:
        raise ValueError("Packed bundle contains unexpected or missing files")
    return {"status": "copy-fidelity-bundle-verified", "bundleVerified": True,
            "bundleManifestSha256": sha256_file(bundle_dir / "manifest.json"),
            "candidateSha256": CANDIDATE_SHA, "tokenizerSha256": TOKENIZER_SHA,
            **plan, "trainingPerformed": False, "optimizerUpdates": 0,
            "finalHoldoutOpened": False}


def prepare(*, output_dir: Path = OUTPUT, source_bundle: Path = SOURCE,
            artifact_root: Path = ARTIFACT_ROOT, storage_limit_bytes: int = MAX_STORAGE) -> dict[str, Any]:
    """Publish only a new verified bundle, atomically. No model access."""
    if type(storage_limit_bytes) is not int or storage_limit_bytes <= 0:
        raise ValueError("Invalid storage budget")
    root = artifact_root.resolve()
    output = path_within_root(output_dir, root)
    source = source_bundle.resolve(strict=True)
    if (output == root or output.exists() or output == source
            or output in source.parents or source in output.parents
            or output in (HERE, ROOT) or output in HERE.parents):
        raise ValueError("Bundle output must be new and separate from all inputs")
    tokenizer, identity = _tokenizer(source_bundle)
    files, plan = pack(tokenizer)
    manifest = _manifest(plan, identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".copy-fidelity-bundle-", dir=output.parent))
    try:
        for name in TOKENIZER_FILES:
            raw = _regular_file(source / name, 16 * 1024 * 1024 if name == "tokenizer.json" else 1024 * 1024)
            (stage / name).write_bytes(raw)
        for split in SPLITS:
            for suffix, raw in files[split].items():
                (stage / f"{split}.{suffix}").write_bytes(raw)
        (stage / "manifest.json").write_bytes(
            (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        )
        enforce_storage_limit(root, limit_bytes=storage_limit_bytes)
        if not inspect(stage)["bundleVerified"]:
            raise ValueError("Packed output failed reconstruction")
        if output.exists():
            raise FileExistsError("Bundle output already exists")
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"status": "copy-fidelity-bundle-prepared-no-training",
            "bundleDirectory": str(output), **inspect(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prepare", action="store_true", help="Write one new immutable bundle")
    group.add_argument("--verify", action="store_true", help="Inspect the existing packed bundle")
    parser.add_argument("--bundle-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        if args.prepare:
            output = prepare(output_dir=args.bundle_dir)
        elif args.verify:
            output = inspect(args.bundle_dir)
        else:
            output = preflight()
        print(json.dumps(output, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"p3-copy-fidelity-bundle: {exc}\n")


if __name__ == "__main__":
    main()
