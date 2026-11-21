"""M12-A owner-supplied copy-data audit. Read-only and not training authorization.

Accepted inputs are explicitly supplied local train/development JSONL and one
manifest. The sealed holdout is NEVER opened. This module does not load model
weights, initialize an optimizer, train, write files, or approve data rights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

SCHEMA = "plex-m12-a-source-copy-review-v1"
MODEL = "plex-600m-v1"
TOKENIZER_SHA = "9feef87bdf5fc28b3315df589e4e0b8ffcf5f40b18a4c357c91759182340a2e2"
EXPECTED_COUNTS = {"train": 256, "development": 64}
TEMPLATES = {"copy-space", "copy-newline", "console-general"}
SHA_PATTERN = re.compile(r"[a-f0-9]{64}\Z")
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
MAX_RECORD_BYTES = 8192
MAX_SOURCE_BYTES = 2048
MAX_MANIFEST_BYTES = 32768
MAX_JSONL_BYTES = 4 * 1024 * 1024


def refuse_if(not_ok: bool, reason: str) -> None:
    if not not_ok:
        raise ValueError(reason)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(value) -> bool:
    return type(value) is str and SHA_PATTERN.fullmatch(value) is not None


def name(value) -> bool:
    return type(value) is str and NAME_PATTERN.fullmatch(value) is not None


def checked_file(root: Path, filename: str) -> Path:
    refuse_if(filename in {"train.jsonl", "development.jsonl"}, "Only frozen train/development filenames are permitted.")
    path = root / filename
    refuse_if(not path.is_symlink() and path.resolve(strict=True).parent == root
              and path.is_file(), "Split file must be a regular, non-symlink direct child of manifest directory.")
    return path


def prompt_for(row: dict) -> str:
    instruction, source, template = row["instruction"], row["source"], row["promptTemplate"]
    if template == "copy-space":
        return f"Instruction: {instruction}\nCode:\n{source}\nCopy: "
    if template == "copy-newline":
        return f"Instruction: {instruction}\nCode:\n{source}\nCopy:\n"
    if template == "console-general":
        return f"Request: {instruction}\nSource file:\n{source}\nEdited file:\n"
    raise ValueError("Unrecognized prompt template")


def checked_record(row, split):
    refuse_if(type(row) is dict and set(row) == {
        "id", "split", "source", "expected", "instruction", "promptTemplate",
        "identifierFamily", "sourceTemplate", "provenanceReference",
        "sourceOrigin", "rightsClaim", "sourceSha256", "expectedSha256",
        "trainingAuthorized", "developmentSlices",
    }, "M12 record has missing or unknown fields")
    refuse_if(row["split"] == split and name(row["id"])
              and name(row["identifierFamily"]) and name(row["sourceTemplate"])
              and name(row["provenanceReference"]), "Invalid M12 record or group identity")
    refuse_if(row["sourceOrigin"] == "owner-authored"
              and row["rightsClaim"] == "owner-claims-rights"
              and row["trainingAuthorized"] is False,
              "M12-A accepts only owner-claimed provenance with no training approval")
    refuse_if(type(row["source"]) is str and type(row["expected"]) is str
              and len(row["source"].encode("utf-8")) <= MAX_SOURCE_BYTES
              and len(row["source"]) > 0 and row["source"].strip() != ""
              and row["expected"] == row["source"],
              "M12 copy target must equal a nonempty source of at most 2048 bytes")
    refuse_if(sha(row["sourceSha256"]) and sha(row["expectedSha256"])
              and digest(row["source"].encode("utf-8")) == row["sourceSha256"]
              and digest(row["expected"].encode("utf-8")) == row["expectedSha256"],
              "Source/expected content SHA-256 mismatch")
    refuse_if(type(row["instruction"]) is str
              and 5 <= len(row["instruction"]) <= 180
              and "\n" not in row["instruction"] and "\r" not in row["instruction"]
              and type(row["promptTemplate"]) is str
              and row["promptTemplate"] in TEMPLATES,
              "Invalid one-line copy request or template")
    refuse_if(type(row["developmentSlices"]) is list
              and all(type(tag) is str and tag in {"unseen-identifier", "prompt-transfer"}
                      for tag in row["developmentSlices"])
              and len(row["developmentSlices"]) == len(set(row["developmentSlices"]))
              and (split == "development" or not row["developmentSlices"]),
              "Unsupported evaluation slice annotation")
    # Membership labels come from the author; they are NOT independent evidence of transfer.
    return row


def audited_split(root, split, expected_meta):
    refuse_if(type(expected_meta) is dict and set(expected_meta) == {"file", "count", "sha256"},
              "Split metadata must include file, count and sha256 only")
    refuse_if(expected_meta["file"] == split + ".jsonl"
              and type(expected_meta["count"]) is int
              and expected_meta["count"] == EXPECTED_COUNTS[split]
              and sha(expected_meta["sha256"]), "Frozen M12 split identity/count mismatch")
    path = checked_file(root, expected_meta["file"])
    refuse_if(path.stat().st_size <= MAX_JSONL_BYTES, "Split JSONL exceeds safety size limit")
    raw = path.read_bytes()
    refuse_if(digest(raw) == expected_meta["sha256"], "Split SHA-256 mismatch")
    rows = []
    seen = set()
    for number, raw_line in enumerate(raw.splitlines(keepends=True), start=1):
        refuse_if(len(raw_line) <= MAX_RECORD_BYTES and raw_line.endswith(b"\n"),
                  "M12 record is too large or not newline terminated")
        value = json.loads(raw_line.decode("utf-8"))
        row = checked_record(value, split)
        refuse_if(row["id"] not in seen, "Repeated ID within M12 split")
        seen.add(row["id"])
        rows.append(row)
    refuse_if(len(rows) == expected_meta["count"], "Wrong M12 row count")
    return rows


def audit_manifest(path: Path):
    refuse_if(not path.is_symlink() and path.is_file()
              and path.stat().st_size <= MAX_MANIFEST_BYTES,
              "Manifest must be a regular, non-symlink small JSON file")
    root = path.parent.resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    refuse_if(type(manifest) is dict and set(manifest) == {
        "schema", "modelFamily", "status", "trainingAuthorized",
        "rightsVerification", "splits", "sealedHoldout",
    }, "M12-A manifest fields are not the frozen contract")
    refuse_if(manifest["schema"] == SCHEMA and manifest["modelFamily"] == MODEL
              and manifest["status"] == "research-only-not-training-authorized"
              and manifest["trainingAuthorized"] is False
              and manifest["rightsVerification"] == "author-claims-unverified",
              "Manifest cannot authorize training or claim verified rights")
    holdout = manifest["sealedHoldout"]
    refuse_if(type(holdout) is dict and set(holdout) == {
        "count", "commitmentSha256", "contentNotIncluded",
    } and type(holdout["count"]) is int and holdout["count"] == 64
              and sha(holdout["commitmentSha256"]) and holdout["contentNotIncluded"] is True,
              "Only a sealed holdout commitment, never holdout examples, is permitted")
    splits = manifest["splits"]
    refuse_if(type(splits) is dict and set(splits) == set(EXPECTED_COUNTS),
              "M12-A must have only train/development data files")
    data = {split: audited_split(root, split, splits[split]) for split in EXPECTED_COUNTS}
    train, development = data["train"], data["development"]
    for key in ("id", "identifierFamily", "sourceTemplate", "promptTemplate"):
        left = {row[key] for row in train}
        right = {row[key] for row in development}
        refuse_if(left.isdisjoint(right), f"M12-A train/development {key} leakage")
    train_hashes = {row["sourceSha256"] for row in train}
    refuse_if(train_hashes.isdisjoint({row["sourceSha256"] for row in development}),
              "Train/development identical source leakage")
    slices = {name: sum(name in row["developmentSlices"] for row in development)
              for name in ("unseen-identifier", "prompt-transfer")}
    refuse_if(slices["unseen-identifier"] >= 32 and slices["prompt-transfer"] >= 16,
              "Development slices do not meet predeclared minimum counts")
    return {
        "schema": SCHEMA,
        "manifestSha256": digest(path.read_bytes()),
        "trainingAuthorized": False,
        "rightsIndependentlyVerified": False,
        "holdoutContentOpened": False,
        "holdoutExistenceIndependentlyVerified": False,
        "trainCount": len(train),
        "developmentCount": len(development),
        "developmentSliceClaims": slices,
        "splitHashes": {split: splits[split]["sha256"] for split in EXPECTED_COUNTS},
    }, data


def audit_tokenizer_rows(tokenizer, data: dict):
    """Pure token roundtrip and context-boundary audit, independently testable."""
    count, max_positions, max_new = 0, 0, 0
    for split in EXPECTED_COUNTS:
        for row in data[split]:
            prompt = prompt_for(row)
            for value in (row["source"], row["expected"], prompt):
                ids = tokenizer.encode(value)
                refuse_if(len(ids) > 0 and tokenizer.decode(ids) == value,
                          "Tokenizer roundtrip failed on M12 source/target/prompt")
                count += 1
            prompt_ids = tokenizer.encode(prompt)
            target_ids = tokenizer.encode(row["expected"])
            refuse_if(len(target_ids) + 1 <= 96,
                      "M12 copy target and EOS exceed the frozen 96-token generation budget")
            token_positions = len(prompt_ids) + len(target_ids) + 1
            refuse_if(token_positions <= 1024, "M12 input/target/EOS exceeds 1024 positions")
            max_positions = max(max_positions, token_positions)
            max_new = max(max_new, len(target_ids) + 1)
    return {"checkedStrings": count, "maxPositionsIncludingTargetEos": max_positions,
            "maxTargetTokensIncludingEos": max_new, "tokenizerSha256": TOKENIZER_SHA}


def audit_tokenizer(bundle: Path, data: dict):
    """Optional, read-only check with the existing M11 tokenizer in uv/WSL."""
    refuse_if(not bundle.is_symlink() and bundle.is_dir(), "Tokenizer must be a non-symlink directory")
    config = json.loads((bundle / "tokenizer-config.json").read_text(encoding="utf-8"))
    refuse_if(config.get("tokenizerSha256") == TOKENIZER_SHA,
              "Tokenizer differs from frozen M11 identity")
    from plex_training.tokenizer import PlexTokenizer
    return audit_tokenizer_rows(PlexTokenizer.load(bundle), data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Owner-supplied manifest.json (read-only)")
    parser.add_argument("--tokenizer-bundle", type=Path, help="Optional frozen M11 tokenizer bundle")
    args = parser.parse_args()
    report, data = audit_manifest(args.manifest)
    if args.tokenizer_bundle is not None:
        report["tokenizerAudit"] = audit_tokenizer(args.tokenizer_bundle, data)
    else:
        report["tokenizerAudit"] = "NOT RUN"
    report["decision"] = "DATA STRUCTURE REVIEW ONLY - NO TRAINING APPROVAL"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
