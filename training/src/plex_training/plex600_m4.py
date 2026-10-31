"""Plex 600M M4: offline, bounded, approval-gated candidate corpus preparation.

Consumes pinned historical data; never fits a tokenizer or trains a model.
Previously observed validation is audited but is not a fresh evaluation split.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from . import plex600_m3_scope as m3
from .dataset import _SECRET_PATTERNS

MAX_INPUT = 16 * 1024**2
MAX_METADATA = 1024**2
MAX_RECORD = 128 * 1024
HEX = re.compile(r"[a-f0-9]{64}\Z")
DEFAULT_DATASETS = Path(__file__).resolve().parents[2] / "artifacts" / "datasets"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def read_bounded(path: Path, maximum: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Input is missing or is a symbolic link")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("M4 input exceeds the byte bound")
    return raw


def reviewed_decision(decision: dict, review_sha: str, holds_sha: str, scope_sha: str) -> None:
    """Independent approval of M4 and scoped rights; M3 itself never approves."""
    if (not isinstance(decision, dict) or
            decision.get("kind") != "plex600-m4-rights-and-preparation-v1" or
            decision.get("m4PreparationApproved") is not True or
            decision.get("trainingAuthorized") is not False or
            decision.get("finalHoldoutAccessed") is not False or
            decision.get("sourceReviewSha256") != review_sha or
            decision.get("holdsSha256") != holds_sha or
            decision.get("scopeSha256") != scope_sha):
        raise ValueError("Missing or mismatched independent rights/preparation decision")
    expected = {k for k, (_, split) in m3.ROLES.items() if split == "train"}
    sources = decision.get("sources")
    if not isinstance(sources, list) or len(sources) != len(expected):
        raise ValueError("A scoped rights decision is required for every candidate source")
    seen = set()
    for row in sources:
        if not isinstance(row, dict) or set(row) != {
                "sourceId", "rightsStatus", "reviewer", "evidence", "approvedScope"}:
            raise ValueError("Invalid rights decision row")
        name = row["sourceId"]
        if name not in expected or name in seen or row["rightsStatus"] != "approved":
            raise ValueError("Unknown, duplicate or unapproved rights source")
        if any(not isinstance(row[k], str) or not row[k].strip()
               for k in ("reviewer", "evidence", "approvedScope")):
            raise ValueError("A source needs reviewer, evidence and approved scope")
        seen.add(name)
    if seen != expected:
        raise ValueError("Incomplete rights decisions")


def historical_records(root: Path, review: dict) -> tuple[list[dict], dict[str, bytes]]:
    """Recheck complete split hashes, records, source lineage, and packaged notices."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected local historical datasets directory")
    output_rows, notices = [], {}
    ids, contents = set(), set()
    for corpus in review["corpora"]:
        directory = root / corpus["id"]
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Missing or linked historical corpus")
        raw_manifest = read_bounded(directory / "manifest.json", MAX_METADATA)
        if sha(raw_manifest) != corpus["manifestSha256"]:
            raise ValueError("Historical manifest identity mismatch")
        manifest = json.loads(raw_manifest)
        sources = {s["id"]: s for s in manifest["sources"]}
        prior_sources = {s["id"]: s for s in corpus["sources"]}
        if (set(sources) != set(prior_sources) or
                manifest.get("sourceCatalogSha256") != corpus["catalogSha256"]):
            raise ValueError("Historical source catalog mismatch")
        found_paths = {name: [] for name in sources}
        for name, prior in prior_sources.items():
            source = sources[name]
            if (source.get("groupId") != prior["groupId"] or
                    source.get("revision") != prior["revision"] or
                    source.get("licenseNoticeSha256") != prior["noticeSha256"] or
                    source.get("licenseNoticeFile") != f"licenses/{name}.txt"):
                raise ValueError("Historical source or license identity mismatch")
            notice = read_bounded(directory / source["licenseNoticeFile"], MAX_METADATA)
            if sha(notice) != prior["noticeSha256"]:
                raise ValueError("Historical license notice changed")
            if name in notices and notices[name] != notice:
                raise ValueError("Conflicting source notices")
            notices[name] = notice
        identities = []
        for split in ("train", "validation"):
            raw = read_bounded(directory / f"{split}.jsonl", MAX_INPUT)
            if sha(raw) != manifest["summary"][split + "JsonlSha256"]:
                raise ValueError("Historical split hash mismatch")
            lines = raw.splitlines(keepends=True)
            if len(lines) != manifest["summary"][split + "Records"]:
                raise ValueError("Historical split count mismatch")
            for line in lines:
                if not line.endswith(b"\n") or len(line) > 1024**2 + 8192:
                    raise ValueError("Invalid historical record framing")
                row = json.loads(line)
                required = ("sourceId", "path", "recordId", "contentSha256", "groupId", "text")
                if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in required):
                    raise ValueError("Invalid historical record")
                name = row["sourceId"]
                if name not in prior_sources:
                    raise ValueError("Unknown historical source")
                group, required_split = m3.ROLES[name]
                if (row["groupId"] != group or required_split != split or
                        not HEX.fullmatch(row["recordId"]) or
                        not HEX.fullmatch(row["contentSha256"])):
                    raise ValueError("Historical source family/split/identity mismatch")
                if sha(row["text"].encode("utf-8")) != row["contentSha256"]:
                    raise ValueError("Historical text content hash mismatch")
                if row["recordId"] in ids or row["contentSha256"] in contents:
                    raise ValueError("Duplicate historical record or text")
                ids.add(row["recordId"])
                contents.add(row["contentSha256"])
                found_paths[name].append(row["path"])
                identities.append({k: (corpus["id"] if k == "corpus" else
                                        split if k == "split" else row[k])
                                   for k in ("corpus", "split", "sourceId",
                                             "path", "recordId", "contentSha256")})
                output_rows.append({"_historicalCorpus": corpus["id"],
                                    "_historicalSplit": split, **row})
        expected_identity = sha(json.dumps(identities, sort_keys=True, separators=(",", ":")).encode())
        if expected_identity != corpus["orderedRecordIdentitySha256"]:
            raise ValueError("Historical ordered record identities changed")
        for name, prior in prior_sources.items():
            if sorted(found_paths[name]) != prior["acceptedPaths"]:
                raise ValueError("Historical accepted path set mismatch")
    return output_rows, notices


def select_candidate(rows: list[dict], holds: dict, scope: dict, plan: dict):
    """Return deterministic train candidates and exact reasons for every skipped row."""
    omitted = {}
    for reason, records in (("embedded-library-hold", holds["proposedOmissions"]),
                            ("unclassified-notice", holds["noticeTriageRecords"])):
        for item in records:
            key = item["sourceId"], item["path"]
            omitted.setdefault(key, set()).add(reason)
    # Detect identities that appeared in the historical validation corpus.
    validation_hashes = {r["contentSha256"] for r in rows if r["_historicalSplit"] == "validation"}
    kept, skipped, counts = [], [], Counter()
    seen = set()
    for row in sorted(rows, key=lambda r: (r["sourceId"], r["path"], r["recordId"])):
        if row["_historicalSplit"] != "train":
            continue
        key = row["sourceId"], row["path"]
        if key in seen:
            raise ValueError("Duplicate source path in historical candidate")
        seen.add(key)
        reasons = set(omitted.get(key, ()))
        raw = row["text"].encode("utf-8")
        if not row["text"].strip():
            reasons.add("empty")
        if len(raw) > MAX_RECORD:
            reasons.add("record-byte-limit")
        if "\x00" in row["text"] or "\ufffd" in row["text"]:
            reasons.add("invalid-text-marker")
        if any(pattern.search(row["text"]) for pattern in _SECRET_PATTERNS):
            reasons.add("secret-pattern")
        if row["contentSha256"] in validation_hashes:
            reasons.add("historical-validation-content-overlap")
        if reasons:
            reasons = sorted(reasons)
            skipped.append({"sourceId": row["sourceId"], "path": row["path"],
                            "recordId": row["recordId"],
                            "contentSha256": row["contentSha256"], "reasons": reasons})
            counts.update(reasons)
        else:
            kept.append({k: v for k, v in row.items() if not k.startswith("_historical")})
    if len(kept) + len(skipped) != scope["candidateTrainRecords"] + scope["trainQuarantined"]:
        raise ValueError("M4 omitted a historical training record")
    if len(kept) > plan["maxCandidateRecords"]:
        raise ValueError("Candidate record-count budget exceeded")
    summary = {"historicTrainRecords": len(kept) + len(skipped),
               "candidateTrainRecords": len(kept),
               "skippedTrainRecords": len(skipped),
               "reasons": dict(sorted(counts.items())),
               "historicalValidationRecords": sum(r["_historicalSplit"] == "validation" for r in rows),
               "freshValidationRecords": 0}
    return kept, {"summary": summary, "skipped": skipped}, summary


def build(historical_root: Path, output_dir: Path, decision_file: Path) -> dict:
    scope = m3.inspect()  # Verifies pinned source/hold scope before any raw data.
    review = m3.read_json(*m3.REVIEW)
    holds = m3.read_json(*m3.HOLDS)
    plan = m3.read_json("plex600-m3-candidate-scope-v1.json")
    policy_sha = sha(read_bounded(m3.PRE / "plex600-m3-candidate-scope-v1.json", MAX_METADATA))
    decision_raw = read_bounded(decision_file, MAX_METADATA)
    reviewed_decision(json.loads(decision_raw), m3.REVIEW[1], m3.HOLDS[1], policy_sha)
    root = historical_root.resolve(strict=True)
    output = output_dir.resolve(strict=False)
    if output.exists() or output == root or output in root.parents or root in output.parents:
        raise ValueError("Output must be a new directory separate from historical datasets")
    rows, notices = historical_records(root, review)
    train, audit, summary = select_candidate(rows, holds, scope, plan)
    if not train:
        raise ValueError("No usable training candidates")
    train_raw = b"".join(json_bytes(r) for r in train)
    audit_raw = json_bytes(audit)
    notice_files = {f"licenses/{name}.txt": notices[name] for name, (_, split)
                    in sorted(m3.ROLES.items()) if split == "train"}
    manifest = {"schemaVersion": 1, "pipelineVersion": "plex600-m4-candidate-v1",
                "status": "review-only-not-ready-for-tokenizer",
                "sourceReviewSha256": m3.REVIEW[1], "holdsSha256": m3.HOLDS[1],
                "scopeSha256": policy_sha, "rightsDecisionSha256": sha(decision_raw),
                "trainRecords": len(train), "trainJsonlSha256": sha(train_raw),
                "trainBytes": len(train_raw), "auditSha256": sha(audit_raw),
                "noticeSha256": {k: sha(v) for k, v in notice_files.items()},
                "summary": summary, "freshEvaluationCreated": False,
                "tokenizerFitted": False, "trainingAuthorized": False,
                "optimizerUpdates": 0}
    manifest_raw = json_bytes(manifest)
    total = sum(map(len, (train_raw, audit_raw, manifest_raw))) + sum(map(len, notice_files.values()))
    if total > plan["maxOutputBytes"]:
        raise ValueError("M4 total output exceeds 16 MiB; refusing partial publication")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Output appeared before staging")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        (staging / "licenses").mkdir()
        (staging / "train.jsonl").write_bytes(train_raw)
        (staging / "skipped-audit.json").write_bytes(audit_raw)
        (staging / "manifest.json").write_bytes(manifest_raw)
        for name, raw in notice_files.items():
            (staging / name).write_bytes(raw)
        if output.exists():
            raise FileExistsError("Refusing to overwrite an output published during staging")
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"output": str(output), "manifestSha256": sha(manifest_raw),
            "candidateTrainRecords": len(train), "bytes": total,
            "status": "review-only", "tokenizerFitted": False,
            "trainingAuthorized": False}


def verify_output(output_dir: Path) -> dict:
    root = output_dir.resolve(strict=True)
    manifest_raw = read_bounded(root / "manifest.json", MAX_METADATA)
    manifest = json.loads(manifest_raw)
    if (manifest.get("pipelineVersion") != "plex600-m4-candidate-v1"
            or manifest.get("status") != "review-only-not-ready-for-tokenizer"
            or manifest.get("trainingAuthorized") is not False
            or manifest.get("freshEvaluationCreated") is not False):
        raise ValueError("Not a review-only M4 artifact")
    if (root / "validation.jsonl").exists():
        raise ValueError("Old validation must never be published as fresh")
    train_raw = read_bounded(root / "train.jsonl", MAX_INPUT)
    audit_raw = read_bounded(root / "skipped-audit.json", MAX_INPUT)
    if sha(train_raw) != manifest["trainJsonlSha256"] or sha(audit_raw) != manifest["auditSha256"]:
        raise ValueError("Published content SHA mismatch")
    count = 0
    for line in train_raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ValueError("Invalid published JSONL framing")
        row = json.loads(line)
        if sha(row["text"].encode("utf-8")) != row["contentSha256"]:
            raise ValueError("Published record content changed")
        count += 1
    if count != manifest["trainRecords"] or len(train_raw) != manifest["trainBytes"]:
        raise ValueError("Published train count/size changed")
    for path, expected in manifest["noticeSha256"].items():
        if not path.startswith("licenses/") or "/" in path[len("licenses/"):]:
            raise ValueError("Unsafe packaged notice name")
        if sha(read_bounded(root / path, MAX_METADATA)) != expected:
            raise ValueError("Published notice changed")
    return {"status": "verified-review-only", "trainRecords": count,
            "manifestSha256": sha(manifest_raw), "readyForTokenizer": False,
            "optimizerUpdates": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--historical-datasets-root", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--reviewed-decision", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        report = {**m3.inspect(), "m4MaterializationAuthorized": False}
    elif args.command == "build":
        report = build(args.historical_datasets_root, args.output_dir, args.reviewed_decision)
    else:
        report = verify_output(args.dataset_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
