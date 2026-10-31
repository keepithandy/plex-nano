"""Read-only verifier for the first owner-authored Plex 600M preparation pilot.

This is a separate local-use path; it must NOT approve any of the six historical
external sources, open the final holdout, fit a tokenizer, or train a model.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "training" / "pretraining" / "plex600-owner-authored-pilot-v1"
PIPELINE = "plex600-owner-authored-pilot-v1"
MAX_BYTES = 1024 * 1024
CATALOGS = (
    ("code-pairs-v2", "training/phase2/drafts/p2-02-code-pairs-v2.jsonl",
     "training/phase2/data/authored/p2-02-code-pairs-v2/approval.json", 36, 24),
    ("request-following-v2", "training/phase2/drafts/p2-02-request-following-v2/candidate.jsonl",
     "training/phase2/data/authored/p2-02-request-following-v2/approval.json", 180, 120),
)
NOTICE = "licenses/owner-authored-local-pilot.txt"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked evidence file")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Pilot evidence exceeds 1 MiB")
    return path.read_bytes()


def jsonfile(path: Path) -> dict:
    value = json.loads(read(path))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def source_records() -> tuple[dict[str, dict[str, Any]], set[str], dict[str, dict]]:
    """Authenticate original owner approval and return ONLY old train records."""
    original = {}
    forbidden = set()
    identities = {}
    for kind, catalog, approval_file, total, usable in CATALOGS:
        raw = read(ROOT / catalog)
        approval_raw = read(ROOT / approval_file)
        approval = json.loads(approval_raw)
        if (approval.get("approvalStatus") != "approved"
                or approval.get("approvedBy") != "repository-owner"
                or approval.get("scope") != "local-P2-training"
                or approval.get("externalSourceTextIncluded") is not False
                or approval.get("records") != total
                or approval.get("candidateJsonlSha256") != sha(raw)):
            raise ValueError("Original owner approval or candidate hash changed")
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != total:
            raise ValueError("Original candidate record count changed")
        count = 0
        for item in rows:
            if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                    or item.get("language") not in {"html", "css", "javascript"}
                    or not isinstance(item.get("request"), str)
                    or not isinstance(item.get("solution"), str)
                    or not item["request"] or not item["solution"]
                    or not isinstance(item.get("splitGroupId"), str)
                    or not item["splitGroupId"]
                    or item.get("candidateSplit") not in {"train", "validation"}):
                raise ValueError("Malformed original candidate")
            rid = sha(f"{kind}:{item['id']}".encode("utf-8"))
            if rid in original or rid in forbidden:
                raise ValueError("Repeated original candidate identity")
            if item["candidateSplit"] == "validation":
                forbidden.add(rid)
                continue
            count += 1
            family = f"plex-owner-authored-{kind}-{item['language']}"
            content = f"Request: {item['request']}\nAnswer:\n{item['solution']}"
            original[rid] = {
                "recordId": rid, "sourceId": family, "groupId": family,
                "sourceFamilyId": family,
                "splitGroupId": f"{family}:{item['splitGroupId']}",
                "contentSha256": sha(content.encode("utf-8")),
                "path": f"{catalog}#{item['id']}",
                "language": item["language"],
                "originalCandidateId": item["id"], "text": content,
            }
        if count != usable:
            raise ValueError("Changed original train-only candidate count")
        identities[kind] = {
            "kind": kind, "candidatePath": catalog, "candidateJsonlSha256": sha(raw),
            "originalApprovalPath": approval_file,
            "originalApprovalSha256": sha(approval_raw), "originalTotal": total,
            "originalTrainSelected": usable, "originalValidationExcluded": total - usable,
        }
    if len(original) != 144 or len(forbidden) != 72:
        raise ValueError("Owner-authored source inventory changed")
    return original, forbidden, identities


def verify(dataset_dir: Path = DATASET) -> dict:
    if dataset_dir.is_symlink():
        raise ValueError("Dataset root cannot be a symlink")
    root = dataset_dir.resolve(strict=True)
    manifest_raw = read(root / "manifest.json")
    manifest = json.loads(manifest_raw)
    receipt = jsonfile(root / "receipt.json")
    if (manifest.get("schemaVersion") != 1
            or manifest.get("pipelineVersion") != PIPELINE
            or manifest.get("modelFamily") != "plex-600m-v1"
            or manifest.get("status") != "tokenizer-input-only-not-training-authorization"
            or manifest.get("datasetOrigin") != "repo-owner-authored-preapproved-p2-training-split-only"
            or manifest.get("sourceRightsApprovedForLocalPreparation") is not True
            or manifest.get("externalHistoricalSourcesUsed") is not False
            or manifest.get("m4PreparationApproved") is not False
            or manifest.get("trainingAuthorized") is not False
            or manifest.get("finalHoldoutOpened") is not False):
        raise ValueError("Dataset provenance, format or training authorization changed")
    split = manifest.get("split")
    if (not isinstance(split, dict)
            or split.get("seed") != 1337
            or split.get("method") != "original-p2-train-only-family-grouped-sha256-1337"
            or split.get("previousHistoricalValidationReused") is not False
            or split.get("previousP2ValidationOrEvaluationExcluded") is not True
            or split.get("independentDevelopmentEvaluationCreated") is not False):
        raise ValueError("Pilot split contract changed")
    original, forbidden, identities = source_records()
    if {p["kind"]: p for p in manifest["provenance"]} != identities:
        raise ValueError("Original source/approval provenance changed")
    sources = manifest["sources"]
    if not isinstance(sources, list) or len(sources) != 6:
        raise ValueError("Unexpected approved owner-authored sources")
    expected_families = {row["sourceFamilyId"] for row in original.values()}
    notice = read(root / NOTICE)
    for source in sources:
        sid = source["id"]
        if (sid not in expected_families or source.get("groupId") != sid
                or source.get("sourceFamilyId") != sid
                or source.get("licenseNoticeFile") != NOTICE
                or source.get("licenseNoticeSha256") != sha(notice)
                or source.get("rightsReviewStatus") != "approved"
                or source.get("rightsApprovalScope") != "owner-requested-local-600m-dataset-preparation-only"
                or source.get("externalSourceTextIncluded") is not False):
            raise ValueError("Unapproved or changed source identity")
    if {s["id"] for s in sources} != expected_families:
        raise ValueError("Unexpected or repeated source")
    partitions = {}
    seen_ids, seen_content = set(), set()
    for name in ("train", "validation"):
        raw = read(root / f"{name}.jsonl")
        items = [json.loads(line) for line in raw.splitlines()]
        if (sha(raw) != manifest["summary"].get(name + "JsonlSha256")
                or len(items) != manifest["summary"].get(name + "Records")
                or not items or not raw.endswith(b"\n")):
            raise ValueError("Split hash, framing or count mismatch")
        previous = None
        for row in items:
            if (not isinstance(row, dict) or row.get("recordId") not in original
                    or row != original[row["recordId"]]):
                raise ValueError("Record differs from source-approved original train record")
            if row["recordId"] in forbidden or row["recordId"] in seen_ids:
                raise ValueError("Previous validation or duplicate identity included")
            if row["contentSha256"] in seen_content:
                raise ValueError("Duplicate content across split")
            seen_ids.add(row["recordId"])
            seen_content.add(row["contentSha256"])
            position = (row["sourceId"], row["splitGroupId"], row["recordId"])
            if previous is not None and position < previous:
                raise ValueError("Noncanonical record ordering")
            previous = position
        partitions[name] = items
    if seen_ids != set(original):
        raise ValueError("Dataset skipped or added a source record")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in original.values():
        grouped[row["sourceFamilyId"]][row["splitGroupId"]].append(row)
    expected_assignment = {}
    expected_groups = {}
    for family, groups in grouped.items():
        if len(groups) < 2:
            raise ValueError("Not enough groups to hold out one per family")
        ordered = sorted(groups, key=lambda name: (sha(f"1337:{family}:{name}".encode()), name))
        held = ordered[0]
        expected_groups[family] = {"allGroups": len(groups), "validationGroups": 1,
                                   "trainGroups": len(groups) - 1}
        for group, rows in groups.items():
            for row in rows:
                expected_assignment[row["recordId"]] = "validation" if group == held else "train"
    if split.get("families") != expected_groups:
        raise ValueError("Group allocation receipt changed")
    for split_name, rows in partitions.items():
        if any(expected_assignment[row["recordId"]] != split_name for row in rows):
            raise ValueError("Source-related groups cross split boundary")
    if (receipt.get("kind") != PIPELINE or receipt.get("status") != "prepared-for-tokenizer-not-training"
            or receipt.get("manifestSha256") != sha(manifest_raw)
            or receipt.get("splitHashes") != {
                "train": manifest["summary"]["trainJsonlSha256"],
                "validation": manifest["summary"]["validationJsonlSha256"]}
            or receipt.get("sourceRecordsUsed") != 144
            or receipt.get("trainRecords") != len(partitions["train"])
            or receipt.get("validationRecords") != len(partitions["validation"])
            or receipt.get("excludedPreviousP2ValidationRecords") != 72
            or receipt.get("externalSixSourcesIncluded") != 0
            or receipt.get("trainingAuthorized") is not False):
        raise ValueError("Receipt does not match prepared dataset")
    return {"status": "verified-owner-authored-tokenizer-ready",
            "manifestSha256": sha(manifest_raw),
            "trainRecords": len(partitions["train"]),
            "validationRecords": len(partitions["validation"]),
            "priorValidationExcluded": 72,
            "externalHistoricalRecords": 0,
            "trainingAuthorized": False, "finalHoldoutOpened": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.dataset_dir), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
