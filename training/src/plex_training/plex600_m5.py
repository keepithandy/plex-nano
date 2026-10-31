"""Plex 600M M5: rights-gated split of reviewed M4 candidates for tokenizer fitting.

Produces a new deterministic train/validation *development* split. This is not
the independent coding evaluation or final holdout. No model or optimizer work.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from . import plex600_m3_scope as m3
from . import plex600_m4 as m4

VERSION = "plex600-m5.0"
DEFAULT_SEED = 1337
DEFAULT_VALIDATION_PERCENT = 10
MAX_OUTPUT = 32 * 1024**2


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _local(parent: Path, relative: str) -> Path:
    path = parent / relative
    if path.is_symlink():
        raise ValueError("Linked input/output is not permitted")
    resolved = path.resolve(strict=True)
    resolved.relative_to(parent.resolve(strict=True))
    if not resolved.is_file():
        raise ValueError("Expected regular input")
    return resolved


def _input(candidate: Path, decision_file: Path):
    report = m4.verify_output(candidate)
    if report.get("status") != "verified-review-only":
        raise ValueError("M4 candidate is not independently verified")
    raw_manifest = m4.read_bounded(_local(candidate, "manifest.json"), m4.MAX_METADATA)
    manifest = json.loads(raw_manifest)
    raw_decision = m4.read_bounded(decision_file, m4.MAX_METADATA)
    decision = json.loads(raw_decision)
    scope_sha = _sha(m4.read_bounded(m3.PRE / "plex600-m3-candidate-scope-v1.json", m4.MAX_METADATA))
    m4.reviewed_decision(decision, m3.REVIEW[1], m3.HOLDS[1], scope_sha)
    if (manifest.get("sourceReviewSha256") != m3.REVIEW[1] or
            manifest.get("holdsSha256") != m3.HOLDS[1] or
            manifest.get("scopeSha256") != scope_sha or
            manifest.get("rightsDecisionSha256") != _sha(raw_decision)):
        raise ValueError("M4 candidate does not match reviewed rights decision")
    raw = m4.read_bounded(_local(candidate, "train.jsonl"), m4.MAX_INPUT)
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != manifest["trainRecords"] or not rows:
        raise ValueError("M4 candidate train count mismatch")
    return rows, manifest, raw_manifest, raw_decision


def _split(rows: list[dict], seed: int, percent: int):
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be a uint32")
    if type(percent) is not int or not 1 <= percent <= 40:
        raise ValueError("Validation percent must be 1..40")
    families: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    all_ids: set[str] = set()
    all_content: set[str] = set()
    roles = {name: group for name, (group, split) in m3.ROLES.items() if split == "train"}
    for old in rows:
        if not isinstance(old, dict) or any(not isinstance(old.get(k), str) or not old[k]
                                           for k in ("sourceId", "recordId", "path",
                                                     "groupId", "contentSha256", "text")):
            raise ValueError("Incomplete M4 candidate identity")
        name = old["sourceId"]
        if name not in roles or old["groupId"] != roles[name]:
            raise ValueError("M4 record is outside training source roles")
        if old["recordId"] in all_ids or old["contentSha256"] in all_content:
            raise ValueError("Duplicate record ID or content")
        all_ids.add(old["recordId"])
        all_content.add(old["contentSha256"])
        if _sha(old["text"].encode("utf-8")) != old["contentSha256"]:
            raise ValueError("M4 record content hash mismatch")
        # Preserve upstream group when present; otherwise each file is a
        # separate group. Prefix with source ID to prevent group collisions.
        original_group = old.get("splitGroupId")
        if original_group is not None and (not isinstance(original_group, str)
                                            or not original_group.strip()):
            raise ValueError("Invalid historical split group")
        split_group = (f"{roles[name]}:{original_group}" if original_group is not None
                       else f"{name}:{old['recordId']}")
        row = {**old, "sourceFamilyId": roles[name], "splitGroupId": split_group}
        families[roles[name]][split_group].append(row)
    if len(families) < 2:
        raise ValueError("At least two source families are required")
    result = {"train": [], "validation": []}
    selection = {}
    for family, groups in sorted(families.items()):
        names = sorted(groups)
        if len(names) < 2:
            raise ValueError("Source family cannot be present in both splits")
        count = max(1, min(len(names) - 1, round(len(names) * percent / 100)))
        ordered = sorted(names, key=lambda name: (
            _sha(f"{seed}:{family}:{name}".encode("utf-8")), name))
        validation = set(ordered[:count])
        selection[family] = {"groups": len(names), "validationGroups": count,
                             "trainGroups": len(names) - count}
        for name in names:
            split = "validation" if name in validation else "train"
            result[split].extend(groups[name])
    for split in result:
        result[split].sort(key=lambda r: (r["sourceFamilyId"],
                                         r["sourceId"], r["path"], r["recordId"]))
        if not result[split]:
            raise ValueError("Split is empty")
    return result, selection


def build(candidate_dir: Path, output_dir: Path, rights_decision: Path,
          *, seed: int = DEFAULT_SEED,
          validation_percent: int = DEFAULT_VALIDATION_PERCENT):
    """Create-only M5 output. M4 rights decision must match exact M4 receipt."""
    candidate = candidate_dir.resolve(strict=True)
    if candidate_dir.is_symlink() or not candidate.is_dir():
        raise ValueError("M4 input must be a regular directory")
    output = output_dir.resolve(strict=False)
    if output.exists() or output in candidate.parents or candidate in output.parents or output == candidate:
        raise ValueError("M5 output must be new and separate from M4 candidate")
    rows, m4_manifest, m4_raw, decision_raw = _input(candidate, rights_decision)
    partitions, groups = _split(rows, seed, validation_percent)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["sourceId"]].append(row)
    sources = []
    license_bytes = {}
    for source_id, (group, role) in sorted(m3.ROLES.items()):
        if role != "train":
            continue
        if source_id not in by_source:
            raise ValueError("Missing candidate source; rights scope must not shrink silently")
        relative = f"licenses/{source_id}.txt"
        notice = m4.read_bounded(_local(candidate, relative), m4.MAX_METADATA)
        if _sha(notice) != m4_manifest["noticeSha256"][relative]:
            raise ValueError("Source license notice hash changed")
        license_bytes[relative] = notice
        sources.append({"id": source_id, "groupId": group,
                        "sourceFamilyId": group, "rightsReviewStatus": "approved",
                        "licenseNoticeFile": relative, "licenseNoticeSha256": _sha(notice),
                        "rightsDecisionSha256": _sha(decision_raw)})
    split_raw = {key: b"".join(_json(row) for row in partitions[key])
                 for key in ("train", "validation")}
    summary = {key + "Records": len(partitions[key]) for key in split_raw}
    summary.update({key + "JsonlSha256": _sha(split_raw[key]) for key in split_raw})
    manifest = {
        "schemaVersion": 1, "pipelineVersion": VERSION,
        "modelFamily": "plex-600m-v1",
        "status": "tokenizer-input-only-not-training-authorization",
        "m4ManifestSha256": _sha(m4_raw),
        "rightsDecisionSha256": _sha(decision_raw),
        "split": {"method": "family-grouped-seeded-sha256-v1",
                  "seed": seed, "validationPercent": validation_percent,
                  "families": groups, "previousHistoricalValidationReused": False,
                  "independentDevelopmentEvaluationCreated": False,
                  "finalHoldoutOpened": False},
        "sources": sources, "summary": summary,
        "tokenizerTrainingSplit": "train", "trainingAuthorized": False,
    }
    files = {"train.jsonl": split_raw["train"],
             "validation.jsonl": split_raw["validation"],
             "manifest.json": _json(manifest), **license_bytes}
    used = sum(len(b) for b in files.values())
    if used > MAX_OUTPUT:
        raise ValueError("M5 output exceeds 32 MiB budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("M5 output already exists")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.m5-stage-", dir=output.parent))
    try:
        (stage / "licenses").mkdir()
        for relative, content in files.items():
            (stage / relative).write_bytes(content)
        if output.exists():
            raise FileExistsError("M5 output appeared during construction")
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"status": "tokenizer-input-only", "manifestSha256": _sha(files["manifest.json"]),
            "trainRecords": len(partitions["train"]),
            "validationRecords": len(partitions["validation"]),
            "sourceFamilies": len(groups), "outputBytes": used,
            "trainingAuthorized": False, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4-candidate-dir", type=Path, required=True)
    parser.add_argument("--reviewed-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validation-percent", type=int, default=DEFAULT_VALIDATION_PERCENT)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.m4_candidate_dir, args.output_dir, args.reviewed_decision,
                           seed=args.seed, validation_percent=args.validation_percent),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
