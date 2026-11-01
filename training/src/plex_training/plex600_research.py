"""Plex600 research readiness: immutable dev benchmark, synthetic-candidate quarantine,
and local M7 report/checkpoint audit. Never instantiates a model or trains.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "training/phase3/evaluation/plex600-m8-independent-v1.jsonl"
CANDIDATES = ROOT / "training/phase3/candidates/plex600-r2-synthetic-edit-v1.jsonl"
M7_EVIDENCE = ROOT / "training/pretraining/plex600-m7-owner-reported-result-2026-10-31.json"
MAX_JSONL = 2 * 1024**2
LANGUAGES = ("html", "css", "javascript")
SOURCE_SHA = "4bb5742df59f0c56aaac34fc3129012b4617acf7d86a1804e47c5aa69e541a07"
BUNDLE_SHA = "14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55"


class ResearchRefusal(ValueError):
    """Refuse invalid or unauthorized input, never return a fabricated pass."""


def digest_file(path: Path, limit: int | None = MAX_JSONL) -> str:
    if path.is_symlink() or not path.is_file():
        raise ResearchRefusal("Expected regular non-symlink source")
    digest, n = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            n += len(block)
            if limit is not None and n > limit:
                raise ResearchRefusal("Input exceeds review byte budget")
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path, *, limit: int = MAX_JSONL) -> list[dict[str, Any]]:
    digest_file(path, limit)
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise ResearchRefusal("Expected nonempty canonical LF JSONL")
    try:
        entries = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    except (UnicodeError, ValueError) as exc:
        raise ResearchRefusal("Invalid UTF-8 JSONL") from exc
    if any(type(value) is not dict for value in entries):
        raise ResearchRefusal("Every JSONL row must be an object")
    return entries


def exact_patch(row: dict[str, Any], before: str, after: str) -> bool:
    original, replacement = row["oldText"], row["newText"]
    return before.count(original) == 1 and before.replace(original, replacement, 1) == after


def benchmark() -> list[dict[str, Any]]:
    rows = read_jsonl(BENCHMARK)
    if len(rows) != 36:
        raise ResearchRefusal("Benchmark count changed")
    ids, families, fingerprints = set(), set(), set()
    per_lang = Counter()
    for row in rows:
        required = {"schemaVersion", "id", "language", "family", "request", "inputFile",
                    "expectedFile", "oldText", "newText", "sourceOrigin", "split",
                    "trainingAuthorized"}
        if set(row) != required or row["schemaVersion"] != 1:
            raise ResearchRefusal("Malformed benchmark row")
        if (row["language"] not in LANGUAGES or row["id"] in ids or row["family"] in families
                or row["sourceOrigin"] != "independently-authored-m8-evaluation"
                or row["split"] != "development-evaluation-only"
                or row["trainingAuthorized"] is not False
                or not all(type(row[k]) is str and row[k] for k in
                           ("id", "family", "request", "inputFile", "expectedFile",
                            "oldText", "newText"))
                or not exact_patch(row, row["inputFile"], row["expectedFile"])):
            raise ResearchRefusal("Benchmark provenance, uniqueness or patch invalid")
        ids.add(row["id"])
        families.add(row["family"])
        per_lang[row["language"]] += 1
        fingerprints.add(hashlib.sha256(row["inputFile"].encode()).hexdigest())
    if dict(per_lang) != {key: 12 for key in LANGUAGES} or len(fingerprints) != 36:
        raise ResearchRefusal("Benchmark language balance or distinct input violated")
    return rows


def candidate_audit() -> dict[str, Any]:
    """Review the staged records only: NEVER upgrades rights or training approval."""
    held = benchmark()
    reserved_texts = {
        hashlib.sha256(r[k].encode("utf-8")).hexdigest()
        for r in held for k in ("inputFile", "expectedFile", "request")
    }
    rows = read_jsonl(CANDIDATES)
    if len(rows) != 1152:
        raise ResearchRefusal("Expected exactly 1152 synthetic candidate records")
    seen_ids, seen_pairs = set(), set()
    languages, families = Counter(), Counter()
    for r in rows:
        required = {"schemaVersion", "id", "language", "family", "templateGroup",
                    "syntheticVariation", "request", "sourceFile", "expectedFile",
                    "oldText", "newText", "provenance", "rightsReviewStatus",
                    "candidateStatus", "trainingAuthorized", "evaluationOnly"}
        if set(r) != required or r["schemaVersion"] != 1:
            raise ResearchRefusal("Unexpected synthetic candidate fields")
        if (r["language"] not in LANGUAGES
                or r["rightsReviewStatus"] != "pending-owner-review"
                or r["candidateStatus"] != "quarantined-pending-qa-and-approval"
                or r["trainingAuthorized"] is not False
                or r["evaluationOnly"] is not False
                or r["provenance"] != "original-project-authored-synthetic-edit-template"
                or type(r["syntheticVariation"]) is not int
                or not 0 <= r["syntheticVariation"] < 64
                or not all(type(r[k]) is str and r[k] for k in
                           ("id", "language", "family", "templateGroup", "request",
                            "sourceFile", "expectedFile", "oldText", "newText"))
                or not exact_patch(r, r["sourceFile"], r["expectedFile"])):
            raise ResearchRefusal("Unapproved candidate or invalid patch semantics")
        if r["id"] in seen_ids or (r["sourceFile"], r["expectedFile"]) in seen_pairs:
            raise ResearchRefusal("Duplicate synthetic candidate")
        if any(hashlib.sha256(r[k].encode("utf-8")).hexdigest() in reserved_texts
               for k in ("sourceFile", "expectedFile", "request")):
            raise ResearchRefusal("Exact benchmark contamination")
        seen_ids.add(r["id"])
        seen_pairs.add((r["sourceFile"], r["expectedFile"]))
        languages[r["language"]] += 1
        families[(r["language"], r["family"])] += 1
        if r["templateGroup"] != f"r2-{r['language']}-{r['family']}":
            raise ResearchRefusal("Synthetic lineage does not match declared group")
    if (dict(languages) != {lang: 384 for lang in LANGUAGES}
            or len(families) != 18 or set(families.values()) != {64}):
        raise ResearchRefusal("Candidate distribution unexpected")
    return {
        "status": "quarantined-pending-owner-review",
        "candidateJsonlSha256": digest_file(CANDIDATES),
        "benchmarkJsonlSha256": digest_file(BENCHMARK),
        "recordCount": len(rows),
        "byLanguage": dict(languages), "templateFamilies": len(families),
        "variationsPerTemplateFamily": 64,
        "distinctInputOutputPairs": len(seen_pairs),
        "exactBenchmarkStringOverlaps": 0, "evaluationRows": len(held),
        "approvedTrainingRows": 0, "trainingAuthorized": False,
        "qualityCaveat": "Only 18 synthetic edit patterns; 1152 rows are not 1152 independent source documents",
    }


def grade(responses: Path) -> dict[str, Any]:
    """Exact-file, minimal edit and preserved surrounding context; no code execution."""
    truth = benchmark()
    reference = {r["id"]: r for r in truth}
    supplied = read_jsonl(responses)
    answers: dict[str, str] = {}
    for item in supplied:
        if (set(item) != {"id", "output"} or type(item["id"]) is not str
                or item["id"] not in reference or item["id"] in answers
                or type(item["output"]) is not str
                or len(item["output"].encode("utf-8")) > 65536):
            raise ResearchRefusal("Malformed, duplicate or unknown model response")
        answers[item["id"]] = item["output"]
    details = []
    counters = {lang: Counter() for lang in LANGUAGES}
    for r in truth:
        output = answers.get(r["id"])
        score = {"id": r["id"], "language": r["language"], "family": r["family"],
                 "scored": output is not None, "exactFile": False,
                 "targetPresent": False, "untouchedContextPreserved": False}
        if output is not None:
            prefix, suffix = r["inputFile"].split(r["oldText"], 1)
            score.update(
                exactFile=(output == r["expectedFile"]),
                targetPresent=(r["newText"] in output and r["oldText"] not in output.replace(r["newText"], "", 1)),
                untouchedContextPreserved=(output.startswith(prefix) and output.endswith(suffix)),
            )
            counters[r["language"]]["scored"] += 1
            for label in ("exactFile", "targetPresent", "untouchedContextPreserved"):
                counters[r["language"]][label] += int(score[label])
        details.append(score)
    summaries = {lang: {**{k: c[k] for k in
                           ("scored", "exactFile", "targetPresent", "untouchedContextPreserved")},
                         "total": 12, "unscored": 12 - c["scored"]}
                 for lang, c in counters.items()}
    aggregate = {k: sum(s[k] for s in summaries.values())
                 for k in ("scored", "exactFile", "targetPresent", "untouchedContextPreserved", "unscored")}
    aggregate["total"] = 36
    return {
        "protocol": "plex600-m8-independent-edit-evaluation-v1",
        "benchmarkSha256": digest_file(BENCHMARK),
        "responseSha256": digest_file(responses),
        "source": "caller-provided-model-outputs-not-executed-by-scorer",
        "notIndependentFinalHoldout": True,
        "metrics": aggregate, "byLanguage": summaries, "cases": details,
        "behavioralTestsRun": False, "trainingAuthorized": False,
    }


def m7_local_receipt(report_path: Path, checkpoint: Path | None = None) -> dict[str, Any]:
    """Validate original report structure and optionally stream-hash weights.

    A parsed report without weights is *not* independent proof of a checkpoint.
    """
    digest = digest_file(report_path, 512 * 1024)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if type(report) is not dict:
        raise ResearchRefusal("M7 JSON report must be an object")
    steps = report.get("steps")
    before, after = report.get("validationBefore"), report.get("validationAfter")
    if (report.get("protocol") != "plex600-m7-four-update-v1"
            or report.get("status") != "pass"
            or report.get("optimizerUpdatesCompleted") != 4
            or report.get("trainingAuthorized") is not True
            or report.get("source", {}).get("sourceManifestSha256") != SOURCE_SHA
            or report.get("source", {}).get("bundleManifestSha256") != BUNDLE_SHA
            or type(steps) is not list or len(steps) != 4
            or type(before) is not dict or type(after) is not dict):
        raise ResearchRefusal("M7 run identity/status/scope invalid")
    for i, row in enumerate(steps, start=1):
        if (type(row) is not dict or row.get("step") != i
                or type(row.get("supervisedTokens")) is not int
                or not 1 <= row["supervisedTokens"] <= 128
                or type(row.get("lossBeforeUpdate")) not in (int, float)
                or not math.isfinite(row["lossBeforeUpdate"])):
            raise ResearchRefusal("M7 update measurement invalid")
    if len({row["recordId"] for row in steps}) != 4:
        raise ResearchRefusal("Duplicate M7 train example")
    for obj in (before, after):
        if (obj.get("records") != 24 or obj.get("targetTokens") != 1744
                or type(obj.get("loss")) not in (int, float)
                or not math.isfinite(obj["loss"])):
            raise ResearchRefusal("M7 internal validation invalid")
    spec = report.get("checkpoint")
    if (type(spec) is not dict or spec.get("file") != "weights.pt"
            or spec.get("weightsOnly") is not True
            or type(spec.get("bytes")) is not int
            or not 0 < spec["bytes"] <= 3 * 1024**3
            or type(spec.get("sha256")) is not str
            or len(spec["sha256"]) != 64):
        raise ResearchRefusal("Checkpoint evidence absent or invalid")
    checked = None
    if checkpoint is not None:
        if checkpoint.stat().st_size != spec["bytes"]:
            raise ResearchRefusal("Checkpoint size mismatch")
        checked = digest_file(checkpoint, 3 * 1024**3) == spec["sha256"]
        if checked is not True:
            raise ResearchRefusal("Checkpoint SHA-256 mismatch")
    return {
        "protocol": "plex600-m7-local-evidence-review-v1",
        "status": "report-verified" if checkpoint is None else "report-and-checkpoint-verified",
        "localReportSha256": digest,
        "localCheckpointSha256": spec["sha256"],
        "checkpointReadAndHashVerified": checked,
        "updates": len(steps),
        "supervisedTokenPositions": sum(row["supervisedTokens"] for row in steps),
        "validationBefore": before, "validationAfter": after,
        "lossDecrease": before["loss"] - after["loss"],
        "optimizerTrainingPerformedByThisAudit": False,
        "trainingAuthorizedByThisAudit": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="operation", required=True)
    actions.add_parser("candidate-audit")
    actions.add_parser("benchmark-inspect")
    score = actions.add_parser("score")
    score.add_argument("--responses", type=Path, required=True)
    evidence = actions.add_parser("m7-receipt")
    evidence.add_argument("--report", type=Path, required=True)
    evidence.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.operation == "candidate-audit":
            result = candidate_audit()
        elif args.operation == "benchmark-inspect":
            result = {"status": "development-benchmark-only", "rows": len(benchmark()),
                      "sha256": digest_file(BENCHMARK), "trainingAuthorized": False}
        elif args.operation == "score":
            result = grade(args.responses)
        else:
            result = m7_local_receipt(args.report, args.checkpoint)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
        print("Plex 600M research refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
