"""Plex 600M Phase 1: locally verify and register the original M9/M10 evidence.

No model or optimizer is instantiated. Default CLI is inert. Only an explicit
--register creates a NEW gitignored output directory. Historical training and
GPU-origin authenticity cannot be established solely from saved receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from . import plex600_phase1_registry as registry
from . import plex600_m10_admission as gate
from . import plex600_m10_copy_proposal as proposal
from . import plex600_m10_worker as worker
from . import plex600_research as research

PROTOCOL = "plex600-phase1-local-history-reconciliation-v1"
ORIGINAL_M9_SHA = (
    "e913be27855548a8f7dea766a2d367c7eaea357014ca82c80dec088958dfb123"
)
MILESTONES = (10, 25, 50)
SCOPES = ("train", "heldout")
DATASET_STATUS = "12-train-six-evaluation-only-not-final-holdout"
MAX_REPORT = 2 * 1024**2
MAX_RESPONSES = 2 * 1024**2
MAX_WEIGHTS = 3 * 1024**3


class Refusal(ValueError):
    pass


def plan() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "status": "plan-only-no-file-access",
        "requiredEvidence": [
            "separate-M9-backup", "frozen-M7-tokenizer",
            "full-M9-M10-baseline-18-by-2",
            "completed-M10-run-with-all-claimed-snapshots",
        ],
        "modelLoaded": False,
        "optimizerCreated": False,
        "trainingExecuted": False,
        "historicalWeightsModified": False,
        "finalHoldoutUsed": False,
    }


def _regular_file(path: Path, limit: int) -> dict[str, Any]:
    if path.is_symlink() or path.parent.is_symlink():
        raise Refusal("Refuse linked checkpoint or receipt path")
    if not path.is_file():
        raise Refusal("Required checkpoint or receipt file is missing")
    evidence = registry._file_hash(path, limit)
    return {"path": str(path.resolve(strict=True)), **evidence}


def _read_report(path: Path) -> dict[str, Any]:
    _regular_file(path, MAX_REPORT)
    return registry._read_json(path)


def _snapshot_checks(
    run_dir: Path, report: dict[str, Any], source: dict[str, Any],
    baseline_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    updates = report.get("optimizerUpdatesCompleted")
    if type(updates) is not int or updates not in (25, 50):
        raise Refusal("M10 history must be a completed 25- or 50-update run")
    milestones = MILESTONES[:2] if updates == 25 else MILESTONES
    saved = report.get("snapshots")
    steps = report.get("steps")
    schedule = worker.exact_schedule()
    if (type(saved) is not list
            or len(saved) != len(milestones)
            or type(steps) is not list or len(steps) != updates):
        raise Refusal("M10 final report has incomplete snapshots or optimizer receipts")
    for recorded, expected in zip(steps, schedule):
        if (type(recorded) is not dict
                or recorded.get("step") != expected["optimizerUpdate"]
                or recorded.get("recordId") != expected["recordIds"][0]
                or recorded.get("promptPositionsIncorrectlySupervised") != 0
                or recorded.get("eosTargetSupervised") is not True):
            raise Refusal("M10 optimizer ledger differs from exact frozen schedule")
    snapshots = []
    comparisons = []
    previous_entry = "m9-original"
    previous_sha = source["m9WeightsSha256"]
    for step, recorded in zip(milestones, saved):
        if type(recorded) is not dict or recorded.get("optimizerUpdatesCompleted") != step:
            raise Refusal("M10 milestone order differs from the training contract")
        folder = run_dir / f"step-{step}"
        checkpoint = _regular_file(folder / "weights.pt", MAX_WEIGHTS)
        raw = _regular_file(folder / "responses.jsonl", MAX_RESPONSES)
        saved_report = _regular_file(folder / "report.json", MAX_REPORT)
        snapshot_report = _read_report(folder / "report.json")
        if snapshot_report != recorded:
            raise Refusal("M10 milestone report differs from parent summary")
        checkpoint_meta = recorded.get("checkpoint")
        if (type(checkpoint_meta) is not dict
                or recorded.get("checkpointSha256") != checkpoint["sha256"]
                or checkpoint_meta.get("sha256") != checkpoint["sha256"]
                or checkpoint_meta.get("bytes") != checkpoint["bytes"]
                or recorded.get("responsesSha256") != raw["sha256"]
                or recorded.get("candidateSha256") != source["candidateSha256"]
                or recorded.get("optimizerUpdatesCompleted") != step
                or recorded.get("verifiedGeneration") is not True
                or recorded.get("decodingPolicies") != list(gate.DECODERS)
                or recorded.get("finalHoldoutUsed") is not False):
            raise Refusal("M10 milestone checkpoint/response source binding invalid")
        worker.verify_checkpoint_responses(
            folder / "responses.jsonl", checkpoint["sha256"], recorded["scores"])
        comparison = registry.compare_m10(baseline_dir, folder)
        comparisons.append({"step": step, **comparison})
        snapshots.append({
            "entryId": f"m10-step-{step}",
            "modelFamily": "plex-600m",
            "optimizerUpdatesDeclared": step,
            "checkpoint": checkpoint, "report": saved_report,
            "responses": raw, "scoreVerification": "recomputed-from-raw-responses",
            "parentEntryId": previous_entry,
            "parentCheckpointSha256": previous_sha,
        })
        previous_entry = f"m10-step-{step}"
        previous_sha = checkpoint["sha256"]
    return snapshots, comparisons


def audit(
    *, bundle_dir: Path, m9_report: Path, m9_weights: Path,
    baseline_dir: Path, m10_run_dir: Path,
    approval_file: Path | None = None,
) -> dict[str, Any]:
    """Rehash original local evidence and rescore all M10 responses; no GPU."""
    if (baseline_dir.is_symlink() or m10_run_dir.is_symlink()
            or baseline_dir.resolve(strict=True) == m10_run_dir.resolve(strict=True)):
        raise Refusal("Baseline and M10 run must be distinct real directories")
    proof = gate.inspect_sources(
        bundle_dir, research.BUNDLE_SHA, m9_report, m9_weights,
        baseline_dir / "report.json", baseline_dir / "responses.jsonl")
    backup = proof["backup"]
    baseline = proof["baseline"]
    source = proof["tokenAudit"]
    if backup["m9WeightsSha256"] != ORIGINAL_M9_SHA:
        raise Refusal("Historical M9 model hash differs from recorded original")
    expected_candidate_sha = proposal.audit()["candidateSha256"]
    if source["candidateSha256"] != expected_candidate_sha:
        raise Refusal("M10 data source drifted from the historical proposal")
    m10_report_path = m10_run_dir / "report.json"
    parent = _read_report(m10_report_path)
    approval = None if approval_file is None else _regular_file(
        approval_file, gate.APPROVAL_MAX_BYTES)
    if (parent.get("protocol") != worker.PROTOCOL
            or parent.get("status") not in (
                "bounded-50-updates-complete-unreviewed",
                "stopped-after-two-perfect-milestones")
            or parent.get("trainingAuthorized") is not True
            or parent.get("maxOptimizerUpdates") != 50
            or parent.get("optimizerStateSaved") is not False
            or parent.get("finalHoldoutUsed") is not False
            or parent.get("historicalCheckpointWrites") is not False
            or parent.get("originalM9WeightsSha256") != backup["m9WeightsSha256"]
            or parent.get("originalM9ReportSha256") != backup["m9ReportSha256"]
            or parent.get("baselineReportSha256") != baseline["baselineReportSha256"]
            or parent.get("baselineResponsesSha256") != baseline["baselineResponsesSha256"]
            or parent.get("candidateSha256") != expected_candidate_sha
            or parent.get("scheduleSha256")
               != gate.schedule_sha256(worker.exact_schedule())
            or type(parent.get("approvalSha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", parent["approvalSha256"]) is None
            or (approval is not None
                and approval["sha256"] != parent["approvalSha256"])):
        raise Refusal("M10 parent report source/approval/baseline scope invalid")
    snapshot_items, comparisons = _snapshot_checks(
        m10_run_dir, parent, {
            "m9WeightsSha256": backup["m9WeightsSha256"],
            "candidateSha256": expected_candidate_sha,
        }, baseline_dir)
    m9_entry = {
        "entryId": "m9-original",
        "modelFamily": "plex-600m",
        "optimizerUpdatesDeclared": backup["optimizerUpdatesVerified"],
        "checkpoint": _regular_file(m9_weights, MAX_WEIGHTS),
        "report": _regular_file(m9_report, MAX_REPORT),
        "responses": _regular_file(baseline_dir / "responses.jsonl", MAX_RESPONSES),
        "baselineReport": _regular_file(baseline_dir / "report.json", MAX_REPORT),
        "parentEntryId": None,
        "parentCheckpointSha256": None,
        "originalM7WeightsSha256FromReport":
            "69a31b4e710b284581a0b74c9e2661cc27bf134d24b8b1fe7a906f6efc004224",
        "sourceStatus": "separate-M9-backup-checkpoint-and-report-verified",
    }
    if m9_entry["checkpoint"]["sha256"] != backup["m9WeightsSha256"]:
        raise Refusal("M9 checkpoint unexpectedly changed during audit")
    if m9_entry["report"]["sha256"] != backup["m9ReportSha256"]:
        raise Refusal("M9 report unexpectedly changed during audit")
    if m9_entry["responses"]["sha256"] != baseline["baselineResponsesSha256"]:
        raise Refusal("M9 baseline responses changed during audit")
    if m9_entry["baselineReport"]["sha256"] != baseline["baselineReportSha256"]:
        raise Refusal("M9 baseline report changed during audit")
    identities = {
        "bundleDirectory": str(bundle_dir.resolve(strict=True)),
        "bundleManifestSha256": research.BUNDLE_SHA,
        "tokenizerSha256": source["tokenizerSha256"],
        "candidateSha256": expected_candidate_sha,
        "scheduleSha256": parent["scheduleSha256"],
        "originalM9WeightsSha256": backup["m9WeightsSha256"],
        "m9BackupReportSha256": backup["m9ReportSha256"],
        "baselineReportSha256": baseline["baselineReportSha256"],
        "baselineResponsesSha256": baseline["baselineResponsesSha256"],
        "m10Report": _regular_file(m10_report_path, MAX_REPORT),
        "historicalApprovalReceipt": approval,
        "historicalApprovalSha256Declared": parent["approvalSha256"],
        "datasetRole": DATASET_STATUS,
    }
    ledger = [{
        "experimentId": "m9-r3-eight-update", "decision": "retain-original-baseline",
        "checkpointEntryId": "m9-original",
        "optimizerUpdatesCompleted": backup["optimizerUpdatesVerified"],
        "reason": "Baseline identity validated; task improvement not established here.",
    }]
    for item in comparisons:
        gains = [v["deltaExactAndEos"] for v in item["scores"].values()]
        exact = [v["candidateExactAndEos"] for v in item["scores"].values()]
        ledger.append({
            "experimentId": f"m10-update-{item['step']}",
            "decision": ("no-promotion-on-exact-copy" if sum(exact) == 0
                         else "manual-review-needed"),
            "checkpointEntryId": f"m10-step-{item['step']}",
            "optimizerUpdatesCompleted": item["step"],
            "exactAndEosAcross36": sum(exact),
            "exactAndEosDeltaAcross36": sum(gains),
            "reason": "Autonomous exact copy must improve; EOS-only changes do not qualify.",
        })
    ledger.append({
        "experimentId": "m11-source-conditioning",
        "decision": "not-evaluated",
        "checkpointEntryId": None, "optimizerUpdatesCompleted": 0,
        "reason": "M11 M9-space-prompt baseline and approved GPU worker not completed.",
    })
    return {
        "protocol": PROTOCOL,
        "status": "owner-local-checkpoint-files-and-response-scores-reverified",
        "verificationScope": (
            "actual-local-byte-hashes-and-M10-receipt-consistency;"
            "saved-receipts-do-not-independently-prove-GPU-origin"
        ),
        "identities": identities,
        "entries": [m9_entry, *snapshot_items],
        "comparisons": comparisons,
        "decisionLedger": ledger,
        "sourceApprovalStatus": (
            "existing-M10-approval-receipt-hash-verified" if approval is not None
            else "historical-approval-sha-declared-only"),
        "trainingExecutedByThisTool": False,
        "modelLoaded": False,
        "finalHoldoutUsedByThisTool": False,
        "historicalCheckpointWrites": False,
        "trainingAuthorizedByThisTool": False,
    }


def _local_output(directory: Path) -> Path:
    root = (research.ROOT / "training/artifacts").resolve(strict=True)
    resolved = directory.resolve(strict=False)
    if (directory.is_symlink() or directory.parent.is_symlink()
            or resolved.parent != root or resolved.exists()):
        raise Refusal("Output must be a NEW direct child of gitignored training/artifacts")
    return resolved


def register(output: Path, record: dict[str, Any]) -> dict[str, Any]:
    target = _local_output(output)
    target.mkdir(mode=0o700)
    for filename, body in (
        ("checkpoint-registry.json", {
            "protocol": record["protocol"], "status": record["status"],
            "identities": record["identities"], "entries": record["entries"],
        }),
        ("historical-comparisons.json", {
            "protocol": record["protocol"], "comparisons": record["comparisons"],
            "verificationScope": record["verificationScope"],
        }),
        ("experiment-decisions.json", {
            "protocol": record["protocol"], "decisionLedger": record["decisionLedger"],
            "verificationScope": record["verificationScope"],
        }),
    ):
        with (target / filename).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(body, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return {
        "status": record["status"], "outputDirectory": str(target),
        "checkpointCount": len(record["entries"]),
        "comparisonCount": len(record["comparisons"]),
        "decisionCount": len(record["decisionLedger"]),
        "trainingExecutedByThisTool": False,
        "modelLoaded": False,
        "historicalWeightsModified": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit", action="store_true")
    modes.add_argument("--register", action="store_true")
    for option in ("bundle-dir", "m9-report", "m9-weights",
                   "baseline-dir", "m10-run-dir", "approval-file", "output-dir"):
        parser.add_argument("--" + option, type=Path)
    args = parser.parse_args(argv)
    try:
        if not (args.audit or args.register):
            if any(getattr(args, k) is not None for k in (
                "bundle_dir", "m9_report", "m9_weights", "baseline_dir",
                "m10_run_dir", "approval_file", "output_dir")):
                raise Refusal("Default mode does not accept checkpoint arguments")
            output = plan()
        else:
            if any(getattr(args, k) is None for k in (
                "bundle_dir", "m9_report", "m9_weights", "baseline_dir",
                "m10_run_dir")):
                raise Refusal("Audit requires exact original M9/M10 evidence paths")
            if args.audit and args.output_dir is not None:
                raise Refusal("Read-only audit does not accept output path")
            if args.register:
                if args.output_dir is None:
                    raise Refusal("Registration requires a create-only output directory")
                _local_output(args.output_dir)  # refuse before hashing large weights
            record = audit(
                bundle_dir=args.bundle_dir, m9_report=args.m9_report,
                m9_weights=args.m9_weights, baseline_dir=args.baseline_dir,
                m10_run_dir=args.m10_run_dir, approval_file=args.approval_file)
            output = (register(args.output_dir, record)
                      if args.register else record)
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (Refusal, registry.Refusal, gate.Refusal, proposal.Refusal,
            worker.Refusal, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
