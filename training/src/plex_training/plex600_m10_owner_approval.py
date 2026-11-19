"""Owner-operated M10 approval receipt, separate from the GPU training worker.

DEFAULT and --list-baselines: read only. --review-exact-scope: CPU-only
verified source review. --record-owner-approval: explicit owner-invoked
external JSON file creation with O_EXCL, no overwrite and no training.

This tool does NOT claim cryptographic authentication or run a GPU worker.
The owner must separately execute M10 training after source/disk/GPU gates.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import sys
from typing import Any

from . import plex600_m10_admission as gate
from . import plex600_m10_copy_proposal as m10
from . import plex600_research as research

PROTOCOL = "plex600-m10-owner-external-consent-v1"
CONFIRM = "APPROVE M10 12 TRAIN 6 HELDOUT 50 UPDATES 3600 SECONDS"


class Refusal(ValueError):
    pass


def plan() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "status": "owner-approval-not-recorded",
        "sourceOrigin": m10.ORIGIN,
        "approvalKind": gate.KIND,
        "trainingExamples": 12, "heldoutEvaluationExamples": 6,
        "maximumOptimizerUpdates": 50, "hardTimeoutSeconds": 3600,
        "ownerConfirmationRequired": CONFIRM,
        "doesNotApproveAutomatically": True,
        "trainingStarted": False,
        "checkpointWrites": False,
        "modelLoaded": False,
    }


def list_baselines(directory: Path) -> list[dict[str, Any]]:
    """List candidate paths, clearly UNVERIFIED, without loading model/data."""
    if directory.is_symlink() or not directory.is_dir():
        raise Refusal("M10 baseline root is missing or a symlink")
    found: list[dict[str, Any]] = []
    for folder in sorted(directory.glob("plex600-m10-m9-full-baseline-*")):
        if folder.is_symlink() or not folder.is_dir():
            continue
        report = folder / "report.json"
        responses = folder / "responses.jsonl"
        if (not report.is_file() or not responses.is_file()
                or report.is_symlink() or responses.is_symlink()
                or report.stat().st_size > gate.BASELINE_MAX_BYTES
                or responses.stat().st_size > gate.RESPONSES_MAX_BYTES):
            continue
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            if (type(data) is not dict
                    or data.get("selectedCases") != 18
                    or data.get("decodingPolicies") != list(gate.DECODERS)
                    or data.get("status") != "read-only-m9-generation-complete"):
                continue
        except (OSError, ValueError, TypeError):
            continue
        found.append({
            "status": "located-not-yet-SHA-verified",
            "baselineDir": str(folder),
            "baselineReport": str(report),
            "baselineResponses": str(responses),
        })
    return found


def _required_sources(args: Any) -> tuple[Path, str, Path, Path, Path, Path]:
    values = (
        args.bundle_dir, args.bundle_manifest_sha256,
        args.m9_report, args.m9_weights, args.baseline_report,
        args.baseline_responses,
    )
    if not all(values):
        raise Refusal("Owner review requires exact bundle, M9 backup and full M9 baseline paths")
    return values


def review_sources(args: Any) -> dict[str, Any]:
    # All of these checks are CPU only, and sha-verify the real 2.4 GB backup.
    inspected = gate.inspect_sources(*_required_sources(args))
    template = inspected["unapprovedTemplate"]
    rows = m10.rows()
    return {
        **plan(),
        "status": "exact-scope-reviewed-source-verified-no-training",
        "candidateSha256": template["candidateSha256"],
        "m9WeightsSha256": template["m9WeightsSha256"],
        "m9ReportSha256": template["m9ReportSha256"],
        "baselineReportSha256": template["baselineReportSha256"],
        "baselineResponsesSha256": template["baselineResponsesSha256"],
        "trainAndHeldoutRecordDetails": [
            {
                "id": row["id"], "language": row["language"],
                "split": row["split"], "snippet": row["snippet"],
                "sourceOrigin": row["sourceOrigin"],
                "trainingAuthorizedInCandidate": row["trainingAuthorized"],
            }
            for row in rows
        ],
        "unapprovedExactScopeTemplate": template,
        "ownerApprovalRecorded": False,
        "trainingStarted": False,
    }


def _external_output(path: Path) -> Path:
    """Approval cannot be committed to the repo or replace an existing file."""
    if path.name != "m10-owner-approval.json":
        raise Refusal("M10 owner approval filename must be m10-owner-approval.json")
    if path.exists() or path.is_symlink():
        raise Refusal("M10 owner approval path already exists; cannot overwrite")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise Refusal("M10 owner approval parent must be an existing real folder")
    resolved = path.resolve(strict=False)
    root = research.ROOT.resolve(strict=True)
    if resolved == root or root in resolved.parents:
        raise Refusal("M10 approval must remain outside the Git repository")
    if path.parent.resolve(strict=True) == root:
        raise Refusal("M10 approval must remain outside Git")
    return path


def record_owner_approval(args: Any) -> dict[str, Any]:
    if args.confirm_owner_review != CONFIRM:
        raise Refusal("M10 explicit owner confirmation does not match exact 12/6/50/3600 scope")
    if args.approval_file is None:
        raise Refusal("M10 approval requires an external owner-controlled file path")
    target = _external_output(args.approval_file)
    verified = review_sources(args)
    template = verified["unapprovedExactScopeTemplate"]
    # No change of scope, limits, optimizer, source hashes, or candidate rows.
    approved = {
        **template,
        "approvedBy": "keepithandy",
        "approvedAt": date.today().isoformat(),
        "approvedForTraining": True,
        "sourceReviewedByOwner": True,
        "assistantProposedSourcesAcceptedForTraining": True,
    }
    if (approved["maxOptimizerUpdates"] != 50
            or approved["timeBudgetSeconds"] != 3600
            or approved["phaseUpdateCounts"] != [10, 15, 25]
            or len(approved["trainRecordIds"]) != 12
            or len(approved["heldoutRecordIds"]) != 6
            or approved["finalHoldoutAllowed"] is not False):
        raise Refusal("M10 exact approved schedule or source boundaries changed")
    data = (json.dumps(approved, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > gate.APPROVAL_MAX_BYTES:
        raise Refusal("M10 external approval file would exceed strict size cap")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(target, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    # The existing independent validator is the final authority.
    proof = gate.check_approval(target, template)
    return {
        "protocol": PROTOCOL,
        "status": "explicit-external-owner-approval-recorded-not-training",
        "ownerApprovalPath": str(target),
        "ownerApprovalSha256": proof["approvalFileSha256"],
        "candidateSha256": approved["candidateSha256"],
        "m9WeightsSha256": approved["m9WeightsSha256"],
        "baselineResponsesSha256": approved["baselineResponsesSha256"],
        "maxOptimizerUpdates": approved["maxOptimizerUpdates"],
        "hardTimeoutSeconds": approved["timeBudgetSeconds"],
        "trainingStarted": False,
        "checkpointWrites": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-baselines", action="store_true")
    modes.add_argument("--review-exact-scope", action="store_true")
    modes.add_argument("--record-owner-approval", action="store_true")
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--bundle-manifest-sha256")
    parser.add_argument("--m9-report", type=Path)
    parser.add_argument("--m9-weights", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--baseline-responses", type=Path)
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--confirm-owner-review")
    args = parser.parse_args(argv)
    try:
        if args.list_baselines:
            if any(getattr(args, field) is not None for field in (
                "bundle_dir", "bundle_manifest_sha256", "m9_report",
                "m9_weights", "baseline_report", "baseline_responses",
                "approval_file", "confirm_owner_review",
            )):
                raise Refusal("Read-only baseline discovery does not accept approval inputs")
            result = {
                **plan(),
                "status": "unverified-baseline-path-discovery-no-training",
                "baselines": list_baselines(research.ROOT / "training/artifacts"),
            }
        elif args.review_exact_scope:
            if args.approval_file is not None or args.confirm_owner_review is not None:
                raise Refusal("Read-only review cannot record approval")
            result = review_sources(args)
        elif args.record_owner_approval:
            result = record_owner_approval(args)
        else:
            if any(getattr(args, field) is not None for field in (
                "bundle_dir", "bundle_manifest_sha256", "m9_report",
                "m9_weights", "baseline_report", "baseline_responses",
                "approval_file", "confirm_owner_review",
            )):
                raise Refusal("Approval inputs require an explicit review or record flag")
            result = plan()
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print("M10 owner consent refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
