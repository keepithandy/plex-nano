"""Phase 1: choose the exact owner-local M10 baseline and verify history.

Path discovery reads no weights. An explicit audit/registration reuses the
bounded, source-linked M9/M10 CPU evidence checks; no GPU, optimizer or model
is loaded. Local output is create-only and stays in ignored artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

from . import plex600_phase1_historical as history
from . import plex600_phase1_registry as registry
from . import plex600_m10_owner_approval as consent
from . import plex600_m10_worker as worker
from . import plex600_research as research

PROTOCOL = "plex600-phase1-auto-registration-v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
FILES = (
    "checkpoint-registry.json", "historical-comparisons.json",
    "experiment-decisions.json",
)


class Refusal(ValueError):
    pass


def plan() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL, "status": "plan-only-no-file-access",
        "availableModes": [
            "discover-paths-only", "audit-original-evidence",
            "register-create-only", "verify-registered-history",
        ],
        "trainingExecuted": False, "modelLoaded": False,
        "checkpointsModified": False, "finalHoldoutUsed": False,
    }


def _artifact_root() -> Path:
    root = research.ROOT / "training/artifacts"
    if root.is_symlink() or not root.is_dir():
        raise Refusal("Local artifact root missing or linked")
    return root.resolve(strict=True)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else research.ROOT / path


def _direct_child(directory: Path, parent: Path) -> Path:
    if directory.is_symlink() or not directory.is_dir():
        raise Refusal("Artifact directory missing or linked")
    dest = directory.resolve(strict=True)
    if dest.parent != parent:
        raise Refusal("Artifact directory must be a direct child of local artifact root")
    return dest


def _run_candidates(artifacts: Path) -> list[Path]:
    matches = []
    for path in sorted(artifacts.glob("plex600-m10-approved-*")):
        if path.is_symlink() or not path.is_dir():
            continue
        report = path / "report.json"
        if report.is_symlink() or not report.is_file():
            continue
        try:
            obj = registry._read_json(report)
        except (registry.Refusal, OSError):
            continue
        if (obj.get("protocol") == worker.PROTOCOL
                and obj.get("status") in (
                    "bounded-50-updates-complete-unreviewed",
                    "stopped-after-two-perfect-milestones")):
            matches.append(path)
        if len(matches) > 50:
            raise Refusal("Too many matching M10 histories; select explicitly")
    return matches


def discover(
    backup_dir: Path, *, run_dir: Path | None = None,
    bundle_dir: Path | None = None,
) -> dict[str, Any]:
    """No checkpoint hashing; require an exact baseline receipt-hash match.

    An absent or ambiguous baseline is a refusal. File-name similarity never
    establishes a correct source-to-checkpoint relationship.
    """
    artifacts = _artifact_root()
    backup = _repo_path(backup_dir)
    if backup.is_symlink() or not backup.is_dir():
        raise Refusal("Original M9 backup directory missing or linked")
    if (backup / "weights.pt").is_symlink() or (backup / "report.json").is_symlink():
        raise Refusal("Original M9 backup files must not be links")
    if not (backup / "weights.pt").is_file() or not (backup / "report.json").is_file():
        raise Refusal("Original M9 backup must contain weights.pt and report.json")
    bundle = _repo_path(bundle_dir) if bundle_dir else artifacts / "plex600-fresh-tokenizer"
    if bundle.is_symlink() or not bundle.is_dir():
        raise Refusal("Frozen M7 bundle directory is missing or linked")
    if run_dir is None:
        runs = _run_candidates(artifacts)
        if len(runs) != 1:
            raise Refusal(
                "Expected exactly one completed M10 run; specify --m10-run-dir"
                if runs else "No completed M10 run discovered")
        run = _direct_child(runs[0], artifacts)
    else:
        run = _direct_child(_repo_path(run_dir), artifacts)
    parent = registry._read_json(run / "report.json")
    if (parent.get("protocol") != worker.PROTOCOL
            or parent.get("status") not in (
                "bounded-50-updates-complete-unreviewed",
                "stopped-after-two-perfect-milestones")):
        raise Refusal("Selected M10 run is not a completed original worker run")
    report_sha = parent.get("baselineReportSha256")
    response_sha = parent.get("baselineResponsesSha256")
    if (type(report_sha) is not str or HEX64.fullmatch(report_sha) is None
            or type(response_sha) is not str
            or HEX64.fullmatch(response_sha) is None):
        raise Refusal("M10 run lacks exact historical baseline receipt hashes")
    baselines = consent.list_baselines(artifacts)
    if len(baselines) > 100:
        raise Refusal("Too many M10 baseline directories")
    matching = []
    for item in baselines:
        directory = _direct_child(Path(item["baselineDir"]), artifacts)
        left = registry._file_hash(directory / "report.json", 256 * 1024)
        right = registry._file_hash(directory / "responses.jsonl", 512 * 1024)
        if left["sha256"] == report_sha and right["sha256"] == response_sha:
            matching.append(directory)
    if len(matching) != 1:
        raise Refusal(
            "Cannot uniquely match the exact baseline to M10 run receipt hashes;"
            " preserve all baseline artifacts and investigate")
    return {
        "protocol": PROTOCOL,
        "status": "paths-and-baseline-receipt-hashes-matched-not-yet-full-audit",
        "m9Report": str(backup / "report.json"),
        "m9Weights": str(backup / "weights.pt"),
        "bundleDir": str(bundle.resolve(strict=True)),
        "m10RunDir": str(run),
        "baselineDir": str(matching[0]),
        "matchedBaselineReportSha256": report_sha,
        "matchedBaselineResponsesSha256": response_sha,
        "modelLoaded": False,
        "trainingExecuted": False,
    }


def verified_history(
    *, backup_dir: Path, run_dir: Path | None = None,
    bundle_dir: Path | None = None,
    approval_file: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = discover(backup_dir, run_dir=run_dir, bundle_dir=bundle_dir)
    result = history.audit(
        bundle_dir=Path(selected["bundleDir"]),
        m9_report=Path(selected["m9Report"]),
        m9_weights=Path(selected["m9Weights"]),
        baseline_dir=Path(selected["baselineDir"]),
        m10_run_dir=Path(selected["m10RunDir"]),
        approval_file=(_repo_path(approval_file)
                       if approval_file is not None else None),
    )
    return selected, result


def _existing_output(output: Path) -> Path:
    artifacts = _artifact_root()
    folder = _repo_path(output)
    return _direct_child(folder, artifacts)


def verify_registered(output: Path, fresh: dict[str, Any]) -> dict[str, Any]:
    """Re-evaluate source bytes, then check all three published JSON records."""
    folder = _existing_output(output)
    desired = {
        "checkpoint-registry.json": {
            "protocol": fresh["protocol"], "status": fresh["status"],
            "identities": fresh["identities"], "entries": fresh["entries"],
        },
        "historical-comparisons.json": {
            "protocol": fresh["protocol"], "comparisons": fresh["comparisons"],
            "verificationScope": fresh["verificationScope"],
        },
        "experiment-decisions.json": {
            "protocol": fresh["protocol"], "decisionLedger": fresh["decisionLedger"],
            "verificationScope": fresh["verificationScope"],
        },
    }
    digests = {}
    for filename in FILES:
        target = folder / filename
        actual = registry._read_json(target)
        if actual != desired[filename]:
            raise Refusal("Registered historical evidence disagrees with fresh audit: " + filename)
        digests[filename] = registry._file_hash(target, 2 * 1024**2)["sha256"]
    return {
        "protocol": PROTOCOL,
        "status": "local-registration-and-original-file-bytes-reverified",
        "registeredOutputDirectory": str(folder),
        "checkpointEntries": len(fresh["entries"]),
        "matchedCheckpointComparisons": len(fresh["comparisons"]),
        "experimentDecisions": len(fresh["decisionLedger"]),
        "registeredFileSha256": digests,
        "historicalInferenceOriginIndependentlyProven": False,
        "modelLoaded": False,
        "trainingExecuted": False,
        "historicalWeightsModified": False,
        "finalHoldoutUsed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--discover", action="store_true")
    action.add_argument("--audit", action="store_true")
    action.add_argument("--register", action="store_true")
    action.add_argument("--verify", action="store_true")
    for argument in ("m9-backup-dir", "m10-run-dir", "bundle-dir",
                     "approval-file", "output-dir"):
        parser.add_argument("--" + argument, type=Path)
    args = parser.parse_args(argv)
    try:
        if not any((args.discover, args.audit, args.register, args.verify)):
            if any(getattr(args, field) is not None for field in (
                "m9_backup_dir", "m10_run_dir", "bundle_dir",
                "approval_file", "output_dir")):
                raise Refusal("Default plan cannot accept filesystem arguments")
            result = plan()
        else:
            if args.m9_backup_dir is None:
                raise Refusal("An explicit original M9 backup directory is required")
            if args.discover and (args.output_dir or args.approval_file):
                raise Refusal("Discovery does not use approval or output locations")
            if args.audit and args.output_dir is not None:
                raise Refusal("Read-only audit does not create an output")
            if (args.register or args.verify) and args.output_dir is None:
                raise Refusal("Registration/verification requires an exact output directory")
            if args.register:
                # Validate output without opening or hashing multi-GB sources.
                history._local_output(_repo_path(args.output_dir))
            if args.discover:
                result = discover(
                    args.m9_backup_dir, run_dir=args.m10_run_dir,
                    bundle_dir=args.bundle_dir)
            else:
                selected, audited = verified_history(
                    backup_dir=args.m9_backup_dir, run_dir=args.m10_run_dir,
                    bundle_dir=args.bundle_dir, approval_file=args.approval_file)
                if args.audit:
                    result = {
                        "protocol": PROTOCOL, "selected": selected,
                        "audit": audited,
                    }
                elif args.register:
                    created = history.register(_repo_path(args.output_dir), audited)
                    result = {**created, "discovery": selected}
                else:
                    result = verify_registered(args.output_dir, audited)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (Refusal, history.Refusal, registry.Refusal, OSError, ValueError,
            RuntimeError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
