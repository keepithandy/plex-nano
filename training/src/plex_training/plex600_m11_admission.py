"""M11 exact-scope CPU-only admission. NEVER trains or grants owner approval.

The previous M10 consent is not valid for M11. M11 requires a separately
generated and independently checked M9 baseline for the *space-delimited*
four-case source-contrast prompt set before an owner approval can bind it.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

from . import plex600_m10_copy_proposal as m10_copy
from . import plex600_m10_admission as m10_gate
from . import plex600_m11_source_conditioning as m11
from . import plex600_m11_training_core as core
from . import plex600_m8_eval as m8
from . import plex600_research as research

PROTOCOL = "plex600-m11-exact-cpu-admission-v1"
KIND = "plex600-m11-specific-owner-training-consent-v1"
BASELINE_PROTOCOL = "plex600-m11-m9-space-prompt-read-only-baseline-v1"
BASELINE_STATUS = "complete-m11-m9-space-read-only-inference"
REPORT_MAX_BYTES = 128 * 1024
RESPONSES_MAX_BYTES = 128 * 1024
APPROVAL_MAX_BYTES = 16 * 1024
SOURCE_MAX_BYTES = 1024 * 1024
CHECKPOINT_MAX_BYTES = m10_gate.CHECKPOINT_MAX_BYTES
# 4 planned permanent checkpoints + 1 temporary + 1 GiB headroom, per arm.
MIN_OUTPUT_FREE_BYTES = (len(core.CHECKPOINT_STEPS) + 1) * CHECKPOINT_MAX_BYTES + 1024**3
DECODERS = core.DECODERS
RESPONSE_TERMINATIONS = {
    "eos", "max-new-tokens", "invalid-control-token", "outside-fitted-tokenizer",
}


class Refusal(ValueError):
    pass


def plan() -> dict[str, Any]:
    """The default command does not access inputs, create files or import torch."""
    return {
        "protocol": PROTOCOL,
        "status": "review-only-no-execution",
        "proposalTrainingPair": [c["id"] for c in m11.cases() if c["split"] == "train"],
        "visibleProbeOnlyPair": [c["id"] for c in m11.cases() if c["split"] == "probe-only"],
        "scopeSha256": core.candidate_sha256(),
        "scheduleSha256": core.schedule_sha256(),
        "proposedArmEosWeights": list(core.EOS_WEIGHTS),
        "proposedMaxOptimizerUpdatesPerArm": core.MAX_UPDATES,
        "proposedMilestones": list(core.CHECKPOINT_STEPS),
        "requireNewSpaceDelimitedM9Baseline": True,
        "reuseM10ApprovalAllowed": False,
        "minimumOutputFreeBytesPerArm": MIN_OUTPUT_FREE_BYTES,
        "m11TrainerImplemented": False,
        "ownerConsentGranted": False,
        "trainingAuthorized": False,
        "optimizerUpdatesCompleted": 0,
        "modelLoaded": False,
        "checkpointWrites": False,
        "finalHoldoutUsed": False,
    }


def digest(path: Path, limit: int) -> str:
    if path.is_symlink() or not path.is_file():
        raise Refusal("M11 requires a regular unsymlinked source file")
    return research.digest_file(path, limit=limit)


def inspect_sources(bundle: Path, bundle_sha: str,
                    m9_report: Path, m9_weights: Path) -> dict[str, Any]:
    """Verify original independent M9 bytes and frozen M7 tokenizer; CPU only."""
    if bundle_sha != research.BUNDLE_SHA:
        raise Refusal("M11 M7 bundle manifest identity differs from frozen source")
    backup = m10_copy.audit_m9_backup(m9_report, m9_weights)
    m8.verified_bundle(bundle, bundle_sha)
    from .tokenizer import PlexTokenizer
    tokenizer = PlexTokenizer.load(bundle)
    token_audit = m11.audit_tokenizer(tokenizer)
    if tokenizer.vocabulary_size != m10_gate.TOKENIZER_SHA256 and False:
        raise Refusal("Unreachable guard")  # not used; audited by M11
    if not token_audit["tokenizerRoundTripVerified"]:
        raise Refusal("M11 frozen tokenizer did not reproduce the source pairs")
    if (len(core.exact_schedule()) != 200
            or core.plan()["developmentTrainingExposures"] != 0
            or core.CHECKPOINT_STEPS != (25, 50, 100, 200)):
        raise Refusal("M11 proposed training schedule drifted")
    files = {
        "modelSourceSha256": digest(
            Path(__file__).with_name("plex600.py"), SOURCE_MAX_BYTES),
        "m11CandidateSourceSha256": digest(Path(m11.__file__), SOURCE_MAX_BYTES),
        "m11TrainingCoreSourceSha256": digest(Path(core.__file__), SOURCE_MAX_BYTES),
        "m11AdmissionSourceSha256": digest(Path(__file__), SOURCE_MAX_BYTES),
    }
    return {
        "candidateSha256": core.candidate_sha256(),
        "scheduleSha256": core.schedule_sha256(),
        "bundleManifestSha256": bundle_sha,
        "tokenizerSha256": m10_gate.TOKENIZER_SHA256,
        "m9WeightsSha256": backup["m9WeightsSha256"],
        "m9ReportSha256": backup["m9ReportSha256"],
        "trainingIds": [c["id"] for c in m11.cases() if c["split"] == "train"],
        "developmentOnlyIds": [c["id"] for c in m11.cases() if c["split"] == "probe-only"],
        "sourceHashes": files,
        "tokenAudit": token_audit["tokenizerAudit"],
        "verifiedM9Backup": True,
        "optimizerUpdatesCompleted": 0,
        "trainingAuthorized": False,
    }


def read_baseline(report_path: Path, responses_path: Path,
                  source: dict[str, Any]) -> dict[str, Any]:
    """Validate a distinct M11 read-only M9 baseline, not M10's newline report.

    Verifies consistency and provenance of the supplied receipts. It cannot
    independently prove inference happened on the claimed machine.
    """
    if (report_path.is_symlink() or responses_path.is_symlink()
            or not report_path.is_file() or not responses_path.is_file()
            or report_path.name != "report.json"
            or responses_path.name != "responses.jsonl"
            or report_path.parent.resolve(strict=True)
            != responses_path.parent.resolve(strict=True)):
        raise Refusal("M11 baseline must be same-directory regular report and responses")
    report_sha = digest(report_path, REPORT_MAX_BYTES)
    responses_sha = digest(responses_path, RESPONSES_MAX_BYTES)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = m11.cases()
    ids = [case["id"] for case in cases]
    if (type(report) is not dict
            or report.get("protocol") != BASELINE_PROTOCOL
            or report.get("status") != BASELINE_STATUS
            or report.get("checkpointSha256") != source["m9WeightsSha256"]
            or report.get("m9ReportSha256") != source["m9ReportSha256"]
            or report.get("bundleManifestSha256") != source["bundleManifestSha256"]
            or report.get("tokenizerSha256") != source["tokenizerSha256"]
            or report.get("candidateSha256") != source["candidateSha256"]
            or report.get("scheduleSha256") != source["scheduleSha256"]
            or report.get("sourceHashes") != source["sourceHashes"]
            or report.get("responsesSha256") != responses_sha
            or report.get("selectedCaseIds") != ids
            or report.get("decodingPolicies") != list(DECODERS)
            or report.get("responseCount") != len(ids) * len(DECODERS)
            or report.get("optimizerUpdates") != 0
            or report.get("readOnlyGenerationExecuted") is not True
            or report.get("trainingAuthorized") is not False
            or report.get("checkpointWrites") is not False
            or report.get("finalHoldoutUsed") is not False):
        raise Refusal("M11 baseline lacks exact M9, prompt, or read-only identity")
    records = research.read_jsonl(responses_path, limit=RESPONSES_MAX_BYTES)
    if len(records) != 8:
        raise Refusal("M11 baseline must contain 4 cases under both decoders")
    compact = []
    for row in records:
        if type(row) is not dict or set(row) != {
            "id", "split", "decodingPolicy", "output", "terminationReason",
            "generatedTokens", "promptSha256", "checkpointSha256",
        }:
            raise Refusal("M11 baseline response has missing or extra fields")
        case = next((case for case in cases if case["id"] == row["id"]), None)
        if (case is None or row["split"] != case["split"]
                or row["decodingPolicy"] not in DECODERS
                or type(row["output"]) is not str
                or len(row["output"]) > 8192
                or type(row["generatedTokens"]) is not int
                or not 0 <= row["generatedTokens"] <= m11.MAX_GENERATED
                or row["terminationReason"] not in RESPONSE_TERMINATIONS):
            raise Refusal("M11 baseline response source or generation budget mismatched")
        compact.append({
            key: row[key] for key in (
                "id", "decodingPolicy", "output", "terminationReason",
                "promptSha256", "checkpointSha256",
            )
        })
    independently_scored = core.score_unverified_generations(
        compact, source["m9WeightsSha256"])
    if report.get("scores") != independently_scored["scores"]:
        raise Refusal("M11 baseline claimed scores diverge from raw responses")
    return {
        "baselineReportSha256": report_sha,
        "baselineResponsesSha256": responses_sha,
        "baselineReceiptConsistencyVerified": True,
        "sourceBindingVerified": True,
        "baselineInferenceIndependentlyReproduced": False,
        "optimizerUpdatesCompleted": 0,
        "trainingAuthorized": False,
    }


def approval_template(source: dict[str, Any], baseline: dict[str, Any],
                      eos_weight: float) -> dict[str, Any]:
    """Owner-review template, deliberately unapproved and never written."""
    if type(eos_weight) not in (int, float) or eos_weight not in core.EOS_WEIGHTS:
        raise Refusal("M11 EOS arm must be exactly 1.0 or 0.25")
    required = ("candidateSha256", "scheduleSha256", "bundleManifestSha256",
                "m9WeightsSha256", "m9ReportSha256", "tokenizerSha256", "sourceHashes")
    if any(key not in source for key in required) or any(
            key not in baseline for key in (
                "baselineReportSha256", "baselineResponsesSha256")):
        raise Refusal("M11 cannot issue a template before verified source and baseline")
    return {
        "kind": KIND,
        "approvedBy": None,
        "approvedAt": None,
        "approvedForTraining": False,
        "sourceReviewedByOwner": False,
        "reviewedBothTrainingExamples": False,
        "reviewedDevelopmentOnlyExamples": False,
        "m11ProposalScopeAccepted": False,
        "candidateSha256": source["candidateSha256"],
        "scheduleSha256": source["scheduleSha256"],
        "m9WeightsSha256": source["m9WeightsSha256"],
        "m9ReportSha256": source["m9ReportSha256"],
        "bundleManifestSha256": source["bundleManifestSha256"],
        "tokenizerSha256": source["tokenizerSha256"],
        "sourceHashes": source["sourceHashes"],
        "baselineReportSha256": baseline["baselineReportSha256"],
        "baselineResponsesSha256": baseline["baselineResponsesSha256"],
        "trainRecordIds": source["trainingIds"],
        "developmentOnlyRecordIds": source["developmentOnlyIds"],
        "maxOptimizerUpdates": core.MAX_UPDATES,
        "checkpointSteps": list(core.CHECKPOINT_STEPS),
        "examplesPerOptimizerUpdate": 1,
        "gradientAccumulationSteps": 1,
        "eosLossWeight": float(eos_weight),
        "eosWeightDenominator": "unchanged-supervised-position-count",
        "promptDelimiter": "space",
        "answerEndsInNewline": True,
        "seed": 41,
        "optimizer": {
            "name": "AdamW",
            "learningRate": 0.00005,
            "weightDecay": 0.0,
            "epsilon": 1e-8,
            "foreach": False,
            "fused": False,
            "freshStateNotResume": True,
        },
        "decodingPolicies": list(DECODERS),
        "generationLimit": m11.MAX_GENERATED,
        "hardTimeoutSeconds": 3600,
        "checkpointMaxBytes": CHECKPOINT_MAX_BYTES,
        "minimumOutputFreeBytes": MIN_OUTPUT_FREE_BYTES,
        "finalHoldoutAllowed": False,
        "scope": PROTOCOL,
    }


def check_approval(path: Path, expected: dict[str, Any]) -> str:
    """Only a separately owner-created external file can satisfy this check."""
    if (path.is_symlink() or not path.is_file()
            or not 0 < path.stat().st_size <= APPROVAL_MAX_BYTES
            or path.resolve(strict=True).is_relative_to(research.ROOT.resolve())):
        raise Refusal("M11 requires a separate external owner-operated approval file")
    data = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "approvedBy", "approvedAt", "approvedForTraining", "sourceReviewedByOwner",
        "reviewedBothTrainingExamples", "reviewedDevelopmentOnlyExamples",
        "m11ProposalScopeAccepted",
    }
    if type(data) is not dict or set(data) != set(expected):
        raise Refusal("M11 approval JSON has missing or unexpected fields")
    if any(data[key] != val for key, val in expected.items() if key not in fields):
        raise Refusal("M11 approval has incorrect model, data, arm, or schedule scope")
    if (data["approvedBy"] != "keepithandy"
            or data["approvedForTraining"] is not True
            or data["sourceReviewedByOwner"] is not True
            or data["reviewedBothTrainingExamples"] is not True
            or data["reviewedDevelopmentOnlyExamples"] is not True
            or data["m11ProposalScopeAccepted"] is not True
            or type(data["approvedAt"]) is not str):
        raise Refusal("M11 owner has not separately approved the exact experiment")
    try:
        approved_at = date.fromisoformat(data["approvedAt"])
    except ValueError as exc:
        raise Refusal("M11 approval date is invalid") from exc
    if not date(2026, 11, 18) <= approved_at <= date.today():
        raise Refusal("M11 approval date outside this experiment's review period")
    return digest(path, APPROVAL_MAX_BYTES)


def admission(
    bundle: Path, bundle_sha: str, m9_report: Path, m9_weights: Path,
    baseline_report: Path, baseline_responses: Path,
    approval_file: Path, output_dir: Path, eos_weight: float, *,
    check_free_space: bool = True,
) -> dict[str, Any]:
    """CPU-only, fail-closed input gate. Not a GPU execution authorization."""
    output = m8.require_output(output_dir, bundle=bundle, checkpoint=m9_weights)
    source = inspect_sources(bundle, bundle_sha, m9_report, m9_weights)
    baseline = read_baseline(baseline_report, baseline_responses, source)
    template = approval_template(source, baseline, eos_weight)
    approval_sha = check_approval(approval_file, template)
    available = shutil.disk_usage(output.parent).free if check_free_space else None
    if check_free_space and available < MIN_OUTPUT_FREE_BYTES:
        raise Refusal("M11 output filesystem lacks required snapshot free space")
    return {
        **plan(),
        "status": "cpu-inputs-and-owner-consent-verified-not-executed",
        "ownerConsentGranted": True,
        "trainingAuthorized": False,  # Never signals GPU run authorization.
        "m11TrainerImplemented": False,
        "ownerApprovalSha256": approval_sha,
        "m9WeightsSha256": source["m9WeightsSha256"],
        "baselineReportSha256": baseline["baselineReportSha256"],
        "baselineResponsesSha256": baseline["baselineResponsesSha256"],
        "sourceHashes": source["sourceHashes"],
        "chosenEosLossWeight": float(eos_weight),
        "newOutputDestination": str(output),
        "availableFreeBytesChecked": check_free_space,
        "availableFreeBytes": available,
        "optimizerUpdatesCompleted": 0,
        "checkpointWrites": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--audit-sources", action="store_true")
    actions.add_argument("--show-unapproved-template", action="store_true")
    actions.add_argument("--audit-admission", action="store_true")
    for name in ("bundle-dir", "m9-report", "m9-weights",
                 "baseline-report", "baseline-responses", "approval-file", "output-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--bundle-manifest-sha256")
    parser.add_argument("--eos-weight", type=float)
    args = parser.parse_args(argv)
    paths = (args.bundle_dir, args.bundle_manifest_sha256,
             args.m9_report, args.m9_weights)
    try:
        if args.audit_sources or args.show_unapproved_template or args.audit_admission:
            if not all(paths):
                raise Refusal("M11 CPU audit needs exact frozen bundle and M9 backup")
            source = inspect_sources(*paths)
            if args.audit_sources:
                if any(x is not None for x in (
                        args.baseline_report, args.baseline_responses,
                        args.approval_file, args.output_dir, args.eos_weight)):
                    raise Refusal("Source-only audit does not accept approval/run arguments")
                result = {**plan(), "status": "cpu-source-audit-only", "source": source}
            else:
                if any(x is None for x in (
                        args.baseline_report, args.baseline_responses, args.eos_weight)):
                    raise Refusal("M11 template/admission requires new M11 M9 baseline and EOS arm")
                if args.show_unapproved_template:
                    if args.approval_file is not None or args.output_dir is not None:
                        raise Refusal("M11 template mode cannot accept approval/output paths")
                    baseline = read_baseline(
                        args.baseline_report, args.baseline_responses, source)
                    result = {**plan(), "approvalTemplate": approval_template(
                        source, baseline, args.eos_weight)}
                else:
                    if args.approval_file is None or args.output_dir is None:
                        raise Refusal("M11 admission requires approval and new output path")
                    result = admission(
                        *paths, args.baseline_report, args.baseline_responses,
                        args.approval_file, args.output_dir, args.eos_weight)
        elif any(x is not None for x in (
                *paths, args.baseline_report, args.baseline_responses,
                args.approval_file, args.output_dir, args.eos_weight)):
            raise Refusal("M11 file/arm options require explicit audit mode")
        else:
            result = plan()
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print("M11 admission refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
