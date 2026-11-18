"""M10 owner-review admission gate, CPU ONLY. This module NEVER trains.

M10 candidates in PR #205 are assistant-proposed and explicitly unapproved.
A successful read-only baseline is not permission to train. This module
prepares an unapproved exact-scope template and validates separately provided
owner consent *without* importing PyTorch, loading model weights, creating
optimizer state, invoking CUDA, or creating an experiment directory.

A future training worker MUST independently revalidate everything here and
must implement its own hard time, memory, snapshot, and inference gates.
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

from . import plex600_m10_copy_proposal as m10
from . import plex600_m8_eval as m8
from . import plex600_research as research

PROTOCOL = "plex600-m10-exact-scope-cpu-admission-v1"
KIND = "plex600-m10-exact-owner-training-approval-v1"
BASELINE_PROTOCOL = "plex600-m10-m9-read-only-copy-baseline-v1"
BASELINE_MAX_BYTES = 256 * 1024
RESPONSES_MAX_BYTES = 512 * 1024
APPROVAL_MAX_BYTES = 16 * 1024
CHECKPOINT_MAX_BYTES = 3 * 1024**3
SNAPSHOTS = 3
# Require room for 3 permanent snapshots, one temporary in-flight save,
# plus 1 GiB safety margin before any authorized worker creates its output.
MIN_OUTPUT_FREE_BYTES = (SNAPSHOTS + 1) * CHECKPOINT_MAX_BYTES + 1024**3
DECODERS = ("unmasked-v1", "fitted-vocab-v1")
TOKENIZER_SHA256 = "9feef87bdf5fc28b3315df589e4e0b8ffcf5f40b18a4c357c91759182340a2e2"


class Refusal(ValueError):
    pass


def plan() -> dict[str, Any]:
    """Safe default, no reads, no model, no files written."""
    return {
        "protocol": PROTOCOL,
        "status": "plan-only-no-training",
        "candidateOrigin": m10.ORIGIN,
        "candidateTrainingApproved": False,
        "maxOptimizerUpdates": m10.MAX_UPDATES,
        "warmupRecordId": m10.WARMUP_RECORD_ID,
        "examplesPerOptimizerUpdate": 1,
        "phaseUpdates": [10, 15, 25],
        "requiredEvaluationUpdates": [0, 10, 25, 50],
        "minimumOutputDiskFreeBytes": MIN_OUTPUT_FREE_BYTES,
        "requireCompleted18CaseM9Baseline": True,
        "requireExternalOwnerApproval": True,
        "runEntrypointImplemented": False,
        "optimizerUpdatesCompleted": 0,
        "trainingAuthorized": False,
        "modelLoaded": False,
        "checkpointWrites": False,
        "artifactWrites": False,
        "finalHoldoutUsed": False,
    }


def _digest(path: Path, cap: int) -> str:
    return research.digest_file(path, limit=cap)


def read_baseline_receipt(
    report_path: Path, responses_path: Path, *,
    m9_weights_sha: str, m9_report_sha: str, candidate_sha: str,
) -> dict[str, Any]:
    """Read-only verify the *full* M9 18-case / 2-policy baseline and JSONL.

    The caller-supplied report alone is not proof of GPU inference. This
    validates consistency, hashes, identities and scored output counts.
    """
    if (report_path.is_symlink() or responses_path.is_symlink()
            or not report_path.is_file() or not responses_path.is_file()
            or report_path.parent.resolve(strict=True)
               != responses_path.parent.resolve(strict=True)
            or report_path.name != "report.json"
            or responses_path.name != "responses.jsonl"):
        raise Refusal("M10 requires independent baseline report.json and responses.jsonl")
    raw_report_sha = _digest(report_path, BASELINE_MAX_BYTES)
    raw_response_sha = _digest(responses_path, RESPONSES_MAX_BYTES)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if type(report) is not dict:
        raise Refusal("Baseline report must be a JSON object")
    required_bools = {
        "readOnlyGenerationExecuted": True,
        "generatedCodeEvaluated": True,
        "trainingAuthorized": False,
        "checkpointWrites": False,
        "finalHoldoutUsed": False,
    }
    if (report.get("protocol") != BASELINE_PROTOCOL
            or report.get("status") != "read-only-m9-generation-complete"
            or any(report.get(key) is not value for key, value in required_bools.items())
            or type(report.get("optimizerUpdates")) is not int
            or report["optimizerUpdates"] != 0
            or report.get("m9WeightsSha256") != m9_weights_sha
            or report.get("m9ReportSha256") != m9_report_sha
            or report.get("candidateSha256") != candidate_sha
            or report.get("bundleManifestSha256") != research.BUNDLE_SHA
            or report.get("tokenizerSha256") != TOKENIZER_SHA256
            or report.get("responsesSha256") != raw_response_sha
            or report.get("selectedCases") != 18
            or report.get("decodingPolicies") != list(DECODERS)
            or type(report.get("selectedCaseIds")) is not list
            or type(report.get("results")) is not dict
            or set(report["results"]) != set(DECODERS)):
        raise Refusal("M9 full baseline identity or source policy mismatch")
    candidates = m10.rows()
    by_id = {row["id"]: row for row in candidates}
    if (len(by_id) != 18 or len(report["selectedCaseIds"]) != 18
            or len(set(report["selectedCaseIds"])) != 18
            or set(report["selectedCaseIds"]) != set(by_id)):
        raise Refusal("M9 baseline did not cover all 18 exact review candidates")
    responses = research.read_jsonl(responses_path, limit=RESPONSES_MAX_BYTES)
    if len(responses) != 36:
        raise Refusal("M9 baseline must contain exactly 18 cases under both decoders")
    for decoder in DECODERS:
        decoder_rows = []
        for item in responses:
            if type(item) is not dict or set(item) != {
                "id", "output", "terminationReason", "split", "language",
                "decodingPolicy", "generatedTokens", "promptSha256", "m9WeightsSha256",
            }:
                raise Refusal("M9 baseline response contains unexpected fields")
            if item.get("decodingPolicy") != decoder:
                continue
            candidate = by_id.get(item["id"])
            if (candidate is None
                    or item["split"] != candidate["split"]
                    or item["language"] != candidate["language"]
                    or item["m9WeightsSha256"] != m9_weights_sha
                    or item["promptSha256"] != hashlib.sha256(
                        m10.prompt_for(candidate).encode("utf-8")).hexdigest()
                    or type(item["generatedTokens"]) is not int
                    or not 0 <= item["generatedTokens"] <= m10.MAX_GENERATION
                    or type(item["output"]) is not str
                    or len(item["output"]) > m10.MAX_RESPONSE_CHARACTERS
                    or item["terminationReason"] not in {
                        "eos", "max-new-tokens", "invalid-control-token",
                        "outside-fitted-tokenizer",
                    }):
                raise Refusal("M9 baseline response checkpoint/prompt/token mismatch")
            decoder_rows.append({
                "id": item["id"], "output": item["output"],
                "terminationReason": item["terminationReason"],
            })
        scores = m10.score_predictions(candidates, decoder_rows)
        reference = report["results"].get(decoder)
        if type(reference) is not dict or type(reference.get("bySplit")) is not dict:
            raise Refusal("M9 baseline missing independently recomputable scores")
        for split in m10.SPLITS:
            actual = scores[split]
            stored = reference["bySplit"].get(split)
            if type(stored) is not dict:
                raise Refusal("M9 baseline split summary is invalid")
            for name in ("total", "exact", "eos", "exactWithEos"):
                if stored.get(name) != actual[name]:
                    raise Refusal("M9 baseline score differs from raw responses")
            similarity = stored.get("meanCharacterSimilarity")
            if (type(similarity) not in (int, float)
                    or not math.isfinite(similarity)
                    or abs(similarity - actual["meanCharacterSimilarity"]) > 0.000002):
                raise Refusal("M9 baseline character score differs from raw responses")
        if reference.get("complete18CaseComparison") is not True:
            raise Refusal("M9 baseline comparison is incomplete")
    return {
        "status": "complete-m9-read-only-baseline-consistent",
        "m9WeightsSha256": m9_weights_sha,
        "m9ReportSha256": m9_report_sha,
        "baselineReportSha256": raw_report_sha,
        "baselineResponsesSha256": raw_response_sha,
        "complete18CasesBothDecoders": True,
        "baselineNoTraining": True,
        "modelLoaded": False,
        "trainingAuthorized": False,
        "checkpointWrites": False,
    }


def approval_template(
    source: dict[str, Any], baseline: dict[str, Any],
) -> dict[str, Any]:
    """Return only a proposal with false/no-approval fields; never write it."""
    required = {"m9WeightsSha256", "m9ReportSha256"}
    if not required.issubset(source) or not {
        "baselineReportSha256", "baselineResponsesSha256"
    }.issubset(baseline):
        raise Refusal("Missing SHA-verified source and completed baseline")
    inspected = m10.audit()
    return {
        "kind": KIND,
        "approvedBy": None,
        "approvedAt": None,
        "approvedForTraining": False,
        "sourceReviewedByOwner": False,
        "assistantProposedSourcesAcceptedForTraining": False,
        "candidateSha256": inspected["candidateSha256"],
        "m9WeightsSha256": source["m9WeightsSha256"],
        "m9ReportSha256": source["m9ReportSha256"],
        "baselineReportSha256": baseline["baselineReportSha256"],
        "baselineResponsesSha256": baseline["baselineResponsesSha256"],
        "tokenizerBundleManifestSha256": research.BUNDLE_SHA,
        "trainRecordIds": inspected["trainRecordIds"],
        "heldoutRecordIds": inspected["heldoutRecordIds"],
        "maxOptimizerUpdates": m10.MAX_UPDATES,
        "evaluationUpdates": list(m10.EVALUATION_UPDATES),
        "examplesPerOptimizerUpdate": m10.EXAMPLES_PER_UPDATE,
        "gradientAccumulationSteps": m10.GRADIENT_ACCUMULATION_STEPS,
        "phaseUpdateCounts": [10, 15, 25],
        "warmupRecordId": m10.WARMUP_RECORD_ID,
        "maxInputPositions": m10.MAX_POSITIONS,
        "maxNewTokens": m10.MAX_GENERATION,
        "seed": m10.SEED,
        "optimizer": {
            "name": "AdamW",
            "learningRate": 0.00005,
            "weightDecay": 0.0,
            "epsilon": 0.00000001,
            "foreach": False,
            "fused": False,
            "freshStateNotResume": True,
        },
        "decodingPolicies": list(DECODERS),
        "timeBudgetSeconds": 3600,
        "snapshotCount": SNAPSHOTS,
        "checkpointMaxBytes": CHECKPOINT_MAX_BYTES,
        "finalHoldoutAllowed": False,
        "m8DevelopmentUsedForTraining": False,
        "scope": m10.PROTOCOL,
    }


def inspect_sources(
    bundle: Path, bundle_sha: str, m9_report: Path, m9_weights: Path,
    baseline_report: Path, baseline_responses: Path,
) -> dict[str, Any]:
    """Pure admission inputs, no external owner permission and no GPU."""
    if bundle_sha != research.BUNDLE_SHA:
        raise Refusal("M10 frozen M7 bundle SHA mismatch")
    backup = m10.audit_m9_backup(m9_report, m9_weights)
    m8.verified_bundle(bundle, bundle_sha)
    from .tokenizer import PlexTokenizer
    tokenizer = PlexTokenizer.load(bundle)
    token_audit = m10.audit(tokenizer)
    baseline = read_baseline_receipt(
        baseline_report, baseline_responses,
        m9_weights_sha=backup["m9WeightsSha256"],
        m9_report_sha=backup["m9ReportSha256"],
        candidate_sha=token_audit["candidateSha256"],
    )
    if token_audit["tokenizerSha256"] != TOKENIZER_SHA256:
        raise Refusal("M10 frozen tokenizer SHA drifted")
    return {
        "backup": backup,
        "baseline": baseline,
        "tokenAudit": token_audit,
        "unapprovedTemplate": approval_template(backup, baseline),
    }


def check_approval(path: Path, proposed: dict[str, Any]) -> dict[str, Any]:
    """Require exact, external owner signoff; prior M9 consent is never reused."""
    if (path.is_symlink() or not path.is_file()
            or not 0 < path.stat().st_size <= APPROVAL_MAX_BYTES):
        raise Refusal("M10 requires a separate small, regular owner approval file")
    data = json.loads(path.read_text(encoding="utf-8"))
    if type(data) is not dict or set(data) != set(proposed):
        raise Refusal("M10 external approval schema changed or has extra/missing fields")
    fixed = {
        key: value for key, value in proposed.items()
        if key not in {
            "approvedBy", "approvedAt", "approvedForTraining",
            "sourceReviewedByOwner",
            "assistantProposedSourcesAcceptedForTraining",
        }
    }
    if any(data.get(key) != value for key, value in fixed.items()):
        raise Refusal("M10 owner approval does not bind exact data, model or limits")
    if (data.get("approvedBy") != "keepithandy"
            or data.get("approvedForTraining") is not True
            or data.get("sourceReviewedByOwner") is not True
            or data.get("assistantProposedSourcesAcceptedForTraining") is not True
            or type(data.get("approvedAt")) is not str):
        raise Refusal("M10 candidate is not separately authorized by owner")
    try:
        approved_at = date.fromisoformat(data["approvedAt"])
    except (TypeError, ValueError) as exc:
        raise Refusal("M10 approval date is invalid") from exc
    if approved_at < date(2026, 11, 17) or approved_at > date.today():
        raise Refusal("M10 owner approval predates this experiment or is future-dated")
    return {
        "externalApprovalValidated": True,
        "approvalFileSha256": _digest(path, APPROVAL_MAX_BYTES),
        "trainingExecuted": False,
    }


def admission(
    bundle: Path, bundle_sha: str, m9_report: Path, m9_weights: Path,
    baseline_report: Path, baseline_responses: Path, approval: Path,
    output: Path, *, check_free_space: bool = True,
) -> dict[str, Any]:
    """CPU-only admission; never grants approval or starts training."""
    if bundle_sha != research.BUNDLE_SHA:
        raise Refusal("M10 frozen M7 bundle SHA mismatch")
    # Output must not overwrite any existing result. Reject it BEFORE
    # hashing multi-GB weights or reading any owner-consent file.
    output_path = m8.require_output(output, bundle=bundle, checkpoint=m9_weights)
    inspected = inspect_sources(
        bundle, bundle_sha, m9_report, m9_weights,
        baseline_report, baseline_responses,
    )
    backup = inspected["backup"]
    token_audit = inspected["tokenAudit"]
    baseline = inspected["baseline"]
    consent = check_approval(approval, inspected["unapprovedTemplate"])
    free_bytes = shutil.disk_usage(output_path.parent).free if check_free_space else None
    if check_free_space and free_bytes < MIN_OUTPUT_FREE_BYTES:
        raise Refusal("M10 output filesystem lacks room for bounded checkpoint snapshots")
    schedule = m10.update_schedule()
    if (len(schedule) != m10.MAX_UPDATES
            or any(s["examplesInOptimizerUpdate"] != 1 for s in schedule)
            or {s["recordIds"][0] for s in schedule[:10]}
               != {m10.WARMUP_RECORD_ID}
            or len({s["recordIds"][0] for s in schedule[10:]}) != 12):
        raise Refusal("M10 exact 10+15+25 schedule was modified")
    return {
        **plan(),
        "status": "cpu-admission-validated-not-executed",
        "candidateSha256": token_audit["candidateSha256"],
        "m9WeightsSha256": backup["m9WeightsSha256"],
        "m9ReportSha256": backup["m9ReportSha256"],
        "baselineReportSha256": baseline["baselineReportSha256"],
        "baselineResponsesSha256": baseline["baselineResponsesSha256"],
        "tokenizerSha256": token_audit["tokenizerSha256"],
        "sourceBindingVerified": True,
        "trainingAuthorized": consent["externalApprovalValidated"],
        "externalApprovalSha256": consent["approvalFileSha256"],
        "outputReservationCheck": {
            "destination": str(output_path),
            "mustBeNew": True,
            "freeBytesChecked": check_free_space,
            "availableBytes": free_bytes,
            "requiredFreeBytes": MIN_OUTPUT_FREE_BYTES,
        },
        "runEntrypointImplemented": False,
        "optimizerUpdatesCompleted": 0,
        "checkpointWrites": False,
        "artifactWrites": False,
        "modelLoaded": False,
        "finalHoldoutUsed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-admission", action="store_true")
    modes.add_argument("--show-approval-template", action="store_true")
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--bundle-manifest-sha256")
    parser.add_argument("--m9-report", type=Path)
    parser.add_argument("--m9-weights", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--baseline-responses", type=Path)
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.audit_admission or args.show_approval_template:
            required = [
                args.bundle_dir, args.bundle_manifest_sha256,
                args.m9_report, args.m9_weights,
                args.baseline_report, args.baseline_responses,
            ]
            if not all(required):
                raise Refusal("M10 admission requires frozen tokenizer, backup, baseline")
            if args.show_approval_template:
                if args.approval_file is not None or args.output_dir is not None:
                    raise Refusal("Unapproved template cannot accept approval or output paths")
                inspected = inspect_sources(
                    args.bundle_dir, args.bundle_manifest_sha256,
                    args.m9_report, args.m9_weights,
                    args.baseline_report, args.baseline_responses,
                )
                result = {
                    **plan(),
                    "status": "exact-scope-template-unapproved",
                    "approvalTemplate": inspected["unapprovedTemplate"],
                }
            else:
                if args.approval_file is None or args.output_dir is None:
                    raise Refusal("M10 admission additionally requires owner approval and new output")
                result = admission(
                    args.bundle_dir, args.bundle_manifest_sha256,
                    args.m9_report, args.m9_weights,
                    args.baseline_report, args.baseline_responses,
                    args.approval_file, args.output_dir,
                )
        elif any(value is not None for value in (
            args.bundle_dir, args.bundle_manifest_sha256, args.m9_report,
            args.m9_weights, args.baseline_report, args.baseline_responses,
            args.approval_file, args.output_dir,
        )):
            raise Refusal("M10 file paths require explicit CPU-only admission/template flag")
        else:
            result = plan()
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print("M10 CPU-only admission refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
