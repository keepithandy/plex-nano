"""M10 copy microprobe: REVIEW-ONLY proposal, no training or model loading.

The assistant-proposed candidate rows are unapproved. This module can only
audit and display proposed sources, check frozen-tokenizer coverage, and
score externally produced read-only predictions. It has no optimizer,
checkpoint-writing, approval-granting, or GPU execution entry point.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

from . import plex600_m8_eval as m8
from . import plex600_research as research

PROTOCOL = "plex600-m10-copy-microprobe-proposal-v1"
CANDIDATE = research.ROOT / (
    "training/phase3/candidates/plex600-m10-copy-proposal-v1.jsonl"
)
ORIGIN = "assistant-proposed-candidate-not-owner-approved"
RIGHTS = "review-only-no-training"
SPLITS = ("train", "heldout")
TRAIN_PER_LANGUAGE = 4
HELDOUT_PER_LANGUAGE = 2
EVALUATION_UPDATES = (10, 25, 50)
MAX_UPDATES = 50
WARMUP_RECORD_ID = "m10-js-counter"
EXAMPLES_PER_UPDATE = 1
GRADIENT_ACCUMULATION_STEPS = 1
# Total optimizer steps, not epochs; no resetting between multi-example phases.
PHASES = (
    ("single-example-copy", 1, 10),
    ("full-set-copy", 11, 25),
    ("full-set-consolidation", 26, 50),
)
MAX_POSITIONS = 256
MAX_GENERATION = 96
TOKENIZER_VOCABULARY = 1191
SEED = 41
MAX_SOURCE_BYTES = 32 * 1024
NO_TRAINING = "quarantined-candidate-awaiting-source-and-scope-approval"


class Refusal(ValueError):
    pass


def plan() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "status": "proposal-only-no-training-execution",
        "modelFamily": "plex-600m-v1",
        "modelParameterCount": 600372480,
        "startingCheckpoint": "separately-sha-verified-owner-backed-up-M9-weights",
        "optimizer": "proposed-new-AdamW-not-optimizer-resume",
        "optimizerUpdatesMaximum": MAX_UPDATES,
        "evaluationAtOptimizerUpdates": list(EVALUATION_UPDATES),
        "warmupRecordId": WARMUP_RECORD_ID,
        "trainingPhases": [
            {"name": name, "firstUpdate": first, "lastUpdate": last}
            for name, first, last in PHASES
        ],
        "examplesPerOptimizerUpdate": EXAMPLES_PER_UPDATE,
        "gradientAccumulationSteps": GRADIENT_ACCUMULATION_STEPS,
        "examplesPerUpdateDefined": True,
        "maxInputPositionsPerExample": MAX_POSITIONS,
        "maxGeneratedTokensPerExample": MAX_GENERATION,
        "seed": SEED,
        "trainCandidateRecords": TRAIN_PER_LANGUAGE * 3,
        "unseenHeldoutCandidateRecords": HELDOUT_PER_LANGUAGE * 3,
        "task": "exact-reproduction-of-supplied-code-with-EOS",
        "supervision": "copied-code-and-EOS-only-masked-prompt",
        "earlyStop": "train-and-unseen-heldout-100pct-exact-plus-EOS-at-two-consecutive-evaluations",
        "decodingPolicy": "compare-unmasked-v1-and-fitted-vocab-v1-without-changing-checkpoint",
        "datasetOrigin": ORIGIN,
        "sourceApprovedForTraining": False,
        "ownerApprovalBundled": False,
        "trainingAuthorized": False,
        "runEntrypointImplemented": False,
        "optimizerUpdatesCompleted": 0,
        "modelLoaded": False,
        "checkpointWrites": False,
        "historicalCheckpointsModified": False,
        "finalHoldoutUsed": False,
        "m8DevelopmentUsedForTraining": False,
    }


def audit_m9_backup(report_path: Path, weights_path: Path) -> dict[str, Any]:
    """Stream-check an independent M9 backup; never load weights into PyTorch.

    The original completed M9 report has a known checkpointWrites metadata
    inconsistency; use actual checkpoint metadata and bytes as evidence.
    """
    from . import plex600_m7 as m7
    from . import plex600_r3_curriculum as r3

    original = research.ROOT / "training/artifacts/plex600-m9-r3-first-eight"
    if (
        report_path.is_symlink() or weights_path.is_symlink()
        or report_path.parent.resolve(strict=False)
        != weights_path.parent.resolve(strict=False)
        or weights_path.parent.resolve(strict=False)
        == original.resolve(strict=False)
        or report_path.name != "report.json"
        or weights_path.name != "weights.pt"
    ):
        raise Refusal("M10 admission requires separate M9 backup report and weights")
    report_sha = research.digest_file(report_path, 512 * 1024)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    if type(data) is not dict:
        raise Refusal("M9 backup report is not an object")
    checkpoint = data.get("checkpoint")
    approval = data.get("approval")
    steps = data.get("steps")
    if (
        data.get("protocol") != "plex600-m9-r3-eight-update-pilot-v1"
        or data.get("status") != "eight-updates-complete-unreviewed-improvement"
        or data.get("optimizerUpdatesCompleted") != 8
        or data.get("trainingAuthorized") is not True
        or data.get("modelLoaded") is not True
        or data.get("finalHoldoutUsed") is not False
        or data.get("m8DevelopmentUsedForTraining") is not False
        or data.get("codingImprovementProven") is not False
        or data.get("generatedCodeEvaluated") is not False
        or data.get("originalCheckpointSha256") !=
            "69a31b4e710b284581a0b74c9e2661cc27bf134d24b8b1fe7a906f6efc004224"
        or data.get("originalTokenizerSha256") != m7.TOKENIZER_SHA
        or data.get("bundleManifestSha256") != research.BUNDLE_SHA
        or data.get("r3CandidateSha256") != r3.source_sha256()
        or type(approval) is not dict
        or approval.get("approved") is not True
        or type(approval.get("approvalSha256")) is not str
        or len(approval["approvalSha256"]) != 64
        or type(steps) is not list
        or len(steps) != 8
        or any(
            type(step) is not dict or step.get("step") != i
            for i, step in enumerate(steps, 1)
        )
        or type(checkpoint) is not dict
        or checkpoint.get("file") != "weights.pt"
        or checkpoint.get("weightsOnly") is not True
        or checkpoint.get("optimizerStateSaved") is not False
        or type(checkpoint.get("bytes")) is not int
        or not 0 < checkpoint["bytes"] <= 3 * 1024**3
        or type(checkpoint.get("sha256")) is not str
        or len(checkpoint["sha256"]) != 64
    ):
        raise Refusal("M9 backup report provenance, approval or scope invalid")
    if weights_path.stat().st_size != checkpoint["bytes"]:
        raise Refusal("M9 backup checkpoint byte size differs from report")
    weight_sha = m7.hash_file(weights_path, maximum=3 * 1024**3)
    if weight_sha != checkpoint["sha256"]:
        raise Refusal("M9 backup checkpoint SHA-256 differs from report")
    return {
        "protocol": PROTOCOL,
        "status": "read-only-m9-backup-verified-not-training-authorized",
        "m9ReportSha256": report_sha,
        "m9WeightsSha256": weight_sha,
        "weightsBytes": checkpoint["bytes"],
        "optimizerUpdatesVerified": 8,
        "checkpointWritesReportMetadata": data.get("checkpointWrites"),
        "checkpointBytesAndHashVerified": True,
        "noCheckpointDeserialization": True,
        "trainingAuthorized": False,
        "optimizerUpdatesCompleted": 0,
        "checkpointWrites": False,
        "finalHoldoutUsed": False,
    }


def prompt_for(record: dict[str, Any]) -> str:
    """Source-conditioned identity copy. Target is the *entire* code snippet."""
    return (
        "Copy this code exactly. Return only the copied code.\n"
        "Code:\n" + record["snippet"] + "Copy:\n"
    )


def rows() -> list[dict[str, Any]]:
    items = research.read_jsonl(CANDIDATE, limit=MAX_SOURCE_BYTES)
    if len(items) != 18:
        raise Refusal("M10 proposal must contain 12 train and 6 unseen heldout candidates")
    required = {
        "schemaVersion", "id", "language", "split", "family",
        "snippet", "sourceOrigin", "trainingAuthorized", "rightsScope"
    }
    counts: Counter[tuple[str, str]] = Counter()
    ids: set[str] = set()
    families: set[str] = set()
    snippets: set[str] = set()
    for item in items:
        if (
            set(item) != required
            or type(item["schemaVersion"]) is not int
            or item["schemaVersion"] != 1
            or type(item["id"]) is not str
            or not item["id"].startswith("m10-")
            or type(item["family"]) is not str
            or not item["family"]
            or item["language"] not in research.LANGUAGES
            or item["split"] not in SPLITS
            or item["sourceOrigin"] != ORIGIN
            or item["rightsScope"] != RIGHTS
            or item["trainingAuthorized"] is not False
            or type(item["snippet"]) is not str
        ):
            raise Refusal("M10 candidate identity, provenance or permissions invalid")
        snippet = item["snippet"]
        if (
            not snippet.isascii()
            or not snippet.endswith("\n")
            or snippet.count("\n") != 1
            or not snippet.strip()
            or len(snippet.encode("utf-8")) > 96
            or "\r" in snippet
            or item["id"] in ids
            or item["family"] in families
            or snippet in snippets
        ):
            raise Refusal("M10 candidate snippet not short, unique or canonical")
        ids.add(item["id"])
        families.add(item["family"])
        snippets.add(snippet)
        counts[(item["language"], item["split"])] += 1
    for language in research.LANGUAGES:
        if (counts[(language, "train")] != TRAIN_PER_LANGUAGE
                or counts[(language, "heldout")] != HELDOUT_PER_LANGUAGE):
            raise Refusal("M10 train/heldout language balance invalid")

    # The independent M8 *development* benchmark is read-only here and never
    # enters M10 training. Its own final holdout is not accessed.
    development = research.benchmark()
    reserved = {
        row[key].strip() for row in development
        for key in ("inputFile", "expectedFile")
    }
    if any(item["snippet"].strip() in reserved for item in items):
        raise Refusal("M10 proposal duplicates an M8 development source or answer")
    return items


def update_schedule(items: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Pure 50-update *proposal*, never trains or mutates checkpoint.

    Updates 1..10 replay the selected owner-review candidate exactly once
    per optimizer update. Updates 11..50 round-robin all 12 train candidates
    without resetting at update 26. Heldout IDs are never in this schedule.
    """
    proposed = rows() if items is None else items
    train = [record["id"] for record in proposed if record["split"] == "train"]
    heldout = {record["id"] for record in proposed if record["split"] == "heldout"}
    if (len(train) != 12 or len(set(train)) != 12 or WARMUP_RECORD_ID not in train
            or len(heldout) != 6 or set(train) & heldout):
        raise Refusal("M10 schedule requires exact disjoint 12/6 proposal")
    schedule: list[dict[str, Any]] = []
    for step in range(1, MAX_UPDATES + 1):
        phase = next(name for name, first, last in PHASES
                     if first <= step <= last)
        record_id = (WARMUP_RECORD_ID if step <= 10
                     else train[(step - 11) % len(train)])
        schedule.append({
            "optimizerUpdate": step,
            "phase": phase,
            "recordIds": [record_id],
            "examplesInOptimizerUpdate": EXAMPLES_PER_UPDATE,
            "gradientAccumulationSteps": GRADIENT_ACCUMULATION_STEPS,
        })
    if (len(schedule) != MAX_UPDATES
            or any(item["recordIds"][0] in heldout for item in schedule)
            or len({item["recordIds"][0] for item in schedule[10:]}) != 12):
        raise Refusal("M10 schedule violated its split or update limits")
    return schedule


def schedule_audit() -> dict[str, Any]:
    schedule = update_schedule()
    exposures = dict(sorted(Counter(
        row["recordIds"][0] for row in schedule
    ).items()))
    return {
        "protocol": PROTOCOL,
        "status": "review-only-schedule-not-training",
        "optimizerUpdatesMaximum": MAX_UPDATES,
        "examplesPerOptimizerUpdate": EXAMPLES_PER_UPDATE,
        "gradientAccumulationSteps": GRADIENT_ACCUMULATION_STEPS,
        "warmupRecordId": WARMUP_RECORD_ID,
        "candidateSha256": research.digest_file(CANDIDATE, MAX_SOURCE_BYTES),
        "phaseUpdateCounts": {
            phase: sum(s["phase"] == phase for s in schedule)
            for phase, _, _ in PHASES
        },
        "trainRecordExposures": exposures,
        "heldoutTrainingExposures": 0,
        "schedule": schedule,
        "trainingAuthorized": False,
        "optimizerUpdatesCompleted": 0,
        "modelLoaded": False,
        "checkpointWrites": False,
        "finalHoldoutUsed": False,
    }


def audit(tokenizer: Any | None = None) -> dict[str, Any]:
    items = rows()
    result = {
        **plan(),
        "status": NO_TRAINING,
        "candidateSha256": research.digest_file(CANDIDATE, MAX_SOURCE_BYTES),
        "m8DevelopmentBenchmarkSha256": research.digest_file(research.BENCHMARK),
        "trainRecordIds": [r["id"] for r in items if r["split"] == "train"],
        "heldoutRecordIds": [r["id"] for r in items if r["split"] == "heldout"],
        "sourceAndTargetUnique": True,
        "trainHeldoutFamilyDisjoint": True,
        "m8DevelopmentExactFileOverlap": 0,
    }
    if tokenizer is None:
        return result

    # This branch loads only a verified tokenizer; no PyTorch or GPU.
    from . import plex600_r3_curriculum as r3
    if tokenizer.vocabulary_size != TOKENIZER_VOCABULARY:
        raise Refusal("M10 must use exactly the frozen 1191-token M7 fitted vocabulary")
    total_train_labels = 0
    max_input = 0
    max_answer = 0
    for record in items:
        prompt, snippet = prompt_for(record), record["snippet"]
        prefix = tokenizer.encode(prompt)
        answer = tokenizer.encode(snippet)
        if (
            not prefix or not answer
            or tokenizer.decode(prefix) != prompt
            or tokenizer.decode(answer) != snippet
            or len(answer) + 1 > MAX_GENERATION
        ):
            raise Refusal("M10 tokenizer failed exact copy round-trip or generation cap")
        packed = r3.masked_record(
            prefix, answer + [3],
            fitted_vocab_size=tokenizer.vocabulary_size,
        )
        if len(packed["inputIds"]) > MAX_POSITIONS:
            raise Refusal("M10 code-copy prompt exceeds proposed position cap")
        if record["split"] == "train":
            total_train_labels += packed["supervisedAnswerPositions"]
        max_input = max(max_input, len(packed["inputIds"]))
        max_answer = max(max_answer, len(answer) + 1)
    result.update(
        tokenizerSha256="9feef87bdf5fc28b3315df589e4e0b8ffcf5f40b18a4c357c91759182340a2e2",
        tokenizerVocabularySize=tokenizer.vocabulary_size,
        maxObservedInputPositions=max_input,
        maxObservedAnswerTokensIncludingEos=max_answer,
        totalTrainCopyAndEosSupervisedPositions=total_train_labels,
        tokenizerRoundTripVerified=True,
        generatedCodeEvaluated=False,
        optimizerUpdatesCompleted=0,
    )
    return result


def proposal_approval_template() -> dict[str, Any]:
    """Only prints an unapproved template. No approval file is ever created."""
    inspected = audit()
    return {
        "kind": "plex600-m10-copy-owner-review-template-v1",
        "approvedBy": None,
        "approvedAt": None,
        "approvedForTraining": False,
        "sourceReviewedByOwner": False,
        "candidateSha256": inspected["candidateSha256"],
        "m9CheckpointSha256": "OWNER_M9_CHECKPOINT_SHA256_NOT_YET_PINNED",
        "tokenizerBundleManifestSha256": research.BUNDLE_SHA,
        "trainRecordIds": inspected["trainRecordIds"],
        "heldoutRecordIds": inspected["heldoutRecordIds"],
        "maxOptimizerUpdates": MAX_UPDATES,
        "evaluationUpdates": list(EVALUATION_UPDATES),
        "warmupRecordId": WARMUP_RECORD_ID,
        "phaseUpdateCounts": [10, 15, 25],
        "examplesPerOptimizerUpdate": EXAMPLES_PER_UPDATE,
        "gradientAccumulationSteps": GRADIENT_ACCUMULATION_STEPS,
        "maxInputPositions": MAX_POSITIONS,
        "maxNewTokens": MAX_GENERATION,
        "seed": SEED,
        "finalHoldoutAllowed": False,
        "m8DevelopmentUsedForTraining": False,
        "scope": PROTOCOL,
    }


def score_predictions(
    expected: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    """CPU-only exact autonomous-copy scoring for later independent evaluation."""
    ids = {row["id"] for row in expected}
    if (
        len(expected) != 18 or len(ids) != 18 or len(responses) != 18
        or any(type(row) is not dict or set(row) != {"id", "output", "terminationReason"}
               or type(row["id"]) is not str or type(row["output"]) is not str
               or type(row["terminationReason"]) is not str
               for row in responses)
        or {row["id"] for row in responses} != ids
    ):
        raise Refusal("M10 score requires exactly 18 unique, complete generated responses")
    answer_by_id = {row["id"]: row for row in responses}
    if len(answer_by_id) != len(responses):
        raise Refusal("M10 scoring rejects duplicate response IDs")
    summary = {
        split: {"total": 0, "exact": 0, "eos": 0, "exactWithEos": 0}
        for split in SPLITS
    }
    for row in expected:
        prediction = answer_by_id[row["id"]]
        target = summary[row["split"]]
        target["total"] += 1
        exact = prediction["output"] == row["snippet"]
        eos = prediction["terminationReason"] == "eos"
        target["exact"] += int(exact)
        target["eos"] += int(eos)
        target["exactWithEos"] += int(exact and eos)
    all_exact = (
        summary["train"]["exactWithEos"] == 12
        and summary["heldout"]["exactWithEos"] == 6
    )
    return {
        "train": summary["train"],
        "heldout": summary["heldout"],
        "allExactAndEos": all_exact,
        "generalizationClaimEstablished": False,
        "finalHoldoutUsed": False,
    }


def milestone_score(
    updates: int, decoding_policy: str, responses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score supplied responses for a declared milestone, no model execution.

    This records only caller-provided outputs, not proof that inference ran.
    A future authorized worker must bind the checkpoint SHA to the inference.
    """
    if type(updates) is not int or updates not in EVALUATION_UPDATES:
        raise Refusal("M10 evaluation update must be exactly 10, 25 or 50")
    if decoding_policy not in m8.DECODE_POLICIES:
        raise Refusal("M10 evaluation must use a declared versioned decoding policy")
    return {
        **score_predictions(rows(), responses),
        "optimizerUpdatesCompleted": updates,
        "decodingPolicy": decoding_policy,
        "candidateSha256": research.digest_file(CANDIDATE, MAX_SOURCE_BYTES),
        "source": "caller-provided-responses-not-verified-generation",
    }


def eligible_to_stop_early(prior: dict[str, Any], current: dict[str, Any]) -> bool:
    """Never allow a single, repeated or cross-policy milestone to stop."""
    steps = (prior.get("optimizerUpdatesCompleted"),
             current.get("optimizerUpdatesCompleted"))
    sha = research.digest_file(CANDIDATE, MAX_SOURCE_BYTES)
    return (
        steps in ((10, 25), (25, 50))
        and prior.get("allExactAndEos") is True
        and current.get("allExactAndEos") is True
        and prior.get("finalHoldoutUsed") is False
        and current.get("finalHoldoutUsed") is False
        and prior.get("candidateSha256") == sha
        and current.get("candidateSha256") == sha
        and prior.get("decodingPolicy") in m8.DECODE_POLICIES
        and current.get("decodingPolicy") == prior.get("decodingPolicy")
        and prior.get("source") == "caller-provided-responses-not-verified-generation"
        and current.get("source") == "caller-provided-responses-not-verified-generation"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit-candidate", action="store_true")
    mode.add_argument("--show-schedule", action="store_true")
    mode.add_argument("--show-candidate", action="store_true")
    mode.add_argument("--approval-template", action="store_true")
    mode.add_argument("--audit-tokenizer", action="store_true")
    mode.add_argument("--audit-m9-backup", action="store_true")
    parser.add_argument("--m9-report", type=Path)
    parser.add_argument("--m9-weights", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--bundle-manifest-sha256")
    args = parser.parse_args(argv)
    try:
        if args.audit_m9_backup:
            if (args.m9_report is None or args.m9_weights is None
                    or args.bundle_dir is not None or args.bundle_manifest_sha256 is not None):
                raise Refusal("Backup audit needs only --m9-report and --m9-weights")
            result = audit_m9_backup(args.m9_report, args.m9_weights)
        elif args.m9_report is not None or args.m9_weights is not None:
            raise Refusal("M9 paths require --audit-m9-backup")
        elif args.audit_tokenizer:
            if args.bundle_dir is None or args.bundle_manifest_sha256 != research.BUNDLE_SHA:
                raise Refusal("Tokenizer audit requires frozen M7 bundle and manifest SHA")
            m8.verified_bundle(args.bundle_dir, args.bundle_manifest_sha256)
            from .tokenizer import PlexTokenizer
            result = audit(PlexTokenizer.load(args.bundle_dir))
        elif args.bundle_dir is not None or args.bundle_manifest_sha256 is not None:
            raise Refusal("Bundle arguments require --audit-tokenizer")
        elif args.approval_template:
            result = proposal_approval_template()
        elif args.show_schedule:
            result = schedule_audit()
        elif args.show_candidate:
            result = {"plan": audit(), "rows": rows()}
        elif args.audit_candidate:
            result = audit()
        else:
            result = plan()
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print("Plex M10 review-only proposal refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

