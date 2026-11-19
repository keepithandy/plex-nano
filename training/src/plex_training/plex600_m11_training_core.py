"""M11 training core: deterministic candidate schedule, loss, and read-only probes.

This module is NOT a training runner or an approval path. A future bounded
worker must independently verify source scope, owner consent, tokenizer, M9
weights, output destination and receipts before importing CUDA or optimizing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import plex600_m11_source_conditioning as m11
from . import plex600_m10_admission as m10_gate
from . import plex600_r3_curriculum as r3

PROTOCOL = "plex600-m11-training-core-review-only-v1"
MAX_UPDATES = 200
CHECKPOINT_STEPS = (25, 50, 100, 200)
EOS_WEIGHTS = (1.0, 0.25)
EOS_ID = 3
IGNORE = -100
DECODERS = m10_gate.DECODERS


class Refusal(ValueError):
    pass


def candidate_sha256() -> str:
    """Digest the proposed *four* cases; no file or mutable source lookup."""
    rows = m11.cases()
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exact_schedule() -> list[dict[str, Any]]:
    """Equal, alternating 0/1 exposures. The probe-only cases NEVER train."""
    train = [case for case in m11.cases() if case["split"] == "train"]
    heldout = {case["id"] for case in m11.cases() if case["split"] != "train"}
    if (len(train) != 2 or len(heldout) != 2
            or [case["snippet"] for case in train]
            != ["let score = 0;\n", "let score = 1;\n"]
            or len({case["id"] for case in m11.cases()}) != 4):
        raise Refusal("Exact M11 paired-source candidate scope changed")
    schedule = [{
        "optimizerUpdate": step,
        "recordId": train[(step - 1) % 2]["id"],
        "examplesInOptimizerUpdate": 1,
        "gradientAccumulationSteps": 1,
    } for step in range(1, MAX_UPDATES + 1)]
    counts = Counter(step["recordId"] for step in schedule)
    if (len(schedule) != MAX_UPDATES or set(counts) != {case["id"] for case in train}
            or any(count != 100 for count in counts.values())
            or heldout & set(counts)):
        raise Refusal("M11 schedule violates contrast balance or split isolation")
    return schedule


def schedule_sha256() -> str:
    raw = json.dumps(exact_schedule(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def plan() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "status": "review-only-no-gpu-training-entrypoint",
        "source": "separately-verified-independent-M9-backup-proposed",
        "candidateSha256": candidate_sha256(),
        "scheduleSha256": schedule_sha256(),
        "trainingCandidates": [c["id"] for c in m11.cases() if c["split"] == "train"],
        "developmentOnlyCandidates": [c["id"] for c in m11.cases()
                                      if c["split"] == "probe-only"],
        "trainingExposuresPerCandidate": 100,
        "developmentTrainingExposures": 0,
        "maxOptimizerUpdatesProposed": MAX_UPDATES,
        "checkpointStepsProposed": list(CHECKPOINT_STEPS),
        "candidateTerminalEosWeights": list(EOS_WEIGHTS),
        "lossDenominator": "all-nonmasked-target-positions-unchanged-between-arms",
        "requiresNewExactScopeOwnerApproval": True,
        "m10ApprovalCannotAuthorizeM11": True,
        "trainingAuthorized": False,
        "optimizerUpdatesCompleted": 0,
        "executionEntrypointImplemented": False,
        "modelLoaded": False,
        "checkpointWrites": False,
        "finalHoldoutUsed": False,
    }


def packed_training_samples(tokenizer: Any) -> dict[str, dict[str, Any]]:
    """Supervise answer + final EOS only. Inspect both train and probe tokenization."""
    m11.audit_tokenizer(tokenizer)
    packed = {}
    for case in m11.cases():
        if case["split"] != "train":
            continue
        prompt = tokenizer.encode(m11.prompt_for(case))
        answer = tokenizer.encode(case["snippet"])
        if not prompt or not answer or answer[-1] != tokenizer.encode("\n")[0]:
            raise Refusal("M11 answer newline must be preserved")
        target = answer + [EOS_ID]
        record = r3.masked_record(prompt, target, fitted_vocab_size=tokenizer.vocabulary_size)
        if (record["inputIds"] != prompt + answer
                or record["targetIds"] != [IGNORE] * (len(prompt) - 1) + target
                or record["supervisedAnswerPositions"] != len(target)
                or len(record["inputIds"]) > m11.MAX_POSITIONS
                or len(target) > m11.MAX_GENERATED):
            raise Refusal("M11 answer/EOS-only mask or budget changed")
        packed[case["id"]] = record
    if set(packed) != {case["id"] for case in m11.cases() if case["split"] == "train"}:
        raise Refusal("M11 packed samples include a nontraining record")
    return packed


def weighted_copy_loss(torch: Any, logits: Any, target_ids: Any,
                       eos_weight: float) -> Any:
    """Keep the SAME label-count denominator in both EOS-weight arms.

    Only the per-example terminal EOS contribution is reweighted. Content
    gradients are identical for identical logits in the two arms.
    """
    if (type(eos_weight) not in (float, int)
            or not math.isfinite(eos_weight) or eos_weight not in EOS_WEIGHTS):
        raise Refusal("M11 EOS weight must be a predeclared 1.0 or 0.25")
    if (logits.ndim != 3 or target_ids.ndim != 2
            or logits.shape[:2] != target_ids.shape
            or logits.shape[2] <= EOS_ID):
        raise Refusal("M11 logits/targets have invalid alignment")
    valid = target_ids.ne(IGNORE)
    is_eos = target_ids.eq(EOS_ID)
    if not bool(valid.any().item()):
        raise Refusal("M11 has no supervised positions")
    for row_mask, eos_mask in zip(valid, is_eos):
        positions = row_mask.nonzero().flatten()
        if (len(positions) < 2 or int(eos_mask.sum().item()) != 1
                or int(positions[-1].item()) != int(eos_mask.nonzero()[0].item())):
            raise Refusal("M11 requires content supervision and exactly one terminal EOS")
    nonmasked = target_ids[valid]
    if not bool(((nonmasked >= 0) & (nonmasked < logits.shape[2])).all().item()):
        raise Refusal("M11 target ID exceeds fitted model logits")
    ce = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1), ignore_index=IGNORE, reduction="none",
    ).reshape(target_ids.shape)
    weights = torch.where(is_eos, eos_weight, 1.0)
    # Deliberately NOT weights.sum(): no hidden content gradient rescaling.
    loss = (ce * valid * weights).sum() / valid.sum()
    if not bool(torch.isfinite(loss).item()):
        raise Refusal("M11 weighted loss is nonfinite")
    return loss


def source_pair_measurement(torch: Any, model: Any, tokenizer: Any,
                            split: str) -> dict[str, Any]:
    """Gold-prefix contrast from actual logits; NOT autonomous exact copying."""
    if split not in ("train", "probe-only"):
        raise Refusal("Unrecognized M11 source-pair split")
    m11.audit_tokenizer(tokenizer)
    group = [case for case in m11.cases() if case["split"] == split]
    answers = [tokenizer.encode(case["snippet"]) for case in group]
    differing_index = m11.one_change(*answers)
    pair = [answer[differing_index] for answer in answers]
    margins = []
    greedies = []
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for case, answer in zip(group, answers):
            context = tokenizer.encode(m11.prompt_for(case)) + answer[:differing_index]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits, _ = model(x)
            last = logits[0, -1].float()
            if not bool(torch.isfinite(last).all().item()):
                raise Refusal("M11 source-pair logits are nonfinite")
            margins.append(float((last[pair[1]] - last[pair[0]]).item()))
            greedies.append(int(last.argmax().item()))
    return {
        "split": split,
        "caseIds": [case["id"] for case in group],
        "contrastTokenIds": pair,
        "oneMinusZeroLogitMargins": margins,
        "fullVocabularyGreedyIds": greedies,
        "tracksSourceInBothDirections": m11.paired_preference(
            *margins, *greedies, *pair),
        "usesGoldAnswerPrefix": True,
        "autonomousExactCopyingEvaluated": False,
    }


def score_unverified_generations(rows: list[dict[str, Any]],
                                 checkpoint_sha: str) -> dict[str, Any]:
    """Validate purported receipts but NEVER call them verified model results."""
    if (type(checkpoint_sha) is not str or len(checkpoint_sha) != 64
            or any(c not in "0123456789abcdef" for c in checkpoint_sha)):
        raise Refusal("Malformed checkpoint digest")
    cases = {case["id"]: case for case in m11.cases()}
    expected = {(policy, case_id) for policy in DECODERS for case_id in cases}
    if len(rows) != len(expected):
        raise Refusal("M11 requires every case under both decoder policies")
    hits: dict[tuple[str, str], bool] = {}
    for row in rows:
        if (type(row) is not dict
                or set(row) != {"id", "decodingPolicy", "output",
                                "terminationReason", "promptSha256", "checkpointSha256"}
                or (row["decodingPolicy"], row["id"]) not in expected
                or (row["decodingPolicy"], row["id"]) in hits
                or type(row["output"]) is not str or type(row["terminationReason"]) is not str):
            raise Refusal("M11 generation receipt has invalid identity or duplicate")
        case = cases[row["id"]]
        prompt_sha = hashlib.sha256(m11.prompt_for(case).encode("utf-8")).hexdigest()
        if row["checkpointSha256"] != checkpoint_sha or row["promptSha256"] != prompt_sha:
            raise Refusal("M11 generation receipt has invalid source or checkpoint SHA")
        hits[row["decodingPolicy"], row["id"]] = (
            row["output"] == case["snippet"] and row["terminationReason"] == "eos")
    scores = {}
    for policy in DECODERS:
        scores[policy] = {
            "trainingPairExactAndEos": all(
                hits[policy, case["id"]] for case in cases.values() if case["split"] == "train"),
            "developmentPairExactAndEos": all(
                hits[policy, case["id"]] for case in cases.values()
                if case["split"] == "probe-only"),
        }
    return {
        "protocol": PROTOCOL,
        "claimedCheckpointSha256": checkpoint_sha,
        "decodingPolicies": list(DECODERS),
        "scores": scores,
        "receiptOrigin": "caller-supplied-unverified",
        "verifiedModelGeneration": False,
        "sourcePairLogitGateNotMeasured": True,
        "trainingAuthorized": False,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-schedule", action="store_true")
    parser.add_argument("--audit-frozen-tokenizer", action="store_true")
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args(argv)
    if args.bundle is not None and not args.audit_frozen_tokenizer:
        parser.error("--bundle is only valid with --audit-frozen-tokenizer")
    if args.audit_frozen_tokenizer and args.bundle is None:
        parser.error("--audit-frozen-tokenizer requires --bundle")
    result = plan()
    if args.show_schedule:
        result["schedule"] = exact_schedule()
    if args.audit_frozen_tokenizer:
        from . import plex600_m8_eval as m8
        from . import plex600_research as research
        from .tokenizer import PlexTokenizer
        m8.verified_bundle(args.bundle, research.BUNDLE_SHA)
        result["tokenizerAudit"] = m11.audit_tokenizer(PlexTokenizer.load(args.bundle))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
