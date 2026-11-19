"""Review-only M11 source-conditioning proposal. No GPU, optimizer or training entry point.

Pairs are assistant-authored candidates, not owner-approved training data.
Scoring of caller-supplied outputs is *not* evidence of model generation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import plex600_m10_copy_proposal as m10

PROTOCOL = "plex600-m11-source-conditioning-proposal-v1"
PAIRS = (
    ("train", "score", ("0", "1")),
    ("probe-only", "value", ("0", "1")),
)
EVALUATION_STEPS = (0, 25, 50, 100, 200)
MAX_POSITIONS = 256
MAX_GENERATED = 96
EOS_ID = 3


class Refusal(ValueError):
    pass


def cases() -> list[dict]:
    """The probe-only pair is visible development data, never a final holdout."""
    return [
        {
            "id": f"m11-{split}-{name}-{digit}",
            "split": split,
            "snippet": f"let {name} = {digit};\n",
            "sourceOrigin": "assistant-proposed-not-owner-approved",
            "trainingAuthorized": False,
        }
        for split, name, digits in PAIRS
        for digit in digits
    ]


def prompt_for(case: dict) -> str:
    """Keep M10 instruction and replace only its final prompt newline."""
    original = m10.prompt_for(case)
    if not original.endswith("Copy:\n"):
        raise Refusal("Unexpected M10 prompt contract")
    return original[:-1] + " "


def plan() -> dict:
    return {
        "protocol": PROTOCOL,
        "status": "review-only-unapproved-no-training",
        "parentEvidence": "M10 source-pair and delimiter diagnostics, PR #209",
        "startingCheckpointProposed": "separately-verified-independent-M9-backup",
        "promptBoundary": "space; reference answer still ends with newline",
        "proposedOptimizerUpdateMilestones": list(EVALUATION_STEPS),
        "optimizerUpdatesMaximumProposed": 200,
        "terminalEosWeighting": "compare normal versus reduced; not selected",
        "scoringRequirement": "both source variants exact with EOS, plus changed-token preference",
        "trainingCandidateIds": [c["id"] for c in cases() if c["split"] == "train"],
        "probeOnlyIds": [c["id"] for c in cases() if c["split"] == "probe-only"],
        "probeOnlyTrainingExposures": 0,
        "sourceApprovedForTraining": False,
        "newOwnerAuthorizationRequired": True,
        "executionEntrypointImplemented": False,
        "optimizerUpdatesCompleted": 0,
        "modelLoaded": False,
        "checkpointWrites": False,
        "finalHoldoutUsed": False,
    }


def one_change(a: list[int], b: list[int]) -> int:
    positions = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(a) != len(b) or len(positions) != 1:
        raise Refusal("A source pair must differ at exactly one token")
    return positions[0]


def audit_tokenizer(tokenizer) -> dict:
    """Token-only contract check. Caller must supply the verified M7 tokenizer."""
    if tokenizer.vocabulary_size != m10.TOKENIZER_VOCABULARY:
        raise Refusal("Frozen M7 vocabulary identity mismatch")
    pairs = []
    data = cases()
    for i in (0, 2):
        group = data[i:i + 2]
        prompts = [prompt_for(c) for c in group]
        answers = [c["snippet"] for c in group]
        p = [tokenizer.encode(s) for s in prompts]
        a = [tokenizer.encode(s) for s in answers]
        if any(tokenizer.decode(ids) != text for ids, text in zip(p, prompts)):
            raise Refusal("Prompt fails exact tokenizer round trip")
        if any(tokenizer.decode(ids) != text for ids, text in zip(a, answers)):
            raise Refusal("Answer fails exact tokenizer round trip")
        if any(not x or not y or len(x) + len(y) > MAX_POSITIONS
               or len(y) + 1 > MAX_GENERATED for x, y in zip(p, a)):
            raise Refusal("Position or generation budget exceeded")
        src = one_change(*p)
        tgt = one_change(*a)
        if ([p[0][src], p[1][src]] != [a[0][tgt], a[1][tgt]]
                or p[0][-1] != tokenizer.encode(" ")[0]
                or tokenizer.encode(" ") != [p[0][-1]]
                or tokenizer.decode([a[0][-1]]) != "\n"):
            raise Refusal("Source and output tokens must be paired; prompt must end in space")
        pairs.append({
            "split": group[0]["split"],
            "ids": [c["id"] for c in group],
            "sourceTokenIndex": src,
            "answerTokenIndex": tgt,
            "contrastTokenIds": [a[0][tgt], a[1][tgt]],
            "sameLengthSourceAndTargets": True,
        })
    return {**plan(), "tokenizerAudit": pairs, "tokenizerRoundTripVerified": True}


def paired_preference(margin_for_zero: float, margin_for_one: float,
                      greedy_for_zero: int, greedy_for_one: int,
                      zero_token: int, one_token: int) -> bool:
    """Margins are logit(one)-logit(zero) at matched *gold-prefix* contexts."""
    values = (margin_for_zero, margin_for_one)
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise Refusal("Non-finite or non-numeric contrast margin")
    if any(type(x) is not int for x in (
            greedy_for_zero, greedy_for_one, zero_token, one_token)):
        raise Refusal("Expected integer token IDs")
    return (zero_token != one_token and margin_for_zero < 0 < margin_for_one
            and greedy_for_zero == zero_token and greedy_for_one == one_token)


def score_caller_supplied(responses: list[dict]) -> dict:
    """Pure rubric only; does not verify checkpoint, decoder, or inference origin."""
    expected = {case["id"]: case for case in cases()}
    if len(responses) != len(expected):
        raise Refusal("Exactly four response records are required")
    found = {}
    for row in responses:
        if (type(row) is not dict
                or set(row) != {"id", "output", "terminationReason"}
                or type(row["id"]) is not str or row["id"] not in expected
                or row["id"] in found or type(row["output"]) is not str
                or type(row["terminationReason"]) is not str):
            raise Refusal("Invalid, duplicated or missing output identity")
        found[row["id"]] = (
            row["output"] == expected[row["id"]]["snippet"]
            and row["terminationReason"] == "eos"
        )
    train = all(ok for key, ok in found.items() if expected[key]["split"] == "train")
    probe = all(ok for key, ok in found.items() if expected[key]["split"] == "probe-only")
    return {
        "protocol": PROTOCOL,
        "evidenceSource": "caller-supplied-unverified-not-a-model-receipt",
        "exactAndEosById": found,
        "trainPairExactAndEos": train,
        "probePairExactAndEos": probe,
        "completeOutputGate": train and probe,
        "sourcePreferenceGateNotEvaluated": True,
        "trainingAuthorized": False,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-frozen-tokenizer", action="store_true")
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args(argv)
    if not args.audit_frozen_tokenizer:
        print(json.dumps(plan(), indent=2))
        return
    if args.bundle is None:
        parser.error("--audit-frozen-tokenizer requires --bundle")
    from . import plex600_m8_eval as m8
    from . import plex600_research as research
    from .tokenizer import PlexTokenizer

    m8.verified_bundle(args.bundle, research.BUNDLE_SHA)
    print(json.dumps(audit_tokenizer(PlexTokenizer.load(args.bundle)), indent=2))


if __name__ == "__main__":
    main()
