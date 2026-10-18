"""P3-SB03: read-only frozen-tokenizer token-PARITY preflight for comparison.

This reports blockers; it does NOT load a model, stage weights, pack a bundle,
sample training batches, generate text or start optimizer steps.

Two arms are a proposed causal comparison, each requiring its own separate
approval. The entire local historic training inventory is NOT attested here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
GEN_PATH = HERE / "prepare_namespace_diversity_candidate.py"
TOKENIZER = Path.home() / "PlexArtifacts/file-edits/p3-copy-fidelity-v1-bundle/tokenizer.json"
EXPECTED_PROPOSAL_SHA = "4bbdc55782d9447271323378367831656dc1f05cb5bb174138738a6a7e5e7dea"

SPEC = importlib.util.spec_from_file_location("p3_sb03_candidate", GEN_PATH)
assert SPEC and SPEC.loader
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


def frozen_proposal() -> dict[str, Any]:
    raw = GEN._checked(GEN.OUTPUT, EXPECTED_PROPOSAL_SHA)
    if raw != GEN.serialized():
        raise ValueError("Frozen SB03 proposal differs from deterministic generator")
    candidate = json.loads(raw)
    if (candidate.get("status") != "frozen-proposal-not-token-matched-not-authorized"
            or candidate["authorization"].get("trainingAuthorized") is not False
            or candidate["authorization"].get("entireHistoricalTrainingInventoryVerified") is not False
            or candidate["authorization"].get("tokenizerParityVerified") is not False):
        raise ValueError("Candidate preparation permissions changed")
    return candidate


def audit(tokenizer: Any) -> dict[str, Any]:
    candidate = frozen_proposal()
    old = json.loads(GEN._checked(GEN.FROZEN, GEN.SOURCE_SHA))
    if tokenizer.vocabulary_size != 16384:
        raise ValueError("Frozen tokenizer vocabulary not 16384")
    mismatches: list[dict[str, Any]] = []
    source_max = {"prompt": 0, "answerAndEos": 0}
    new_max = {"prompt": 0, "answerAndEos": 0}
    totals = {"comparator": 0, "intervention": 0}
    train_target_lengths = {"comparator": [], "intervention": []}
    for left, right in zip(old["rows"], candidate["rows"], strict=True):
        if (left["split"] != right["split"] or left["language"] != right["language"]
                or left["condition"] != right["condition"]):
            raise ValueError("Comparison train/validation row pairing changed")
        details = []
        for variant, row, maxima in (
            ("comparator", left, source_max), ("intervention", right, new_max)
        ):
            pids = tokenizer.encode(row["prompt"])
            aids = tokenizer.encode(row["expectedFile"])
            if (not pids or not aids or tokenizer.decode(pids) != row["prompt"]
                    or tokenizer.decode(aids) != row["expectedFile"]
                    or any(type(t) is not int or not 4 <= t < 16384 for t in pids + aids)
                    or len(pids) + len(aids) + 1 > 512
                    or len(pids) + 193 > 512
                    or len(aids) + 1 > 192):
                raise ValueError(f"Tokenizer roundtrip/budget failure: {variant}: {row['id']}")
            maxima["prompt"] = max(maxima["prompt"], len(pids))
            maxima["answerAndEos"] = max(maxima["answerAndEos"], len(aids) + 1)
            if row["split"] == "train":
                train_target_lengths[variant].append(len(aids) + 1)
            totals[variant] += len(aids) + 1
            details.append((len(pids), len(aids) + 1))
        if details[0] != details[1]:
            mismatches.append({
                "comparatorRecordId": left["id"],
                "interventionRecordId": right["id"],
                "split": left["split"], "comparatorPromptAndAnswerTokens": list(details[0]),
                "interventionPromptAndAnswerTokens": list(details[1]),
            })
    draw = candidate["sampler"]["drawIndexArray"]
    if (len(draw) != 1600 or len(set(draw)) != 288
            or len(train_target_lengths["comparator"]) != 288
            or len(train_target_lengths["intervention"]) != 288):
        raise ValueError("Frozen sampler/target coverage changed")
    draw_targets = {
        variant: sum(lengths[i] for i in draw)
        for variant, lengths in train_target_lengths.items()
    }
    return {
        "status": "token-count-parity-audit-complete",
        "candidateSha256": EXPECTED_PROPOSAL_SHA,
        "tokenizerSha256": GEN.TOKENIZER_SHA,
        "rows": 384,
        "trainRows": 288, "validationRows": 96,
        "tokenParityExact": len(mismatches) == 0,
        "tokenParityMismatchCount": len(mismatches),
        "firstMismatches": mismatches[:16],
        "perArmMaximums": {"comparator": source_max, "intervention": new_max},
        "supervisedAnswerAndEosPositionTotals": totals,
        "drawWeightedTrainTargetPositions": draw_targets,
        "drawCoverage": 288,
        "globalHistoricalTrainingInventoryVerified": False,
        "readyForTraining": False,
        "modelLoaded": False, "optimizerUpdates": 0, "trainingAuthorized": False,
        "finalHoldoutOpened": False,
    }


def audit_file(path: Path = TOKENIZER) -> dict[str, Any]:
    frozen_proposal()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Frozen tokenizer missing, linked or oversized")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != GEN.TOKENIZER_SHA:
        raise ValueError("Frozen tokenizer SHA changed")
    from tokenizers import Tokenizer
    from plex_training.tokenizer import PlexTokenizer
    return audit(PlexTokenizer(Tokenizer.from_str(raw.decode("utf-8"))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    args = parser.parse_args()
    try:
        print(json.dumps(audit_file(args.tokenizer), indent=2, sort_keys=True))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"p3-sb03-tokenizer-audit: {exc}\n")


if __name__ == "__main__":
    main()
