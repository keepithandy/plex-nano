"""M8 checkpoint-bound read-only teacher-forced *diagnostic*, not generation.

Only explicitly selected owner-local frozen M7 checkpoints can be evaluated.
Each gold answer is provided as context to score reference-token probabilities;
these numbers are NOT independent generated-edit success or a training signal.
No gradients, optimizer, checkpoint mutation, final holdout or external data.
Default command prints a no-load plan. Execution is explicit and create-only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from . import plex600_feasibility as hardware
from . import plex600_m8_eval as m8

PROTOCOL = "plex600-m8-teacher-forced-reference-diagnostic-v1"


def plan(*, max_cases: int = m8.DEFAULT_MAX_CASES) -> dict[str, Any]:
    m8.plan(max_cases=max_cases, max_new_tokens=96)
    return {
        "protocol": PROTOCOL, "status": "plan-only",
        "maxCases": max_cases, "modelFamily": "plex-600m-v1",
        "mode": "read-only-gold-conditioned-next-token-diagnostic",
        "goldAnswerProvidedToModel": True, "generatedAnswers": 0,
        "independentEditSuccessMeasured": False,
        "trainingAuthorized": False, "optimizerUpdates": 0,
        "finalHoldoutUsed": False, "modelLoaded": False,
        "checkpointWrites": False, "output": "create-only-diagnostic-report",
    }


def score_logits(logits: Any, targets: list[int], *,
                 fitted_vocab_size: int, torch: Any) -> dict[str, Any]:
    """Token-weighted sums; unmasked 32K logits are the model's real distribution.

    Conditional-on-fitted-vocab likelihood is a diagnostic ablation ONLY.
    Predictions are never masked or edited in this scoring path.
    """
    if (len(targets) < 2 or targets[-1] != 3
            or any(type(t) is not int or t < 4 or t >= fitted_vocab_size
                   for t in targets[:-1])
            or type(fitted_vocab_size) is not int or fitted_vocab_size < 5
            or not hasattr(logits, "ndim") or logits.ndim != 2
            or logits.shape[0] != len(targets)
            or logits.shape[1] < fitted_vocab_size):
        raise m8.Refusal("Teacher-forced reference shape/vocabulary/EOS mismatch")
    scores = logits.detach().float()
    if not bool(torch.isfinite(scores).all()):
        raise m8.Refusal("Teacher-forced model logits are non-finite")
    reference = torch.tensor(targets, dtype=torch.long, device=scores.device)
    all_norm = torch.logsumexp(scores, dim=-1)
    fitted_norm = torch.logsumexp(scores[:, :fitted_vocab_size], dim=-1)
    chosen = scores.gather(1, reference[:, None]).squeeze(1)
    full_nll = all_norm - chosen
    fitted_nll = fitted_norm - chosen
    valid_mass = (fitted_norm - all_norm).exp()
    greedy_ids = torch.argmax(scores, dim=-1)
    n = len(targets)
    return {
        "referenceTokensIncludingEos": n,
        "unmaskedReferenceNllSum": float(full_nll.sum().item()),
        "conditionalFittedVocabReferenceNllSum": float(fitted_nll.sum().item()),
        "fittedVocabProbabilityMassSum": float(valid_mass.sum().item()),
        "unmaskedGreedyReferenceMatches": int((greedy_ids == reference).sum().item()),
        "unmaskedGreedyOutsideFittedVocab": int(
            (greedy_ids >= fitted_vocab_size).sum().item()),
        "unmaskedGreedyInvalidControl": int(
            ((greedy_ids < 4) & (greedy_ids != 3)).sum().item()),
        "unmaskedGreedyPrematureEos": int(
            ((greedy_ids == 3) & (reference != 3)).sum().item()),
        "goldConditionedNotGenerated": True,
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise m8.Refusal("Empty teacher-forced diagnostic")
    total = sum(case["referenceTokensIncludingEos"] for case in cases)
    if total < 1:
        raise m8.Refusal("No reference tokens")
    def weighted(key: str) -> float:
        return sum(float(case[key]) for case in cases) / total
    full = weighted("unmaskedReferenceNllSum")
    result = {
        "caseCount": len(cases), "referenceTokensIncludingEos": total,
        "unmaskedReferenceMeanNllNats": full,
        "conditionalFittedVocabMeanNllNats": weighted(
            "conditionalFittedVocabReferenceNllSum"),
        "meanFittedVocabProbabilityMass": weighted(
            "fittedVocabProbabilityMassSum"),
        "unmaskedGreedyReferenceTokenAccuracy": sum(
            case["unmaskedGreedyReferenceMatches"] for case in cases) / total,
        "unmaskedGreedyOutsideFittedVocabRate": sum(
            case["unmaskedGreedyOutsideFittedVocab"] for case in cases) / total,
        "unmaskedGreedyInvalidControlRate": sum(
            case["unmaskedGreedyInvalidControl"] for case in cases) / total,
        "unmaskedGreedyPrematureEosRate": sum(
            case["unmaskedGreedyPrematureEos"] for case in cases) / total,
    }
    if not all(math.isfinite(x) for x in result.values() if type(x) is float):
        raise m8.Refusal("Non-finite teacher-forced summary")
    return result


def reference_alignment(prompt_ids: list[int],
                        reference_ids: list[int]) -> tuple[list[int], int]:
    """Build causal inputs without feeding EOS; first answer scored after prompt."""
    if (not prompt_ids or not reference_ids or reference_ids[-1] != 3
            or len(reference_ids) > 96
            or len(prompt_ids) + len(reference_ids) - 1 > 1024):
        raise m8.Refusal("M8 teacher-forced prompt/reference exceeds context")
    return prompt_ids + reference_ids[:-1], len(prompt_ids) - 1


def run(bundle: Path, report: Path, weights: Path, output: Path, *,
        bundle_sha: str, max_cases: int) -> dict[str, Any]:
    proposal = plan(max_cases=max_cases)
    target = m8.require_output(output, bundle=bundle, checkpoint=weights)
    proof = m8.prerequisites(bundle, report, weights, bundle_sha)
    from .tokenizer import PlexTokenizer
    tokenizer = PlexTokenizer.load(bundle)
    cases = m8.choose_cases(max_cases)
    budget = m8.token_budget(cases, tokenizer, max_new_tokens=96)
    if budget["blockedByBudget"]:
        raise m8.Refusal("Frozen M8 reference exceeds bounded 96-token diagnostic")
    prepared = []
    for case in cases:
        prompt_ids = tokenizer.encode(m8.prompt_for(case))
        reference_ids = tokenizer.encode(case["expectedFile"]) + [3]
        # Only diagnostic gold tokens are appended to the model input;
        # the first scored logit is conditioned on the final prompt token.
        input_ids, first_answer_logit = reference_alignment(
            prompt_ids, reference_ids)
        prepared.append((case, input_ids, first_answer_logit, reference_ids))

    # Everything above is CPU-side; the expensive model is never initialized
    # for invalid source/benchmark/context/output requests.
    import torch
    from .plex600 import Plex600Config, Plex600Model, parameter_count
    hardware._check_device(torch, 0)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise m8.Refusal("Native BF16 GPU support is required")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(hardware.MEMORY_FRACTION, device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.device("cpu"):
            model = Plex600Model(Plex600Config())
        if parameter_count(model) != hardware.PARAMETERS:
            raise m8.Refusal("Frozen M7 checkpoint architecture mismatch")
        state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        if type(state) is not dict:
            raise m8.Refusal("M7 checkpoint must be a model-only state dictionary")
        model.load_state_dict(state, strict=True)
        del state
        model.to(device=device, dtype=torch.float32)
        model.eval()

        observations: list[dict[str, Any]] = []
        with torch.inference_mode():
            for case, input_ids, first_answer_logit, reference_ids in prepared:
                inputs = torch.tensor(
                    [input_ids], dtype=torch.long, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw_logits, _ = model(inputs)
                # Align the logit after the last prompt token with the first
                # expected-file token, and the last logit with EOS.
                answer_logits = raw_logits[0, first_answer_logit:, :]
                metric = score_logits(
                    answer_logits, reference_ids,
                    fitted_vocab_size=tokenizer.vocabulary_size, torch=torch)
                observations.append({
                    "id": case["id"], "language": case["language"],
                    "promptTokens": first_answer_logit + 1,
                    **metric,
                })
                del raw_logits, answer_logits, inputs
        result = {
            **proposal,
            "status": "teacher-forced-reference-diagnostic-complete",
            "modelLoaded": True,
            "checkpointSha256": proof["m7Receipt"]["localCheckpointSha256"],
            "originalReportSha256": proof["m7Receipt"]["localReportSha256"],
            "bundleManifestSha256": bundle_sha,
            "tokenizerSha256": proof["tokenizerSha256"],
            "benchmarkSha256": proof["benchmarkSha256"],
            "sourceBindingVerified": True,
            "modelOutputVocabularyCapacity": 32768,
            "fittedTokenizerVocabularySize": tokenizer.vocabulary_size,
            "referenceBudget": budget,
            "aggregate": summarize(observations),
            "byLanguage": {
                language: summarize([c for c in observations if c["language"] == language])
                for language in sorted({c["language"] for c in observations})
            },
            "cases": observations,
            "peakCudaReservedBytes": torch.cuda.max_memory_reserved(device),
        }
        with (stage / "teacher-forced-report.json").open(
                "x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        if target.exists():
            raise m8.Refusal("Destination appeared during diagnostic")
        os.rename(stage, target)
        stage = None
        return result
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-read-only-teacher-forced", action="store_true")
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--bundle-manifest-sha256")
    parser.add_argument("--m7-report", type=Path)
    parser.add_argument("--m7-weights", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-cases", type=int, default=m8.DEFAULT_MAX_CASES)
    args = parser.parse_args(argv)
    try:
        proposal = plan(max_cases=args.max_cases)
        if args.run_read_only_teacher_forced:
            if not all((args.bundle_dir, args.bundle_manifest_sha256,
                        args.m7_report, args.m7_weights, args.output_dir)):
                raise m8.Refusal("Explicit diagnostic requires all pinned local inputs and new output directory")
            result = run(
                args.bundle_dir, args.m7_report, args.m7_weights, args.output_dir,
                bundle_sha=args.bundle_manifest_sha256, max_cases=args.max_cases)
        else:
            result = proposal
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print("Plex M8 teacher-forced diagnostic refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
