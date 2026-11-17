"""M10 read-only M9 copying baseline: no optimizer, gradients or checkpoint writes.

Default invocation prints a plan. --audit-inputs validates a separately backed-up
M9 model-only checkpoint and the frozen M7 tokenizer WITHOUT importing torch.
Only --run-read-only-baseline opts into bounded GPU inference and create-only
JSON artifacts. No training mode or approval semantics exist in this module.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from . import plex600_m10_copy_proposal as proposal
from . import plex600_m8_eval as m8
from . import plex600_research as research
from . import plex600_feasibility as hardware

PROTOCOL = "plex600-m10-m9-read-only-copy-baseline-v1"
MAX_CASES = 18
DEFAULT_CASES = 3
MAX_NEW_TOKENS = 96
MAX_WALL_SECONDS = 1800
DECODERS = ("unmasked-v1", "fitted-vocab-v1")
MODES = ("both",) + DECODERS


class Refusal(ValueError):
    pass


def plan() -> dict[str, Any]:
    """Safe default: no filesystem reads, model loads or GPU."""
    return {
        "protocol": PROTOCOL,
        "status": "plan-only-no-model",
        "source": "independent-sha-verified-M9-backup",
        "tokenizer": "frozen-source-bound-original-M7",
        "dataset": "18-unapproved-review-only-copy-proposal-records",
        "selectedCasesDefault": DEFAULT_CASES,
        "selectedCasesMaximum": MAX_CASES,
        "maxNewTokensPerCase": MAX_NEW_TOKENS,
        "decodingPolicies": list(DECODERS),
        "defaultDecodingMode": "both",
        "runFlag": "--run-read-only-baseline",
        "timeBudgetSeconds": MAX_WALL_SECONDS,
        "trainingAuthorized": False,
        "optimizerUpdates": 0,
        "gradientEnabled": False,
        "optimizerStateLoaded": False,
        "modelLoaded": False,
        "checkpointWrites": False,
        "artifactWrites": False,
        "finalHoldoutUsed": False,
    }


def selected_candidates(limit: int) -> list[dict[str, Any]]:
    """Warmup example first, then round-robin language-balanced selection."""
    if type(limit) is not int or not 1 <= limit <= MAX_CASES:
        raise Refusal("M10 baseline requires 1..18 selected review cases")
    candidates = proposal.rows()
    warmup = next((r for r in candidates if r["id"] == proposal.WARMUP_RECORD_ID), None)
    if warmup is None:
        raise Refusal("M10 baseline warmup ID missing")
    ordered = [warmup]
    grouped = {language: [r for r in candidates if r["language"] == language]
               for language in research.LANGUAGES}
    for index in range(6):
        for language in research.LANGUAGES:
            record = grouped[language][index]
            if record["id"] != warmup["id"]:
                ordered.append(record)
    if len(ordered) != MAX_CASES or len({r["id"] for r in ordered}) != MAX_CASES:
        raise Refusal("M10 baseline selected duplicate or missing candidate ID")
    return ordered[:limit]


def policy_names(mode: str) -> tuple[str, ...]:
    if mode not in MODES:
        raise Refusal("M10 baseline decoder selection is invalid")
    return DECODERS if mode == "both" else (mode,)


def audit_inputs(bundle: Path, bundle_sha: str, report: Path, weights: Path,
                 *, max_cases: int = DEFAULT_CASES, mode: str = "both") -> dict[str, Any]:
    """CPU-only identity, exact-copy data and complete-context audit."""
    if bundle_sha != research.BUNDLE_SHA:
        raise Refusal("M10 baseline requires pinned M7 tokenizer bundle identity")
    policies = policy_names(mode)
    backup = proposal.audit_m9_backup(report, weights)
    bound = m8.verified_bundle(bundle, bundle_sha)
    from .tokenizer import PlexTokenizer
    tokenizer = PlexTokenizer.load(bundle)
    tokenizer_proof = proposal.audit(tokenizer)
    selected = selected_candidates(max_cases)
    input_details = []
    for row in selected:
        prompt = proposal.prompt_for(row)
        prompt_ids = tokenizer.encode(prompt)
        expected_ids = tokenizer.encode(row["snippet"])
        if (not prompt_ids or not expected_ids
                or len(prompt_ids) + MAX_NEW_TOKENS > 1024
                or len(expected_ids) + 1 > MAX_NEW_TOKENS
                or tokenizer.decode(prompt_ids) != prompt
                or tokenizer.decode(expected_ids) != row["snippet"]):
            raise Refusal("M10 baseline token budget or round-trip invalid")
        input_details.append({
            "id": row["id"], "split": row["split"],
            "language": row["language"],
            "promptTokens": len(prompt_ids),
            "expectedTokensIncludingEos": len(expected_ids) + 1,
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        })
    return {
        **plan(),
        "status": "read-only-inputs-verified-no-model",
        "selectedCases": len(selected),
        "selectedCaseIds": [r["id"] for r in selected],
        "selectedCasesIncludeWarmup": selected[0]["id"] == proposal.WARMUP_RECORD_ID,
        "m9Backup": backup,
        "m9WeightsSha256": backup["m9WeightsSha256"],
        "m9ReportSha256": backup["m9ReportSha256"],
        "tokenizerSha256": tokenizer_proof["tokenizerSha256"],
        "bundleManifestSha256": bundle_sha,
        "candidateSha256": tokenizer_proof["candidateSha256"],
        "decodingPolicies": list(policies),
        "sampleBudget": input_details,
        "sourceBindingVerified": bound["sourceBinding"]["sourceBindingVerified"],
        "trainingAuthorized": False, "optimizerUpdates": 0,
        "modelLoaded": False, "checkpointWrites": False,
        "artifactWrites": False, "finalHoldoutUsed": False,
    }


def score_observations(cases: list[dict[str, Any]],
                       responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Score exactly the supplied set under one policy; never infer training."""
    reference = {row["id"]: row for row in cases}
    if (not 1 <= len(cases) <= MAX_CASES
            or len(reference) != len(cases)
            or len(responses) != len(cases)
            or any(type(x) is not dict
                   or set(x) != {"id", "output", "terminationReason"}
                   or type(x["id"]) is not str or type(x["output"]) is not str
                   or type(x["terminationReason"]) is not str
                   or len(x["output"]) > proposal.MAX_RESPONSE_CHARACTERS
                   for x in responses)):
        raise Refusal("M10 baseline observations require exact bounded case coverage")
    answer = {row["id"]: row for row in responses}
    if set(answer) != set(reference) or len(answer) != len(responses):
        raise Refusal("M10 baseline responses have missing, extra or duplicate IDs")
    stats: dict[str, dict[str, Any]] = {
        s: {"total": 0, "exact": 0, "exactWithEos": 0, "eos": 0,
            "spacesOnly": 0, "periodsOnly": 0, "repetitionCollapse": 0,
            "meanCharacterSimilarity": 0.0}
        for s in proposal.SPLITS
    }
    details = []
    for row in cases:
        response = answer[row["id"]]
        output = response["output"]
        similarity = proposal.character_similarity(row["snippet"], output)
        repetition = proposal.repetition_diagnostics(output)
        exact = output == row["snippet"]
        eos = response["terminationReason"] == "eos"
        score = stats[row["split"]]
        score["total"] += 1
        score["exact"] += int(exact)
        score["exactWithEos"] += int(exact and eos)
        score["eos"] += int(eos)
        score["spacesOnly"] += int(repetition["spaceOnly"])
        score["periodsOnly"] += int(repetition["periodOnly"])
        score["repetitionCollapse"] += int(repetition["repetitionCollapse"])
        score["meanCharacterSimilarity"] += similarity
        details.append({
            "id": row["id"], "split": row["split"],
            "exact": exact, "eos": eos,
            "characterSimilarity": round(similarity, 6),
            "terminationReason": response["terminationReason"],
            **repetition,
        })
    for split, val in stats.items():
        val["meanCharacterSimilarity"] = (
            round(val["meanCharacterSimilarity"] / val["total"], 6)
            if val["total"] else None
        )
    return {
        "bySplit": stats,
        "byExample": details,
        "complete18CaseComparison": len(cases) == 18,
        "allExactAndEos": len(cases) == 18 and all(
            stats[s]["exactWithEos"] == stats[s]["total"]
            for s in proposal.SPLITS
        ),
        "codeEditingImprovementProven": False,
        "generatedCodeEvaluated": True,
        "finalHoldoutUsed": False,
    }


def run_baseline(bundle: Path, bundle_sha: str, report: Path, weights: Path,
                 output: Path, *, max_cases: int, mode: str) -> dict[str, Any]:
    """Explicit read-only GPU inference; only writes fresh response/report files."""
    # Refuse existing output *before* expensive checkpoint/hash/model/GPU work.
    target = m8.require_output(output, bundle=bundle, checkpoint=weights)
    audit = audit_inputs(bundle, bundle_sha, report, weights,
                         max_cases=max_cases, mode=mode)
    selected = selected_candidates(max_cases)
    policies = policy_names(mode)
    from .tokenizer import PlexTokenizer
    tokenizer = PlexTokenizer.load(bundle)
    prepared = [(row, tokenizer.encode(proposal.prompt_for(row))) for row in selected]

    # GPU and model imports only after all local identity and no-overwrite gates.
    import torch
    from .plex600 import Plex600Config, Plex600Model, parameter_count
    hardware.source_identity()
    hardware._check_device(torch, 0)
    stage: Path | None = Path(tempfile.mkdtemp(
        prefix=f".{target.name}.staging-", dir=target.parent))
    started = time.monotonic()
    try:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(hardware.MEMORY_FRACTION, device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.device("cpu"):
            model = Plex600Model(Plex600Config())
        if parameter_count(model) != hardware.PARAMETERS:
            raise Refusal("M10 baseline checkpoint architecture mismatch")
        state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        if type(state) not in (dict, OrderedDict):
            raise Refusal("M10 baseline requires native model-only PyTorch state dictionary")
        model.load_state_dict(state, strict=True)
        del state
        model.to(device=device, dtype=torch.float32)
        model.eval()

        output_records = []
        scores = {}
        for policy in policies:
            predictions = []
            for row, prompt_ids in prepared:
                if time.monotonic() - started > MAX_WALL_SECONDS:
                    raise Refusal("M10 read-only baseline exceeded elapsed time budget")
                with torch.inference_mode():
                    generated, generated_count, termination = m8.generate_one(
                        model, torch, tokenizer, prompt_ids, device=device,
                        max_new_tokens=MAX_NEW_TOKENS, decoding_policy=policy)
                item = {"id": row["id"], "output": generated,
                        "terminationReason": termination}
                predictions.append(item)
                output_records.append({
                    **item, "split": row["split"], "language": row["language"],
                    "decodingPolicy": policy, "generatedTokens": generated_count,
                    "promptSha256": hashlib.sha256(
                        proposal.prompt_for(row).encode("utf-8")).hexdigest(),
                    "m9WeightsSha256": audit["m9WeightsSha256"],
                })
            scores[policy] = score_observations(selected, predictions)

        raw = ("".join(json.dumps(row, sort_keys=True, ensure_ascii=False)
                       + "\n" for row in output_records)).encode("utf-8")
        with (stage / "responses.jsonl").open("xb") as stream:
            stream.write(raw)
        result = {
            **plan(),
            "status": "read-only-m9-generation-complete",
            "m9WeightsSha256": audit["m9WeightsSha256"],
            "m9ReportSha256": audit["m9ReportSha256"],
            "candidateSha256": audit["candidateSha256"],
            "bundleManifestSha256": bundle_sha,
            "tokenizerSha256": audit["tokenizerSha256"],
            "selectedCaseIds": audit["selectedCaseIds"],
            "selectedCases": len(selected),
            "decodingPolicies": list(policies),
            "checkpointSource": "independently-verified-m9-weights",
            "responsesSha256": hashlib.sha256(raw).hexdigest(),
            "results": scores,
            "modelLoaded": True,
            "artifactWrites": True,
            "responseFilesWritten": ["responses.jsonl", "report.json"],
            "peakCudaReservedBytes": torch.cuda.max_memory_reserved(device),
            "elapsedSeconds": round(time.monotonic() - started, 3),
            "optimizerUpdates": 0, "trainingAuthorized": False,
            "checkpointWrites": False, "finalHoldoutUsed": False,
        }
        with (stage / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        if target.exists():
            raise Refusal("M10 baseline destination appeared during generation")
        os.rename(stage, target)
        stage = None
        return result
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit-inputs", action="store_true")
    mode.add_argument("--run-read-only-baseline", action="store_true")
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--bundle-manifest-sha256")
    parser.add_argument("--m9-report", type=Path)
    parser.add_argument("--m9-weights", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-cases", type=int, default=DEFAULT_CASES)
    parser.add_argument("--decoding", choices=MODES, default="both")
    args = parser.parse_args(argv)
    try:
        if not args.audit_inputs and not args.run_read_only_baseline:
            if any(x is not None for x in (
                    args.bundle_dir, args.bundle_manifest_sha256,
                    args.m9_report, args.m9_weights, args.output_dir)):
                raise Refusal("Input paths require explicit read-only audit or run flag")
            result = plan()
        elif not all((args.bundle_dir, args.bundle_manifest_sha256,
                      args.m9_report, args.m9_weights)):
            raise Refusal("M10 baseline requires M7 bundle identity and M9 backup paths")
        elif args.audit_inputs:
            if args.output_dir is not None:
                raise Refusal("CPU audit does not accept an output destination")
            result = audit_inputs(args.bundle_dir, args.bundle_manifest_sha256,
                                  args.m9_report, args.m9_weights,
                                  max_cases=args.max_cases, mode=args.decoding)
        elif args.output_dir is None:
            raise Refusal("Explicit baseline needs a new create-only output directory")
        else:
            result = run_baseline(args.bundle_dir, args.bundle_manifest_sha256,
                                  args.m9_report, args.m9_weights, args.output_dir,
                                  max_cases=args.max_cases, mode=args.decoding)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print("M10 read-only baseline refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
