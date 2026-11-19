"""M11: read-only M9 source-conditioning baseline under literal-space prompts.

Default CLI never accesses model sources. --audit-inputs rehashes original M9
backup and M7 tokenizer on CPU only. --run-read-only-baseline is explicit GPU
INFERENCE (not training), exactly 4 M11 prompts x 2 frozen decoders, with
create-only checkpoint-bound JSON receipts. No optimizer or training branch.
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

from . import plex600_m11_admission as admission
from . import plex600_m11_source_conditioning as m11
from . import plex600_m11_training_core as core
from . import plex600_m8_eval as m8
from . import plex600_research as research
from . import plex600_feasibility as hardware

PROTOCOL = admission.BASELINE_PROTOCOL
STATUS = admission.BASELINE_STATUS
DECODERS = core.DECODERS
MAX_NEW_TOKENS = m11.MAX_GENERATED
MAX_WALL_SECONDS = 1800
MAX_RESPONSES_BYTES = admission.RESPONSES_MAX_BYTES
MAX_REPORT_BYTES = admission.REPORT_MAX_BYTES


class Refusal(ValueError):
    pass


def plan() -> dict[str, Any]:
    """No filesystem read, torch import, GPU, optimizer, or model loading."""
    return {
        "protocol": PROTOCOL,
        "status": "plan-only-no-model",
        "checkpointSource": "separately-verified-independent-M9-backup",
        "promptBoundary": "literal-space-not-M10-final-newline",
        "selectedCaseIds": [row["id"] for row in m11.cases()],
        "caseCount": 4,
        "decodingPolicies": list(DECODERS),
        "responseCount": 8,
        "maxNewTokensPerCase": MAX_NEW_TOKENS,
        "timeBudgetSeconds": MAX_WALL_SECONDS,
        "timeBudgetEnforcement": "cooperative-between-inference-cases",
        "runFlag": "--run-read-only-baseline",
        "trainingAuthorized": False,
        "optimizerUpdates": 0,
        "gradientEnabled": False,
        "optimizerStateLoaded": False,
        "modelLoaded": False,
        "checkpointWrites": False,
        "artifactWrites": False,
        "finalHoldoutUsed": False,
    }


def audit_inputs(bundle: Path, bundle_sha: str, report: Path,
                 weights: Path) -> dict[str, Any]:
    """The exact M11 admission source audit, plus 4 complete prompt budgets."""
    source = admission.inspect_sources(bundle, bundle_sha, report, weights)
    cases = m11.cases()
    if (len(cases) != 4 or len({row["id"] for row in cases}) != 4
            or [row["split"] for row in cases]
               != ["train", "train", "probe-only", "probe-only"]):
        raise Refusal("M11 baseline requires the frozen 2+2 source contrast pairs")
    from .tokenizer import PlexTokenizer
    tokenizer = PlexTokenizer.load(bundle)
    audit = m11.audit_tokenizer(tokenizer)
    if audit["tokenizerRoundTripVerified"] is not True:
        raise Refusal("M11 tokenizer round-trip failed")
    details = []
    for row in cases:
        prompt = m11.prompt_for(row)
        if not prompt.endswith("Copy: ") or prompt.endswith("Copy:\n"):
            raise Refusal("M11 source prompt did not end with literal space")
        prefix = tokenizer.encode(prompt)
        answer = tokenizer.encode(row["snippet"])
        if (not prefix or not answer
                or tokenizer.decode(prefix) != prompt
                or tokenizer.decode(answer) != row["snippet"]
                or len(prefix) + MAX_NEW_TOKENS > 1024
                or len(prefix) + len(answer) > m11.MAX_POSITIONS
                or len(answer) + 1 > MAX_NEW_TOKENS):
            raise Refusal("M11 source prompt or expected completion exceeds budget")
        details.append({
            "id": row["id"], "split": row["split"],
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "promptTokens": len(prefix),
            "referenceAnswerTokensWithEos": len(answer) + 1,
        })
    return {
        "protocol": PROTOCOL,
        "status": "source-and-tokenizer-verified-no-model",
        "checkpointSha256": source["m9WeightsSha256"],
        "m9ReportSha256": source["m9ReportSha256"],
        "bundleManifestSha256": source["bundleManifestSha256"],
        "tokenizerSha256": source["tokenizerSha256"],
        "candidateSha256": source["candidateSha256"],
        "scheduleSha256": source["scheduleSha256"],
        "sourceHashes": source["sourceHashes"],
        "selectedCaseIds": [row["id"] for row in cases],
        "decodingPolicies": list(DECODERS),
        "responseCount": len(cases) * len(DECODERS),
        "promptBudget": details,
        "trainingAuthorized": False, "optimizerUpdates": 0,
        "modelLoaded": False, "checkpointWrites": False,
        "artifactWrites": False, "finalHoldoutUsed": False,
    }


def _receipt_rows(source: dict[str, Any],
                  observations: list[dict[str, Any]]) -> tuple[bytes, dict[str, Any]]:
    """Pure CPU sealing; scored independently by M11 admission from raw bytes."""
    cases = m11.cases()
    identities = {(policy, case["id"]) for policy in DECODERS for case in cases}
    if type(observations) is not list or len(observations) != len(identities):
        raise Refusal("M11 read-only baseline requires exactly eight actual observations")
    compact = []
    seen = set()
    allowed = {
        "id", "split", "decodingPolicy", "output", "terminationReason",
        "generatedTokens", "promptSha256", "checkpointSha256",
    }
    for row in observations:
        if type(row) is not dict or set(row) != allowed:
            raise Refusal("M11 response has missing or unexpected fields")
        pair = (row["decodingPolicy"], row["id"])
        case = next((x for x in cases if x["id"] == row["id"]), None)
        if (pair not in identities or pair in seen or case is None
                or row["split"] != case["split"]
                or type(row["output"]) is not str or len(row["output"]) > 8192
                or row["terminationReason"] not in admission.RESPONSE_TERMINATIONS
                or type(row["generatedTokens"]) is not int
                or not 0 <= row["generatedTokens"] <= MAX_NEW_TOKENS
                or row["promptSha256"] != hashlib.sha256(
                    m11.prompt_for(case).encode("utf-8")).hexdigest()
                or row["checkpointSha256"] != source["checkpointSha256"]):
            raise Refusal("M11 response missing or duplicated, or source/checkpoint changed")
        seen.add(pair)
        compact.append({
            key: row[key] for key in (
                "id", "decodingPolicy", "output", "terminationReason",
                "promptSha256", "checkpointSha256")
        })
    if seen != identities:
        raise Refusal("M11 baseline did not cover every prompt under both decoders")
    score = core.score_unverified_generations(compact, source["checkpointSha256"])
    raw = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                  for row in observations).encode("utf-8")
    if len(raw) > MAX_RESPONSES_BYTES:
        raise Refusal("M11 baseline raw responses exceed admitted maximum bytes")
    return raw, score["scores"]


def _write_receipts(stage: Path, source: dict[str, Any],
                    observations: list[dict[str, Any]],
                    *, peak_cuda_reserved: int, elapsed_seconds: float) -> dict[str, Any]:
    """Create two new receipt files, recheck by independent M11 admission."""
    raw, scores = _receipt_rows(source, observations)
    with (stage / "responses.jsonl").open("xb") as stream:
        stream.write(raw)
    result = {
        **source,
        "protocol": PROTOCOL,
        "status": STATUS,
        "responseCount": 8,
        "responsesSha256": hashlib.sha256(raw).hexdigest(),
        "scores": scores,
        "readOnlyGenerationExecuted": True,
        "generatedCodeEvaluated": True,
        "checkpointSource": "independently-verified-m9-model-only-weights",
        "responseProvenance": "actual-generator-in-this-process-receipt-only",
        "sourcePairLogitGateNotMeasured": True,
        "peakCudaReservedBytes": peak_cuda_reserved,
        "elapsedSeconds": round(elapsed_seconds, 3),
        "modelLoaded": True,
        "artifactWrites": True,
        "responseFilesWritten": ["responses.jsonl", "report.json"],
        "trainingAuthorized": False,
        "optimizerUpdates": 0,
        "gradientEnabled": False,
        "optimizerStateLoaded": False,
        "checkpointWrites": False,
        "finalHoldoutUsed": False,
    }
    with (stage / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    if (stage / "report.json").stat().st_size > MAX_REPORT_BYTES:
        raise Refusal("M11 baseline report exceeds admitted maximum bytes")
    verified = admission.read_baseline(
        stage / "report.json", stage / "responses.jsonl", {
            "m9WeightsSha256": source["checkpointSha256"],
            "m9ReportSha256": source["m9ReportSha256"],
            "bundleManifestSha256": source["bundleManifestSha256"],
            "tokenizerSha256": source["tokenizerSha256"],
            "candidateSha256": source["candidateSha256"],
            "scheduleSha256": source["scheduleSha256"],
            "sourceHashes": source["sourceHashes"],
        },
    )
    if (verified["baselineReceiptConsistencyVerified"] is not True
            or verified["trainingAuthorized"] is not False):
        raise Refusal("M11 independently rescored receipt was not accepted")
    return result


def run_baseline(bundle: Path, bundle_sha: str, report: Path,
                 weights: Path, output: Path) -> dict[str, Any]:
    """Explicit, bounded read-only GPU inference against the original M9."""
    # Existing destinations are rejected BEFORE opening multi-GB M9 weights.
    target = m8.require_output(output, bundle=bundle, checkpoint=weights)
    source = audit_inputs(bundle, bundle_sha, report, weights)
    from .tokenizer import PlexTokenizer
    tokenizer = PlexTokenizer.load(bundle)
    cases = m11.cases()
    prepared = [(row, tokenizer.encode(m11.prompt_for(row))) for row in cases]

    # CUDA and torch imports only after all source and output checks succeed.
    import torch
    from .plex600 import Plex600Config, Plex600Model, parameter_count
    hardware.source_identity()
    hardware._check_device(torch, 0)
    stage: Path | None = Path(tempfile.mkdtemp(
        prefix=f".{target.name}.staging-", dir=target.parent))
    start = time.monotonic()
    try:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(hardware.MEMORY_FRACTION, device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.device("cpu"):
            model = Plex600Model(Plex600Config())
        if parameter_count(model) != hardware.PARAMETERS:
            raise Refusal("M11 M9 checkpoint architecture parameter count mismatch")
        state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        if type(state) not in (dict, OrderedDict):
            raise Refusal("M11 baseline requires model-only state dictionary")
        model.load_state_dict(state, strict=True)
        del state
        model.to(device=device, dtype=torch.float32)
        model.eval()

        outputs = []
        for policy in DECODERS:
            for row, ids in prepared:
                if time.monotonic() - start > MAX_WALL_SECONDS:
                    raise Refusal("M11 read-only inference exceeded time budget")
                with torch.inference_mode():
                    generated, count, termination = m8.generate_one(
                        model, torch, tokenizer, ids, device=device,
                        max_new_tokens=MAX_NEW_TOKENS, decoding_policy=policy)
                outputs.append({
                    "id": row["id"], "split": row["split"], "output": generated,
                    "terminationReason": termination, "generatedTokens": count,
                    "decodingPolicy": policy,
                    "promptSha256": hashlib.sha256(
                        m11.prompt_for(row).encode("utf-8")).hexdigest(),
                    "checkpointSha256": source["checkpointSha256"],
                })

        # Verify the original checkpoint was not swapped during model load.
        actual = research.digest_file(weights, limit=admission.CHECKPOINT_MAX_BYTES)
        if actual != source["checkpointSha256"]:
            raise Refusal("Original M9 weights changed during read-only generation")
        result = _write_receipts(
            stage, source, outputs, peak_cuda_reserved=int(
                torch.cuda.max_memory_reserved(device)),
            elapsed_seconds=time.monotonic() - start)
        if target.exists():
            raise Refusal("Create-only output destination appeared during inference")
        os.rename(stage, target)
        stage = None
        return result
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--audit-inputs", action="store_true")
    flags.add_argument("--run-read-only-baseline", action="store_true")
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--bundle-manifest-sha256")
    parser.add_argument("--m9-report", type=Path)
    parser.add_argument("--m9-weights", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if not (args.audit_inputs or args.run_read_only_baseline):
            if any(value is not None for value in (
                    args.bundle_dir, args.bundle_manifest_sha256,
                    args.m9_report, args.m9_weights, args.output_dir)):
                raise Refusal("M11 source paths require explicit read-only audit or run")
            result = plan()
        elif not all((args.bundle_dir, args.bundle_manifest_sha256,
                      args.m9_report, args.m9_weights)):
            raise Refusal("M11 requires exact M7 tokenizer and independent M9 backup")
        elif args.audit_inputs:
            if args.output_dir is not None:
                raise Refusal("CPU audit cannot write a generation result")
            result = audit_inputs(
                args.bundle_dir, args.bundle_manifest_sha256,
                args.m9_report, args.m9_weights)
        elif args.output_dir is None:
            raise Refusal("M11 GPU inference requires a NEW create-only destination")
        else:
            result = run_baseline(
                args.bundle_dir, args.bundle_manifest_sha256,
                args.m9_report, args.m9_weights, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print("M11 read-only baseline refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
