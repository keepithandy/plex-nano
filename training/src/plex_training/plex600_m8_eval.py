"""M8 read-only, checkpoint-bound greedy evaluation of an explicitly selected M7 run.

No training mode, gradients, optimizer, checkpoint writes, or external data.
Default CLI shows plan only. Model loading requires verified local M7 report,
SHA-256-matching weights, source-bound tokenizer and explicit eval opt-in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from . import plex600_research as r
from . import plex600_feasibility as hardware
from . import plex600_setup as setup

PROTOCOL = "plex600-m8-read-only-generation-v1"
OUTPUT_ROOT = r.ROOT / "training" / "artifacts"
DEFAULT_MAX_CASES = 6
DEFAULT_MAX_NEW = 32


class Refusal(ValueError):
    pass


def plan(*, max_cases: int = DEFAULT_MAX_CASES,
         max_new_tokens: int = DEFAULT_MAX_NEW) -> dict[str, Any]:
    if (type(max_cases) is not int or not 1 <= max_cases <= 36
            or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 64):
        raise Refusal("Read-only evaluation is bounded to 1..36 cases and 1..64 new tokens")
    return {
        "protocol": PROTOCOL, "status": "plan",
        "modelFamily": "plex-600m-v1", "modelParameterCount": 600372480,
        "mode": "read-only-greedy-generation-no-grad",
        "maxCases": max_cases, "maxNewTokensPerCase": max_new_tokens,
        "benchmarkCasesTotal": 36, "promptFormat": "request-source-file-edited-file",
        "modelLoaded": False, "trainingAuthorized": False, "optimizerUpdates": 0,
        "input": "sha-verified-owner-m7-checkpoint-and-source-bound-local-tokenizer",
        "checkpointWrites": False, "output": "create-only-raw-responses-and-report",
        "externalSourcesUsed": False, "finalHoldoutUsed": False,
    }


def prompt_for(task: dict[str, Any]) -> str:
    return ("Request: " + task["request"] + "\nSource file:\n"
            + task["inputFile"] + "\nEdited file:\n")


def require_output(path: Path, *, bundle: Path, checkpoint: Path) -> Path:
    root = OUTPUT_ROOT.resolve(strict=True)
    dest = path.resolve(strict=False)
    if (dest.exists() or path.is_symlink() or dest == root or root not in dest.parents
            or bundle.resolve(strict=True) in dest.parents
            or checkpoint.resolve(strict=True) in dest.parents
            or not dest.parent.is_dir() or dest.parent.is_symlink()):
        raise Refusal("M8 output must be a new directory directly under training/artifacts")
    if dest.parent != root:
        raise Refusal("M8 output must be a direct child of training/artifacts")
    return dest


def prerequisites(bundle: Path, report: Path, weights: Path,
                  bundle_sha: str) -> dict[str, Any]:
    if bundle_sha != r.BUNDLE_SHA:
        raise Refusal("M8 requires separately pinned M7 bundle manifest SHA-256")
    # Includes SHA-256 validation of the local >=2 GiB checkpoint bytes.
    receipt = r.m7_local_receipt(report, weights)
    if receipt["checkpointReadAndHashVerified"] is not True:
        raise Refusal("Checkpoint hash not verified")
    bound = setup.check_source_binding(bundle, r.ROOT / "training/pretraining/plex600-owner-authored-pilot-v1",
                                       bundle_manifest_sha256=bundle_sha,
                                       source_manifest_sha256=r.SOURCE_SHA)
    checked = setup.check_tokenizer_bundle(bundle)
    if (bound["sourceBindingVerified"] is not True
            or checked["actualVocabularySize"] != 1191
            or checked["tokenizerSha256"] !=
               "9feef87bdf5fc28b3315df589e4e0b8ffcf5f40b18a4c357c91759182340a2e2"):
        raise Refusal("M8 tokenizer/source identity mismatch")
    return {"m7Receipt": receipt, "sourceBinding": bound,
            "tokenizerSha256": checked["tokenizerSha256"],
            "benchmarkSha256": r.digest_file(r.BENCHMARK)}


def generate_one(model: Any, torch: Any, tokenizer: Any, tokens: list[int],
                 *, device: Any, max_new_tokens: int) -> tuple[str, int, bool]:
    """Greedy decode with full recomputation; no KV cache or sampled claims."""
    generated: list[int] = []
    reached_eos = False
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = tokens + generated
            if len(context) > 1024:
                raise Refusal("Generation context exceeded 1024 model positions")
            inputs = torch.tensor([context], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(inputs)
            nxt = int(torch.argmax(logits[0, -1, :]).item())
            if nxt == 3:
                reached_eos = True
                break
            if nxt < 4 or nxt >= 1191:
                # Special control tokens are not valid generated text; stop without
                # laundering them into a matching output.
                break
            generated.append(nxt)
    return tokenizer.decode(generated), len(generated), reached_eos


def evaluate(bundle: Path, report: Path, weights: Path, output: Path, *,
             bundle_sha: str, max_cases: int, max_new_tokens: int) -> dict[str, Any]:
    proposal = plan(max_cases=max_cases, max_new_tokens=max_new_tokens)
    import torch  # GPU/model imported ONLY inside the explicit evaluation branch.
    from .plex600 import Plex600Model, Plex600Config, parameter_count
    from .tokenizer import PlexTokenizer
    target = require_output(output, bundle=bundle, checkpoint=weights)
    proof = prerequisites(bundle, report, weights, bundle_sha)
    hardware._check_device(torch, 0)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise Refusal("Native BF16 GPU support is required")
    tasks = r.benchmark()[:max_cases]
    tokenizer = PlexTokenizer.load(bundle)
    prepared = [(task, tokenizer.encode(prompt_for(task))) for task in tasks]
    if any(not ids or len(ids) + max_new_tokens > 1024 for _, ids in prepared):
        raise Refusal("Input + expected generated tokens exceeds context")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(hardware.MEMORY_FRACTION, device)
        torch.cuda.reset_peak_memory_stats(device)
        # No optimizer, no backward, no train() call anywhere in this path.
        config = Plex600Config()
        with torch.device("cpu"):
            model = Plex600Model(config)
        if parameter_count(model) != hardware.PARAMETERS:
            raise Refusal("Checkpoint architecture mismatch")
        state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        if type(state) is not dict:
            raise Refusal("M7 checkpoint must be model-only parameter dictionary")
        model.load_state_dict(state, strict=True)
        del state
        model.to(device=device, dtype=torch.float32)
        model.eval()
        responses = []
        outputs = []
        for task, ids in prepared:
            generated, count, eos = generate_one(model, torch, tokenizer, ids, device=device,
                                                 max_new_tokens=max_new_tokens)
            response = {"id": task["id"], "output": generated}
            outputs.append(json.dumps(response, ensure_ascii=False, sort_keys=True))
            responses.append({"id": task["id"], "promptTokens": len(ids),
                              "generatedTokens": count, "reachedEos": eos})
        raw = ("\n".join(outputs) + "\n").encode("utf-8")
        with (stage / "responses.jsonl").open("xb") as stream:
            stream.write(raw)
        result = {**proposal,
                  "status": "read-only-generation-complete",
                  "modelLoaded": True,
                  "checkpointSha256": proof["m7Receipt"]["localCheckpointSha256"],
                  "originalReportSha256": proof["m7Receipt"]["localReportSha256"],
                  "bundleManifestSha256": bundle_sha,
                  "benchmarkSha256": proof["benchmarkSha256"],
                  "responseSha256": hashlib.sha256(raw).hexdigest(),
                  "generatedCases": len(responses),
                  "cases": responses,
                  "peakCudaReservedBytes": torch.cuda.max_memory_reserved(device)}
        with (stage / "generation-report.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        if target.exists():
            raise Refusal("Destination appeared during evaluation")
        os.rename(stage, target)
        stage = None
        return result
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-read-only-eval", action="store_true")
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--bundle-manifest-sha256")
    parser.add_argument("--m7-report", type=Path)
    parser.add_argument("--m7-weights", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-cases", type=int, default=DEFAULT_MAX_CASES)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW)
    args = parser.parse_args(argv)
    try:
        proposed = plan(max_cases=args.max_cases, max_new_tokens=args.max_new_tokens)
        if not args.run_read_only_eval:
            result = proposed
        else:
            if not all([args.bundle_dir, args.bundle_manifest_sha256, args.m7_report,
                        args.m7_weights, args.output_dir]):
                raise Refusal("Explicit evaluation requires all pinned local inputs and a new output directory")
            result = evaluate(args.bundle_dir, args.m7_report, args.m7_weights,
                              args.output_dir, bundle_sha=args.bundle_manifest_sha256,
                              max_cases=args.max_cases, max_new_tokens=args.max_new_tokens)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print("Plex 600M read-only evaluation refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
