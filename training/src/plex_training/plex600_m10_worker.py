"""M10: strictly owner-approved, bounded 50-update copying microprobe.

DEFAULT: plan only. No GPU, files, optimizer or model imports.
--execute-approved-run requires independently hashed M9 backup, complete
18-case × 2-decoder read-only baseline, an *external exact-scope owner
approval*, clean pinned model source, free disk/VRAM, and a new output path.

All training is confined to a child killed at 3600s. No branch merge,
approval file creation, final holdout access or historical weight writes.
This is an EXPERIMENT, not a claim that the 600M checkpoint can code.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from . import plex600_m10_admission as gate
from . import plex600_m10_copy_proposal as copy
from . import plex600_m8_eval as m8
from . import plex600_feasibility as hw
from . import plex600_m7 as m7
from . import plex600_r3_curriculum as r3
from . import plex600_research as research

PROTOCOL = "plex600-m10-approved-copy-worker-v1"
CONTROL = "PLEX600_M10_WORKER_NONCE"
HARD_TIMEOUT = 3600
SNAPSHOTS = (10, 25, 50)
DECODERS = gate.DECODERS
MAX_WEIGHTS = gate.CHECKPOINT_MAX_BYTES
SEED = copy.SEED
PROBE_ID = copy.WARMUP_RECORD_ID


class Refusal(ValueError):
    pass


def plan() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL, "status": "plan-only-no-training",
        "requiresExternalExactScopeOwnerApproval": True,
        "source": "independent-SHA-verified-M9-backup",
        "baseline": "completed-18-case-two-decoder-M9",
        "trainingCandidates": "assistant-proposed-unapproved",
        "optimizer": "fresh-AdamW-no-state-resume",
        "maxOptimizerUpdates": copy.MAX_UPDATES,
        "schedule": "10-single-copy-15-all-train-25-all-train",
        "snapshotSteps": list(SNAPSHOTS),
        "hardTimeoutSeconds": HARD_TIMEOUT,
        "maxNewTokens": copy.MAX_GENERATION,
        "gpu": "NVIDIA GeForce RTX 4080 SUPER-native-BF16",
        "trainingAuthorized": False,
        "optimizerUpdatesCompleted": 0, "modelLoaded": False,
        "checkpointWrites": False, "finalHoldoutUsed": False,
        "historicalCheckpointWrites": False,
    }


def exact_schedule() -> list[dict[str, Any]]:
    schedule = copy.update_schedule()
    if (len(schedule) != 50
            or [row["optimizerUpdate"] for row in schedule] != list(range(1, 51))
            or any(row["examplesInOptimizerUpdate"] != 1
                   or row["gradientAccumulationSteps"] != 1
                   or len(row["recordIds"]) != 1 for row in schedule)
            or {x["recordIds"][0] for x in schedule[:10]} != {PROBE_ID}
            or {x["recordIds"][0] for x in schedule[10:]}
               != {r["id"] for r in copy.rows() if r["split"] == "train"}
            or gate.schedule_sha256(schedule)
               != gate.approval_template(
                   {"m9WeightsSha256": "a"*64, "m9ReportSha256": "b"*64},
                   {"baselineReportSha256": "c"*64,
                    "baselineResponsesSha256": "d"*64})["scheduleSha256"]):
        raise Refusal("M10 update schedule no longer agrees with the proposed approval")
    return schedule


def packaged_samples(tokenizer: Any) -> dict[str, dict[str, Any]]:
    """Pack only the 12 proposed training rows; heldout never receives a label."""
    items = copy.rows()
    packed: dict[str, dict[str, Any]] = {}
    for record in items:
        if record["split"] != "train":
            continue
        prompt = tokenizer.encode(copy.prompt_for(record))
        answer = tokenizer.encode(record["snippet"]) + [3]
        value = r3.masked_record(
            prompt, answer, fitted_vocab_size=tokenizer.vocabulary_size)
        if (len(value["inputIds"]) > copy.MAX_POSITIONS
                or value["targetIds"][:len(prompt) - 1] != [-100] * (len(prompt) - 1)
                or value["targetIds"][len(prompt) - 1:] != answer
                or value["targetIds"][-1] != 3
                or value["supervisedAnswerPositions"] != len(answer)):
            raise Refusal("M10 exact answer/EOS-only supervision changed")
        packed[record["id"]] = value
    if len(packed) != 12 or PROBE_ID not in packed:
        raise Refusal("M10 exact 12 training IDs required")
    return packed


def checked_early_stop(previous: dict[str, Any] | None,
                       current: dict[str, Any]) -> bool:
    """Only real, separate, adjacent checkpoint-bound scores may stop early."""
    if previous is None:
        return False
    pair = (previous.get("optimizerUpdatesCompleted"),
            current.get("optimizerUpdatesCompleted"))
    if pair not in ((10, 25), (25, 50)):
        return False
    if (previous.get("checkpointSha256") == current.get("checkpointSha256")
            or previous.get("candidateSha256") != current.get("candidateSha256")
            or previous.get("candidateSha256") != copy.audit()["candidateSha256"]):
        return False
    for stage in (previous, current):
        if (stage.get("verifiedGeneration") is not True
                or stage.get("decodingPolicies") != list(DECODERS)
                or type(stage.get("scores")) is not dict
                or set(stage["scores"]) != set(DECODERS)
                or any(stage["scores"][policy].get("allExactAndEos") is not True
                       for policy in DECODERS)):
            return False
    return True


def _sha(path: Path) -> str:
    return m7.hash_file(path, maximum=MAX_WEIGHTS)


def _source_identity() -> dict[str, Any]:
    protected = (
        Path(__file__), Path(gate.__file__), Path(copy.__file__),
    )
    names = [path.relative_to(research.ROOT).as_posix() for path in protected]
    state = subprocess.run(
        ["git", "status", "--porcelain", "--", *names],
        cwd=research.ROOT, capture_output=True, text=True,
        check=True, timeout=8,
    )
    if state.stdout.strip():
        raise Refusal("M10 worker/admission/proposal source is locally modified")
    return {
        **hw.source_identity(),
        "m10WorkerSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "m10AdmissionSha256": hashlib.sha256(
            Path(gate.__file__).read_bytes()).hexdigest(),
        "m10ProposalSha256": hashlib.sha256(
            Path(copy.__file__).read_bytes()).hexdigest(),
    }


def _paths_from_args(args: Any) -> dict[str, Path]:
    names = ("bundle_dir", "m9_report", "m9_weights",
             "baseline_report", "baseline_responses",
             "approval_file", "output_dir")
    if any(getattr(args, name) is None for name in names):
        raise Refusal("M10 execution requires all independently verified paths")
    return {name: getattr(args, name) for name in names}


def _admit(paths: dict[str, Path], bundle_sha: str) -> dict[str, Any]:
    if bundle_sha != research.BUNDLE_SHA:
        raise Refusal("M10 requires original M7 bundle manifest SHA")
    return gate.admission(
        paths["bundle_dir"], bundle_sha,
        paths["m9_report"], paths["m9_weights"],
        paths["baseline_report"], paths["baseline_responses"],
        paths["approval_file"], paths["output_dir"],
    )


def _fit_check(path: Path) -> None:
    """Conservatively retain room for the next model-only checkpoint."""
    if shutil.disk_usage(path).free < MAX_WEIGHTS + 1024**3:
        raise Refusal("M10 model-only save lacks next-checkpoint headroom")


def _reload_parity(model: Any, torch: Any, weights: Path, probe_ids: list[int],
                   config: Any) -> dict[str, Any]:
    """Save a detached CPU copy, compare before/after reload in CPU FP32.

    NEVER move the live training model off its current device. In particular,
    moving a live CUDA module to CPU/back while AdamW retains references and
    momentum buffers can silently break optimizer continuity.
    """
    from .plex600 import Plex600Model

    if weights.exists() or weights.is_symlink():
        raise Refusal("M10 checkpoint save must be create-only")
    model.eval()
    live_parameters = tuple(model.parameters())
    live_devices = tuple(parameter.device for parameter in live_parameters)
    live_ids = tuple(id(parameter) for parameter in live_parameters)
    with torch.device("cpu"):
        reference = Plex600Model(config)
    source = model.state_dict()
    destination = reference.state_dict()
    if source.keys() != destination.keys():
        raise Refusal("M10 CPU checkpoint copy differs from source architecture")
    # Copy only one tensor at a time: do not allocate a second complete 600M
    # state dictionary or any additional CUDA model.
    with torch.no_grad():
        for name, target in destination.items():
            original = source[name].detach()
            if original.shape != target.shape or original.dtype != target.dtype:
                raise Refusal("M10 CPU snapshot tensor differs from live model")
            target.copy_(original.to(device="cpu"))
    del source, destination
    reference.eval()
    inputs = torch.tensor([probe_ids], dtype=torch.long, device="cpu")
    with torch.inference_mode():
        before_logits, _ = reference(inputs)
        before = before_logits[0, -1].float().detach().clone()
        before_greedy = int(before.argmax().item())
    with weights.open("xb") as stream:
        torch.save(reference.state_dict(), stream)
    if not 0 < weights.stat().st_size <= MAX_WEIGHTS:
        raise Refusal("M10 model-only snapshot exceeds 3 GiB")
    digest = _sha(weights)
    del reference, before_logits
    gc.collect()

    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if type(state) not in (dict, OrderedDict):
        raise Refusal("M10 saved checkpoint must be a native model-only state dictionary")
    with torch.device("cpu"):
        restored = Plex600Model(config)
    restored.load_state_dict(state, strict=True)
    del state
    restored.eval()
    with torch.inference_mode():
        after_logits, _ = restored(inputs)
        after = after_logits[0, -1].float()
        difference = float((after - before).abs().max().item())
        same_tokens = int(after.argmax().item()) == before_greedy
    del after_logits, restored, before
    gc.collect()
    if not same_tokens or not math.isfinite(difference) or difference > 0.0001:
        raise Refusal("M10 saved checkpoint fails greedy/logit reload parity")
    if (tuple(id(p) for p in model.parameters()) != live_ids
            or tuple(p.device for p in model.parameters()) != live_devices):
        raise Refusal("M10 snapshot changed live model parameter identity/device")
    return {
        "file": "weights.pt", "sha256": digest,
        "bytes": weights.stat().st_size,
        "weightsOnly": True, "optimizerStateSaved": False,
        "reloadedCheckpointSha256": _sha(weights),
        "reloadedGreedyTokensMatch": same_tokens,
        "maxReloadLogitAbsoluteDifference": difference,
        "liveModelParameterDevicePreserved": True,
        "liveModelParameterIdentityPreserved": True,
    }


def _measure_generation(model: Any, torch: Any, tokenizer: Any,
                        rows: list[dict[str, Any]], device: Any,
                        checkpoint_sha: str) -> tuple[dict[str, Any], bytes]:
    """Actual autoregressive inference, not caller-supplied scoring claims."""
    outputs: list[dict[str, Any]] = []
    scores: dict[str, Any] = {}
    for policy in DECODERS:
        predictions = []
        for row in rows:
            raw, generated_tokens, termination = m8.generate_one(
                model, torch, tokenizer, tokenizer.encode(copy.prompt_for(row)),
                device=device, max_new_tokens=copy.MAX_GENERATION,
                decoding_policy=policy,
            )
            predictions.append({
                "id": row["id"], "output": raw,
                "terminationReason": termination,
            })
            outputs.append({
                "id": row["id"], "output": raw,
                "terminationReason": termination,
                "generatedTokens": generated_tokens,
                "language": row["language"], "split": row["split"],
                "decodingPolicy": policy,
                "promptSha256": hashlib.sha256(
                    copy.prompt_for(row).encode("utf-8")).hexdigest(),
                "checkpointSha256": checkpoint_sha,
            })
        scores[policy] = copy.score_predictions(rows, predictions)
    raw = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                  for row in outputs).encode("utf-8")
    return scores, raw


def verify_checkpoint_responses(
    response_path: Path, checkpoint_sha: str,
    recorded_scores: dict[str, Any],
) -> dict[str, Any]:
    """Independently re-score all 36 checkpoint-bound predictions on CPU.

    Matching a file hash alone cannot prove that the recorded exact/EOS
    outcomes agree with the serialized outputs. This checker is shared by
    normal publication and interrupted milestone salvage.
    """
    if type(recorded_scores) is not dict or set(recorded_scores) != set(DECODERS):
        raise Refusal("M10 checkpoint scores require both decoding policies")
    expected = copy.rows()
    cases = {case["id"]: case for case in expected}
    response_rows = research.read_jsonl(
        response_path, limit=gate.RESPONSES_MAX_BYTES)
    if len(response_rows) != len(cases) * len(DECODERS):
        raise Refusal("M10 checkpoint must have exactly 18 responses per decoder")
    required = {
        "id", "output", "terminationReason", "generatedTokens",
        "language", "split", "decodingPolicy",
        "promptSha256", "checkpointSha256",
    }
    seen: set[tuple[str, str]] = set()
    predictions: dict[str, list[dict[str, Any]]] = {p: [] for p in DECODERS}
    for row in response_rows:
        if type(row) is not dict or set(row) != required:
            raise Refusal("M10 checkpoint response has missing or extra fields")
        policy = row["decodingPolicy"]
        candidate = cases.get(row["id"])
        identity = (policy, row["id"])
        if (policy not in DECODERS or candidate is None or identity in seen
                or row["checkpointSha256"] != checkpoint_sha
                or row["language"] != candidate["language"]
                or row["split"] != candidate["split"]
                or row["promptSha256"] != hashlib.sha256(
                    copy.prompt_for(candidate).encode("utf-8")).hexdigest()
                or type(row["output"]) is not str
                or len(row["output"]) > copy.MAX_RESPONSE_CHARACTERS
                or type(row["generatedTokens"]) is not int
                or not 0 <= row["generatedTokens"] <= copy.MAX_GENERATION
                or row["terminationReason"] not in (
                    "eos", "max-new-tokens", "invalid-control-token",
                    "outside-fitted-tokenizer",
                )):
            raise Refusal("M10 checkpoint response source/checkpoint/policy mismatched")
        seen.add(identity)
        predictions[policy].append({
            "id": row["id"], "output": row["output"],
            "terminationReason": row["terminationReason"],
        })
    for policy in DECODERS:
        try:
            computed = copy.score_predictions(expected, predictions[policy])
        except copy.Refusal as exc:
            raise Refusal("M10 checkpoint responses incomplete or duplicated") from exc
        if recorded_scores[policy] != computed:
            raise Refusal("M10 checkpoint scores diverge from raw generated responses")
    return {
        "responseCountVerified": len(response_rows),
        "decodingPoliciesVerified": list(DECODERS),
        "sourceCheckpointSha256": checkpoint_sha,
    }


def warmup_comparison(
    baseline_responses: list[dict[str, Any]], scores: dict[str, Any],
) -> dict[str, Any]:
    """Check the *same* one-line prompt before/after actual inference."""
    candidate = next(row for row in copy.rows() if row["id"] == PROBE_ID)
    output = {}
    for decoder in DECODERS:
        prior = [x for x in baseline_responses
                 if x.get("id") == PROBE_ID and x.get("decodingPolicy") == decoder]
        now = [x for x in scores[decoder]["byExample"] if x["id"] == PROBE_ID]
        if len(prior) != 1 or len(now) != 1:
            raise Refusal("M10 must compare the same exact warmup case under both decoders")
        before = prior[0]
        after = now[0]
        expected = candidate["snippet"]
        before_similarity = copy.character_similarity(expected, before["output"])
        output[decoder] = {
            "beforeExactAndEos": (before["output"] == expected
                                 and before["terminationReason"] == "eos"),
            "afterExactAndEos": after["exact"] and after["eos"],
            "beforeCharacterSimilarity": round(before_similarity, 6),
            "afterCharacterSimilarity": after["characterSimilarity"],
            "characterSimilarityDelta": round(
                after["characterSimilarity"] - before_similarity, 6),
            "beforeTerminationReason": before["terminationReason"],
            "afterTerminationReason": after["terminationReason"],
        }
    return output


def teacher_forced_probe(
    model: Any, torch: Any, packed: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    """Report EOS and first-answer ranks without confusing NLL with generation."""
    model.eval()
    x = torch.tensor([packed["inputIds"]], dtype=torch.long, device=device)
    y = torch.tensor([packed["targetIds"]], dtype=torch.long, device=device)
    first_index = packed["lossMaskPromptPositions"]
    eos_index = len(packed["targetIds"]) - 1
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits, loss = model(x, y)
    if loss is None or not bool(torch.isfinite(loss)):
        raise Refusal("M10 teacher-forced warmup loss invalid")
    first_logits = logits[0, first_index].float()
    eos_logits = logits[0, eos_index].float()
    first_token = y[0, first_index]
    eos_logit = eos_logits[3]
    return {
        "teacherForcedAnswerAndEosNll": float(loss.item()),
        "firstAnswerTokenRank": int((first_logits > first_logits[first_token]).sum().item()) + 1,
        "eosTokenRank": int((eos_logits > eos_logit).sum().item()) + 1,
        "eosTokenProbability": float(eos_logits.softmax(-1)[3].item()),
        "supervisedAnswerPositions": packed["supervisedAnswerPositions"],
        "notAutonomousGeneration": True,
    }


def _train_worker(paths: dict[str, Path], bundle_sha: str,
                  stage: Path) -> dict[str, Any]:
    # The child independently revalidates every owner consent/checkpoint.
    admitted = _admit(paths, bundle_sha)
    identity = _source_identity()
    from .tokenizer import PlexTokenizer
    tokenizer = PlexTokenizer.load(paths["bundle_dir"])
    packed = packaged_samples(tokenizer)
    schedule = exact_schedule()
    all_rows = copy.rows()
    # Admission independently verified all 36 raw baseline responses and
    # their SHA; only copy them for source-bound before/after comparisons.
    baseline_responses = research.read_jsonl(
        paths["baseline_responses"], limit=gate.RESPONSES_MAX_BYTES)
    probe_ids = tokenizer.encode(copy.prompt_for(
        next(row for row in all_rows if row["id"] == PROBE_ID)))

    import torch  # Only AFTER owner approval, source and baseline preflight.
    from .plex600 import Plex600Config, Plex600Model, parameter_count
    hardware = hw._check_device(torch, 0)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(hw.MEMORY_FRACTION, device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(SEED)
    config = Plex600Config()
    with torch.device("cpu"):
        model = Plex600Model(config)
    if parameter_count(model) != hw.PARAMETERS:
        raise Refusal("M10 must use unchanged 600M architecture")
    state = torch.load(paths["m9_weights"], map_location="cpu",
                       weights_only=True, mmap=True)
    if type(state) not in (dict, OrderedDict):
        raise Refusal("M10 requires a native, model-only M9 checkpoint")
    model.load_state_dict(state, strict=True)
    del state
    model.to(device=device, dtype=torch.float32)
    model.gradient_checkpointing_enable()
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=5e-5, weight_decay=0.0, eps=1e-8,
        foreach=False, fused=False,
    )
    start = time.monotonic()
    completed = []
    updates = []
    previous: dict[str, Any] | None = None
    early_stopped = False
    for item in schedule:
        step = item["optimizerUpdate"]
        if step > copy.MAX_UPDATES or time.monotonic() - start > HARD_TIMEOUT:
            raise Refusal("M10 update/time budget exceeded")
        sample = packed[item["recordIds"][0]]
        inputs = torch.tensor([sample["inputIds"]], dtype=torch.long, device=device)
        targets = torch.tensor([sample["targetIds"]], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Fixed probe of one actual learned embedding row; changes are checked
        # after every update, not merely assumed from changing loss values.
        row_index = sample["inputIds"][-1]
        before_slice = model.token_embedding.weight[row_index].detach().clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(inputs, targets)
        if loss is None or not bool(torch.isfinite(loss)):
            raise Refusal("M10 answer/EOS loss is not finite")
        before_loss = float(loss.item())
        loss.backward()
        gradient_squared = 0.0
        for parameter in parameters:
            grad = parameter.grad
            if grad is None or not bool(torch.isfinite(grad).all()):
                raise Refusal("M10 gradient missing or nonfinite")
            gradient_squared += float(grad.float().square().sum().item())
        gradient_norm = math.sqrt(gradient_squared)
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise Refusal("M10 gradients did not reach model")
        optimizer.step()
        torch.cuda.synchronize(device)
        delta = float((model.token_embedding.weight[row_index].detach()
                       - before_slice).float().norm().item())
        if not math.isfinite(delta) or delta <= 0:
            raise Refusal("M10 optimizer did not change the measured embedding row")
        updates.append({
            "step": step, "recordId": item["recordIds"][0], "phase": item["phase"],
            "loss": before_loss, "gradientL2Norm": gradient_norm,
            "measuredEmbeddingRowDeltaL2": delta,
            "supervisedAnswerPositions": sample["supervisedAnswerPositions"],
            "promptPositionsIncorrectlySupervised": 0,
            "eosTargetSupervised": sample["targetIds"][-1] == 3,
        })
        del inputs, targets, before_slice, loss
        if step not in SNAPSHOTS:
            continue
        # Refuse lack of storage rather than truncating a checkpoint.
        _fit_check(stage)
        snapshot = stage / f"step-{step}"
        snapshot.mkdir(mode=0o700)
        weights = snapshot / "weights.pt"
        evidence = _reload_parity(model, torch, weights, probe_ids, config)
        model.eval()
        scores, response_raw = _measure_generation(
            model, torch, tokenizer, all_rows, device, evidence["sha256"])
        diagnostics = teacher_forced_probe(
            model, torch, packed[PROBE_ID], device)
        warmup_delta = warmup_comparison(baseline_responses, scores)
        with (snapshot / "responses.jsonl").open("xb") as stream:
            stream.write(response_raw)
        score_proof = {
            "optimizerUpdatesCompleted": step,
            "checkpointSha256": evidence["sha256"],
            "candidateSha256": admitted["candidateSha256"],
            "verifiedGeneration": True,
            "decodingPolicies": list(DECODERS),
            "scores": scores,
        }
        record = {
            **score_proof,
            "checkpoint": evidence,
            "warmupBeforeAfter": warmup_delta,
            "warmupTeacherForcedDiagnostics": diagnostics,
            "responsesSha256": hashlib.sha256(response_raw).hexdigest(),
            "latestGradientL2Norm": gradient_norm,
            "latestEmbeddingRowDeltaL2": delta,
            "peakCudaReservedBytes": torch.cuda.max_memory_reserved(device),
            "finalHoldoutUsed": False,
        }
        with (snapshot / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        completed.append(record)
        if checked_early_stop(previous, score_proof):
            early_stopped = True
            break
        previous = score_proof
    if not early_stopped and len(updates) != 50:
        raise Refusal("M10 did not reach the 50-update ceiling")
    if [x["optimizerUpdatesCompleted"] for x in completed] not in (
        [10, 25], [10, 25, 50]):
        raise Refusal("M10 skipped a required checkpoint")
    result = {
        "protocol": PROTOCOL,
        "status": "stopped-after-two-perfect-milestones" if early_stopped
                  else "bounded-50-updates-complete-unreviewed",
        "trainingAuthorized": True,
        "optimizerUpdatesCompleted": len(updates),
        "maxOptimizerUpdates": copy.MAX_UPDATES,
        "originalM9WeightsSha256": admitted["m9WeightsSha256"],
        "originalM9ReportSha256": admitted["m9ReportSha256"],
        "baselineReportSha256": admitted["baselineReportSha256"],
        "baselineResponsesSha256": admitted["baselineResponsesSha256"],
        "candidateSha256": admitted["candidateSha256"],
        "scheduleSha256": admitted["scheduleSha256"],
        "approvalSha256": admitted["externalApprovalSha256"],
        "identity": identity,
        "snapshots": completed,
        "steps": updates,
        "peakCudaReservedBytes": torch.cuda.max_memory_reserved(device),
        "elapsedSeconds": round(time.monotonic() - start, 3),
        "finalHoldoutUsed": False,
        "historicalCheckpointWrites": False,
        "optimizerStateSaved": False,
        "autonomousCodeEditingImprovementProven": False,
    }
    return result


def preserve_verified_partial(
    stage: Path, target: Path, proof: dict[str, Any], *,
    failure_reason: str,
) -> bool:
    """On failure, preserve only complete SHA-checked milestones as INCOMPLETE.

    A hard-killed run cannot claim the final optimizer update count or
    completion. This protects the scientifically useful step-10 diagnostic
    without accepting a half-written weight file as a valid checkpoint.
    """
    if failure_reason not in {"hard-timeout", "worker-error"}:
        raise Refusal("M10 unknown failure classification")
    if target.exists() or target.is_symlink():
        raise Refusal("M10 cannot overwrite an occupied partial-output path")
    checks: list[dict[str, Any]] = []
    for step in SNAPSHOTS:
        folder = stage / f"step-{step}"
        weights = folder / "weights.pt"
        response = folder / "responses.jsonl"
        report_path = folder / "report.json"
        if not all(path.is_file() and not path.is_symlink()
                   for path in (weights, response, report_path)):
            break
        if (report_path.stat().st_size > 1024**2
                or response.stat().st_size > gate.RESPONSES_MAX_BYTES):
            break
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            checkpoint = report.get("checkpoint")
            digest = _sha(weights)
            if (report.get("optimizerUpdatesCompleted") != step
                    or report.get("candidateSha256") != proof["candidateSha256"]
                    or type(checkpoint) is not dict
                    or checkpoint.get("sha256") != digest
                    or checkpoint.get("bytes") != weights.stat().st_size
                    or report.get("checkpointSha256") != digest
                    or report.get("verifiedGeneration") is not True
                    or report.get("decodingPolicies") != list(DECODERS)
                    or report.get("responsesSha256") != research.digest_file(
                        response, limit=gate.RESPONSES_MAX_BYTES)):
                break
            verify_checkpoint_responses(response, digest, report["scores"])
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            break
        checks.append({
            "optimizerUpdatesCompletedThroughThisMilestone": step,
            "checkpointSha256": digest,
            "responseSha256": report["responsesSha256"],
        })
    if not checks:
        return False
    # This staging directory belongs exclusively to the failed worker. Do
    # not publish any unfinished future snapshot or unverified extra files.
    verified_names = {
        f"step-{item['optimizerUpdatesCompletedThroughThisMilestone']}"
        for item in checks
    }
    for item in stage.iterdir():
        if item.name in verified_names:
            continue
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()
    interrupted = {
        "protocol": PROTOCOL,
        "status": "interrupted-partial-unreviewed-not-training-success",
        "failureReason": failure_reason,
        "trainingMayHaveContinuedBeyondLastVerifiedMilestone": True,
        "lastVerifiedMilestone": checks[-1]["optimizerUpdatesCompletedThroughThisMilestone"],
        "verifiedPartialSnapshots": checks,
        "originalM9WeightsSha256": proof["m9WeightsSha256"],
        "candidateSha256": proof["candidateSha256"],
        "approvalSha256": proof["externalApprovalSha256"],
        "trainingCompleted": False,
        "resumeAuthorized": False,
        "optimizerStateSaved": False,
        "finalHoldoutUsed": False,
        "historicalCheckpointWrites": False,
    }
    with (stage / "interruption.json").open("x", encoding="utf-8") as stream:
        json.dump(interrupted, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    if target.exists():
        raise Refusal("M10 output appeared during interrupted evidence publication")
    os.rename(stage, target)
    return True


def run_approved(paths: dict[str, Path], *, bundle_sha: str,
                 explicit_flag: bool) -> dict[str, Any]:
    if explicit_flag is not True:
        raise Refusal("M10 requires explicit opt-in to execute an approved run")
    proof = _admit(paths, bundle_sha)
    identity = _source_identity()
    target = m8.require_output(
        paths["output_dir"], bundle=paths["bundle_dir"],
        checkpoint=paths["m9_weights"])
    stage: Path | None = Path(tempfile.mkdtemp(
        prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        nonce = secrets.token_hex(32)
        env = dict(os.environ, **{CONTROL: nonce, "PYTHONDONTWRITEBYTECODE": "1"})
        payload = {
            "nonce": nonce,
            "bundle_sha": bundle_sha,
            "stage": str(stage.resolve(strict=True)),
            "paths": {name: str(path.resolve(strict=True) if name != "output_dir"
                                else path.resolve(strict=False))
                      for name, path in paths.items()},
        }
        rc, out, err, timed_out = hw._supervise(
            [sys.executable, "-m", "plex_training.plex600_m10_worker",
             "--_worker"],
            payload, env, timeout=HARD_TIMEOUT)
        if timed_out or rc != 0:
            reason = "hard-timeout" if timed_out else "worker-error"
            preserved = preserve_verified_partial(
                stage, target, proof, failure_reason=reason)
            if preserved:
                stage = None
                raise Refusal(
                    f"M10 {reason}: verified PARTIAL evidence stored at {target}; "
                    "training NOT complete, no optimizer-state resume")
            raise Refusal("M10 hard-timeout or worker failure; no completed result published")
        outcome = json.loads(out)
        if (outcome.get("protocol") != PROTOCOL
                or outcome.get("identity") != identity
                or outcome.get("originalM9WeightsSha256") != proof["m9WeightsSha256"]
                or outcome.get("candidateSha256") != proof["candidateSha256"]
                or outcome.get("approvalSha256") != proof["externalApprovalSha256"]
                or outcome.get("scheduleSha256") != proof["scheduleSha256"]
                or outcome.get("finalHoldoutUsed") is not False
                or outcome.get("historicalCheckpointWrites") is not False
                or outcome.get("optimizerStateSaved") is not False
                or outcome.get("status") not in (
                    "bounded-50-updates-complete-unreviewed",
                    "stopped-after-two-perfect-milestones",
                )
                or outcome.get("optimizerUpdatesCompleted") not in (25, 50)):
            raise Refusal("M10 child outcome has invalid source, budget or owner scope")
        expected_steps = ([10, 25] if outcome["optimizerUpdatesCompleted"] == 25
                          else [10, 25, 50])
        snapshots = outcome.get("snapshots")
        if (type(snapshots) is not list
                or [s["optimizerUpdatesCompleted"] for s in snapshots]
                   != expected_steps):
            raise Refusal("M10 failed exact checkpoint progression")
        schedule = exact_schedule()
        steps = outcome.get("steps")
        if (type(steps) is not list
                or len(steps) != outcome["optimizerUpdatesCompleted"]
                or any(type(row) is not dict
                       or row.get("step") != expected["optimizerUpdate"]
                       or row.get("recordId") != expected["recordIds"][0]
                       or row.get("eosTargetSupervised") is not True
                       or row.get("promptPositionsIncorrectlySupervised") != 0
                       for row, expected in zip(steps, schedule))):
            raise Refusal("M10 child update ledger diverged from exact approved schedule")
        for item in snapshots:
            folder = stage / f"step-{item['optimizerUpdatesCompleted']}"
            checkpoint = folder / "weights.pt"
            responses = folder / "responses.jsonl"
            checkpoint_sha = _sha(checkpoint)
            response_sha = research.digest_file(
                responses, limit=gate.RESPONSES_MAX_BYTES)
            with (folder / "report.json").open("r", encoding="utf-8") as stream:
                snapshot_record = json.load(stream)
            if (checkpoint_sha != item["checkpointSha256"]
                    or checkpoint.stat().st_size != item["checkpoint"]["bytes"]
                    or checkpoint_sha != item["checkpoint"]["sha256"]
                    or response_sha != item["responsesSha256"]
                    or snapshot_record != item):
                raise Refusal("M10 saved checkpoint/response/scores differ from child receipts")
            verify_checkpoint_responses(responses, checkpoint_sha, item["scores"])
        with (stage / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(outcome, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        if target.exists():
            raise Refusal("M10 final target appeared during execution")
        os.rename(stage, target)
        stage = None
        return outcome
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def _worker_entry() -> None:
    payload = json.loads(sys.stdin.buffer.read(64 * 1024))
    nonce = os.environ.pop(CONTROL, "")
    if not nonce or payload.get("nonce") != nonce or set(payload) != {
        "nonce", "bundle_sha", "stage", "paths",
    } or type(payload.get("paths")) is not dict:
        raise Refusal("M10 worker can only be started by the approved parent process")
    path_fields = {"bundle_dir", "m9_report", "m9_weights",
                   "baseline_report", "baseline_responses",
                   "approval_file", "output_dir"}
    if set(payload["paths"]) != path_fields:
        raise Refusal("M10 worker inputs differ from exact approval contract")
    result = _train_worker(
        {name: Path(value) for name, value in payload["paths"].items()},
        payload["bundle_sha"], stage=Path(payload["stage"]))
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-approved-run", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
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
        if args._worker:
            if args.execute_approved_run or any(
                getattr(args, name) is not None for name in (
                    "bundle_dir", "bundle_manifest_sha256", "m9_report",
                    "m9_weights", "baseline_report", "baseline_responses",
                    "approval_file", "output_dir",
                )
            ):
                raise Refusal("M10 private worker cannot accept public CLI flags")
            _worker_entry()
            return 0
        if args.execute_approved_run:
            paths = _paths_from_args(args)
            if args.bundle_manifest_sha256 is None:
                raise Refusal("M10 training requires pinned original M7 manifest")
            result = run_approved(
                paths, bundle_sha=args.bundle_manifest_sha256, explicit_flag=True)
        elif any(getattr(args, name) is not None for name in (
            "bundle_dir", "bundle_manifest_sha256", "m9_report",
            "m9_weights", "baseline_report", "baseline_responses",
            "approval_file", "output_dir",
        )):
            raise Refusal("M10 paths require --execute-approved-run")
        else:
            result = plan()
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        print("M10 worker refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
