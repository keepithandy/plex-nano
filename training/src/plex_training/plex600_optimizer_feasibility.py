"""Plex 600M M6: one opt-in synthetic optimizer update on RTX 4080 SUPER.

The default command prints a plan. No dataset, tokenizer, holdout, checkpoint,
or model weights are loaded from disk. A worker runs the bounded GPU experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import time
from typing import Any

from . import plex600_feasibility as prior

PROTOCOL = "plex-600m-optimizer-feasibility-v1"
ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = Path(__file__).resolve()
TIMEOUT_SECONDS = 600
SEED = 17
SEQUENCE = 1024
LEARNING_RATE = 0.0001
STATE_TENSORS = prior.PARAMETER_TENSORS * 2
CONTROL_ENV = "PLEX600_OPTIMIZER_CONTROL"


def plan(device_index: int = 0) -> dict[str, Any]:
    if type(device_index) is not int or device_index < 0:
        raise prior.Refusal("Device index must be a nonnegative integer")
    return {
        "protocol": PROTOCOL, "status": "plan", "deviceIndex": device_index,
        "modelSha256Accepted": list(prior.ACCEPTED_MODEL_SHA256),
        "parameterCount": prior.PARAMETERS, "parameterTensors": prior.PARAMETER_TENSORS,
        "configuration": dict(prior.CONFIGURATION), "seed": SEED,
        "batchSize": 1, "sequenceLength": SEQUENCE, "supervisedTargets": SEQUENCE - 1,
        "parameterDtype": "float32", "autocastDtype": "bfloat16",
        "gradientCheckpointing": True, "optimizer": "AdamW",
        "optimizerConfiguration": {"learningRate": LEARNING_RATE, "weightDecay": 0.0,
                                   "eps": 1e-8, "foreach": False, "fused": False},
        "updatesPlanned": 1, "minimumFreeBytes": prior.MIN_FREE_BYTES,
        "allocatorFraction": prior.MEMORY_FRACTION, "timeoutSeconds": TIMEOUT_SECONDS,
        "input": "synthetic-sequential-token-ids-only", "checkpointWrites": False,
        "datasetAccess": False, "finalHoldoutAccess": False,
    }


def source_identity() -> dict[str, str | bool]:
    identity = prior.source_identity()  # Pins original model bytes and clean source.
    relative = RUNNER_PATH.relative_to(ROOT).as_posix()
    options = dict(cwd=ROOT, capture_output=True, text=True, check=True, timeout=5)
    state = subprocess.run(["git", "status", "--porcelain", "--", relative], **options)
    if state.stdout.strip():
        raise prior.Refusal("Optimizer runner has uncommitted changes")
    return {**identity, "optimizerRunnerSha256": hashlib.sha256(RUNNER_PATH.read_bytes()).hexdigest()}


def check_optimizer_state(optimizer: Any, parameters: list[Any], torch: Any) -> dict[str, Any]:
    """Inspect every FP32 AdamW moment and every updated parameter after one step."""
    state_tensors = 0
    for parameter in parameters:
        if parameter.dtype != torch.float32 or not bool(torch.isfinite(parameter).all()):
            raise RuntimeError("Parameter is not finite FP32 after update")
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError("Missing or unexpected AdamW state")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = state[name]
            if tensor.shape != parameter.shape or tensor.dtype != torch.float32:
                raise RuntimeError("AdamW moment shape/dtype mismatch")
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError("AdamW state is not finite")
            state_tensors += 1
        step = state["step"]
        if not bool(torch.isfinite(step)) or float(step.item()) != 1.0:
            raise RuntimeError("AdamW update count must be exactly one")
    return {"stateTensorsVerified": state_tensors, "stateFinite": True,
            "allParametersFinite": True}


def workload(torch: Any, model_module: Any, report: dict[str, Any]) -> None:
    """Only called in child after source and GPU guards; at most one update."""
    config = model_module.Plex600Config()
    if config.to_dict() != prior.CONFIGURATION or config.parameter_count() != prior.PARAMETERS:
        raise prior.Refusal("Model architecture differs from fixed proposal")
    device_index = report["deviceIndex"]
    report["environment"] = {"python": sys.version, "platform": platform.platform(),
                             "torch": str(torch.__version__), "cudaRuntime": str(torch.version.cuda)}
    report["gpu"] = prior._check_device(torch, device_index)
    device = torch.device("cuda", device_index)
    torch.cuda.set_per_process_memory_fraction(prior.MEMORY_FRACTION, device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(SEED)
    report["phase"] = "initialization"
    start = time.perf_counter()
    with torch.device("cpu"):
        model = model_module.Plex600Model(config)
    if model_module.parameter_count(model) != prior.PARAMETERS:
        raise RuntimeError("Actual model parameter count mismatch")
    free, _ = torch.cuda.mem_get_info(device_index)
    report["gpu"]["freeBytesBeforeAllocation"] = free
    if free < prior.MIN_FREE_BYTES:
        raise prior.Refusal("Free GPU memory dropped below 14,000 MiB")
    model.to(device=device, dtype=torch.float32)
    model.train()
    model.gradient_checkpointing_enable()
    parameters = list(model.parameters())
    if len(parameters) != prior.PARAMETER_TENSORS:
        raise RuntimeError("Unexpected parameter tensor count")
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, eps=1e-8,
                                  weight_decay=0.0, foreach=False, fused=False)
    torch.cuda.synchronize(device)
    report["initializationSeconds"] = time.perf_counter() - start

    # A single deterministic next-token workload, no real corpus.
    inputs = torch.arange(SEQUENCE, device=device, dtype=torch.long).unsqueeze(0)
    targets = inputs + 1
    targets[:, -1] = -100
    report["phase"] = "forward"
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(inputs, targets)
    torch.cuda.synchronize(device)
    report["forwardSeconds"] = time.perf_counter() - start
    if (tuple(logits.shape) != (1, SEQUENCE, config.vocab_size)
            or loss is None or loss.ndim != 0 or not bool(torch.isfinite(loss))):
        raise RuntimeError("Invalid forward result or non-finite loss")
    report["lossBeforeUpdate"] = float(loss.item())

    report["phase"] = "backward"
    start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    report["backwardSeconds"] = time.perf_counter() - start
    for parameter in parameters:
        if (parameter.grad is None or parameter.grad.dtype != torch.float32
                or not bool(torch.isfinite(parameter.grad).all())):
            raise RuntimeError("Missing or non-finite FP32 gradient")
    report["gradientsFinite"] = True
    # Inspect an actual used input token without cloning the full 2.4 GiB model.
    before = model.token_embedding.weight[0, :64].detach().clone()

    report["phase"] = "optimizer-update"
    start = time.perf_counter()
    report["optimizerUpdatesCompleted"] = None  # Unknown if step fails partway.
    optimizer.step()  # Exactly one update; no loop or checkpoint.
    torch.cuda.synchronize(device)
    report["optimizerUpdatesCompleted"] = 1
    report["optimizerSeconds"] = time.perf_counter() - start
    changed = bool((model.token_embedding.weight[0, :64].detach() != before).any())
    if not changed:
        raise RuntimeError("Sampled trained parameter slice did not change")
    report["parameterSliceChanged"] = changed
    report["phase"] = "optimizer-state-verification"
    report.update(check_optimizer_state(optimizer, parameters, torch))
    if report["stateTensorsVerified"] != STATE_TENSORS:
        raise RuntimeError("Unexpected AdamW state tensor count")
    torch.cuda.synchronize(device)
    report["peakAllocatedBytes"] = torch.cuda.max_memory_allocated(device)
    report["peakReservedBytes"] = torch.cuda.max_memory_reserved(device)
    report["status"] = "pass"
    report["phase"] = "complete"


def worker_entry() -> None:
    started = time.perf_counter()
    report: dict[str, Any] = {"protocol": PROTOCOL, "status": "refused"}
    torch_module = None
    try:
        # Remove the secret from the environment before importing heavy libraries.
        token = os.environ.pop(CONTROL_ENV, "")
        control = json.loads(sys.stdin.read(4097))
        if (not token or not isinstance(control, dict)
                or set(control) != {"nonce", "deviceIndex"}
                or control.get("nonce") != token):
            raise prior.Refusal("Optimizer worker must be launched through opt-in parent")
        report = plan(control["deviceIndex"])
        report["status"] = "refused"
        report["optimizerUpdatesCompleted"] = 0
        report["source"] = source_identity()
        import torch
        from . import plex600

        torch_module = torch
        workload(torch, plex600, report)
    except prior.Refusal as exc:
        report.update(status="refused", error=str(exc))
    except Exception as exc:
        report.update(status="failed", errorType=type(exc).__name__, error=str(exc))
    if torch_module is not None and "gpu" in report:
        try:
            report["peakAllocatedBytes"] = torch_module.cuda.max_memory_allocated(report["deviceIndex"])
            report["peakReservedBytes"] = torch_module.cuda.max_memory_reserved(report["deviceIndex"])
        except Exception:
            report["peakMeasurementUnavailable"] = True
    report["workerSeconds"] = time.perf_counter() - started
    print(json.dumps(report, allow_nan=False), flush=True)


def trusted_result(result: Any, device: int, identity: dict, *, require_pass: bool) -> bool:
    if not isinstance(result, dict) or result.get("protocol") != PROTOCOL:
        return False
    if result.get("status") not in ("pass", "failed", "refused"):
        return False
    expected = plan(device)
    for key, value in expected.items():
        if key != "status" and (key not in result or type(result[key]) != type(value)
                                or result[key] != value):
            return False
    if result.get("source") != identity:
        return False
    if not require_pass or result["status"] != "pass":
        return True
    if (result.get("gradientsFinite") is not True or
            result.get("stateFinite") is not True or
            result.get("allParametersFinite") is not True or
            result.get("parameterSliceChanged") is not True or
            result.get("stateTensorsVerified") != STATE_TENSORS or
            result.get("optimizerUpdatesCompleted") != 1):
        return False
    gpu = result.get("gpu")
    if (not isinstance(gpu, dict) or
            gpu.get("deviceName") != "NVIDIA GeForce RTX 4080 SUPER" or
            not isinstance(gpu.get("driverVersion"), str) or
            not gpu["driverVersion"] or
            not all(type(gpu.get(k)) is int and gpu[k] >= prior.MIN_FREE_BYTES
                    for k in ("totalBytes", "freeBytesAtPreflight", "freeBytesBeforeAllocation"))):
        return False
    for field in ("lossBeforeUpdate", "forwardSeconds", "backwardSeconds",
                  "optimizerSeconds", "initializationSeconds", "workerSeconds",
                  "peakAllocatedBytes", "peakReservedBytes"):
        v = result.get(field)
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            return False
    if result["workerSeconds"] > TIMEOUT_SECONDS:
        return False
    return True


def execute(device: int, *, authorized: bool = False) -> dict[str, Any]:
    report = plan(device)
    report["status"] = "refused"
    if authorized is not True:
        report["error"] = "Explicit --execute-optimizer-check flag required"
        return report
    try:
        identity = source_identity()
        report["source"] = identity
        nonce = secrets.token_hex(32)
        env = dict(os.environ, **{CONTROL_ENV: nonce, "PYTHONDONTWRITEBYTECODE": "1"})
        command = [sys.executable, "-m", "plex_training.plex600_optimizer_feasibility", "--_worker"]
        code, stdout, stderr, timed_out = prior._supervise(
            command, {"nonce": nonce, "deviceIndex": device}, env, timeout=TIMEOUT_SECONDS
        )
        report.update(workerExitCode=code, workerStderr=stderr[-8192:])
        if timed_out:
            report.update(status="timeout", error="Optimizer worker exceeded 600 seconds")
        elif code != 0:
            report.update(status="inconclusive", error="Worker exited without validated report")
        else:
            def refuse_constant(value: str) -> None:
                raise ValueError("Invalid non-finite JSON: " + value)
            result = json.loads(stdout, parse_constant=refuse_constant)
            if not trusted_result(result, device, identity, require_pass=True):
                report.update(status="inconclusive", error="Untrusted optimizer result")
            else:
                report.update(result)
    except prior.Refusal as exc:
        report.update(status="refused", error=str(exc))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report.update(status="inconclusive", errorType=type(exc).__name__, error=str(exc))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--execute-optimizer-check", action="store_true",
                        help="Explicit opt-in: one GPU AdamW update on synthetic input")
    parser.add_argument("--output", type=Path, help="Create-only JSON report file")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        worker_entry()
        return 0
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        parser.error("Refusing to overwrite an existing optimizer report")
    try:
        result = execute(args.device, authorized=True) if args.execute_optimizer_check else plan(args.device)
    except prior.Refusal as exc:
        parser.error(str(exc))
    raw = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(raw)
    print(raw, end="")
    return 0 if result["status"] in ("plan", "pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
