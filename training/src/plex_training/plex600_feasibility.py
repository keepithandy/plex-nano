"""One opt-in, synthetic CUDA forward/backward check, with no optimizer.

The default CLI only prints the plan. Importing this module does not import
PyTorch, initialize CUDA, construct a model, or write an artifact.
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

PROTOCOL = "plex-600m-cuda-feasibility-v1"
MODEL_SHA256 = "e7ad49628b1ebcf083991c385a9c55cff159c5567a6a0f6062d48418f26d1af2"
ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = Path(__file__).with_name("plex600.py")
TIMEOUT_SECONDS = 600
MIN_FREE_BYTES = 14_000 * 1024**2
MEMORY_FRACTION = 0.80
PARAMETERS = 600_372_480
PARAMETER_TENSORS = 218  # 24 * (2 norms + 4 attention + 3 FFN) + embedding + final norm.
TOKEN_COUNT = 1024
SEED = 17
_CONTROL_ENV = "PLEX600_FEASIBILITY_CONTROL"
CONFIGURATION = {
    "vocab_size": 32768, "context_length": 1024, "width": 1280, "layers": 24,
    "attention_heads": 20, "key_value_heads": 5, "feed_forward_width": 4992,
    "rope_theta": 10000.0, "norm_epsilon": 1e-5, "dropout": 0.0,
}


class Refusal(ValueError):
    """A prerequisite failed before constructing the full model."""


def plan(device_index: int = 0) -> dict[str, Any]:
    if type(device_index) is not int or device_index < 0:
        raise Refusal("Device index must be a nonnegative integer")
    return {
        "protocol": PROTOCOL, "status": "plan", "deviceIndex": device_index,
        "modelSha256": MODEL_SHA256, "configuration": dict(CONFIGURATION),
        "parameterCount": PARAMETERS, "seed": SEED, "batchSize": 1,
        "sequenceLength": TOKEN_COUNT, "supervisedTargets": TOKEN_COUNT - 1,
        "parameterDtype": "float32", "autocastDtype": "bfloat16",
        "gradientCheckpointing": True, "forwardCalls": 1, "backwardCalls": 1,
        "minimumFreeBytes": MIN_FREE_BYTES, "allocatorFraction": MEMORY_FRACTION,
        "timeoutSeconds": TIMEOUT_SECONDS, "optimizerUpdates": 0,
        "artifactWrites": False,
    }


def source_identity() -> dict[str, Any]:
    """Verify on-disk source and Git state without importing the model."""
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise Refusal("Model source SHA-256 does not match the fixed proposal")
    relative = MODEL_PATH.relative_to(ROOT).as_posix()
    options = dict(cwd=ROOT, text=True, capture_output=True, check=True, timeout=5)
    state = subprocess.run(["git", "status", "--porcelain", "--", relative], **options)
    if state.stdout.strip():
        raise Refusal("Model source has local or staged modifications")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], **options).stdout.strip()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise Refusal("Cannot identify the current Git commit")
    return {
        "gitCommit": commit, "modelSourceClean": True, "modelSha256": digest,
        "runnerSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _check_device(torch: Any, device_index: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise Refusal("CUDA is unavailable")
    if device_index >= torch.cuda.device_count():
        raise Refusal("Selected CUDA device does not exist")
    torch.cuda.set_device(device_index)
    name = torch.cuda.get_device_name(device_index)
    if name != "NVIDIA GeForce RTX 4080 SUPER":
        raise Refusal("Selected device is not the proposed NVIDIA GeForce RTX 4080 SUPER")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise Refusal("Native CUDA BF16 support is required")
    free, total = torch.cuda.mem_get_info(device_index)
    if free < MIN_FREE_BYTES:
        raise Refusal("Less than 14,000 MiB GPU memory is free")
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True, capture_output=True, check=True, timeout=5,
    ).stdout.strip().splitlines()
    if not driver or not driver[0].strip():
        raise Refusal("Cannot identify NVIDIA driver")
    return {
        "deviceName": name, "driverVersion": driver[0].strip(),
        "totalBytes": total, "freeBytesAtPreflight": free,
        "cudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _workload(torch: Any, model_module: Any, report: dict[str, Any]) -> None:
    """Called only inside the authorized worker after source verification."""
    device_index = report["deviceIndex"]
    config = model_module.Plex600Config()
    if config.to_dict() != CONFIGURATION or config.parameter_count() != PARAMETERS:
        raise Refusal("Model configuration differs from the fixed proposal")
    report["environment"] = {
        "python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "cudaRuntime": torch.version.cuda,
    }
    report["gpu"] = _check_device(torch, device_index)
    device = torch.device("cuda", device_index)
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device)
    torch.cuda.reset_peak_memory_stats(device)
    report["phase"] = "initialization"
    started = time.perf_counter()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(SEED)
    # CPU scratch initialization avoids a temporary full-size GPU copy.
    with torch.device("cpu"):
        model = model_module.Plex600Model(config)
    if model_module.parameter_count(model) != PARAMETERS:
        raise RuntimeError("Actual model parameter count differs from the proposal")
    # Recheck after CPU initialization, immediately before the GPU allocation.
    free, _ = torch.cuda.mem_get_info(device_index)
    report["gpu"]["freeBytesBeforeAllocation"] = free
    if free < MIN_FREE_BYTES:
        # The model already exists on CPU; distinguish this from a pre-model refusal.
        raise RuntimeError("Free GPU memory dropped below 14,000 MiB before transfer")
    model.to(device=device, dtype=torch.float32)
    model.train()
    model.gradient_checkpointing_enable()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Model parameters must remain FP32")
    torch.cuda.synchronize(device)
    report["initializationSeconds"] = time.perf_counter() - started
    report["actualParameterCount"] = model_module.parameter_count(model)
    inputs = torch.arange(TOKEN_COUNT, device=device, dtype=torch.long).unsqueeze(0)
    targets = inputs + 1
    targets[:, -1] = -100
    report["phase"] = "forward"
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(inputs, targets)
    torch.cuda.synchronize(device)
    report["forwardSeconds"] = time.perf_counter() - started
    if tuple(logits.shape) != (1, TOKEN_COUNT, CONFIGURATION["vocab_size"]):
        raise RuntimeError("Unexpected output shape")
    if loss is None or loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise RuntimeError("Loss is absent or non-finite")
    report["loss"] = loss.item()
    report["outputShape"] = list(logits.shape)
    report["phase"] = "backward"
    started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    report["backwardSeconds"] = time.perf_counter() - started
    report["phase"] = "gradient-validation"
    tensors = 0
    for parameter in model.parameters():
        if parameter.grad is None or parameter.grad.dtype != torch.float32:
            raise RuntimeError("A parameter has no FP32 gradient")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("A parameter gradient is non-finite")
        tensors += 1
    if tensors != PARAMETER_TENSORS:
        raise RuntimeError("Unexpected number of trainable parameter tensors")
    torch.cuda.synchronize(device)
    report["gradientTensorCount"] = tensors
    report["allGradientsPresentFinite"] = True
    report["peakAllocatedBytes"] = torch.cuda.max_memory_allocated(device)
    report["peakReservedBytes"] = torch.cuda.max_memory_reserved(device)
    report["status"] = "pass"
    report["phase"] = "complete"


def _worker_entry() -> None:
    """Internal subprocess entry; require the parent handshake before imports."""
    report: dict[str, Any] = {"protocol": PROTOCOL, "status": "refused"}
    started = time.perf_counter()
    torch_module = None
    try:
        control = json.loads(sys.stdin.read(4097))
        nonce = os.environ.pop(_CONTROL_ENV, "")
        if (not nonce or not isinstance(control, dict)
                or control.get("nonce") != nonce or set(control) != {"nonce", "deviceIndex"}):
            raise Refusal("Worker must be launched by the opt-in parent command")
        report = plan(control["deviceIndex"])
        report["status"] = "refused"
        report["source"] = source_identity()
        # Lazy imports: neither plan nor an unauthenticated worker touches CUDA.
        import torch
        from . import plex600

        torch_module = torch
        _workload(torch, plex600, report)
    except Refusal as error:
        report.update(status="refused", error=str(error))
    except Exception as error:
        report.update(status="failed", errorType=type(error).__name__, error=str(error))
    # Retain available peaks even for OOM/non-finite failures. Do not mask the
    # original error if a damaged CUDA context cannot report these statistics.
    if torch_module is not None and "gpu" in report:
        try:
            report["peakAllocatedBytes"] = torch_module.cuda.max_memory_allocated(report["deviceIndex"])
            report["peakReservedBytes"] = torch_module.cuda.max_memory_reserved(report["deviceIndex"])
        except Exception:
            report["memoryMeasurementUnavailable"] = True
    report["workerSeconds"] = time.perf_counter() - started
    print(json.dumps(report, allow_nan=False), flush=True)


def _supervise(command: list[str], control: dict[str, Any], env: dict[str, str],
               timeout: float = TIMEOUT_SECONDS) -> tuple[int, str, str, bool]:
    """Kill and reap the single worker on timeout or parent interruption."""
    worker = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, cwd=ROOT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        stdout, stderr = worker.communicate(json.dumps(control), timeout=timeout)
        return worker.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        worker.kill()
        stdout, stderr = worker.communicate()
        return worker.returncode, stdout, stderr, True
    except BaseException:
        worker.kill()
        worker.communicate()
        raise


def _valid_result(result: Any, device_index: int, identity: dict[str, Any]) -> bool:
    if not isinstance(result, dict) or result.get("protocol") != PROTOCOL:
        return False
    if result.get("status") not in ("pass", "failed", "refused"):
        return False
    for key, value in plan(device_index).items():
        if key != "status" and result.get(key) != value:
            return False
    if result.get("source") != identity:
        return False
    if result["status"] == "pass":
        if (result.get("actualParameterCount") != PARAMETERS
                or result.get("outputShape") != [1, TOKEN_COUNT, CONFIGURATION["vocab_size"]]
                or result.get("allGradientsPresentFinite") is not True
                or type(result.get("gradientTensorCount")) is not int
                or result["gradientTensorCount"] != PARAMETER_TENSORS):
            return False
        gpu, environment = result.get("gpu"), result.get("environment")
        if (not isinstance(gpu, dict) or not isinstance(environment, dict)
                or gpu.get("deviceName") != "NVIDIA GeForce RTX 4080 SUPER"
                or not all(isinstance(environment.get(key), str) and environment[key]
                           for key in ("python", "platform", "torch", "cudaRuntime"))
                or not isinstance(gpu.get("driverVersion"), str) or not gpu["driverVersion"]):
            return False
        for key in ("freeBytesAtPreflight", "freeBytesBeforeAllocation", "totalBytes"):
            if type(gpu.get(key)) is not int or gpu[key] < MIN_FREE_BYTES:
                return False
        for key in ("loss", "initializationSeconds", "forwardSeconds", "backwardSeconds",
                    "workerSeconds", "peakAllocatedBytes", "peakReservedBytes"):
            value = result.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                return False
        if result["workerSeconds"] > TIMEOUT_SECONDS:
            return False
    return True


def execute(device_index: int, *, authorized: bool = False) -> dict[str, Any]:
    report = plan(device_index)
    report["status"] = "refused"
    if authorized is not True:
        report["error"] = "Explicit --execute-cuda-check is required"
        return report
    try:
        identity = source_identity()
        report["source"] = identity
        nonce = secrets.token_hex(32)
        env = dict(os.environ, **{_CONTROL_ENV: nonce, "PYTHONDONTWRITEBYTECODE": "1"})
        command = [sys.executable, "-m", "plex_training.plex600_feasibility", "--_worker"]
        code, stdout, stderr, timed_out = _supervise(
            command, {"nonce": nonce, "deviceIndex": device_index}, env,
        )
        report["workerExitCode"] = code
        report["workerStderr"] = stderr[-8192:]
        if timed_out:
            report.update(status="timeout", error="Worker exceeded the 600-second limit")
        elif code != 0:
            report.update(status="inconclusive", error="Worker exited without a trusted result")
        else:
            # Python otherwise accepts non-standard JSON NaN/Infinity constants.
            def reject_constant(value: str) -> None:
                raise ValueError("Non-finite JSON constant in worker result: " + value)

            result = json.loads(stdout, parse_constant=reject_constant)
            if not _valid_result(result, device_index, identity):
                report.update(status="inconclusive", error="Worker result failed provenance or measurement checks")
            else:
                report.update(result)
    except Refusal as error:
        report["error"] = str(error)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report.update(status="inconclusive", errorType=type(error).__name__, error=str(error))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--execute-cuda-check", action="store_true",
                        help="Explicit opt-in for one separately authorized CUDA check")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        _worker_entry()
        return 0
    try:
        result = execute(args.device, authorized=True) if args.execute_cuda_check else plan(args.device)
    except Refusal as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] in ("plan", "pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
