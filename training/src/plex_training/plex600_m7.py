"""Plex 600M M7: plan/preflight and separately authorized four-update GPU pilot.

No optimizer update occurs without BOTH an external owner approval record and
--execute-approved-run. The current repository includes neither authorization.
"""
from __future__ import annotations

import argparse
from array import array
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

from . import plex600_feasibility as hw
from . import plex600_owner_pilot as source
from . import plex600_setup as setup

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "training" / "artifacts"
PROTOCOL = "plex600-m7-four-update-v1"
APPROVAL_KIND = "plex600-m7-owner-training-approval-v1"
SOURCE_SHA = "4bb5742df59f0c56aaac34fc3129012b4617acf7d86a1804e47c5aa69e541a07"
TOKENIZER_SHA = "9feef87bdf5fc28b3315df589e4e0b8ffcf5f40b18a4c357c91759182340a2e2"
TRAIN_SHA = "3819b4fe0939e5f12b82c5a0e78d536ccc746fb558e419647eb28d0969886500"
VALID_SHA = "57cea37870cc9b5fde4e2bc525b778282951c13a06c87762ca18065a53d42999"
SEED, UPDATES, WINDOW = 17, 4, 128
TIMEOUT = 900
MAX_CHECKPOINT_BYTES = 3 * 1024**3
MIN_FREE_DISK = 5 * 1024**3
CONTROL = "PLEX600_M7_WORKER_NONCE"


class Refusal(ValueError):
    pass


def hash_file(path: Path, *, maximum: int | None = None) -> str:
    if path.is_symlink() or not path.is_file():
        raise Refusal("Missing or linked file")
    hasher, read_bytes = hashlib.sha256(), 0
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            read_bytes += len(chunk)
            if maximum is not None and read_bytes > maximum:
                raise Refusal("Evidence exceeds maximum bytes")
            hasher.update(chunk)
    return hasher.hexdigest()


def plan() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL, "status": "plan", "optimizerUpdatesPlanned": UPDATES,
        "optimizerUpdatesCompleted": 0, "trainingAuthorized": False,
        "modelFamily": "plex-600m-v1", "parameterCount": hw.PARAMETERS,
        "modelSha256Accepted": list(hw.ACCEPTED_MODEL_SHA256),
        "sourceManifestSha256": SOURCE_SHA, "tokenizerSha256": TOKENIZER_SHA,
        "trainPackedSha256": TRAIN_SHA, "validationPackedSha256": VALID_SHA,
        "trainRecords": 120, "validationRecords": 24,
        "trainTokens": 6970, "validationTokens": 1845,
        "sampling": "seed-17-shuffled-record-order-no-cross-record-128-token-window",
        "validation": "all-24-internal-records-before-after-no-grad",
        "optimizer": {"name": "AdamW", "lr": 0.0001, "weightDecay": 0,
                      "eps": 1e-8, "foreach": False, "fused": False},
        "seed": SEED, "batchSize": 1, "sequenceLengthMax": WINDOW,
        "dtypeParameters": "float32", "dtypeAutocast": "bfloat16",
        "gradientCheckpointing": True, "maxUpdates": UPDATES,
        "minimumFreeBytes": hw.MIN_FREE_BYTES,
        "allocatorFraction": hw.MEMORY_FRACTION, "timeoutSeconds": TIMEOUT,
        "checkpoint": "one-model-state-dict-only-create-only-sha256",
        "checkpointMaxBytes": MAX_CHECKPOINT_BYTES,
        "minimumFreeDiskBytes": MIN_FREE_DISK,
        "historicalValidationAccessed": False, "finalHoldoutAccessed": False,
        "externalSixSourcesUsed": False,
    }


def preflight(bundle: Path, dataset: Path = source.DATASET,
              *, expected_bundle_sha: str) -> dict[str, Any]:
    """No CUDA, no optimizer: fail closed on bundle/rights/provenance identity."""
    if dataset.resolve(strict=True) != source.DATASET.resolve(strict=True):
        raise Refusal("M7 accepts only the committed owner-authored dataset")
    if source.verify(dataset)["status"] != "verified-owner-authored-tokenizer-ready":
        raise Refusal("Owner-authored dataset failed verification")
    if not isinstance(expected_bundle_sha, str) or len(expected_bundle_sha) != 64:
        raise Refusal("Explicit bundle manifest SHA-256 required")
    if hash_file(dataset / "manifest.json", maximum=1024**2) != SOURCE_SHA:
        raise Refusal("Changed owner-authored source manifest")
    if hash_file(bundle / "manifest.json", maximum=1024**2) != expected_bundle_sha:
        raise Refusal("Bundle manifest SHA-256 mismatch")
    packed = setup.check_source_binding(
        bundle, dataset, bundle_manifest_sha256=expected_bundle_sha,
        source_manifest_sha256=SOURCE_SHA,
    )
    checked = setup.check_tokenizer_bundle(bundle)
    if (packed["sourceBindingVerified"] is not True
            or checked["actualVocabularySize"] != 1191
            or checked["sourceDatasetManifestSha256"] != SOURCE_SHA
            or checked["tokenizerSha256"] != TOKENIZER_SHA
            or packed["trainingAuthorized"] is not False):
        raise Refusal("M7 requires the exact verified tokenizer")
    for split, digest, nrecords, ntokens in (
        ("train", TRAIN_SHA, 120, 6970),
        ("validation", VALID_SHA, 24, 1845),
    ):
        got = checked["verifiedSplits"][split]
        if got["sha256"] != digest or got["records"] != nrecords or got["tokens"] != ntokens:
            raise Refusal("Split identity differs from M7 fixed pilot")
    return {
        "status": "source-bound-plan-only", "trainingAuthorized": False,
        "bundleManifestSha256": expected_bundle_sha,
        "sourceManifestSha256": SOURCE_SHA,
        "actualVocabularySize": 1191,
        "trainTokens": 6970, "validationTokens": 1845,
        "sourceBindingVerified": True, "optimizerUpdatesCompleted": 0,
    }


def checked_approval(path: Path, *, bundle_sha: str, source_sha: str = SOURCE_SHA) -> dict:
    """Independent, noncommitted owner decision; cannot be inferred from data."""
    raw = path.read_bytes() if path.is_file() and not path.is_symlink() and path.stat().st_size <= 8192 else b""
    try:
        decision = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise Refusal("Missing independently reviewed training approval") from exc
    if (not isinstance(decision, dict)
            or set(decision) != {"kind", "approvalStatus", "approvedBy",
                                "trainingScope", "approvedUpdates", "sourceManifestSha256",
                                "bundleManifestSha256", "finalHoldoutAllowed"}
            or decision["kind"] != APPROVAL_KIND
            or decision["approvalStatus"] != "approved"
            or decision["approvedBy"] != "repository-owner"
            or decision["trainingScope"] != "local-four-update-m7-owner-authored-pilot-only"
            or type(decision["approvedUpdates"]) is not int
            or decision["approvedUpdates"] != UPDATES
            or decision["sourceManifestSha256"] != source_sha
            or decision["bundleManifestSha256"] != bundle_sha
            or decision["finalHoldoutAllowed"] is not False):
        raise Refusal("No valid owner-scoped M7 training authorization")
    return {"approvalSha256": hashlib.sha256(raw).hexdigest(),
            "trainingAuthorized": True, "approvedUpdates": UPDATES}


def runner_identity() -> dict[str, str]:
    """Refuse uncommitted edits to this executable and record its exact bytes."""
    runner = Path(__file__).resolve()
    expected = runner.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", expected],
        cwd=ROOT, text=True, capture_output=True, check=True, timeout=5)
    if result.stdout.strip():
        raise Refusal("M7 runner has local/staged changes")
    return {"m7RunnerSha256": hash_file(runner, maximum=1024 * 1024)}


def output_path(path: Path, *, bundle: Path, dataset: Path) -> Path:
    """Restrict write destination to local ignored artifacts outside input roots."""
    root = ARTIFACTS.resolve(strict=True)
    target = path.resolve(strict=False)
    if (target == root or root not in target.parents
            or target.exists() or path.is_symlink()
            or dataset.resolve(strict=True) in target.parents
            or bundle.resolve(strict=True) in target.parents):
        raise Refusal("Output must be a new directory inside training/artifacts")
    if (not target.parent.is_dir() or target.parent.is_symlink()
            or target.parent.resolve(strict=True) != target.parent):
        raise Refusal("Output parent must already exist and be a real directory")
    if shutil.disk_usage(target.parent).free < MIN_FREE_DISK:
        raise Refusal("At least five GiB of free disk required for checkpoint")
    return target


def record_batches(bundle: Path, split: str) -> list[tuple[str, list[int]]]:
    """Bounded file-aligned samples only; no mixing adjacent records."""
    if split not in {"train", "validation"}:
        raise Refusal("Unsupported training split")
    manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
    info = manifest[split]
    length = info["tokenCount"]
    if length > 8192 or info["sha256"] != (TRAIN_SHA if split == "train" else VALID_SHA):
        raise Refusal("Unexpected token stream size/hash")
    raw = (bundle / f"{split}.tokens.u16le").read_bytes()
    if len(raw) != length * 2 or hash_file(bundle / f"{split}.tokens.u16le") != info["sha256"]:
        raise Refusal("Packed token stream changed")
    indices = json.loads((bundle / f"{split}.index.json").read_text("utf-8"))
    if not isinstance(indices, list) or len(indices) != (120 if split == "train" else 24):
        raise Refusal("Invalid split record inventory")
    all_tokens = array("H")
    all_tokens.frombytes(raw)
    if sys.byteorder != "little":
        all_tokens.byteswap()
    vocab = manifest["actualVocabularySize"]
    offset = 0
    result = []
    for index in indices:
        count = index["tokenCount"]
        if (index["startToken"] != offset or type(count) is not int
                or count < 2 or count > 8192):
            raise Refusal("Invalid indexed token window")
        ids = all_tokens[offset:offset + count].tolist()
        if ids[-1] != 3 or any(t < 4 or t >= vocab for t in ids[:-1]):
            raise Refusal("Invalid data/EOS token values")
        result.append((index["recordId"], ids))
        offset += count
    if offset != length or len({rid for rid, _ in result}) != len(result):
        raise Refusal("Unexpected offsets or duplicate records")
    return result


def choose_windows(records: list[tuple[str, list[int]]], *, seed=SEED):
    """Four deterministic windows (context <=128) from distinct train records."""
    import random
    ids = list(range(len(records)))
    random.Random(seed).shuffle(ids)
    if len(ids) < UPDATES:
        raise Refusal("Not enough independent train records")
    return [(records[i][0], records[i][1][:WINDOW + 1]) for i in ids[:UPDATES]]


def evaluate(model: Any, torch: Any, validation: list, device: Any) -> dict[str, Any]:
    model.eval()
    weighted_loss, targets = 0., 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _, tokens in validation:
            observed = tokens[:WINDOW + 1]
            inputs = torch.tensor([observed[:-1]], device=device, dtype=torch.long)
            labels = torch.tensor([observed[1:]], device=device, dtype=torch.long)
            _, loss = model(inputs, labels)
            if loss is None or not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite internal validation loss")
            weighted_loss += float(loss.item()) * labels.numel()
            targets += labels.numel()
    if not targets:
        raise RuntimeError("Empty internal validation targets")
    return {"loss": weighted_loss / targets, "targetTokens": targets, "records": len(validation)}


def run_worker(bundle: Path, dataset: Path, output_stage: Path,
               bundle_sha: str, approval: Path, report: dict[str, Any]):
    """Restricted worker, invoked only after parent and worker approval gates."""
    from . import plex600
    import torch
    checked = preflight(bundle, dataset, expected_bundle_sha=bundle_sha)
    permit = checked_approval(approval, bundle_sha=bundle_sha)
    identity = {**hw.source_identity(), **runner_identity()}
    if (plex600.Plex600Config().to_dict() != hw.CONFIGURATION
            or identity["modelSha256"] not in hw.ACCEPTED_MODEL_SHA256):
        raise Refusal("Model differs from fixed M6 architecture")
    report.update({"trainingAuthorized": True, "source": checked, "approval": permit, "codeIdentity": identity,
                   "torch": str(torch.__version__), "cudaRuntime": str(torch.version.cuda)})
    report["gpu"] = hw._check_device(torch, 0)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(hw.MEMORY_FRACTION, device)
    torch.cuda.reset_peak_memory_stats(device)
    report["phase"] = "dataset-sampling"
    train = choose_windows(record_batches(bundle, "train"))
    validation = record_batches(bundle, "validation")
    report["trainingRecordIds"] = [row[0] for row in train]
    report["phase"] = "model-initialization"
    torch.manual_seed(SEED)
    with torch.device("cpu"):
        model = plex600.Plex600Model()
    if plex600.parameter_count(model) != hw.PARAMETERS:
        raise Refusal("Full model parameter count differs")
    model.to(device, dtype=torch.float32)
    model.gradient_checkpointing_enable()
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=0.0001, weight_decay=0, eps=1e-8,
                                  foreach=False, fused=False)
    report["phase"] = "baseline-validation"
    report["validationBefore"] = evaluate(model, torch, validation, device)
    report["phase"] = "four-optimizer-updates"
    report["steps"] = []
    report["optimizerUpdatesCompleted"] = 0
    start_updates = time.perf_counter()
    for step, (rid, ids) in enumerate(train, 1):
        model.train()
        inputs = torch.tensor([ids[:-1]], device=device, dtype=torch.long)
        labels = torch.tensor([ids[1:]], device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(inputs, labels)
        if loss is None or not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite training loss")
        loss_before = float(loss.item())
        loss.backward()
        for param in parameters:
            if param.grad is None or not bool(torch.isfinite(param.grad).all()):
                raise RuntimeError("Non-finite or absent gradient")
        # No optimizer step may happen after step four.
        optimizer.step()
        torch.cuda.synchronize(device)
        report["optimizerUpdatesCompleted"] = step
        report["steps"].append({"step": step, "recordId": rid,
                                "lossBeforeUpdate": loss_before,
                                "supervisedTokens": labels.numel()})
    report["updatesSeconds"] = time.perf_counter() - start_updates
    report["phase"] = "post-validation"
    report["validationAfter"] = evaluate(model, torch, validation, device)
    # Report final train loss on same selected windows WITHOUT another update.
    report["phase"] = "post-training-train-loss"
    report["trainLossAfter"] = evaluate(model, torch, train, device)["loss"]
    report["peakAllocatedBytes"] = torch.cuda.max_memory_allocated(device)
    report["peakReservedBytes"] = torch.cuda.max_memory_reserved(device)
    total_trained = sum(step["supervisedTokens"] for step in report["steps"])
    report["trainedTokenPositions"] = total_trained
    report["tokensPerSecondUpdatesOnly"] = total_trained / report["updatesSeconds"]
    # Create-only model state; NO pickle file is loaded by this pilot.
    report["phase"] = "checkpoint"
    checkpoint = output_stage / "weights.pt"
    if checkpoint.exists():
        raise Refusal("Checkpoint already exists")
    torch.save(model.state_dict(), checkpoint)
    size = checkpoint.stat().st_size
    if not 0 < size <= MAX_CHECKPOINT_BYTES:
        raise Refusal("Checkpoint size beyond pinned ceiling")
    report["checkpoint"] = {"file": "weights.pt", "bytes": size,
                            "sha256": hash_file(checkpoint, maximum=MAX_CHECKPOINT_BYTES),
                            "weightsOnly": True, "optimizerStateSaved": False,
                            "resumeAuthorized": False}
    report["phase"] = "complete"
    report["status"] = "pass"
    return report


def worker() -> None:
    report = {**plan(), "status": "refused"}
    try:
        nonce = os.environ.pop(CONTROL, "")
        msg = json.loads(sys.stdin.read(4097))
        if (not nonce or type(msg) is not dict or msg.get("nonce") != nonce
                or set(msg) != {"nonce", "bundle", "dataset", "stage", "bundleSha", "approval"}):
            raise Refusal("M7 worker cannot run independently")
        report["phase"] = "authorization-preflight"
        run_worker(Path(msg["bundle"]), Path(msg["dataset"]), Path(msg["stage"]),
                   msg["bundleSha"], Path(msg["approval"]), report)
    except Exception as exc:
        report["status"] = "refused" if isinstance(exc, (Refusal, ValueError)) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)


def execute(bundle: Path, approval: Path, output: Path, *, bundle_sha: str,
            execute_flag: bool = False) -> dict[str, Any]:
    report = {**plan(), "status": "refused"}
    if execute_flag is not True:
        report["error"] = "Explicit --execute-approved-run required"
        return report
    stage = None
    try:
        # Authorize and pin ALL inputs BEFORE model/GPU access or output creation.
        source_report = preflight(bundle, expected_bundle_sha=bundle_sha)
        rights = checked_approval(approval, bundle_sha=bundle_sha)
        identity = {**hw.source_identity(), **runner_identity()}
        if identity["modelSha256"] not in hw.ACCEPTED_MODEL_SHA256:
            raise Refusal("Unexpected model SHA")
        target = output_path(output, bundle=bundle, dataset=source.DATASET)
        stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
        env = dict(os.environ, **{CONTROL: secrets.token_hex(32), "PYTHONDONTWRITEBYTECODE": "1"})
        token = env[CONTROL]
        payload = {"nonce": token, "bundle": str(bundle.resolve(strict=True)),
                   "dataset": str(source.DATASET.resolve(strict=True)),
                   "stage": str(stage), "bundleSha": bundle_sha,
                   "approval": str(approval.resolve(strict=True))}
        command = [sys.executable, "-m", "plex_training.plex600_m7", "--_worker"]
        code, stdout, stderr, timeout = hw._supervise(command, payload, env, timeout=TIMEOUT)
        report.update({"workerExitCode": code, "workerStderr": stderr[-4096:]})
        if timeout:
            raise Refusal("M7 worker exceeded 900 seconds")
        if code != 0:
            raise Refusal("M7 worker failed without a validated result")
        result = json.loads(stdout)
        checkpoint = result.get("checkpoint")
        if (result.get("protocol") != PROTOCOL or result.get("status") != "pass"
                or result.get("optimizerUpdatesCompleted") != UPDATES
                or result.get("trainingAuthorized") is not True
                or result.get("source") != source_report
                or result.get("approval") != rights
                or result.get("codeIdentity") != identity
                or result.get("phase") != "complete" or not isinstance(checkpoint, dict)
                or checkpoint.get("file") != "weights.pt" or checkpoint.get("weightsOnly") is not True
                or checkpoint.get("optimizerStateSaved") is not False
                or not all(isinstance(step, dict) and step.get("step") == index
                           for index, step in enumerate(result.get("steps", []), 1))
                or len(result.get("steps", [])) != UPDATES):
            raise Refusal("Untrusted or incomplete worker result")
        if (checkpoint.get("bytes") != (stage / "weights.pt").stat().st_size
                or checkpoint.get("sha256") != hash_file(stage / "weights.pt",
                                                          maximum=MAX_CHECKPOINT_BYTES)):
            raise Refusal("Checkpoint hash/size verification failed")
        for key in ("validationBefore", "validationAfter"):
            val = result.get(key)
            if (not isinstance(val, dict) or val.get("records") != 24
                    or type(val.get("targetTokens")) is not int or val["targetTokens"] <= 0
                    or type(val.get("loss")) not in (float, int)
                    or not math.isfinite(val["loss"])):
                raise Refusal("Invalid validation evidence")
        # Report is part of the atomically published checkpoint directory.
        result["outputPublished"] = True
        with (stage / "report.json").open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        if target.exists():
            raise Refusal("Output appeared during M7 execution")
        os.rename(stage, target)
        stage = None
        return result
    except (OSError, ValueError, subprocess.SubprocessError, KeyError, TypeError) as exc:
        report.update(status="refused", error=f"{type(exc).__name__}: {exc}")
        return report
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--bundle-manifest-sha256")
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight", action="store_true", help="Read-only source/bundle check; no GPU")
    parser.add_argument("--execute-approved-run", action="store_true",
                        help="Only with external owner training approval: four real optimizer updates")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        worker()
        return 0
    if args.preflight and args.execute_approved_run:
        parser.error("--preflight and --execute-approved-run are mutually exclusive")
    if args.preflight or args.execute_approved_run:
        if not args.bundle_dir or not args.bundle_manifest_sha256:
            parser.error("Bundle directory and independent manifest SHA-256 required")
    if args.execute_approved_run and (not args.approval_file or not args.output_dir):
        parser.error("Execution needs a separately approved packet and new output directory")
    try:
        if args.preflight:
            answer = preflight(args.bundle_dir, expected_bundle_sha=args.bundle_manifest_sha256)
        elif args.execute_approved_run:
            answer = execute(args.bundle_dir, args.approval_file, args.output_dir,
                             bundle_sha=args.bundle_manifest_sha256, execute_flag=True)
        else:
            answer = plan()
    except (OSError, ValueError) as exc:
        answer = {**plan(), "status": "refused", "error": str(exc)}
    print(json.dumps(answer, indent=2, sort_keys=True, allow_nan=False))
    return 0 if answer["status"] in ("plan", "source-bound-plan-only", "pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
