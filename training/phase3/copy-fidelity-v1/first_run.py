"""Owner-approved, SHA-pinned ONE bounded P3 copy-fidelity research training run.

Default is still read-only evidence/sampler preflight; --baseline is read-only.
Only explicit --authorize-first-run can initiate a new CUDA run from the
frozen stage and dataset with the exact separately approved contract.
No resume, checkpoint reselection, or final project holdout. No generation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import struct
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from plex_training.artifacts import enforce_storage_limit, path_within_root
from plex_training.checkpoint import read_checkpoint, save_checkpoint
from plex_training.model import parameter_count
from plex_training.tokenizer import CODEC, sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARTIFACT_ROOT = Path.home() / "PlexArtifacts"
STAGE_CHECKPOINT = ARTIFACT_ROOT / "file-edits/p3-copy-fidelity-v1-stage0/stage-checkpoint.pt"
BUNDLE_DIR = ARTIFACT_ROOT / "file-edits/p3-copy-fidelity-v1-bundle"
OUTPUT_DIR = ARTIFACT_ROOT / "file-edits/p3-copy-fidelity-v1-first-run"
APPROVAL = ROOT / "training/pretraining/p3-copy-fidelity-first-run-authorization-draft.json"
STAGE_SHA256 = "3524620d830d363585d501285a9cfdea09d1c2448be08239e2516d78291b3ca9"
BUNDLE_SHA256 = "8eceed468fdcb7202484eba82c8ebb8159a0d9e05787538f2c81f06f62719e2d"
CANDIDATE_SHA256 = "9414d1da7672a386a920d1935723e9133d011c9c3a4029e5d5800f643835ca7a"
SAMPLER_SHA256 = "6d21424636cdb6b5912dacd1564f51b63cc3f9897ec13cb5235e99c17930e9f4"
# Owner approved exactly one first run on 2026-10-17. Both frozen hashes
# must match byte-for-byte; no earlier P3-02 authorization is inherited.
APPROVED_CONTRACT_SHA256: str | None = "a7432a3bfa7942b6086bc8e64fa3b24290351acfd9fb700aac2af613a8c0911b"
APPROVED_BASELINE_SHA256: str | None = "5183726a621fb82539137b009c1514b5ed59d750e3ef4ad4bcd71453e644ff01"
BASELINE_PATH = ROOT / "training/pretraining/p3-copy-fidelity-step-zero-baseline-evidence.json"
SEED = 1337
MAX_STEPS = 100
MAX_SECONDS = 600
GRAD_ACCUMULATION = 16
EXPECTED_RECORDS = {"train": 288, "validation": 96}
EXPECTED_TARGETS = {"train": 7300, "validation": 2420}
VALIDATION_STEPS = (0, 25, 50, 75, 100)
CHECKPOINT_STEPS = (25, 50, 75, 100)

def _module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise ValueError("Frozen sibling module not found")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

PACK = _module("cf_training_pack", "pack_bundle.py")
STAGE = _module("cf_training_stage", "zero_stage.py")


def _rows(split: str, bundle_dir: Path) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    if split not in EXPECTED_RECORDS:
        raise ValueError("Only train and validation splits are allowed")
    # PACK.inspect has independently reconstructed every input/output/mask byte.
    index_path = bundle_dir / f"{split}.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    raw = (bundle_dir / f"{split}.tokens.u16le").read_bytes()
    masks = (bundle_dir / f"{split}.weights.u8").read_bytes()
    if len(raw) % 2 or not isinstance(index, list) or len(index) != EXPECTED_RECORDS[split]:
        raise ValueError("Indexed record count or token layout changed")
    tokens = struct.unpack("<" + "H" * (len(raw) // 2), raw)
    if len(masks) != len(tokens) - len(index):
        raise ValueError("Packed next-token target count mismatched")
    result = []
    offset_tokens = offset_masks = 0
    for entry in index:
        start, count = entry["startToken"], entry["tokenCount"]
        first, mask_count = entry["startTarget"], entry["targetCount"]
        token_ids = tokens[start:start+count]
        target_mask = masks[first:first+mask_count]
        if (start != offset_tokens or first != offset_masks or len(token_ids) != count
                or len(target_mask) != count - 1
                or count > 512
                or target_mask != bytes([0] * (entry["promptTokens"] - 1)
                                        + [1] * entry["answerTokensIncludingEos"])
                or token_ids[-1] != 3
                or any(token < 0 or token >= 16384 for token in token_ids)):
            raise ValueError("Packed record is not exact complete-file answer/EOS supervision")
        result.append((entry["recordId"], token_ids, tuple(target_mask)))
        offset_tokens += count
        offset_masks += mask_count
    if offset_tokens != len(tokens) or offset_masks != len(masks):
        raise ValueError("Packed index leaves unaccounted data")
    if sum(sum(record[2]) for record in result) != EXPECTED_TARGETS[split]:
        raise ValueError("Frozen target positions changed")
    return result


def _draw_indices(count: int) -> tuple[list[int], str]:
    if count != 288:
        raise ValueError("Only the exact 288-record frozen train split can be sampled")
    rng = random.Random(SEED)
    draws: list[int] = []
    while len(draws) < MAX_STEPS * GRAD_ACCUMULATION:
        epoch = list(range(count))
        rng.shuffle(epoch)
        draws.extend(epoch[:MAX_STEPS * GRAD_ACCUMULATION - len(draws)])
    raw = struct.pack("<" + "H" * len(draws), *draws)
    return draws, hashlib.sha256(raw).hexdigest()


def _draws(records: list[tuple[str, tuple[int, ...], tuple[int, ...]]]) -> tuple[list[int], dict[str, Any]]:
    draws, digest = _draw_indices(len(records))
    targets = sum(sum(records[i][2]) for i in draws)
    if (digest != SAMPLER_SHA256 or len(set(draws)) != 288 or targets != 40477):
        raise ValueError("Frozen deterministic sampler SHA/coverage/target positions changed")
    proposal = PACK._sampler(288, [sum(r[2]) for r in records])
    if (proposal["drawIndexSha256"] != digest
            or proposal["expectedAnswerAndEosTargetPositions"] != targets
            or proposal["samplingExecutionAuthorized"] is not False):
        raise ValueError("Sampler differs from frozen bundle contract")
    return draws, proposal


def preflight(*, bundle_dir: Path = BUNDLE_DIR,
              stage_checkpoint: Path = STAGE_CHECKPOINT) -> dict[str, Any]:
    proof = STAGE.verify_stage(checkpoint=stage_checkpoint, bundle_dir=bundle_dir)
    packed = PACK.inspect(bundle_dir)
    if (proof.get("stageCheckpointSha256") != STAGE_SHA256
            or proof.get("baseWeightsPreservedExactly") is not True
            or proof.get("optimizerStateEmpty") is not True
            or packed["bundleManifestSha256"] != BUNDLE_SHA256
            or packed["candidateSha256"] != CANDIDATE_SHA256
            or packed["splits"]["train"]["answerAndEosTargetPositions"] != 7300
            or packed["splits"]["validation"]["answerAndEosTargetPositions"] != 2420):
        raise ValueError("Step-zero source/bundle changed after owner verification")
    train = _rows("train", bundle_dir)
    val = _rows("validation", bundle_dir)
    _, sampler = _draws(train)
    return {
        "status": "copy-fidelity-first-run-read-only-preflight-passed",
        "sourceCheckpointSha256": STAGE.SOURCE_SHA256,
        "stageCheckpointSha256": STAGE_SHA256,
        "bundleManifestSha256": BUNDLE_SHA256,
        "candidateSha256": CANDIDATE_SHA256,
        "trainRecords": len(train), "validationRecords": len(val),
        "validationAnswerAndEosTargets": 2420, "sampler": sampler,
        "baselineMeasured": False, "trainingAuthorized": False,
        "trainingPerformed": False, "optimizerUpdates": 0,
        "finalHoldoutOpened": False,
    }


def _record_loss(model: Any, row: tuple[str, tuple[int, ...], tuple[int, ...]],
                 device: torch.device) -> tuple[torch.Tensor, int]:
    _, tokens, mask = row
    inputs = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
    weights = torch.tensor([mask], dtype=torch.float32, device=device)
    _, loss = model(inputs, targets, target_weights=weights, loss_vocabulary_size=16384)
    if loss is None or not bool(torch.isfinite(loss.detach()).all()):
        raise ValueError("Nonfinite or missing complete-answer/EOS loss")
    return loss, sum(mask)


def _validation_loss(model: Any, records: list, device: torch.device,
                     *, detailed: bool = False) -> dict[str, Any]:
    was_training = bool(getattr(model, "training", False))
    model.eval()
    sums: dict[str, float] = {"all": 0.0}
    counts: Counter[str] = Counter()
    language_by_id = {
        r["id"]: r["language"] for r in PACK._candidate()["rows"] if r["split"] == "validation"
    }
    with torch.inference_mode():
        for row in records:
            loss, n = _record_loss(model, row, device)
            sums["all"] += loss.item() * n
            counts["all"] += n
            lang = language_by_id[row[0]]
            sums[lang] = sums.get(lang, 0.0) + loss.item() * n
            counts[lang] += n
    if (counts["all"] != 2420 or any(counts[l] <= 0 for l in
                                     ("html", "css", "javascript"))):
        raise ValueError("Validation did not cover all 2,420 supervised targets")
    if was_training:
        model.train()
    result = {
        "meanValidationAnswerAndEosLoss": sums["all"] / counts["all"],
        "validationRecordsEvaluated": len(records), "supervisedTargetsEvaluated": counts["all"],
    }
    if detailed:
        result["perLanguage"] = {
            l: {"meanLoss": sums[l] / counts[l], "supervisedTargets": counts[l]}
            for l in ("html", "css", "javascript")
        }
    return result


def baseline(*, bundle_dir: Path = BUNDLE_DIR, stage_checkpoint: Path = STAGE_CHECKPOINT,
             device_name: str = "cpu") -> dict[str, Any]:
    proof = preflight(bundle_dir=bundle_dir, stage_checkpoint=stage_checkpoint)
    if device_name not in ("cpu", "cuda") or (device_name == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Baseline requires an available cpu/cuda device")
    model, payload = read_checkpoint(stage_checkpoint, torch.device("cpu"))
    if (payload.get("step") != 0 or payload.get("tokensProcessedTotal") != 0
            or parameter_count(model) != 27566080):
        raise ValueError("Step-zero stage payload changed")
    model = model.to(torch.device(device_name))
    measured = _validation_loss(model, _rows("validation", bundle_dir),
                                torch.device(device_name), detailed=True)
    return {
        **proof, **measured,
        "status": "copy-fidelity-step-zero-validation-baseline-measured-read-only",
        "baselineMeasured": True, "device": device_name,
        "trainingAuthorized": False, "trainingPerformed": False,
        "optimizerUpdates": 0, "finalHoldoutOpened": False,
    }


def _approved_packet() -> tuple[dict[str, Any], str]:
    """Must refuse before touching the model unless new owner approval is reviewed."""
    if APPROVED_CONTRACT_SHA256 is None or APPROVED_BASELINE_SHA256 is None:
        raise PermissionError(
            "P3 copy-fidelity training NOT authorized: approved contract and "
            "independent step-zero baseline have not been reviewed/pinned"
        )
    if (not isinstance(APPROVED_CONTRACT_SHA256, str)
            or not isinstance(APPROVED_BASELINE_SHA256, str)
            or len(APPROVED_CONTRACT_SHA256) != 64 or len(APPROVED_BASELINE_SHA256) != 64):
        raise PermissionError("Approved evidence is not SHA-256 pinned")
    if (not APPROVAL.is_file() or APPROVAL.is_symlink()
            or sha256_file(APPROVAL) != APPROVED_CONTRACT_SHA256
            or not BASELINE_PATH.is_file() or BASELINE_PATH.is_symlink()
            or sha256_file(BASELINE_PATH) != APPROVED_BASELINE_SHA256):
        raise PermissionError("Frozen owner approval or measured baseline evidence changed")
    packet = json.loads(APPROVAL.read_text(encoding="utf-8"))
    evidence = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    expected = {
        "schemaVersion": 1, "kind": "p3-copy-fidelity-first-run-contract-v1",
        "status": "owner-approved-one-run", "approvedBy": "keepithandy",
        "modelTrainingAuthorized": True,
        "stageCheckpointSha256": STAGE_SHA256, "bundleManifestSha256": BUNDLE_SHA256,
        "candidateSha256": CANDIDATE_SHA256, "samplerSha256": SAMPLER_SHA256,
        "maximumOptimizerUpdates": 100, "maximumWallTimeSeconds": 600,
        "seed": 1337, "gradientAccumulation": 16, "microBatch": 1,
        "resumeAllowed": False, "checkpointReselectionAllowed": False,
        "finalHoldoutOpened": False,
    }
    if any(type(packet.get(k)) is not type(v) or packet[k] != v for k,v in expected.items()):
        raise PermissionError("Owner-approved training settings diverge from reviewed values")
    if (packet.get("approvedDate") != "2026-10-17"
            or packet.get("executionState") != {
                "trainingPerformed": False,
                "optimizerUpdates": 0,
                "stageVerifiedByOwner": True,
                "baselinePending": False,
                "baselineRecordedByOwner": True,
                "runAuthorized": True,
                "oneRunOnly": True,
            }
            or not isinstance(packet.get("authorizationReference"), str)
            or "one bounded P3 copy-fidelity training run" not in packet["authorizationReference"]
            or packet.get("baselineEvidenceSha256") != APPROVED_BASELINE_SHA256
            or evidence.get("status") != "copy-fidelity-step-zero-validation-baseline-measured-read-only"
            or evidence.get("device") != "cuda"
            or evidence.get("stageCheckpointSha256") != STAGE_SHA256
            or evidence.get("bundleManifestSha256") != BUNDLE_SHA256
            or evidence.get("sourceCheckpointSha256") != STAGE.SOURCE_SHA256
            or evidence.get("candidateSha256") != CANDIDATE_SHA256
            or evidence.get("sampler", {}).get("drawIndexSha256") != SAMPLER_SHA256
            or evidence.get("validationRecordsEvaluated") != 96
            or evidence.get("supervisedTargetsEvaluated") != 2420
            or evidence.get("trainingAuthorized") is not False
            or evidence.get("finalHoldoutOpened") is not False
            or evidence.get("trainingPerformed") is not False
            or evidence.get("optimizerUpdates") != 0
            or type(evidence.get("meanValidationAnswerAndEosLoss")) is not float
            or not math.isfinite(evidence["meanValidationAnswerAndEosLoss"])):
        raise PermissionError("Owner baseline and approval packet are not independently pinned")
    return packet, APPROVED_CONTRACT_SHA256


def _training_step(model: Any, optimizer: Any, rows: list, selected: list[int],
                   device: torch.device, deadline: float) -> tuple[float, int] | None:
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    targets = 0
    for index in selected:
        if time.perf_counter() >= deadline:
            optimizer.zero_grad(set_to_none=True)
            return None
        loss, count = _record_loss(model, rows[index], device)
        (loss / len(selected)).backward()
        loss_sum += float(loss.detach().item()) / len(selected)
        targets += count
    if time.perf_counter() >= deadline:
        optimizer.zero_grad(set_to_none=True)
        return None
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0, error_if_nonfinite=True
    )
    if not bool(torch.isfinite(grad_norm).all()):
        raise ValueError("Nonfinite gradient norm")
    optimizer.step()
    return loss_sum, targets


def run_training(*, authorize_first_run: bool = False, bundle_dir: Path = BUNDLE_DIR,
                 stage_checkpoint: Path = STAGE_CHECKPOINT,
                 output_dir: Path = OUTPUT_DIR,
                 artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    """One owner-approved CUDA run; refuses any existing output or altered evidence."""
    if authorize_first_run is not True:
        raise PermissionError("Explicit --authorize-first-run is required")
    packet, authorization_sha = _approved_packet()  # FIRST: refuse before model reads/writes
    root = artifact_root.resolve()
    output = path_within_root(output_dir, root)
    if output != OUTPUT_DIR.resolve(strict=False) or output.exists() or output == root:
        raise ValueError("Run output must be the one unused reviewed directory")
    for protected in (bundle_dir, stage_checkpoint, APPROVAL, BASELINE_PATH):
        source = protected.resolve(strict=False)
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Run output aliases a protected research input")
    if not torch.cuda.is_available():
        raise ValueError("CUDA required; never use silent CPU fallback")
    preflight(bundle_dir=bundle_dir, stage_checkpoint=stage_checkpoint)
    train = _rows("train", bundle_dir)
    validation = _rows("validation", bundle_dir)
    draws, sampler = _draws(train)
    # Compare step-zero validation to exact separately reviewed baseline.
    # Compare on the SAME device as the separately frozen owner baseline.
    # CPU and CUDA loss can differ without any model weight change.
    observed = baseline(bundle_dir=bundle_dir, stage_checkpoint=stage_checkpoint,
                        device_name="cuda")
    expected = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if abs(observed["meanValidationAnswerAndEosLoss"] -
           expected["meanValidationAnswerAndEosLoss"]) > 0.02:
        raise ValueError("Fresh step-zero baseline differs from separately frozen evidence")
    if sha256_file(stage_checkpoint) != STAGE_SHA256:
        raise ValueError("Stage checkpoint changed before the run")
    enforce_storage_limit(root, additional_bytes=2 * 1024**3)
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, stage_payload = read_checkpoint(stage_checkpoint, torch.device("cpu"))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    if optimizer.state:
        raise ValueError("Optimizer must have zero inherited moments")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    metrics_path = output / "metrics.jsonl"
    settings = {
        "kind": "p3-copy-fidelity-owner-authorized-answer-eos-v1",
        "authorizationContractSha256": authorization_sha,
        "baselineEvidenceSha256": APPROVED_BASELINE_SHA256,
        "stageCheckpointSha256": STAGE_SHA256,
        "bundleManifestSha256": BUNDLE_SHA256,
        "samplerSha256": SAMPLER_SHA256,
        "checkpointReselectionAllowed": False,
        "automaticContinuation": False,
    }
    def record(row: dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    started = time.perf_counter()
    deadline = started + MAX_SECONDS
    validations = [{"step": 0, "meanValidationAnswerAndEosLoss":
                    observed["meanValidationAnswerAndEosLoss"]}]
    checkpoints: list[dict[str, Any]] = []
    steps = processed = 0
    interrupted = False
    try:
        model.train()
        record({"event": "start", "atUtc": datetime.now(timezone.utc).isoformat(),
                "stageCheckpointSha256": STAGE_SHA256, "bundleManifestSha256": BUNDLE_SHA256})
        while steps < MAX_STEPS and time.perf_counter() < deadline:
            chosen = draws[steps*GRAD_ACCUMULATION:(steps+1)*GRAD_ACCUMULATION]
            update = _training_step(model, optimizer, train, chosen, device, deadline)
            if update is None:
                break
            mean_loss, targets = update
            steps += 1
            processed += targets
            record({"event": "update", "step": steps, "meanTrainLoss": mean_loss,
                    "answerAndEosTargetsProcessed": processed})
            if steps in VALIDATION_STEPS:
                val = _validation_loss(model, validation, device)
                if not math.isfinite(val["meanValidationAnswerAndEosLoss"]):
                    raise ValueError("Nonfinite validation loss")
                validations.append({"step": steps, **val})
                record({"event": "validation", "step": steps, **val})
            if steps in CHECKPOINT_STEPS:
                destination = checkpoint_dir / f"step-{steps:04d}.pt"
                saved = save_checkpoint(
                    model, optimizer, step=steps, seed=SEED, codec=CODEC,
                    sampling_rng=random.Random(SEED), device=device,
                    destination=destination, artifact_root=root,
                    initialization_record=stage_payload["initializationRecord"],
                    stage_transition_record=stage_payload["stageTransitionRecord"],
                    tokenizer_record=stage_payload["tokenizerRecord"],
                    dataset_record=stage_payload["datasetRecord"],
                    training_settings=settings,
                    schedule_state={"kind": "constant-v1", "step": steps,
                                    "learningRate": 0.0001},
                    tokens_processed_total=processed,
                )
                saved["sha256"] = sha256_file(destination)
                checkpoints.append(saved)
                record({"event": "checkpoint", **saved})
    except KeyboardInterrupt:
        interrupted = True
    finally:
        optimizer.zero_grad(set_to_none=True)
    if steps == MAX_STEPS and processed != 40477:
        raise ValueError("Actual target positions diverge from the frozen sampler")
    if sha256_file(stage_checkpoint) != STAGE_SHA256:
        raise ValueError("Source stage was modified during training")
    report = {
        "status": "bounded-research-run-completed" if steps == MAX_STEPS else "bounded-research-run-incomplete",
        "trainingPerformed": steps > 0, "optimizerUpdates": steps,
        "maximumOptimizerUpdates": MAX_STEPS, "maximumWallTimeSeconds": MAX_SECONDS,
        "completedMaximumSteps": steps == MAX_STEPS, "interrupted": interrupted,
        "elapsedSeconds": time.perf_counter() - started,
        "answerAndEosTargetsProcessed": processed,
        "stageCheckpointSha256": STAGE_SHA256, "bundleManifestSha256": BUNDLE_SHA256,
        "authorizationContractSha256": authorization_sha,
        "validation": validations, "checkpoints": checkpoints,
        "fixedEndpointSha256": (checkpoints[-1]["sha256"] if steps == MAX_STEPS else None),
        "resumeAllowed": False, "automaticContinuation": False,
        "checkpointReselectionAllowed": False,
        "developmentEvaluationPerformed": False, "finalHoldoutOpened": False,
    }
    with (output / "result.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--baseline", action="store_true",
                        help="Read-only teacher-forced step-zero validation loss")
    action.add_argument("--authorize-first-run", action="store_true",
                        help="Execute exactly one owner-approved, SHA-pinned bounded CUDA run")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        if args.authorize_first_run:
            result = run_training(authorize_first_run=True)
        elif args.baseline:
            result = baseline(device_name=args.device)
        else:
            result = preflight()
        print(json.dumps(result, sort_keys=True, indent=2))
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        parser.exit(2, f"p3-copy-fidelity-first-run: {exc}\n")


if __name__ == "__main__":
    main()
