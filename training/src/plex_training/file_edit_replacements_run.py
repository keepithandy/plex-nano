"""P3-02 bounded answer/EOS file-edit training: draft gated until owner approval.

Training is impossible with the committed draft packet. The explicit CLI flag,
an exact owner-approved packet, a frozen stage SHA and baseline check are separate
prerequisites. No development/final-holdout reader or resume path is included.
"""
from __future__ import annotations

import json
import math
import os
import random
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .artifacts import enforce_storage_limit, path_within_root
from .checkpoint import read_checkpoint, save_checkpoint
from .file_edit_replacements import MILESTONE, PROPOSED_RUN, ROOT
from .file_edit_replacements_bundle import (
    DEFAULT_CHECKPOINT, DEFAULT_OUTPUT as DEFAULT_BUNDLE, DEFAULT_P248_RESULT,
    inspect_replacement_bundle,
)
from .file_edit_replacements_stage import (
    BUNDLE_SHA256, DEFAULT_STAGE_DIR, STAGE_FILENAME, _read_evidence,
    baseline_replacement_preflight, preflight_replacement_stage,
)
from .model import parameter_count
from .runner import ADAMW_BETAS, ADAMW_EPSILON, ADAMW_WEIGHT_DECAY
from .tokenizer import CODEC, sha256_file

DEFAULT_AUTHORIZATION = ROOT / "training/pretraining/p3-02-first-run-contract.json"
DEFAULT_BASELINE = ROOT / "training/pretraining/p3-02-step-zero-baseline-result.json"
DEFAULT_STAGE = DEFAULT_STAGE_DIR / STAGE_FILENAME
DEFAULT_OUTPUT = ROOT / "training/artifacts/file-edits/p3-02-first-run"
AUTHORIZATION_KIND = "plex-p3-02-first-exact-replacement-run-contract-v1"
APPROVED_STATUS = "owner-approved-first-run"
DRAFT_AUTHORIZATION_SCOPE = (
    "Draft only; no training until separate explicit owner approval changes "
    "the frozen packet and CLI --authorize-first-run flag is present."
)
APPROVED_AUTHORIZATION_SCOPE = (
    "Owner approved on 2026-10-13 for one bounded P3-02 first run only: "
    "at most 100 optimizer updates or 600 seconds, whichever comes first. "
    "CUDA, fixed source/stage/bundle, no resume, automatic continuation, "
    "checkpoint reselection, or development/final holdout access. "
    "Execution separately requires --authorize-first-run."
)
EXPECTED_BASELINE_LOSS = 10.566839536031088
EXPECTED_STAGE_SHA256 = "50f18e463d6a51bac5a21c0101b13d4b33c7e0d3eb41904600f6fce8bc622246"
EXPECTED_TOKENIZER_SHA256 = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
EXPECTED_SAMPLER_POSITIONS = 60445
MAX_STEPS = 100
MAX_SECONDS = 600
SEED = 1337
ACCUMULATION = 16
VALIDATION_STEPS = (0, 25, 50, 75, 100)
CHECKPOINT_STEPS = (25, 50, 75, 100)
NO_TRAINING = {
    "modelTrainingAuthorized": False, "trainingPerformed": False,
    "researchOptimizerUpdates": 0, "finalHoldoutOpened": False,
    "automaticContinuation": False,
}


def _bounded_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("P3-02 contract/baseline must be a bounded regular JSON file")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("P3-02 contract/baseline must be a JSON object")
    return obj


def _require(actual: Any, expected: Any, label: str) -> None:
    # bool must never pass as int. Likewise refuse additional/missing dict fields.
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError("P3-02 frozen authorization mismatch: " + label)


def _expected_source() -> dict[str, Any]:
    return {
        "sourceCheckpointSha256": PROPOSED_RUN["sourceCheckpointSha256"],
        "stageCheckpointSha256": EXPECTED_STAGE_SHA256,
        "bundleManifestSha256": BUNDLE_SHA256,
        "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
        "trainRecords": 144, "validationRecords": 36,
        "trainAnswerAndEosTargets": 5424,
        "validationAnswerAndEosTargets": 1320,
        "modelParameters": 27566080, "contextLength": 512, "vocabularySize": 16384,
    }


def _expected_training() -> dict[str, Any]:
    return {
        "device": "cuda", "seed": SEED,
        "objective": "answer-eos-only-complete-record-v1",
        "optimizer": "AdamW", "learningRate": PROPOSED_RUN["learningRate"],
        "betas": list(ADAMW_BETAS), "epsilon": ADAMW_EPSILON,
        "weightDecay": ADAMW_WEIGHT_DECAY, "gradientClippingNorm": 1,
        "microBatch": 1, "gradientAccumulation": ACCUMULATION,
        "lossReduction": "mean-weighted-answer-eos-per-example-then-mean-across-accumulation-v1",
        "selection": "uniform-complete-train-record-with-replacement",
        "expectedExamplesAt100Steps": 1600,
        "expectedAnswerAndEosPositionsAt100Steps": EXPECTED_SAMPLER_POSITIONS,
        "maximumOptimizerUpdates": MAX_STEPS,
        "maximumWallTimeSeconds": MAX_SECONDS,
        "validationSteps": list(VALIDATION_STEPS),
        "checkpointSteps": list(CHECKPOINT_STEPS),
        "maxNewTokens": 192,
        "checkpointReselectionAllowed": False,
        "automaticContinuation": False,
        "resumeAllowed": False,
    }


def validate_authorization(packet: dict[str, Any], *, allow_draft: bool) -> bool:
    """Only exact reviewed packet settings are acceptable.

    Returns True solely for explicit owner-approved status; draft is read-only.
    """
    _require(set(packet), {
        "schemaVersion", "milestone", "kind", "status", "approvalPacketComplete",
        "modelTrainingAuthorized", "approvedBy", "approvedDate", "authorizationScope",
        "source", "baseline", "training", "outputDirectory",
        "protectedEvaluation", "executionState", "continuationRule",
    }, "fields")
    for k, v in {
        "schemaVersion": 1, "milestone": MILESTONE, "kind": AUTHORIZATION_KIND,
        "approvalPacketComplete": True,
        "source": _expected_source(),
        "baseline": {
            "kind": "teacher-forced-validation-answer-eos-only-v1",
            "meanLoss": EXPECTED_BASELINE_LOSS, "validationRecords": 36,
            "supervisedTargets": 1320,
            "recordPath": "training/pretraining/p3-02-step-zero-baseline-result.json",
            "absTolerance": 0.02,
        },
        "training": _expected_training(),
        "outputDirectory": "file-edits/p3-02-first-run",
        "protectedEvaluation": {
            "validationExcludedFromGradients": True,
            "developmentExcludedFromGradients": True,
            "finalProjectHoldoutMustRemainClosed": True,
        },
        "executionState": {
            "trainingExecuted": False, "researchOptimizerUpdates": 0,
            "finalHoldoutOpened": False,
        },
        "continuationRule": (
            "Stop at the first 100 updates or 600 seconds, whichever happens first. "
            "No automatic restart, checkpoint reselection, or continuation."
        ),
    }.items():
        _require(packet.get(k), v, k)
    if packet.get("status") == "owner-approval-required":
        _require(packet.get("authorizationScope"), DRAFT_AUTHORIZATION_SCOPE, "draft authorizationScope")
        _require(packet.get("modelTrainingAuthorized"), False, "draft training flag")
        _require(packet.get("approvedBy"), None, "draft approvedBy")
        _require(packet.get("approvedDate"), None, "draft approvedDate")
        if not allow_draft:
            raise ValueError("P3-02 model training is NOT authorized by the committed draft")
        return False
    _require(packet.get("status"), APPROVED_STATUS, "status")
    _require(packet.get("authorizationScope"), APPROVED_AUTHORIZATION_SCOPE, "approved authorizationScope")
    _require(packet.get("modelTrainingAuthorized"), True, "approval training flag")
    _require(packet.get("approvedBy"), "keepithandy", "approvedBy")
    date = packet.get("approvedDate")
    if type(date) is not str or len(date) != 10:
        raise ValueError("P3-02 approval date is missing")
    try:
        if datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d") != date:
            raise ValueError("P3-02 approvedDate is not a valid ISO date")
    except ValueError as exc:
        raise ValueError("P3-02 approvedDate is not a valid ISO date") from exc
    return True


def validate_frozen_baseline(result: dict[str, Any]) -> None:
    for k, v in {
        "schemaVersion": 1, "milestone": MILESTONE,
        "kind": "p3-02-owner-reported-step-zero-baseline-v1",
        "bundleManifestSha256": BUNDLE_SHA256,
        "stageCheckpointSha256": EXPECTED_STAGE_SHA256,
        "sourceCheckpointSha256": PROPOSED_RUN["sourceCheckpointSha256"],
        "metric": "validation-answer-and-eos-cross-entropy-per-supervised-target",
        "meanValidationAnswerAndEosLoss": EXPECTED_BASELINE_LOSS,
        "validationRecordsEvaluated": 36, "supervisedTargetsEvaluated": 1320,
        "modelWeightsPreservedExactly": True, "modelTrainingAuthorized": False,
        "trainingPerformed": False, "researchOptimizerUpdates": 0,
        "finalHoldoutOpened": False, "automaticContinuation": False,
    }.items():
        _require(result.get(k), v, "baseline." + k)
    for language, count, loss in (
        ("css", 521, 10.045777009682096),
        ("html", 455, 10.963397183261074),
        ("javascript", 344, 10.831490253293238),
    ):
        data = result.get("perLanguage", {}).get(language)
        if not isinstance(data, dict):
            raise ValueError("P3-02 baseline language missing")
        _require(data.get("supervisedTargets"), count, language + " targets")
        _require(data.get("meanLoss"), loss, language + " loss")


def _verified_packet(bundle_dir: Path, stage_checkpoint: Path, source_checkpoint: Path,
                     result_path: Path, baseline_path: Path, require_baseline: bool,
                     device: str = "cpu") -> tuple[dict[str, Any], dict[str, Any]]:
    """No artifact writes. Recheck exact local source, stage, and train/validation."""
    recorded = _bounded_json(baseline_path)
    validate_frozen_baseline(recorded)
    proof = preflight_replacement_stage(
        stage_checkpoint=stage_checkpoint, source_checkpoint=source_checkpoint,
        bundle_dir=bundle_dir, result_path=result_path,
    )
    if (proof.get("stageCheckpointSha256") != EXPECTED_STAGE_SHA256
            or proof.get("bundleManifestSha256") != BUNDLE_SHA256
            or proof.get("baseWeightsPreservedExactly") is not True
            or proof.get("optimizerStateEmpty") is not True):
        raise ValueError("P3-02 exact source/stage preflight differs from owner's frozen evidence")
    bundle = inspect_replacement_bundle(bundle_dir)
    if (bundle["bundleManifestSha256"] != BUNDLE_SHA256
            or bundle["records"] != {"train": 144, "validation": 36}
            or bundle["answerAndEosTargetPositions"] != {"train": 5424, "validation": 1320}
            or bundle["sampler"]["expectedAnswerAndEosTargetPositions"] != EXPECTED_SAMPLER_POSITIONS):
        raise ValueError("P3-02 source data or deterministic sampler differs from the reviewed packet")
    if require_baseline:
        measured = baseline_replacement_preflight(
            stage_checkpoint=stage_checkpoint, source_checkpoint=source_checkpoint,
            bundle_dir=bundle_dir, result_path=result_path, device=device,
        )
        current = measured["meanValidationAnswerAndEosLoss"]
        if (measured.get("supervisedTargetsEvaluated") != 1320
                or measured.get("validationRecordsEvaluated") != 36
                or not math.isfinite(current)
                or abs(current - EXPECTED_BASELINE_LOSS) > 0.02):
            raise ValueError("P3-02 locally recomputed stage-zero validation baseline differs")
    return proof, bundle


def _packed_records(bundle_dir: Path, split: str) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    """Caller must first inspect the entire exact frozen bundle."""
    if split not in ("train", "validation"):
        raise ValueError("P3-02 training/evaluation split must be train or validation")
    index = json.loads((bundle_dir / f"{split}.index.json").read_text(encoding="utf-8"))
    raw = (bundle_dir / f"{split}.tokens.u16le").read_bytes()
    masks = (bundle_dir / f"{split}.weights.u8").read_bytes()
    if len(raw) % 2:
        raise ValueError("P3-02 packed tokens must be little-endian uint16")
    ids = struct.unpack("<" + "H" * (len(raw) // 2), raw)
    expected_count = 144 if split == "train" else 36
    if not isinstance(index, list) or len(index) != expected_count:
        raise ValueError("P3-02 indexed split record count changed")
    result = []
    for entry in index:
        start, length = entry["startToken"], entry["tokenCount"]
        target_start, count = entry["startTarget"], entry["targetCount"]
        token_ids = ids[start:start + length]
        mask = masks[target_start:target_start + count]
        if (len(token_ids) != length or len(mask) != length - 1
                or not 1 <= sum(mask) < len(token_ids)
                or any(v not in (0, 1) for v in mask)
                or any(token >= 16384 for token in token_ids)):
            raise ValueError("P3-02 indexed answer-only training record differs")
        result.append((entry["recordId"], token_ids, tuple(mask)))
    return result


def _replay_positions(records: list[tuple[str, tuple[int, ...], tuple[int, ...]]]) -> dict[str, int]:
    if len(records) != 144:
        raise ValueError("P3-02 training sampler must have 144 records")
    rng = random.Random(SEED)
    chosen = [rng.randrange(len(records)) for _ in range(1600)]
    positions = sum(sum(records[i][2]) for i in chosen)
    if positions != EXPECTED_SAMPLER_POSITIONS or len(set(chosen)) != 144:
        raise ValueError("P3-02 actual record order or deterministic training sampler changed")
    return {"examples": 1600, "recordsSelected": len(set(chosen)),
            "expectedAnswerAndEosTargetPositions": positions}


def preflight_replacement_first_run(
    *, bundle_dir: Path = DEFAULT_BUNDLE, stage_checkpoint: Path = DEFAULT_STAGE,
    source_checkpoint: Path = DEFAULT_CHECKPOINT, result_path: Path = DEFAULT_P248_RESULT,
    baseline_path: Path = DEFAULT_BASELINE, authorization_path: Path = DEFAULT_AUTHORIZATION,
) -> dict[str, Any]:
    """Read-only preflight; intentionally leaves the committed training draft blocked."""
    packet = _bounded_json(authorization_path)
    approved = validate_authorization(packet, allow_draft=True)
    proof, bundle = _verified_packet(
        bundle_dir, stage_checkpoint, source_checkpoint, result_path,
        baseline_path, require_baseline=True, device="cpu",
    )
    sampler = _replay_positions(_packed_records(bundle_dir, "train"))
    return {
        "schemaVersion": 1, "milestone": MILESTONE,
        "status": "first-run-source-and-baseline-preflight-passed",
        "authorizationStatus": packet["status"],
        "trainingReadyIfSeparatelyApproved": True,
        "modelTrainingAuthorized": approved,
        "sourceCheckpointSha256": PROPOSED_RUN["sourceCheckpointSha256"],
        "stageCheckpointSha256": proof["stageCheckpointSha256"],
        "bundleManifestSha256": bundle["bundleManifestSha256"],
        "baselineValidationLoss": EXPECTED_BASELINE_LOSS,
        "baselineTolerance": 0.02,
        "sampler": sampler,
        "maximumOptimizerUpdatesIfAuthorized": MAX_STEPS,
        "maximumWallTimeSecondsIfAuthorized": MAX_SECONDS,
        **{k: v for k, v in NO_TRAINING.items() if k != "modelTrainingAuthorized"},
    }


def _record_loss(
    model: Any, record: tuple[str, tuple[int, ...], tuple[int, ...]],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    _, tokens, mask = record
    inputs = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
    weights = torch.tensor([mask], dtype=torch.float32, device=device)
    _, loss = model(inputs, targets, target_weights=weights, loss_vocabulary_size=16384)
    if loss is None or not bool(torch.isfinite(loss.detach()).all()):
        raise RuntimeError("P3-02 nonfinite or missing answer-only model loss")
    return loss, sum(mask)


def _validation_loss(model: Any, validation: list, device: torch.device) -> float:
    model.eval()
    total, positions = 0.0, 0
    with torch.inference_mode():
        for record in validation:
            loss, count = _record_loss(model, record, device)
            total += loss.item() * count
            positions += count
    model.train()
    if positions != 1320:
        raise ValueError("P3-02 validation did not cover all 1320 targets")
    return total / positions


def _bounded_training_step(
    model: Any, optimizer: Any, records: list, rng: random.Random,
    device: torch.device, deadline: float,
) -> tuple[float, int] | None:
    """One update, 16 independent complete examples; discard partial grads at timeout."""
    optimizer.zero_grad(set_to_none=True)
    mean_loss, target_positions = 0.0, 0
    for _ in range(ACCUMULATION):
        if time.perf_counter() >= deadline:
            optimizer.zero_grad(set_to_none=True)
            return None
        idx = rng.randrange(len(records))
        loss, count = _record_loss(model, records[idx], device)
        (loss / ACCUMULATION).backward()
        mean_loss += float(loss.detach().item()) / ACCUMULATION
        target_positions += count
    if time.perf_counter() >= deadline:
        optimizer.zero_grad(set_to_none=True)
        return None
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                               error_if_nonfinite=True)
    if not bool(torch.isfinite(grad_norm).all()):
        raise RuntimeError("P3-02 gradient norm is not finite")
    optimizer.step()
    return mean_loss, target_positions


def _append_metrics(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_replacement_first_training(
    *, authorize_first_run: bool = False, bundle_dir: Path = DEFAULT_BUNDLE,
    stage_checkpoint: Path = DEFAULT_STAGE, source_checkpoint: Path = DEFAULT_CHECKPOINT,
    result_path: Path = DEFAULT_P248_RESULT, baseline_path: Path = DEFAULT_BASELINE,
    authorization_path: Path = DEFAULT_AUTHORIZATION,
    output_dir: Path = DEFAULT_OUTPUT, artifact_root: Path = ROOT / "training/artifacts",
) -> dict[str, Any]:
    """One real run only after *both* later owner approval and explicit CLI opt-in."""
    # Nothing from the model, dataset or CUDA is touched until both gates pass.
    if authorize_first_run is not True:
        raise ValueError("P3-02 training requires explicit --authorize-first-run")
    packet = _bounded_json(authorization_path)
    validate_authorization(packet, allow_draft=False)
    root = artifact_root.resolve()
    output = path_within_root(output_dir, root)
    if output != root / packet["outputDirectory"] or output.exists():
        raise ValueError("P3-02 first-run output must be the exact unused authorized directory")
    if any(output == p.resolve(strict=False) or output in p.resolve(strict=False).parents
           for p in (bundle_dir, stage_checkpoint, source_checkpoint, result_path,
                     authorization_path, baseline_path)):
        raise ValueError("P3-02 training output aliases a protected input")
    if not torch.cuda.is_available():
        raise ValueError("P3-02 first run requires CUDA; no silent CPU fallback")
    proof, bundle = _verified_packet(
        bundle_dir, stage_checkpoint, source_checkpoint, result_path,
        baseline_path, require_baseline=True, device="cpu",
    )
    train = _packed_records(bundle_dir, "train")
    val = _packed_records(bundle_dir, "validation")
    _replay_positions(train)
    if sum(sum(x[2]) for x in val) != 1320:
        raise ValueError("P3-02 validation target count changed")
    # Reserve four new ~330MiB model+optimizer checkpoints, metrics, and a safety margin.
    enforce_storage_limit(root, additional_bytes=2 * 1024**3)
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng = random.Random(SEED)
    model, stage_payload = read_checkpoint(stage_checkpoint, torch.device("cpu"))
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PROPOSED_RUN["learningRate"],
        betas=ADAMW_BETAS, eps=ADAMW_EPSILON, weight_decay=ADAMW_WEIGHT_DECAY,
    )
    if optimizer.state:
        raise ValueError("P3-02 first-run optimizer must start empty")
    model.train()
    start_sha = sha256_file(stage_checkpoint)
    if start_sha != EXPECTED_STAGE_SHA256:
        raise ValueError("P3-02 stage changed before run output was created")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    metrics_path = output / "metrics.jsonl"
    settings = {
        "schemaVersion": 1, "kind": "p3-02-authorized-answer-eos-training-v1",
        "authorizationContractSha256": sha256_file(authorization_path),
        "stageCheckpointSha256": start_sha,
        **_expected_training(),
    }
    baseline_value = EXPECTED_BASELINE_LOSS
    results = [{"step": 0, "meanValidationAnswerAndEosLoss": baseline_value}]
    checkpoints = []
    steps = positions = 0
    start = time.perf_counter()
    deadline = start + MAX_SECONDS
    interrupted = False
    _append_metrics(metrics_path, {
        "event": "started", "atUtc": datetime.now(timezone.utc).isoformat(),
        "stageCheckpointSha256": start_sha, "bundleManifestSha256": BUNDLE_SHA256,
        "authorizationContractSha256": settings["authorizationContractSha256"],
        "baselineValidationLoss": baseline_value,
    })
    try:
        while steps < MAX_STEPS and time.perf_counter() < deadline:
            update = _bounded_training_step(model, optimizer, train, rng, device, deadline)
            if update is None:
                break
            loss, target_positions = update
            steps += 1
            positions += target_positions
            _append_metrics(metrics_path, {
                "event": "update", "step": steps, "meanTrainLoss": loss,
                "answerAndEosTargetsProcessed": positions,
            })
            if steps in VALIDATION_STEPS:
                measured = _validation_loss(model, val, device)
                if not math.isfinite(measured):
                    raise RuntimeError("P3-02 validation loss is nonfinite")
                results.append({"step": steps, "meanValidationAnswerAndEosLoss": measured})
                _append_metrics(metrics_path, {"event": "validation", "step": steps,
                                               "meanValidationAnswerAndEosLoss": measured})
            if steps in CHECKPOINT_STEPS:
                destination = checkpoint_dir / f"step-{steps:04d}.pt"
                saved = save_checkpoint(
                    model, optimizer, step=steps, seed=SEED, codec=CODEC,
                    sampling_rng=rng, device=device,
                    destination=destination, artifact_root=root,
                    initialization_record=stage_payload["initializationRecord"],
                    stage_transition_record=stage_payload["stageTransitionRecord"],
                    tokenizer_record=stage_payload["tokenizerRecord"],
                    dataset_record=stage_payload["datasetRecord"],
                    training_settings=settings,
                    schedule_state={
                        "kind": "constant-v1", "step": steps,
                        "learningRate": PROPOSED_RUN["learningRate"],
                    }, tokens_processed_total=positions,
                )
                saved["sha256"] = sha256_file(destination)
                checkpoints.append(saved)
                _append_metrics(metrics_path, {"event": "checkpoint", **saved})
    except KeyboardInterrupt:
        interrupted = True
    finally:
        optimizer.zero_grad(set_to_none=True)
    if steps == MAX_STEPS and positions != EXPECTED_SAMPLER_POSITIONS:
        raise ValueError("P3-02 actual sampler progress differs from frozen 1600-example replay")
    if sha256_file(stage_checkpoint) != EXPECTED_STAGE_SHA256:
        raise ValueError("P3-02 source stage was modified during training")
    report = {
        "schemaVersion": 1, "milestone": MILESTONE,
        "kind": "p3-02-first-bounded-answer-eos-run-result-v1",
        "trainingPerformed": steps > 0, "researchOptimizerUpdates": steps,
        "maximumOptimizerUpdates": MAX_STEPS, "maximumWallTimeSeconds": MAX_SECONDS,
        "completedMaximumSteps": steps == MAX_STEPS,
        "interrupted": interrupted,
        "elapsedSeconds": time.perf_counter() - start,
        "answerAndEosTargetsProcessed": positions,
        "stageCheckpointSha256": start_sha,
        "bundleManifestSha256": BUNDLE_SHA256,
        "authorizationContractSha256": settings["authorizationContractSha256"],
        "validation": results, "checkpoints": checkpoints,
        "fixedEndpointStep": steps,
        "fixedEndpointSha256": checkpoints[-1]["sha256"] if steps == MAX_STEPS else None,
        "resumeAllowed": False, "automaticContinuation": False,
        "continuationAuthorized": False, "finalHoldoutOpened": False,
        "developmentEvaluationPerformed": False,
    }
    # Never claim the final 100-step endpoint if the deadline/interrupt ended early.
    with (output / "result.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report
