"""P3-02 zero-update stage and validation-only baseline preflight.

No train command, gradient step, development evaluator, or owner training approval.
A stage requires an explicit --authorize-zero-stage flag; baseline reads files only.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .artifacts import path_within_root
from .checkpoint import read_checkpoint, save_checkpoint
from .file_edit_replacements import (
    DEFAULT_CONTRACT, MILESTONE, PROPOSED_RUN, ROOT, SAFE_FLAGS,
    build_replacement_rows, render_prompt,
)
from .file_edit_replacements_bundle import (
    DEFAULT_CHECKPOINT, DEFAULT_OUTPUT as DEFAULT_BUNDLE, DEFAULT_P248_RESULT,
    inspect_replacement_bundle, preflight_replacement_checkpoint_source,
)
from .model import parameter_count
from .tokenizer import CODEC, sha256_file

EVIDENCE_PATH = ROOT / "training/pretraining/p3-02-local-verification-evidence.json"
DEFAULT_STAGE_DIR = ROOT / "training/artifacts/file-edits/p3-02-stage0"
STAGE_FILENAME = "stage-checkpoint.pt"
STAGE_KIND = "plex-p3-02-exact-replacements-stage-transition-v1"
STAGE_SETTINGS_KIND = "p3-02-zero-stage-serialization-only-v1"
BUNDLE_SHA256 = "0cd6df3a6d8eec1d7da8c7a02af067b15299393d8b0e1756a071be17b9aca354"
SOURCE_SHA256 = PROPOSED_RUN["sourceCheckpointSha256"]


def _read_evidence(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("P3-02 recorded local evidence must be a bounded regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("P3-02 evidence must be a JSON object")
    for field, expected in {
        "schemaVersion": 1,
        "milestone": MILESTONE,
        "candidateSha256": "98d7ff276cf28d2509ad775c95c49ae0ffdb975de7e47f11e54430f3ca426403",
        "tokenizerSha256": "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697",
        "trainingBundleManifestSha256": BUNDLE_SHA256,
        "sourceCheckpointSha256": SOURCE_SHA256,
        "sourceCheckpointStep": 100,
        "sourceModelParameters": 27566080,
        "sourceTokensProcessed": 429375,
        "trainRecords": 144,
        "validationRecords": 36,
        "trainRecordTokens": 16644,
        "validationRecordTokens": 4074,
        "trainAnswerAndEosTargets": 5424,
        "validationAnswerAndEosTargets": 1320,
    }.items():
        if type(value.get(field)) is not type(expected) or value[field] != expected:
            raise ValueError("P3-02 recorded local evidence mismatch: " + field)
    checks = value.get("checks")
    if not isinstance(checks, dict) or any(checks.get(k) is not True for k in (
            "answerOnlySupervisionPreflight", "packedBundleVerified",
            "sourceCheckpointCompatibilityVerified")):
        raise ValueError("P3-02 local preflight evidence is incomplete")
    return value


def _verified_source(
    *, bundle_dir: Path, source_checkpoint: Path, result_path: Path,
    contract_path: Path, evidence_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = _read_evidence(evidence_path)
    source = preflight_replacement_checkpoint_source(
        checkpoint_path=source_checkpoint, bundle_dir=bundle_dir,
        result_path=result_path, contract_path=contract_path,
    )
    if (source.get("status") != "checkpoint-source-read-only-preflight-passed"
            or source.get("baseCheckpointSha256") != SOURCE_SHA256
            or source.get("bundleManifestSha256") != BUNDLE_SHA256
            or source.get("bundleCompatibilityVerified") is not True
            or source.get("modelTrainingAuthorized") is not False):
        raise ValueError("P3-02 stage source differs from verified local checkpoint/bundle")
    return source, evidence


def _transition() -> dict[str, Any]:
    return {
        "schemaVersion": 1, "kind": STAGE_KIND, "milestone": MILESTONE,
        "baseMilestone": "P2-48", "baseCheckpointSha256": SOURCE_SHA256,
        "baseCheckpointStep": 100, "baseTokensProcessed": 429375,
        "bundleManifestSha256": BUNDLE_SHA256,
        "baseModelWeightsPreserved": True,
        "baseOptimizerStateReused": False, "baseSamplerStateReused": False,
        "baseTrainingStepReused": False,
        "p302StageStep": 0, "p302TokensProcessed": 0,
        "trainingPerformed": False, "researchOptimizerUpdates": 0,
        "finalHoldoutOpened": False,
    }


def _settings() -> dict[str, Any]:
    return {
        "kind": STAGE_SETTINGS_KIND,
        "purpose": "checkpoint serialization only, without optimizer stepping",
        "modelTrainingAuthorized": False,
        "optimizerUpdatesAuthorized": 0,
        "optimizerKind": "AdamW",
        "proposedLearningRate": PROPOSED_RUN["learningRate"],
        "microBatch": PROPOSED_RUN["microBatch"],
        "gradientAccumulation": PROPOSED_RUN["gradientAccumulation"],
        "objective": PROPOSED_RUN["objective"],
        "automaticContinuation": False,
    }


def _check_stage_payload(payload: dict[str, Any]) -> None:
    optimizer = payload.get("optimizerStateDict")
    tokenizer = payload.get("tokenizerRecord")
    dataset = payload.get("datasetRecord")
    if (type(payload.get("step")) is not int or payload["step"] != 0
            or type(payload.get("tokensProcessedTotal")) is not int
            or payload["tokensProcessedTotal"] != 0
            or payload.get("seed") != PROPOSED_RUN["seed"]
            or payload.get("codec") != CODEC
            or payload.get("stageTransitionRecord") != _transition()
            or payload.get("trainingSettings") != _settings()
            or payload.get("scheduleState") is not None
            or not isinstance(optimizer, dict)
            or optimizer.get("state") != {}
            or not isinstance(optimizer.get("param_groups"), list)
            or len(optimizer["param_groups"]) != 1
            or not isinstance(tokenizer, dict)
            or tokenizer.get("tokenizerSha256") != _read_evidence(EVIDENCE_PATH)["tokenizerSha256"]
            or not isinstance(dataset, dict)
            or dataset.get("bundleManifestSha256") != BUNDLE_SHA256
            or dataset.get("trainRecords") != 144
            or dataset.get("validationRecords") != 36):
        raise ValueError("P3-02 stage checkpoint is not an untouched zero-step stage")


def preflight_replacement_stage(
    *, stage_checkpoint: Path,
    source_checkpoint: Path = DEFAULT_CHECKPOINT,
    bundle_dir: Path = DEFAULT_BUNDLE,
    result_path: Path = DEFAULT_P248_RESULT,
    contract_path: Path = DEFAULT_CONTRACT,
    evidence_path: Path = EVIDENCE_PATH,
) -> dict[str, Any]:
    """Read-only source and frozen step-zero/weights comparison."""
    source, evidence = _verified_source(
        bundle_dir=bundle_dir, source_checkpoint=source_checkpoint,
        result_path=result_path, contract_path=contract_path, evidence_path=evidence_path,
    )
    if stage_checkpoint.is_symlink() or not stage_checkpoint.is_file():
        raise ValueError("P3-02 stage checkpoint must be a regular local file")
    source_model, _ = read_checkpoint(source_checkpoint, torch.device("cpu"))
    stage_model, stage_payload = read_checkpoint(stage_checkpoint, torch.device("cpu"))
    _check_stage_payload(stage_payload)
    source_state = source_model.state_dict()
    stage_state = stage_model.state_dict()
    if (source_model.config != stage_model.config
            or parameter_count(stage_model) != evidence["sourceModelParameters"]
            or source_state.keys() != stage_state.keys()
            or any(not torch.equal(source_state[k], stage_state[k]) for k in source_state)):
        raise ValueError("P3-02 stage did not preserve exact P2-48 model tensor weights")
    return {
        "schemaVersion": 1, "milestone": MILESTONE,
        "status": "read-only-stage-preflight-passed",
        "sourceCheckpointSha256": SOURCE_SHA256,
        "stageCheckpointSha256": sha256_file(stage_checkpoint),
        "bundleManifestSha256": source["bundleManifestSha256"],
        "baseWeightsPreservedExactly": True,
        "stageStep": 0, "stageTokensProcessed": 0,
        "optimizerStateEmpty": True, "optimizerCreatedForSerializationOnly": True,
        "stageCheckpointCreated": True,
        **SAFE_FLAGS,
    }


def create_replacement_stage(
    *, authorize_zero_stage: bool = False,
    source_checkpoint: Path = DEFAULT_CHECKPOINT,
    bundle_dir: Path = DEFAULT_BUNDLE,
    output_dir: Path = DEFAULT_STAGE_DIR,
    artifact_root: Path = ROOT / "training/artifacts",
    result_path: Path = DEFAULT_P248_RESULT,
    contract_path: Path = DEFAULT_CONTRACT,
    evidence_path: Path = EVIDENCE_PATH,
) -> dict[str, Any]:
    """Explicitly consented one-time serialization of unchanged model plus fresh empty optimizer."""
    if authorize_zero_stage is not True:
        raise ValueError("P3-02 zero-step staging needs explicit --authorize-zero-stage")
    root = artifact_root.resolve()
    output = path_within_root(output_dir, root)
    bundle = bundle_dir.resolve(strict=True)
    base = source_checkpoint.resolve(strict=True)
    if (output == root or output.exists() or output == bundle
            or output in bundle.parents or bundle in output.parents
            or output == base or output in base.parents
            or output in result_path.resolve().parents):
        raise ValueError("P3-02 stage output must be fresh and separate from all source artifacts")
    _verified_source(
        bundle_dir=bundle_dir, source_checkpoint=source_checkpoint,
        result_path=result_path, contract_path=contract_path, evidence_path=evidence_path,
    )
    model, payload = read_checkpoint(source_checkpoint, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROPOSED_RUN["learningRate"])
    if optimizer.state:
        raise RuntimeError("P3-02 fresh zero-stage optimizer must be empty")
    dataset = inspect_replacement_bundle(bundle_dir)
    if dataset["bundleManifestSha256"] != BUNDLE_SHA256:
        raise ValueError("P3-02 source bundle changed before staging")
    record = {
        "kind": "p3-02-answer-eos-verified-train-val-v1",
        "bundleManifestSha256": BUNDLE_SHA256,
        "trainRecords": dataset["records"]["train"],
        "validationRecords": dataset["records"]["validation"],
        "trainAnswerAndEosTargets": dataset["answerAndEosTargetPositions"]["train"],
        "validationAnswerAndEosTargets": dataset["answerAndEosTargetPositions"]["validation"],
        "developmentInTrainingBundle": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    checkpoint = output / STAGE_FILENAME
    try:
        save_checkpoint(
            model, optimizer, step=0, seed=PROPOSED_RUN["seed"], codec=CODEC,
            sampling_rng=random.Random(PROPOSED_RUN["seed"]),
            device=torch.device("cpu"), destination=checkpoint, artifact_root=root,
            initialization_record=payload["initializationRecord"],
            stage_transition_record=_transition(),
            tokenizer_record=payload["tokenizerRecord"],
            dataset_record=record, training_settings=_settings(),
            tokens_processed_total=0,
        )
        proof = preflight_replacement_stage(
            stage_checkpoint=checkpoint, source_checkpoint=source_checkpoint,
            bundle_dir=bundle_dir, result_path=result_path, contract_path=contract_path,
            evidence_path=evidence_path,
        )
        if not proof["baseWeightsPreservedExactly"]:
            raise ValueError("P3-02 stage weights differ after serialization")
    except Exception:
        shutil.rmtree(output)
        raise
    return {"status": "zero-update-stage-created-and-verified",
            "stageDirectory": str(output),
            **proof}


def baseline_replacement_preflight(
    *, stage_checkpoint: Path,
    source_checkpoint: Path = DEFAULT_CHECKPOINT,
    bundle_dir: Path = DEFAULT_BUNDLE,
    result_path: Path = DEFAULT_P248_RESULT,
    contract_path: Path = DEFAULT_CONTRACT,
    evidence_path: Path = EVIDENCE_PATH,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate exactly 36 frozen validation records with answer/EOS-only weighting."""
    proof = preflight_replacement_stage(
        stage_checkpoint=stage_checkpoint, source_checkpoint=source_checkpoint,
        bundle_dir=bundle_dir, result_path=result_path, contract_path=contract_path,
        evidence_path=evidence_path,
    )
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("P3-02 baseline device must be an available CPU or CUDA device")
    model, _ = read_checkpoint(stage_checkpoint, torch.device("cpu"))
    model = model.to(torch.device(device))
    model.eval()
    index_path = bundle_dir / "validation.index.json"
    tokens_path = bundle_dir / "validation.tokens.u16le"
    weights_path = bundle_dir / "validation.weights.u8"
    if any(p.is_symlink() or not p.is_file() for p in (index_path, tokens_path, weights_path)):
        raise ValueError("P3-02 validation files must be regular files")
    # inspect_replacement_bundle regenerates and verifies every byte before evaluation.
    inspected = inspect_replacement_bundle(bundle_dir)
    if inspected["bundleManifestSha256"] != BUNDLE_SHA256:
        raise ValueError("P3-02 baseline requires the exact frozen local validation bundle")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    packed = tokens_path.read_bytes()
    masks = weights_path.read_bytes()
    languages = {row["id"]: row["language"] for row in build_replacement_rows()
                 if row["candidateSplit"] == "validation"}
    if (not isinstance(index, list) or len(index) != 36 or len(packed) % 2 != 0
            or len(masks) != inspected["recordTokens"]["validation"] - 36):
        raise ValueError("P3-02 baseline validation record/index dimensions differ")
    ids = struct.unpack("<" + "H" * (len(packed) // 2), packed)
    sums = Counter()
    counts = Counter()
    evaluated: list[dict[str, Any]] = []
    with torch.inference_mode():
        for entry in index:
            start, length = entry["startToken"], entry["tokenCount"]
            target_start, target_count = entry["startTarget"], entry["targetCount"]
            record_tokens = ids[start:start + length]
            mask = masks[target_start:target_start + target_count]
            if (len(record_tokens) != length or len(mask) != length - 1
                    or any(m not in (0, 1) for m in mask) or sum(mask) < 1):
                raise ValueError("P3-02 validation record has an invalid supervised mask")
            inputs = torch.tensor([record_tokens[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([record_tokens[1:]], dtype=torch.long, device=device)
            weights = torch.tensor([mask], dtype=torch.float32, device=device)
            _, loss = model(
                inputs, targets,
                target_weights=weights,
                loss_vocabulary_size=PROPOSED_RUN["vocabularySize"],
            )
            if loss is None or not math.isfinite(loss.item()):
                raise ValueError("P3-02 baseline measured a nonfinite validation loss")
            language = languages[entry["recordId"]]
            supervised = sum(mask)
            sums[language] += float(loss.item()) * supervised
            counts[language] += supervised
            evaluated.append({"recordId": entry["recordId"],
                              "language": language,
                              "supervisedTargets": supervised,
                              "meanAnswerAndEosLoss": float(loss.item())})
    if (len(evaluated) != 36 or sum(counts.values()) != 1320
            or any(counts[lang] == 0 for lang in ("html", "css", "javascript"))
            or len({item["recordId"] for item in evaluated}) != 36):
        raise ValueError("P3-02 baseline did not evaluate every validation record")
    return {
        "schemaVersion": 1, "milestone": MILESTONE,
        "status": "answer-only-step-zero-baseline-measured-no-training",
        "metric": "validation-answer-and-eos-cross-entropy-per-supervised-target",
        "validationRecordsEvaluated": len(evaluated),
        "supervisedTargetsEvaluated": sum(counts.values()),
        "meanValidationAnswerAndEosLoss": sum(sums.values()) / sum(counts.values()),
        "perLanguage": {lang: {"supervisedTargets": counts[lang],
                               "meanLoss": sums[lang] / counts[lang]}
                        for lang in ("html", "css", "javascript")},
        "perRecord": evaluated,
        "checkpointSha256": proof["stageCheckpointSha256"],
        "bundleManifestSha256": BUNDLE_SHA256,
        "modelWeightsPreservedExactly": True,
        "baselineWrittenToDisk": False,
        **SAFE_FLAGS,
    }
