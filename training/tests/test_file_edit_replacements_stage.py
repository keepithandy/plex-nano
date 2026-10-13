"""P3-02 stage/validation-only baseline safety and deterministic fixture tests."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from plex_training import file_edit_replacements_stage as stage
from plex_training.cli import build_parser
from plex_training.file_edit_replacements import build_replacement_rows


def test_owner_reported_frozen_hashes_and_stage_guard_are_explicit():
    value = stage._read_evidence(stage.EVIDENCE_PATH)
    assert value["trainingBundleManifestSha256"] == stage.BUNDLE_SHA256
    assert value["sourceCheckpointSha256"] == stage.SOURCE_SHA256
    assert value["validationAnswerAndEosTargets"] == 1320
    assert value["checks"]["sourceCheckpointCompatibilityVerified"] is True
    assert value["executionState"]["modelTrainingAuthorized"] is False
    assert value["executionState"]["checkpointStaged"] is False
    assert stage._settings()["optimizerUpdatesAuthorized"] == 0
    assert stage._transition()["p302StageStep"] == 0


def test_stage_rejects_missing_explicit_authorization_without_source_access(tmp_path):
    with pytest.raises(ValueError, match="authorize-zero-stage"):
        stage.create_replacement_stage(
            output_dir=tmp_path / "stage", bundle_dir=tmp_path / "absent",
            artifact_root=tmp_path)
    assert not (tmp_path / "stage").exists()


def test_stage_creates_zero_update_checkpoint_without_stepping_and_refuses_overwrite(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    source = tmp_path / "source.pt"
    source.write_bytes(b"original")
    bundle = artifact_root / "bundle"
    bundle.mkdir()
    output = artifact_root / "stage0"
    result = tmp_path / "source-result.json"
    result.write_text("{}")
    frozen = b"original"
    fake_model = torch.nn.Linear(2, 2)
    fake_payload = {
        "initializationRecord": {"pretrainedCheckpointLoaded": False},
        "tokenizerRecord": {"tokenizerSha256": "frozen"},
    }
    monkeypatch.setattr(stage, "_verified_source", lambda **_: ({"bundleManifestSha256": stage.BUNDLE_SHA256}, {}))
    monkeypatch.setattr(stage, "read_checkpoint", lambda *_: (fake_model, fake_payload))
    monkeypatch.setattr(stage, "inspect_replacement_bundle", lambda *_: {
        "bundleManifestSha256": stage.BUNDLE_SHA256,
        "records": {"train": 144, "validation": 36},
        "answerAndEosTargetPositions": {"train": 5424, "validation": 1320},
    })
    saves = []
    def save_without_training(model, optimizer, **kwargs):
        assert optimizer.state == {}
        assert kwargs["step"] == kwargs["tokens_processed_total"] == 0
        assert kwargs["training_settings"]["modelTrainingAuthorized"] is False
        assert kwargs["stage_transition_record"]["baseOptimizerStateReused"] is False
        kwargs["destination"].write_bytes(b"stage-zero-fixture")
        saves.append(kwargs["destination"])
    monkeypatch.setattr(stage, "save_checkpoint", save_without_training)
    monkeypatch.setattr(stage, "preflight_replacement_stage", lambda **_: {
        "baseWeightsPreservedExactly": True, "stageCheckpointCreated": True,
        "modelTrainingAuthorized": False,
    })
    result = stage.create_replacement_stage(
        authorize_zero_stage=True, source_checkpoint=source, bundle_dir=bundle,
        output_dir=output, artifact_root=artifact_root, result_path=result)
    assert result["status"] == "zero-update-stage-created-and-verified"
    assert result["zeroStageExplicitlyAuthorized"] is True
    assert saves == [output / stage.STAGE_FILENAME]
    assert source.read_bytes() == frozen
    with pytest.raises(ValueError, match="fresh"):
        stage.create_replacement_stage(
            authorize_zero_stage=True, source_checkpoint=source, bundle_dir=bundle,
            output_dir=output, artifact_root=artifact_root, result_path=tmp_path / "source-result.json")


def test_failed_zero_stage_serialization_rolls_back_without_source_mutation(tmp_path, monkeypatch):
    artifact_root = tmp_path / "root"
    artifact_root.mkdir()
    bundle = artifact_root / "bundle"
    bundle.mkdir()
    source = tmp_path / "original.pt"
    source.write_bytes(b"original")
    result = tmp_path / "source.json"
    result.write_text("{}")
    monkeypatch.setattr(stage, "_verified_source", lambda **_: ({}, {}))
    monkeypatch.setattr(stage, "read_checkpoint", lambda *_: (
        torch.nn.Linear(2, 2), {"initializationRecord": {}, "tokenizerRecord": {}}
    ))
    monkeypatch.setattr(stage, "inspect_replacement_bundle", lambda *_: {
        "bundleManifestSha256": stage.BUNDLE_SHA256,
        "records": {"train": 144, "validation": 36},
        "answerAndEosTargetPositions": {"train": 5424, "validation": 1320},
    })
    def partial_write(*_, **kwargs):
        kwargs["destination"].write_bytes(b"partial")
        raise RuntimeError("synthetic serialization failure")
    monkeypatch.setattr(stage, "save_checkpoint", partial_write)
    with pytest.raises(RuntimeError, match="synthetic serialization failure"):
        stage.create_replacement_stage(
            authorize_zero_stage=True, source_checkpoint=source,
            bundle_dir=bundle, output_dir=artifact_root / "stage0",
            artifact_root=artifact_root, result_path=result)
    assert source.read_bytes() == b"original"
    assert not (artifact_root / "stage0").exists()
    assert bundle.exists()


def test_zero_stage_verifier_rejects_changed_weights(tmp_path, monkeypatch):
    original = tmp_path / "source.pt"
    staged = tmp_path / "stage.pt"
    original.write_bytes(b"original")
    staged.write_bytes(b"zero")
    monkeypatch.setattr(stage, "_verified_source", lambda **_: (
        {"bundleManifestSha256": stage.BUNDLE_SHA256},
        {"sourceModelParameters": 27566080},
    ))
    class Model:
        config = "same"
        def __init__(self, number): self.number = number
        def state_dict(self): return {"weight": torch.tensor([self.number])}
    models = iter([(Model(1), {}), (Model(2), {"step": 0})])
    monkeypatch.setattr(stage, "read_checkpoint", lambda *_: next(models))
    monkeypatch.setattr(stage, "_check_stage_payload", lambda *_: None)
    monkeypatch.setattr(stage, "parameter_count", lambda *_: 27566080)
    with pytest.raises(ValueError, match="tensor weights"):
        stage.preflight_replacement_stage(stage_checkpoint=staged, source_checkpoint=original)


def test_validation_baseline_is_masked_and_complete_without_training(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    rows = [row for row in build_replacement_rows() if row["candidateSplit"] == "validation"]
    assert len(rows) == 36
    tokens = []
    masks = bytearray()
    index = []
    for i, row in enumerate(rows):
        # Construct exactly 1320 supervised targets: 24 records x 37 + 12 x 36.
        answer_count = 37 if i < 24 else 36
        packed = [4] * (answer_count + 2)
        index.append({
            "recordId": row["id"], "startToken": len(tokens),
            "tokenCount": len(packed), "startTarget": len(masks),
            "targetCount": len(packed) - 1,
        })
        tokens.extend(packed)
        masks.extend(b"\x00" + b"\x01" * answer_count)
    (root / "validation.index.json").write_text(json.dumps(index))
    (root / "validation.tokens.u16le").write_bytes(
        struct.pack("<" + "H" * len(tokens), *tokens))
    (root / "validation.weights.u8").write_bytes(masks)
    monkeypatch.setattr(stage, "preflight_replacement_stage", lambda **_: {
        "stageCheckpointSha256": "fixture", "baseWeightsPreservedExactly": True})
    monkeypatch.setattr(stage, "inspect_replacement_bundle", lambda *_: {
        "bundleManifestSha256": stage.BUNDLE_SHA256,
        "recordTokens": {"validation": len(tokens)}})
    class ConstantLossModel:
        def to(self, *_): return self
        def eval(self): return self
        def __call__(self, inputs, targets, *, target_weights, loss_vocabulary_size):
            assert inputs.shape == targets.shape == target_weights.shape
            assert int(target_weights.sum()) > 0
            assert loss_vocabulary_size == 16384
            return None, torch.tensor(2.5)
    monkeypatch.setattr(stage, "read_checkpoint", lambda *_: (ConstantLossModel(), {}))
    result = stage.baseline_replacement_preflight(
        stage_checkpoint=tmp_path / "unused-stage.pt", bundle_dir=root)
    assert result["validationRecordsEvaluated"] == 36
    assert result["supervisedTargetsEvaluated"] == 1320
    assert result["meanValidationAnswerAndEosLoss"] == pytest.approx(2.5)
    assert all(result["perLanguage"][l]["meanLoss"] == pytest.approx(2.5)
               for l in ("html", "css", "javascript"))
    assert result["trainingPerformed"] is False
    assert result["baselineWrittenToDisk"] is False
    assert not any(p.name.startswith("development.") for p in root.iterdir())


def test_cli_exposes_separate_stage_and_baseline_without_training():
    parser = build_parser()
    stages = parser.parse_args(["file-edit-replacements-stage"])
    assert stages.authorize_zero_stage is False
    assert stages.source_checkpoint.is_absolute()
    assert stages.output_dir.is_absolute()
    assert stages.artifact_root.is_absolute()
    allowed = parser.parse_args(["file-edit-replacements-stage", "--authorize-zero-stage"])
    assert allowed.authorize_zero_stage is True
    baseline = parser.parse_args(["file-edit-replacements-baseline-preflight"])
    assert baseline.device == "cpu"
    assert baseline.stage_checkpoint.is_absolute()
    verification = parser.parse_args(["file-edit-replacements-stage-preflight"])
    assert verification.bundle_dir.is_absolute()
    for args in (stages, allowed, baseline, verification):
        assert not hasattr(args, "maximum_optimizer_updates")
