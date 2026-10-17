"""P3 copy-fidelity first-run preparation: no model training or checkpoint writes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parents[1] / "phase3" / "copy-fidelity-v1"
SPEC = importlib.util.spec_from_file_location("cf_first_run_tests", HERE / "first_run.py")
assert SPEC and SPEC.loader
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_exact_owner_approval_is_pinned_and_not_inherited() -> None:
    assert RUN.APPROVED_CONTRACT_SHA256 == (
        "a7432a3bfa7942b6086bc8e64fa3b24290351acfd9fb700aac2af613a8c0911b"
    )
    assert RUN.APPROVED_BASELINE_SHA256 == (
        "5183726a621fb82539137b009c1514b5ed59d750e3ef4ad4bcd71453e644ff01"
    )
    raw = RUN.APPROVAL.read_bytes()
    assert b"\r" not in raw
    assert hashlib.sha256(raw).hexdigest() == RUN.APPROVED_CONTRACT_SHA256
    packet, pinned = RUN._approved_packet()
    assert pinned == RUN.APPROVED_CONTRACT_SHA256
    assert packet["status"] == "owner-approved-one-run"
    assert packet["approvedDate"] == "2026-10-17"
    assert packet["approvedBy"] == "keepithandy"
    assert packet["modelTrainingAuthorized"] is True
    assert packet["baselineEvidenceSha256"] == RUN.APPROVED_BASELINE_SHA256
    assert packet["stageCheckpointSha256"] == RUN.STAGE_SHA256
    assert packet["samplerSha256"] == RUN.SAMPLER_SHA256
    assert packet["maximumOptimizerUpdates"] == 100
    assert packet["maximumWallTimeSeconds"] == 600
    assert packet["resumeAllowed"] is False
    assert packet["checkpointReselectionAllowed"] is False
    assert packet["finalHoldoutOpened"] is False
    assert packet["executionState"]["trainingPerformed"] is False
    assert packet["executionState"]["runAuthorized"] is True
    assert packet["executionState"]["oneRunOnly"] is True


def test_training_requires_explicit_flag_and_pinned_contract_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RUN, "preflight", lambda **_: pytest.fail("preflight read before authorization"))
    monkeypatch.setattr(RUN, "read_checkpoint", lambda *a: pytest.fail("model read before approval"))
    monkeypatch.setattr(RUN.torch.cuda, "is_available",
                        lambda: pytest.fail("cuda touched before authorization"))
    with pytest.raises(PermissionError, match="Explicit --authorize-first-run"):
        RUN.run_training()
    monkeypatch.setattr(RUN, "APPROVED_CONTRACT_SHA256", "0" * 64)
    with pytest.raises(PermissionError, match="Frozen owner approval"):
        RUN.run_training(authorize_first_run=True)
    with pytest.raises(PermissionError, match="Frozen owner approval"):
        RUN._approved_packet()


def test_exact_sampler_sha_uses_epoch_shuffles_and_never_selects_validation() -> None:
    draws, sha = RUN._draw_indices(288)
    assert len(draws) == 1600
    assert len(set(draws)) == 288
    assert all(0 <= x < 288 for x in draws)
    assert sha == "6d21424636cdb6b5912dacd1564f51b63cc3f9897ec13cb5235e99c17930e9f4"
    # Each full epoch visits all training rows exactly once.
    assert sorted(draws[:288]) == list(range(288))
    assert sorted(draws[288:576]) == list(range(288))
    for invalid in [0, 96, 287, 289]:
        with pytest.raises(ValueError, match="288-record"):
            RUN._draw_indices(invalid)


def test_real_bundle_sampler_target_count_must_match_40477() -> None:
    rows = [(f"train-{i}", (4, 5, 3), (0, 1)) for i in range(288)]
    with pytest.raises(ValueError, match="SHA/coverage/target"):
        RUN._draws(rows)


def _fixtures(monkeypatch: pytest.MonkeyPatch):
    training = [(f"train-{i}", (4, 5, 3), (0, 1)) for i in range(288)]
    ids = [r["id"] for r in RUN.PACK._candidate()["rows"] if r["split"] == "validation"]
    assert len(ids) == 96
    # Exact 2420 target positions across 96 validation records.
    validation = [
        (name, tuple([4] * (26 if i < 76 else 27) + [3]),
         tuple([0] + [1] * (25 if i < 76 else 26)))
        for i, name in enumerate(ids)
    ]
    assert sum(sum(mask) for _, _, mask in validation) == 2420
    monkeypatch.setattr(RUN.STAGE, "verify_stage", lambda **_: {
        "status": "copy-fidelity-zero-stage-verified",
        "stageCheckpointSha256": RUN.STAGE_SHA256,
        "baseWeightsPreservedExactly": True,
        "optimizerStateEmpty": True,
    })
    monkeypatch.setattr(RUN.PACK, "inspect", lambda *_: {
        "bundleManifestSha256": RUN.BUNDLE_SHA256,
        "candidateSha256": RUN.CANDIDATE_SHA256,
        "splits": {"train": {"answerAndEosTargetPositions": 7300},
                   "validation": {"answerAndEosTargetPositions": 2420}},
    })
    monkeypatch.setattr(RUN, "_rows",
                        lambda split, _: training if split == "train" else validation)
    monkeypatch.setattr(RUN, "_draws", lambda _: (list(range(288)) * 6,
                        {"drawIndexSha256": RUN.SAMPLER_SHA256,
                         "expectedAnswerAndEosTargetPositions": 40477,
                         "optimizerUpdatesPerformed": 0,
                         "samplingExecutionAuthorized": False}))
    return training, validation


def test_read_only_preflight_reports_no_optimizer_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    _fixtures(monkeypatch)
    result = RUN.preflight()
    assert result["status"] == "copy-fidelity-first-run-read-only-preflight-passed"
    assert result["baselineMeasured"] is False
    assert result["trainingAuthorized"] is False
    assert result["trainingPerformed"] is False
    assert result["optimizerUpdates"] == 0
    assert result["validationAnswerAndEosTargets"] == 2420
    assert result["stageCheckpointSha256"] == RUN.STAGE_SHA256


def test_preflight_rejects_swapped_source_stage_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    _fixtures(monkeypatch)
    monkeypatch.setattr(RUN.STAGE, "verify_stage", lambda **_: {
        "stageCheckpointSha256": "wrong",
        "baseWeightsPreservedExactly": True,
        "optimizerStateEmpty": True,
    })
    with pytest.raises(ValueError, match="changed after owner verification"):
        RUN.preflight()


def test_baseline_teacher_forcing_checks_all_targets_and_never_trains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, validation = _fixtures(monkeypatch)
    model = SimpleNamespace()
    class ConstantLoss:
        def eval(self):
            return self
        def to(self, device):
            return self
        def __call__(self, inputs, targets, *, target_weights, loss_vocabulary_size):
            assert inputs.shape == targets.shape == target_weights.shape
            assert loss_vocabulary_size == 16384
            return None, torch.tensor(2.5, dtype=torch.float32)
    monkeypatch.setattr(RUN, "read_checkpoint",
                        lambda *_: (ConstantLoss(), {"step": 0, "tokensProcessedTotal": 0}))
    monkeypatch.setattr(RUN, "parameter_count", lambda *_: 27566080)
    result = RUN.baseline(device_name="cpu")
    assert result["status"] == "copy-fidelity-step-zero-validation-baseline-measured-read-only"
    assert result["meanValidationAnswerAndEosLoss"] == pytest.approx(2.5)
    assert result["validationRecordsEvaluated"] == 96
    assert result["supervisedTargetsEvaluated"] == 2420
    assert all(result["perLanguage"][lang]["meanLoss"] == pytest.approx(2.5)
               for lang in ("html", "css", "javascript"))
    assert result["trainingPerformed"] is False
    assert result["optimizerUpdates"] == 0
    assert result["finalHoldoutOpened"] is False


def test_baseline_fails_for_missing_masked_targets(monkeypatch: pytest.MonkeyPatch):
    _, validation = _fixtures(monkeypatch)
    class ConstantLoss:
        def eval(self):
            return self
        def __call__(self, *args, **kwargs):
            return None, torch.tensor(1.0)
    with pytest.raises(ValueError, match="2,420"):
        RUN._validation_loss(ConstantLoss(), validation[:-1], torch.device("cpu"))


def test_cli_exposes_safe_separate_read_only_actions():
    parser_script = (HERE / "first_run.py").read_text(encoding="utf-8")
    assert 'action.add_argument("--baseline"' in parser_script
    assert 'action.add_argument("--authorize-first-run"' in parser_script
    assert "APPROVED_CONTRACT_SHA256: str | None = " in parser_script
    assert "APPROVED_BASELINE_SHA256: str | None = " in parser_script
    assert "if authorize_first_run is not True:" in parser_script
    assert "packet, authorization_sha = _approved_packet()" in parser_script
    assert "finalHoldoutOpened" in parser_script
    assert "optimizer.step()" in parser_script


def test_training_step_discards_partial_gradients_on_timeout():
    class FakeOptimizer:
        def __init__(self): self.clears = 0
        def zero_grad(self, **kwargs): self.clears += 1
        def step(self): pytest.fail("optimizer must not step on timeout")
    optimizer = FakeOptimizer()
    result = RUN._training_step(model=None, optimizer=optimizer,
                                rows=[("a", (4,5,3), (0,1))],
                                selected=[0] * 16, device=torch.device("cpu"),
                                deadline=-1)
    assert result is None
    assert optimizer.clears == 2



def test_owner_reported_cuda_baseline_snapshot_is_frozen_and_not_training_approval() -> None:
    """GitHub checks owner-reported numbers; it cannot rerun the owner's checkpoint."""
    raw = RUN.BASELINE_PATH.read_bytes()
    digest = "5183726a621fb82539137b009c1514b5ed59d750e3ef4ad4bcd71453e644ff01"
    assert hashlib.sha256(raw).hexdigest() == digest
    assert b"\r" not in raw
    record = json.loads(raw)
    assert record["kind"] == "p3-copy-fidelity-owner-reported-step-zero-cuda-baseline-v1"
    assert "Owner-supplied" in record["source"]
    assert record["device"] == "cuda"
    assert record["status"] == "copy-fidelity-step-zero-validation-baseline-measured-read-only"
    assert record["bundleManifestSha256"] == RUN.BUNDLE_SHA256
    assert record["stageCheckpointSha256"] == RUN.STAGE_SHA256
    assert record["candidateSha256"] == RUN.CANDIDATE_SHA256
    assert record["sourceCheckpointSha256"] == RUN.STAGE.SOURCE_SHA256
    assert record["sampler"]["drawIndexSha256"] == RUN.SAMPLER_SHA256
    assert record["sampler"]["expectedAnswerAndEosTargetPositions"] == 40477
    assert record["validationRecordsEvaluated"] == 96
    assert record["supervisedTargetsEvaluated"] == 2420
    assert record["trainingAuthorized"] is False
    assert record["trainingPerformed"] is False
    assert record["optimizerUpdates"] == 0
    assert record["finalHoldoutOpened"] is False
    per = record["perLanguage"]
    assert {name: per[name]["supervisedTargets"] for name in per} == {
        "css": 800, "html": 980, "javascript": 640,
    }
    assert per["css"]["meanLoss"] == 7.546227005124092
    assert per["html"]["meanLoss"] == 6.3204158690510965
    assert per["javascript"]["meanLoss"] == 7.799167118966579
    weighted = sum(row["meanLoss"] * row["supervisedTargets"]
                   for row in per.values()) / 2420
    assert weighted == pytest.approx(7.116717401614859, abs=1e-12)
    assert record["meanValidationAnswerAndEosLoss"] == pytest.approx(weighted, abs=1e-12)
    packet = json.loads(RUN.APPROVAL.read_text(encoding="utf-8"))
    assert packet["baselineEvidenceSha256"] == digest
    assert packet["executionState"]["baselinePending"] is False
    assert packet["executionState"]["baselineRecordedByOwner"] is True
    assert packet["status"] == "owner-approved-one-run"
    assert packet["modelTrainingAuthorized"] is True
    assert RUN.APPROVED_CONTRACT_SHA256 == hashlib.sha256(RUN.APPROVAL.read_bytes()).hexdigest()
    assert RUN.APPROVED_BASELINE_SHA256 == digest


def test_one_run_refuses_preexisting_output_before_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    existing_output = artifacts / "previous-run"
    existing_output.mkdir()
    monkeypatch.setattr(RUN, "OUTPUT_DIR", existing_output)
    monkeypatch.setattr(RUN, "preflight", lambda **_: pytest.fail("preflight read before output check"))
    monkeypatch.setattr(RUN.torch.cuda, "is_available",
                        lambda: pytest.fail("cuda touched before output check"))
    with pytest.raises(ValueError, match="unused reviewed directory"):
        RUN.run_training(authorize_first_run=True, output_dir=existing_output,
                         artifact_root=artifacts)


def test_owner_approval_rejects_changed_baseline_sha_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RUN, "APPROVED_BASELINE_SHA256", "f" * 64)
    monkeypatch.setattr(RUN.torch.cuda, "is_available",
                        lambda: pytest.fail("cuda touched with invalid baseline"))
    with pytest.raises(PermissionError, match="Frozen owner approval"):
        RUN.run_training(authorize_first_run=True)


def test_training_compares_baseline_on_the_original_cuda_device() -> None:
    code = (HERE / "first_run.py").read_text(encoding="utf-8")
    assert 'observed = baseline(bundle_dir=bundle_dir, stage_checkpoint=stage_checkpoint,' in code
    assert 'device_name="cuda")' in code
    assert 'if abs(observed["meanValidationAnswerAndEosLoss"] -' in code
    assert "MAX_STEPS = 100" in code
    assert "MAX_SECONDS = 600" in code
    assert "GRAD_ACCUMULATION = 16" in code
    assert 'while steps < MAX_STEPS and time.perf_counter() < deadline' in code
    assert 'output.mkdir(exist_ok=False)' in code
    assert '"resumeAllowed": False' in code
    assert '"checkpointReselectionAllowed": False' in code



def test_authorized_runner_orchestration_uses_fixed_endpoint_without_real_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic only: cover checkpoint/report writing without training weights."""
    root = tmp_path / "artifacts"
    root.mkdir()
    output = root / "authorized-run"
    monkeypatch.setattr(RUN, "OUTPUT_DIR", output)
    monkeypatch.setattr(RUN.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(RUN.torch.cuda, "manual_seed_all", lambda *_: None)
    monkeypatch.setattr(RUN, "preflight", lambda **_: {
        "status": "copy-fidelity-first-run-read-only-preflight-passed"
    })
    training = [(f"t-{i}", (4, 5, 3), (0, 1)) for i in range(288)]
    validation = [("v", (4, 5, 3), (0, 1))] * 96
    monkeypatch.setattr(
        RUN, "_rows", lambda split, _: training if split == "train" else validation
    )
    monkeypatch.setattr(RUN, "_draws", lambda _: (
        list(range(288)) * 5 + list(range(160)),
        {"drawIndexSha256": RUN.SAMPLER_SHA256,
         "expectedAnswerAndEosTargetPositions": 40477}
    ))
    monkeypatch.setattr(RUN, "baseline", lambda **_: {
        "meanValidationAnswerAndEosLoss": 7.116717401614859
    })
    monkeypatch.setattr(RUN, "_validation_loss", lambda *args, **kwargs: {
        "meanValidationAnswerAndEosLoss": 1.5,
        "validationRecordsEvaluated": 96,
        "supervisedTargetsEvaluated": 2420,
    })
    monkeypatch.setattr(RUN, "enforce_storage_limit", lambda *a, **kw: None)
    monkeypatch.setattr(RUN, "sha256_file",
                        lambda path: (RUN.STAGE_SHA256
                                      if Path(path) == RUN.STAGE_CHECKPOINT
                                      else hashlib.sha256(Path(path).read_bytes()).hexdigest()))

    class FakeModel:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.zeros(1))
        def to(self, device):
            return self
        def parameters(self):
            return [self.parameter]
        def train(self):
            return self

    monkeypatch.setattr(RUN, "read_checkpoint", lambda *a: (
        FakeModel(), {
            "initializationRecord": {},
            "stageTransitionRecord": {},
            "tokenizerRecord": {},
            "datasetRecord": {},
        }
    ))
    updates = []
    def mock_update(model, optimizer, records, selected, device, deadline):
        assert device.type == "cuda"
        assert len(selected) == 16
        updates.append(tuple(selected))
        assert len(updates) <= 100
        return 0.5, (877 if len(updates) == 100 else 400)
    monkeypatch.setattr(RUN, "_training_step", mock_update)

    saved = []
    def fake_save(model, optimizer, **kwargs):
        checkpoint = kwargs["destination"]
        checkpoint.write_bytes(f"mock-checked-step-{kwargs['step']}".encode())
        saved.append(kwargs["step"])
        assert kwargs["training_settings"]["checkpointReselectionAllowed"] is False
        assert kwargs["tokens_processed_total"] <= 40477
        return {"path": checkpoint.name, "step": kwargs["step"],
                "bytes": checkpoint.stat().st_size}
    monkeypatch.setattr(RUN, "save_checkpoint", fake_save)

    result = RUN.run_training(authorize_first_run=True, output_dir=output,
                              artifact_root=root)
    assert result["status"] == "bounded-research-run-completed"
    assert result["optimizerUpdates"] == 100
    assert result["answerAndEosTargetsProcessed"] == 40477
    assert result["completedMaximumSteps"] is True
    assert [e["step"] for e in result["validation"]] == [0, 25, 50, 75, 100]
    assert saved == [25, 50, 75, 100]
    assert result["fixedEndpointSha256"] == hashlib.sha256(
        (output / "checkpoints/step-0100.pt").read_bytes()
    ).hexdigest()
    assert result["finalHoldoutOpened"] is False
    assert len(updates) == 100
    assert all(len(group) == 16 for group in updates)
    assert (output / "result.json").is_file()
    assert (output / "metrics.jsonl").is_file()

    with pytest.raises(ValueError, match="unused reviewed directory"):
        RUN.run_training(authorize_first_run=True, output_dir=output,
                         artifact_root=root)
