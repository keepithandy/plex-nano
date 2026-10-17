"""Model-free P3 copy-fidelity bundle/stage integrity and refusal tests."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parents[1] / "phase3" / "copy-fidelity-v1"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PACK = _load("p3_cf_test_pack", "pack_bundle.py")
STAGE = _load("p3_cf_test_stage", "zero_stage.py")


class ByteTokenizer:
    vocabulary_size = 16384

    def encode(self, value: str) -> list[int]:
        return [int(b) + 4 for b in value.encode("ascii")]

    def decode(self, values: list[int]) -> str:
        return bytes(int(v) - 4 for v in values).decode("ascii")


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tokenizer = ByteTokenizer()
    candidate = PACK._candidate()
    targets = {
        split: sum(len(r["expectedFile"]) + 1 for r in candidate["rows"] if r["split"] == split)
        for split in PACK.SPLITS
    }
    monkeypatch.setattr(PACK, "EXPECTED_TARGET_POSITIONS", targets)
    root = tmp_path / "artifacts"
    source = root / "source"
    source.mkdir(parents=True)
    for filename in PACK.TOKENIZER_FILES:
        (source / filename).write_bytes(b"{}\n")
    model_config_sha = hashlib.sha256(b"{}\n").hexdigest()

    def local_tokenizer(folder):
        identity = {
            "tokenizerSha256": PACK.TOKENIZER_SHA,
            "modelConfigSha256": model_config_sha,
            "bundleManifestSha256": (
                PACK.SOURCE_BUNDLE_SHA if Path(folder) == source
                else hashlib.sha256((Path(folder) / "manifest.json").read_bytes()).hexdigest()
            ),
        }
        return tokenizer, identity

    monkeypatch.setattr(PACK, "_completion_tokenizer", local_tokenizer)
    output = root / "bundle"
    result = PACK.prepare(
        output_dir=output, source_bundle=source, artifact_root=root,
    )
    return {"root": root, "bundle": output, "source": source,
            "result": result, "targets": targets, "tokenizer": tokenizer}


def test_packed_bundle_verifies_source_text_tokens_masks_and_sampler(prepared):
    result = prepared["result"]
    assert result["bundleVerified"] is True
    assert result["trainingPerformed"] is False
    assert result["optimizerUpdates"] == 0
    bundle = prepared["bundle"]
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["trainingAuthorized"] is False
    assert manifest["optimizerUpdatesPerformed"] == 0
    assert manifest["proposalSampler"]["distinctRecords"] == 288
    assert manifest["proposalSampler"]["draws"] == 1600
    assert manifest["proposalSampler"]["samplingExecutionAuthorized"] is False
    assert manifest["splits"]["train"]["answerAndEosTargetPositions"] == prepared["targets"]["train"]
    assert manifest["splits"]["validation"]["answerAndEosTargetPositions"] == prepared["targets"]["validation"]
    for split, count in (("train", 288), ("validation", 96)):
        entries = json.loads((bundle / f"{split}.index.json").read_text(encoding="utf-8"))
        raw_mask = (bundle / f"{split}.weights.u8").read_bytes()
        raw_tokens = (bundle / f"{split}.tokens.u16le").read_bytes()
        assert len(entries) == count
        assert len(raw_mask) == len(raw_tokens) // 2 - count
        for entry in entries:
            begin, length = entry["startTarget"], entry["targetCount"]
            mask = raw_mask[begin:begin + length]
            assert mask == b"\x00" * (entry["promptTokens"] - 1) + b"\x01" * entry["answerTokensIncludingEos"]
            assert sum(mask) == entry["answerTokensIncludingEos"]
    assert PACK.inspect(bundle)["bundleManifestSha256"] == result["bundleManifestSha256"]


@pytest.mark.parametrize("suffix", PACK.PARTS)
def test_inspector_rejects_tampered_train_files(prepared, suffix):
    path = prepared["bundle"] / f"train.{suffix}"
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError):
        PACK.inspect(prepared["bundle"])


def test_inspector_rejects_bundled_development_material(prepared):
    (prepared["bundle"] / "development.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="unexpected"):
        PACK.inspect(prepared["bundle"])


def test_pack_never_overwrites_and_does_not_alias_source(prepared):
    original = (prepared["bundle"] / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="new"):
        PACK.prepare(output_dir=prepared["bundle"], source_bundle=prepared["source"],
                     artifact_root=prepared["root"])
    with pytest.raises(ValueError, match="new"):
        PACK.prepare(output_dir=prepared["source"], source_bundle=prepared["source"],
                     artifact_root=prepared["root"])
    assert (prepared["bundle"] / "manifest.json").read_bytes() == original


def test_pack_low_storage_rolls_back_without_touching_source(prepared):
    root = prepared["root"]
    output = root / "low-storage"
    original = (prepared["source"] / "tokenizer.json").read_bytes()
    with pytest.raises(RuntimeError, match="allocation exceeded"):
        PACK.prepare(output_dir=output, source_bundle=prepared["source"],
                     artifact_root=root, storage_limit_bytes=1)
    assert not output.exists()
    assert not list(root.glob(".copy-fidelity-bundle-*"))
    assert (prepared["source"] / "tokenizer.json").read_bytes() == original


def test_no_stage_without_explicit_opt_in(tmp_path: Path):
    with pytest.raises(ValueError, match="authorize-zero-stage"):
        STAGE.create_stage(
            output_dir=tmp_path / "stage0",
            source_checkpoint=tmp_path / "missing.pt",
            bundle_dir=tmp_path / "missing-bundle",
            artifact_root=tmp_path,
        )
    assert not (tmp_path / "stage0").exists()


def _mock_stage_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    bundle = root / "bundle"
    bundle.mkdir()
    source = tmp_path / "source.pt"
    source.write_bytes(b"original")
    result = tmp_path / "result.json"
    result.write_text("{}")
    model = torch.nn.Linear(2, 2)
    payload = {"initializationRecord": {}, "tokenizerRecord": {"tokenizerSha256": STAGE.TOKENIZER_SHA}}
    proof = {
        "bundleManifestSha256": "fixture-bundle-sha",
        "sourceCheckpointSha256": STAGE.SOURCE_SHA256,
        "sourceWeightsVerified": True,
    }
    data = {
        "bundleManifestSha256": "fixture-bundle-sha",
        "candidateSha256": STAGE.CANDIDATE_SHA,
        "bundleVerified": True,
        "splits": {
            "train": {"records": 288, "answerAndEosTargetPositions": 7300},
            "validation": {"records": 96, "answerAndEosTargetPositions": 2420},
        },
    }
    monkeypatch.setattr(STAGE, "source_preflight", lambda **kwargs: proof)
    monkeypatch.setattr(STAGE.PACK, "inspect", lambda _: data)
    monkeypatch.setattr(STAGE, "read_checkpoint", lambda *_: (model, payload))
    return root, bundle, source, result, proof, model, payload


def test_zero_stage_is_serialization_only_and_preserves_source(tmp_path, monkeypatch):
    root, bundle, source, result, proof, model, payload = _mock_stage_sources(tmp_path, monkeypatch)
    staged = root / "stage0"
    saved = []

    def fake_save(model_arg, optimizer, **kwargs):
        assert model_arg is model
        assert optimizer.state == {}
        assert kwargs["step"] == 0
        assert kwargs["tokens_processed_total"] == 0
        assert kwargs["training_settings"]["trainingAuthorized"] is False
        assert kwargs["stage_transition_record"]["baseOptimizerStateReused"] is False
        assert kwargs["dataset_record"]["developmentInTrainingBundle"] is False
        kwargs["destination"].write_bytes(b"synthetic-zero-stage")
        saved.append(kwargs["destination"])

    monkeypatch.setattr(STAGE, "save_checkpoint", fake_save)
    monkeypatch.setattr(STAGE, "verify_stage", lambda **kwargs: {
        "status": "copy-fidelity-zero-stage-verified",
        "baseWeightsPreservedExactly": True,
    })
    output = STAGE.create_stage(
        authorize_zero_stage=True, source_checkpoint=source, source_result=result,
        bundle_dir=bundle, output_dir=staged, artifact_root=root,
    )
    assert output["status"] == "copy-fidelity-zero-stage-created-and-verified"
    assert saved == [staged / STAGE.STAGE_NAME]
    assert source.read_bytes() == b"original"
    assert output["optimizerUpdates"] == 0
    with pytest.raises(ValueError, match="fresh"):
        STAGE.create_stage(
            authorize_zero_stage=True, source_checkpoint=source, source_result=result,
            bundle_dir=bundle, output_dir=staged, artifact_root=root,
        )


def test_stage_serialization_failure_rolls_back(tmp_path, monkeypatch):
    root, bundle, source, result, *_ = _mock_stage_sources(tmp_path, monkeypatch)
    staged = root / "failed-stage"

    def fail(*args, **kwargs):
        kwargs["destination"].write_bytes(b"partial")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(STAGE, "save_checkpoint", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        STAGE.create_stage(
            authorize_zero_stage=True, source_checkpoint=source, source_result=result,
            bundle_dir=bundle, output_dir=staged, artifact_root=root,
        )
    assert not staged.exists()
    assert source.read_bytes() == b"original"


def test_verifier_rejects_mutated_tensor_weights(tmp_path, monkeypatch):
    root, bundle, source, result, proof, model, payload = _mock_stage_sources(tmp_path, monkeypatch)
    original = torch.nn.Linear(2, 2)
    changed = torch.nn.Linear(2, 2)
    with torch.no_grad():
        original.weight.fill_(1)
        changed.weight.fill_(2)
    original.config = changed.config = SimpleNamespace(context_length=512, vocab_size=16384)
    staged = root / STAGE.STAGE_NAME
    staged.write_bytes(b"dummy")
    metadata = {
        "step": 0, "tokensProcessedTotal": 0, "seed": 1337,
        "codec": STAGE.CODEC, "trainingSettings": STAGE._settings(),
        "stageTransitionRecord": STAGE._transition(proof["bundleManifestSha256"]),
        "datasetRecord": STAGE._dataset_record(STAGE.PACK.inspect(bundle)),
        "scheduleState": None, "tokenizerRecord": {"tokenizerSha256": STAGE.TOKENIZER_SHA},
        "optimizerStateDict": {"state": {}, "param_groups": [{}]},
    }
    loads = iter([(original, {}), (changed, metadata)])
    monkeypatch.setattr(STAGE, "read_checkpoint", lambda *args: next(loads))
    monkeypatch.setattr(STAGE, "parameter_count", lambda model: 27566080)
    with pytest.raises(ValueError, match="tensor weights"):
        STAGE.verify_stage(checkpoint=staged, source_checkpoint=source,
                           source_result=result, bundle_dir=bundle)



def test_source_preflight_accepts_p248_tokenizer_and_p302_dataset_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Reproduce original P3-02 step-100 metadata without private weights."""
    assert STAGE.ORIGINAL_P248_TOKENIZER_BUNDLE_SHA == (
        "71fec0882692f5eb1b47b72f4a9f539e53e77882d003649beacd5f341b5a4ea7"
    )
    assert STAGE.ORIGINAL_P248_TOKENIZER_BUNDLE_SHA != STAGE.ORIGINAL_P302_BUNDLE_SHA
    checkpoint = tmp_path / "step-0100.pt"
    checkpoint.write_bytes(b"synthetic-frozen-step100")
    monkeypatch.setattr(STAGE, "SOURCE_SIZE", checkpoint.stat().st_size)
    monkeypatch.setattr(STAGE, "SOURCE_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    monkeypatch.setattr(STAGE, "_frozen_run_report", lambda **_: {})
    monkeypatch.setattr(STAGE.PACK, "inspect", lambda _: {
        "candidateSha256": STAGE.CANDIDATE_SHA,
        "bundleVerified": True,
        "bundleManifestSha256": "synthetic-copy-fidelity-bundle",
    })
    model = SimpleNamespace(config=SimpleNamespace(context_length=512, vocab_size=16384))
    monkeypatch.setattr(STAGE, "parameter_count", lambda model: 27566080)
    payload = {
        "step": 100, "codec": STAGE.CODEC, "initializationRecord": {},
        "tokenizerRecord": {
            "tokenizerSha256": STAGE.TOKENIZER_SHA,
            "bundleManifestSha256": STAGE.ORIGINAL_P248_TOKENIZER_BUNDLE_SHA,
        },
        "datasetRecord": {
            "bundleManifestSha256": STAGE.ORIGINAL_P302_BUNDLE_SHA,
        },
    }
    monkeypatch.setattr(STAGE, "read_checkpoint", lambda *_: (model, payload))
    kwargs = {
        "source_checkpoint": checkpoint,
        "source_result": tmp_path / "result.json",
        "bundle_dir": tmp_path / "copy-fidelity-bundle",
    }
    result = STAGE.source_preflight(**kwargs)
    assert result["status"] == "copy-fidelity-source-stage-read-only-preflight-passed"
    assert result["sourceWeightsVerified"] is True
    assert result["optimizerCreated"] is False
    assert result["optimizerUpdates"] == 0

    # The old bug incorrectly compared BOTH fields with the P3-02 dataset hash.
    payload["tokenizerRecord"]["bundleManifestSha256"] = STAGE.ORIGINAL_P302_BUNDLE_SHA
    with pytest.raises(ValueError, match="original P3-02 fixed endpoint"):
        STAGE.source_preflight(**kwargs)
    payload["tokenizerRecord"]["bundleManifestSha256"] = STAGE.ORIGINAL_P248_TOKENIZER_BUNDLE_SHA
    payload["datasetRecord"]["bundleManifestSha256"] = STAGE.ORIGINAL_P248_TOKENIZER_BUNDLE_SHA
    with pytest.raises(ValueError, match="original P3-02 fixed endpoint"):
        STAGE.source_preflight(**kwargs)
    payload["datasetRecord"]["bundleManifestSha256"] = STAGE.ORIGINAL_P302_BUNDLE_SHA
    payload["step"] = True
    with pytest.raises(ValueError, match="original P3-02 fixed endpoint"):
        STAGE.source_preflight(**kwargs)
    payload["step"] = 100
    checkpoint.write_bytes(b"modified-frozen-step100")
    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        STAGE.source_preflight(**kwargs)


def test_read_only_stage_verifier_reports_stage_status_and_matching_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = tmp_path / "source.pt"
    source.write_bytes(b"unchanged-original")
    checkpoint = tmp_path / "stage.pt"
    checkpoint.write_bytes(b"synthetic-stage-checkpoint")
    mock_dataset = {
        "bundleManifestSha256": "verified-new-copy-fidelity-bundle",
        "candidateSha256": STAGE.CANDIDATE_SHA,
        "bundleVerified": True,
        "splits": {
            "train": {"records": 288, "answerAndEosTargetPositions": 7300},
            "validation": {"records": 96, "answerAndEosTargetPositions": 2420},
        },
    }
    monkeypatch.setattr(STAGE.PACK, "inspect", lambda _: mock_dataset)
    monkeypatch.setattr(STAGE, "source_preflight", lambda **_: {
        "status": "copy-fidelity-source-stage-read-only-preflight-passed",
        "bundleManifestSha256": mock_dataset["bundleManifestSha256"],
        "sourceWeightsVerified": True,
        "optimizerUpdates": 0,
    })
    original, copied = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    copied.load_state_dict(original.state_dict())
    original.config = copied.config = SimpleNamespace(context_length=512, vocab_size=16384)
    stage_payload = {
        "step": 0, "tokensProcessedTotal": 0, "seed": 1337,
        "codec": STAGE.CODEC, "trainingSettings": STAGE._settings(),
        "stageTransitionRecord": STAGE._transition(mock_dataset["bundleManifestSha256"]),
        "datasetRecord": STAGE._dataset_record(mock_dataset),
        "scheduleState": None,
        "optimizerStateDict": {"state": {}, "param_groups": [{}]},
        "tokenizerRecord": {"tokenizerSha256": STAGE.TOKENIZER_SHA},
    }
    checkpoints = iter([(original, {}), (copied, stage_payload)])
    monkeypatch.setattr(STAGE, "read_checkpoint", lambda *_: next(checkpoints))
    monkeypatch.setattr(STAGE, "parameter_count", lambda _: 27566080)
    result = STAGE.verify_stage(
        checkpoint=checkpoint, source_checkpoint=source, bundle_dir=bundle,
    )
    assert result["status"] == "copy-fidelity-zero-stage-verified"
    assert result["baseWeightsPreservedExactly"] is True
    assert result["optimizerStateEmpty"] is True
    assert result["optimizerUpdates"] == 0
