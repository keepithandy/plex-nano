"""Portable P3-02 bundle, loss-mask, source-hash and authorization regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from plex_training.cli import build_parser
from plex_training.config import DEFAULT_CONFIG
from plex_training.file_edit_replacements import (
    DEFAULT_CONTRACT, EXPECTED_TOKENIZER_SHA256, TOKENIZER_BUNDLE_SHA256,
    build_replacement_rows, generate_replacement_candidate,
    plan_replacement_supervision,
)
from plex_training import file_edit_replacements_bundle as bundle
from plex_training.tokenizer import CODEC


class ByteTokenizer:
    vocabulary_size = 16384

    def encode(self, text):
        return [value + 4 for value in text.encode("utf-8")]

    def decode(self, ids):
        return bytes(value - 4 for value in ids).decode("utf-8")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    tokenizer = ByteTokenizer()
    source = tmp_path / "artifacts" / "p2-48-tokenizer"
    source.mkdir(parents=True)
    for name in bundle.TOKENIZER_FILES:
        (source / name).write_bytes(b"{}\n")
    identity = {
        "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
        "bundleManifestSha256": TOKENIZER_BUNDLE_SHA256,
        "modelConfigSha256": hashlib.sha256(b"{}\n").hexdigest(),
    }
    monkeypatch.setattr(bundle, "_completion_tokenizer", lambda *_: (tokenizer, identity))
    plan = plan_replacement_supervision(build_replacement_rows(), tokenizer)
    monkeypatch.setattr(bundle, "preflight_replacement_supervision", lambda **_: {
        "recordTokens": plan["recordTokens"],
        "packedTokenSha256": plan["packedTokenSha256"],
        "sampler": plan["sampler"],
    })
    candidate = tmp_path / "candidate.jsonl"
    review = tmp_path / "review.json"
    generate_replacement_candidate(candidate_path=candidate, review_path=review)
    output = tmp_path / "artifacts" / "p3-02"
    kwargs = {
        "candidate_path": candidate, "review_path": review,
        "source_bundle_dir": source, "output_dir": output,
        "artifact_root": tmp_path / "artifacts",
    }
    result = bundle.prepare_replacement_bundle(**kwargs)
    return {"output": output, "kwargs": kwargs, "result": result,
            "source": source, "plan": plan, "tokenizer": tokenizer}


def test_bundle_packs_only_train_validation_with_exact_next_token_masks(prepared):
    output = prepared["output"]
    result = prepared["result"]
    plan = prepared["plan"]
    assert result["status"] == "training-bundle-prepared-zero-update"
    assert result["bundleVerified"] and result["researchOptimizerUpdates"] == 0
    assert result["records"] == {"train": 144, "validation": 36}
    assert result["recordTokens"] == plan["recordTokens"]
    assert result["answerAndEosTargetPositions"] == plan["answerAndEosTargetPositions"]
    assert result["sampler"] == plan["sampler"]
    assert bundle.inspect_replacement_bundle(output)["bundleVerified"]
    assert not (output / "development.jsonl").exists()
    assert not (output / "development.tokens.u16le").exists()
    assert not (output / "development.weights.u8").exists()
    for split, count in (("train", 144), ("validation", 36)):
        rows = [json.loads(line) for line in (output / (split + ".jsonl")).read_text().splitlines()]
        index = json.loads((output / (split + ".index.json")).read_text())
        mask = (output / (split + ".weights.u8")).read_bytes()
        tokens = (output / (split + ".tokens.u16le")).read_bytes()
        assert len(rows) == len(index) == count
        assert len(tokens) // 2 == plan["recordTokens"][split]
        assert len(mask) == plan["recordTokens"][split] - count
        for line, entry in zip(rows, index, strict=True):
            start = entry["startTarget"]
            stop = start + entry["targetCount"]
            weights = mask[start:stop]
            assert sum(weights) == entry["answerTokensIncludingEos"]
            assert weights == (b"\x00" * (entry["promptTokens"] - 1)
                               + b"\x01" * entry["answerTokensIncludingEos"])
            assert line["recordId"] == entry["recordId"]
        assert sum(mask) == plan["answerAndEosTargetPositions"][split]
    assert sum(result["sampler"][key] for key in ["optimizerUpdatesPerformed"]) == 0


@pytest.mark.parametrize("suffix", ("tokens.u16le", "weights.u8", "jsonl", "index.json"))
def test_bundle_inspector_rejects_tampering(prepared, suffix):
    target = prepared["output"] / ("train." + suffix)
    raw = target.read_bytes()
    target.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError):
        bundle.inspect_replacement_bundle(prepared["output"])


def test_bundle_refuses_replacement_and_output_source_alias(prepared):
    kwargs = prepared["kwargs"]
    original = (prepared["output"] / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="fresh output"):
        bundle.prepare_replacement_bundle(**kwargs)
    assert (prepared["output"] / "manifest.json").read_bytes() == original
    with pytest.raises(ValueError, match="fresh output"):
        bundle.prepare_replacement_bundle(**{**kwargs, "output_dir": prepared["source"]})


def test_bundle_low_storage_budget_rolls_back_before_publication(prepared):
    kwargs = prepared["kwargs"]
    target = prepared["output"].with_name("insufficient")
    with pytest.raises(RuntimeError, match="allocation exceeded"):
        bundle.prepare_replacement_bundle(
            **{**kwargs, "output_dir": target, "storage_limit_bytes": 1})
    assert not target.exists()
    assert not list(target.parent.glob(".p3-02-bundle-*"))


def test_missing_real_tokenizer_blocks_preflight_before_any_write(tmp_path):
    candidate = tmp_path / "missing-candidate"
    output = tmp_path / "artifacts" / "uncreated"
    with pytest.raises((ValueError, FileNotFoundError)):
        bundle.prepare_replacement_bundle(
            candidate_path=candidate,
            source_bundle_dir=tmp_path / "no-real-tokenizer",
            output_dir=output, artifact_root=tmp_path / "artifacts")
    assert not output.exists()


def _source_payload():
    return {
        "step": 100, "tokensProcessedTotal": 429375,
        "seed": 1337, "codec": CODEC,
        "stageTransitionRecord": {
            "kind": "plex-request-grounded-coding-stage-transition-v1",
            "milestone": "P2-48",
        },
        "trainingSettings": {"kind": "p2-48-authorized-request-grounded-training-v1"},
        "tokenizerRecord": {
            "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
            "bundleManifestSha256": TOKENIZER_BUNDLE_SHA256,
            "actualVocabularySize": 16384,
        },
        "initializationRecord": {
            "pretrainedCheckpointLoaded": False,
            "pretrainedModelWeightsLoaded": False,
        },
    }


def test_checkpoint_source_requires_frozen_sha_then_checks_payload(prepared, tmp_path, monkeypatch):
    checkpoint = tmp_path / "step-0100.pt"
    checkpoint.write_bytes(b"frozen-checkpoint-fixture")
    good_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(bundle, "SOURCE_CHECKPOINT_SHA256", good_hash)
    monkeypatch.setattr(bundle, "SOURCE_CHECKPOINT_BYTES", checkpoint.stat().st_size)
    final = {"step": 100, "path": "step-0100.pt",
             "sha256": good_hash, "bytes": checkpoint.stat().st_size}
    result = tmp_path / "p2-48-result.json"
    result.write_text(json.dumps({
        "milestone": "P2-48", "trainingPerformed": True,
        "researchOptimizerUpdates": 100, "finalCheckpoint": final,
    }))
    class Model:
        config = DEFAULT_CONFIG
    payload = _source_payload()
    with patch.object(bundle, "read_checkpoint", return_value=(Model(), payload)) as load, \
         patch.object(bundle, "parameter_count", return_value=27566080):
        checked = bundle.preflight_replacement_checkpoint_source(
            checkpoint_path=checkpoint, bundle_dir=prepared["output"], result_path=result)
        assert checked["status"] == "checkpoint-source-read-only-preflight-passed"
        assert checked["bundleCompatibilityVerified"]
        assert not checked["checkpointStaged"]
        assert not checked["optimizerCreated"]
        assert not checked["modelTrainingAuthorized"]
        load.assert_called_once()
        checkpoint.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="exact local"):
            bundle.preflight_replacement_checkpoint_source(
                checkpoint_path=checkpoint, bundle_dir=prepared["output"], result_path=result)
        assert load.call_count == 1
        checkpoint.write_bytes(b"frozen-checkpoint-fixture")
        with patch.object(bundle, "read_checkpoint", return_value=(
             Model(), {**payload, "step": True})):
            with pytest.raises(ValueError, match="provenance"):
                bundle.preflight_replacement_checkpoint_source(
                    checkpoint_path=checkpoint, bundle_dir=prepared["output"], result_path=result)


def test_p302_cli_commands_use_rooted_paths_and_no_training_switch():
    parser = build_parser()
    pack = parser.parse_args(["file-edit-replacements-bundle-prepare"])
    assert pack.candidate.is_absolute() and pack.review.is_absolute()
    assert pack.output_dir.is_absolute() and pack.source_bundle_dir.is_absolute()
    assert pack.artifact_root.is_absolute()
    inspect = parser.parse_args(["file-edit-replacements-bundle-verify"])
    assert inspect.bundle_dir.is_absolute()
    source = parser.parse_args(["file-edit-replacements-source-preflight"])
    assert source.bundle_dir.is_absolute()
    assert source.checkpoint.is_absolute() and source.result.is_absolute()
    for args in (pack, inspect, source):
        assert not hasattr(args, "train") and not hasattr(args, "steps")
