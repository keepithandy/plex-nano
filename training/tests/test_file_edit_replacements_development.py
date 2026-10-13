"""P3-02 fixed 100-step endpoint evidence and development-only gate regressions."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from plex_training.cli import build_parser
from plex_training import file_edit_replacements_development as evaluation


def _responses_from_frozen_targets():
    return [
        {"recordId": row["id"], "actual": row["expectedFile"],
         "generatedTokenCount": 40, "stoppedAtEos": True,
         "decodeSucceeded": True, "fullPromptVisibleForEveryGenerationStep": True}
        for row in evaluation._development_rows()
    ]


def test_frozen_endpoint_identity_and_exact_declared_gate():
    frozen = evaluation._json(evaluation.DEFAULT_FROZEN_REPORT)
    assert frozen["fixedEndpoint"]["sha256"] == evaluation.EXPECTED_ENDPOINT_SHA256
    assert frozen["fixedEndpoint"]["step"] == 100
    assert frozen["researchOptimizerUpdates"] == 100
    assert frozen["automaticContinuation"] is False
    assert frozen["finalHoldoutOpened"] is False
    assert frozen["developmentEvaluationPerformed"] is False
    assert frozen["validation"][-1]["meanValidationAnswerAndEosLoss"] == 1.5892543557015333
    assert [x["step"] for x in frozen["checkpoints"]] == [25, 50, 75, 100]
    assert frozen["fixedEndpoint"]["checkpointReselectionAllowed"] is False
    assert len(evaluation._development_rows()) == 36


def test_perfect_development_responses_pass_exactly_predeclared_gate():
    scores = evaluation.score_development_responses(_responses_from_frozen_targets())
    assert scores["totalRecords"] == 36
    assert scores["exactEdits"] == 36
    assert scores["unrelatedCodePreserved"] == 36
    assert scores["fullPromptVisibleRecords"] == 36
    assert scores["allThreeCorrectContrastGroups"] == 12
    assert scores["developmentGatePassed"] is True
    assert scores["gateChecks"] == dict.fromkeys(scores["gateChecks"], True)
    assert {r["language"] for r in scores["perRecord"]} == {"html", "css", "javascript"}
    assert len(scores["perFamily"]) == 12
    assert all(v["exactEdits"] == 12 for v in scores["perLanguage"].values())


def test_wrong_request_outputs_reject_contrast_group_and_per_language_gate():
    answers = _responses_from_frozen_targets()
    first_group = answers[0]["recordId"].rsplit("-", 1)[0]
    group = [answer for answer in answers if answer["recordId"].startswith(first_group + "-")]
    assert len(group) == 3
    answers[0]["actual"], answers[1]["actual"] = (
        answers[1]["actual"], answers[0]["actual"]
    )
    result = evaluation.score_development_responses(answers)
    assert result["exactEdits"] == 34
    assert result["allThreeCorrectContrastGroups"] == 11
    assert result["developmentGatePassed"] is True
    # Failure of whole-language and group gate, not just overall pass count.
    for answer in answers:
        if answer["recordId"].startswith("p302-javascript-"):
            answer["actual"] = "unrelated code"
    result = evaluation.score_development_responses(answers)
    assert result["perLanguage"]["javascript"]["exactEdits"] == 0
    assert result["allThreeCorrectContrastGroups"] < 10
    assert result["developmentGatePassed"] is False


def test_exact_output_with_prompt_eviction_fails_policy_gate():
    answers = _responses_from_frozen_targets()
    answers[-1]["fullPromptVisibleForEveryGenerationStep"] = False
    result = evaluation.score_development_responses(answers)
    assert result["exactEdits"] == 36
    assert result["gateChecks"]["fullPromptVisibleEveryStep"] is False
    assert result["developmentGatePassed"] is False


def test_unrelated_edits_loss_and_non_eos_are_recorded():
    answers = _responses_from_frozen_targets()
    answers[0]["actual"] += "\n// unrelated"
    answers[0]["stoppedAtEos"] = False
    result = evaluation.score_development_responses(answers)
    assert result["exactEdits"] == 35
    assert result["unrelatedCodePreserved"] == 35
    assert result["eosTerminatedRecords"] == 35


def test_decode_failure_cannot_count_as_correct_even_if_output_matches():
    answers = _responses_from_frozen_targets()
    answers[0]["decodeSucceeded"] = False
    scores = evaluation.score_development_responses(answers)
    assert scores["exactEdits"] == 35
    assert scores["unrelatedCodePreserved"] == 35


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unexpected", "invalid-output", "invalid-flag"])
def test_development_scoring_rejects_incomplete_or_modified_protocol(problem):
    answers = _responses_from_frozen_targets()
    if problem == "missing":
        answers.pop()
    elif problem == "duplicate":
        answers[-1] = copy.deepcopy(answers[0])
    elif problem == "unexpected":
        answers[0]["recordId"] = "p301-final-project-holdout"
    elif problem == "invalid-output":
        answers[0]["actual"] = None
    else:
        answers[0]["stoppedAtEos"] = 1
    with pytest.raises(ValueError):
        evaluation.score_development_responses(answers)


def test_local_frozen_report_must_match_checkpoint_hash_and_run_result(tmp_path, monkeypatch):
    source = tmp_path / "first-run"
    checkpoint_dir = source / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "step-0100.pt"
    checkpoint.write_bytes(b"test")
    local = evaluation._json(evaluation.DEFAULT_FROZEN_REPORT)
    # Actual trainer result has a different kind and fixedEndpoint fields from frozen evidence.
    local.pop("fixedEndpoint")
    local["kind"] = "p3-02-first-bounded-answer-eos-run-result-v1"
    local["fixedEndpointStep"] = 100
    local["fixedEndpointSha256"] = evaluation.EXPECTED_ENDPOINT_SHA256
    local["elapsedSeconds"] = 19.9099784999853
    local["checkpoints"] = evaluation._json(
        evaluation.DEFAULT_FROZEN_REPORT)["checkpoints"]
    result_path = source / "result.json"
    result_path.write_text(json.dumps(local))
    monkeypatch.setattr(evaluation, "EXPECTED_BYTES", 4)
    monkeypatch.setattr(evaluation, "sha256_file", lambda _: evaluation.EXPECTED_ENDPOINT_SHA256)
    evidence = evaluation._frozen_run_report(
        local_result=result_path, frozen_report=evaluation.DEFAULT_FROZEN_REPORT,
        checkpoint=checkpoint)
    assert evidence["fixedEndpoint"]["step"] == 100
    local["fixedEndpointSha256"] = "tampered"
    result_path.write_text(json.dumps(local))
    with pytest.raises(ValueError, match="differs"):
        evaluation._frozen_run_report(
            local_result=result_path, frozen_report=evaluation.DEFAULT_FROZEN_REPORT,
            checkpoint=checkpoint)


def test_local_frozen_endpoint_rejects_wrong_checkpoint_hash(tmp_path, monkeypatch):
    source = tmp_path / "first-run"
    directory = source / "checkpoints"
    directory.mkdir(parents=True)
    ckpt = directory / "step-0100.pt"
    ckpt.write_bytes(b"test")
    local = evaluation._json(evaluation.DEFAULT_FROZEN_REPORT)
    local.pop("fixedEndpoint")
    local["kind"] = "p3-02-first-bounded-answer-eos-run-result-v1"
    local.update(fixedEndpointStep=100, fixedEndpointSha256=evaluation.EXPECTED_ENDPOINT_SHA256)
    source_result = source / "result.json"
    source_result.write_text(json.dumps(local))
    monkeypatch.setattr(evaluation, "EXPECTED_BYTES", 4)
    monkeypatch.setattr(evaluation, "sha256_file", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 changed"):
        evaluation._frozen_run_report(
            local_result=source_result, frozen_report=evaluation.DEFAULT_FROZEN_REPORT,
            checkpoint=ckpt)


def test_cli_has_one_read_only_fixed_development_command():
    parser = build_parser()
    arg = parser.parse_args(["file-edit-replacements-development-evaluate"])
    assert arg.device == "cpu"
    assert arg.checkpoint.name == "step-0100.pt"
    assert arg.run_result.name == "result.json"
    assert arg.bundle_dir.is_absolute()
    assert arg.output_dir.is_absolute()
    assert arg.artifact_root.is_absolute()
    assert not hasattr(arg, "authorize_first_run")
    assert not hasattr(arg, "optimizer_updates")
    assert not hasattr(arg, "final_holdout")


def test_evaluator_rejects_existing_output_before_checkpoint_or_model_read(tmp_path, monkeypatch):
    root = tmp_path / "root"
    output = root / "used"
    output.mkdir(parents=True)
    monkeypatch.setattr(evaluation, "read_checkpoint",
                        lambda *_: (_ for _ in ()).throw(AssertionError("must not load")))
    with pytest.raises(ValueError, match="fresh"):
        evaluation.evaluate_replacement_development(
            output_dir=output, artifact_root=root, device="cpu")


def test_synthetic_end_to_end_read_only_36_responses(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    bundle = root / "frozen-bundle"
    bundle.mkdir()
    ckpt = root / "first-run/checkpoints/step-0100.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"fixture")
    report = root / "first-run/result.json"
    report.write_text("{}")
    output = root / "development"
    monkeypatch.setattr(evaluation, "_frozen_run_report", lambda **_: {
        "sourceCheckpointSha256": evaluation.PROPOSED_RUN["sourceCheckpointSha256"]
    })
    monkeypatch.setattr(evaluation, "inspect_replacement_bundle", lambda *_: {
        "bundleManifestSha256": evaluation.BUNDLE_SHA256})
    class FakeTokenizer:
        vocabulary_size = 16384
        def __init__(self):
            self.last_prompt = ""
        def encode(self, prompt):
            self.last_prompt = prompt
            return [4, 5, 6]
        def decode(self, tokens):
            return self.last_prompt + "wrong edit"
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(evaluation, "_completion_tokenizer", lambda *_: (
        tokenizer, {"tokenizerSha256": evaluation.EXPECTED_TOKENIZER_SHA256}
    ))
    class FakeModel:
        config = SimpleNamespace(context_length=512, vocab_size=16384)
        def to(self, *_): return self
        def eval(self): return self
    monkeypatch.setattr(evaluation, "read_checkpoint", lambda *_: (FakeModel(), {
        "codec": evaluation.CODEC, "step": 100, "tokensProcessedTotal": 60445,
        "tokenizerRecord": {"tokenizerSha256": evaluation.EXPECTED_TOKENIZER_SHA256},
        "initializationRecord": {"pretrainedCheckpointLoaded": False},
        "stageTransitionRecord": {"baseCheckpointSha256": evaluation.PROPOSED_RUN["sourceCheckpointSha256"]},
        "datasetRecord": {"bundleManifestSha256": evaluation.BUNDLE_SHA256},
        "trainingSettings": {
            "authorizationContractSha256": evaluation.EXPECTED_AUTHORIZATION_SHA256,
            "stageCheckpointSha256": evaluation.EXPECTED_STAGE_SHA256,
            "kind": "p3-02-authorized-answer-eos-training-v1",
        },
    }))
    monkeypatch.setattr(evaluation, "parameter_count", lambda *_: 27566080)
    calls = []
    def fake_generate(*args, **kwargs):
        calls.append((kwargs["max_new_tokens"], kwargs["temperature"], kwargs["seed"]))
        return [7], True, {
            "promptTokenCount": 3, "fullPromptVisibleForEveryGenerationStep": True,
        }
    monkeypatch.setattr(evaluation, "_generate_token_ids", fake_generate)
    summary = evaluation.evaluate_replacement_development(
        checkpoint=ckpt, run_result=report, bundle_dir=bundle,
        output_dir=output, artifact_root=root, device="cpu")
    full = json.loads((output / "development-result.json").read_text())
    assert len(calls) == 36
    assert set(calls) == {(192, 0.0, 1337)}
    assert summary["totalRecords"] == 36
    assert summary["exactEdits"] == 0
    assert full["developmentGatePassed"] is False
    assert full["trainingPerformedDuringEvaluation"] is False
    assert full["finalProjectHoldoutOpened"] is False
    assert len(full["perRecord"]) == 36
    assert ckpt.read_bytes() == b"fixture"
    assert report.read_text() == "{}"
    with pytest.raises(ValueError, match="fresh"):
        evaluation.evaluate_replacement_development(
            checkpoint=ckpt, run_result=report, bundle_dir=bundle,
            output_dir=output, artifact_root=root, device="cpu")
