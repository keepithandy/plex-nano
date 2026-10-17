"""Read-only P3 copy-fidelity step-100 generation gates. No live model in CI."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parents[1] / "phase3" / "copy-fidelity-v1"
SPEC = importlib.util.spec_from_file_location("p3_cf_eval_step100_test", HERE / "evaluate_step100.py")
assert SPEC and SPEC.loader
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(E.PACK, "inspect", lambda _: {
        "bundleManifestSha256": E.BUNDLE_SHA, "candidateSha256": E.CANDIDATE_SHA,
    })
    val, control, _ = E._frozen_inputs(E.BUNDLE, E.CONTROL_PATH)
    return val, control


def responses_from_rows(val, controls, *, raw=False):
    rows = []
    for r in val + controls:
        rows.append({
            "recordId": r.get("id", r.get("taskId")), "actual": r["expectedFile"],
            "stoppedAtEos": True, "decodeSucceeded": True,
            "fullPromptVisibleForEveryGenerationStep": True,
            "generatedTokenCount": len(r["expectedFile"]),
        })
    return rows


def test_exact_frozen_control_and_validation_compositions(frozen):
    val, controls = frozen
    assert len(val) == 96 and len(controls) == 42
    assert {r["namespace"] for r in val} == {"granite", "saffron"}
    assert {r["namespace"] for r in controls} == {"cobalt", "spruce"}
    assert {r["language"] for r in val} == {"html", "css", "javascript"}
    assert all(r["condition"] in ("copy", "edit") for r in val)


def test_all_correct_complete_outputs_pass_every_gate(frozen):
    val, controls = frozen
    scores = E.score_responses(val, controls, responses_from_rows(val, controls))
    assert scores["validation"]["copies"] == {"exact": 24, "total": 24}
    assert scores["validation"]["edits"] == {"exact": 72, "total": 72}
    assert scores["validation"]["fullyCorrectContrastGroups"] == 24
    assert scores["allValidationGatesPassed"] is True
    assert scores["development"]["copies"] == {"exact": 6, "total": 6}
    assert scores["development"]["edits"] == {"exact": 36, "total": 36}
    assert scores["development"]["fullyCorrectEditContrastGroups"] == 12
    assert scores["allDevelopmentGatesPassed"] is True
    for lang in ("html", "css", "javascript"):
        assert scores["validation"]["perLanguage"][lang] == {
            "copies": {"exact": 8, "total": 8},
            "edits": {"exact": 24, "total": 24},
        }
        assert scores["development"]["perLanguage"][lang] == {
            "copies": {"exact": 2, "total": 2},
            "edits": {"exact": 12, "total": 12},
        }
    assert len(scores["perRecord"]) == 138
    assert scores["validation"]["fullPromptVisibleCount"] == 96


def test_missing_eos_and_prompt_visibility_never_count_as_exact(frozen):
    val, controls = frozen
    rows = responses_from_rows(val, controls)
    rows[0]["stoppedAtEos"] = False
    rows[1]["fullPromptVisibleForEveryGenerationStep"] = False
    rows[2]["decodeSucceeded"] = False
    scores = E.score_responses(val, controls, rows)
    assert scores["validation"]["copies"]["exact"] < 24
    assert scores["allValidationGatesPassed"] is False
    assert all(scores["perRecord"][i]["exactCompleteFile"] is False for i in range(3))


def test_context_preservation_scoring_does_not_invent_requested_values(frozen):
    val, controls = frozen
    rows = responses_from_rows(val, controls)
    i = next(i for i, row in enumerate(val) if row["condition"] == "edit")
    rows[i]["actual"] = "garbled source"
    summary = E.score_responses(val, controls, rows)
    bad = summary["perRecord"][i]
    assert bad["exactCompleteFile"] is False
    assert bad["unrelatedBytesPreserved"] is False
    assert bad["requestedValueCorrect"] is None
    assert bad["requestedValueExtractable"] is False
    assert summary["allValidationGatesPassed"] is False


def test_language_gate_rejects_one_weak_language_even_if_aggregate_threshold_passes(frozen):
    val, controls = frozen
    rows = responses_from_rows(val, controls)
    # Five CSS edits fail, so aggregate = 67/72 but CSS = 19/24.
    css_ids = [i for i, row in enumerate(val) if
               row["language"] == "css" and row["condition"] == "edit"]
    for i in css_ids[:5]:
        rows[i]["actual"] = "broken"
    scores = E.score_responses(val, controls, rows)
    assert scores["validation"]["edits"] == {"exact": 67, "total": 72}
    assert scores["validation"]["gates"]["editsAtLeast60of72"] is True
    assert scores["validation"]["gates"]["perLanguageAtLeast20of24Edits"] is False
    assert scores["allValidationGatesPassed"] is False


def test_original_developer_controls_have_exact_frozen_boundary_semantics(frozen):
    val, controls = frozen
    copy = next(r for r in controls if r["condition"] == "copy")
    edit = next(r for r in controls if r["condition"] == "minimal-edit")
    assert E._score_control(copy, copy["expectedFile"])["exact"] is True
    result = E._score_control(edit, edit["expectedFile"])
    assert result == {
        "exact": True, "unrelatedBytesPreserved": True,
        "requestedValueCorrect": True, "targetExtractable": True,
    }
    result = E._score_control(edit, edit["inputFile"])
    assert result["exact"] is False
    assert result["unrelatedBytesPreserved"] is True
    assert result["requestedValueCorrect"] is False


def test_scorer_rejects_missing_duplicate_or_reordered_cases(frozen):
    val, controls = frozen
    replies = responses_from_rows(val, controls)
    for rows in (replies[:-1], replies + replies[:1], [replies[1], replies[0]] + replies[2:]):
        with pytest.raises(ValueError, match="all 138 frozen"):
            E.score_responses(val, controls, rows)


def test_training_endpoint_report_requires_pinned_sha_and_history(tmp_path, monkeypatch):
    report_file = tmp_path / "result.json"
    endpoint = tmp_path / "step-0100.pt"
    endpoint.write_bytes(b"synthetic checkpoint")
    monkeypatch.setattr(E, "CHECKPOINT_SHA", hashlib.sha256(endpoint.read_bytes()).hexdigest())
    report = {
        "status": "bounded-research-run-completed", "trainingPerformed": True,
        "completedMaximumSteps": True, "optimizerUpdates": 100,
        "answerAndEosTargetsProcessed": 40477,
        "authorizationContractSha256": E.CONTRACT_SHA,
        "stageCheckpointSha256": E.STAGE_SHA, "bundleManifestSha256": E.BUNDLE_SHA,
        "fixedEndpointSha256": E.CHECKPOINT_SHA, "resumeAllowed": False,
        "automaticContinuation": False, "checkpointReselectionAllowed": False,
        "developmentEvaluationPerformed": False, "finalHoldoutOpened": False,
        "interrupted": False, "maximumOptimizerUpdates": 100,
        "maximumWallTimeSeconds": 600, "elapsedSeconds": 102.94426369998837,
        "checkpoints": [
            {"step": k, "sha256": E.CHECKPOINT_SHA if k == 100 else "0" * 64,
             "path": f"step-{k:04d}.pt"}
            for k in (25, 50, 75, 100)
        ],
        "validation": [{"step": s, "meanValidationAnswerAndEosLoss": loss}
                       for s, loss in [(0, 7.116717401614859), (25, 3.375139713878474),
                                       (50, 2.793581586140246), (75, 2.6199502862189425),
                                       (100, 2.7984008651134396)]],
    }
    report_file.write_text(json.dumps(report), encoding="utf-8")
    proof = E._run_metadata(report_file, endpoint)
    assert proof["fixedCheckpointSha256"] == E.CHECKPOINT_SHA
    report["fixedEndpointSha256"] = "0" * 64
    report_file.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="authorized fixed endpoint"):
        E._run_metadata(report_file, endpoint)


class FakeTokenizer:
    """Synthetic reversible whole-string tokenizer: not an inference tokenizer."""
    vocabulary_size = 16384
    _reverse = {}
    _forward = {}
    def encode(self, text):
        if text not in self._forward:
            token_id = 4 + len(self._forward)
            self._forward[text] = token_id
            self._reverse[token_id] = text
        return [self._forward[text]]
    def decode(self, ids):
        return "".join(self._reverse[x] for x in ids)


def test_preflight_and_evaluation_use_no_optimizers_or_existing_output(frozen, tmp_path, monkeypatch):
    val, controls = frozen
    root = tmp_path / "artifacts"
    root.mkdir()
    destination = root / "generation"
    monkeypatch.setattr(E, "_run_metadata", lambda *args: {
        "runResultSha256": "a" * 64, "fixedCheckpointSha256": E.CHECKPOINT_SHA,
        "authorizedUpdates": 100, "trainTargetsProcessed": 40477,
    })
    monkeypatch.setattr(E, "_completion_tokenizer", lambda _: (FakeTokenizer(), {
        "tokenizerSha256": E.TOKENIZER_SHA,
        "bundleManifestSha256": E.BUNDLE_SHA,
    }))
    monkeypatch.setattr(E, "_checkpoint_model", lambda *args: SimpleNamespace(
        config=SimpleNamespace(context_length=512)
    ))
    expected_by_prompt = {r["prompt"]: r["expectedFile"] for r in val + controls}
    def fake_generate(model, prompt_ids, **kwargs):
        prompt = FakeTokenizer().decode(prompt_ids)
        expected = expected_by_prompt[prompt]
        assert kwargs["temperature"] == 0.0 and kwargs["max_new_tokens"] == 192
        return FakeTokenizer().encode(expected), True, {
            "fullPromptVisibleForEveryGenerationStep": True
        }
    monkeypatch.setattr(E, "_generate_token_ids", fake_generate)
    out = E.evaluate(
        checkpoint=tmp_path / "source.pt", run_result=tmp_path / "result.json",
        bundle_dir=E.BUNDLE, control_path=E.CONTROL_PATH,
        output_dir=destination, artifact_root=root, device_name="cpu",
    )
    assert out["validation"]["copies"]["exact"] == 24
    assert out["validation"]["edits"]["exact"] == 72
    assert out["development"]["copies"]["exact"] == 6
    assert out["development"]["edits"]["exact"] == 36
    assert out["trainingPerformed"] is False and out["optimizerUpdates"] == 0
    assert out["finalHoldoutOpened"] is False
    assert (destination / "evaluation-result.json").exists()
    with pytest.raises(ValueError, match="NEW directory"):
        E.evaluate(
            checkpoint=tmp_path / "source.pt", run_result=tmp_path / "result.json",
            bundle_dir=E.BUNDLE, control_path=E.CONTROL_PATH,
            output_dir=destination, artifact_root=root, device_name="cpu",
        )
