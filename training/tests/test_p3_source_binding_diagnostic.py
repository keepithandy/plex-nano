"""Synthetic only: P3-SB00 model-free diagnosis of existing saved completions."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1] / "phase3" / "copy-fidelity-v1"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = _load("p3sb_diagnose_tests", "diagnose_saved_outputs.py")
E = _load("p3sb_generate_tests", "evaluate_step100.py")


@pytest.fixture
def frozen():
    val, dev = D.frozen_rows()
    return val, dev


def mock_report(val, dev, *, nonempty=None):
    nonempty = nonempty or {}
    responses = []
    for row in val + dev:
        ident = row["id"] if "id" in row else row["taskId"]
        actual = nonempty.get(ident, "")
        responses.append({
            "recordId": ident, "actual": actual, "generatedTokenCount": len(actual),
            "stoppedAtEos": True, "decodeSucceeded": True,
            "fullPromptVisibleForEveryGenerationStep": True,
        })
    scores = E.score_responses(val, dev, responses)
    return {
        "schemaVersion": 1, "kind": "p3-copy-fidelity-fixed-step100-generation-v1",
        "status": "fixed-endpoint-generation-evaluation-complete",
        "checkpointStep": 100, "checkpointSha256": D.ENDPOINT_SHA,
        "bundleManifestSha256": D.BUNDLE_SHA,
        "candidateSha256": D.CANDIDATE_SHA,
        "controlManifestSha256": D.CONTROL_SHA,
        "trainingPerformed": False, "optimizerUpdates": 0,
        "finalHoldoutOpened": False,
        "developmentControlsPreviouslyInspected": True,
        "generation": {
            "kind": "greedy-complete-file-v1",
            "maxNewTokens": 192, "temperature": 0.0, "seed": 1337,
            "checkpointReselectionAllowed": False,
        },
        **scores,
    }


def test_frozen_inputs_are_exact_and_never_training_split(frozen):
    val, dev = frozen
    assert len(val) == 96
    assert len(dev) == 42
    assert {r["namespace"] for r in val} == set(D.VALID_NAMES)
    assert {r["namespace"] for r in dev} == set(D.CONTROL_NAMES)
    assert set(D.TRAIN_NAMES).isdisjoint(D.VALID_NAMES)
    assert set(D.TRAIN_NAMES).isdisjoint(D.CONTROL_NAMES)
    assert all(r["split"] == "validation" for r in val)


def test_empty_outputs_preserve_unknown_values_instead_of_inventing_values(frozen):
    val, dev = frozen
    result = D.analyze(mock_report(val, dev), val, dev)
    assert result["status"] == "p3-source-binding-read-only-morphology-reviewed"
    assert result["totalCases"] == 138
    assert result["byScope"]["validation"]["cases"] == 96
    assert result["byScope"]["development"]["cases"] == 42
    assert result["byScope"]["validation"]["emptyOutput"] == 96
    assert result["byScope"]["development"]["emptyOutput"] == 42
    assert result["byScope"]["validation"]["requestedValueCorrectWhenExtractable"] == 0
    assert result["byScope"]["validation"]["requestedValueExtractable"] == 0
    assert result["byScope"]["validation"]["trainingNamespaceIntrusion"] == 0
    assert result["responseRepetition"]["reusedOutputRecords"] == 138
    assert result["allOriginalGeneratedFileGatesPassed"] is False
    assert result["trainingPerformed"] is False
    assert result["finalHoldoutOpened"] is False
    assert result["optimizerUpdates"] == 0


def test_training_namespace_intrusion_and_requested_literal_are_only_heuristics(frozen):
    val, dev = frozen
    target = next(r for r in val if r["language"] == "html"
                  and r["condition"] == "edit" and r["requestedLiteral"] == "Start")
    name = target["id"]
    report = mock_report(val, dev, nonempty={
        name: '<button id="meadow-launch">Start</button>\n',
    })
    summary = D.analyze(report, val, dev)
    assert summary["byScope"]["validation"]["trainingNamespaceIntrusion"] == 1
    assert summary["byScope"]["validation"]["sourceNamespacePresent"] == 0
    assert summary["byScope"]["validation"]["requestedLiteralMentioned"] == 1
    assert summary["byScope"]["validation"]["requestedValueCorrectWhenExtractable"] == 0
    assert summary["illustrativeFailures"][0]["sourceName"] in D.VALID_NAMES
    assert summary["sourceNamespacePresentIsNotExactPreservation"] is True
    assert summary["requestedLiteralMentionIsNotProvenValueBinding"] is True


def test_report_order_tampering_and_stale_aggregate_refused(frozen):
    val, dev = frozen
    report = mock_report(val, dev)
    modified = copy.deepcopy(report)
    modified["perRecord"][0], modified["perRecord"][1] = (
        modified["perRecord"][1], modified["perRecord"][0]
    )
    with pytest.raises(ValueError, match="reordered or malformed"):
        D.analyze(modified, val, dev)
    modified = copy.deepcopy(report)
    modified["validation"]["eosCount"] = 0
    with pytest.raises(ValueError, match="Saved summary disagrees"):
        D.analyze(modified, val, dev)
    modified = copy.deepcopy(report)
    modified["perRecord"][0]["exactCompleteFile"] = True
    with pytest.raises(ValueError, match="inconsistent"):
        D.analyze(modified, val, dev)


def test_changed_provenance_or_training_field_refused(frozen):
    val, dev = frozen
    report = mock_report(val, dev)
    for key, bad in [
        ("checkpointSha256", "0" * 64),
        ("trainingPerformed", True),
        ("optimizerUpdates", 1),
        ("finalHoldoutOpened", True),
        ("bundleManifestSha256", "1" * 64),
    ]:
        modified = copy.deepcopy(report)
        modified[key] = bad
        with pytest.raises(ValueError, match="original fixed-endpoint"):
            D.analyze(modified, val, dev)


def test_manifest_bytes_are_frozen_and_wrong_hash_fails(tmp_path: Path):
    path = tmp_path / "candidate.json"
    path.write_bytes(D.CANDIDATE.read_bytes())
    manifest = D.read_frozen(path, D.CANDIDATE_SHA, 4 * 1024 * 1024)
    assert len(manifest["rows"]) == 384
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash changed"):
        D.read_frozen(path, D.CANDIDATE_SHA, 4 * 1024 * 1024)


def test_diagnostic_does_not_import_or_write_model_artifacts():
    script = (HERE / "diagnose_saved_outputs.py").read_text(encoding="utf-8")
    assert "import torch" not in script
    assert "optimizer.step()" not in script
    assert "read_checkpoint(" not in script
    assert ".write_text(" not in script
    assert ".write_bytes(" not in script
    assert 'parser.add_argument("--report"' in script
