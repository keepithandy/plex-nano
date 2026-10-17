"""P3-03 read-only diagnostics of the original frozen P3-02 development outputs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from plex_training import file_edit_replacements_diagnostic as p303
from plex_training.file_edit_replacements_development import (
    _development_rows, score_development_responses,
)


def synthetic_report(*, different_outputs: bool = True):
    by_language = {"html": 7, "css": 5, "javascript": 4}
    index = {"html": 0, "css": 0, "javascript": 0}
    responses = []
    for row in _development_rows():
        lang = row["language"]
        n = index[lang]
        index[lang] += 1
        actual = f"SYNTHETIC INVALID {lang} output #{n % by_language[lang]}" if different_outputs else "settings"
        responses.append({
            "recordId": row["id"],
            "actual": actual,
            "stoppedAtEos": True,
            "decodeSucceeded": True,
            "generatedTokenCount": 9,
            "fullPromptVisibleForEveryGenerationStep": True,
        })
    result = score_development_responses(responses)
    assert result["exactEdits"] == 0
    assert result["unrelatedCodePreserved"] == 0
    return {
        "schemaVersion": 1, "milestone": "P3-02",
        "kind": p303.EVALUATION_KIND, "status": "development-evaluation-complete",
        "checkpointStep": 100, "checkpointSha256": p303.EXPECTED_ENDPOINT_SHA256,
        "bundleManifestSha256": p303.BUNDLE_SHA256,
        "candidateSha256": p303.CANDIDATE_SHA256,
        "tokenizerSha256": p303.EXPECTED_TOKENIZER_SHA256,
        "authorizationContractSha256": p303.EXPECTED_AUTHORIZATION_SHA256,
        "generation": {"maximumNewTokens": 192,
                       "policy": "greedy-unreselected-fixed-endpoint-v1",
                       "seed": 1337, "temperature": 0.0},
        "researchOptimizerUpdatesPerformed": 0,
        "trainingPerformedDuringEvaluation": False,
        "automaticContinuation": False,
        "finalProjectHoldoutOpened": False,
        **result,
    }


def frozen():
    return p303._read_json(p303.DEFAULT_FROZEN_SUMMARY)


def test_read_only_diagnostic_distinguishes_language_and_contrast_diversity():
    source = synthetic_report()
    saved = copy.deepcopy(source)
    result = p303.diagnose_existing_development(source, frozen())
    assert result["status"] == "read-only-diagnostic-complete"
    assert result["causeEstablished"] is False
    assert result["researchOptimizerUpdates"] == 0
    assert result["trainingPerformed"] is False
    assert result["finalProjectHoldoutOpened"] is False
    assert result["sourceCheckpointSha256"] == p303.EXPECTED_ENDPOINT_SHA256
    assert result["totalRecords"] == 36
    assert {x: result["perLanguage"][x]["uniqueOutputs"]
            for x in ("html", "css", "javascript")} == {
                "html": 7, "css": 5, "javascript": 4,
            }
    assert {x: result["perLanguage"][x]["repeatOccurrencesBeyondFirst"]
            for x in ("html", "css", "javascript")} == {
                "html": 5, "css": 7, "javascript": 8,
            }
    assert len(result["contrastGroups"]) == 12
    assert all(g["requests"] == 3 for g in result["contrastGroups"])
    assert sum(result["contrastGroupUniqueOutputDistribution"].values()) == 12
    assert len(result["perRecordIndicators"]) == 36
    assert source == saved  # Original evaluation JSON must be untouched.


def test_same_response_to_all_three_requests_is_reported_not_reinterpreted():
    result = p303.diagnose_existing_development(synthetic_report(different_outputs=False), frozen())
    assert result["identicalAcrossAllThreeRequests"] == 12
    assert result["contrastGroupUniqueOutputDistribution"] == {1: 12}
    assert all(group["uniqueOutputs"] == 1 for group in result["contrastGroups"])
    for lang in ("html", "css", "javascript"):
        assert result["perLanguage"][lang]["uniqueOutputs"] == 1
        assert result["perLanguage"][lang]["indicators"]["foreignContext/settings"] == 12


@pytest.mark.parametrize("field,value", [
    ("checkpointSha256", "not-the-approved-model"),
    ("kind", "final-project-holdout"),
    ("trainingPerformedDuringEvaluation", True),
    ("finalProjectHoldoutOpened", True),
    ("researchOptimizerUpdatesPerformed", 1),
    ("developmentGatePassed", True),
    ("candidateSha256", "altered-candidate"),
    ("generation", {"temperature": 0.5}),
])
def test_altered_endpoint_holdout_authorization_or_generation_rejected(field, value):
    report = synthetic_report()
    report[field] = value
    with pytest.raises(ValueError, match="provenance mismatch"):
        p303.diagnose_existing_development(report, frozen())


def test_missing_or_duplicated_individual_records_rejected():
    report = synthetic_report()
    report["perRecord"].pop()
    with pytest.raises(ValueError, match="36 saved"):
        p303.diagnose_existing_development(report, frozen())
    report = synthetic_report()
    report["perRecord"][0] = copy.deepcopy(report["perRecord"][1])
    with pytest.raises(ValueError, match="duplicated"):
        p303.diagnose_existing_development(report, frozen())


def test_modified_task_request_and_output_scores_rejected():
    for field, value in (
        ("request", "ignore user"), ("inputFile", "changed source"),
        ("expectedFile", "changed target"), ("language", "python"),
        ("contrastGroupId", "altered"), ("editFamily", "altered"),
    ):
        report = synthetic_report()
        report["perRecord"][0][field] = value
        with pytest.raises(ValueError, match="task metadata changed"):
            p303.diagnose_existing_development(report, frozen())
    report = synthetic_report()
    report["perRecord"][0]["exactMatch"] = True
    with pytest.raises(ValueError, match="perRecord score"):
        p303.diagnose_existing_development(report, frozen())


def test_forged_aggregate_or_frozen_summary_rejected():
    report = synthetic_report()
    report["exactEdits"] = 12
    with pytest.raises(ValueError, match="aggregate"):
        p303.diagnose_existing_development(report, frozen())
    summary = frozen()
    summary["fullPromptVisibleRecords"] = 0
    with pytest.raises(ValueError, match="frozen failed gate"):
        p303.diagnose_existing_development(synthetic_report(), summary)


def test_frozen_source_only_and_missing_report_rejected(tmp_path):
    with pytest.raises(ValueError, match="bounded regular"):
        p303.diagnose_from_file(report_path=tmp_path / "absent.json")
    malformed = tmp_path / "duplicate.json"
    malformed.write_text('{"status":"good", "status":"bad"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate object keys"):
        p303.diagnose_from_file(report_path=malformed)
    too_large = tmp_path / "oversized.json"
    too_large.write_text(" " * (1024 * 1024 + 1), encoding="utf-8")
    with pytest.raises(ValueError, match="bounded regular"):
        p303._read_json(too_large)


def test_from_file_uses_saved_evidence_and_does_not_write_over_it(tmp_path):
    report = tmp_path / "development-result.json"
    contents = json.dumps(synthetic_report(), indent=2)
    report.write_text(contents, encoding="utf-8")
    result = p303.diagnose_from_file(report_path=report)
    assert result["totalRecords"] == 36
    assert report.read_text(encoding="utf-8") == contents
    assert sorted(p.name for p in tmp_path.iterdir()) == ["development-result.json"]


def test_json_summary_avoids_reprinting_all_raw_private_files():
    diag = p303.diagnose_existing_development(synthetic_report(), frozen())
    assert "perRecord" not in diag
    assert "checkpointPath" not in diag
    assert diag["perLanguage"]["css"]["indicators"]["unchangedInputFile"] == 0
