"""P3-02-D1 read-only diagnostics of the existing frozen P3-02 development JSON.

No checkpoint loading, model inference, dataset mutation, optimizer updates, or
access to the final project holdout. Descriptive counts are NOT a new gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .file_edit_replacements import CONTEXT_NAMES, ROOT, build_replacement_rows
from .file_edit_replacements_development import (
    EVALUATION_KIND, EXPECTED_AUTHORIZATION_SHA256,
    EXPECTED_ENDPOINT_SHA256, EXPECTED_TOKENIZER_SHA256,
    BUNDLE_SHA256, CANDIDATE_SHA256, _development_rows,
    score_development_responses,
)

DEFAULT_REPORT = (
    Path.home() / "PlexArtifacts/file-edits/p3-02-development-evaluation/development-result.json"
)
DEFAULT_FROZEN_SUMMARY = (
    ROOT / "training/pretraining/p3-02-development-gate-result-summary.json"
)
DIAGNOSTIC_KIND = "p3-02-d1-read-only-frozen-development-output-diagnostic-v1"
LANGUAGES = ("html", "css", "javascript")


def _read_json(path: Path, *, max_bytes: int = 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= max_bytes:
        raise ValueError("P3-02-D1 input must be an existing bounded regular JSON file")
    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("P3-02-D1 JSON contains duplicate object keys")
            result[key] = value
        return result
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)
    if not isinstance(result, dict):
        raise ValueError("P3-02-D1 JSON root must be an object")
    return result


def diagnose_existing_development(
    report: dict[str, Any], frozen_summary: dict[str, Any],
) -> dict[str, Any]:
    """Inspect original 36 decoded outputs; never generate or score a new model."""
    expected_fields = {
        "schemaVersion": 1, "milestone": "P3-02",
        "kind": EVALUATION_KIND, "status": "development-evaluation-complete",
        "checkpointStep": 100, "checkpointSha256": EXPECTED_ENDPOINT_SHA256,
        "bundleManifestSha256": BUNDLE_SHA256,
        "candidateSha256": CANDIDATE_SHA256,
        "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
        "authorizationContractSha256": EXPECTED_AUTHORIZATION_SHA256,
        "generation": {
            "maximumNewTokens": 192,
            "policy": "greedy-unreselected-fixed-endpoint-v1",
            "seed": 1337, "temperature": 0.0,
        },
        "researchOptimizerUpdatesPerformed": 0,
        "trainingPerformedDuringEvaluation": False,
        "finalProjectHoldoutOpened": False, "automaticContinuation": False,
        "developmentGatePassed": False,
    }
    for name, value in expected_fields.items():
        if type(report.get(name)) is not type(value) or report[name] != value:
            raise ValueError("P3-02-D1 existing development report provenance mismatch: " + name)
    frozen_fields = {
        "schemaVersion": 1, "milestone": "P3-02",
        "kind": "p3-02-owner-reported-frozen-step100-development-result-summary-v1",
        "status": "development-gate-failed",
        "checkpointStep": 100, "checkpointSha256": EXPECTED_ENDPOINT_SHA256,
        "bundleManifestSha256": BUNDLE_SHA256,
        "candidateSha256": CANDIDATE_SHA256,
        "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
        "authorizationContractSha256": EXPECTED_AUTHORIZATION_SHA256,
        "generation": expected_fields["generation"],
        "developmentGatePassed": False,
        "researchOptimizerUpdatesPerformed": 0,
        "trainingPerformedDuringEvaluation": False,
        "finalProjectHoldoutOpened": False, "automaticContinuation": False,
    }
    for name, value in frozen_fields.items():
        if type(frozen_summary.get(name)) is not type(value) or frozen_summary[name] != value:
            raise ValueError("P3-02-D1 committed development summary mismatch: " + name)

    rows = report.get("perRecord")
    if not isinstance(rows, list) or len(rows) != 36:
        raise ValueError("P3-02-D1 requires all 36 saved perRecord completions")
    frozen_rows = {r["id"]: r for r in _development_rows()}
    by_id = {}
    for audit in rows:
        if not isinstance(audit, dict):
            raise ValueError("P3-02-D1 development audit row is not an object")
        ident = audit.get("recordId")
        if not isinstance(ident, str) or ident not in frozen_rows or ident in by_id:
            raise ValueError("P3-02-D1 missing, duplicated or unexpected development ID")
        expected = frozen_rows[ident]
        for key, value in {
            "language": expected["language"], "editFamily": expected["editFamily"],
            "contrastGroupId": expected["contrastGroupId"],
            "request": expected["request"], "inputFile": expected["inputFile"],
            "expectedFile": expected["expectedFile"],
        }.items():
            if audit.get(key) != value:
                raise ValueError("P3-02-D1 original perRecord task metadata changed: " + key)
        if type(audit.get("actual")) is not str:
            raise ValueError("P3-02-D1 raw response text is missing")
        by_id[ident] = audit

    rescored = score_development_responses([
        {
            "recordId": row["recordId"], "actual": row["actual"],
            "stoppedAtEos": row["stoppedAtEos"],
            "decodeSucceeded": row["decodeSucceeded"],
            "generatedTokenCount": row["generatedTokenCount"],
            "fullPromptVisibleForEveryGenerationStep":
                row["fullPromptVisibleForEveryGenerationStep"],
        } for row in rows
    ])
    for key in (
        "totalRecords", "exactEdits", "unrelatedCodePreserved",
        "fullPromptVisibleRecords", "eosTerminatedRecords",
        "decodeSucceededRecords", "allThreeCorrectContrastGroups",
        "perLanguage", "perFamily", "gateChecks", "developmentGatePassed",
    ):
        if report.get(key) != rescored[key]:
            raise ValueError("P3-02-D1 saved aggregate contradicts raw responses: " + key)
        if key in frozen_summary and frozen_summary[key] != rescored[key]:
            raise ValueError("P3-02-D1 raw responses contradict frozen failed gate: " + key)
    for row in rescored["perRecord"]:
        audit = by_id[row["recordId"]]
        if any(audit.get(key) != value for key, value in row.items()):
            raise ValueError("P3-02-D1 raw perRecord score disagrees with frozen scorer")

    all_authored = build_replacement_rows()
    previous_inputs = {
        r["inputFile"] for r in all_authored if r["candidateSplit"] in ("train", "validation")
    }
    previous_targets = {
        r["expectedFile"] for r in all_authored if r["candidateSplit"] in ("train", "validation")
    }
    distinct = defaultdict(set)
    contrast = defaultdict(list)
    output_occurrences = defaultdict(Counter)
    diagnostics = defaultdict(Counter)
    samples = defaultdict(list)
    for ident in sorted(by_id):
        row = frozen_rows[ident]
        actual = by_id[ident]["actual"].replace("\r\n", "\n").replace("\r", "\n")
        language = row["language"]
        group = row["contrastGroupId"]
        distinct[language].add(actual)
        contrast[group].append(actual)
        output_occurrences[language][actual] += 1
        expected_context = "account"  # P3-02's only frozen development context
        assert expected_context in row["inputFile"]
        foreign_contexts = [name for name in CONTEXT_NAMES if name != expected_context and name in actual]
        indicators = {
            "unchangedInputFile": actual == row["inputFile"],
            "requestedValueLiteralPresent": row["newText"] in actual,
            "developmentContextLiteralPresent": expected_context in actual,
            "foreignContextLiteralPresent": bool(foreign_contexts),
            "exactPriorTrainOrValidationInputMatch": actual in previous_inputs,
            "exactPriorTrainOrValidationTargetMatch": actual in previous_targets,
            "emptyOutput": actual == "",
        }
        for k, value in indicators.items():
            diagnostics[language][k] += int(value)
        for name in foreign_contexts:
            diagnostics[language]["foreignContext/" + name] += 1
        samples[language].append({
            "recordId": ident, "family": row["editFamily"], "group": group,
            "outputSha256": hashlib.sha256(actual.encode("utf-8")).hexdigest(),
            "requestedValueLiteralPresent": indicators["requestedValueLiteralPresent"],
            "developmentContextLiteralPresent": indicators["developmentContextLiteralPresent"],
            "foreignContextLiterals": foreign_contexts,
            "actualExactlyInput": indicators["unchangedInputFile"],
        })
    groups = []
    for group, outputs in sorted(contrast.items()):
        groups.append({
            "group": group, "language": group.split("-")[1],
            "requests": len(outputs), "uniqueOutputs": len(set(outputs)),
            "identicalAcrossThreeDifferentRequests": len(set(outputs)) == 1,
        })
    if len(groups) != 12 or any(g["requests"] != 3 for g in groups):
        raise ValueError("P3-02-D1 contrast group accounting changed")
    distribution = dict(sorted(Counter(g["uniqueOutputs"] for g in groups).items()))
    return {
        "schemaVersion": 1, "milestone": "P3-02-D1",
        "kind": DIAGNOSTIC_KIND,
        "status": "read-only-diagnostic-complete",
        "sourceCheckpointSha256": EXPECTED_ENDPOINT_SHA256,
        "inputScope": "only-frozen-P3-02-36-development-records",
        "developmentGatePassed": False,
        "trainingPerformed": False, "researchOptimizerUpdates": 0,
        "finalProjectHoldoutOpened": False,
        "causeEstablished": False,
        "limits": (
            "Exact repeats and literal matches are descriptive indicators, not proof of "
            "memorization or a causal account of failed request binding."
        ),
        "totalRecords": 36,
        "perLanguage": {
            language: {
                "tasks": 12, "uniqueOutputs": len(distinct[language]),
                "repeatOccurrencesBeyondFirst": 12 - len(distinct[language]),
                "largestIdenticalOutputCluster":
                    max(output_occurrences[language].values()),
                "indicators": dict(sorted(diagnostics[language].items())),
                "repeatedOutputClusters": [
                    {
                        "occurrences": count,
                        "outputSha256": hashlib.sha256(actual.encode("utf-8")).hexdigest(),
                        "preview": actual[:160],
                    }
                    for actual, count in sorted(
                        output_occurrences[language].items(),
                        key=lambda pair: (-pair[1], pair[0])
                    ) if count > 1
                ],
            } for language in LANGUAGES
        },
        "contrastGroups": groups,
        "contrastGroupUniqueOutputDistribution": distribution,
        "identicalAcrossAllThreeRequests": sum(
            g["identicalAcrossThreeDifferentRequests"] for g in groups
        ),
        "perRecordIndicators": [
            item for language in LANGUAGES for item in samples[language]
        ],
    }


def diagnose_from_file(
    *, report_path: Path = DEFAULT_REPORT,
    frozen_summary_path: Path = DEFAULT_FROZEN_SUMMARY,
) -> dict[str, Any]:
    return diagnose_existing_development(
        _read_json(report_path), _read_json(frozen_summary_path),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only P3-02-D1 diagnosis of frozen P3-02 development responses")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        result = diagnose_from_file(report_path=args.report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"p3-02-d1-diagnose: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
