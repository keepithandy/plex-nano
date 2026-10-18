"""Read-only P3-SB00 diagnosis of the ONE completed fixed step-100 generation report.

Does NOT load PyTorch, run model inference, train, select checkpoints, rewrite
historical artifacts, or open the sealed final project holdout.

The user's saved perRecord output stays in PlexArtifacts. This module prints
aggregates and the byte SHA-256 of that unmodified input for owner review.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE / "copy-fidelity-candidate-v1.json"
CONTROLS = ROOT / "training/phase3/diagnostics/p3-copy-edit-controls-v1/copy-edit-controls-v1.json"
REPORT = Path.home() / "PlexArtifacts/file-edits/p3-copy-fidelity-v1-step100-evaluation/evaluation-result.json"

CANDIDATE_SHA = "9414d1da7672a386a920d1935723e9133d011c9c3a4029e5d5800f643835ca7a"
CONTROL_SHA = "17834efb801ac963a4672ae616ca47e9eea40b2568f01951c3891c7e4997f01a"
ENDPOINT_SHA = "c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230"
BUNDLE_SHA = "8eceed468fdcb7202484eba82c8ebb8159a0d9e05787538f2c81f06f62719e2d"
TOKENIZER_SHA = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
STAGE_SHA = "3524620d830d363585d501285a9cfdea09d1c2448be08239e2516d78291b3ca9"
CONTRACT_SHA = "a7432a3bfa7942b6086bc8e64fa3b24290351acfd9fb700aac2af613a8c0911b"
TRAIN_NAMES = ("cedar", "lunar", "prism", "amber", "zephyr", "meadow")
VALID_NAMES = ("granite", "saffron")
CONTROL_NAMES = ("cobalt", "spruce")
LANGUAGES = ("html", "css", "javascript")


def read_frozen(path: Path, expected_sha: str, maximum_bytes: int) -> dict[str, Any]:
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size > maximum_bytes):
        raise ValueError("Required frozen file is missing, linked, or too large")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Frozen manifest hash changed")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Frozen manifest is not a JSON object")
    return data


def frozen_rows(candidate_path: Path = CANDIDATE,
                controls_path: Path = CONTROLS) -> tuple[list[dict], list[dict]]:
    source = read_frozen(candidate_path, CANDIDATE_SHA, 4 * 1024 * 1024)
    controls = read_frozen(controls_path, CONTROL_SHA, 1024 * 1024)
    val = [r for r in source["rows"] if r["split"] == "validation"]
    dev = controls["tasks"]
    if (len(source["rows"]) != 384 or len(val) != 96 or len(dev) != 42
            or source["namespaceSplits"] != {
                "train": list(TRAIN_NAMES), "validation": list(VALID_NAMES),
            }
            or Counter(r["condition"] for r in val) != {"copy": 24, "edit": 72}
            or Counter(r["condition"] for r in dev) != {
                "copy": 6, "minimal-edit": 18, "context-edit": 18,
            }
            or {r["namespace"] for r in dev} != set(CONTROL_NAMES)
            or len({r["id"] for r in val}) != 96
            or len({r["taskId"] for r in dev}) != 42):
        raise ValueError("Frozen task set or split composition changed")
    return val, dev


def _result_check(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise ValueError("Input report must be a JSON object")
    expected = {
        "schemaVersion": 1,
        "kind": "p3-copy-fidelity-fixed-step100-generation-v1",
        "status": "fixed-endpoint-generation-evaluation-complete",
        "checkpointStep": 100,
        "checkpointSha256": ENDPOINT_SHA,
        "bundleManifestSha256": BUNDLE_SHA,
        "candidateSha256": CANDIDATE_SHA,
        "controlManifestSha256": CONTROL_SHA,
        "tokenizerSha256": TOKENIZER_SHA,
        "stageCheckpointSha256": STAGE_SHA,
        "authorizationContractSha256": CONTRACT_SHA,
        "trainingPerformed": False,
        "optimizerUpdates": 0,
        "finalHoldoutOpened": False,
        "developmentControlsPreviouslyInspected": True,
    }
    if any(type(report.get(key)) is not type(value) or
           report[key] != value for key, value in expected.items()):
        raise ValueError("Input is not the original fixed-endpoint generation report")
    generation = report.get("generation")
    policy = {"kind": "greedy-complete-file-v1", "maxNewTokens": 192,
              "temperature": 0.0, "seed": 1337, "checkpointReselectionAllowed": False}
    if (not isinstance(generation, dict) or any(
            type(generation.get(k)) is not type(v) or generation[k] != v
            for k, v in policy.items())):
        raise ValueError("Generation policy changed from the frozen evaluation")


def _boundary_score(row: dict, actual: str) -> tuple[bool, bool | None]:
    if row["condition"] == "copy":
        return actual == row["expectedFile"], None
    source = row["inputFile"]
    if "targetStart" in row:
        begin, end, requested = row["targetStart"], row["targetEnd"], row["requestedLiteral"]
    else:
        target = row["target"]
        begin, end, requested = target["startCharacter"], target["endCharacter"], target["requestedLiteral"]
    before, after = source[:begin], source[end:]
    boundaries = actual.startswith(before) and actual.endswith(after)
    slice_end = len(actual) - len(after) if after else len(actual)
    middle = actual[len(before):slice_end] if boundaries else None
    return boundaries, middle == requested if boundaries else None


def analyze(report: dict[str, Any], val: list[dict], dev: list[dict]) -> dict[str, Any]:
    """Pure scorer-derived morphology; no missing-value inference from mismatched text."""
    _result_check(report)
    expected_rows = val + dev
    saved = report.get("perRecord")
    if (len(val) != 96 or len(dev) != 42 or not isinstance(saved, list)
            or len(saved) != 138):
        raise ValueError("Expected exactly 96 validation + 42 development rows")
    by_scope: dict[str, Counter] = defaultdict(Counter)
    by_language: dict[str, Counter] = defaultdict(Counter)
    by_condition: dict[str, Counter] = defaultdict(Counter)
    by_context: dict[str, Counter] = defaultdict(Counter)
    response_counts: Counter[tuple[str, str, str, str]] = Counter()
    illustrative: list[dict[str, Any]] = []
    all_ids = set()
    group_results: dict[tuple[str, str], list[bool]] = defaultdict(list)
    language_exact: Counter[tuple[str, str, str]] = Counter()

    for i, (row, saved_row) in enumerate(zip(expected_rows, saved)):
        scope = "validation" if i < len(val) else "development"
        condition = "copy" if row["condition"] == "copy" else "edit"
        expected_id = row["id"] if scope == "validation" else row["taskId"]
        if (not isinstance(saved_row, dict) or saved_row.get("recordId") != expected_id
                or expected_id in all_ids
                or saved_row.get("scope") != scope
                or saved_row.get("language") != row["language"]
                or saved_row.get("condition") != row["condition"]
                or type(saved_row.get("actual")) is not str
                or saved_row.get("expectedFile") != row["expectedFile"]
                or any(type(saved_row.get(k)) is not bool for k in
                       ("stoppedAtEos", "decodeSucceeded", "fullPromptVisible"))
                or type(saved_row.get("generatedTokenCount")) is not int
                or not 0 <= saved_row["generatedTokenCount"] <= 192
                or (saved_row["stoppedAtEos"] and saved_row["generatedTokenCount"] == 192)):
            raise ValueError("Saved per-record response changed, reordered or malformed")
        all_ids.add(expected_id)
        actual = saved_row["actual"]
        complete = all(saved_row.get(k) is True for k in
                       ("stoppedAtEos", "decodeSucceeded", "fullPromptVisible"))
        boundaries, value_correct = _boundary_score(row, actual)
        exact = complete and actual == row["expectedFile"]
        language_exact[(scope, row["language"], condition)] += exact
        if scope == "validation" or condition == "edit":
            group_id = row["groupId"] if scope == "validation" else row["contrastGroupId"]
            group_results[(scope, group_id)].append(exact)
        preserved = complete and boundaries
        extracted = boundaries if complete and condition == "edit" else (
            None if complete and condition == "copy" else False)
        requested = value_correct if complete else None
        if (saved_row.get("exactCompleteFile") is not exact
                or saved_row.get("unrelatedBytesPreserved") is not preserved
                or saved_row.get("requestedValueCorrect") is not requested
                or saved_row.get("requestedValueExtractable") is not extracted):
            raise ValueError("Raw response is inconsistent with original exact-file scorer")
        name = row["namespace"]
        train_hits = tuple(n for n in TRAIN_NAMES if n in actual)
        same_name = name in actual  # heuristic, NOT proof of exact identifier copying
        literal = (row.get("requestedLiteral") if scope == "validation"
                   else row["target"]["requestedLiteral"])
        literal_mentioned = (
            condition == "edit" and literal is not None and literal in actual
        )  # weak text occurrence, NOT requested-value extraction
        looks_like_code = (
            "<button" in actual and "</button>" in actual if row["language"] == "html"
            else "gap:" in actual and "{" in actual and "}" in actual
            if row["language"] == "css"
            else "const " in actual and ";" in actual
        )  # heuristic shape check, NOT a syntax parser
        facts = {
            "cases": 1,
            "exact": int(exact),
            "preserved": int(preserved),
            "sourceNamespacePresent": int(same_name),
            "trainingNamespaceIntrusion": int(bool(train_hits)),
            "emptyOutput": int(actual == ""),
            "maxTokenBudgetReached": int(saved_row["generatedTokenCount"] == 192),
            "eos": int(saved_row.get("stoppedAtEos") is True),
            "decodeSucceeded": int(saved_row.get("decodeSucceeded") is True),
            "fullPromptVisible": int(saved_row.get("fullPromptVisible") is True),
            "structureHeuristic": int(looks_like_code),
            "requestedLiteralMentioned": int(literal_mentioned),
            "requestedValueExtractable": int(extracted is True),
            "requestedValueCorrectWhenExtractable": int(requested is True),
            "totalGeneratedTokens": saved_row["generatedTokenCount"],
            "totalActualCharacters": len(actual),
        }
        for key, increment in facts.items():
            by_scope[scope][key] += increment
            by_language[f"{scope}.{row['language']}"][key] += increment
            by_condition[f"{scope}.{condition}"][key] += increment
            context = row["context"] if scope == "validation" else row["condition"]
            by_context[f"{scope}.{context}"][key] += increment
        response_counts[(scope, row["language"], condition, actual)] += 1
        if train_hits:
            for hit in train_hits:
                by_scope[scope][f"trainName:{hit}"] += 1
        if len(illustrative) < 12 and (not exact or not preserved):
            illustrative.append({
                "recordId": expected_id, "language": row["language"],
                "scope": scope, "condition": condition,
                "expected": row["expectedFile"][:160], "actual": actual[:160],
                "sourceName": name, "sourceNamePresent": same_name,
                "trainingNameHits": list(train_hits),
                "requestedLiteralMentioned": literal_mentioned,
                "requestedValueCorrectWhenExtractable": requested,
            })

    # Reconstruct EVERY original aggregate, including gates, from raw responses.
    # Do not trust a saved pass flag merely because its per-case counts agree.
    gates_by_scope = {}
    for scope, source in (("validation", report.get("validation")),
                          ("development", report.get("development"))):
        if not isinstance(source, dict):
            raise ValueError("Saved aggregate scores are missing")
        c = by_scope[scope]
        copies = by_condition[f"{scope}.copy"]
        edits = by_condition[f"{scope}.edit"]
        is_val = scope == "validation"
        groups = [g for (s, _), g in group_results.items() if s == scope]
        if (len(groups) != (24 if is_val else 12)
                or any(len(g) != (4 if is_val else 3) for g in groups)):
            raise ValueError("Frozen contrast group composition changed")
        correct_groups = sum(all(g) for g in groups)
        # Language totals and copy/edit counts come from the frozen task rows.
        per_language = {
            lang: {kind: {
                "exact": language_exact[(scope, lang, "copy" if kind == "copies" else "edit")],
                "total": (8 if is_val else 2) if kind == "copies" else (24 if is_val else 12),
            } for kind in ("copies", "edits")}
            for lang in LANGUAGES
        }
        gates = ({
            "copy24of24": copies["exact"] == 24,
            "editsAtLeast60of72": edits["exact"] >= 60,
            "perLanguageAtLeast20of24Edits": all(
                per_language[lang]["edits"]["exact"] >= 20 for lang in LANGUAGES),
            "groupsAtLeast18of24": correct_groups >= 18,
        } if is_val else {
            "copy6of6": copies["exact"] == 6,
            "editsAtLeast30of36": edits["exact"] >= 30,
        })
        derived = {
            "cases": c["cases"], "copies": {"exact": copies["exact"], "total": copies["cases"]},
            "edits": {"exact": edits["exact"], "total": edits["cases"]},
            "perLanguage": per_language, "preservedSourceCount": c["preserved"],
            "eosCount": c["eos"], "fullPromptVisibleCount": c["fullPromptVisible"],
            "gates": gates,
            "fullyCorrectContrastGroups" if is_val else "fullyCorrectEditContrastGroups": correct_groups,
            "contrastGroups" if is_val else "editContrastGroups": len(groups),
        }
        gates_by_scope[scope] = all(gates.values())
        # JSON equality distinguishes bool/int/float (Python dict equality does not).
        if json.dumps(source, sort_keys=True) != json.dumps(derived, sort_keys=True):
            raise ValueError("Saved summary disagrees with original raw file outputs")
        key = "allValidationGatesPassed" if is_val else "allDevelopmentGatesPassed"
        if report.get(key) is not gates_by_scope[scope]:
            raise ValueError("Saved summary disagrees with derived capability gates")
    duplicates = {
        "reusedOutputRecords": sum(n for n in response_counts.values() if n > 1),
        "reusedOutputGroups": sum(1 for n in response_counts.values() if n > 1),
        "uniqueOutputsWithinScopeLanguageCondition": len(response_counts),
    }
    return {
        "status": "p3-source-binding-read-only-morphology-reviewed",
        "sourceCheckpointSha256": ENDPOINT_SHA,
        "totalCases": 138,
        "allOriginalGeneratedFileGatesPassed": (
            gates_by_scope["validation"] and gates_by_scope["development"]
        ),
        "trainNamespaceIntrusionIsHeuristic": True,
        "sourceNamespacePresentIsNotExactPreservation": True,
        "requestedLiteralMentionIsNotProvenValueBinding": True,
        "byScope": {k: dict(v) for k, v in sorted(by_scope.items())},
        "byLanguage": {k: dict(v) for k, v in sorted(by_language.items())},
        "byCondition": {k: dict(v) for k, v in sorted(by_condition.items())},
        "byContext": {k: dict(v) for k, v in sorted(by_context.items())},
        "responseRepetition": duplicates,
        "illustrativeFailures": illustrative,
        "trainingPerformed": False, "optimizerUpdates": 0,
        "finalHoldoutOpened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT,
                        help="Already generated local evaluation-result.json; NEVER mutate")
    args = parser.parse_args()
    try:
        if (args.report.is_symlink() or not args.report.is_file()
                or args.report.stat().st_size > 2 * 1024 * 1024):
            raise ValueError("Missing, linked or oversized existing evaluation report")
        raw = args.report.read_bytes()
        val, dev = frozen_rows()
        output = analyze(json.loads(raw), val, dev)
        output["inputReportSha256"] = hashlib.sha256(raw).hexdigest()
        print(json.dumps(output, sort_keys=True, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"p3-source-binding-diagnostic: {exc}\n")


if __name__ == "__main__":
    main()
