"""Controlled, model-free checks of the frozen P3-03 diagnostic decision axes."""

import importlib.util
import json
from pathlib import Path

import pytest


PACKET = Path(__file__).resolve().parents[1] / "phase3" / "diagnostics" / "p3-03-source-preservation-v1"
SPEC = importlib.util.spec_from_file_location("p303_diagnostic", PACKET / "analyze_responses.py")
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)
MANIFEST = json.loads((PACKET / "p3-03-source-preservation-v1.json").read_text(encoding="utf-8"))
FAMILY_TASKS = list({
    (task["language"], task["family"]): task
    for task in reversed(MANIFEST["tasks"])
    if task["sourceNamespace"] == "harbor"
}.values())
assert len(FAMILY_TASKS) == 12


@pytest.mark.parametrize(
    ("change_name", "change_value", "expected_outcome", "expected_name", "expected_value"),
    [
        (False, False, "both-pass", True, True),
        (True, False, "preservation-fails-value-passes", False, True),
        (False, True, "preservation-passes-value-fails", True, False),
        (True, True, "both-fail", False, False),
    ],
)
@pytest.mark.parametrize("task", FAMILY_TASKS,
                         ids=[f"{task['language']}-{task['family']}" for task in FAMILY_TASKS])
def test_name_and_value_axes_are_independent(
    task: dict,
    change_name: bool, change_value: bool, expected_outcome: str,
    expected_name: bool, expected_value: bool,
) -> None:
    responses = [{"taskId": task["taskId"], "actual": task["expectedFile"]}
                 for task in MANIFEST["tasks"]]
    target_response = next(response for response in responses if response["taskId"] == task["taskId"])
    actual = target_response["actual"]
    if change_name:
        actual = actual.replace("harbor", "settings")
    if change_value:
        requested = task["target"]["requestedLiteral"]
        other = next(member["target"]["requestedLiteral"] for member in MANIFEST["tasks"]
                     if member["contrastGroupId"] == task["contrastGroupId"]
                     and member["target"]["requestedLiteral"] != requested)
        start = task["target"]["startCharacter"]
        end = task["target"]["endCharacter"]
        source = task["inputFile"]
        replacement = source[:start] + other + source[end:]
        actual = replacement.replace("harbor", "settings") if change_name else replacement
    target_response["actual"] = actual

    summary, audit = ANALYZER._score(MANIFEST, {"responses": responses}, set())
    observed = next(record for record in audit["tasks"] if record["taskId"] == task["taskId"])

    assert observed["sourceNamePreservation"] is expected_name
    assert observed["requestedValueSelection"]["pass"] is expected_value
    assert summary["decision"]["outcome"] == expected_outcome


def test_unlocatable_target_does_not_prove_unrelated_code_changed() -> None:
    task = next(task for task in MANIFEST["tasks"]
                if task["language"] == "javascript" and task["family"] == "retry-limit"
                and task["sourceNamespace"] == "harbor"
                and "= 3;" in task["expectedFile"])
    responses = [{"taskId": task["taskId"], "actual": task["expectedFile"]}
                 for task in MANIFEST["tasks"]]
    target_response = next(response for response in responses if response["taskId"] == task["taskId"])
    target_response["actual"] = target_response["actual"].replace("= 3;", "= unknown;", 1)

    summary, audit = ANALYZER._score(MANIFEST, {"responses": responses}, set())
    observed = next(record for record in audit["tasks"] if record["taskId"] == task["taskId"])

    assert observed["sourceNamePreservation"] is True
    assert observed["requestedValueSelection"]["pass"] is False
    assert summary["decision"]["unscorablePreservationTasks"] == 1
    assert summary["decision"]["preservationFailureObserved"] is False
    assert summary["decision"]["outcome"] == "scoring-ambiguous"


def test_report_retains_language_and_family_breakdowns() -> None:
    responses = [{"taskId": task["taskId"], "actual": task["expectedFile"]}
                 for task in MANIFEST["tasks"]]
    summary, audit = ANALYZER._score(MANIFEST, {"responses": responses}, set())
    execution = {
        "checkpointStep": 100, "checkpointSha256": "synthetic",
        "tokenizerSha256": "synthetic", "p302BundleManifestSha256": "synthetic",
        "policy": "synthetic", "seed": 0, "temperature": 0,
        "maximumNewTokens": 0, "codeRevision": "synthetic",
        "evaluationAuthorizationEvidenceReference": "synthetic-test-only",
    }
    report = ANALYZER._markdown_report(MANIFEST, "synthetic", execution,
                                       "synthetic", "synthetic", summary, audit)

    assert "| html | 24 | 24 | 24 | 24 | 0 | 24 |" in report
    assert "| css | 24 | 24 | 24 | 24 | 0 | 24 |" in report
    assert "| javascript | 24 | 24 | 24 | 24 | 0 | 24 |" in report
    assert "| html / button-text | 6 | 6 | 6 | 6 | 0 | 6 |" in report
    assert "| css / gap | 6 | 6 | 6 | 6 | 0 | 6 |" in report
    assert "| javascript / retry-limit | 6 | 6 | 6 | 6 | 0 | 6 |" in report


@pytest.mark.parametrize("task", [t for t in FAMILY_TASKS if t["language"] == "javascript"],
                         ids=lambda t: t["family"])
@pytest.mark.parametrize("wrapper", ["comment", "template", "unclosed-comment"])
def test_inactive_javascript_is_not_scored_as_declarations(task, wrapper):
    first, declarations = task["expectedFile"].split("\n", 1)
    if wrapper == "comment":
        actual = first + "\n/*\n" + declarations + "*/\n"
    elif wrapper == "template":
        actual = first + "\nconst example = `\n" + declarations + "`;\n"
    else:
        actual = first + "\n/*\n" + declarations
    extracted = ANALYZER._extract_target(task, actual)
    roles = ANALYZER._preservation_observations(task, actual)
    assert extracted["valid"] is False
    assert extracted["reason"] == "target-declaration-missing"
    assert not all(role["pass"] for role in roles)


def test_commented_duplicate_does_not_hide_active_target_or_shift_spans():
    task = next(t for t in FAMILY_TASKS if t["family"] == "retry-limit")
    actual = "/*\nconst harborRetries = 999;\n*/\n" + task["expectedFile"]
    target = ANALYZER._extract_target(task, actual)
    assert target["valid"]
    assert target["value"] == task["target"]["requestedLiteral"]
    assert actual[target["startCharacter"]:target["endCharacter"]] == target["value"]


@pytest.mark.parametrize("literal", ['"https://example.test/*path*/"', '"a\\\"//b"'])
def test_comment_markers_in_strings_do_not_hide_following_declarations(literal):
    text = f"const harborMessage = {literal};\nconst harborEnabled = true;\n"
    declarations = ANALYZER._js_declarations(text)
    assert [d["name"] for d in declarations] == ["harborMessage", "harborEnabled"]
    assert declarations[0]["initializer"] == literal


def test_windows_checkout_preserves_frozen_manifest_hash():
    import hashlib
    import subprocess
    root = PACKET.parents[3]
    relative = (PACKET / "p3-03-source-preservation-v1.json").relative_to(root).as_posix()
    checkout = subprocess.check_output(
        ["git", "-c", "core.autocrlf=true", "cat-file", "--filters", f"HEAD:{relative}"],
        cwd=root,
    )
    assert b"\r\n" not in checkout
    assert hashlib.sha256(checkout).hexdigest() == ANALYZER.EXPECTED_MANIFEST_SHA256


@pytest.mark.parametrize("kind", ["missing", "invalid-json", "changed-manifest"])
def test_cli_reports_input_errors_without_traceback(tmp_path, capsys, kind):
    bad = tmp_path / "input.json"
    if kind == "invalid-json":
        bad.write_text("{broken", encoding="utf-8")
    elif kind == "changed-manifest":
        bad.write_text("{}", encoding="utf-8")
    report, audit = tmp_path / "report.md", tmp_path / "audit.json"
    args = ["--responses", str(bad), "--report", str(report), "--audit", str(audit)]
    if kind == "changed-manifest":
        args += ["--manifest", str(bad)]
    with pytest.raises(SystemExit) as error:
        ANALYZER.main(args)
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("p3-03-analyze: ")
    assert "Traceback" not in captured.err
    assert not report.exists() and not audit.exists()
