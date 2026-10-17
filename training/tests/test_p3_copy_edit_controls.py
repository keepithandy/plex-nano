"""Pure-Python, no-inference checks for P3 copy-versus-edit control preparation."""
import hashlib
import importlib.util
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

PACKET = Path(__file__).resolve().parents[1] / "phase3" / "diagnostics" / "p3-copy-edit-controls-v1"
SPEC = importlib.util.spec_from_file_location("copy_edit_control_prep", PACKET / "prepare_controls.py")
assert SPEC is not None and SPEC.loader is not None
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


def test_fixed_controls_and_prior_overlap_identity() -> None:
    manifest = PREP.make_manifest()
    rows = manifest["tasks"]
    assert len(rows) == 42
    assert len({row["taskId"] for row in rows}) == 42
    assert Counter(row["condition"] for row in rows) == {
        "copy": 6, "minimal-edit": 18, "context-edit": 18,
    }
    assert all(sum(row["language"] == lang and row["condition"] == cond for row in rows) == n
               for lang in ("html", "css", "javascript")
               for cond, n in (("copy", 2), ("minimal-edit", 6), ("context-edit", 6)))
    assert len({r["contrastGroupId"] for r in rows}) == 18
    assert set(manifest["overlapWithFrozenP302TrainAndP303Diagnostic"].values()) == {0}
    assert manifest["authorization"]["evaluationAuthorized"] is False
    assert manifest["authorization"]["trainingAuthorized"] is False


def test_same_request_three_different_values_per_edit_group() -> None:
    rows = PREP.task_rows()
    for condition in ("minimal-edit", "context-edit"):
        for lang in ("html", "css", "javascript"):
            for ns in PREP.NAMES:
                group = [r for r in rows if (r["condition"], r["language"], r["namespace"]) ==
                         (condition, lang, ns)]
                assert len(group) == 3
                assert len({r["inputFile"] for r in group}) == 1
                assert len({r["expectedFile"] for r in group}) == 3
                assert len({r["request"] for r in group}) == 3


@pytest.mark.parametrize("condition", ["copy", "minimal-edit", "context-edit"])
@pytest.mark.parametrize("language", ["html", "css", "javascript"])
def test_expected_complete_files_pass_all_applicable_checks(condition: str, language: str) -> None:
    selected = [t for t in PREP.task_rows() if t["condition"] == condition and t["language"] == language]
    assert selected
    for task in selected:
        result = PREP.score_one(task, task["expectedFile"])
        assert result["exact"] and result["unrelatedBytesPreserved"]
        assert result["requestedValueCorrect"] is (None if condition == "copy" else True)


def test_wrong_requested_value_is_not_confused_with_unrelated_preservation() -> None:
    for task in (t for t in PREP.task_rows() if t["condition"] != "copy"):
        old = task["inputFile"]
        result = PREP.score_one(task, old)
        assert not result["exact"]
        assert result["unrelatedBytesPreserved"]
        assert result["requestedValueCorrect"] is False


def test_changed_unrelated_context_is_scoring_ambiguous_not_value_failure() -> None:
    for task in (t for t in PREP.task_rows() if t["condition"] == "context-edit"):
        changed = task["expectedFile"].replace(task["namespace"], "unexpected", 1)
        result = PREP.score_one(task, changed)
        assert not result["exact"]
        assert result["unrelatedBytesPreserved"] is False
        assert result["requestedValueCorrect"] is None


def test_copy_controls_measure_exact_bytes_not_namespace_substrings() -> None:
    for task in (t for t in PREP.task_rows() if t["condition"] == "copy"):
        actual = task["expectedFile"] + "\n"
        assert task["namespace"] in actual
        assert PREP.score_one(task, actual)["exact"] is False


def test_builder_never_overwrites_existing_frozen_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "frozen.json"
    sentinel = b'{"ownerData": true}\n'
    target.write_bytes(sentinel)
    monkeypatch.setattr("sys.argv", ["prepare_controls.py", "--output", str(target)])
    with pytest.raises(FileExistsError):
        PREP.main()
    assert target.read_bytes() == sentinel


def test_explicit_verify_does_not_rewrite_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "frozen.json"
    frozen = PREP.serialized_manifest()
    target.write_bytes(frozen)
    monkeypatch.setattr("sys.argv", ["prepare_controls.py", "--output", str(target), "--verify"])
    PREP.main()
    assert target.read_bytes() == frozen
    json.loads(target.read_text(encoding="utf-8"))
    target.write_bytes(frozen + b" ")
    with pytest.raises(SystemExit):
        PREP.main()

FROZEN_MANIFEST = PACKET / "copy-edit-controls-v1.json"
FROZEN_SHA256 = "17834efb801ac963a4672ae616ca47e9eea40b2568f01951c3891c7e4997f01a"


def test_frozen_manifest_matches_builder_and_hash() -> None:
    raw = FROZEN_MANIFEST.read_bytes()
    assert raw == PREP.serialized_manifest()
    assert hashlib.sha256(raw).hexdigest() == FROZEN_SHA256
    assert b"\\r\\n" not in raw
    frozen = json.loads(raw)
    assert frozen["authorization"]["evaluationAuthorized"] is False
    assert frozen["authorization"]["optimizerUpdates"] == 0


def test_all_prompts_fit_conservative_byte_bpe_budget() -> None:
    # Byte-level BPE starts from individual UTF-8 bytes and merges them.
    # Even with no merges, the 192 requested tokens + EOS remain in context.
    frozen = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    for row in frozen["tasks"]:
        assert len(row["prompt"].encode("utf-8")) + 192 + 1 <= 512
        assert len(row["inputFile"].encode("utf-8")) == len(row["inputFile"])
        assert row["prompt"] == PREP.PROMPT.format(
            language=row["language"], request=row["request"], input_file=row["inputFile"],
        )


def test_frozen_manifest_windows_checkout_maintains_sha256() -> None:
    repo = PACKET.parents[3]
    path = FROZEN_MANIFEST.relative_to(repo).as_posix()
    checked_out = subprocess.check_output(
        ["git", "-c", "core.autocrlf=true", "cat-file", "--filters", f"HEAD:{path}"],
        cwd=repo,
    )
    assert hashlib.sha256(checked_out).hexdigest() == FROZEN_SHA256


SCORE_SPEC = importlib.util.spec_from_file_location(
    "copy_edit_control_score", PACKET / "score_frozen_responses.py",
)
assert SCORE_SPEC is not None and SCORE_SPEC.loader is not None
SCORE = importlib.util.module_from_spec(SCORE_SPEC)
SCORE_SPEC.loader.exec_module(SCORE)


def _synthetic_envelope() -> dict:
    from hashlib import sha256
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    return {
        "schemaVersion": 1,
        "manifestSha256": FROZEN_SHA256,
        "evaluationAuthorization": {
            "status": "separately-authorized",
            "evidenceReference": "synthetic-unit-test-only",
        },
        "execution": {
            "checkpointStep": 100, "checkpointSha256": SCORE.CHECKPOINT_SHA256,
            "checkpointHashVerified": True,
            "tokenizerSha256": SCORE.TOKENIZER_SHA256,
            "p302BundleManifestSha256": SCORE.BUNDLE_SHA256,
            "policy": "greedy-unreselected-fixed-endpoint-v1",
            "seed": 1337, "temperature": 0,
            "maximumNewTokens": 192, "checkpointReselectionAllowed": False,
            "trainingPerformedDuringDiagnostic": False,
            "optimizerUpdatesDuringDiagnostic": 0,
            "finalProjectHoldoutOpened": False,
            "promptTemplateSha256": sha256(PREP.PROMPT.encode()).hexdigest(),
            "codeRevision": "synthetic", "runnerSha256": "synthetic", "device": "synthetic",
        },
        "responses": [
            {"taskId": t["taskId"], "actual": t["expectedFile"]}
            for t in manifest["tasks"]
        ],
        "generationAudit": [
            {"taskId": t["taskId"], "generatedTokenCount": 10,
             "stoppedAtEos": True, "fullPromptVisibleForEveryGenerationStep": True}
            for t in manifest["tasks"]
        ],
    }


def test_scorer_perfect_synthetic_envelope_and_reporting() -> None:
    summary = SCORE.score(SCORE.read_manifest(), _synthetic_envelope())
    assert summary["exact"] == 42
    assert summary["byConditionExact"] == {
        "copy": 6, "minimal-edit": 18, "context-edit": 18,
    }
    assert summary["fullyCorrectEditContrastGroups"] == 12
    assert summary["decision"] == "all-controls-passed-require-fresh-independent-confirmation"
    assert len(summary["byLanguageAndCondition"]) == 9
    report = SCORE.report_md(summary)
    assert "| html/minimal-edit | 6 | 6 |" in report
    assert "| javascript/context-edit | 6 | 6 |" in report


@pytest.mark.parametrize(
    ("condition", "decision"),
    [
        ("copy", "copy-control-failed-review-source-copying"),
        ("minimal-edit", "minimal-edit-failed-review-target-and-request-binding"),
        ("context-edit", "context-edit-failed-review-unrelated-code-preservation"),
    ],
)
def test_scorer_decision_follows_predeclared_failure_order(condition, decision) -> None:
    frozen = SCORE.read_manifest()
    env = _synthetic_envelope()
    index = next(i for i, t in enumerate(frozen["tasks"]) if t["condition"] == condition)
    env["responses"][index]["actual"] = "not the expected file"
    result = SCORE.score(frozen, env)
    assert result["decision"] == decision
    assert result["exact"] == 41


def test_scorer_changed_target_boundaries_are_unscorable() -> None:
    frozen = SCORE.read_manifest()
    env = _synthetic_envelope()
    index = next(i for i, t in enumerate(frozen["tasks"]) if t["condition"] == "context-edit")
    env["responses"][index]["actual"] = "completely different content"
    result = SCORE.score(frozen, env)
    record = result["perTask"][index]
    assert record["unrelatedBytesPreserved"] is False
    assert record["requestedValueCorrect"] is None
    assert result["byLanguageAndCondition"][record["language"] + "/context-edit"]["requestedValueUnscorable"] == 1


@pytest.mark.parametrize("case", [
    "missing", "duplicate", "wrong-id", "not-string", "wrong-checkpoint",
    "invalid-authorization", "non-eos-boolean", "prompt-eviction",
    "unexpected-training", "wrong-manifest", "wrong-step-type",
])
def test_scorer_rejects_invalid_envelopes(case: str) -> None:
    frozen = SCORE.read_manifest()
    env = _synthetic_envelope()
    if case == "missing":
        env["responses"].pop()
    elif case == "duplicate":
        env["responses"][1]["taskId"] = env["responses"][0]["taskId"]
    elif case == "wrong-id":
        env["responses"][0]["taskId"] = "unfrozen-task"
    elif case == "not-string":
        env["responses"][0]["actual"] = None
    elif case == "wrong-checkpoint":
        env["execution"]["checkpointSha256"] = "bad"
    elif case == "invalid-authorization":
        env["evaluationAuthorization"]["status"] = "unapproved"
    elif case == "non-eos-boolean":
        env["generationAudit"][0]["stoppedAtEos"] = 1
    elif case == "prompt-eviction":
        env["generationAudit"][0]["fullPromptVisibleForEveryGenerationStep"] = False
    elif case == "unexpected-training":
        env["execution"]["optimizerUpdatesDuringDiagnostic"] = 1
    elif case == "wrong-manifest":
        env["manifestSha256"] = "bad"
    else:
        env["execution"]["checkpointStep"] = True
    with pytest.raises(ValueError):
        SCORE.score(frozen, env)


def test_scorer_requires_eos_for_exact_credit() -> None:
    env = _synthetic_envelope()
    env["generationAudit"][0]["stoppedAtEos"] = False
    result = SCORE.score(SCORE.read_manifest(), env)
    assert result["exact"] == 41
    assert result["perTask"][0]["rawTextExact"]
    assert not result["perTask"][0]["exact"]


def test_scorer_writes_distinct_read_only_report_once(tmp_path: Path, monkeypatch) -> None:
    input_path, output = tmp_path / "responses.json", tmp_path / "report"
    input_path.write_text(json.dumps(_synthetic_envelope()), encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "score_frozen_responses.py", "--responses", str(input_path),
        "--output-dir", str(output),
    ])
    SCORE.main()
    assert json.loads((output / "audit.json").read_text())["exact"] == 42
    assert "42/42" in (output / "report.md").read_text()
    with pytest.raises(SystemExit):
        SCORE.main()
