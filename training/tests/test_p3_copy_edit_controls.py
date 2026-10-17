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
