"""P3 copy-fidelity candidate: read-only, model-free deterministic regressions."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "phase3" / "copy-fidelity-v1"
REPO = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("copy_fidelity_preparation", HERE / "prepare_copy_fidelity.py")
assert SPEC and SPEC.loader
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)
FROZEN = HERE / "copy-fidelity-candidate-v1.json"
EXPECTED_SHA256 = "9414d1da7672a386a920d1935723e9133d011c9c3a4029e5d5800f643835ca7a"


def test_candidate_frozen_sha_and_byte_reproducibility() -> None:
    raw = FROZEN.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
    assert raw == PREP.serialized()
    assert b"\r" not in raw
    manifest = json.loads(raw)
    assert manifest["authorization"] == {
        "dataPreparationOnly": True, "trainingAuthorized": False,
        "evaluationAuthorized": False, "trainingPerformed": False,
        "optimizerUpdatesPerformed": 0, "finalHoldoutOpened": False,
    }
    assert manifest["model"]["startingCheckpointSha256"] == PREP.STEP100_SHA256
    assert manifest["model"]["tokenizerSha256"] == PREP.TOKENIZER_SHA256
    assert manifest["proposedRunNotAuthorized"]["checkpointReselectionAllowed"] is False
    assert manifest["proposedRunNotAuthorized"]["resumeOrAutomaticContinuation"] is False


def test_candidate_exact_distribution_by_language_condition_context() -> None:
    rows = PREP.rows()
    assert len(rows) == 384
    assert len({r["id"] for r in rows}) == 384
    assert Counter(r["split"] for r in rows) == {"train": 288, "validation": 96}
    for split, per_language in (("train", 96), ("validation", 32)):
        for language in PREP.LANGUAGES:
            subset = [r for r in rows if r["split"] == split and r["language"] == language]
            assert len(subset) == per_language
            assert Counter(r["condition"] for r in subset) == {
                "copy": per_language // 4, "edit": per_language * 3 // 4,
            }
            assert Counter(r["context"] for r in subset) == {
                "minimal": per_language // 2, "context": per_language // 2,
            }


def test_four_case_groups_keep_full_source_in_same_split() -> None:
    groups = defaultdict(list)
    for row in PREP.rows():
        groups[row["groupId"]].append(row)
    assert len(groups) == 96
    for group in groups.values():
        assert len(group) == 4
        assert {r["contrastIndex"] for r in group} == {0, 1, 2, 3}
        assert {r["condition"] for r in group} == {"copy", "edit"}
        assert len({r["inputFile"] for r in group}) == 1
        assert len({r["expectedFile"] for r in group}) == 4
        assert len({r["split"] for r in group}) == 1
        assert len({r["namespace"] for r in group}) == 1


def test_exact_copy_and_single_literal_edit() -> None:
    for r in PREP.rows():
        source = r["inputFile"]
        assert source.count(r["sourceLiteral"]) == 1
        if r["condition"] == "copy":
            assert r["expectedFile"] == source
            assert r["requestedLiteral"] is None
        else:
            assert r["expectedFile"] != source
            assert r["expectedFile"] == (
                source[:r["targetStart"]] + r["requestedLiteral"] + source[r["targetEnd"]:]
            )
            assert r["requestedLiteral"] not in (
                r["sourceLiteral"], None,
            )


def test_train_validation_and_historical_sources_do_not_overlap() -> None:
    details = PREP.check_rows(PREP.rows())
    assert details["splitOverlap"] == {
        "inputFile": 0, "expectedFile": 0, "prompt": 0, "groupId": 0,
    }
    assert details["historicalCompleteFileOverlap"] == 0
    assert set(PREP.TRAIN_NAMES).isdisjoint(PREP.VALIDATION_NAMES)
    assert set(PREP.TRAIN_NAMES + PREP.VALIDATION_NAMES).isdisjoint(("cobalt", "spruce"))
    for split in ("train", "validation"):
        names = set(PREP.TRAIN_NAMES if split == "train" else PREP.VALIDATION_NAMES)
        assert {r["namespace"] for r in PREP.rows() if r["split"] == split} == names


def test_byte_conservative_frozen_token_budget() -> None:
    for row in PREP.rows():
        assert row["prompt"] == PREP.PROMPT.format(
            language=row["language"], request=row["request"], input_file=row["inputFile"],
        )
        assert row["prompt"].isascii() and row["expectedFile"].isascii()
        assert len(row["prompt"].encode("utf-8")) + PREP.MAX_NEW_TOKENS + 1 <= 512
        assert len(row["expectedFile"].encode("utf-8")) + 1 <= PREP.MAX_NEW_TOKENS
    details = PREP.check_rows(PREP.rows())
    assert details["maximumPromptBytes"] == 313
    assert details["maximumAnswerBytes"] == 127


def test_frozen_bytes_survive_windows_git_filters() -> None:
    path = FROZEN.relative_to(REPO).as_posix()
    checkout = subprocess.check_output(
        ["git", "-c", "core.autocrlf=true", "cat-file", "--filters", f"HEAD:{path}"],
        cwd=REPO,
    )
    assert hashlib.sha256(checkout).hexdigest() == EXPECTED_SHA256


def test_deterministic_builder_refuses_overwrite_and_detects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "candidate.json"
    target.write_bytes(b"existing owner data")
    monkeypatch.setattr("sys.argv", ["prepare_copy_fidelity.py", "--output", str(target)])
    with pytest.raises(FileExistsError):
        PREP.main()
    assert target.read_bytes() == b"existing owner data"
    monkeypatch.setattr("sys.argv", [
        "prepare_copy_fidelity.py", "--verify", "--output", str(target),
    ])
    with pytest.raises(SystemExit):
        PREP.main()
    assert target.read_bytes() == b"existing owner data"
    target.write_bytes(PREP.serialized())
    PREP.main()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == EXPECTED_SHA256


def test_frozen_prior_hash_mismatch_blocks_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    injected = tmp_path / "modified-prior.json"
    injected.write_bytes(b"{}")
    monkeypatch.setattr(PREP, "P302", injected)
    monkeypatch.setattr(PREP, "FROZEN_SOURCES", {
        injected: "0" * 64,
        PREP.P303: PREP.FROZEN_SOURCES[PREP.P303],
        PREP.COPY_CONTROL: PREP.FROZEN_SOURCES[PREP.COPY_CONTROL],
    })
    with pytest.raises(ValueError, match="Historical frozen source hash changed"):
        PREP.check_rows(PREP.rows())
