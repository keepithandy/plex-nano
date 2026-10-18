"""P3-SB01: new frozen 48-case *proposed* source-binding fixture, no model inference."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1] / "phase3" / "copy-fidelity-v1"
SPEC = importlib.util.spec_from_file_location(
    "p3_source_binding_microprobe_test", HERE / "prepare_source_binding_microprobe.py"
)
assert SPEC and SPEC.loader
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


def test_committed_microprobe_is_byte_exact_and_proposal_only():
    original = PREP.OUTPUT.read_bytes()
    assert original == PREP.serialized()
    digest = hashlib.sha256(original).hexdigest()
    assert digest == "0a30df999f8e1c0f12cf0f6e4a71e0eab814bdd99b3bd83d0e2184637ad31fda"
    manifest = PREP.make_manifest()
    assert manifest["status"] == "frozen-proposal-only-evaluation-not-authorized"
    assert manifest["authorization"] == {
        "modelLoaded": False, "trainingPerformed": False,
        "optimizerUpdates": 0, "evaluationAuthorized": False,
        "finalHoldoutOpened": False,
    }
    assert manifest["originalResearchCheckpointSha256"] == (
        "c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230"
    )
    assert manifest["generationProposalNotAuthorized"] == {
        "temperature": 0, "seed": 1337, "maximumNewTokens": 192,
        "checkpointReselectionAllowed": False,
    }


def test_exact_counts_and_complete_contrast_groups():
    rows = PREP.records()
    assert len(rows) == len({r["id"] for r in rows}) == 48
    assert Counter(r["condition"] for r in rows) == {"copy": 12, "edit": 36}
    assert Counter(r["language"] for r in rows) == {
        "html": 16, "css": 16, "javascript": 16,
    }
    assert Counter(r["context"] for r in rows) == {
        "minimal": 24, "context": 24,
    }
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["groupId"]].append(row)
    assert len(grouped) == 12
    for group in grouped.values():
        assert len(group) == 4
        assert {r["contrastIndex"] for r in group} == {0, 1, 2, 3}
        assert len({r["inputFile"] for r in group}) == 1
        assert len({r["expectedFile"] for r in group}) == 4
        assert {r["condition"] for r in group} == {"copy", "edit"}


def test_exact_copy_edit_and_namespace_relation():
    rows = PREP.records()
    for r in rows:
        assert r["sourceNamespace"] in r["inputFile"]
        assert r["sourceNamespace"] in r["expectedFile"]
        assert r["prompt"].endswith(r["inputFile"] + "\nEdited file:")
        assert len(r["prompt"].encode("utf-8")) + 192 + 1 <= 512
        assert len(r["expectedFile"].encode("utf-8")) + 1 <= 192
        assert r["inputFile"].count(r["sourceLiteral"]) == 1
        if r["condition"] == "copy":
            assert r["requestedLiteral"] is None
            assert r["expectedFile"] == r["inputFile"]
        else:
            assert r["expectedFile"] == (
                r["inputFile"][:r["targetStart"]] + r["requestedLiteral"]
                + r["inputFile"][r["targetEnd"]:]
            )


def test_prior_corpora_are_sha_pinned_and_new_names_never_occur():
    old_sources, old_targets = PREP._checked_prior()
    assert len(old_sources) > 0 and len(old_targets) > 0
    new_sources = {r["inputFile"] for r in PREP.records()}
    new_targets = {r["expectedFile"] for r in PREP.records()}
    assert not ((new_sources | new_targets) & (old_sources | old_targets))
    for name in PREP.NAMES:
        assert all(name not in text for text in old_sources | old_targets)
    assert PREP.make_manifest()["counts"]["overlapWithPreviousFiles"] == {
        "newInputsInPreviousInputs": 0, "newInputsInPreviousTargets": 0,
        "newTargetsInPreviousInputs": 0, "newTargetsInPreviousTargets": 0,
    }


def test_mutated_frozen_corpus_is_rejected_before_microprobe_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prior = next(iter(PREP.PRIOR))
    forged = tmp_path / prior.name
    forged.write_bytes(prior.read_bytes() + b" ")
    monkeypatch.setattr(PREP, "PRIOR", {forged: PREP.PRIOR[prior]})
    with pytest.raises(ValueError, match="SHA changed"):
        PREP.make_manifest()


def test_module_is_strictly_no_training_and_no_inference():
    source = (HERE / "prepare_source_binding_microprobe.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "read_checkpoint(" not in source
    assert "optimizer.step(" not in source
    assert "_generate_token_ids(" not in source
    assert "with OUTPUT.open(\"xb\")" in source
    assert "evaluationAuthorized" in source
    assert "modelLoaded" in source
