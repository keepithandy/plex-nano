"""P3-SB03 V2: read-only tokenizer-aware candidate search and refusal gates.

These tests use synthetic tokenizers only. The real frozen Plex tokenizer exists
in the owner's local artifact directory, not GitHub CI.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "phase3" / "copy-fidelity-v1"


def _load(file: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V2 = _load("review_namespace_diversity_v2.py", "p3sb03_v2_feasibility_test")


class FakeBackend:
    def get_vocab(self):
        return {
            "taper": 25, "cycle": 30, "other": 31,
            "marble": 32, "purple": 33, "butter": 34,
            "cedar": 35, "meadow": 36, "x" * 4: 3,
        }


class ByteTokenizer:
    vocabulary_size = 16384
    backend = FakeBackend()

    def encode(self, text):
        return [ord(c) + 4 for c in text]

    def decode(self, ids):
        return "".join(chr(t - 4) for t in ids)


class NonMatchingTokenizer(ByteTokenizer):
    """Old cedar is one token; random replacements do not match token count."""
    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            if text.startswith("cedar", i):
                tokens.append(1000)
                i += 5
            else:
                tokens.append(ord(text[i]) + 4)
                i += 1
        return tokens

    def decode(self, ids):
        return "".join("cedar" if t == 1000 else chr(t - 4) for t in ids)


@pytest.fixture(scope="module")
def synthetic_success():
    receipt, proposal = V2.search(ByteTokenizer(), attempts_per_slot=10)
    assert proposal is not None
    return receipt, proposal


def test_full_48_slot_token_count_parity_is_proposal_only(synthetic_success):
    receipt, proposal = synthetic_success
    assert receipt["status"] == "token-parity-feasible-for-five-p3-corpora"
    assert receipt["sampledSlots"] == 48
    assert receipt["newNamesSelected"] == 48
    assert receipt["matchedRows"] == 384
    assert receipt["matchedValidationRows"] == 96
    assert receipt["entireHistoricalTrainingInventoryVerified"] is False
    assert receipt["trainingReady"] is False
    assert receipt["trainingAuthorized"] is False
    assert receipt["newInferenceAuthorized"] is False
    assert receipt["finalHoldoutOpened"] is False
    assert receipt["optimizerUpdates"] == 0
    assert proposal["authorization"] == {
        "modelLoaded": False, "optimizerUpdates": 0,
        "trainingAuthorized": False, "inferenceAuthorized": False,
        "stageCreated": False, "bundleCreated": False,
        "finalHoldoutOpened": False,
    }
    assert hashlib.sha256(V2.V1._canonical(proposal)).hexdigest() == (
        receipt["unapprovedV2ProposalSha256"]
    )


def test_only_names_change_and_every_original_token_signature_matches(synthetic_success):
    receipt, proposal = synthetic_success
    original = json.loads(V2.V1._checked(V2.V1.FROZEN, V2.V1.SOURCE_SHA))["rows"]
    tokenizer = ByteTokenizer()
    assert len(proposal["rows"]) == len(original) == 384
    count = Counter(r["split"] for r in proposal["rows"])
    assert count == {"train": 288, "validation": 96}
    for old, row in zip(original, proposal["rows"], strict=True):
        assert V2._signature(tokenizer, old) == V2._signature(tokenizer, row)
        assert old["language"] == row["language"]
        assert old["request"] == row["request"]
        assert old["targetStart"] == row["targetStart"]
        assert old["targetEnd"] == row["targetEnd"]
        assert len(old["inputFile"]) == len(row["inputFile"])
        assert len(old["expectedFile"]) == len(row["expectedFile"])
        assert len(old["prompt"]) == len(row["prompt"])
        if old["split"] == "validation":
            assert row == old
    groups = {}
    for row in proposal["rows"]:
        groups.setdefault(row["groupId"], []).append(row)
    assert len(groups) == 96
    assert all(len(xs) == 4 for xs in groups.values())
    assert all(len({r["namespace"] for r in xs}) == 1
               for xs in groups.values())
    assert len({row["namespace"] for row in proposal["rows"]
                if row["split"] == "train"}) == 48
    assert proposal["sampler"]["drawIndexCanonicalSha256"] == V2.V1.EXPECTED_DRAW_SHA
    assert len(proposal["sampler"]["drawIndexArray"]) == 1600


def test_tokenizer_counterexample_exhausts_only_a_bounded_search():
    receipt, proposal = V2.search(NonMatchingTokenizer(), attempts_per_slot=5)
    assert proposal is None
    assert receipt["status"] == "token-parity-infeasible-within-search-budget"
    assert receipt["sampledSlots"] == 1
    assert receipt["newNamesSelected"] == 0
    assert receipt["testedCandidates"] == 5
    assert receipt["trainingReady"] is False
    assert receipt["matchedRows"] is None
    assert receipt["optimizerUpdates"] == 0
    assert receipt["slotResults"]["cedar.0"]["matched"] is False


def test_name_stream_is_deterministic_and_never_reuses_previous_name(synthetic_success):
    _, proposal = synthetic_success
    pools = proposal["selectedTrainNamespaces"]
    assert list(pools) == list(V2.V1.OLD_NAMES)
    names = [name for pool in pools.values() for name in pool]
    assert len(names) == len(set(names)) == 48
    assert set(names).isdisjoint(V2.V1.OLD_NAMES)
    assert set(names).isdisjoint(V2.V1.VAL_NAMES)
    assert set(names).isdisjoint(V2.V1.MICRO_NAMES)
    a = list(next(V2._candidate_stream(ByteTokenizer(), "cedar", 0)) for _ in range(2))
    assert a[0] == a[1]
    assert len(a[0]) == len("cedar")


def test_corrupt_original_v1_proposal_blocks_search(monkeypatch):
    monkeypatch.setattr(V2, "V1_PROPOSAL_SHA", "f" * 64)
    with pytest.raises(ValueError, match="SHA mismatch"):
        V2.search(ByteTokenizer(), attempts_per_slot=1)


def test_reject_bad_tokenizer_or_bad_search_budget():
    t = ByteTokenizer()
    t.vocabulary_size = 16383
    with pytest.raises(ValueError, match="16384"):
        V2.search(t)
    for val in (0, -3, 20_001, True, 1.5):
        with pytest.raises(ValueError, match="bounded"):
            V2.search(ByteTokenizer(), attempts_per_slot=val)


def test_explicit_training_history_inventory_sha_and_collision(tmp_path):
    data = tmp_path / "p3-history.txt"
    data.write_bytes(b"extra nonholdout data taper\n")
    meta = tmp_path / "history.json"
    meta.write_text(json.dumps({
        "scope": "non-final-holdout-training-corpora",
        "files": [{
            "path": str(data.resolve()),
            "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        }],
    }), encoding="utf-8")
    extra, shas = V2._optional_history(meta)
    assert len(extra) == len(shas) == 1
    assert b"taper" in extra[0]
    assert len(shas) == 1
    meta.write_text(json.dumps({
        "scope": "non-final-holdout-training-corpora",
        "files": [{
            "path": str(data.resolve()),
            "sha256": "0" * 64,
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        V2._optional_history(meta)


def test_explicit_final_holdout_path_is_refused(tmp_path):
    directory = tmp_path / "final-holdout"
    directory.mkdir()
    data = directory / "data.jsonl"
    data.write_bytes(b"private")
    meta = tmp_path / "inventory.json"
    meta.write_text(json.dumps({
        "scope": "non-final-holdout-training-corpora",
        "files": [{
            "path": str(data.resolve()),
            "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="holdout"):
        V2._optional_history(meta)


def test_incomplete_inventory_never_marks_global_training_history_verified(synthetic_success):
    receipt, proposal = synthetic_success
    assert receipt["entireHistoricalTrainingInventoryVerified"] is False
    assert proposal["selection"]["completeHistoricalTrainingInventoryVerified"] is False
    assert proposal["status"] == "local-proposal-only-global-history-unverified-no-training"


def test_script_is_read_only_by_default_and_no_model_apis():
    path = HERE / "review_namespace_diversity_v2.py"
    source = path.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "read_checkpoint(" not in source
    assert "optimizer.step(" not in source
    assert "_generate_token_ids(" not in source
    assert "model.generate(" not in source
    assert 'parser.add_argument("--prepare-local"' in source
    assert 'with target.open("xb")' in source
    assert "trainingAuthorized" in source
    assert "finalHoldoutOpened" in source
