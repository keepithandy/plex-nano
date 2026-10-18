"""P3-SB03 namespace-diversity proposal: deterministic model-free gates only."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "phase3" / "copy-fidelity-v1"


def load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = load("prepare_namespace_diversity_candidate.py", "sb03_frozen_candidate_test")
AUDIT = load("audit_namespace_diversity_token_parity.py", "sb03_token_parity_test")


class SyntheticByteTokenizer:
    vocabulary_size = 16384

    def encode(self, text):
        return [ord(ch) + 4 for ch in text]

    def decode(self, ids):
        return "".join(chr(i - 4) for i in ids)


class CollapsedOriginalNameTokenizer(SyntheticByteTokenizer):
    SPECIAL = 1000

    def encode(self, text):
        result = []
        i = 0
        while i < len(text):
            if text.startswith("cedar", i):
                result.append(self.SPECIAL)
                i += 5
            else:
                result.append(ord(text[i]) + 4)
                i += 1
        return result

    def decode(self, ids):
        return "".join("cedar" if i == self.SPECIAL else chr(i - 4) for i in ids)


def test_frozen_proposal_bytes_sha_and_refusal_to_train():
    raw = GEN.OUTPUT.read_bytes()
    assert raw == GEN.serialized()
    assert hashlib.sha256(raw).hexdigest() == AUDIT.EXPECTED_PROPOSAL_SHA
    assert AUDIT.EXPECTED_PROPOSAL_SHA == (
        "4bbdc55782d9447271323378367831656dc1f05cb5bb174138738a6a7e5e7dea"
    )
    m = AUDIT.frozen_proposal()
    assert m["status"] == "frozen-proposal-not-token-matched-not-authorized"
    assert m["fixedStartingCheckpointSha256"] == GEN.ENDPOINT_SHA
    assert m["tokenizerSha256"] == GEN.TOKENIZER_SHA
    assert m["comparison"]["comparatorRequiresSeparateOwnerAuthorization"] is True
    assert m["comparison"]["interventionRequiresSeparateOwnerAuthorization"] is True
    assert m["authorization"] == {
        "candidatePreparationApproved": True,
        "tokenizerParityVerified": False,
        "entireHistoricalTrainingInventoryVerified": False,
        "modelLoaded": False, "optimizerUpdates": 0,
        "stageCreated": False, "bundleCreated": False,
        "trainingAuthorized": False, "newGenerationAuthorized": False,
        "finalHoldoutOpened": False,
    }


def test_only_train_identifiers_differ_and_validation_is_unchanged():
    old = json.loads(GEN._checked(GEN.FROZEN, GEN.SOURCE_SHA))["rows"]
    proposal = GEN.make()["rows"]
    assert len(old) == len(proposal) == 384
    assert [r for r in old if r["split"] == "validation"] == [
        r for r in proposal if r["split"] == "validation"
    ]
    seen = set()
    groups = defaultdict(list)
    for source, result in zip(old, proposal):
        assert source["split"] == result["split"]
        assert source["language"] == result["language"]
        assert source["condition"] == result["condition"]
        assert source["context"] == result["context"]
        assert source["shape"] == result["shape"]
        assert source["request"] == result["request"]
        assert source["targetStart"] == result["targetStart"]
        assert source["targetEnd"] == result["targetEnd"]
        for name in ("inputFile", "expectedFile", "prompt"):
            assert len(source[name]) == len(result[name])
        if source["split"] == "train":
            seen.add(result["namespace"])
            assert result["namespace"] not in GEN.OLD_NAMES
            assert result["namespace"] not in GEN.VAL_NAMES
            assert result["namespace"] not in GEN.MICRO_NAMES
        else:
            assert result == source
        groups[result["groupId"]].append(result)
    assert len(seen) == 48
    assert len(groups) == 96
    assert all(len(rows) == 4 for rows in groups.values())
    assert sum(r["condition"] == "copy" for r in proposal) == 96
    assert sum(r["condition"] == "edit" for r in proposal) == 288


def test_sha256_sampler_covers_every_record_with_5_or_6_draws():
    m = GEN.make()
    sampler = m["sampler"]
    draws = sampler["drawIndexArray"]
    assert len(draws) == 1600
    assert len(set(draws)) == 288
    counts = Counter(draws)
    assert min(counts.values()) == 5
    assert max(counts.values()) == 6
    assert GEN._draws() == draws
    assert hashlib.sha256(GEN._canonical(draws)).hexdigest() == (
        "9ddf29a3a2972cee0c8c2f79db97d7ef3f823065a2db6ee0840e89bb01760c13"
    )
    assert sampler["drawIndexCanonicalSha256"] == GEN.EXPECTED_DRAW_SHA


def test_historical_source_corruption_blocks_preparation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    prior = next(iter(GEN.HISTORICAL))
    altered = tmp_path / prior.name
    altered.write_bytes(prior.read_bytes() + b" ")
    monkeypatch.setattr(GEN, "HISTORICAL", {altered: GEN.HISTORICAL[prior]})
    with pytest.raises(ValueError, match="SHA mismatch"):
        GEN.make()


def test_byte_tokenizer_parity_is_possible_but_never_authorizes_training():
    report = AUDIT.audit(SyntheticByteTokenizer())
    assert report["tokenParityExact"] is True
    assert report["tokenParityMismatchCount"] == 0
    assert report["drawCoverage"] == 288
    assert report["readyForTraining"] is False
    assert report["globalHistoricalTrainingInventoryVerified"] is False
    assert report["trainingAuthorized"] is False
    assert report["optimizerUpdates"] == 0
    assert report["finalHoldoutOpened"] is False


def test_tokenizer_parity_failure_is_reported_not_silently_promoted():
    result = AUDIT.audit(CollapsedOriginalNameTokenizer())
    assert result["tokenParityExact"] is False
    assert result["tokenParityMismatchCount"] > 0
    assert result["firstMismatches"]
    assert result["drawCoverage"] == 288
    assert result["readyForTraining"] is False


def test_wrong_proposal_sha_or_permissions_refused(monkeypatch):
    monkeypatch.setattr(AUDIT, "EXPECTED_PROPOSAL_SHA", "0" * 64)
    with pytest.raises(ValueError, match="SHA mismatch"):
        AUDIT.frozen_proposal()


def test_no_optimizer_checkpoint_or_inference_apis_present():
    for filename in (
        "prepare_namespace_diversity_candidate.py",
        "audit_namespace_diversity_token_parity.py",
    ):
        content = (HERE / filename).read_text(encoding="utf-8")
        assert "import torch" not in content
        assert "read_checkpoint(" not in content
        assert "optimizer.step(" not in content
        assert "_generate_token_ids(" not in content
        assert "model.generate(" not in content
        assert "final_holdout" not in content
