"""SB01/SB05 synthetic scoring and read-only tokenizer gates; no model execution."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "phase3/copy-fidelity-v1"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = load("sb05_test_scoring", "score_source_binding_microprobe.py")
T = load("sb01_test_tokenizer_review", "review_source_binding_microprobe.py")


def perfect_responses():
    return [{"recordId": row["id"], "actual": row["expectedFile"],
             "stoppedAtEos": True, "decodeSucceeded": True, "fullPromptVisible": True,
             "generatedTokenCount": len(row["expectedFile"])}
            for row in S.frozen_probe()["rows"]]


def score(responses):
    return S.score_responses(responses, probe_sha256=S.PROBE_SHA,
                             checkpoint_sha256=S.CHECKPOINT_SHA, tokenizer_sha256=S.TOKENIZER_SHA)


def test_perfect_synthetic_files_and_gates():
    result = score(perfect_responses())
    assert result["totals"]["exactCopies"] == 12
    assert result["totals"]["exactEdits"] == 36
    assert result["totals"]["unrelatedSourcePreserved"] == 48
    assert result["fullyCorrectContrastGroups"] == 12
    assert result["allProposedGatesPassed"] is True
    assert result["executionProvenanceIndependentlyVerified"] is False
    assert result["liveEvaluationAuthorized"] is False
    assert result["modelLoaded"] is False


def test_exact_threshold_and_each_independent_capability_gate():
    rows = S.frozen_probe()["rows"]
    replies = perfect_responses()
    # Fail two edits in one group per language: 30 edits, 10 per language, 9 groups.
    for lang in S.LANGUAGES:
        indices = [i for i, r in enumerate(rows) if r["language"] == lang and r["condition"] == "edit"]
        for i in indices[:2]:
            replies[i]["actual"] = ""
    assert score(replies)["allProposedGatesPassed"] is True
    replies[0]["actual"] = ""
    assert score(replies)["gates"]["copies12of12"] is False
    # Three wrong edits in one language: overall count/group gates pass, language fails.
    replies = perfect_responses()
    for i in (1, 2, 3):
        replies[i]["actual"] = ""
    gates = score(replies)["gates"]
    assert gates["editsAtLeast30of36"] and gates["completeGroupsAtLeast9of12"]
    assert not gates["perLanguageAtLeast10of12Edits"]
    # Four scattered edits: 32 edits and >=10 per language, but only 8 whole groups.
    replies = perfect_responses()
    for i in (1, 5, 17, 21):
        replies[i]["actual"] = ""
    gates = score(replies)["gates"]
    assert gates["editsAtLeast30of36"] and gates["perLanguageAtLeast10of12Edits"]
    assert not gates["completeGroupsAtLeast9of12"]


def test_wrong_value_is_distinct_from_unextractable_or_literal_mention():
    replies = perfect_responses()
    # A requested literal appearing in a different namespace is not value binding.
    replies[1]["actual"] = replies[1]["actual"].replace("ulvar", "meadow")
    result = score(replies)
    assert result["perRecord"][1]["requestedValueCorrect"] is None
    assert result["totals"]["requestedValueUnknown"] == 1
    # Correct anchors but wrong value: independently extractable and incorrect.
    replies = perfect_responses()
    replies[1]["actual"] = replies[1]["actual"].replace("Start", "Wrong")
    row = score(replies)["perRecord"][1]
    assert row["requestedValueExtractable"] is True
    assert row["requestedValueCorrect"] is False
    assert row["unrelatedBytesPreserved"] is True
    assert row["namespaceLexemesMatch"] is True
    assert row["exactCompleteFile"] is False


@pytest.mark.parametrize("field", ["stoppedAtEos", "decodeSucceeded", "fullPromptVisible"])
def test_completion_failures_cannot_pass_even_with_exact_text(field):
    replies = perfect_responses()
    replies[1][field] = False
    row = score(replies)["perRecord"][1]
    assert row["exactCompleteFile"] is False
    assert row["requestedValueCorrect"] is None
    replies[1][field] = 1
    with pytest.raises(ValueError, match="Malformed"):
        score(replies)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "reordered", "scope"])
def test_response_identity_and_scope_fail_closed(change):
    replies = perfect_responses()
    if change == "missing":
        replies.pop()
    elif change == "extra":
        replies.append(copy.deepcopy(replies[0]))
    elif change == "duplicate":
        replies[1] = copy.deepcopy(replies[0])
    elif change == "reordered":
        replies[0], replies[1] = replies[1], replies[0]
    else:
        replies[0]["recordId"] = "final-holdout-task"
    with pytest.raises(ValueError, match="48 frozen"):
        score(replies)


@pytest.mark.parametrize("field", ["probe_sha256", "checkpoint_sha256", "tokenizer_sha256"])
def test_bad_provenance_rejected(field):
    args = dict(probe_sha256=S.PROBE_SHA, checkpoint_sha256=S.CHECKPOINT_SHA,
                tokenizer_sha256=S.TOKENIZER_SHA)
    args[field] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        S.score_responses(perfect_responses(), **args)


def test_lexical_match_does_not_override_unrelated_code_or_structure_failure():
    replies = perfect_responses()
    replies[0]["actual"] = "ulvar-launch"  # same name, destroyed file structure
    row = score(replies)["perRecord"][0]
    assert row["namespaceLexemesMatch"] is True
    assert row["exactCompleteFile"] is False
    assert row["unrelatedBytesPreserved"] is False
    replies = perfect_responses()
    replies[0]["actual"] += " "  # no stripping/normalization
    assert score(replies)["perRecord"][0]["exactCompleteFile"] is False


@pytest.mark.parametrize("bad", [True, -1, 193, 1.0])
def test_token_count_types_and_bounds(bad):
    replies = perfect_responses()
    replies[0]["generatedTokenCount"] = bad
    with pytest.raises(ValueError, match="Malformed"):
        score(replies)


class ByteTokenizer:
    vocabulary_size = 16384

    def encode(self, text):
        return [c + 4 for c in text.encode("ascii")]

    def decode(self, ids):
        return bytes(c - 4 for c in ids).decode("ascii")


def test_tokenizer_budget_mask_roundtrip_without_claiming_byte_identity():
    result = T.review(ByteTokenizer())
    assert result["records"] == 48
    assert result["maximum"]["promptTokens"] + 193 <= 512
    assert result["maximum"]["answerTokensPlusEos"] <= 192
    assert result["tokenizerByteIdentityVerified"] is False


@pytest.mark.parametrize("failure", ["roundtrip", "budget", "control", "vocabulary"])
def test_bad_tokenizer_fails_closed(failure):
    tokenizer = ByteTokenizer()
    if failure == "roundtrip":
        tokenizer.decode = lambda ids: "wrong"
    elif failure == "budget":
        # Exercise budget gate after successful roundtrip, using ID-to-source lookup.
        cache = {}
        def encode(text):
            ids = [len(cache) + 4] * 513
            cache[ids[0]] = text
            return ids
        tokenizer.encode = encode
        tokenizer.decode = lambda ids: cache[ids[0]]
    elif failure == "control":
        tokenizer.encode = lambda text: [0] + ByteTokenizer.encode(tokenizer, text)
        tokenizer.decode = lambda ids: ByteTokenizer.decode(tokenizer, ids[1:])
    else:
        tokenizer.vocabulary_size = 16383
    with pytest.raises(ValueError):
        T.review(tokenizer)


def test_changed_manifest_and_tokenizer_bytes_are_refused(tmp_path, monkeypatch):
    changed = tmp_path / "probe.json"
    changed.write_bytes(S.PROBE.read_bytes() + b" ")
    monkeypatch.setattr(S, "PROBE", changed)
    with pytest.raises(ValueError, match="SHA changed"):
        S.frozen_probe()
    bad = tmp_path / "tokenizer.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer SHA changed"):
        T.review_file(bad)


def test_new_modules_have_no_model_execution_or_publication_entrypoint():
    for name in ("score_source_binding_microprobe.py", "review_source_binding_microprobe.py"):
        source = (HERE / name).read_text(encoding="utf-8")
        for forbidden in ("import torch", "read_checkpoint(", "optimizer.step(",
                          "_generate_token_ids(", ".write_text(", ".write_bytes("):
            assert forbidden not in source


def test_recorded_evidence_binds_reviewed_tools_and_failed_historical_result():
    evidence = HERE.parents[2] / "training/pretraining/p3-source-binding-technical-review-evidence.json"
    raw = evidence.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "064c52150951c2510f2f87f6406b57634a0e9369525dd0f603e5ffafceb2e43b"
    saved = json.loads(raw)
    for name, digest in saved["reviewSourceSha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
    assert saved["sb00"]["totalCases"] == 138
    assert saved["sb00"]["allOriginalGeneratedFileGatesPassed"] is False
    assert saved["sb00"]["inputReportSha256"] == "971a78971387200df9523c6ca177006f60f2834be0664c2b590e6ec72f06a197"
    assert saved["sb01Tokenizer"]["probeSha256"] == S.PROBE_SHA
    assert saved["sb01Tokenizer"]["tokenizerByteIdentityVerified"] is True
