"""P3-02 request binding, split integrity, identity, and preparation boundaries."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from plex_training.cli import build_parser
from plex_training.file_conditioned_contract import _canonical_bytes, build_candidate_rows, EXPECTED_CANDIDATE_SHA256
from plex_training.file_edit_replacements import (
    DEFAULT_CONTRACT, EXPECTED_BYTES, EXPECTED_SHA256, DEFAULT_BUNDLE,
    build_replacement_rows, validate_replacement_rows, render_prompt,
    generate_replacement_candidate, review_replacement_candidate, score_replacement,
    answer_only_target_mask, plan_replacement_supervision, preflight_replacement_supervision,
    EXPECTED_TOKENIZER_SHA256, TOKENIZER_BUNDLE_SHA256,
)


class ByteFixtureTokenizer:
    vocabulary_size = 16384

    def encode(self, text):
        return [value + 4 for value in text.encode("utf-8")]

    def decode(self, ids):
        return bytes(value - 4 for value in ids).decode("utf-8")


def tokenizer_fixture():
    return ByteFixtureTokenizer(), {
        "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
        "bundleManifestSha256": TOKENIZER_BUNDLE_SHA256,
    }


def candidate_fixture(root):
    candidate, review = root / "candidate.jsonl", root / "review.json"
    generate_replacement_candidate(candidate_path=candidate, review_path=review)
    return candidate, review


def test_balanced_groups_and_zero_exact_split_overlap():
    import hashlib
    rows = build_replacement_rows()
    report = validate_replacement_rows(rows)
    assert report["records"] == 216 and report["contrastGroups"] == 72
    assert report["splitCounts"] == {"train": 144, "validation": 36, "development": 36}
    assert report["perLanguage"] == {"html": 72, "css": 72, "javascript": 72}
    assert not any(count for pair in report["splitOverlap"].values() for count in pair.values())
    assert not any(report["p301ExactOverlap"].values())
    raw = _canonical_bytes(rows)
    assert len(raw) == EXPECTED_BYTES and hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
    assert hashlib.sha256(_canonical_bytes(build_candidate_rows())).hexdigest() == EXPECTED_CANDIDATE_SHA256


def test_same_file_contrast_requires_request_specific_replacement():
    first, second, third = build_replacement_rows()[:3]
    assert first["inputFile"] == second["inputFile"] == third["inputFile"]
    assert first["request"] == "For the button with id cart-save, change its text to Confirm."
    assert second["request"] == "For the button with id cart-save, change its text to Save."
    assert '<button id="cart-save">Confirm</button>' in first["expectedFile"]
    assert score_replacement(first, first["expectedFile"])["exactMatch"]
    wrong = score_replacement(first, second["expectedFile"])
    assert not wrong["exactMatch"] and wrong["unrelatedCodePreserved"]
    assert 'Keep this note.' in first["expectedFile"]
    assert "contrastGroupId" not in render_prompt(first)
    assert "candidateSplit" not in render_prompt(first)


def test_css_and_javascript_requests_match_the_authored_target_and_preserve_neighbors():
    rows = build_replacement_rows()
    css = next(r for r in rows if r["id"] == "p302-css-gap-00-1")
    assert css["request"] == "In .cart-gap, change gap to 16px."
    assert css["expectedFile"] == (
        "/* Keep this note for cart. */\n.cart-gap {\n  gap: 16px;\n  display: block;\n}"
        "\n.cart-aside { opacity: 0.8; }"
    )
    js = next(r for r in rows if r["id"] == "p302-javascript-default-size-00-1")
    assert js["request"] == "Change the default size argument in cartSize to 20."
    assert js["expectedFile"] == (
        "// Keep this note for cart.\nfunction cartSize(size = 20) {\n  return size;\n}"
        "\nconst cartEnabled = true;"
    )


@pytest.mark.parametrize("field,value", [
    ("expectedFile", "unrelated change"), ("request", "different request"),
    ("candidateSplit", "validation"), ("schemaVersion", True),
])
def test_changed_targets_requests_and_split_assignments_fail_closed(field, value):
    rows = build_replacement_rows()
    rows[0][field] = value
    with pytest.raises(ValueError, match="differs"):
        validate_replacement_rows(rows)


def test_duplicate_and_missing_rows_rejected():
    rows = build_replacement_rows()
    with pytest.raises(ValueError, match="216"):
        validate_replacement_rows(rows[:-1])
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_replacement_rows(rows)


def test_scorer_preservation_and_strict_schema():
    row = build_replacement_rows()[0]
    damaged = row["expectedFile"].replace("Keep this note.", "Removed.")
    assert score_replacement(row, damaged)["unnecessaryEdits"]
    assert not score_replacement(row, row["expectedFile"] + "\n")["exactMatch"]
    assert score_replacement(row, row["expectedFile"].replace("\n", "\r\n"))["exactMatch"]
    for version in (True, 1.0):
        with pytest.raises(ValueError, match="frozen"):
            score_replacement({**row, "schemaVersion": version}, row["expectedFile"])


def test_preparation_flags_reject_numeric_boolean_aliases_before_writes():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = json.loads(DEFAULT_CONTRACT.read_text())
        for field, value in (("modelTrainingAuthorized", True), ("dataPreparationAuthorized", 1),
                             ("researchOptimizerUpdates", False), ("optimizerCreationAuthorized", 0)):
            contract = root / "contract.json"
            contract.write_text(json.dumps({**data, field: value}))
            with pytest.raises(ValueError, match="boundary"):
                generate_replacement_candidate(candidate_path=root / "candidate", review_path=root / "review",
                                               contract_path=contract)
            assert not (root / "candidate").exists()
            assert not (root / "review").exists()


def test_generation_protects_contract_and_existing_input_aliases():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        contract = root / "contract.json"
        original = DEFAULT_CONTRACT.read_bytes()
        contract.write_bytes(original)
        with pytest.raises(ValueError, match="protected input"):
            generate_replacement_candidate(candidate_path=contract, review_path=root / "review",
                                           contract_path=contract)
        assert contract.read_bytes() == original


def test_review_requires_frozen_tokenizer_and_roundtrip_without_model_work():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        candidate, review = candidate_fixture(root)
        with patch("plex_training.file_edit_replacements._completion_tokenizer", return_value=tokenizer_fixture()), \
             patch("plex_training.completion.read_checkpoint", side_effect=AssertionError("model load forbidden")):
            report = review_replacement_candidate(candidate_path=candidate, review_path=review)
        assert report["status"] == "candidate-review-passed"
        assert report["tokenizerPreflight"]["allFit512Tokens"]
        assert not report["modelTrainingAuthorized"] and not report["finalHoldoutOpened"]
        tokenizer, identity = tokenizer_fixture()
        with patch("plex_training.file_edit_replacements._completion_tokenizer",
                   return_value=(tokenizer, {**identity, "tokenizerSha256": "wrong"})):
            with pytest.raises(ValueError, match="frozen"):
                review_replacement_candidate(candidate_path=candidate, review_path=review)


def test_review_rejects_identity_and_metadata_tampering_before_tokenizer():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        candidate, review = candidate_fixture(root)
        data = json.loads(review.read_text())
        review.write_text(json.dumps({**data, "modelTrainingAuthorized": 0}))
        with patch("plex_training.file_edit_replacements._completion_tokenizer") as load:
            with pytest.raises(ValueError, match="draft review"):
                review_replacement_candidate(candidate_path=candidate, review_path=review)
            load.assert_not_called()
        candidate.write_bytes(candidate.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="identity changed"):
            review_replacement_candidate(candidate_path=candidate, review_path=review)


def test_context_overflow_and_roundtrip_failure_cannot_pass_review():
    class OversizedTokenizer(ByteFixtureTokenizer):
        def encode(self, text):
            return [token for token in super().encode(text) for _ in range(4)]

        def decode(self, ids):
            return super().decode(ids[::4])

    class BrokenRoundtripTokenizer(ByteFixtureTokenizer):
        def decode(self, ids):
            return "wrong decoded text"

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        candidate, review = candidate_fixture(root)
        for tokenizer, message in ((OversizedTokenizer(), "budget"), (BrokenRoundtripTokenizer(), "roundtrip")):
            _, identity = tokenizer_fixture()
            with patch("plex_training.file_edit_replacements._completion_tokenizer", return_value=(tokenizer, identity)):
                with pytest.raises(ValueError, match=message):
                    review_replacement_candidate(candidate_path=candidate, review_path=review)


def test_report_alias_rejected_before_tokenizer_or_candidate_read():
    with TemporaryDirectory() as temporary:
        candidate = Path(temporary) / "candidate.jsonl"
        with patch("plex_training.file_edit_replacements._completion_tokenizer") as load:
            with pytest.raises(ValueError, match="protected input"):
                review_replacement_candidate(candidate_path=candidate, report_path=candidate)
            load.assert_not_called()


def test_cli_commands_keep_repository_rooted_defaults():
    parser = build_parser()
    assert parser.parse_args(["environment"]).command == "environment"
    for command in ("file-edit-replacements-generate", "file-edit-replacements-review"):
        args = parser.parse_args([command])
        assert args.candidate.is_absolute() and args.review.is_absolute() and args.contract.is_absolute()
        if command.endswith("review"):
            assert args.bundle_dir == DEFAULT_BUNDLE


def test_answer_only_mask_supervises_first_answer_token_and_eos():
    assert answer_only_target_mask(3, 2) == (0, 0, 1, 1, 1)
    assert answer_only_target_mask(1, 1) == (1, 1)
    for prompt, target in ((0, 1), (1, 0), (True, 1), (2, 1.0)):
        with pytest.raises(ValueError, match="nonempty"):
            answer_only_target_mask(prompt, target)


def test_p302_supervision_plan_excludes_development_and_replays_exact_targets():
    import random
    rows = build_replacement_rows()
    tokenizer = ByteFixtureTokenizer()
    first = plan_replacement_supervision(rows, tokenizer)
    assert plan_replacement_supervision(rows, tokenizer) == first
    assert first["records"] == {"train": 144, "validation": 36}
    assert first["sampler"]["examples"] == 1600
    assert first["sampler"]["recordsSelected"] == 144
    assert first["sampler"]["optimizerUpdatesPerformed"] == 0
    assert first["promptLossWeights"] == 0
    assert first["eosTargetWeight"] == 1
    assert first["paddingTargetWeight"] == 0
    assert set(first["answerAndEosTargetsPerLanguage"]) == {"train", "validation"}
    assert sum(first["answerAndEosTargetsPerLanguage"]["train"].values()) == first["answerAndEosTargetPositions"]["train"]
    assert sum(first["answerAndEosTargetsPerLanguage"]["validation"].values()) == first["answerAndEosTargetPositions"]["validation"]
    assert len(first["packedTokenSha256"]["train"]) == 64
    assert len(first["packedTokenSha256"]["validation"]) == 64
    train = [row for row in rows if row["candidateSplit"] == "train"]
    targets = [len(tokenizer.encode(row["expectedFile"])) + 1 for row in train]
    rng = random.Random(1337)
    assert first["sampler"]["expectedAnswerAndEosTargetPositions"] == sum(
        targets[rng.randrange(len(train))] for _ in range(1600)
    )
    changed_dev = [dict(row) for row in rows]
    dev_index = next(i for i, row in enumerate(changed_dev) if row["candidateSplit"] == "development")
    changed_dev[dev_index]["expectedFile"] = "tampered"
    with pytest.raises(ValueError, match="differs"):
        plan_replacement_supervision(changed_dev, tokenizer)


def test_supervision_preflight_cannot_claim_frozen_review_from_fake_token_counts():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        candidate, review = candidate_fixture(root)
        with patch("plex_training.file_edit_replacements._completion_tokenizer",
                   return_value=tokenizer_fixture()), patch(
                   "plex_training.completion.read_checkpoint",
                   side_effect=AssertionError("checkpoint load forbidden")) as load:
            with pytest.raises(ValueError, match="frozen tokenizer evidence"):
                preflight_replacement_supervision(candidate_path=candidate, review_path=review)
            load.assert_not_called()
        assert sorted(path.name for path in root.iterdir()) == ["candidate.jsonl", "review.json"]


def test_supervision_preflight_cli_remains_read_only_and_rooted():
    args = build_parser().parse_args(["file-edit-replacements-supervision-preflight"])
    assert args.candidate.is_absolute()
    assert args.review.is_absolute()
    assert args.contract.is_absolute()
    assert args.bundle_dir == DEFAULT_BUNDLE
    assert not hasattr(args, "report")
