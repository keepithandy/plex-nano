"""R3 source-conditioned candidate packing tests; no model training."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from plex_training import plex600_r3_curriculum as r3


class ByteTokenizer:
    # Test-only byte codec; integration CI reconstructs the real 1,191 BPE.
    vocabulary_size = 260

    def encode(self, value):
        return list(value.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8")


class R3FileEditingCurriculumTests(unittest.TestCase):
    def test_all_new_rows_are_valid_source_conditioned_and_unapproved(self):
        examples = r3.rows()
        report = r3.audit()
        self.assertEqual(len(examples), 111)
        self.assertEqual(report["candidateRows"], 111)
        self.assertEqual(report["distinctEditFamilies"], 37)
        self.assertEqual(report["uniqueEditOperations"], 37)
        self.assertEqual(report["languages"],
                         {"html": 36, "css": 39, "javascript": 36})
        self.assertEqual(report["trainCandidateRows"], 84)
        self.assertEqual(report["validationCandidateRows"], 27)
        self.assertEqual(report["exactM8EditOverlap"], 0)
        self.assertFalse(report["trainingAuthorized"])
        self.assertFalse(report["sourceApprovedForTraining"])
        self.assertFalse(report["finalHoldoutUsed"])
        self.assertTrue(report["familySplitDisjoint"])
        for example in examples:
            self.assertEqual(
                example["inputFile"].replace(example["oldText"],
                                             example["newText"], 1),
                example["expectedFile"])
            self.assertFalse(example["trainingAuthorized"])

    def test_family_isolation_with_three_variants_kept_together(self):
        examples = r3.rows()
        split = r3.family_splits(examples)
        self.assertEqual(len(split["train"]), 28)
        self.assertEqual(len(split["validation"]), 9)
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        for row in examples:
            self.assertIn(row["family"], split["train"] + split["validation"])

    def test_masked_causal_targets_only_supervise_complete_file_and_eos(self):
        masked = r3.masked_record(
            [11, 12, 13], [21, 22, 3], fitted_vocab_size=260)
        self.assertEqual(masked["inputIds"], [11, 12, 13, 21, 22])
        self.assertEqual(masked["targetIds"], [-100, -100, 21, 22, 3])
        self.assertEqual(masked["supervisedAnswerPositions"], 3)
        self.assertEqual(masked["lossMaskPromptPositions"], 2)
        self.assertEqual(len(masked["inputIds"]), len(masked["targetIds"]))
        for bad_prompt, bad_answer in (
            ([], [4, 3]), ([4], []), ([4], [4]), ([4], [0, 3]),
            ([4] * 1024, [4, 3]), ([4], [260, 3])):
            with self.subTest(prompt=bad_prompt[:4], answer=bad_answer), \
                 self.assertRaisesRegex(ValueError, "context|malformed"):
                r3.masked_record(
                    bad_prompt, bad_answer, fitted_vocab_size=260)

    def test_cpu_tokenized_all_111_examples_have_complete_target_only_masks(self):
        result = r3.audit(ByteTokenizer())
        self.assertTrue(result["packedPreviewNoModel"])
        self.assertGreater(result["supervisedAnswerPositions"], 1000)
        self.assertGreater(result["maskedPromptPositions"], 1000)
        self.assertLessEqual(result["maxInputPositions"], 1024)
        self.assertFalse(result["trainingAuthorized"])
        self.assertFalse(result["modelLoaded"])
        self.assertEqual(result["optimizerUpdates"], 0)

    def test_create_only_packed_candidate_bundle_cannot_be_authorization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "training/artifacts").mkdir(parents=True)
            destination = root / "training/artifacts/plex600-r3-test"
            with patch.object(r3, "ROOT", root):
                manifest = r3.materialize(destination, ByteTokenizer())
                self.assertEqual(manifest["trainCandidateRows"], 84)
                self.assertEqual(manifest["validationCandidateRows"], 27)
                self.assertFalse(manifest["trainingAuthorized"])
                self.assertIsNone(manifest["authorizingApprovalRecord"])
                self.assertEqual(manifest["tokenizerSha256"], r3.m7.TOKENIZER_SHA)
                train = [json.loads(x) for x in
                         (destination / "train.jsonl").read_text().splitlines()]
                valid = [json.loads(x) for x in
                         (destination / "validation.jsonl").read_text().splitlines()]
                self.assertEqual(len(train), 84)
                self.assertEqual(len(valid), 27)
                self.assertFalse(
                    {x["family"] for x in train} &
                    {x["family"] for x in valid})
                self.assertEqual(
                    train[0]["targetIds"].count(-100),
                    train[0]["promptTokens"] - 1)
                self.assertEqual(train[0]["targetIds"][-1], 3)
                with self.assertRaisesRegex(ValueError, "must be new"):
                    r3.materialize(destination, ByteTokenizer())

    def test_mutation_to_training_approval_refused(self):
        examples = r3.rows()
        examples[0]["trainingAuthorized"] = True
        with tempfile.TemporaryDirectory() as temp:
            malicious = Path(temp) / "candidate.jsonl"
            malicious.write_text(
                "".join(json.dumps(item) + "\n" for item in examples))
            with patch.object(r3, "CANDIDATE", malicious):
                with self.assertRaisesRegex(ValueError, "provenance"):
                    r3.rows()

    def test_default_cli_never_fits_or_trains(self):
        with patch.object(r3, "with_frozen_tokenizer",
                          side_effect=AssertionError("tokenizer fitted")), \
             redirect_stdout(StringIO()) as out:
            self.assertEqual(r3.main([]), 0)
        result = json.loads(out.getvalue())
        self.assertFalse(result["trainingAuthorized"])
        self.assertFalse(result["modelLoaded"])
        self.assertEqual(result["optimizerUpdates"], 0)


if __name__ == "__main__":
    unittest.main()
