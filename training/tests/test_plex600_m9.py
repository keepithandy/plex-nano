"""M9 600M eight-update pilot gates: CPU-only, never executes optimizer."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from plex_training import plex600_m9 as m9


class CompactTokenizer:
    """Test-only deterministic bounded tokenizer, NOT original M7 BPE."""
    vocabulary_size = 1191

    def encode(self, text):
        raw = text.encode("utf-8")
        return [4 + sum(raw[i:i + 4]) % 250
                for i in range(0, len(raw), 4)]


class M9BoundedPilotTests(unittest.TestCase):
    def test_default_no_training_checkpoint_source_or_cuda(self):
        with patch.object(m9.r3, "rows", side_effect=AssertionError("data accessed")), \
             patch.object(m9.m8, "prerequisites", side_effect=AssertionError("checkpoint read")), \
             patch.object(m9, "train_worker", side_effect=AssertionError("GPU training")), \
             redirect_stdout(StringIO()) as out:
            self.assertEqual(m9.main([]), 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["protocol"], m9.PROTOCOL)
        self.assertFalse(data["trainingAuthorized"])
        self.assertEqual(data["optimizerUpdatesCompleted"], 0)
        self.assertEqual(data["optimizerUpdatesMaximum"], 8)
        self.assertFalse(data["modelLoaded"])
        self.assertFalse(data["finalHoldoutUsed"])
        self.assertFalse(data["ownerApprovalBundled"])

    def test_source_grouped_no_truncation_three_languages(self):
        sample = m9.selected_records(CompactTokenizer())
        self.assertEqual(len(sample["trainRecordIds"]), 8)
        self.assertEqual(len(sample["validationRecordIds"]), 3)
        self.assertLessEqual(sample["trainMaxPositions"], 224)
        self.assertLessEqual(sample["validationMaxPositions"], 224)
        self.assertGreaterEqual(sample["trainLanguages"].count("html"), 2)
        self.assertGreaterEqual(sample["trainLanguages"].count("css"), 2)
        self.assertGreaterEqual(sample["trainLanguages"].count("javascript"), 2)
        self.assertFalse(sample["trainingAuthorized"])
        self.assertEqual(sample["optimizerUpdatesCompleted"], 0)
        train = sample["data"]["train"]
        validation = sample["data"]["validation"]
        self.assertFalse(
            {row["family"] for row, _ in train} &
            {row["family"] for row, _ in validation})
        for _, packed in train + validation:
            self.assertEqual(len(packed["inputIds"]), len(packed["targetIds"]))
            self.assertEqual(
                packed["targetIds"].count(-100),
                packed["promptTokens"] - 1)
            self.assertEqual(packed["targetIds"][-1], 3)

    def test_candidate_selection_deterministic_and_review_list_pinned(self):
        first = m9.selected_records(CompactTokenizer())
        second = m9.selected_records(CompactTokenizer())
        self.assertEqual(
            m9.public_selection(first), m9.public_selection(second))
        self.assertNotIn("data", m9.public_selection(first))
        proposed = m9.approval_draft(first)
        self.assertEqual(proposed["maxOptimizerUpdates"], 8)
        self.assertEqual(proposed["trainRecordIds"], first["trainRecordIds"])
        self.assertEqual(proposed["r3CandidateSha256"], first["r3CandidateSha256"])
        self.assertFalse(proposed["approvedForTraining"])
        self.assertFalse(proposed["sourceReviewedByOwner"])
        self.assertFalse(proposed["m8DevelopmentUsedForTraining"])
        self.assertFalse(proposed["finalHoldoutAllowed"])
        self.assertIsNone(proposed["approvedBy"])

    def test_approval_needs_exact_owner_review_checkpoint_and_scope(self):
        selected = m9.selected_records(CompactTokenizer())
        original = "f" * 64
        template = m9.approval_draft(selected, checkpoint_sha=original)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "approval.json"
            path.write_text(json.dumps(template))
            with self.assertRaisesRegex(ValueError, "source-scoped"):
                m9.check_approval(path, selected, original_checkpoint_sha=original)
            approved = {**template, "approvedBy": "keepithandy",
                        "approvedAt": "2026-11-16",
                        "approvedForTraining": True,
                        "sourceReviewedByOwner": True}
            path.write_text(json.dumps(approved))
            proof = m9.check_approval(
                path, selected, original_checkpoint_sha=original)
            self.assertTrue(proof["approved"])
            self.assertEqual(len(proof["approvalSha256"]), 64)
            variants = [
                {**approved, "approvedBy": "not-owner"},
                {**approved, "trainRecordIds": approved["trainRecordIds"][:-1]},
                {**approved, "r3CandidateSha256": "0" * 64},
                {**approved, "m7CheckpointSha256": "0" * 64},
                {**approved, "maxOptimizerUpdates": 1000},
                {**approved, "finalHoldoutAllowed": True},
                {**approved, "approvedForTraining": False},
                {**approved, "sourceReviewedByOwner": False},
                {**approved, "scope": "other-model"},
            ]
            for i, value in enumerate(variants):
                with self.subTest(variant=i):
                    path.write_text(json.dumps(value))
                    with self.assertRaisesRegex(ValueError, "source-scoped"):
                        m9.check_approval(
                            path, selected, original_checkpoint_sha=original)

    def test_source_identity_pins_model_and_m9_runner_before_cuda(self):
        with patch.object(m9.hardware, "source_identity", return_value={
                "gitCommit": "a" * 40,
                "modelSourceClean": True,
                "modelSha256": m9.hardware.MODEL_SHA256,
                "runnerSha256": "b" * 64}), \
             patch.object(m9.m7, "hash_file", return_value="c" * 64):
            identity = m9.code_identity()
        self.assertEqual(identity["modelSha256"], m9.hardware.MODEL_SHA256)
        self.assertEqual(identity["m9RunnerSha256"], "c" * 64)
        with patch.object(m9.hardware, "source_identity",
                          side_effect=ValueError("Modified model source")):
            with self.assertRaisesRegex(ValueError, "Modified model"):
                m9.code_identity()

    def test_impossible_date_is_not_accepted_as_owner_approval(self):
        sample = m9.selected_records(CompactTokenizer())
        document = m9.approval_draft(sample, checkpoint_sha="f" * 64)
        document.update(approvedBy="keepithandy",
                        approvedAt="2026-99-99",
                        approvedForTraining=True,
                        sourceReviewedByOwner=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "approval.json"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "calendar date"):
                m9.check_approval(
                    path, sample, original_checkpoint_sha="f" * 64)

    def test_missing_approval_prevents_gpu_check_even_with_fake_m7_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            from plex_training.tokenizer import PlexTokenizer
            fake_proof = {
                "m7Receipt": {"localCheckpointSha256": "f" * 64},
                "tokenizerSha256": m9.m7.TOKENIZER_SHA,
            }
            with patch.object(m9.m8, "prerequisites", return_value=fake_proof), \
                 patch.object(PlexTokenizer, "load", return_value=CompactTokenizer()), \
                 patch.object(m9.hardware, "_check_device",
                              side_effect=AssertionError("CUDA accessed")):
                with self.assertRaises(ValueError):
                    m9.train_worker(
                        Path(folder), Path("unused"), Path("unused"),
                        Path(folder) / "missing-approval.json",
                        stage=Path(folder), bundle_sha=m9.r3.research.BUNDLE_SHA)

    def test_cli_refuses_run_without_all_explicit_inputs(self):
        with patch.object(m9, "execute", side_effect=AssertionError("GPU invoked")), \
             redirect_stderr(StringIO()) as err:
            self.assertEqual(m9.main(["--execute-approved-run"]), 2)
        self.assertIn("requires explicit", err.getvalue())

    def test_approval_template_always_unapproved_and_no_gpu(self):
        with patch.object(m9.r3, "with_frozen_tokenizer",
                          side_effect=lambda fn: fn(CompactTokenizer())), \
             patch.object(m9, "execute", side_effect=AssertionError("GPU invoked")), \
             redirect_stdout(StringIO()) as out:
            self.assertEqual(m9.main(["--approval-template"]), 0)
        report = json.loads(out.getvalue())
        self.assertTrue(report["notAnApproval"])
        self.assertTrue(report["ownerReviewRequired"])
        self.assertFalse(report["approvalDraft"]["approvedForTraining"])
        self.assertEqual(report["trainRecordIds"],
                         report["approvalDraft"]["trainRecordIds"])


if __name__ == "__main__":
    unittest.main()
