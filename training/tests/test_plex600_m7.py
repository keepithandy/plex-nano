"""M7 tests: only plans, source checks, simulated supervision and refusals.

These tests never instantiate the 600M model or execute an optimizer step.
"""
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from plex_training import plex600_m7 as m7


class Plex600M7Tests(unittest.TestCase):
    def test_plan_is_exactly_four_and_non_authorizing(self):
        p = m7.plan()
        self.assertEqual(p["status"], "plan")
        self.assertEqual(p["optimizerUpdatesPlanned"], 4)
        self.assertEqual(p["optimizerUpdatesCompleted"], 0)
        self.assertFalse(p["trainingAuthorized"])
        self.assertFalse(p["externalSixSourcesUsed"])
        self.assertFalse(p["historicalValidationAccessed"])
        self.assertFalse(p["finalHoldoutAccessed"])
        self.assertEqual(p["sequenceLengthMax"], 128)
        self.assertEqual(p["trainRecords"], 120)
        self.assertEqual(p["validationRecords"], 24)
        self.assertEqual(p["sourceManifestSha256"], m7.SOURCE_SHA)

    def test_default_cli_never_reads_model_or_bundle(self):
        with patch.object(m7, "preflight", side_effect=AssertionError("source touched")), \
             patch.object(m7, "execute", side_effect=AssertionError("training touched")):
            with redirect_stdout(StringIO()) as out:
                self.assertEqual(m7.main([]), 0)
        self.assertEqual(json.loads(out.getvalue())["optimizerUpdatesCompleted"], 0)

    def test_explicit_execution_flag_is_required_before_source_access(self):
        with patch.object(m7, "preflight", side_effect=AssertionError("source read")):
            self.assertEqual(m7.execute(Path("unused"), Path("unused"), Path("unused"),
                                        bundle_sha="a" * 64)["status"], "refused")

    def test_unknown_or_unapproved_owner_packet_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "approval.json"
            decision = {
                "kind": m7.APPROVAL_KIND,
                "approvalStatus": "approved",
                "approvedBy": "repository-owner",
                "trainingScope": "local-four-update-m7-owner-authored-pilot-only",
                "approvedUpdates": 4, "sourceManifestSha256": m7.SOURCE_SHA,
                "bundleManifestSha256": "b" * 64, "finalHoldoutAllowed": False,
            }
            with self.assertRaisesRegex(ValueError, "approval"):
                m7.checked_approval(path, bundle_sha="b" * 64)
            path.write_text(json.dumps(decision), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "approval"):
                m7.checked_approval(path, bundle_sha="a" * 64)
            for field, wrong in (("approvedUpdates", 5), ("approvedUpdates", True),
                                 ("trainingScope", "unlimited"),
                                 ("finalHoldoutAllowed", True),
                                 ("approvalStatus", "pending")):
                with self.subTest(field=field, wrong=wrong):
                    altered = {**decision, field: wrong}
                    path.write_text(json.dumps(altered))
                    with self.assertRaisesRegex(ValueError, "authorization"):
                        m7.checked_approval(path, bundle_sha="b" * 64)
            path.write_text(json.dumps(decision))
            permitted = m7.checked_approval(path, bundle_sha="b" * 64)
            self.assertTrue(permitted["trainingAuthorized"])
            self.assertEqual(permitted["approvedUpdates"], 4)

    def test_bundle_preflight_exact_binding_without_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            (bundle / "manifest.json").write_bytes(b'{"file":"bundle"}\n')
            sha = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
            fake_report = {
                "actualVocabularySize": 1191,
                "sourceDatasetManifestSha256": m7.SOURCE_SHA,
                "tokenizerSha256": m7.TOKENIZER_SHA,
                "verifiedSplits": {
                    "train": {"sha256": m7.TRAIN_SHA, "records": 120, "tokens": 6970},
                    "validation": {"sha256": m7.VALID_SHA, "records": 24, "tokens": 1845},
                },
            }
            with patch.object(m7.setup, "check_source_binding", return_value={
                    "sourceBindingVerified": True, "trainingAuthorized": False}), \
                 patch.object(m7.setup, "check_tokenizer_bundle", return_value=fake_report), \
                 patch.object(m7.hw, "_check_device", side_effect=AssertionError("GPU called")):
                result = m7.preflight(bundle, expected_bundle_sha=sha)
                self.assertEqual(result["status"], "source-bound-plan-only")
                self.assertFalse(result["trainingAuthorized"])
                self.assertEqual(result["optimizerUpdatesCompleted"], 0)
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    m7.preflight(bundle, expected_bundle_sha="b" * 64)

    def test_preflight_rejects_wrong_tokenizer_and_train_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            (bundle / "manifest.json").write_bytes(b"{}")
            sha = hashlib.sha256(b"{}").hexdigest()
            baseline = {"actualVocabularySize": 1191,
                        "sourceDatasetManifestSha256": m7.SOURCE_SHA,
                        "tokenizerSha256": m7.TOKENIZER_SHA,
                        "verifiedSplits": {
                            "train": {"sha256": m7.TRAIN_SHA, "records": 120, "tokens": 6970},
                            "validation": {"sha256": m7.VALID_SHA, "records": 24, "tokens": 1845}}}
            for changed in ({"tokenizerSha256": "f" * 64},
                            {"verifiedSplits": {**baseline["verifiedSplits"],
                                "train": {"sha256": "0" * 64, "records": 120, "tokens": 6970}}}):
                with self.subTest(changed=changed), \
                     patch.object(m7.setup, "check_source_binding", return_value={
                        "sourceBindingVerified": True, "trainingAuthorized": False}), \
                     patch.object(m7.setup, "check_tokenizer_bundle",
                                  return_value={**baseline, **changed}):
                    with self.assertRaisesRegex(ValueError, "exact verified tokenizer|fixed pilot"):
                        m7.preflight(bundle, expected_bundle_sha=sha)

    def test_record_windows_do_not_cross_neighbors_or_touch_evaluation(self):
        records = [(str(i), [4, 5, 6, 7, 8, 9, 3]) for i in range(120)]
        first = m7.choose_windows(records)
        second = m7.choose_windows(records)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len({rid for rid, _ in first}), 4)
        self.assertTrue(all(window == [4, 5, 6, 7, 8, 9, 3] for _, window in first))
        with self.assertRaisesRegex(ValueError, "Not enough"):
            m7.choose_windows(records[:3])

    def test_output_destination_is_create_only_and_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(m7, "ARTIFACTS", Path(temp)):
                root = Path(temp)
                (root / "tokenizer").mkdir()
                with patch.object(m7, "MIN_FREE_DISK", 0):
                    result = m7.output_path(root / "first-run", bundle=root / "tokenizer",
                                            dataset=m7.source.DATASET)
                    self.assertEqual(result, (root / "first-run").resolve())
                    result.mkdir()
                    with self.assertRaisesRegex(ValueError, "new directory"):
                        m7.output_path(result, bundle=root / "tokenizer",
                                       dataset=m7.source.DATASET)
                    with self.assertRaisesRegex(ValueError, "new directory"):
                        m7.output_path(root / "tokenizer" / "overwrite",
                                       bundle=root / "tokenizer", dataset=m7.source.DATASET)

    def test_missing_owner_approval_refuses_before_supervisor(self):
        with patch.object(m7, "preflight", return_value={
                "status": "source-bound-plan-only"}), \
             patch.object(m7, "checked_approval", side_effect=m7.Refusal("not approved")), \
             patch.object(m7.hw, "_supervise", side_effect=AssertionError("worker launched")):
            result = m7.execute(Path("."), Path("missing"), Path("."), bundle_sha="a" * 64,
                                execute_flag=True)
            self.assertEqual(result["status"], "refused")
            self.assertIn("not approved", result["error"])

    def test_unauthorized_worker_entry_cannot_launch_training(self):
        with patch.dict(os.environ, {}, clear=False), \
             patch.object(m7.sys, "stdin", StringIO("{}")), \
             patch.object(m7, "run_worker", side_effect=AssertionError("model instantiated")), \
             redirect_stdout(StringIO()) as out:
            os.environ.pop(m7.CONTROL, None)
            m7.worker()
        result = json.loads(out.getvalue())
        self.assertEqual(result["status"], "refused")
        self.assertEqual(result["optimizerUpdatesCompleted"], 0)

    def test_corrupt_or_tampered_saved_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "weights.pt"
            file.write_bytes(b"sample bytes (not a trained model)")
            digest = m7.hash_file(file)
            file.write_bytes(b"tampered sample bytes")
            self.assertNotEqual(digest, m7.hash_file(file))
            with self.assertRaisesRegex(ValueError, "maximum"):
                m7.hash_file(file, maximum=4)


if __name__ == "__main__":
    unittest.main()
