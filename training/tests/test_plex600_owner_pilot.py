"""Safety/identity tests for first real owner-authored Plex 600M dataset.

Tests may fit a tiny 600M-family BPE in temporary space but never construct the
600M model, open historical corpus/holdout, or execute optimizer updates.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from plex_training import plex600_owner_pilot as pilot
from plex_training.plex600_setup import check_source_binding, check_tokenizer_bundle
from plex_training.tokenizer import train_tokenizer


class Plex600OwnerPilotTests(unittest.TestCase):
    def test_committed_dataset_is_original_authored_train_only(self):
        outcome = pilot.verify()
        self.assertEqual(outcome["status"], "verified-owner-authored-tokenizer-ready")
        self.assertEqual(outcome["trainRecords"], 120)
        self.assertEqual(outcome["validationRecords"], 24)
        self.assertEqual(outcome["priorValidationExcluded"], 72)
        self.assertEqual(outcome["externalHistoricalRecords"], 0)
        self.assertFalse(outcome["trainingAuthorized"])
        self.assertFalse(outcome["finalHoldoutOpened"])

    def test_unchanged_training_source_approvals_are_only_p2(self):
        _, forbidden, approvals = pilot.source_records()
        self.assertEqual(len(forbidden), 72)
        self.assertEqual(len(approvals), 2)
        manifest = pilot.jsonfile(pilot.DATASET / "manifest.json")
        self.assertEqual(manifest["pipelineVersion"], pilot.PIPELINE)
        self.assertFalse(manifest["m4PreparationApproved"])
        self.assertFalse(manifest["trainingAuthorized"])
        self.assertTrue(manifest["sourceRightsApprovedForLocalPreparation"])
        self.assertFalse(manifest["externalHistoricalSourcesUsed"])

    def test_pilot_tampering_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "dataset"
            shutil.copytree(pilot.DATASET, copy)
            target = copy / "validation.jsonl"
            target.write_bytes(target.read_bytes().replace(b"Request:", b"Replaced:", 1))
            with self.assertRaisesRegex(ValueError, "Split hash"):
                pilot.verify(copy)

    def test_pilot_provenance_tampering_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "dataset"
            shutil.copytree(pilot.DATASET, copy)
            manifest_file = copy / "manifest.json"
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            manifest["externalHistoricalSourcesUsed"] = True
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance"):
                pilot.verify(copy)

    def test_fits_600m_tokenizer_from_real_pilot_train_only(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "tokenizer"
            fitted = train_tokenizer(pilot.DATASET, bundle, vocab_size=512,
                                     model_family="plex-600m-v1", storage_limit_bytes=1024**2)
            self.assertEqual(fitted["train"]["records"], 120)
            self.assertEqual(fitted["validation"]["records"], 24)
            checked = check_tokenizer_bundle(bundle)
            self.assertEqual(checked["verifiedSplits"]["train"]["records"], 120)
            self.assertEqual(checked["verifiedSplits"]["validation"]["records"], 24)
            hash_file = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            source = check_source_binding(bundle, pilot.DATASET,
                bundle_manifest_sha256=hash_file(bundle / "manifest.json"),
                source_manifest_sha256=hash_file(pilot.DATASET / "manifest.json"))
            self.assertTrue(source["sourceBindingVerified"])
            self.assertFalse(source["trainingAuthorized"])

    def test_old_nano_tokenizer_does_not_accept_600m_pilot(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "600M pilot"):
                train_tokenizer(pilot.DATASET, Path(temp) / "bad",
                                vocab_size=512, model_family="plex-nano")


if __name__ == "__main__":
    unittest.main()
