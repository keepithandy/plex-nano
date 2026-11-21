"""CI checks read-only historical M12-A inventory; no model or protected holdout."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("plex_m12_history_inventory.py")
spec = importlib.util.spec_from_file_location("plex_m12_history_inventory", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HistoricalInventoryTests(unittest.TestCase):
    def test_original_owner_prep_data_is_verified_without_scope_escalation(self):
        report = module.inventory()
        self.assertEqual(report["javascriptRecordsTotal"], 44)
        self.assertEqual(report["priorCorpus"]["train"]["byLanguage"]["javascript"], 36)
        self.assertEqual(report["priorCorpus"]["validation"]["byLanguage"]["javascript"], 8)
        self.assertEqual(report["previousP2ValidationExcluded"], 72)
        self.assertFalse(report["m12ReuseRightsEstablished"])
        self.assertFalse(report["m12TrainingAuthorized"])
        self.assertFalse(report["m12NewOwnerCorpusPrepared"])
        self.assertFalse(report["independentM12DevelopmentEstablished"])
        self.assertFalse(report["sealedFinalHoldoutAccessed"])
        self.assertFalse(report["modelWeightsLoaded"])
        self.assertFalse(report["trainingPerformed"])
        self.assertTrue(report["priorCorpus"]["train"]["requestAnswerFormat"])

    def test_cli_reads_original_without_gpu_or_new_artifacts(self):
        result = subprocess.run(
            [sys.executable, "-I", "-S", str(SCRIPT)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["kind"], "m12-a-read-only-historical-inventory-v1")
        self.assertFalse(report["m12TrainingAuthorized"])

    def test_changed_corpus_is_refused_by_source_bound_verifier(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "pilot"
            shutil.copytree(module.DATASET, copy)
            train = copy / "train.jsonl"
            train.write_bytes(train.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Split hash"):
                module.inventory(copy)


if __name__ == "__main__":
    unittest.main()
