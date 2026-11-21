"""CPU-only tests of M12-A teacher-forced scorer. No torch or checkpoint."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("plex_m12_source_conditioning.py")
spec = importlib.util.spec_from_file_location("plex_m12_source_conditioning", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class SourceConditioningSmoke(unittest.TestCase):
    def test_frozen_input_and_delimiters(self):
        cases = [
            {"id": f"m11-{split}-{name}-{digit}", "split": split,
             "snippet": f"let {name} = {digit};\n"}
            for split, name in (("train", "score"), ("probe-only", "value"))
            for digit in ("0", "1")
        ]
        prepared = probe.planned_cases(
            cases,
            lambda case: f"Code:\n{case['snippet']}Copy: ",
            lambda task: f"Request: {task['request']}\nSource file:\n"
                         f"{task['inputFile']}\nEdited file:\n",
        )
        self.assertEqual(len(prepared), 8)
        self.assertEqual({r["template"] for r in prepared}, set(probe.TEMPLATES))
        self.assertEqual(sum(r["split"] == "probe-only" for r in prepared), 4)
        self.assertEqual(prepared[0]["digit"], "0")
        self.assertTrue(prepared[0]["prompt"].endswith("Copy: "))
        self.assertTrue(prepared[1]["prompt"].endswith("Edited file:\n"))
        with self.assertRaisesRegex(ValueError, "frozen M11"):
            probe.planned_cases(cases[:3], lambda case: "bad", lambda task: "bad")
        with self.assertRaisesRegex(ValueError, "delimiter"):
            probe.planned_cases(cases, lambda case: "Code:\nCopy:\n",
                                lambda task: "Edited file:\n")

    def test_gold_ranking_and_eos_loss_are_scored_independently(self):
        # fitted vocabulary IDs 3..6, EOS 3, code 4..6.
        data = probe.score_fitted_logits([3.0, 2.0, 5.0, 1.0], 5)
        self.assertEqual(data["goldRank"], 1)
        self.assertEqual(data["fittedGreedyId"], 5)
        self.assertEqual(data["eosMinusBestCodeLogit"], -2.0)
        self.assertEqual(data["goldMinusEosLogit"], 2.0)
        self.assertGreater(data["goldNegativeLogProbability"], 0)
        eos = probe.score_fitted_logits([7.0, 2.0, 5.0, 1.0], 3)
        self.assertEqual(eos["goldRank"], 1)
        sequence = probe.score_target([
            [3.0, 2.0, 5.0, 1.0], [7.0, 2.0, 5.0, 1.0],
        ], [5, 3])
        self.assertEqual(sequence["goldGreedyMatches"], 2)
        self.assertEqual(sequence["prematureEosSteps"], 0)
        self.assertTrue(sequence["teacherForced"])
        self.assertFalse(sequence["autonomousGenerationPerformed"])
        with self.assertRaisesRegex(ValueError, "fitted-vocabulary"):
            probe.score_fitted_logits([float("nan"), 2.0, 5.0], 5)
        with self.assertRaisesRegex(ValueError, "gold EOS"):
            probe.score_target([[1.0, 2.0]], [5])
        with self.assertRaisesRegex(ValueError, "missing or identical"):
            probe.contrast_logit([1.0, 2.0, 3.0], 4, 4)

    def test_source_sensitive_contrast_checks_changed_token(self):
        i, score, value = probe.first_difference([10, 13, 8], [10, 15, 8])
        self.assertEqual((i, score, value), (1, 13, 15))
        self.assertEqual(probe.contrast_logit([1, 5, 2], 4, 5), -3)
        self.assertEqual(probe.contrast_logit([1, 2, 5], 4, 5), 3)
        with self.assertRaisesRegex(ValueError, "common answer prefix"):
            probe.first_difference([4, 5], [6, 5])
        with self.assertRaisesRegex(ValueError, "identifier divergence"):
            probe.first_difference([4, 5], [4, 5])

    @staticmethod
    def synthetic_rows():
        rows = []
        for template in probe.TEMPLATES:
            for split, name in (("train", "score"), ("probe-only", "value")):
                for digit in ("0", "1"):
                    original = template == "m11-trained"
                    gold = (5 if original else 3)
                    rows.append({
                        "id": f"m11-{split}-{name}-{digit}",
                        "template": template, "split": split, "digit": digit,
                        "valueMinusScoreLogit": (-2 if split == "train" else 3)
                        if original else -4,
                        "scores": {
                            "meanFittedGoldNll": 0.5 if original else 7.5,
                            "goldGreedyMatches": 2 if original else 0,
                            "targetTokensPlusEos": 2,
                            "tokens": [{"fittedGreedyId": gold}],
                        },
                    })
        return rows

    def test_summary_separates_gold_prefix_from_actual_generation(self):
        result = probe.summarize(self.synthetic_rows())
        self.assertEqual(result["groups"]["m11-trained/train"]["firstTokenEosCases"], 0)
        self.assertEqual(result["groups"]["console-general/train"]["firstTokenEosCases"], 2)
        self.assertEqual(result["sourceContrasts"]["m11-trained"]["switchedPreferencePairs"], 2)
        self.assertEqual(result["sourceContrasts"]["console-general"]["switchedPreferencePairs"], 0)
        self.assertFalse(result["modelImprovementClaimed"])
        with self.assertRaisesRegex(ValueError, "eight unique"):
            probe.summarize(self.synthetic_rows()[:-1])

    def test_new_only_artifact_receipt_is_integrity_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dest = root / "fresh"
            rows = self.synthetic_rows()
            summary = probe.summarize(rows)
            probe.save_report(dest, root, rows, summary, "c" * 64)
            report = json.loads((dest / "report.json").read_text(encoding="utf-8"))
            raw = (dest / "rows.jsonl").read_bytes()
            self.assertEqual(report["responsesSha256"], hashlib.sha256(raw).hexdigest())
            for flag in ("autonomousGenerationPerformed", "trainingPerformed",
                         "optimizerCreated", "checkpointWrites", "finalHoldoutAccessed"):
                self.assertFalse(report[flag])
            self.assertTrue(report["goldTargetSuppliedToModel"])
            with self.assertRaises(FileExistsError):
                probe.save_report(dest, root, rows, summary, "c" * 64)

    def test_unguarded_cli_refuses_before_loading_cuda_or_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "future"
            r = subprocess.run(
                [sys.executable, "-I", "-S", str(SCRIPT),
                 "--output-dir", str(target)],
                capture_output=True, text=True, timeout=15,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("Add --execute-read-only", r.stderr)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
