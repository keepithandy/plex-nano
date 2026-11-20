"""CPU-only tests for the M11 EOS diagnostic. No torch, CUDA or model weights."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "plex_phase2_eos_probe.py"
_spec = importlib.util.spec_from_file_location("plex_phase2_eos_probe", SCRIPT)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


class EosDiagnosticSmoke(unittest.TestCase):
    def test_frozen_prompt_variants_only_change_the_intended_boundary(self):
        cases = [
            {"id": f"m11-{split}-{n}", "split": split, "snippet": f"let {name} = {n};\n"}
            for split, name in (("train", "score"), ("probe-only", "value"))
            for n in (0, 1)
        ]
        prompts = probe.make_prompts(
            cases,
            lambda case: f"Code:\n{case['snippet']}Copy: ",
            lambda task: "Request: " + task["request"] + "\nSource file:\n"
                         + task["inputFile"] + "\nEdited file:\n",
        )
        self.assertEqual(len(prompts), 12)
        self.assertEqual({p["template"] for p in prompts}, set(probe.TEMPLATES))
        for case in cases:
            views = {p["template"]: p["prompt"] for p in prompts if p["id"] == case["id"]}
            self.assertTrue(views["m11-trained"].endswith("Copy: "))
            self.assertEqual(views["m11-newline"], views["m11-trained"][:-1] + "\n")
            self.assertIn("Source file:\n" + case["snippet"], views["console-general"])
            self.assertTrue(views["console-general"].endswith("Edited file:\n"))
        with self.assertRaisesRegex(ValueError, "Frozen M11"):
            probe.make_prompts(cases, lambda c: "wrong", lambda c: "console")

    def test_first_token_metrics_identify_fitted_eos_and_unmasked_differences(self):
        # eos=token 3; code token 4; out-of-vocab token 8
        logits = [-2, -2, -2, 8, 5, 3, 0, 1, 12]
        result = probe.first_token_metrics(logits, 8, 4)
        self.assertEqual(result["fittedFirstTokenId"], 3)
        self.assertEqual(result["unmaskedFirstTokenId"], 8)
        self.assertEqual(result["eosFittedRank"], 1)
        self.assertEqual(result["eosMinusBestCodeLogit"], 3)
        self.assertEqual(result["goldMinusEosLogit"], -3)
        logits[4] = 9
        result = probe.first_token_metrics(logits, 8, 4)
        self.assertEqual(result["fittedFirstTokenId"], 4)
        self.assertEqual(result["eosFittedRank"], 2)
        self.assertEqual(result["eosMinusBestCodeLogit"], -1)
        with self.assertRaisesRegex(ValueError, "Invalid next-token"):
            probe.first_token_metrics([float("nan")] * 10, 8, 4)

    @staticmethod
    def example_rows():
        result = []
        for template in probe.TEMPLATES:
            for split, name in (("train", "score"), ("probe-only", "value")):
                for n in (0, 1):
                    is_original_train = template == "m11-trained" and split == "train"
                    is_eos = not is_original_train
                    result.append({
                        "id": f"m11-{split}-{name}-{n}",
                        "template": template, "split": split,
                        "exactAndEos": is_original_train,
                        "stop": "eos", "generatedTokens": 0 if is_eos else 7,
                        "firstToken": {"fittedFirstTokenId": 3 if is_eos else 4},
                    })
        return result

    def test_summary_keeps_train_and_unseen_separate(self):
        rows = self.example_rows()
        summary = probe.summarize(rows)
        self.assertEqual(summary["m11-trained"]["trainingExact"], 2)
        self.assertEqual(summary["m11-trained"]["probeExact"], 0)
        self.assertEqual(summary["m11-trained"]["immediateEos"], 2)
        self.assertEqual(summary["console-general"]["immediateEos"], 4)
        self.assertEqual(summary["m11-newline"]["immediateEos"], 4)
        with self.assertRaisesRegex(ValueError, "12 unique"):
            probe.summarize(rows[:-1])

    def test_results_are_new_only_and_receipt_hash_matches_probes(self):
        rows = self.example_rows()
        summary = probe.summarize(rows)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dest = root / "new-diag"
            class Console:
                DIGEST = "f" * 64
            probe.save_results(dest, root, rows, summary, Console, "a" * 64)
            receipt = json.loads((dest / "report.json").read_text(encoding="utf-8"))
            raw = (dest / "probes.jsonl").read_bytes()
            self.assertEqual(receipt["probeResponsesSha256"], hashlib.sha256(raw).hexdigest())
            self.assertFalse(receipt["trainingPerformed"])
            self.assertFalse(receipt["finalHoldoutAccessed"])
            self.assertEqual(receipt["generations"], 12)
            with self.assertRaises(FileExistsError):
                probe.save_results(dest, root, rows, summary, Console, "b" * 64)
            self.assertEqual(raw, (dest / "probes.jsonl").read_bytes())

    def test_output_destination_requires_fresh_artifacts_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = probe.ARTIFACTS
            probe.ARTIFACTS = root
            try:
                good, anchored = probe.output_destination(root / "new")
                self.assertEqual(good, (root / "new").resolve())
                self.assertEqual(anchored, root.resolve())
                with self.assertRaisesRegex(ValueError, "NEW output"):
                    probe.output_destination(root.parent / "wrong")
                existing = root / "exists"
                existing.mkdir()
                with self.assertRaisesRegex(ValueError, "NEW output"):
                    probe.output_destination(existing)
            finally:
                probe.ARTIFACTS = old

    def test_no_flag_refuses_before_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "future"
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(SCRIPT), "--output-dir", str(output)],
                capture_output=True, text=True, timeout=20,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Add --execute-read-only", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
