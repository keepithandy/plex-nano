"""M8 read-only inference harness preflight tests. NO GPU or checkpoint model load."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from plex_training import plex600_m8_eval as m8
from plex_training import plex600_research as research


class Plex600M8EvalTests(unittest.TestCase):
    def test_default_is_true_no_load_no_dataset_touch(self):
        with patch.object(m8, "evaluate", side_effect=AssertionError("GPU eval called")), \
             patch.object(m8, "prerequisites", side_effect=AssertionError("checkpoint opened")), \
             redirect_stdout(StringIO()) as out:
            self.assertEqual(m8.main([]), 0)
        result = json.loads(out.getvalue())
        self.assertEqual(result["status"], "plan")
        self.assertEqual(result["optimizerUpdates"], 0)
        self.assertFalse(result["modelLoaded"])
        self.assertFalse(result["trainingAuthorized"])
        self.assertFalse(result["finalHoldoutUsed"])
        self.assertEqual(result["maxCases"], 6)

    def test_unknown_greedy_bounds_refuse_before_model(self):
        for kw in ({"max_cases": 0}, {"max_cases": 37}, {"max_new_tokens": 0},
                   {"max_new_tokens": 65}, {"max_cases": True}):
            with self.subTest(kw=kw), self.assertRaisesRegex(ValueError, "bounded"):
                m8.plan(**kw)

    def test_prompt_contains_exact_original_and_request(self):
        for case in research.benchmark():
            prompt = m8.prompt_for(case)
            self.assertIn(case["request"], prompt)
            self.assertIn(case["inputFile"], prompt)
            self.assertTrue(prompt.endswith("\nEdited file:\n"))
            self.assertNotIn(case["expectedFile"] + "\nEdited file:\n", prompt)

    def test_execution_requires_explicit_inputs(self):
        with patch.object(m8, "evaluate", side_effect=AssertionError("GPU called")), \
             redirect_stderr(StringIO()) as err:
            self.assertEqual(m8.main(["--run-read-only-eval"]), 2)
        self.assertIn("all pinned local inputs", err.getvalue())

    def test_missing_local_m7_receipt_refused_before_gpu(self):
        with patch.object(m8.r, "m7_local_receipt", side_effect=ValueError("invalid SHA")), \
             patch.object(m8.hardware, "_check_device", side_effect=AssertionError("GPU accessed")):
            with self.assertRaisesRegex(ValueError, "SHA"):
                m8.prerequisites(Path("unused"), Path("unused"), Path("unused"), research.BUNDLE_SHA)

    def test_source_bundle_mismatch_refuses_without_weights(self):
        with patch.object(m8.research, "m7_local_receipt",
                          side_effect=AssertionError("weights touched")):
            with self.assertRaisesRegex(ValueError, "pinned"):
                m8.prerequisites(Path("unused"), Path("unused"), Path("unused"), "0" * 64)

    def test_output_must_be_create_only_in_artifacts(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "bundle").mkdir()
            (root / "weights.pt").write_bytes(b"not an actual checkpoint")
            with patch.object(m8, "OUTPUT_ROOT", root):
                expected = (root / "result").resolve()
                actual = m8.require_output(root / "result", bundle=root / "bundle",
                                           checkpoint=root / "weights.pt")
                self.assertEqual(actual, expected)
                expected.mkdir()
                with self.assertRaisesRegex(ValueError, "new directory"):
                    m8.require_output(expected, bundle=root / "bundle",
                                      checkpoint=root / "weights.pt")
                with self.assertRaisesRegex(ValueError, "direct child"):
                    nested = root / "nested"
                    nested.mkdir()
                    m8.require_output(nested / "run", bundle=root / "bundle",
                                      checkpoint=root / "weights.pt")

    def test_eval_is_not_implicitly_launched_from_scoring(self):
        with patch.object(m8, "evaluate", side_effect=AssertionError("GPU accessed")), \
             redirect_stdout(StringIO()) as output:
            self.assertEqual(research.main(["benchmark-inspect"]), 0)
        self.assertEqual(json.loads(output.getvalue())["rows"], 36)


if __name__ == "__main__":
    unittest.main()
