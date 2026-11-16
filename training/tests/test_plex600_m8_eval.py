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

    def test_balanced_first_six_cases_include_all_languages(self):
        cases = m8.choose_cases(6)
        self.assertEqual(len(cases), 6)
        self.assertEqual({case["language"] for case in cases}, {"html", "css", "javascript"})
        self.assertEqual([case["language"] for case in cases],
                         ["html", "css", "javascript", "html", "css", "javascript"])

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
        with patch.object(m8.r, "m7_local_receipt",
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

    def test_token_budget_counts_gold_file_and_eos_without_generation(self):
        task = research.benchmark()[0]

        class CharacterTokenizer:
            def encode(self, value):
                return list(value.encode("utf-8"))

        result = m8.token_budget([task], CharacterTokenizer(), max_new_tokens=64)
        item = result["cases"][0]
        self.assertEqual(item["goldFileTokens"], len(task["expectedFile"].encode("utf-8")))
        self.assertEqual(item["requiredTokensIncludingEos"], item["goldFileTokens"] + 1)
        self.assertEqual(result["blockedByBudget"], 1)
        self.assertEqual(result["completeWithinBudget"], 0)
        self.assertEqual(result["byLanguage"]["html"]["selected"], 1)
        self.assertFalse(result["modelLoaded"])
        self.assertFalse(result["trainingAuthorized"])

    def test_token_budget_positive_case_and_context_limit(self):
        task = research.benchmark()[0]

        class ShortTokenizer:
            def encode(self, value):
                return [5] * (4 if value == task["expectedFile"] else 10)

        result = m8.token_budget([task], ShortTokenizer(), max_new_tokens=5)
        self.assertEqual(result["completeWithinBudget"], 1)
        self.assertEqual(result["requiredMaxNewTokensIncludingEos"], 5)
        self.assertEqual(result["requiredMaxTotalContext"], 15)
        with self.assertRaisesRegex(ValueError, "budget"):
            m8.token_budget([task], ShortTokenizer(), max_new_tokens=65)

    def test_budget_audit_uses_only_verified_bundle_not_weights_or_gpu(self):
        class CharacterTokenizer:
            def encode(self, value):
                return list(value.encode("utf-8"))

        from plex_training.tokenizer import PlexTokenizer
        with patch.object(m8, "verified_bundle",
                          return_value={"tokenizerSha256": "test",
                                        "benchmarkSha256": "benchmark"}), \\
             patch.object(PlexTokenizer, "load", return_value=CharacterTokenizer()), \\
             patch.object(m8, "prerequisites", side_effect=AssertionError("weights opened")), \\
             patch.object(m8, "evaluate", side_effect=AssertionError("GPU requested")), \\
             redirect_stdout(StringIO()) as output:
            self.assertEqual(m8.main([
                "--audit-token-budget", "--bundle-dir", "dummy",
                "--bundle-manifest-sha256", research.BUNDLE_SHA,
                "--max-cases", "3", "--max-new-tokens", "64"]), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["selectedCases"], 3)
        self.assertEqual(report["bundleManifestSha256"], research.BUNDLE_SHA)
        self.assertEqual(report["blockedByBudget"], 3)
        self.assertFalse(report["modelLoaded"])

    def test_undersized_budget_refuses_before_torch_model_or_cuda(self):
        class CharacterTokenizer:
            def encode(self, value):
                return list(value.encode("utf-8"))

        from plex_training.tokenizer import PlexTokenizer
        with patch.object(m8, "require_output", return_value=Path("unused")), \\
             patch.object(m8, "prerequisites", return_value={"benchmarkSha256": "test"}), \\
             patch.object(PlexTokenizer, "load", return_value=CharacterTokenizer()), \\
             patch.object(m8.hardware, "_check_device", side_effect=AssertionError("GPU accessed")):
            with self.assertRaisesRegex(ValueError, "under-budget"):
                m8.evaluate(Path("bundle"), Path("report"), Path("weights"),
                            Path("output"), bundle_sha=research.BUNDLE_SHA,
                            max_cases=3, max_new_tokens=64)

    def test_eval_is_not_implicitly_launched_from_scoring(self):
        with patch.object(m8, "evaluate", side_effect=AssertionError("GPU accessed")), \
             redirect_stdout(StringIO()) as output:
            self.assertEqual(research.main(["benchmark-inspect"]), 0)
        self.assertEqual(json.loads(output.getvalue())["rows"], 36)


if __name__ == "__main__":
    unittest.main()
