"""M8 read-only inference harness preflight tests. NO GPU or checkpoint model load."""
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from plex_training import plex600_m8_eval as m8
from plex_training import plex600_m8_reproduce as m8_reproduce
from plex_training import plex600_research as research


class Plex600M8EvalTests(unittest.TestCase):
    def test_frozen_m7_training_has_no_m8_file_conditioning_format(self):
        rows = research.read_jsonl(m8_reproduce.owner.DATASET / "train.jsonl")
        report = m8_reproduce.curriculum_format_audit(
            rows, m8.choose_cases(36))
        self.assertEqual(report["m7TrainingRows"], 120)
        self.assertEqual(report["m7RequestAnswerRows"], 120)
        self.assertEqual(report["m7RowsWithSourceFileMarker"], 0)
        self.assertEqual(report["m7RowsWithEditedFileMarker"], 0)
        self.assertEqual(report["m7RowsWithBothFileMarkers"], 0)
        self.assertEqual(report["m8CasesWithSourceAndEditedFilePromptMarkers"], 36)
        self.assertTrue(report["literalPromptFormatMismatch"])
        self.assertFalse(report["semanticModelIncapacityProven"])
        self.assertFalse(report["trainingAuthorized"])
        self.assertFalse(report["finalHoldoutUsed"])

    def test_format_gap_rejects_empty_corpora(self):
        with self.assertRaisesRegex(ValueError, "both frozen"):
            m8_reproduce.curriculum_format_audit([], m8.choose_cases(3))
        with self.assertRaisesRegex(ValueError, "both frozen"):
            m8_reproduce.curriculum_format_audit(
                [{"text": "Request: x\nAnswer:\ny"}], [])

    def test_tokenizer_reconstruction_default_is_plan_only(self):
        with patch.object(m8_reproduce, "reconstruct",
                          side_effect=AssertionError("tokenizer fitted")), \
             redirect_stdout(StringIO()) as stdout:
            self.assertEqual(m8_reproduce.main([]), 0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["status"], "plan-only")
        self.assertFalse(plan["modelLoaded"])
        self.assertFalse(plan["trainingAuthorized"])
        self.assertFalse(plan["finalHoldoutUsed"])

    def test_tokenizer_reconstruction_rejects_oversized_case_count(self):
        with patch.object(m8_reproduce.owner, "verify",
                          side_effect=AssertionError("dataset accessed")), \
             redirect_stderr(StringIO()) as stderr:
            self.assertEqual(m8_reproduce.main([
                "--reconstruct-read-only", "--max-cases", "37"]), 2)
        self.assertIn("1..36", stderr.getvalue())

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
                   {"max_new_tokens": 97}, {"max_cases": True}):
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

    def test_generation_stop_reasons_are_diagnostic_not_token_masking(self):
        self.assertEqual(m8.classify_next_token(3, 1191), "eos")
        self.assertEqual(m8.classify_next_token(1, 1191), "invalid-control-token")
        self.assertEqual(m8.classify_next_token(-1, 1191), "invalid-control-token")
        self.assertEqual(m8.classify_next_token(1191, 1191), "outside-fitted-tokenizer")
        self.assertEqual(m8.classify_next_token(32767, 1191), "outside-fitted-tokenizer")
        self.assertIsNone(m8.classify_next_token(1190, 1191))

        class FakeLogits:
            def __getitem__(self, key):
                return self

        class FakeModel:
            def eval(self):
                pass

            def __call__(self, inputs):
                return FakeLogits(), None

        class FakeTorch:
            long = "long"
            bfloat16 = "bfloat16"

            def __init__(self, ids):
                self.tokens = iter(ids)

            def inference_mode(self):
                return nullcontext()

            def autocast(self, *args, **kwargs):
                return nullcontext()

            def tensor(self, *args, **kwargs):
                return None

            def argmax(self, unused):
                return SimpleNamespace(item=lambda: next(self.tokens))

        class FakeTokenizer:
            vocabulary_size = 1191

            def decode(self, ids):
                return ",".join(map(str, ids))

        checks = (
            ([1191], 2, ("", 0, "outside-fitted-tokenizer")),
            ([1], 2, ("", 0, "invalid-control-token")),
            ([3], 2, ("", 0, "eos")),
            ([1190, 3], 2, ("1190", 1, "eos")),
            ([4, 5], 2, ("4,5", 2, "max-new-tokens")),
        )
        for predicted, budget, expected in checks:
            with self.subTest(predicted=predicted):
                result = m8.generate_one(
                    FakeModel(), FakeTorch(predicted), FakeTokenizer(), [4, 5],
                    device="not-gpu", max_new_tokens=budget)
                self.assertEqual(result, expected)

    def test_fitted_vocab_decoding_is_explicit_and_does_not_modify_logits(self):
        import torch

        scores = torch.tensor([-100.0, -100.0, -100.0, 2.0, 6.0, 8.0,
                               40.0, -4.0])
        frozen = scores.clone()
        # Original unmasked decoder chooses an unavailable ID (6).
        self.assertEqual(m8.select_next_id(
            scores, torch, 6, "unmasked-v1"), 6)
        # Guarded decoding chooses the best *real* code-token ID (5).
        self.assertEqual(m8.select_next_id(
            scores, torch, 6, "fitted-vocab-v1"), 5)
        self.assertTrue(torch.equal(scores, frozen))
        scores[3] = 100.0
        # EOS remains available as a legitimate stop.
        self.assertEqual(m8.select_next_id(
            scores, torch, 6, "fitted-vocab-v1"), 3)
        self.assertEqual(m8.classify_next_token(3, 6), "eos")
        with self.assertRaisesRegex(ValueError, "output head"):
            m8.select_next_id(scores, torch, 10, "fitted-vocab-v1")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            m8.select_next_id(scores, torch, 6, "unrecognized")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            m8.plan(decoding_policy="unrecognized")

    def test_versioned_decoding_policy_stays_off_by_default(self):
        baseline = m8.plan(max_cases=6, max_new_tokens=96)
        candidate = m8.plan(
            max_cases=6, max_new_tokens=96,
            decoding_policy="fitted-vocab-v1")
        self.assertEqual(baseline["decodingPolicy"], "unmasked-v1")
        self.assertTrue(baseline["historicalUnmaskedPolicy"])
        self.assertEqual(candidate["decodingPolicy"], "fitted-vocab-v1")
        self.assertFalse(candidate["historicalUnmaskedPolicy"])
        self.assertFalse(candidate["trainingAuthorized"])
        with patch.object(m8, "evaluate",
                          side_effect=AssertionError("weights loaded")), \
             redirect_stdout(StringIO()) as out:
            self.assertEqual(m8.main([
                "--decoding-policy", "fitted-vocab-v1"]), 0)
        self.assertEqual(
            json.loads(out.getvalue())["decodingPolicy"],
            "fitted-vocab-v1")

    def test_evidence_backed_96_token_budget_requires_explicit_opt_in(self):
        proposal = m8.plan(max_cases=6, max_new_tokens=96)
        self.assertEqual(proposal["maxNewTokensPerCase"], 96)
        self.assertFalse(proposal["modelLoaded"])
        self.assertFalse(proposal["trainingAuthorized"])
        with self.assertRaisesRegex(ValueError, "bounded"):
            m8.plan(max_cases=6, max_new_tokens=97)

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
            m8.token_budget([task], ShortTokenizer(), max_new_tokens=97)

    def test_budget_audit_uses_only_verified_bundle_not_weights_or_gpu(self):
        class CharacterTokenizer:
            def encode(self, value):
                return list(value.encode("utf-8"))

        from plex_training.tokenizer import PlexTokenizer
        with patch.object(m8, "verified_bundle",
                          return_value={"tokenizerSha256": "test",
                                        "benchmarkSha256": "benchmark"}), \
             patch.object(PlexTokenizer, "load", return_value=CharacterTokenizer()), \
             patch.object(m8, "prerequisites", side_effect=AssertionError("weights opened")), \
             patch.object(m8, "evaluate", side_effect=AssertionError("GPU requested")), \
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
        with patch.object(m8, "require_output", return_value=Path("unused")), \
             patch.object(m8, "prerequisites", return_value={"benchmarkSha256": "test"}), \
             patch.object(PlexTokenizer, "load", return_value=CharacterTokenizer()), \
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
