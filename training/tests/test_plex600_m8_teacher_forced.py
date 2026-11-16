"""CPU-only tests for M8 reference-likelihood diagnostic, not model inference."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import math
import unittest
from unittest.mock import patch

import torch

from plex_training import plex600_m8_teacher_forced as diag


class M8TeacherForcedDiagnosticTests(unittest.TestCase):
    def test_cli_is_plan_only_and_no_model_or_checkpoint_load(self):
        with patch.object(diag, "run", side_effect=AssertionError("GPU diagnostic called")), \
             patch.object(diag.m8, "prerequisites", side_effect=AssertionError("checkpoint read")), \
             redirect_stdout(StringIO()) as stdout:
            self.assertEqual(diag.main([]), 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "plan-only")
        self.assertEqual(result["maxCases"], 6)
        self.assertTrue(result["goldAnswerProvidedToModel"])
        self.assertEqual(result["generatedAnswers"], 0)
        self.assertFalse(result["independentEditSuccessMeasured"])
        self.assertFalse(result["trainingAuthorized"])
        self.assertFalse(result["modelLoaded"])
        self.assertFalse(result["finalHoldoutUsed"])

    def test_bounds_and_required_inputs_rejected_before_checkpoint_read(self):
        with patch.object(diag, "run", side_effect=AssertionError("GPU called")), \
             patch.object(diag.m8, "prerequisites",
                          side_effect=AssertionError("checkpoint read")), \
             redirect_stderr(StringIO()) as stderr:
            self.assertEqual(diag.main(["--run-read-only-teacher-forced"]), 2)
            self.assertEqual(diag.main(["--max-cases", "37"]), 2)
        self.assertIn("all pinned local inputs", stderr.getvalue())
        self.assertIn("1..36", stderr.getvalue())

    def test_reference_alignment_scores_first_gold_after_prompt(self):
        inputs, start = diag.reference_alignment([7, 8], [4, 3])
        self.assertEqual(inputs, [7, 8, 4])
        self.assertEqual(start, 1)
        self.assertEqual(len(inputs) - start, 2)
        with self.assertRaisesRegex(ValueError, "context"):
            diag.reference_alignment([], [4, 3])
        with self.assertRaisesRegex(ValueError, "context"):
            diag.reference_alignment([7], [4])
        with self.assertRaisesRegex(ValueError, "context"):
            diag.reference_alignment([7] * 1024, [4, 3])
        with self.assertRaisesRegex(ValueError, "context"):
            diag.reference_alignment([7], [4] * 96 + [3])

    def test_real_torch_cpu_likelihood_flags_unfitted_logit_winner(self):
        # Two source-conditioned, gold reference positions. First predicts the
        # correct code token (4); second predicts an unavailable token (6),
        # although the reference EOS is token 3.
        logits = torch.full((2, 8), -10.0)
        logits[0, 4] = 5.0
        logits[1, 6] = 5.0
        logits[1, 3] = 0.0
        report = diag.score_logits(
            logits, [4, 3], fitted_vocab_size=5, torch=torch)
        self.assertEqual(report["referenceTokensIncludingEos"], 2)
        self.assertEqual(report["unmaskedGreedyReferenceMatches"], 1)
        self.assertEqual(report["unmaskedGreedyOutsideFittedVocab"], 1)
        self.assertEqual(report["unmaskedGreedyInvalidControl"], 0)
        self.assertEqual(report["unmaskedGreedyPrematureEos"], 0)
        self.assertLess(report["fittedVocabProbabilityMassSum"], 2)
        self.assertGreater(report["fittedVocabProbabilityMassSum"], 1)
        self.assertGreater(
            report["unmaskedReferenceNllSum"],
            report["conditionalFittedVocabReferenceNllSum"] + 4)
        self.assertTrue(report["goldConditionedNotGenerated"])

    def test_nll_is_logit_shift_invariant_and_token_weighted(self):
        a = torch.full((2, 8), -2.0)
        a[0, 4], a[1, 3] = 4.0, 1.0
        one = diag.score_logits(a, [4, 3], fitted_vocab_size=5, torch=torch)
        shifted = diag.score_logits(
            a + 1000, [4, 3], fitted_vocab_size=5, torch=torch)
        self.assertAlmostEqual(
            one["unmaskedReferenceNllSum"],
            shifted["unmaskedReferenceNllSum"], places=3)
        combined = diag.summarize([one, one, shifted])
        self.assertEqual(combined["caseCount"], 3)
        self.assertEqual(combined["referenceTokensIncludingEos"], 6)
        self.assertAlmostEqual(
            combined["unmaskedReferenceMeanNllNats"],
            one["unmaskedReferenceNllSum"] / 2, places=3)
        self.assertEqual(combined["unmaskedGreedyReferenceTokenAccuracy"], 1)
        self.assertTrue(math.isfinite(combined["conditionalFittedVocabMeanNllNats"]))

    def test_premature_eos_is_separate_from_control_token_rate(self):
        logits = torch.zeros((2, 8))
        logits[0, 3] = 10.0  # premature EOS instead of first gold token 4
        logits[1, 1] = 10.0  # invalid control token instead of terminal EOS 3
        case = diag.score_logits(
            logits, [4, 3], fitted_vocab_size=5, torch=torch)
        self.assertEqual(case["unmaskedGreedyPrematureEos"], 1)
        self.assertEqual(case["unmaskedGreedyInvalidControl"], 1)
        self.assertEqual(case["unmaskedGreedyReferenceMatches"], 0)
        aggregate = diag.summarize([case])
        self.assertEqual(aggregate["unmaskedGreedyPrematureEosRate"], 0.5)
        self.assertEqual(aggregate["unmaskedGreedyInvalidControlRate"], 0.5)

    def test_invalid_reference_shape_and_nonfinite_logits_refuse(self):
        valid = torch.zeros((2, 8))
        invalid = (
            ([4, 2], 5, valid),  # no EOS
            ([0, 3], 5, valid),  # special token in answer
            ([5, 3], 5, valid),  # invalid fitted-vocab label
            ([4, 3], 3, valid),  # invalid fitted vocabulary
            ([4, 3], 9, valid),  # output vocab too small
            ([4, 3], 5, valid[:1]),  # shape disagreement
            ([3], 5, valid[:1]),  # empty expected file is unsupported
        )
        for reference, vocab, logits in invalid:
            with self.subTest(reference=reference, vocab=vocab), \
                 self.assertRaisesRegex(ValueError, "mismatch"):
                diag.score_logits(
                    logits, reference, fitted_vocab_size=vocab, torch=torch)
        bad = valid.clone()
        bad[0, 4] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            diag.score_logits(
                bad, [4, 3], fitted_vocab_size=5, torch=torch)
        with self.assertRaisesRegex(ValueError, "Empty"):
            diag.summarize([])


if __name__ == "__main__":
    unittest.main()
