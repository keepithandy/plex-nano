"""CPU-only tests for opt-in M9 checkpoint copying baseline; never run CUDA."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from plex_training import plex600_m10_baseline as baseline
from plex_training import plex600_m10_copy_proposal as proposal
from plex_training import plex600_research as research


class TinyTokenizer:
    vocabulary_size = proposal.TOKENIZER_VOCABULARY

    def encode(self, value: str) -> list[int]:
        return [ord(x) + 4 for x in value]

    def decode(self, values: list[int]) -> str:
        return "".join(chr(x - 4) for x in values)


class ReadOnlyBaselineTests(unittest.TestCase):
    def test_default_prints_plan_without_model_checkpoint_dataset_access(self):
        with patch.object(baseline.proposal, "rows",
                          side_effect=AssertionError("loaded unapproved inputs")), \
             patch.object(baseline.proposal, "audit_m9_backup",
                          side_effect=AssertionError("loaded M9 weights")), \
             redirect_stdout(StringIO()) as output:
            self.assertEqual(baseline.main([]), 0)
        result = json.loads(output.getvalue())
        self.assertFalse(result["modelLoaded"])
        self.assertFalse(result["artifactWrites"])
        self.assertFalse(result["trainingAuthorized"])
        self.assertFalse(result["checkpointWrites"])
        self.assertFalse(result["gradientEnabled"])
        self.assertFalse(result["optimizerStateLoaded"])
        self.assertEqual(result["optimizerUpdates"], 0)
        self.assertEqual(result["decodingPolicies"], list(baseline.DECODERS))

    def test_explicit_versioned_decoding_policies(self):
        self.assertEqual(baseline.policy_names("both"), baseline.DECODERS)
        self.assertEqual(baseline.policy_names("unmasked-v1"), ("unmasked-v1",))
        self.assertEqual(baseline.policy_names("fitted-vocab-v1"), ("fitted-vocab-v1",))
        with self.assertRaises(baseline.Refusal):
            baseline.policy_names("temperature-v1")

    def test_warmup_first_and_exact_18_unique_balanced_candidates(self):
        cases = baseline.selected_candidates(18)
        self.assertEqual(len(cases), 18)
        self.assertEqual(len({x["id"] for x in cases}), 18)
        self.assertEqual(cases[0]["id"], proposal.WARMUP_RECORD_ID)
        self.assertEqual({x["id"] for x in cases}, {x["id"] for x in proposal.rows()})
        self.assertEqual(len(baseline.selected_candidates(3)), 3)
        self.assertEqual({x["language"] for x in baseline.selected_candidates(3)},
                         {"html", "css", "javascript"})
        for limit in (0, 19, -1, 1.5, "18", True):
            with self.subTest(limit=limit), self.assertRaises(baseline.Refusal):
                baseline.selected_candidates(limit)

    def test_partial_and_full_score_detects_repetition_eos_and_wrong_lengths(self):
        cases = baseline.selected_candidates(18)
        perfect = [
            {"id": x["id"], "output": x["snippet"], "terminationReason": "eos"}
            for x in cases
        ]
        result = baseline.score_observations(cases, perfect)
        self.assertTrue(result["allExactAndEos"])
        self.assertTrue(result["complete18CaseComparison"])
        self.assertEqual(result["bySplit"]["train"]["exactWithEos"], 12)
        self.assertEqual(result["bySplit"]["heldout"]["exactWithEos"], 6)
        self.assertEqual(result["bySplit"]["train"]["meanCharacterSimilarity"], 1.0)
        self.assertFalse(result["codeEditingImprovementProven"])
        changed = [dict(x) for x in perfect]
        changed[0]["output"] = " " * 96
        changed[-1]["output"] = "." * 96
        changed[-1]["terminationReason"] = "max-new-tokens"
        score = baseline.score_observations(cases, changed)
        self.assertFalse(score["allExactAndEos"])
        self.assertEqual(sum(score["bySplit"][s]["spacesOnly"] for s in proposal.SPLITS), 1)
        self.assertEqual(sum(score["bySplit"][s]["periodsOnly"] for s in proposal.SPLITS), 1)
        self.assertEqual(sum(score["bySplit"][s]["repetitionCollapse"] for s in proposal.SPLITS), 2)
        warmup_score = baseline.score_observations(cases[:1], perfect[:1])
        self.assertFalse(warmup_score["complete18CaseComparison"])
        self.assertFalse(warmup_score["allExactAndEos"])
        self.assertIsNone(warmup_score["bySplit"]["heldout"]["meanCharacterSimilarity"])
        with self.assertRaises(baseline.Refusal):
            baseline.score_observations(cases, perfect[:-1])
        with self.assertRaises(baseline.Refusal):
            baseline.score_observations(cases, perfect[:-1] + [perfect[0]])

    def test_cpu_input_audit_checks_frozen_sources_without_gpu(self):
        checkpoint = {"m9WeightsSha256": "a" * 64, "m9ReportSha256": "b" * 64}
        selection = proposal.audit()
        with patch.object(proposal, "audit_m9_backup", return_value=checkpoint), \
             patch.object(baseline.m8, "verified_bundle",
                          return_value={"sourceBinding": {"sourceBindingVerified": True}}), \
             patch("plex_training.tokenizer.PlexTokenizer.load",
                   return_value=TinyTokenizer()), \
             patch.object(proposal, "audit",
                          return_value={**selection, "tokenizerSha256": "c" * 64}):
            result = baseline.audit_inputs(
                Path("bundle"), research.BUNDLE_SHA,
                Path("m9-report.json"), Path("weights.pt"),
                max_cases=3)
        self.assertEqual(result["selectedCases"], 3)
        self.assertEqual(result["m9WeightsSha256"], "a" * 64)
        self.assertEqual(len(result["sampleBudget"]), 3)
        self.assertTrue(result["selectedCasesIncludeWarmup"])
        self.assertTrue(result["sourceBindingVerified"])
        self.assertEqual(result["optimizerUpdates"], 0)
        self.assertFalse(result["trainingAuthorized"])
        self.assertFalse(result["artifactWrites"])

    def test_wrong_bundle_sha_refused_before_checkpoint_access(self):
        with patch.object(proposal, "audit_m9_backup",
                          side_effect=AssertionError("opened checkpoint")):
            with self.assertRaisesRegex(baseline.Refusal, "pinned"):
                baseline.audit_inputs(
                    Path("missing"), "unapproved",
                    Path("missing"), Path("missing"))

    def test_existing_output_refused_before_any_model_or_checkpoint_reads(self):
        with patch.object(baseline.m8, "require_output",
                          side_effect=baseline.Refusal("destination exists")), \
             patch.object(baseline, "audit_inputs",
                          side_effect=AssertionError("opened checkpoint")):
            with self.assertRaisesRegex(baseline.Refusal, "destination exists"):
                baseline.run_baseline(Path("bundle"), research.BUNDLE_SHA,
                                      Path("report"), Path("weights"), Path("output"),
                                      max_cases=3, mode="both")

    def test_cli_refuses_missing_evidence_training_flags_and_destinations(self):
        with redirect_stderr(StringIO()):
            self.assertEqual(baseline.main(["--run-read-only-baseline"]), 2)
            self.assertEqual(baseline.main(["--audit-inputs"]), 2)
            self.assertEqual(baseline.main(["--m9-weights", "weights.pt"]), 2)
            with self.assertRaises(SystemExit) as refusal:
                baseline.main(["--execute-approved-run"])
        self.assertEqual(refusal.exception.code, 2)
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as refusal:
                baseline.main(["--run-read-only-baseline", "--audit-inputs"])
        self.assertEqual(refusal.exception.code, 2)

    def test_module_has_no_optimizer_backward_train_or_checkpoint_save_calls(self):
        source = Path(baseline.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        forbidden = {
            "torch.save", "optimizer.step", "model.train", "loss.backward",
            "torch.autograd.grad", "torch.enable_grad",
        }
        self.assertFalse(calls & forbidden, calls & forbidden)
        self.assertNotIn("train_worker", source)
        self.assertNotIn("--execute-approved-run", source)


if __name__ == "__main__":
    unittest.main()
