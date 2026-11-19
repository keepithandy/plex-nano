"""CPU-only regression tests for the M11 training core; NO GPU training."""
from collections import Counter
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import unittest

from plex_training import plex600_m11_training_core as core
from plex_training import plex600_m11_source_conditioning as m11


class ByteTokenizer:
    vocabulary_size = 1191

    @staticmethod
    def encode(value):
        return list(value.encode("ascii"))

    @staticmethod
    def decode(ids):
        return bytes(ids).decode("ascii")


class M11TrainingCoreTests(unittest.TestCase):
    def test_default_is_inert_and_no_training_flag(self):
        with redirect_stdout(StringIO()) as output:
            core.main([])
        result = json.loads(output.getvalue())
        self.assertFalse(result["trainingAuthorized"])
        self.assertFalse(result["executionEntrypointImplemented"])
        self.assertEqual(result["optimizerUpdatesCompleted"], 0)
        self.assertFalse(result["modelLoaded"])
        self.assertFalse(result["checkpointWrites"])
        self.assertTrue(result["m10ApprovalCannotAuthorizeM11"])
        with redirect_stdout(StringIO()), self.assertRaises(SystemExit) as rejected:
            core.main(["--execute-approved-run"])
        self.assertEqual(rejected.exception.code, 2)

    def test_exact_alternating_schedule_and_isolated_development_probe(self):
        rows = core.exact_schedule()
        self.assertEqual(len(rows), 200)
        self.assertEqual([row["optimizerUpdate"] for row in rows], list(range(1, 201)))
        train = [case["id"] for case in m11.cases() if case["split"] == "train"]
        probe = {case["id"] for case in m11.cases() if case["split"] == "probe-only"}
        self.assertEqual([row["recordId"] for row in rows[:4]], train * 2)
        self.assertEqual(dict(Counter(row["recordId"] for row in rows)),
                         {train[0]: 100, train[1]: 100})
        self.assertFalse(probe & {row["recordId"] for row in rows})
        self.assertTrue(all(row["gradientAccumulationSteps"] == 1
                            and row["examplesInOptimizerUpdate"] == 1 for row in rows))
        self.assertEqual(core.CHECKPOINT_STEPS, (25, 50, 100, 200))
        self.assertEqual(core.schedule_sha256(), core.schedule_sha256())

    def test_answer_only_masks_and_eos_are_packed_without_probe(self):
        samples = core.packed_training_samples(ByteTokenizer())
        self.assertEqual(len(samples), 2)
        for case in m11.cases():
            if case["split"] != "train":
                self.assertNotIn(case["id"], samples)
                continue
            prompt = ByteTokenizer.encode(m11.prompt_for(case))
            target = ByteTokenizer.encode(case["snippet"]) + [core.EOS_ID]
            packed = samples[case["id"]]
            self.assertEqual(packed["inputIds"], prompt + target[:-1])
            self.assertEqual(packed["targetIds"], [-100] * (len(prompt) - 1) + target)
            self.assertEqual(packed["supervisedAnswerPositions"], len(target))
            self.assertEqual(packed["targetIds"][-1], core.EOS_ID)

    def test_eos_weight_reduces_terminal_gradient_not_content_gradient(self):
        import torch
        torch.manual_seed(117)
        base = torch.randn((1, 4, 32), dtype=torch.float32)
        targets = torch.tensor([[-100, 10, 11, 3]])
        full = base.clone().requires_grad_()
        partial = base.clone().requires_grad_()
        loss_full = core.weighted_copy_loss(torch, full, targets, 1.0)
        loss_partial = core.weighted_copy_loss(torch, partial, targets, 0.25)
        self.assertGreater(loss_full.item(), loss_partial.item())
        loss_full.backward()
        loss_partial.backward()
        self.assertTrue(torch.equal(full.grad[:, 1:3], partial.grad[:, 1:3]))
        self.assertTrue(torch.allclose(partial.grad[:, 3], full.grad[:, 3] * 0.25,
                                       atol=1e-8, rtol=1e-6))
        self.assertTrue(torch.equal(full.grad[:, 0], torch.zeros_like(full.grad[:, 0])))

    def test_eos_weight_invalid_masks_and_weights_refused(self):
        import torch
        logits = torch.zeros((1, 3, 16))
        with self.assertRaisesRegex(core.Refusal, "predeclared"):
            core.weighted_copy_loss(torch, logits, torch.tensor([[-100, 5, 3]]), 0)
        for target in ([-100, 3, 5], [-100, 5, 5], [-100, 3, 3],
                       [-100, -100, 3]):
            with self.subTest(target=target), self.assertRaisesRegex(
                    core.Refusal, "terminal EOS"):
                core.weighted_copy_loss(torch, logits, torch.tensor([target]), 1.0)

    def test_source_pair_logits_are_measured_in_both_directions(self):
        import torch

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
                self.constant = False

            def forward(self, x):
                source = bytes(x[0].tolist()).decode("ascii")
                selected = ord("0") if self.constant or " = 0;" in source else ord("1")
                result = torch.zeros((1, x.shape[1], 1191), device=x.device)
                result[0, -1, selected] = 4.0
                return result, None

        model = Toy()
        for split in ("train", "probe-only"):
            result = core.source_pair_measurement(torch, model, ByteTokenizer(), split)
            self.assertTrue(result["tracksSourceInBothDirections"])
            self.assertTrue(result["usesGoldAnswerPrefix"])
            self.assertFalse(result["autonomousExactCopyingEvaluated"])
        model.constant = True
        self.assertFalse(core.source_pair_measurement(
            torch, model, ByteTokenizer(), "train")["tracksSourceInBothDirections"])
        with self.assertRaisesRegex(core.Refusal, "split"):
            core.source_pair_measurement(torch, model, ByteTokenizer(), "heldout")

    def test_receipt_scores_reject_constant_wrong_eos_and_tampering(self):
        sha = "a" * 64
        cases = m11.cases()
        def make_rows():
            return [
                {
                    "id": case["id"],
                    "decodingPolicy": decoder,
                    "output": case["snippet"],
                    "terminationReason": "eos",
                    "promptSha256": hashlib.sha256(
                        m11.prompt_for(case).encode("utf-8")).hexdigest(),
                    "checkpointSha256": sha,
                }
                for decoder in core.DECODERS for case in cases
            ]
        rows = make_rows()
        scored = core.score_unverified_generations(rows, sha)
        self.assertFalse(scored["verifiedModelGeneration"])
        self.assertTrue(all(s["trainingPairExactAndEos"]
                            and s["developmentPairExactAndEos"]
                            for s in scored["scores"].values()))
        rows[1]["output"] = cases[0]["snippet"]
        self.assertFalse(core.score_unverified_generations(rows, sha)[
            "scores"][core.DECODERS[0]]["trainingPairExactAndEos"])
        rows = make_rows()
        rows[0]["terminationReason"] = "length-cap"
        self.assertFalse(core.score_unverified_generations(rows, sha)[
            "scores"][core.DECODERS[0]]["trainingPairExactAndEos"])
        rows = make_rows()
        rows[0]["promptSha256"] = "0" * 64
        with self.assertRaisesRegex(core.Refusal, "SHA"):
            core.score_unverified_generations(rows, sha)
        rows = make_rows()
        rows[0] = dict(rows[1])  # Preserve matching prompt SHA, isolate duplicate ID.
        with self.assertRaisesRegex(core.Refusal, "duplicate"):
            core.score_unverified_generations(rows, sha)


if __name__ == "__main__":
    unittest.main()
