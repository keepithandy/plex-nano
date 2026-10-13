"""Portable fail-closed P3-02 bounded runner authorization and masked-update tests."""

from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from plex_training import file_edit_replacements_run as p302
from plex_training.cli import build_parser


CONTRACT = Path("training/pretraining/p3-02-first-run-contract.json")


class FileEditReplacementRunTests(unittest.TestCase):
    def _contract(self) -> dict:
        return json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_committed_packet_is_draft_and_exact_prior_evidence(self) -> None:
        packet = self._contract()
        self.assertFalse(p302.validate_authorization(packet, allow_draft=True))
        self.assertFalse(packet["modelTrainingAuthorized"])
        self.assertEqual(packet["source"]["stageCheckpointSha256"], p302.EXPECTED_STAGE_SHA256)
        self.assertEqual(packet["source"]["bundleManifestSha256"], p302.BUNDLE_SHA256)
        self.assertEqual(packet["baseline"]["meanLoss"], p302.EXPECTED_BASELINE_LOSS)
        self.assertEqual(packet["training"]["maximumOptimizerUpdates"], 100)
        self.assertEqual(packet["training"]["maximumWallTimeSeconds"], 600)
        self.assertEqual(packet["training"]["expectedAnswerAndEosPositionsAt100Steps"], 60445)
        self.assertTrue(packet["protectedEvaluation"]["developmentExcludedFromGradients"])
        self.assertFalse(packet["training"]["automaticContinuation"])
        self.assertFalse(packet["training"]["resumeAllowed"])
        with self.assertRaisesRegex(ValueError, "NOT authorized"):
            p302.validate_authorization(packet, allow_draft=False)
        p302.validate_frozen_baseline(p302._bounded_json(p302.DEFAULT_BASELINE))

    def test_approval_requires_owner_and_valid_date(self) -> None:
        packet = self._contract()
        packet.update(status=p302.APPROVED_STATUS, modelTrainingAuthorized=True,
                      approvedBy="keepithandy", approvedDate="2026-10-13")
        self.assertTrue(p302.validate_authorization(packet, allow_draft=False))
        for bad in ("", "2026-13-13", "2026-02-30", True, 20261013):
            broken = copy.deepcopy(packet)
            broken["approvedDate"] = bad
            with self.subTest(date=bad), self.assertRaisesRegex(ValueError, "approval date|approvedDate"):
                p302.validate_authorization(broken, allow_draft=False)
        broken = copy.deepcopy(packet)
        broken["approvedBy"] = "unknown"
        with self.assertRaisesRegex(ValueError, "approvedBy"):
            p302.validate_authorization(broken, allow_draft=False)

    def test_tampered_training_packet_fails_closed(self) -> None:
        packet = self._contract()
        changes = [
            (("training", "maximumOptimizerUpdates"), 101),
            (("training", "maximumOptimizerUpdates"), True),
            (("training", "maximumWallTimeSeconds"), 601),
            (("training", "gradientAccumulation"), 15),
            (("training", "microBatch"), 2),
            (("training", "learningRate"), 0.002),
            (("training", "objective"), "ordinary-next-token-v1"),
            (("training", "resumeAllowed"), True),
            (("training", "checkpointReselectionAllowed"), True),
            (("training", "validationSteps"), [0, 100]),
            (("source", "stageCheckpointSha256"), "changed"),
            (("source", "bundleManifestSha256"), "changed"),
            (("protectedEvaluation", "developmentExcludedFromGradients"), False),
            (("baseline", "meanLoss"), 0.1),
        ]
        for path, value in changes:
            with self.subTest(path=path, value=value):
                bad = copy.deepcopy(packet)
                bad[path[0]][path[1]] = value
                with self.assertRaisesRegex(ValueError, "frozen authorization mismatch"):
                    p302.validate_authorization(bad, allow_draft=True)

    def test_extra_contract_key_rejected(self) -> None:
        value = self._contract()
        value["unsafeOverride"] = True
        with self.assertRaisesRegex(ValueError, "fields"):
            p302.validate_authorization(value, allow_draft=True)

    def test_training_needs_both_flag_and_approved_packet_before_file_reads(self) -> None:
        with patch.object(p302, "read_checkpoint", side_effect=AssertionError("checkpoint read")):
            with patch.object(p302.torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")):
                with self.assertRaisesRegex(ValueError, "authorize-first-run"):
                    p302.run_replacement_first_training(
                        output_dir=Path("nonexistent"), authorize_first_run=False)
                with self.assertRaisesRegex(ValueError, "NOT authorized"):
                    p302.run_replacement_first_training(
                        output_dir=Path("nonexistent"), authorize_first_run=True)

    def test_preflight_does_not_mutate_training_authorization(self) -> None:
        baseline = {"stageCheckpointSha256": p302.EXPECTED_STAGE_SHA256,
                    "baseWeightsPreservedExactly": True}
        bundle = {"bundleManifestSha256": p302.BUNDLE_SHA256,
                  "sampler": {"expectedAnswerAndEosTargetPositions": 60445}}
        with patch.object(p302, "_verified_packet", return_value=(baseline, bundle)):
            with patch.object(p302, "_packed_records", return_value=[
                (str(i), (5, 7), (0, 1)) for i in range(144)
            ]):
                with patch.object(p302, "_replay_positions", return_value={
                    "examples": 1600, "recordsSelected": 144,
                    "expectedAnswerAndEosTargetPositions": 60445,
                }):
                    report = p302.preflight_replacement_first_run()
        self.assertEqual(report["authorizationStatus"], "owner-approval-required")
        self.assertFalse(report["modelTrainingAuthorized"])
        self.assertFalse(report["trainingPerformed"])
        self.assertFalse(report["automaticContinuation"])

    def test_sampler_replay_fails_on_any_change(self) -> None:
        records = [(str(i), (5, 7), (0, 1)) for i in range(144)]
        with self.assertRaisesRegex(ValueError, "sampler changed"):
            p302._replay_positions(records)
        with self.assertRaisesRegex(ValueError, "144 records"):
            p302._replay_positions(records[:-1])

    def test_training_step_supervises_only_masked_targets(self) -> None:
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.seen = []
            def forward(self, input_ids, targets, *, target_weights, loss_vocabulary_size):
                self.seen.append(tuple(target_weights[0].tolist()))
                assert loss_vocabulary_size == 16384
                # Loss depends on the supervised answer only.
                term = ((self.weight - targets.float()) ** 2 * target_weights).sum()
                loss = term / target_weights.sum()
                return None, loss
        model = Tiny()
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        records = [("row", (6, 100, 7), (0, 1))]
        rng = random.Random(1337)
        from time import perf_counter
        result = p302._bounded_training_step(model, opt, records, rng, torch.device("cpu"), perf_counter() + 30)
        self.assertEqual(result[1], 16)
        self.assertEqual(len(model.seen), 16)
        self.assertTrue(all(m == (0.0, 1.0) for m in model.seen))
        self.assertNotEqual(model.weight.detach().item(), 1.0)
        self.assertTrue(opt.state)

    def test_deadline_never_creates_an_optimizer_step(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        before = [p.detach().clone() for p in model.parameters()]
        with patch.object(p302.time, "perf_counter", return_value=101.0):
            result = p302._bounded_training_step(
                model, optimizer, [("row", (5, 7), (1,))],
                random.Random(1337), torch.device("cpu"), deadline=100.0)
        self.assertIsNone(result)
        self.assertFalse(optimizer.state)
        for first, after in zip(before, model.parameters()):
            self.assertTrue(torch.equal(first, after.detach()))

    def test_cli_preflight_has_no_approval_flag_and_run_is_blocked_by_default(self) -> None:
        parser = build_parser()
        view = parser.parse_args(["file-edit-replacements-training-preflight"])
        self.assertFalse(hasattr(view, "authorize_first_run"))
        run = parser.parse_args(["file-edit-replacements-run"])
        self.assertFalse(run.authorize_first_run)
        self.assertTrue(run.stage_checkpoint.is_absolute())
        self.assertTrue(run.authorization_contract.is_absolute())
        approval = parser.parse_args(["file-edit-replacements-run", "--authorize-first-run"])
        self.assertTrue(approval.authorize_first_run)


if __name__ == "__main__":
    unittest.main()
