"""CPU tests for the M10 proposal and a tiny synthetic copying fixture."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from plex_training import plex600_m10_copy_proposal as m10


class FakeTokenizer:
    vocabulary_size = m10.TOKENIZER_VOCABULARY

    def encode(self, value):
        return [byte + 4 for byte in value.encode("ascii")]

    def decode(self, values):
        return bytes(token - 4 for token in values).decode("ascii")


class CopyProposalTests(unittest.TestCase):
    def test_default_is_plan_only_and_never_opens_sources(self):
        with patch.object(m10, "rows", side_effect=AssertionError("opened candidates")), \
             redirect_stdout(StringIO()) as output:
            self.assertEqual(m10.main([]), 0)
        data = json.loads(output.getvalue())
        self.assertFalse(data["trainingAuthorized"])
        self.assertFalse(data["runEntrypointImplemented"])
        self.assertEqual(data["optimizerUpdatesCompleted"], 0)
        self.assertEqual(data["optimizerUpdatesMaximum"], 50)
        self.assertEqual(data["evaluationAtOptimizerUpdates"], [10, 25, 50])
        self.assertEqual(data["examplesPerOptimizerUpdate"], 1)
        self.assertEqual(data["gradientAccumulationSteps"], 1)
        self.assertEqual(data["warmupRecordId"], "m10-js-counter")
        self.assertTrue(data["readOnlyBaselineAtUpdateZero"])
        self.assertFalse(data["finalHoldoutUsed"])
        self.assertFalse(data["checkpointWrites"])

    def test_candidate_source_rights_and_split_are_strict(self):
        rows = m10.rows()
        self.assertEqual(len(rows), 18)
        self.assertEqual(sum(r["split"] == "train" for r in rows), 12)
        self.assertEqual(sum(r["split"] == "heldout" for r in rows), 6)
        self.assertEqual(len({r["snippet"] for r in rows}), len(rows))
        self.assertEqual(len({r["family"] for r in rows}), len(rows))
        self.assertTrue(all(r["trainingAuthorized"] is False for r in rows))
        self.assertTrue(all(r["sourceOrigin"] == m10.ORIGIN for r in rows))
        self.assertTrue(all(r["rightsScope"] == m10.RIGHTS for r in rows))

    def test_rejects_forged_owner_permission_and_overlap(self):
        original = m10.rows()
        for field, value in (
            ("trainingAuthorized", True),
            ("rightsScope", "approved"),
            ("sourceOrigin", "owner-authored"),
        ):
            with self.subTest(field=field):
                modified = deepcopy(original)
                modified[0][field] = value
                with patch.object(m10.research, "read_jsonl", return_value=modified):
                    with self.assertRaises(m10.Refusal):
                        m10.rows()
        modified = deepcopy(original)
        modified[-1]["snippet"] = modified[0]["snippet"]
        with patch.object(m10.research, "read_jsonl", return_value=modified):
            with self.assertRaisesRegex(m10.Refusal, "unique"):
                m10.rows()

    def test_tokenizer_packing_and_exact_round_trip_no_training(self):
        result = m10.audit(FakeTokenizer())
        self.assertTrue(result["tokenizerRoundTripVerified"])
        self.assertLessEqual(result["maxObservedInputPositions"], 256)
        self.assertLessEqual(result["maxObservedAnswerTokensIncludingEos"], 96)
        self.assertGreater(result["totalTrainCopyAndEosSupervisedPositions"], 0)
        self.assertFalse(result["generatedCodeEvaluated"])
        self.assertFalse(result["trainingAuthorized"])

    def test_tokenizer_refusal_preserves_proposal_only_mode(self):
        class WrongVocabulary(FakeTokenizer):
            vocabulary_size = 32768

        with self.assertRaisesRegex(m10.Refusal, "1191"):
            m10.audit(WrongVocabulary())

    def test_owner_approval_template_is_unapproved_and_has_exact_limits(self):
        proposal = m10.proposal_approval_template()
        self.assertIsNone(proposal["approvedBy"])
        self.assertIsNone(proposal["approvedAt"])
        self.assertFalse(proposal["approvedForTraining"])
        self.assertFalse(proposal["sourceReviewedByOwner"])
        self.assertEqual(proposal["evaluationUpdates"], [10, 25, 50])
        self.assertEqual(proposal["maxOptimizerUpdates"], 50)
        self.assertEqual(proposal["examplesPerOptimizerUpdate"], 1)
        self.assertEqual(proposal["phaseUpdateCounts"], [10, 15, 25])
        self.assertEqual(proposal["warmupRecordId"], "m10-js-counter")
        self.assertEqual(proposal["maxNewTokens"], 96)
        self.assertEqual(proposal["maxInputPositions"], 256)
        self.assertTrue(proposal["m9CheckpointSha256"].startswith("OWNER_"))
        self.assertEqual(len(proposal["candidateSha256"]), 64)

    def test_scoring_requires_autonomous_exact_text_plus_eos(self):
        examples = m10.rows()
        perfect = [
            {"id": r["id"], "output": r["snippet"], "terminationReason": "eos"}
            for r in examples
        ]
        result = m10.score_predictions(examples, perfect)
        self.assertTrue(result["allExactAndEos"])
        self.assertEqual(result["train"]["exactWithEos"], 12)
        self.assertEqual(result["heldout"]["exactWithEos"], 6)
        self.assertFalse(result["generalizationClaimEstablished"])
        self.assertEqual(result["train"]["meanCharacterSimilarity"], 1)
        self.assertEqual(result["heldout"]["meanCorrectPrefixFraction"], 1)
        self.assertEqual(result["train"]["repetitionCollapse"], 0)
        self.assertEqual(result["uniqueGeneratedOutputs"], 18)
        incorrect = deepcopy(perfect)
        incorrect[-1]["output"] = " " * 96
        result = m10.score_predictions(examples, incorrect)
        self.assertFalse(result["allExactAndEos"])
        self.assertEqual(result["heldout"]["exactWithEos"], 5)
        self.assertEqual(result["heldout"]["spacesOnly"], 1)
        self.assertEqual(result["heldout"]["repetitionCollapse"], 1)
        self.assertLess(result["heldout"]["meanCharacterSimilarity"], 1)
        incorrect = deepcopy(perfect)
        incorrect[0]["terminationReason"] = "max-new-tokens"
        self.assertFalse(m10.score_predictions(examples, incorrect)["allExactAndEos"])
        incorrect = deepcopy(perfect)
        incorrect[0]["id"] = incorrect[1]["id"]
        with self.assertRaises(m10.Refusal):
            m10.score_predictions(examples, incorrect)

    def test_milestone_scoring_disallows_duplicate_or_cross_policy_early_stop(self):
        examples = m10.rows()
        perfect = [
            {"id": row["id"], "output": row["snippet"], "terminationReason": "eos"}
            for row in examples
        ]
        baseline = m10.milestone_score(0, "unmasked-v1", perfect)
        self.assertEqual(baseline["optimizerUpdatesCompleted"], 0)
        previous = m10.milestone_score(10, "unmasked-v1", perfect)
        current = m10.milestone_score(25, "unmasked-v1", perfect)
        final = m10.milestone_score(50, "unmasked-v1", perfect)
        fitted = m10.milestone_score(25, "fitted-vocab-v1", perfect)
        self.assertTrue(m10.eligible_to_stop_early(previous, current))
        self.assertTrue(m10.eligible_to_stop_early(current, final))
        self.assertFalse(m10.eligible_to_stop_early(baseline, previous))
        self.assertFalse(m10.eligible_to_stop_early(previous, previous))
        self.assertFalse(m10.eligible_to_stop_early(previous, final))
        self.assertFalse(m10.eligible_to_stop_early(previous, fitted))
        mutated = {**current, "candidateSha256": "0" * 64}
        self.assertFalse(m10.eligible_to_stop_early(previous, mutated))
        mutated = {**current, "allExactAndEos": False}
        self.assertFalse(m10.eligible_to_stop_early(previous, mutated))
        mutated = {**current, "finalHoldoutUsed": True}
        self.assertFalse(m10.eligible_to_stop_early(previous, mutated))
        mutated = {**current, "source": "unverified-manual-result"}
        self.assertFalse(m10.eligible_to_stop_early(previous, mutated))
        with self.assertRaises(m10.Refusal):
            m10.milestone_score(11, "unmasked-v1", perfect)
        with self.assertRaises(m10.Refusal):
            m10.milestone_score(25, "other-decoder", perfect)

    def test_explicit_schedule_has_single_example_warmup_and_40_full_set_steps(self):
        schedule = m10.update_schedule()
        self.assertEqual(len(schedule), 50)
        self.assertEqual([s["optimizerUpdate"] for s in schedule], list(range(1, 51)))
        self.assertTrue(all(s["examplesInOptimizerUpdate"] == 1 for s in schedule))
        self.assertTrue(all(s["gradientAccumulationSteps"] == 1 for s in schedule))
        self.assertEqual({s["recordIds"][0] for s in schedule[:10]}, {"m10-js-counter"})
        self.assertEqual({s["recordIds"][0] for s in schedule[10:]},
                         {r["id"] for r in m10.rows() if r["split"] == "train"})
        self.assertFalse({s["recordIds"][0] for s in schedule}
                         & {r["id"] for r in m10.rows() if r["split"] == "heldout"})
        self.assertEqual([s["phase"] for s in schedule].count("single-example-copy"), 10)
        self.assertEqual([s["phase"] for s in schedule].count("full-set-copy"), 15)
        self.assertEqual([s["phase"] for s in schedule].count("full-set-consolidation"), 25)
        counts = m10.schedule_audit()["trainRecordExposures"]
        self.assertEqual(sum(counts.values()), 50)
        self.assertEqual(counts["m10-js-counter"], 13)
        self.assertEqual(len(counts), 12)
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(m10.main(["--show-schedule"]), 0)
        self.assertEqual(len(json.loads(output.getvalue())["schedule"]), 50)

    def test_partial_copy_metrics_handle_repeated_spaces_periods_and_shift(self):
        self.assertEqual(m10.character_similarity("abc" + chr(10), "abc" + chr(10)), 1)
        self.assertEqual(m10.character_similarity("abc" + chr(10), "abc"), 0.75)
        self.assertEqual(m10.character_similarity("abc", ""), 0)
        self.assertGreater(m10.character_similarity("abc", "xbc"), 0.6)
        spaces = m10.repetition_diagnostics(" " * 96)
        self.assertTrue(spaces["spaceOnly"])
        self.assertTrue(spaces["whitespaceOnly"])
        self.assertTrue(spaces["repetitionCollapse"])
        periods = m10.repetition_diagnostics("." * 96)
        self.assertTrue(periods["periodOnly"])
        self.assertTrue(periods["repetitionCollapse"])
        self.assertFalse(m10.repetition_diagnostics("abc abc")["repetitionCollapse"])
        with self.assertRaises(m10.Refusal):
            m10.character_similarity("valid", " " * 4097)
        with self.assertRaises(m10.Refusal):
            m10.repetition_diagnostics("." * 4097)

    def test_mechanical_evidence_is_a_diagnostic_not_training_proof(self):
        evidence = {
            "optimizerUpdatesCompleted": 10,
            "examplesPerOptimizerUpdate": 1,
            "promptPositionsIncorrectlySupervised": 0,
            "answerAndEosPositionsSupervised": 100,
            "eosTargetSupervised": True,
            "gradientL2Norm": 0.25,
            "weightDeltaL2Norm": 0.005,
            "originalCheckpointSha256": "a" * 64,
            "newCheckpointSha256": "b" * 64,
            "newCheckpointHashVerified": True,
            "reloadedCheckpointSha256": "b" * 64,
            "reloadedGreedyTokensMatch": True,
            "maxReloadLogitAbsoluteDifference": 0.0,
        }
        proof = m10.classify_mechanical_evidence(evidence)
        self.assertEqual(proof["status"], "mechanically-consistent-caller-claims")
        self.assertFalse(proof["evidenceVerifiedByThisModule"])
        self.assertFalse(proof["trainingAuthorized"])
        self.assertEqual(proof["failedChecks"], [])
        for key, value, expected in (
            ("promptPositionsIncorrectlySupervised", 1, "promptMaskCorrect"),
            ("gradientL2Norm", 0, "gradientNonzero"),
            ("weightDeltaL2Norm", 0, "weightsChanged"),
            ("reloadedGreedyTokensMatch", False, "reloadGenerationTokensMatch"),
            ("maxReloadLogitAbsoluteDifference", 0.1, "reloadLogitsWithinTolerance"),
            ("newCheckpointHashVerified", False, "checkpointBytesVerified"),
        ):
            with self.subTest(key=key):
                broken = m10.classify_mechanical_evidence({**evidence, key: value})
                self.assertIn(expected, broken["failedChecks"])
                self.assertFalse(broken["evidenceVerifiedByThisModule"])
        for broken in ({**evidence, "optimizerUpdatesCompleted": 60},
                       {**evidence, "gradientL2Norm": float("nan")},
                       {**evidence, "extraClaim": True}):
            with self.assertRaises(m10.Refusal):
                m10.classify_mechanical_evidence(broken)

    def test_single_copy_update_and_reload_on_cpu(self):
        import torch
        from torch.nn import functional as F
        from plex_training.plex600 import Plex600Config, Plex600Model
        from plex_training.plex600_r3_curriculum import masked_record

        # Synthetic IDs only. Exercise the Plex600 forward path on a tiny CPU model.
        prompt, answer = [4, 7, 8, 9], [7, 8, 3]
        config = Plex600Config(vocab_size=16, context_length=12, width=16,
                               layers=1, attention_heads=2, key_value_heads=1,
                               feed_forward_width=32)
        packed = masked_record(prompt, answer, fitted_vocab_size=config.vocab_size)
        inputs = torch.tensor([packed["inputIds"]])
        targets = torch.tensor([packed["targetIds"]])
        start = len(prompt) - 1
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(41)
            model = Plex600Model(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0)
            logits, loss = model(inputs, targets)
            expected_loss = F.cross_entropy(logits[0, start:], torch.tensor(answer))
            torch.testing.assert_close(loss, expected_loss)
            logits.retain_grad()
            loss.backward()
            self.assertEqual(logits.grad[0, :start].count_nonzero().item(), 0)
            self.assertTrue(bool((logits.grad[0, start:].abs().sum(-1) > 0).all()))
            weight = model.token_embedding.weight
            self.assertTrue(bool(torch.isfinite(weight.grad).all()))
            self.assertGreater(weight.grad.norm().item(), 0)
            before = weight.detach().clone()
            optimizer.step()
            self.assertTrue(bool(torch.isfinite(weight).all()))
            self.assertFalse(torch.equal(before, weight))

            with tempfile.TemporaryDirectory() as folder, torch.no_grad():
                checkpoint = Path(folder) / "fixture.pt"
                model.eval()
                expected, _ = model(inputs)
                torch.save(model.state_dict(), checkpoint)
                restored = Plex600Model(config).eval()
                restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
                actual, _ = restored(inputs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(actual.argmax(-1), expected.argmax(-1)))

    def test_copy_mask_positions_match_answer_and_eos(self):
        from plex_training import plex600_r3_curriculum as r3
        tokenizer = FakeTokenizer()
        exemplar = next(r for r in m10.rows() if r["id"] == m10.WARMUP_RECORD_ID)
        prompt_ids = tokenizer.encode(m10.prompt_for(exemplar))
        answer_ids = tokenizer.encode(exemplar["snippet"]) + [3]
        packed = r3.masked_record(
            prompt_ids, answer_ids, fitted_vocab_size=tokenizer.vocabulary_size)
        self.assertEqual(packed["targetIds"][:len(prompt_ids) - 1],
                         [-100] * (len(prompt_ids) - 1))
        self.assertEqual(packed["targetIds"][len(prompt_ids) - 1:], answer_ids)
        self.assertEqual(packed["inputIds"], prompt_ids + answer_ids[:-1])
        self.assertEqual(packed["targetIds"][-1], 3)
        self.assertEqual(packed["supervisedAnswerPositions"], len(answer_ids))

    def test_read_only_m9_backup_verifies_bytes_and_preserves_history(self):
        from plex_training import plex600_r3_curriculum as r3
        from plex_training import plex600_m7 as m7
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            weights = folder / "weights.pt"
            report = folder / "report.json"
            weights.write_bytes(b"small-synthetic-test-only-not-a-real-checkpoint")
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            data = {
                "protocol": "plex600-m9-r3-eight-update-pilot-v1",
                "status": "eight-updates-complete-unreviewed-improvement",
                "optimizerUpdatesCompleted": 8,
                "trainingAuthorized": True,
                "modelLoaded": True,
                "finalHoldoutUsed": False,
                "m8DevelopmentUsedForTraining": False,
                "codingImprovementProven": False,
                "generatedCodeEvaluated": False,
                "originalCheckpointSha256":
                    "69a31b4e710b284581a0b74c9e2661cc27bf134d24b8b1fe7a906f6efc004224",
                "originalTokenizerSha256": m7.TOKENIZER_SHA,
                "bundleManifestSha256": m10.research.BUNDLE_SHA,
                "r3CandidateSha256": r3.source_sha256(),
                "approval": {"approved": True, "approvalSha256": "a" * 64},
                "steps": [{"step": i} for i in range(1, 9)],
                "checkpoint": {
                    "file": "weights.pt", "bytes": weights.stat().st_size,
                    "sha256": digest, "weightsOnly": True,
                    "optimizerStateSaved": False,
                },
                # The historical report has this known misleading metadata.
                "checkpointWrites": False,
            }
            report.write_text(json.dumps(data), encoding="utf-8")
            proof = m10.audit_m9_backup(report, weights)
            self.assertEqual(proof["m9WeightsSha256"], digest)
            self.assertTrue(proof["checkpointBytesAndHashVerified"])
            self.assertFalse(proof["checkpointWritesReportMetadata"])
            self.assertFalse(proof["trainingAuthorized"])
            self.assertFalse(proof["checkpointWrites"])

            weights.write_bytes(weights.read_bytes() + b"tampered")
            with self.assertRaisesRegex(m10.Refusal, "byte size"):
                m10.audit_m9_backup(report, weights)
            weights.write_bytes(b"small-synthetic-test-only-not-a-real-checkpoint")
            bad = {**data, "optimizerUpdatesCompleted": 9}
            report.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(m10.Refusal, "scope invalid"):
                m10.audit_m9_backup(report, weights)
            report.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(m10.research, "ROOT", folder):
                # A source directory for an original M9 experiment cannot
                # substitute for the independent backed-up checkpoint.
                original = folder / "training/artifacts/plex600-m9-r3-first-eight"
                original.mkdir(parents=True)
                original_weights = original / "weights.pt"
                original_report = original / "report.json"
                original_weights.write_bytes(weights.read_bytes())
                original_report.write_bytes(report.read_bytes())
                with self.assertRaisesRegex(m10.Refusal, "separate M9 backup"):
                    m10.audit_m9_backup(original_report, original_weights)

    def test_cli_m9_backup_mode_requires_both_paths_and_never_trains(self):
        with redirect_stderr(StringIO()) as err:
            self.assertEqual(m10.main(["--audit-m9-backup"]), 2)
        self.assertIn("needs only", err.getvalue())
        with redirect_stderr(StringIO()) as err:
            self.assertEqual(m10.main(["--m9-report", "report.json"]), 2)
        self.assertIn("require --audit-m9-backup", err.getvalue())

    def test_cli_does_not_offer_a_training_entrypoint(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as refused:
                m10.main(["--execute-approved-run"])
        self.assertEqual(refused.exception.code, 2)
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(m10.main(["--show-candidate"]), 0)
        data = json.loads(output.getvalue())
        self.assertEqual(len(data["rows"]), 18)
        self.assertFalse(data["plan"]["trainingAuthorized"])


if __name__ == "__main__":
    unittest.main()

