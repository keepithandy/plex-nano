"""CPU-only M10 review contract; no model creation, optimizer or GPU."""
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
        incorrect = deepcopy(perfect)
        incorrect[-1]["output"] = " " * 96
        result = m10.score_predictions(examples, incorrect)
        self.assertFalse(result["allExactAndEos"])
        self.assertEqual(result["heldout"]["exactWithEos"], 5)
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
        previous = m10.milestone_score(10, "unmasked-v1", perfect)
        current = m10.milestone_score(25, "unmasked-v1", perfect)
        final = m10.milestone_score(50, "unmasked-v1", perfect)
        fitted = m10.milestone_score(25, "fitted-vocab-v1", perfect)
        self.assertTrue(m10.eligible_to_stop_early(previous, current))
        self.assertTrue(m10.eligible_to_stop_early(current, final))
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

