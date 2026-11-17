"""CPU-only M10 review contract; no model creation, optimizer or GPU."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
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

    def test_early_stop_requires_two_consecutive_complete_evaluations(self):
        success = {"allExactAndEos": True, "finalHoldoutUsed": False}
        failure = {"allExactAndEos": False, "finalHoldoutUsed": False}
        self.assertFalse(m10.eligible_to_stop_early(success, failure))
        self.assertFalse(m10.eligible_to_stop_early(failure, success))
        self.assertTrue(m10.eligible_to_stop_early(success, success))
        self.assertFalse(m10.eligible_to_stop_early(
            success, {"allExactAndEos": True, "finalHoldoutUsed": True}))

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

