"""CPU-only M11 proposal tests. No model, checkpoint or optimizer."""
from contextlib import redirect_stdout
from io import StringIO
import json
import unittest
from unittest.mock import patch

from plex_training import plex600_m11_source_conditioning as m11


class Tokenizer:
    vocabulary_size = 1191

    @staticmethod
    def encode(value):
        return list(value.encode("ascii"))

    @staticmethod
    def decode(ids):
        return bytes(ids).decode("ascii")


class M11SourceConditioningTests(unittest.TestCase):
    def test_default_is_read_only_and_does_not_audit(self):
        with patch.object(m11, "audit_tokenizer", side_effect=AssertionError), redirect_stdout(StringIO()) as out:
            m11.main([])
        result = json.loads(out.getvalue())
        self.assertFalse(result["sourceApprovedForTraining"])
        self.assertFalse(result["modelLoaded"])
        self.assertFalse(result["executionEntrypointImplemented"])
        self.assertEqual(result["optimizerUpdatesCompleted"], 0)
        self.assertEqual(result["probeOnlyTrainingExposures"], 0)

    def test_source_pairs_and_space_boundary_audit(self):
        rows = m11.audit_tokenizer(Tokenizer())["tokenizerAudit"]
        self.assertEqual([r["split"] for r in rows], ["train", "probe-only"])
        self.assertEqual(len({key for r in rows for key in r["ids"]}), 4)
        for row in rows:
            self.assertEqual(row["contrastTokenIds"], [ord("0"), ord("1")])
        for item in m11.cases():
            prompt = m11.prompt_for(item)
            self.assertTrue(prompt.endswith("Copy: "))
            self.assertTrue(item["snippet"].endswith(";\n"))
            self.assertNotEqual(prompt, m11.m10.prompt_for(item))
            self.assertFalse(item["trainingAuthorized"])

    def test_reject_unfrozen_and_nonpaired_tokenizer(self):
        bad = Tokenizer()
        bad.vocabulary_size = 100
        with self.assertRaisesRegex(m11.Refusal, "vocabulary"):
            m11.audit_tokenizer(bad)
        with self.assertRaisesRegex(m11.Refusal, "one token"):
            m11.one_change([1, 2], [1, 3, 4])

    def test_caller_supplied_perfect_responses_are_not_verified(self):
        rows = [
            {"id": x["id"], "output": x["snippet"], "terminationReason": "eos"}
            for x in m11.cases()
        ]
        result = m11.score_caller_supplied(rows)
        self.assertTrue(result["completeOutputGate"])
        self.assertTrue(result["sourcePreferenceGateNotEvaluated"])
        self.assertIn("unverified", result["evidenceSource"])

    def test_constant_output_or_missing_eos_is_not_a_pass(self):
        rows = [
            {"id": x["id"], "output": "let score = 0;\n", "terminationReason": "eos"}
            for x in m11.cases()
        ]
        self.assertFalse(m11.score_caller_supplied(rows)["completeOutputGate"])
        rows = [
            {"id": x["id"], "output": x["snippet"], "terminationReason": "length-cap"}
            for x in m11.cases()
        ]
        self.assertFalse(m11.score_caller_supplied(rows)["trainPairExactAndEos"])
        rows[1]["id"] = rows[0]["id"]
        with self.assertRaisesRegex(m11.Refusal, "duplicated"):
            m11.score_caller_supplied(rows)

    def test_preference_needs_both_directions_and_actual_greedy_winners(self):
        z, o = ord("0"), ord("1")
        self.assertTrue(m11.paired_preference(-2.0, 3.0, z, o, z, o))
        self.assertFalse(m11.paired_preference(-2.0, 3.0, z, z, z, o))
        self.assertFalse(m11.paired_preference(-2.0, -1.0, z, o, z, o))
        with self.assertRaisesRegex(m11.Refusal, "Non-finite"):
            m11.paired_preference(float("nan"), 1, z, o, z, o)

    def test_no_execution_flag_exists(self):
        with redirect_stdout(StringIO()), self.assertRaises(SystemExit) as ex:
            m11.main(["--execute-approved-run"])
        self.assertEqual(ex.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
