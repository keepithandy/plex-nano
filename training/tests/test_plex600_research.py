"""Plex 600M research tests are read-only, CPU-only, no model instantiation."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from plex_training import plex600_research as research


class Plex600ResearchTests(unittest.TestCase):
    def test_separate_unapproved_1152_record_candidate_audit(self):
        receipt = research.candidate_audit()
        self.assertEqual(receipt["recordCount"], 1152)
        self.assertEqual(receipt["approvedTrainingRows"], 0)
        self.assertFalse(receipt["trainingAuthorized"])
        self.assertEqual(receipt["byLanguage"], {"html": 384, "css": 384, "javascript": 384})
        self.assertEqual(receipt["templateFamilies"], 18)
        self.assertEqual(receipt["variationsPerTemplateFamily"], 64)
        self.assertEqual(receipt["exactBenchmarkStringOverlaps"], 0)
        self.assertEqual(receipt["evaluationRows"], 36)
        self.assertEqual(len(receipt["candidateJsonlSha256"]), 64)

    def test_benchmark_is_36_hand_authored_distinct_dev_only_cases(self):
        rows = research.benchmark()
        self.assertEqual(len(rows), 36)
        self.assertEqual(len({r["id"] for r in rows}), 36)
        self.assertEqual(len({r["family"] for r in rows}), 36)
        self.assertTrue(all(r["trainingAuthorized"] is False for r in rows))
        self.assertTrue(all(research.exact_patch(r, r["inputFile"], r["expectedFile"])
                            for r in rows))

    def test_all_expected_outputs_score_perfect_and_only_eval(self):
        rows = research.benchmark()
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "responses.jsonl"
            path.write_text("".join(json.dumps({"id": r["id"], "output": r["expectedFile"]})
                                    + "\n" for r in rows), encoding="utf-8", newline="\n")
            result = research.grade(path)
        self.assertEqual(result["metrics"],
                         {"scored": 36, "exactFile": 36, "targetPresent": 36,
                          "untouchedContextPreserved": 36, "unscored": 0, "total": 36})
        self.assertTrue(result["notIndependentFinalHoldout"])
        self.assertFalse(result["trainingAuthorized"])
        self.assertFalse(result["behavioralTestsRun"])

    def test_missing_responses_remain_unscored_not_passed(self):
        with tempfile.TemporaryDirectory() as t:
            r = research.benchmark()[0]
            path = Path(t) / "partial.jsonl"
            path.write_text(json.dumps({"id": r["id"], "output": "wrong"}) + "\n", encoding="utf-8")
            outcome = research.grade(path)
        self.assertEqual(outcome["metrics"]["scored"], 1)
        self.assertEqual(outcome["metrics"]["exactFile"], 0)
        self.assertEqual(outcome["metrics"]["unscored"], 35)

    def test_target_edit_without_preservation_is_not_full_credit(self):
        with tempfile.TemporaryDirectory() as t:
            r = research.benchmark()[0]
            path = Path(t) / "partial.jsonl"
            path.write_text(json.dumps({"id": r["id"], "output": r["newText"]}) + "\n", encoding="utf-8")
            result = research.grade(path)
        self.assertEqual(result["metrics"]["targetPresent"], 1)
        self.assertEqual(result["metrics"]["exactFile"], 0)
        self.assertEqual(result["metrics"]["untouchedContextPreserved"], 0)

    def test_duplicate_or_unknown_response_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            r = research.benchmark()[0]
            path = Path(t) / "duplicate.jsonl"
            row = json.dumps({"id": r["id"], "output": r["expectedFile"]}) + "\n"
            path.write_text(row + row, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                research.grade(path)
            path.write_text(json.dumps({"id": "unknown", "output": "anything"}) + "\n")
            with self.assertRaisesRegex(ValueError, "unknown"):
                research.grade(path)

    def test_missing_candidate_or_unapproved_flags_cannot_be_accepted(self):
        candidates = research.read_jsonl(research.CANDIDATES)
        bad = dict(candidates[0], rightsReviewStatus="approved", trainingAuthorized=True)
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "bad.jsonl"
            path.write_text(json.dumps(bad) + "\n")
            with patch.object(research, "CANDIDATES", path), \
                 self.assertRaisesRegex(ValueError, "1152|Unapproved"):
                research.candidate_audit()

    def test_benchmark_can_never_be_used_as_training_candidate(self):
        rows = research.read_jsonl(research.CANDIDATES)
        self.assertTrue(all(r["evaluationOnly"] is False for r in rows))
        self.assertTrue(all(r["trainingAuthorized"] is False for r in rows))
        self.assertTrue(all(r["rightsReviewStatus"] == "pending-owner-review" for r in rows))
        self.assertTrue(all(r["candidateStatus"] == "quarantined-pending-qa-and-approval"
                            for r in rows))

    def test_owner_report_facts_are_verbatim_but_not_independent_checkpoint_proof(self):
        evidence = json.loads(research.M7_EVIDENCE.read_text(encoding="utf-8"))
        self.assertEqual(evidence["optimizerUpdatesCompleted"], 4)
        self.assertEqual(evidence["supervisedPositionsAcrossUpdates"], 220)
        self.assertEqual(evidence["sourceManifestSha256"], research.SOURCE_SHA)
        self.assertEqual(evidence["bundleManifestSha256"], research.BUNDLE_SHA)
        self.assertIsNone(evidence["checkpoint"]["sha256"])
        self.assertFalse(evidence["checkpoint"]["independentlyVerified"])
        self.assertFalse(evidence["assurance"]["newTrainingAuthorizationGranted"])
        self.assertEqual(sum(s["supervisedTokens"] for s in evidence["steps"]), 220)

    def test_local_receipt_check_does_not_trust_wrong_identity_or_hash(self):
        report = {
            "protocol": "plex600-m7-four-update-v1",
            "status": "pass", "optimizerUpdatesCompleted": 4, "trainingAuthorized": True,
            "source": {"sourceManifestSha256": research.SOURCE_SHA,
                       "bundleManifestSha256": research.BUNDLE_SHA},
            "steps": [{"step": i, "recordId": f"record-{i}", "lossBeforeUpdate": 7.0,
                       "supervisedTokens": 50} for i in range(1, 5)],
            "validationBefore": {"loss": 10.0, "records": 24, "targetTokens": 1744},
            "validationAfter": {"loss": 9.0, "records": 24, "targetTokens": 1744},
            "checkpoint": {"file": "weights.pt", "weightsOnly": True,
                           "bytes": 13, "sha256": hashlib.sha256(b"fake pt bytes").hexdigest()},
        }
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t)
            path = folder / "report.json"
            weights = folder / "weights.pt"
            weights.write_bytes(b"fake pt bytes")
            report["checkpoint"]["bytes"] = weights.stat().st_size
            path.write_text(json.dumps(report), encoding="utf-8")
            result = research.m7_local_receipt(path)
            self.assertIsNone(result["checkpointReadAndHashVerified"])
            result = research.m7_local_receipt(path, weights)
            self.assertTrue(result["checkpointReadAndHashVerified"])
            report["source"]["sourceManifestSha256"] = "0" * 64
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity"):
                research.m7_local_receipt(path)
            report["source"]["sourceManifestSha256"] = research.SOURCE_SHA
            path.write_text(json.dumps(report), encoding="utf-8")
            weights.write_bytes(b"changed bytes")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                research.m7_local_receipt(path, weights)

    def test_nontraining_cli_paths(self):
        with patch.object(research, "m7_local_receipt",
                          side_effect=AssertionError("M7 report opened")), \
             redirect_stdout(StringIO()) as out:
            self.assertEqual(research.main(["candidate-audit"]), 0)
        self.assertEqual(json.loads(out.getvalue())["approvedTrainingRows"], 0)
        with redirect_stdout(StringIO()) as out:
            self.assertEqual(research.main(["benchmark-inspect"]), 0)
        self.assertEqual(json.loads(out.getvalue())["rows"], 36)
        with redirect_stderr(StringIO()) as err:
            self.assertEqual(research.main(["m7-receipt", "--report", "nonexistent"]), 2)
        self.assertIn("refused", err.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
