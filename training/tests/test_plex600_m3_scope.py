"""Metadata-only regression tests: no model, optimizer, or dataset materialization."""
import copy
import unittest

from plex_training.plex600_m3_scope import HOLDS, REVIEW, inspect, read_json, verify


class Plex600M3ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = read_json(*REVIEW)
        cls.holds = read_json(*HOLDS)
        cls.plan = read_json("plex600-m3-candidate-scope-v1.json")

    def test_review_candidate_counts_and_no_training(self):
        got = inspect()
        self.assertEqual(got["historicalRecords"], 3228)
        self.assertEqual(got["quarantinedRecords"], 145)
        self.assertEqual(got["trainQuarantined"], 123)
        self.assertEqual(got["candidateTrainRecords"], 2982)
        self.assertFalse(got["rawDataRead"])
        self.assertFalse(got["newCorpusPrepared"])
        self.assertFalse(got["finalHoldoutOpened"])
        self.assertEqual(got["optimizerUpdates"], 0)

    def test_new_or_misassigned_hold_is_rejected(self):
        holds = copy.deepcopy(self.holds)
        holds["proposedOmissions"][0]["path"] = "unknown.js"
        with self.assertRaisesRegex(ValueError, "outside reviewed"):
            verify(self.review, holds, self.plan)

    def test_overlap_identity_conflict_is_rejected(self):
        holds = copy.deepcopy(self.holds)
        keys = {(r["sourceId"], r["path"]) for r in holds["proposedOmissions"]}
        both = next(r for r in holds["noticeTriageRecords"]
                    if (r["sourceId"], r["path"]) in keys)
        both["contentSha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            verify(self.review, holds, self.plan)

    def test_historical_validation_cannot_become_training(self):
        plan = copy.deepcopy(self.plan)
        plan["sourceRoles"]["bulma-templates"] = "candidate-train"
        with self.assertRaisesRegex(ValueError, "Historical split"):
            verify(self.review, self.holds, plan)

    def test_training_authorization_cannot_be_sneaked_in(self):
        plan = copy.deepcopy(self.plan)
        plan["trainingAuthorized"] = True
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            verify(self.review, self.holds, plan)

    def test_all_notice_flagged_records_must_be_quarantined(self):
        plan = copy.deepcopy(self.plan)
        plan["quarantine"] = "only-proposed-omissions"
        with self.assertRaisesRegex(ValueError, "fail-closed"):
            verify(self.review, self.holds, plan)

    def test_strict_preparation_target(self):
        plan = copy.deepcopy(self.plan)
        plan["maxCandidateRecords"] = 100
        with self.assertRaisesRegex(ValueError, "exceeds preparation"):
            verify(self.review, self.holds, plan)


if __name__ == "__main__":
    unittest.main()
