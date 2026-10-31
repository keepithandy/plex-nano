"""M3 six-source rights review stays read-only, reproducible and unapproved.

This checks committed evidence against historical pinned receipts without
opening any corpus, fetching sources, fitting tokenizer or training model.
"""
import unittest

from plex_training import plex600_m3_scope as m3
from plex_training import plex600_m4 as m4


class SixSourceRightsEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = m3.read_json("plex600-m3-six-source-rights-evidence-2026-10-31.json")
        cls.review = m3.read_json(*m3.REVIEW)
        cls.holds = m3.read_json(*m3.HOLDS)

    def test_review_must_not_be_an_approval_or_materialization(self):
        r = self.report
        self.assertEqual(r["kind"], "plex600-m3-six-source-rights-evidence-v1")
        self.assertEqual(r["status"], "review-only-unapproved")
        self.assertEqual(r["approvedSources"], [])
        for field in ("sourceRightsApproved", "m4PreparationApproved",
                      "trainingAuthorized", "independentDevelopmentEvaluationCreated",
                      "finalHoldoutOpened", "newCorpusPrepared", "tokenizerFitted",
                      "fileLevelRightsAndThirdPartyProvenanceComplete",
                      "upstreamCommitTreeAndRawNoticeByteReauthenticationCompleted"):
            self.assertIs(r[field], False, field)
        self.assertEqual(r["optimizerUpdatesPerformedByThisAudit"], 0)
        self.assertEqual(r["upstreamLicenseTextCheckedAtPinnedRevision"], 6)
        self.assertEqual(r["sourceReviewReceiptSha256"], m3.REVIEW[1])
        self.assertEqual(r["quarantineReceiptSha256"], m3.HOLDS[1])
        for source in r["sources"]:
            self.assertEqual(source["rightsDisposition"], "hold-needs-scoped-review")
            self.assertIs(source["approvedForM4"], False)
            self.assertIs(source["rootLicenseBytesIndependentlyMatchedToHistoricalNotice"], False)

    def test_each_count_and_source_identity_matches_pinned_m3_receipts(self):
        # The M3 scope verification checks the same historical split/role
        # assignments plus all 69 omissions and 101 notice records.
        inspected = m3.inspect()
        historical = {s["id"]: s for c in self.review["corpora"] for s in c["sources"]}
        omissions = {}
        notices = {}
        for key, result in (("proposedOmissions", omissions), ("noticeTriageRecords", notices)):
            for item in self.holds[key]:
                result.setdefault(item["sourceId"], set()).add(item["path"])
        expected_names = {name for name, (_, split) in m3.ROLES.items() if split == "train"}
        got_names = [s["sourceId"] for s in self.report["sources"]]
        self.assertEqual(set(got_names), expected_names)
        self.assertEqual(len(got_names), len(expected_names))
        total = held_total = 0
        for item in self.report["sources"]:
            name = item["sourceId"]
            previous = historical[name]
            group, split = m3.ROLES[name]
            self.assertEqual(split, "train")
            self.assertEqual(item["sourceGroup"], group)
            self.assertEqual(item["pinnedRevision"], previous["revision"])
            self.assertEqual(item["declaredRootLicense"], previous["recordedLicenseId"])
            self.assertEqual(item["historicalTrainRecords"], len(previous["acceptedPaths"]))
            omission = omissions.get(name, set())
            flagged = notices.get(name, set())
            self.assertEqual(item["proposedEmbeddedLibraryOmissions"], len(omission))
            self.assertEqual(item["noticeKeywordFlags"], len(flagged))
            self.assertEqual(item["quarantineIntersection"], len(omission & flagged))
            self.assertEqual(item["unionQuarantined"], len(omission | flagged))
            self.assertEqual(item["conditionalCandidateRecords"], len(previous["acceptedPaths"]) - len(omission | flagged))
            total += item["conditionalCandidateRecords"]
            held_total += item["unionQuarantined"]
        self.assertEqual(total, inspected["candidateTrainRecords"])
        self.assertEqual(held_total, inspected["trainQuarantined"])
        summary = self.report["quarantineCounts"]
        self.assertEqual(summary["conditionalTrainingRecordsBeforeQualityFilters"], total)
        self.assertEqual(summary["unionQuarantinedTraining"], held_total)
        self.assertEqual(summary["totalQuarantinedAcrossRoles"], inspected["quarantinedRecords"])
        self.assertEqual(summary["historicalValidationRecordsExcluded"], 123)

    def test_report_cannot_substitute_for_m4_rights_decision(self):
        # M4 must continue rejecting evidence receipts as authority to read
        # real historical training text.
        with self.assertRaisesRegex(ValueError, "Missing or mismatched"):
            m4.reviewed_decision(self.report, m3.REVIEW[1], m3.HOLDS[1], "0" * 64)


if __name__ == "__main__":
    unittest.main()
