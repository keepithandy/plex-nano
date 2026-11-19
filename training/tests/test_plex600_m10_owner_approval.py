"""M10 owner approval is external, explicit, create-only and CPU-only."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from plex_training import plex600_m10_owner_approval as owner
from plex_training import plex600_m10_admission as gate
from plex_training import plex600_m10_copy_proposal as m10
from plex_training import plex600_research as research


def fake_inspected():
    source = {"m9WeightsSha256": "a"*64, "m9ReportSha256": "b"*64}
    baseline = {"baselineReportSha256": "c"*64,
                "baselineResponsesSha256": "d"*64}
    return {"backup": source, "baseline": baseline,
            "unapprovedTemplate": gate.approval_template(source, baseline)}


def args(**fields):
    values = {
        "bundle_dir": Path("bundle"), "bundle_manifest_sha256": research.BUNDLE_SHA,
        "m9_report": Path("m9-report"), "m9_weights": Path("m9-weights"),
        "baseline_report": Path("baseline-report"),
        "baseline_responses": Path("responses"),
        "confirm_owner_review": owner.CONFIRM,
        "approval_file": None,
    }
    values.update(fields)
    return SimpleNamespace(**values)


class OwnerConsentTests(unittest.TestCase):
    def test_no_args_never_reads_sources_or_writes_an_approval(self):
        with patch.object(owner.gate, "inspect_sources",
                          side_effect=AssertionError("read on default")), \
             redirect_stdout(StringIO()) as output:
            self.assertEqual(owner.main([]), 0)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["trainingStarted"])
        self.assertTrue(payload["doesNotApproveAutomatically"])
        self.assertFalse(payload["checkpointWrites"])
        self.assertEqual(payload["maximumOptimizerUpdates"], 50)
        self.assertEqual(payload["hardTimeoutSeconds"], 3600)

    def test_read_only_review_lists_all_exact_candidate_lines_without_approval(self):
        fake = fake_inspected()
        with patch.object(owner.gate, "inspect_sources", return_value=fake):
            scope = owner.review_sources(args())
        self.assertEqual(scope["status"],
                         "exact-scope-reviewed-source-verified-no-training")
        self.assertEqual(len(scope["trainAndHeldoutRecordDetails"]), 18)
        self.assertEqual(sum(x["split"] == "train" for x in
                             scope["trainAndHeldoutRecordDetails"]), 12)
        self.assertEqual(sum(x["split"] == "heldout" for x in
                             scope["trainAndHeldoutRecordDetails"]), 6)
        self.assertEqual(scope["unapprovedExactScopeTemplate"]["approvedForTraining"],
                         False)
        warmup = next(x for x in scope["trainAndHeldoutRecordDetails"]
                      if x["id"] == "m10-js-counter")
        self.assertEqual(warmup["snippet"], "let score = 0;\n")
        self.assertFalse(scope["trainingStarted"])

    def test_explicit_external_owner_record_is_create_only_and_matches_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "m10-owner-approval.json"
            with patch.object(owner.gate, "inspect_sources",
                              return_value=fake_inspected()):
                a = args(approval_file=path)
                report = owner.record_owner_approval(a)
                self.assertTrue(path.exists())
                self.assertEqual(report["status"],
                    "explicit-external-owner-approval-recorded-not-training")
                self.assertFalse(report["trainingStarted"])
                self.assertFalse(report["checkpointWrites"])
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(saved["approvedForTraining"])
                self.assertTrue(saved["assistantProposedSourcesAcceptedForTraining"])
                self.assertEqual(saved["approvedBy"], "keepithandy")
                self.assertEqual(saved["approvedAt"], date.today().isoformat())
                self.assertEqual(saved["phaseUpdateCounts"], [10, 15, 25])
                self.assertEqual(saved["maxOptimizerUpdates"], 50)
                self.assertEqual(saved["timeBudgetSeconds"], 3600)
                self.assertFalse(saved["finalHoldoutAllowed"])
                self.assertEqual(len(saved["trainRecordIds"]), 12)
                self.assertEqual(len(saved["heldoutRecordIds"]), 6)
                self.assertEqual(
                    gate.check_approval(path, fake_inspected()["unapprovedTemplate"])
                        ["approvalFileSha256"],
                    report["ownerApprovalSha256"])
                existing = path.read_bytes()
                with self.assertRaisesRegex(owner.Refusal, "already exists"):
                    owner.record_owner_approval(a)
                self.assertEqual(path.read_bytes(), existing)

    def test_wrong_explicit_owner_phrase_fails_before_source_inspection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "m10-owner-approval.json"
            with patch.object(owner, "review_sources",
                              side_effect=AssertionError("should not inspect")):
                with self.assertRaisesRegex(owner.Refusal, "does not match"):
                    owner.record_owner_approval(
                        args(approval_file=path,
                             confirm_owner_review="approve all"))
            self.assertFalse(path.exists())

    def test_cannot_record_consent_inside_repository_or_reuse_existing_file(self):
        with self.assertRaisesRegex(owner.Refusal, "outside the Git repository"):
            owner._external_output(research.ROOT / "m10-owner-approval.json")
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "m10-owner-approval.json"
            target.write_text("not authorization", encoding="utf-8")
            with self.assertRaisesRegex(owner.Refusal, "already exists"):
                owner._external_output(target)
            with self.assertRaisesRegex(owner.Refusal, "filename"):
                owner._external_output(Path(temp) / "other.json")

    def test_source_validation_failure_creates_no_file(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "m10-owner-approval.json"
            with patch.object(owner.gate, "inspect_sources",
                              side_effect=gate.Refusal("checkpoint SHA mismatch")):
                with self.assertRaisesRegex(gate.Refusal, "SHA mismatch"):
                    owner.record_owner_approval(args(approval_file=target))
            self.assertFalse(target.exists())

    def test_discovers_only_full_18_case_baseline_paths_without_validating(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/"plex600-m10-m9-full-baseline-20261118"
            folder.mkdir()
            (folder / "report.json").write_text(json.dumps({
                "selectedCases": 18,
                "decodingPolicies": list(gate.DECODERS),
                "status": "read-only-m9-generation-complete",
            }), encoding="utf-8")
            (folder / "responses.jsonl").write_bytes(b'{}\n')
            found = owner.list_baselines(Path(temp))
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["status"], "located-not-yet-SHA-verified")
            self.assertTrue(found[0]["baselineResponses"].endswith("responses.jsonl"))
            (folder / "report.json").write_text(json.dumps({
                "selectedCases": 3,
                "decodingPolicies": list(gate.DECODERS),
                "status": "read-only-m9-generation-complete",
            }), encoding="utf-8")
            self.assertEqual(owner.list_baselines(Path(temp)), [])

    def test_all_cli_modes_are_mutually_exclusive_and_never_execute_training(self):
        with redirect_stderr(StringIO()):
            self.assertEqual(owner.main(["--record-owner-approval"]), 2)
            self.assertEqual(owner.main(["--confirm-owner-review", owner.CONFIRM]), 2)
            with self.assertRaises(SystemExit) as result:
                owner.main(["--execute-approved-run"])
            self.assertEqual(result.exception.code, 2)
            with self.assertRaises(SystemExit) as result:
                owner.main(["--list-baselines", "--record-owner-approval"])
            self.assertEqual(result.exception.code, 2)
        content = Path(owner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", content)
        self.assertNotIn("optimizer.step(", content)
        self.assertNotIn("torch.save(", content)
        self.assertNotIn("run_approved(", content)


if __name__ == "__main__":
    unittest.main()
