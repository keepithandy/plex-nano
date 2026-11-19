"""CPU-only M11 admission/receipt controls. No CUDA, optimizer or checkpoint."""
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, timedelta
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from plex_training import plex600_m11_admission as admission
from plex_training import plex600_m11_source_conditioning as m11
from plex_training import plex600_m11_training_core as core


def synthetic_source():
    return {
        "candidateSha256": core.candidate_sha256(),
        "scheduleSha256": core.schedule_sha256(),
        "bundleManifestSha256": "b" * 64,
        "tokenizerSha256": "c" * 64,
        "m9WeightsSha256": "a" * 64,
        "m9ReportSha256": "d" * 64,
        "trainingIds": [c["id"] for c in m11.cases() if c["split"] == "train"],
        "developmentOnlyIds": [
            c["id"] for c in m11.cases() if c["split"] == "probe-only"],
        "sourceHashes": {
            "modelSourceSha256": "e" * 64,
            "m11CandidateSourceSha256": "f" * 64,
            "m11TrainingCoreSourceSha256": "1" * 64,
            "m11AdmissionSourceSha256": "2" * 64,
        },
    }


def synthetic_baseline_files(root, source, output_override=None):
    folder = Path(root)
    folder.mkdir(parents=True, exist_ok=True)
    paths = folder / "report.json", folder / "responses.jsonl"
    rows = [
        {
            "id": case["id"],
            "split": case["split"],
            "decodingPolicy": policy,
            "output": case["snippet"] if output_override is None else output_override,
            "terminationReason": "eos",
            "generatedTokens": 10,
            "promptSha256": hashlib.sha256(
                m11.prompt_for(case).encode("utf-8")).hexdigest(),
            "checkpointSha256": source["m9WeightsSha256"],
        }
        for policy in core.DECODERS for case in m11.cases()
    ]
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    paths[1].write_bytes(raw.encode("utf-8"))
    scorer_rows = [{
        key: row[key] for key in (
            "id", "decodingPolicy", "output", "terminationReason",
            "promptSha256", "checkpointSha256",
        )
    } for row in rows]
    scored = core.score_unverified_generations(
        scorer_rows, source["m9WeightsSha256"])
    report = {
        "protocol": admission.BASELINE_PROTOCOL,
        "status": admission.BASELINE_STATUS,
        "checkpointSha256": source["m9WeightsSha256"],
        "m9ReportSha256": source["m9ReportSha256"],
        "bundleManifestSha256": source["bundleManifestSha256"],
        "tokenizerSha256": source["tokenizerSha256"],
        "candidateSha256": source["candidateSha256"],
        "scheduleSha256": source["scheduleSha256"],
        "sourceHashes": source["sourceHashes"],
        "responsesSha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "selectedCaseIds": [case["id"] for case in m11.cases()],
        "decodingPolicies": list(core.DECODERS),
        "responseCount": 8,
        "optimizerUpdates": 0,
        "readOnlyGenerationExecuted": True,
        "trainingAuthorized": False,
        "checkpointWrites": False,
        "finalHoldoutUsed": False,
        "scores": scored["scores"],
    }
    paths[0].write_text(json.dumps(report) + "\n", encoding="utf-8")
    return paths, report, rows


class M11AdmissionTests(unittest.TestCase):
    def test_default_plan_does_not_read_sources_or_train(self):
        with patch.object(admission, "inspect_sources", side_effect=AssertionError), \
                redirect_stdout(StringIO()) as out:
            self.assertEqual(admission.main([]), 0)
        report = json.loads(out.getvalue())
        self.assertFalse(report["trainingAuthorized"])
        self.assertFalse(report["modelLoaded"])
        self.assertFalse(report["checkpointWrites"])
        self.assertFalse(report["m11TrainerImplemented"])
        self.assertFalse(report["reuseM10ApprovalAllowed"])
        self.assertEqual(report["optimizerUpdatesCompleted"], 0)
        self.assertEqual(report["proposedMaxOptimizerUpdatesPerArm"], 200)
        self.assertEqual(report["minimumOutputFreeBytesPerArm"], 16 * 1024**3)

    def test_public_interface_fails_closed_without_exact_inputs(self):
        with redirect_stderr(StringIO()):
            self.assertEqual(admission.main(["--audit-sources"]), 2)
            self.assertEqual(admission.main([
                "--show-unapproved-template", "--eos-weight", "0.25"]), 2)
            self.assertEqual(admission.main(["--bundle-dir", "placeholder"]), 2)
            self.assertEqual(admission.main([
                "--audit-admission", "--execute-approved-run"]), 2)

    def test_distinct_m11_baseline_scores_every_case_two_decoders(self):
        source = synthetic_source()
        with TemporaryDirectory() as temp:
            (report_path, responses_path), _, _ = synthetic_baseline_files(
                Path(temp) / "m11-baseline", source)
            proof = admission.read_baseline(report_path, responses_path, source)
            self.assertTrue(proof["baselineReceiptConsistencyVerified"])
            self.assertFalse(proof["baselineInferenceIndependentlyReproduced"])
            self.assertFalse(proof["trainingAuthorized"])

    def test_reject_m10_protocol_or_mutated_source_scope(self):
        source = synthetic_source()
        with TemporaryDirectory() as temp:
            (report_path, responses_path), report, rows = synthetic_baseline_files(
                Path(temp) / "m11-baseline", source)
            report["protocol"] = "plex600-m10-m9-read-only-copy-baseline-v1"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(admission.Refusal, "baseline"):
                admission.read_baseline(report_path, responses_path, source)
            report["protocol"] = admission.BASELINE_PROTOCOL
            report["candidateSha256"] = "9" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(admission.Refusal, "identity"):
                admission.read_baseline(report_path, responses_path, source)
            report["candidateSha256"] = source["candidateSha256"]
            report_path.write_text(json.dumps(report), encoding="utf-8")
            rows[0]["promptSha256"] = "0" * 64
            raw = "".join(json.dumps(row) + "\n" for row in rows)
            responses_path.write_text(raw, encoding="utf-8")
            report["responsesSha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(core.Refusal, "SHA"):
                admission.read_baseline(report_path, responses_path, source)

    def test_reject_rewritten_responses_even_with_matching_rehashed_file(self):
        source = synthetic_source()
        with TemporaryDirectory() as temp:
            (report_path, responses_path), report, rows = synthetic_baseline_files(
                Path(temp) / "m11-baseline", source)
            rows[0]["output"] = "let score = 1;\n"
            raw = "".join(json.dumps(row) + "\n" for row in rows)
            responses_path.write_text(raw, encoding="utf-8")
            report["responsesSha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(admission.Refusal, "scores"):
                admission.read_baseline(report_path, responses_path, source)

    def test_template_is_scope_specific_per_arm_and_not_approved(self):
        source = synthetic_source()
        baseline = {
            "baselineReportSha256": "3" * 64,
            "baselineResponsesSha256": "4" * 64,
        }
        normal = admission.approval_template(source, baseline, 1.0)
        reduced = admission.approval_template(source, baseline, 0.25)
        self.assertEqual(normal["eosLossWeight"], 1.0)
        self.assertEqual(reduced["eosLossWeight"], 0.25)
        self.assertNotEqual(normal["eosLossWeight"], reduced["eosLossWeight"])
        self.assertFalse(normal["approvedForTraining"])
        self.assertEqual(normal["scheduleSha256"], core.schedule_sha256())
        self.assertEqual(normal["checkpointSteps"], [25, 50, 100, 200])
        self.assertEqual(normal["hardTimeoutSeconds"], 3600)
        self.assertFalse(normal["finalHoldoutAllowed"])
        with self.assertRaisesRegex(admission.Refusal, "arm"):
            admission.approval_template(source, baseline, 0.0)

    def test_owner_consent_is_external_exact_and_not_reusable_from_m10(self):
        source = synthetic_source()
        baseline = {
            "baselineReportSha256": "3" * 64,
            "baselineResponsesSha256": "4" * 64,
        }
        normal = admission.approval_template(source, baseline, 1.0)
        reduced = admission.approval_template(source, baseline, 0.25)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "m11-owner-consent.json"
            with self.assertRaises(admission.Refusal):
                admission.check_approval(path, normal)
            approved = dict(normal, approvedBy="keepithandy",
                approvedAt=date.today().isoformat(), approvedForTraining=True,
                sourceReviewedByOwner=True, reviewedBothTrainingExamples=True,
                reviewedDevelopmentOnlyExamples=True, m11ProposalScopeAccepted=True)
            path.write_text(json.dumps(approved), encoding="utf-8")
            self.assertEqual(len(admission.check_approval(path, normal)), 64)
            with self.assertRaisesRegex(admission.Refusal, "scope"):
                admission.check_approval(path, reduced)
            approved["kind"] = "plex600-m10-exact-owner-training-approval-v1"
            path.write_text(json.dumps(approved), encoding="utf-8")
            with self.assertRaisesRegex(admission.Refusal, "scope"):
                admission.check_approval(path, normal)
            approved["kind"] = admission.KIND
            approved["approvedAt"] = (date.today() + timedelta(days=1)).isoformat()
            path.write_text(json.dumps(approved), encoding="utf-8")
            with self.assertRaisesRegex(admission.Refusal, "date"):
                admission.check_approval(path, normal)

    def test_output_is_checked_before_checkpoint_or_approval_reads(self):
        with patch.object(admission.m8, "require_output",
                          side_effect=admission.Refusal("occupied output")) as output, \
             patch.object(admission, "inspect_sources", side_effect=AssertionError), \
             self.assertRaisesRegex(admission.Refusal, "occupied"):
            admission.admission(
                Path("bundle"), "x" * 64, Path("m9-report"), Path("m9-weights"),
                Path("baseline-report"), Path("baseline-responses"),
                Path("approval"), Path("occupied"), 1.0)
        output.assert_called_once()


if __name__ == "__main__":
    unittest.main()
