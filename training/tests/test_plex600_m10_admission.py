"""M10 CPU-only admission gates. Synthetic data is NEVER training approval."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import date
from io import StringIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from plex_training import plex600_m10_admission as admission
from plex_training import plex600_m10_copy_proposal as m10
from plex_training import plex600_research as research


def synthetic_baseline(directory: Path) -> tuple[Path, Path]:
    """Create a consistent fake 18×2 report for validation only."""
    rows = m10.rows()
    response_rows = []
    by_policy = {}
    for decoder in admission.DECODERS:
        for item in rows:
            response_rows.append({
                "id": item["id"], "output": item["snippet"],
                "terminationReason": "eos", "split": item["split"],
                "language": item["language"], "decodingPolicy": decoder,
                "generatedTokens": 1,
                "promptSha256": hashlib.sha256(
                    m10.prompt_for(item).encode()).hexdigest(),
                "m9WeightsSha256": "a" * 64,
            })
        score = m10.score_predictions(
            rows,
            [{"id": item["id"], "output": item["snippet"],
              "terminationReason": "eos"} for item in rows],
        )
        by_policy[decoder] = {
            "bySplit": {split: {
                "total": score[split]["total"],
                "exact": score[split]["exact"],
                "eos": score[split]["eos"],
                "exactWithEos": score[split]["exactWithEos"],
                "meanCharacterSimilarity": score[split]["meanCharacterSimilarity"],
            } for split in m10.SPLITS},
            "complete18CaseComparison": True,
        }
    responses = directory / "responses.jsonl"
    responses.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in response_rows
    ), encoding="utf-8", newline="\n")
    report = directory / "report.json"
    data = {
        "protocol": admission.BASELINE_PROTOCOL,
        "status": "read-only-m9-generation-complete",
        "readOnlyGenerationExecuted": True,
        "generatedCodeEvaluated": True,
        "trainingAuthorized": False,
        "checkpointWrites": False,
        "finalHoldoutUsed": False,
        "optimizerUpdates": 0,
        "m9WeightsSha256": "a" * 64,
        "m9ReportSha256": "b" * 64,
        "candidateSha256": research.digest_file(m10.CANDIDATE),
        "bundleManifestSha256": research.BUNDLE_SHA,
        "tokenizerSha256": admission.TOKENIZER_SHA256,
        "responsesSha256": hashlib.sha256(responses.read_bytes()).hexdigest(),
        "selectedCases": 18,
        "selectedCaseIds": [row["id"] for row in rows],
        "decodingPolicies": list(admission.DECODERS),
        "results": by_policy,
    }
    report.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    return report, responses


def owner_document(template: dict) -> dict:
    return {
        **template,
        "approvedBy": "keepithandy",
        "approvedAt": date.today().isoformat(),
        "approvedForTraining": True,
        "sourceReviewedByOwner": True,
        "assistantProposedSourcesAcceptedForTraining": True,
    }


class M10AdmissionTests(unittest.TestCase):
    def test_default_does_not_open_any_files_or_implement_training(self):
        with patch.object(admission, "inspect_sources",
                          side_effect=AssertionError("read inputs")), \
             redirect_stdout(StringIO()) as output:
            self.assertEqual(admission.main([]), 0)
        result = json.loads(output.getvalue())
        self.assertFalse(result["runEntrypointImplemented"])
        self.assertFalse(result["trainingAuthorized"])
        self.assertFalse(result["modelLoaded"])
        self.assertFalse(result["checkpointWrites"])
        self.assertEqual(result["optimizerUpdatesCompleted"], 0)
        self.assertEqual(result["maxOptimizerUpdates"], 50)
        self.assertEqual(result["phaseUpdates"], [10, 15, 25])

    def test_baseline_verifies_full_18_by_two_policy_and_response_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            report, responses = synthetic_baseline(Path(temp))
            result = admission.read_baseline_receipt(
                report, responses, m9_weights_sha="a"*64,
                m9_report_sha="b"*64,
                candidate_sha=research.digest_file(m10.CANDIDATE),
            )
            self.assertTrue(result["complete18CasesBothDecoders"])
            self.assertFalse(result["trainingAuthorized"])
            self.assertFalse(result["checkpointWrites"])
            self.assertEqual(result["baselineResponsesSha256"],
                             hashlib.sha256(responses.read_bytes()).hexdigest())
            data = json.loads(report.read_text(encoding="utf-8"))
            for field, value in (
                ("m9WeightsSha256", "c"*64),
                ("candidateSha256", "d"*64),
                ("selectedCases", 3),
                ("trainingAuthorized", True),
                ("finalHoldoutUsed", True),
            ):
                with self.subTest(field=field):
                    modified = {**data, field: value}
                    report.write_text(json.dumps(modified), encoding="utf-8")
                    with self.assertRaises(admission.Refusal):
                        admission.read_baseline_receipt(
                            report, responses, m9_weights_sha="a"*64,
                            m9_report_sha="b"*64,
                            candidate_sha=research.digest_file(m10.CANDIDATE))
            report.write_text(json.dumps(data), encoding="utf-8")
            responses.write_text(responses.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaises(admission.Refusal):
                admission.read_baseline_receipt(
                    report, responses, m9_weights_sha="a"*64,
                    m9_report_sha="b"*64,
                    candidate_sha=research.digest_file(m10.CANDIDATE))

    def test_baseline_rejects_poisoned_prompt_or_wrong_score(self):
        with tempfile.TemporaryDirectory() as temp:
            report, responses = synthetic_baseline(Path(temp))
            original = json.loads(report.read_text(encoding="utf-8"))
            lines = [json.loads(x) for x in responses.read_text(encoding="utf-8").splitlines()]
            lines[0]["promptSha256"] = "0" * 64
            responses.write_text("".join(json.dumps(x) + "\n" for x in lines),
                                 encoding="utf-8", newline="\n")
            original["responsesSha256"] = hashlib.sha256(responses.read_bytes()).hexdigest()
            report.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(admission.Refusal, "prompt/token"):
                admission.read_baseline_receipt(
                    report, responses, m9_weights_sha="a"*64,
                    m9_report_sha="b"*64,
                    candidate_sha=research.digest_file(m10.CANDIDATE))
            report, responses = synthetic_baseline(Path(temp))
            bad = json.loads(report.read_text(encoding="utf-8"))
            bad["results"]["unmasked-v1"]["bySplit"]["train"]["exact"] = 0
            report.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(admission.Refusal, "score differs"):
                admission.read_baseline_receipt(
                    report, responses, m9_weights_sha="a"*64,
                    m9_report_sha="b"*64,
                    candidate_sha=research.digest_file(m10.CANDIDATE))

    def test_unapproved_template_cannot_serve_as_owner_consent(self):
        source = {"m9WeightsSha256": "a"*64, "m9ReportSha256": "b"*64}
        baseline = {"baselineReportSha256": "c"*64,
                    "baselineResponsesSha256": "d"*64}
        template = admission.approval_template(source, baseline)
        self.assertEqual(template["maxOptimizerUpdates"], 50)
        self.assertEqual(template["examplesPerOptimizerUpdate"], 1)
        self.assertEqual(template["phaseUpdateCounts"], [10, 15, 25])
        self.assertEqual(template["evaluationUpdates"], [10, 25, 50])
        self.assertFalse(template["approvedForTraining"])
        self.assertFalse(template["assistantProposedSourcesAcceptedForTraining"])
        self.assertEqual(template["m9WeightsSha256"], source["m9WeightsSha256"])
        self.assertEqual(template["trainRecordIds"], m10.audit()["trainRecordIds"])
        with tempfile.TemporaryDirectory() as temp:
            owner = Path(temp)/"owner.json"
            owner.write_text(json.dumps(template), encoding="utf-8")
            with self.assertRaisesRegex(admission.Refusal, "not separately authorized"):
                admission.check_approval(owner, template)
            consent = owner_document(template)
            owner.write_text(json.dumps(consent), encoding="utf-8")
            proof = admission.check_approval(owner, template)
            self.assertTrue(proof["externalApprovalValidated"])
            self.assertFalse(proof["trainingExecuted"])
            for field, value in (
                ("candidateSha256", "f"*64),
                ("m9WeightsSha256", "f"*64),
                ("maxOptimizerUpdates", 51),
                ("phaseUpdateCounts", [20, 5, 25]),
                ("assistantProposedSourcesAcceptedForTraining", False),
                ("finalHoldoutAllowed", True),
                ("trainRecordIds", []),
            ):
                with self.subTest(field=field):
                    changed = deepcopy(consent)
                    changed[field] = value
                    owner.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(admission.Refusal):
                        admission.check_approval(owner, template)

    def test_cpu_admission_checks_destination_before_sources_or_disk(self):
        with patch.object(admission.m8, "require_output",
                          side_effect=admission.Refusal("already exists")), \
             patch.object(admission, "inspect_sources",
                          side_effect=AssertionError("read source before output gate")):
            with self.assertRaisesRegex(admission.Refusal, "already exists"):
                admission.admission(
                    Path("bundle"), research.BUNDLE_SHA, Path("report"),
                    Path("weights"), Path("baseline"), Path("responses"),
                    Path("approval"), Path("output"))

    def test_cpu_admission_cannot_run_and_checks_disk_capacity(self):
        source = {"m9WeightsSha256": "a"*64, "m9ReportSha256": "b"*64}
        baseline = {"baselineReportSha256": "c"*64,
                    "baselineResponsesSha256": "d"*64}
        token = {**m10.audit(), "tokenizerSha256": admission.TOKENIZER_SHA256}
        inspected = {"backup": source, "baseline": baseline, "tokenAudit": token,
                     "unapprovedTemplate": admission.approval_template(source, baseline)}
        with patch.object(admission.m8, "require_output",
                          return_value=Path("training/artifacts/new")), \
             patch.object(admission, "inspect_sources", return_value=inspected), \
             patch.object(admission, "check_approval",
                          return_value={"externalApprovalValidated": True,
                                        "approvalFileSha256": "f"*64}), \
             patch.object(admission.shutil, "disk_usage",
                          return_value=SimpleNamespace(free=admission.MIN_OUTPUT_FREE_BYTES)):
            result = admission.admission(
                Path("bundle"), research.BUNDLE_SHA, Path("report"),
                Path("weights"), Path("baseline"), Path("responses"),
                Path("approval"), Path("output"))
        self.assertEqual(result["status"], "cpu-admission-validated-not-executed")
        self.assertTrue(result["trainingAuthorized"])
        self.assertFalse(result["runEntrypointImplemented"])
        self.assertFalse(result["checkpointWrites"])
        self.assertEqual(result["optimizerUpdatesCompleted"], 0)
        with patch.object(admission.m8, "require_output",
                          return_value=Path("training/artifacts/new")), \
             patch.object(admission, "inspect_sources", return_value=inspected), \
             patch.object(admission, "check_approval",
                          return_value={"externalApprovalValidated": True,
                                        "approvalFileSha256": "f"*64}), \
             patch.object(admission.shutil, "disk_usage",
                          return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(admission.Refusal, "room"):
                admission.admission(
                    Path("bundle"), research.BUNDLE_SHA, Path("report"),
                    Path("weights"), Path("baseline"), Path("responses"),
                    Path("approval"), Path("output"))

    def test_cli_requires_explicit_gates_and_refuses_training_mode(self):
        with redirect_stderr(StringIO()):
            self.assertEqual(admission.main(["--audit-admission"]), 2)
            self.assertEqual(admission.main(["--show-approval-template"]), 2)
            self.assertEqual(admission.main(["--approval-file", "approval.json"]), 2)
            with self.assertRaises(SystemExit) as result:
                admission.main(["--execute-approved-run"])
            self.assertEqual(result.exception.code, 2)
            with self.assertRaises(SystemExit) as result:
                admission.main(["--show-approval-template", "--audit-admission"])
            self.assertEqual(result.exception.code, 2)

    def test_no_optimizer_or_torch_in_admission_module(self):
        source = Path(admission.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("torch.save(", source)
        self.assertNotIn("train_worker(", source)
        self.assertNotIn("--execute-approved-run", source)


if __name__ == "__main__":
    unittest.main()
