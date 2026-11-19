"""CPU-only M11 worker admission, scope, source and capability regression tests.

Synthetic dictionaries are deliberately NOT purported GPU or coding results.
No approved GPU run, optimizer or torch model is instantiated here.
"""
from contextlib import redirect_stdout, redirect_stderr
from datetime import date
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from plex_training import plex600_m11_worker as worker
from plex_training import plex600_m11_admission as admission
from plex_training import plex600_m11_source_conditioning as m11
from plex_training import plex600_m11_training_core as core
from plex_training import plex600_research as research


def proof() -> dict:
    return {
        "status": "cpu-inputs-and-owner-consent-verified-not-executed",
        "ownerConsentGranted": True,
        "optimizerUpdatesCompleted": 0,
        "chosenEosLossWeight": 1.0,
        "checkpointWrites": False,
        "candidateSha256": core.candidate_sha256(),
        "scheduleSha256": core.schedule_sha256(),
        "m9WeightsSha256": "a" * 64,
        "baselineReportSha256": "b" * 64,
        "baselineResponsesSha256": "c" * 64,
        "ownerApprovalSha256": "d" * 64,
    }


def observations(checkpoint: str, *, perfect: bool = True) -> list[dict]:
    output = []
    for policy in core.DECODERS:
        for case in m11.cases():
            output.append({
                "id": case["id"], "split": case["split"],
                "decodingPolicy": policy,
                "output": case["snippet"] if perfect else "rrrr",
                "terminationReason": "eos" if perfect else "max-new-tokens",
                "generatedTokens": 10,
                "promptSha256": hashlib.sha256(
                    m11.prompt_for(case).encode("utf-8")).hexdigest(),
                "checkpointSha256": checkpoint,
            })
    return output


def preferences(*, correct: bool = True) -> dict:
    return {
        split: {
            "split": split,
            "caseIds": [c["id"] for c in m11.cases() if c["split"] == split],
            "contrastTokenIds": [5, 6],
            "oneMinusZeroLogitMargins": [-2.0, 2.0],
            "fullVocabularyGreedyIds": [5, 6] if correct else [6, 6],
            "tracksSourceInBothDirections": correct,
            "usesGoldAnswerPrefix": True,
            "autonomousExactCopyingEvaluated": False,
        } for split in ("train", "probe-only")
    }


def milestone(step: int, *, perfect: bool = True) -> dict:
    return {
        "optimizerUpdatesCompleted": step,
        "scores": {
            policy: {
                "trainingPairExactAndEos": perfect,
                "developmentPairExactAndEos": perfect,
            } for policy in core.DECODERS
        },
        "sourcePreferences": preferences(correct=perfect),
    }


def synthetic_child(p: dict, identity: dict, *, weight=1.0) -> dict:
    schedule = core.exact_schedule()
    result = {
        "protocol": worker.PROTOCOL,
        "status": "bounded-200-updates-complete-unreviewed",
        "trainingAuthorized": True,
        "sourceIdentity": identity,
        "originalM9WeightsSha256": p["m9WeightsSha256"],
        "baselineReportSha256": p["baselineReportSha256"],
        "baselineResponsesSha256": p["baselineResponsesSha256"],
        "candidateSha256": p["candidateSha256"],
        "scheduleSha256": p["scheduleSha256"],
        "ownerApprovalSha256": p["ownerApprovalSha256"],
        "chosenEosLossWeight": weight,
        "maxOptimizerUpdates": core.MAX_UPDATES,
        "optimizerUpdatesCompleted": core.MAX_UPDATES,
        "steps": [{
            "step": s["optimizerUpdate"], "recordId": s["recordId"],
            "promptPositionsIncorrectlySupervised": 0,
            "eosTargetSupervised": True,
            "loss": 0.1, "gradientL2Norm": 1.0,
            "measuredEmbeddingRowDeltaL2": 0.01,
        } for s in schedule],
        "snapshots": [milestone(n) for n in core.CHECKPOINT_STEPS],
        "optimizerStateSaved": False,
        "historicalCheckpointWrites": False,
        "finalHoldoutUsed": False,
        "generalCodingCapabilityProven": False,
    }
    result["decision"] = worker._decision(result["snapshots"])
    return result


class M11WorkerTests(unittest.TestCase):
    def test_default_plan_never_runs_gpu_or_checks_owner_files(self):
        with patch.object(worker, "_admit", side_effect=AssertionError), \
             patch.object(worker, "_source_identity", side_effect=AssertionError), \
             patch.object(worker, "_train_worker", side_effect=AssertionError), \
             redirect_stdout(StringIO()) as stdout:
            self.assertEqual(worker.main([]), 0)
        data = json.loads(stdout.getvalue())
        self.assertFalse(data["trainingAuthorized"])
        self.assertFalse(data["modelLoaded"])
        self.assertFalse(data["checkpointWrites"])
        self.assertFalse(data["artifactWrites"])
        self.assertEqual(data["optimizerUpdatesCompleted"], 0)
        self.assertEqual(data["checkpointSteps"], [25, 50, 100, 200])
        self.assertEqual(data["eosLossWeights"], [1.0, 0.25])
        self.assertNotIn("torch", worker.__dict__)

    def test_public_cli_never_accepts_training_without_flag_and_paths(self):
        with redirect_stderr(StringIO()):
            self.assertEqual(worker.main(["--execute-approved-run"]), 2)
            self.assertEqual(worker.main([
                "--eos-weight", "1.0"]), 2)
            self.assertEqual(worker.main([
                "--m9-weights", "weights.pt"]), 2)
            self.assertEqual(worker.main([
                "--execute-approved-run", "--eos-weight", "0.0"]), 2)
            with self.assertRaises(SystemExit) as exc:
                worker.main(["--force-train"])
            self.assertEqual(exc.exception.code, 2)
        self.assertEqual(worker.plan()["m10ApprovalReusable"], False)

    def test_gate_is_unique_and_requires_specific_positive_owner_consent(self):
        paths = {name: Path(name) for name in worker.PATH_FIELDS}
        with patch.object(worker.gate, "admission",
                          return_value=proof()) as called:
            actual = worker._admit(paths, research.BUNDLE_SHA, 1.0)
            self.assertTrue(actual["ownerConsentGranted"])
            called.assert_called_once()
        for field, value in (
            ("ownerConsentGranted", False),
            ("status", "cpu-source-audit-only"),
            ("chosenEosLossWeight", 0.25),
            ("optimizerUpdatesCompleted", 1),
            ("checkpointWrites", True),
        ):
            invalid = {**proof(), field: value}
            with self.subTest(field=field), patch.object(
                    worker.gate, "admission", return_value=invalid):
                with self.assertRaises(worker.Refusal):
                    worker._admit(paths, research.BUNDLE_SHA, 1.0)
        with patch.object(worker.gate, "admission",
                          side_effect=AssertionError("opened sources")):
            with self.assertRaisesRegex(worker.Refusal, "original pinned"):
                worker._admit(paths, "bad-manifest", 1.0)

    def test_owner_consent_is_bound_to_exact_worker_snapshot_decoder_bytes(self):
        source = {
            "candidateSha256": core.candidate_sha256(),
            "scheduleSha256": core.schedule_sha256(),
            "bundleManifestSha256": research.BUNDLE_SHA,
            "m9WeightsSha256": "a" * 64,
            "m9ReportSha256": "b" * 64,
            "tokenizerSha256": "c" * 64,
            "sourceHashes": {"modelSourceSha256": "d" * 64},
            "trainingIds": [x["id"] for x in m11.cases() if x["split"] == "train"],
            "developmentOnlyIds": [
                x["id"] for x in m11.cases() if x["split"] == "probe-only"],
        }
        baseline = {
            "baselineReportSha256": "e" * 64,
            "baselineResponsesSha256": "f" * 64,
        }
        template = admission.approval_template(source, baseline, 1.0)
        expected = {
            "m11WorkerSourceSha256": worker.__file__,
            "m10SnapshotHelperSourceSha256": worker.m10.__file__,
            "m8DecoderSourceSha256": worker.m8.__file__,
        }
        for field, filename in expected.items():
            self.assertEqual(
                template[field], hashlib.sha256(Path(filename).read_bytes()).hexdigest())
        self.assertEqual(template["approvedForTraining"], False)
        with TemporaryDirectory() as temp:
            # Synthetic, throwaway fixture only. Never create owner consent.
            approval = Path(temp) / "synthetic-test-fixture.json"
            content = {**template,
                "approvedBy": "keepithandy",
                "approvedAt": date.today().isoformat(),
                "approvedForTraining": True,
                "sourceReviewedByOwner": True,
                "reviewedBothTrainingExamples": True,
                "reviewedDevelopmentOnlyExamples": True,
                "m11ProposalScopeAccepted": True,
            }
            approval.write_text(json.dumps(content), encoding="utf-8")
            self.assertEqual(len(admission.check_approval(approval, template)), 64)
            for field in expected:
                tampered_expected = {**template, field: "0" * 64}
                with self.subTest(field=field), self.assertRaisesRegex(
                        admission.Refusal, "scope"):
                    admission.check_approval(approval, tampered_expected)

    def test_exact_outputs_are_recomputed_and_are_not_report_only(self):
        checkpoint = "a" * 64
        correct = observations(checkpoint)
        scores = core.score_unverified_generations(
            [{k: row[k] for k in (
                "id", "decodingPolicy", "output", "terminationReason",
                "promptSha256", "checkpointSha256")} for row in correct],
            checkpoint)["scores"]
        verified = worker._verify_observations(correct, checkpoint, scores)
        self.assertEqual(verified, scores)
        for policy in core.DECODERS:
            self.assertTrue(verified[policy]["trainingPairExactAndEos"])
            self.assertTrue(verified[policy]["developmentPairExactAndEos"])
        wrong = observations(checkpoint, perfect=False)
        wrong_score = core.score_unverified_generations(
            [{k: row[k] for k in (
                "id", "decodingPolicy", "output", "terminationReason",
                "promptSha256", "checkpointSha256")} for row in wrong],
            checkpoint)["scores"]
        self.assertFalse(worker._verify_observations(
            wrong, checkpoint, wrong_score)["unmasked-v1"]["trainingPairExactAndEos"])
        with self.assertRaisesRegex(worker.Refusal, "scores"):
            worker._verify_observations(wrong, checkpoint, scores)

    def test_reject_duplicate_case_unknown_policy_bad_digest_and_probe_swap(self):
        checkpoint = "a" * 64
        rows = observations(checkpoint)
        score = core.score_unverified_generations([
            {k: row[k] for k in (
                "id", "decodingPolicy", "output", "terminationReason",
                "promptSha256", "checkpointSha256")} for row in rows
        ], checkpoint)["scores"]
        for field, value in (
            ("checkpointSha256", "b" * 64),
            ("promptSha256", "f" * 64),
            ("split", "probe-only"),
            ("decodingPolicy", "temperature"),
            ("generatedTokens", 97),
            ("output", 1),
        ):
            changed = [dict(x) for x in rows]
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(worker.Refusal):
                worker._verify_observations(changed, checkpoint, score)
        changed = [dict(x) for x in rows]
        changed[1]["id"] = changed[0]["id"]
        with self.assertRaises(worker.Refusal):
            worker._verify_observations(changed, checkpoint, score)
        with self.assertRaises(worker.Refusal):
            worker._verify_observations(rows[:-1], checkpoint, score)

    def test_source_contrast_rejects_unsupported_nonfinite_or_falsified_gate(self):
        worker._validate_preferences(preferences())
        worker._validate_preferences(preferences(correct=False))
        for alteration in ("nan", "fake", "wrong_split", "missing"):
            p = preferences()
            if alteration == "nan":
                p["train"]["oneMinusZeroLogitMargins"][0] = float("nan")
            elif alteration == "fake":
                p["train"]["tracksSourceInBothDirections"] = False
            elif alteration == "wrong_split":
                p["train"]["split"] = "probe-only"
            else:
                del p["probe-only"]
            with self.subTest(alteration=alteration), self.assertRaises(worker.Refusal):
                worker._validate_preferences(p)

    def test_real_filesystem_snapshot_receipts_and_corruption_rejected(self):
        p = proof()
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "step-25"
            folder.mkdir()
            weights = folder / "weights.pt"
            weights.write_bytes(b"SYNTHETIC CHECKPOINT TEST - NO MODEL")
            checkpoint_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
            rows = observations(checkpoint_sha)
            raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            response_path = folder / "responses.jsonl"
            response_path.write_bytes(raw.encode("utf-8"))
            score = core.score_unverified_generations([
                {key: row[key] for key in (
                    "id", "decodingPolicy", "output", "terminationReason",
                    "promptSha256", "checkpointSha256")}
                for row in rows], checkpoint_sha)["scores"]
            record = {
                "optimizerUpdatesCompleted": 25,
                "checkpointSha256": checkpoint_sha,
                "checkpoint": {"sha256": checkpoint_sha,
                               "bytes": weights.stat().st_size},
                "responsesSha256": hashlib.sha256(
                    response_path.read_bytes()).hexdigest(),
                "candidateSha256": p["candidateSha256"],
                "scheduleSha256": p["scheduleSha256"],
                "baselineResponsesSha256": p["baselineResponsesSha256"],
                "ownerApprovalSha256": p["ownerApprovalSha256"],
                "chosenEosLossWeight": 1.0,
                "decodingPolicies": list(core.DECODERS),
                "verifiedGeneration": True,
                "optimizerStateSaved": False,
                "finalHoldoutUsed": False,
                "scores": score,
                "sourcePreferences": preferences(),
            }
            report_path = folder / "report.json"
            report_path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(worker._verify_snapshot(folder, 25, p, 1.0), record)
            response_path.write_bytes((raw + "{}\n").encode("utf-8"))
            with self.assertRaises(worker.Refusal):
                worker._verify_snapshot(folder, 25, p, 1.0)
            response_path.write_bytes(raw.encode("utf-8"))
            for field in ("checkpoint", "scores", "sourcePreferences"):
                bad = {**record, field: "invalid"}
                report_path.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(worker.Refusal):
                    worker._verify_snapshot(folder, 25, p, 1.0)

    def test_decision_does_not_claim_general_coding_from_two_examples(self):
        record = worker._decision([milestone(25), milestone(50)])
        self.assertTrue(record["lastTwoMilestonesMeetNarrowM11Gate"])
        self.assertFalse(record["generalCodingCapabilityProven"])
        self.assertFalse(record["unseenBroadJavaScriptGeneralizationProven"])
        self.assertTrue(record["trainingDataMemorizationNotExcluded"])
        damaged = worker._decision([milestone(25), milestone(50, perfect=False)])
        self.assertFalse(damaged["lastTwoMilestonesMeetNarrowM11Gate"])
        self.assertFalse(damaged["promotionAutomatic"])

    def test_parent_rejects_forged_or_incomplete_child_updates(self):
        p = proof()
        identity = {"m11WorkerSha256": "d" * 64}
        result = synthetic_child(p, identity)
        with patch.object(worker, "_publication_step") as snapshots:
            worker._check_final(result, p, identity, 1.0, Path("stage"))
            self.assertEqual(snapshots.call_count, 4)
        variants = [
            {"steps": result["steps"][:-1]},
            {"snapshots": result["snapshots"][:-1]},
            {"ownerApprovalSha256": "0" * 64},
            {"chosenEosLossWeight": 0.25},
            {"generalCodingCapabilityProven": True},
            {"trainingAuthorized": False},
            {"decision": {"promotionAutomatic": True}},
        ]
        for delta in variants:
            with self.subTest(delta=list(delta)), patch.object(
                    worker, "_publication_step"):
                with self.assertRaises(worker.Refusal):
                    worker._check_final({**result, **delta}, p, identity, 1.0,
                                        Path("stage"))
        altered = synthetic_child(p, identity)
        altered["steps"] = [dict(x) for x in altered["steps"]]
        altered["steps"][4]["recordId"] = "m11-probe-only-value-0"
        with patch.object(worker, "_publication_step"), self.assertRaises(worker.Refusal):
            worker._check_final(altered, p, identity, 1.0, Path("stage"))

    def test_refuses_before_cuda_child_when_no_approval(self):
        with patch.object(worker, "_admit",
                          side_effect=worker.Refusal("approval absent")), \
             patch.object(worker.hw, "_supervise",
                          side_effect=AssertionError("started worker")), \
             self.assertRaisesRegex(worker.Refusal, "approval absent"):
            worker.run_approved({}, research.BUNDLE_SHA, 1.0, explicit_flag=True)
        with patch.object(worker, "_admit",
                          side_effect=AssertionError("opened anything")), \
             self.assertRaisesRegex(worker.Refusal, "explicit"):
            worker.run_approved({}, research.BUNDLE_SHA, 1.0, explicit_flag=False)

    def test_child_must_be_nonce_guarded(self):
        with patch.object(worker.sys, "stdin") as stdin, \
             patch.dict(worker.os.environ, {worker.CONTROL: ""}):
            stdin.buffer.read.return_value = b'{"nonce":"secret"}'
            with self.assertRaisesRegex(worker.Refusal, "approved parent nonce"):
                worker._worker_entry()

    def test_declared_modes_not_training_during_test_collection(self):
        self.assertEqual(core.MAX_UPDATES, 200)
        self.assertEqual(worker.HARD_TIMEOUT, 3600)
        self.assertEqual(worker.SNAPSHOTS, (25, 50, 100, 200))
        self.assertEqual(worker.SEED, 41)
        self.assertEqual(len(worker.plan()["developmentOnlyRecordIds"]), 2)


if __name__ == "__main__":
    unittest.main()
