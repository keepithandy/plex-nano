"""CPU-only source/receipt/decision tests. Synthetic fixtures are NOT Plex runs."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from plex_training import plex600_phase1_historical as history
from plex_training import plex600_phase1_registry as registry
from plex_training import plex600_m10_copy_proposal as copy
from plex_training import plex600_m10_admission as gate
from plex_training import plex600_m10_worker as worker


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, sort_keys=True) + "\n").encode("utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root: Path) -> dict:
    """Produce exact M10-shaped synthetic evidence without model checkpoints."""
    bundle = root / "tokenizer"
    bundle.mkdir()
    m9 = root / "backup"
    m9.mkdir()
    (m9 / "weights.pt").write_bytes(b"SYNTHETIC M9 WEIGHTS\n")
    save(m9 / "report.json", {"synthetic": True})
    baseline = root / "baseline"
    baseline.mkdir()
    run = root / "m10"
    run.mkdir()
    cases = copy.rows()
    policies = gate.DECODERS
    candidate_sha = copy.audit()["candidateSha256"]
    def responses(checkpoint: str, baseline_format: bool) -> tuple[bytes, dict]:
        entries = []
        score = {}
        for policy in policies:
            predictions = []
            for case in cases:
                row = {
                    "id": case["id"], "split": case["split"],
                    "language": case["language"],
                    "decodingPolicy": policy, "output": case["snippet"],
                    "terminationReason": "eos", "generatedTokens": 4,
                    "promptSha256": hashlib.sha256(
                        copy.prompt_for(case).encode("utf-8")).hexdigest(),
                    "m9WeightsSha256" if baseline_format else "checkpointSha256": checkpoint,
                }
                entries.append(row)
                predictions.append({
                    "id": case["id"], "output": case["snippet"],
                    "terminationReason": "eos",
                })
            score[policy] = copy.score_predictions(cases, predictions)
        return ("".join(json.dumps(x, sort_keys=True) + "\n" for x in entries)
                .encode("utf-8"), score)
    original_m9 = sha(m9 / "weights.pt")
    baseline_raw, _ = responses(original_m9, True)
    (baseline / "responses.jsonl").write_bytes(baseline_raw)
    save(baseline / "report.json", {
        "protocol": "plex600-m10-m9-read-only-copy-baseline-v1",
        "m9WeightsSha256": original_m9,
        "selectedCaseIds": [x["id"] for x in cases],
        "responsesSha256": sha(baseline / "responses.jsonl"),
    })
    baseline_report_sha = sha(baseline / "report.json")
    baseline_responses_sha = sha(baseline / "responses.jsonl")
    snapshots = []
    for step in (10, 25):
        folder = run / f"step-{step}"
        folder.mkdir()
        (folder / "weights.pt").write_bytes(f"SYNTHETIC M10 {step}\n".encode("utf-8"))
        checkpoint_sha = sha(folder / "weights.pt")
        raw, score = responses(checkpoint_sha, False)
        (folder / "responses.jsonl").write_bytes(raw)
        record = {
            "optimizerUpdatesCompleted": step,
            "checkpointSha256": checkpoint_sha,
            "checkpoint": {"sha256": checkpoint_sha,
                           "bytes": (folder / "weights.pt").stat().st_size},
            "responsesSha256": sha(folder / "responses.jsonl"),
            "candidateSha256": candidate_sha,
            "verifiedGeneration": True,
            "decodingPolicies": list(policies),
            "scores": score, "finalHoldoutUsed": False,
        }
        save(folder / "report.json", record)
        snapshots.append(record)
    schedule = worker.exact_schedule()
    m9_report_sha = sha(m9 / "report.json")
    save(run / "report.json", {
        "protocol": worker.PROTOCOL,
        "status": "stopped-after-two-perfect-milestones",
        "trainingAuthorized": True,
        "maxOptimizerUpdates": 50,
        "optimizerUpdatesCompleted": 25,
        "optimizerStateSaved": False,
        "finalHoldoutUsed": False,
        "historicalCheckpointWrites": False,
        "originalM9WeightsSha256": original_m9,
        "originalM9ReportSha256": m9_report_sha,
        "baselineReportSha256": baseline_report_sha,
        "baselineResponsesSha256": baseline_responses_sha,
        "candidateSha256": candidate_sha,
        "scheduleSha256": gate.schedule_sha256(schedule),
        "approvalSha256": "c" * 64,
        "snapshots": snapshots,
        "steps": [{
            "step": row["optimizerUpdate"],
            "recordId": row["recordIds"][0],
            "promptPositionsIncorrectlySupervised": 0,
            "eosTargetSupervised": True,
        } for row in schedule[:25]],
    })
    proof = {
        "backup": {
            "m9WeightsSha256": original_m9,
            "m9ReportSha256": m9_report_sha,
            "optimizerUpdatesVerified": 8,
        },
        "baseline": {
            "baselineReportSha256": baseline_report_sha,
            "baselineResponsesSha256": baseline_responses_sha,
        },
        "tokenAudit": {
            "candidateSha256": candidate_sha,
            "tokenizerSha256": "f" * 64,
        },
    }
    return {
        "bundle": bundle, "m9": m9, "baseline": baseline,
        "run": run, "proof": proof,
    }


class HistoricalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.f = fixture(self.root)
        self.scope = {
            "bundle_dir": self.f["bundle"],
            "m9_report": self.f["m9"] / "report.json",
            "m9_weights": self.f["m9"] / "weights.pt",
            "baseline_dir": self.f["baseline"],
            "m10_run_dir": self.f["run"],
        }

    def audit(self) -> dict:
        with patch.object(history.gate, "inspect_sources",
                          return_value=self.f["proof"]), \
             patch.object(history, "ORIGINAL_M9_SHA",
                          self.f["proof"]["backup"]["m9WeightsSha256"]):
            return history.audit(**self.scope)

    def test_default_is_inert(self):
        out = StringIO()
        with redirect_stdout(out):
            rc = history.main([])
        self.assertEqual(rc, 0)
        obj = json.loads(out.getvalue())
        self.assertFalse(obj["trainingExecuted"])
        self.assertFalse(obj["modelLoaded"])
        self.assertFalse(obj["historicalWeightsModified"])

    def test_registers_synthetic_checkpoint_chain_and_comparison(self):
        record = self.audit()
        self.assertEqual(len(record["entries"]), 3)
        self.assertEqual(len(record["comparisons"]), 2)
        self.assertEqual(record["entries"][0]["entryId"], "m9-original")
        self.assertEqual(record["entries"][2]["entryId"], "m10-step-25")
        self.assertEqual(record["decisionLedger"][-1]["decision"], "not-evaluated")
        self.assertFalse(record["trainingExecutedByThisTool"])
        self.assertEqual(record["sourceApprovalStatus"],
                         "historical-approval-sha-declared-only")

    def test_reject_m10_weights_tampering(self):
        record = self.audit()
        self.assertEqual(len(record["entries"]), 3)
        (self.f["run"] / "step-10/weights.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(history.Refusal, "binding"):
            self.audit()

    def test_reject_modified_raw_predictions_even_with_matching_file_hash(self):
        response = self.f["run"] / "step-10/responses.jsonl"
        rows = [json.loads(s) for s in response.read_text(encoding="utf-8").splitlines()]
        rows[0]["output"] = "WRONG"
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
        response.write_bytes(raw)
        report = self.f["run"] / "step-10/report.json"
        item = json.loads(report.read_text(encoding="utf-8"))
        item["responsesSha256"] = sha(response)
        save(report, item)
        parent = self.f["run"] / "report.json"
        outer = json.loads(parent.read_text(encoding="utf-8"))
        outer["snapshots"][0] = item
        save(parent, outer)
        with self.assertRaisesRegex(worker.Refusal, "diverge"):
            self.audit()

    def test_missing_or_disordered_milestone_fails(self):
        parent = self.f["run"] / "report.json"
        record = json.loads(parent.read_text(encoding="utf-8"))
        record["snapshots"].reverse()
        save(parent, record)
        with self.assertRaisesRegex(history.Refusal, "order"):
            self.audit()

    def test_mismatched_parent_m9_report_rejected(self):
        parent = self.f["run"] / "report.json"
        record = json.loads(parent.read_text(encoding="utf-8"))
        record["originalM9WeightsSha256"] = "0" * 64
        save(parent, record)
        with self.assertRaisesRegex(history.Refusal, "scope"):
            self.audit()

    def test_approval_file_hash_checked_when_supplied(self):
        approval = self.root / "consent.json"
        approval.write_bytes(b"not the historical approval\n")
        with self.assertRaisesRegex(history.Refusal, "scope"):
            with patch.object(history.gate, "inspect_sources",
                              return_value=self.f["proof"]), \
                 patch.object(history, "ORIGINAL_M9_SHA",
                              self.f["proof"]["backup"]["m9WeightsSha256"]):
                history.audit(**self.scope, approval_file=approval)

    def test_create_only_local_registration(self):
        record = self.audit()
        output_root = self.root / "training/artifacts"
        output_root.mkdir(parents=True)
        with patch.object(history.research, "ROOT", self.root):
            res = history.register(output_root / "phase1-historical", record)
            self.assertEqual(res["checkpointCount"], 3)
            self.assertEqual(res["decisionCount"], 4)
            self.assertTrue((output_root / "phase1-historical/"
                             "experiment-decisions.json").is_file())
            with self.assertRaisesRegex(history.Refusal, "NEW"):
                history.register(output_root / "phase1-historical", record)

    def test_register_refuses_external_output(self):
        record = self.audit()
        with patch.object(history.research, "ROOT", self.root):
            (self.root / "training/artifacts").mkdir(parents=True)
            with self.assertRaisesRegex(history.Refusal, "NEW direct child"):
                history.register(self.root / "external-output", record)


if __name__ == "__main__":
    unittest.main()
