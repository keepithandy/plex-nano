"""Portable guards for Plex 600M one-update feasibility (no GPU runner in CI)."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from plex_training import plex600_optimizer_feasibility as check


class Plex600OptimizerFeasibilityTests(unittest.TestCase):
    def test_default_plan_is_lightweight_and_does_not_import_torch(self):
        snippet = (
            "import sys; from plex_training import plex600_optimizer_feasibility as c;"
            " assert 'torch' not in sys.modules;"
            " assert c.plan()['status']=='plan';"
            " assert 'torch' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", snippet],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fixed_contract_is_one_update_and_synthetic(self):
        plan = check.plan()
        self.assertEqual(plan["updatesPlanned"], 1)
        self.assertEqual(plan["optimizer"], "AdamW")
        self.assertFalse(plan["optimizerConfiguration"]["foreach"])
        self.assertFalse(plan["optimizerConfiguration"]["fused"])
        self.assertEqual(plan["parameterCount"], 600_372_480)
        self.assertEqual(plan["sequenceLength"], 1024)
        self.assertEqual(plan["input"], "synthetic-sequential-token-ids-only")
        self.assertFalse(plan["datasetAccess"])
        self.assertFalse(plan["checkpointWrites"])
        self.assertFalse(plan["finalHoldoutAccess"])

    def test_execution_requires_explicit_authorization_and_no_worker(self):
        with patch.object(check, "source_identity", side_effect=AssertionError("source access")):
            result = check.execute(0)
        self.assertEqual(result["status"], "refused")
        self.assertIn("Explicit", result["error"])

    def test_invalid_device_is_rejected_before_source_access(self):
        for device in (-1, True, "0", 0.5):
            with self.subTest(device=device), self.assertRaises(check.prior.Refusal):
                check.plan(device)

    def test_validated_source_identity_propagates(self):
        mock = {"gitCommit": "a" * 40, "modelSourceClean": True,
                "modelSha256": "b" * 64, "runnerSha256": "c" * 64}
        with patch.object(check.prior, "source_identity", return_value=mock), \
             patch.object(check.subprocess, "run") as run:
            run.return_value.stdout = ""
            identity = check.source_identity()
        self.assertEqual(identity["gitCommit"], "a" * 40)
        self.assertEqual(len(identity["optimizerRunnerSha256"]), 64)
        self.assertTrue(run.called)

    def test_dirty_optimizer_runner_is_refused(self):
        with patch.object(check.prior, "source_identity", return_value={}), \
             patch.object(check.subprocess, "run") as run:
            run.return_value.stdout = " M training/src/plex_training/plex600_optimizer_feasibility.py"
            with self.assertRaisesRegex(check.prior.Refusal, "uncommitted changes"):
                check.source_identity()

    def test_untrusted_fake_gpu_pass_is_inconclusive(self):
        identity = {"gitCommit": "a" * 40}
        fake = dict(check.plan(), status="pass", source=identity,
                    gradientsFinite=True, stateFinite=True)
        with patch.object(check, "source_identity", return_value=identity), \
             patch.object(check.prior, "_supervise", return_value=(
                 0, json.dumps(fake), "", False)):
            result = check.execute(0, authorized=True)
        self.assertEqual(result["status"], "inconclusive")

    def test_timeout_terminates_untrusted_worker(self):
        with patch.object(check, "source_identity", return_value={}), \
             patch.object(check.prior, "_supervise", return_value=(
                 -9, "", "killed", True)):
            result = check.execute(0, authorized=True)
        self.assertEqual(result["status"], "timeout")

    def test_failed_worker_without_trust_is_not_reported_as_pass(self):
        with patch.object(check, "source_identity", return_value={}), \
             patch.object(check.prior, "_supervise", return_value=(
                 1, "", "out-of-memory", False)):
            result = check.execute(0, authorized=True)
        self.assertEqual(result["status"], "inconclusive")
        self.assertNotEqual(result["status"], "pass")

    def test_optimizer_state_verifier_accepts_real_tiny_cpu_update(self):
        import torch
        p = torch.nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float32))
        opt = torch.optim.AdamW([p], lr=0.01, weight_decay=0, foreach=False, fused=False)
        (p.square().sum()).backward()
        before = p.detach().clone()
        opt.step()
        report = check.check_optimizer_state(opt, [p], torch)
        self.assertEqual(report["stateTensorsVerified"], 2)
        self.assertTrue(report["stateFinite"])
        self.assertFalse(torch.equal(before, p))

    def test_optimizer_state_verifier_rejects_corrupt_moment(self):
        import torch
        p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        opt = torch.optim.AdamW([p], lr=0.01, foreach=False, fused=False)
        (p.square().sum()).backward()
        opt.step()
        opt.state[p]["exp_avg"][0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            check.check_optimizer_state(opt, [p], torch)

    def test_plan_report_writes_create_only_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(check.main(["--output", str(path)]), 0)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "plan")
            with redirect_stdout(StringIO()), self.assertRaises(SystemExit):
                check.main(["--output", str(path)])

    def test_unauthorized_worker_never_constructs_model(self):
        result = subprocess.run(
            [sys.executable, "-m", "plex_training.plex600_optimizer_feasibility", "--_worker"],
            input="{}", capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertNotEqual(payload["status"], "pass")
        self.assertNotIn("gpu", payload)


if __name__ == "__main__":
    unittest.main()
