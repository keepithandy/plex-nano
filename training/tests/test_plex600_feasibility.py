"""Runner guards and lifecycle only: no full model or real CUDA access."""

from contextlib import nullcontext, redirect_stdout
from io import StringIO
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from plex_training import plex600_feasibility as check


class Plex600FeasibilityTests(unittest.TestCase):
    def test_plan_and_import_do_not_load_torch_or_launch_worker(self):
        code = (
            "import sys; from plex_training import plex600_feasibility as c; "
            "assert 'torch' not in sys.modules; c.main([]); "
            "assert 'torch' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, check=True, timeout=15)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["status"], "plan")
        self.assertEqual(plan["parameterCount"], 600_372_480)
        self.assertEqual(plan["optimizerUpdates"], 0)
        self.assertEqual(plan["supervisedTargets"], 1023)
        self.assertFalse(plan["artifactWrites"])

    def test_execute_refuses_before_source_or_process_without_explicit_opt_in(self):
        with patch.object(check, "source_identity") as source, patch.object(check, "_supervise") as run:
            for value in (False, None, 1, "yes"):
                self.assertEqual(check.execute(0, authorized=value)["status"], "refused")
            source.assert_not_called()
            run.assert_not_called()

    def test_bad_device_index_is_rejected(self):
        for value in (-1, True, 1.5, "0"):
            with self.subTest(value=value), self.assertRaises(check.Refusal):
                check.plan(value)

    def test_source_mismatch_or_dirty_source_prevents_worker(self):
        path = MagicMock()
        path.read_bytes.return_value = b"changed"
        with patch.object(check, "MODEL_PATH", path), patch.object(check, "_supervise") as run:
            report = check.execute(0, authorized=True)
            self.assertEqual(report["status"], "refused")
            self.assertIn("SHA-256", report["error"])
            run.assert_not_called()
        with patch.object(check.subprocess, "run") as git:
            git.return_value.stdout = " M training/src/plex_training/plex600.py\n"
            with self.assertRaisesRegex(check.Refusal, "modifications"):
                check.source_identity()

    def test_actual_pinned_source_and_git_identity_are_consistent(self):
        report = check.source_identity()
        self.assertTrue(report["modelSourceClean"])
        self.assertEqual(report["modelSha256"], check.MODEL_SHA256)
        self.assertEqual(len(report["gitCommit"]), 40)
        self.assertEqual(len(report["runnerSha256"]), 64)

    def test_direct_worker_invocation_refuses_without_torch_import(self):
        env = dict(os.environ)
        env.pop(check._CONTROL_ENV, None)
        code = (
            "import sys; from plex_training import plex600_feasibility as c; "
            "c._worker_entry(); assert 'torch' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], input='{}', env=env,
                                capture_output=True, text=True, check=True, timeout=15)
        self.assertEqual(json.loads(result.stdout)["status"], "refused")

    @staticmethod
    def fake_torch():
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4080 SUPER"
        torch.cuda.is_bf16_supported.return_value = True
        torch.cuda.mem_get_info.return_value = (15_000 * 1024**2, 16_376 * 1024**2)
        torch.device.side_effect = lambda *args: nullcontext() if args == ("cpu",) else "cuda:0"
        torch.autocast.side_effect = lambda **kwargs: nullcontext()
        torch.__version__ = "test-only"
        torch.version.cuda = "test-only"
        torch.cuda.max_memory_allocated.return_value = 4_000_000
        torch.cuda.max_memory_reserved.return_value = 5_000_000
        return torch

    def test_device_refusals_prevent_model_construction(self):
        variants = [
            ("cuda.is_available.return_value", False, "unavailable"),
            ("cuda.device_count.return_value", 0, "does not exist"),
            ("cuda.get_device_name.return_value", "Other GPU", "not the proposed"),
            ("cuda.is_bf16_supported.return_value", False, "BF16"),
            ("cuda.mem_get_info.return_value", (check.MIN_FREE_BYTES - 1, 17_000_000_000), "14,000"),
        ]
        for key, value, message in variants:
            torch = self.fake_torch()
            torch.configure_mock(**{key: value})
            module = self.fake_model(torch)
            with self.subTest(key=key), self.assertRaisesRegex(check.Refusal, message):
                check._workload(torch, module, check.plan())
            module.Plex600Model.assert_not_called()

    @staticmethod
    def fake_model(torch):
        module = MagicMock()
        module.Plex600Config.return_value.to_dict.return_value = dict(check.CONFIGURATION)
        module.Plex600Config.return_value.parameter_count.return_value = check.PARAMETERS
        module.parameter_count.return_value = check.PARAMETERS
        parameter = MagicMock()
        parameter.dtype = torch.float32
        parameter.grad.dtype = torch.float32
        model = module.Plex600Model.return_value
        model.parameters.return_value = [parameter] * check.PARAMETER_TENSORS
        logits = MagicMock()
        logits.shape = (1, 1024, 32768)
        loss = MagicMock()
        loss.ndim = 0
        loss.item.return_value = 3.5
        model.return_value = (logits, loss)
        return module

    def test_fixed_workload_calls_forward_backward_once_and_records_measurements(self):
        torch = self.fake_torch()
        module = self.fake_model(torch)
        report = check.plan()
        with patch.object(check.subprocess, "run") as driver:
            driver.return_value.stdout = "610.60\n"
            check._workload(torch, module, report)
        model = module.Plex600Model.return_value
        self.assertEqual(model.call_count, 1)
        model.return_value[1].backward.assert_called_once_with()
        model.gradient_checkpointing_enable.assert_called_once_with()
        torch.cuda.set_per_process_memory_fraction.assert_called_once_with(0.8, "cuda:0")
        torch.arange.assert_called_once_with(1024, device="cuda:0", dtype=torch.long)
        torch.autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)
        torch.cuda.mem_get_info.assert_called_with(0)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["loss"], 3.5)
        self.assertTrue(report["allGradientsPresentFinite"])
        self.assertGreater(report["peakReservedBytes"], 0)

    def test_memory_drop_before_transfer_aborts_without_forward(self):
        torch = self.fake_torch()
        torch.cuda.mem_get_info.side_effect = [
            (check.MIN_FREE_BYTES + 1, 17_000_000_000),
            (check.MIN_FREE_BYTES - 1, 17_000_000_000),
        ]
        module = self.fake_model(torch)
        with patch.object(check.subprocess, "run") as driver:
            driver.return_value.stdout = "610.60\n"
            with self.assertRaisesRegex(RuntimeError, "dropped"):
                check._workload(torch, module, check.plan())
        module.Plex600Model.return_value.to.assert_not_called()
        module.Plex600Model.return_value.assert_not_called()

    def test_nonfinite_loss_and_bad_gradient_cannot_pass(self):
        for failure in ("loss", "missing-gradient", "nonfinite-gradient"):
            torch = self.fake_torch()
            module = self.fake_model(torch)
            if failure == "loss":
                torch.isfinite.return_value = False
            elif failure == "missing-gradient":
                module.Plex600Model.return_value.parameters.return_value[0].grad = None
            else:
                torch.isfinite.return_value.all.return_value = False
            with self.subTest(failure=failure), patch.object(check.subprocess, "run") as driver:
                driver.return_value.stdout = "610.60\n"
                with self.assertRaises(RuntimeError):
                    check._workload(torch, module, check.plan())
            if failure == "loss":
                module.Plex600Model.return_value.return_value[1].backward.assert_not_called()

    def test_real_supervisor_terminates_and_reaps_timed_out_process(self):
        code, stdout, stderr, expired = check._supervise(
            [sys.executable, "-c", "import time; time.sleep(30)"], {}, dict(os.environ), timeout=0.15,
        )
        self.assertTrue(expired)
        self.assertNotEqual(code, 0)

    def test_parent_interruption_kills_and_reaps_worker(self):
        worker = MagicMock()
        worker.communicate.side_effect = [KeyboardInterrupt(), ("", "")]
        with patch.object(check.subprocess, "Popen", return_value=worker):
            with self.assertRaises(KeyboardInterrupt):
                check._supervise(["test-only"], {}, {})
        worker.kill.assert_called_once_with()
        self.assertEqual(worker.communicate.call_count, 2)

    def test_parent_rejects_bad_output_failure_and_timeout_without_retry(self):
        for process_result, expected in [
            ((-9, "", "", True), "timeout"),
            ((1, "", "error", False), "inconclusive"),
            ((0, "not-json", "", False), "inconclusive"),
            ((0, '{"status":"pass"}', "", False), "inconclusive"),
        ]:
            with self.subTest(expected=expected), patch.object(check, "source_identity", return_value={}), \
                    patch.object(check, "_supervise", return_value=process_result) as run:
                report = check.execute(0, authorized=True)
                self.assertEqual(report["status"], expected)
                run.assert_called_once()

    def test_matching_failure_is_preserved_and_missing_pass_measurements_rejected(self):
        failure = dict(check.plan(), status="failed", source={}, error="CUDA out of memory")
        with patch.object(check, "source_identity", return_value={}), patch.object(
            check, "_supervise", return_value=(0, json.dumps(failure), "warning", False),
        ):
            report = check.execute(0, authorized=True)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], "CUDA out of memory")
        self.assertEqual(report["workerStderr"], "warning")
        self.assertFalse(check._valid_result(dict(failure, status="pass"), 0, {}))

    def test_pass_requires_finite_measurements_and_environment_provenance(self):
        result = dict(check.plan(), status="pass", source={},
                      actualParameterCount=check.PARAMETERS, outputShape=[1, 1024, 32768],
                      allGradientsPresentFinite=True, gradientTensorCount=check.PARAMETER_TENSORS,
                      loss=3.5, initializationSeconds=2., forwardSeconds=1., backwardSeconds=1.,
                      workerSeconds=5., peakAllocatedBytes=100, peakReservedBytes=200,
                      environment=dict(python="test", platform="test", torch="test", cudaRuntime="test"),
                      gpu=dict(deviceName="NVIDIA GeForce RTX 4080 SUPER", driverVersion="test",
                               freeBytesAtPreflight=check.MIN_FREE_BYTES,
                               freeBytesBeforeAllocation=check.MIN_FREE_BYTES,
                               totalBytes=16_376 * 1024**2))
        self.assertTrue(check._valid_result(result, 0, {}))
        for key, value in [("loss", float("nan")), ("backwardSeconds", -1),
                           ("workerSeconds", 601), ("peakReservedBytes", None),
                           ("gradientTensorCount", 1), ("environment", {})]:
            with self.subTest(key=key):
                self.assertFalse(check._valid_result(dict(result, **{key: value}), 0, {}))

    def test_authenticated_worker_records_oom_as_failure_without_retry(self):
        import types
        torch = self.fake_torch()
        model = types.ModuleType("plex_training.plex600")

        def oom(torch_arg, model_arg, report):
            report.update(gpu={}, phase="backward")
            raise RuntimeError("CUDA out of memory")

        control = json.dumps({"nonce": "test-only", "deviceIndex": 0})
        output = StringIO()
        with patch.dict(os.environ, {check._CONTROL_ENV: "test-only"}), \
                patch.dict(sys.modules, {"torch": torch, "plex_training.plex600": model}), \
                patch.object(check, "source_identity", return_value={}), \
                patch.object(check, "_workload", side_effect=oom) as work, \
                patch.object(sys, "stdin", StringIO(control)), redirect_stdout(output):
            check._worker_entry()
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["phase"], "backward")
        self.assertEqual(report["error"], "CUDA out of memory")
        self.assertEqual(report["peakReservedBytes"], 5_000_000)
        work.assert_called_once()

    def test_cli_failure_returns_nonzero_exit_code(self):
        with patch.object(check, "execute", return_value={"status": "refused"}) as run, \
                redirect_stdout(StringIO()):
            self.assertEqual(check.main(["--execute-cuda-check", "--device", "0"]), 2)
        run.assert_called_once_with(0, authorized=True)


if __name__ == "__main__":
    unittest.main()
