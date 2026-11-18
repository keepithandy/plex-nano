"""CPU-only unit tests for exact M10 worker gates, schedule and refusal paths."""
from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from plex_training import plex600_m10_worker as worker
from plex_training import plex600_m10_copy_proposal as copy
from plex_training import plex600_m10_admission as gate
from plex_training import plex600_research as research


class MiniTokenizer:
    vocabulary_size = 1191

    def encode(self, text: str) -> list[int]:
        return [ord(c) + 4 for c in text]

    def decode(self, values: list[int]) -> str:
        return "".join(chr(c - 4) for c in values)


def scores(perfect: bool) -> dict:
    return {
        policy: {"allExactAndEos": perfect}
        for policy in worker.DECODERS
    }


def checkpoint_receipt(step: int, perfect: bool, digest: str) -> dict:
    return {
        "optimizerUpdatesCompleted": step,
        "checkpointSha256": digest,
        "candidateSha256": copy.audit()["candidateSha256"],
        "verifiedGeneration": True,
        "decodingPolicies": list(worker.DECODERS),
        "scores": scores(perfect),
    }


class M10WorkerTests(unittest.TestCase):
    def test_default_only_prints_plan_and_has_no_training_authorization(self):
        with patch.object(worker, "_admit",
                          side_effect=AssertionError("approval read")), \
             patch.object(worker.hw, "_check_device",
                          side_effect=AssertionError("CUDA inspected")), \
             redirect_stdout(StringIO()) as output:
            self.assertEqual(worker.main([]), 0)
        info = json.loads(output.getvalue())
        self.assertEqual(info["status"], "plan-only-no-training")
        self.assertFalse(info["trainingAuthorized"])
        self.assertFalse(info["modelLoaded"])
        self.assertFalse(info["checkpointWrites"])
        self.assertEqual(info["optimizerUpdatesCompleted"], 0)
        self.assertEqual(info["maxOptimizerUpdates"], 50)

    def test_exact_schedule_never_trains_heldout_and_limits_steps(self):
        schedule = worker.exact_schedule()
        self.assertEqual(len(schedule), 50)
        self.assertEqual([s["optimizerUpdate"] for s in schedule], list(range(1, 51)))
        self.assertEqual({s["recordIds"][0] for s in schedule[:10]},
                         {worker.PROBE_ID})
        self.assertEqual({s["recordIds"][0] for s in schedule[10:]},
                         {r["id"] for r in copy.rows() if r["split"] == "train"})
        self.assertFalse({s["recordIds"][0] for s in schedule}
                         & {r["id"] for r in copy.rows() if r["split"] == "heldout"})
        self.assertTrue(all(s["examplesInOptimizerUpdate"] == 1 for s in schedule))
        self.assertTrue(all(s["gradientAccumulationSteps"] == 1 for s in schedule))
        self.assertEqual(worker.SNAPSHOTS, (10, 25, 50))

    def test_training_pack_exact_eos_and_mask_for_only_twelve_rows(self):
        samples = worker.packaged_samples(MiniTokenizer())
        rows = {r["id"]: r for r in copy.rows() if r["split"] == "train"}
        self.assertEqual(set(samples), set(rows))
        self.assertEqual(len(samples), 12)
        for id_, record in rows.items():
            prefix = MiniTokenizer().encode(copy.prompt_for(record))
            answer = MiniTokenizer().encode(record["snippet"]) + [3]
            packed = samples[id_]
            self.assertEqual(packed["inputIds"], prefix + answer[:-1])
            self.assertEqual(packed["targetIds"],
                             [-100] * (len(prefix) - 1) + answer)
            self.assertEqual(packed["targetIds"][-1], 3)

    def test_early_stop_must_have_two_real_adjacent_perfect_milestones(self):
        ten = checkpoint_receipt(10, True, "a"*64)
        twentyfive = checkpoint_receipt(25, True, "b"*64)
        fifty = checkpoint_receipt(50, True, "c"*64)
        self.assertTrue(worker.checked_early_stop(ten, twentyfive))
        self.assertTrue(worker.checked_early_stop(twentyfive, fifty))
        self.assertFalse(worker.checked_early_stop(None, twentyfive))
        self.assertFalse(worker.checked_early_stop(ten, fifty))
        self.assertFalse(worker.checked_early_stop(ten, ten))
        for field, value in (
            ("verifiedGeneration", False),
            ("candidateSha256", "9"*64),
            ("checkpointSha256", ten["checkpointSha256"]),
            ("scores", scores(False)),
            ("decodingPolicies", ["unmasked-v1"]),
        ):
            with self.subTest(field=field):
                self.assertFalse(worker.checked_early_stop(
                    ten, {**twentyfive, field: value}))

    def test_cli_refuses_any_unauthorized_run_mode(self):
        with redirect_stderr(StringIO()):
            self.assertEqual(worker.main(["--execute-approved-run"]), 2)
            self.assertEqual(worker.main(["--m9-weights", "weights.pt"]), 2)
            with self.assertRaises(SystemExit) as raised:
                worker.main(["--train-without-approval"])
            self.assertEqual(raised.exception.code, 2)

    def test_external_gate_refusal_precedes_source_load_gpu_and_staging(self):
        paths = {
            name: Path(name) for name in (
                "bundle_dir", "m9_report", "m9_weights",
                "baseline_report", "baseline_responses",
                "approval_file", "output_dir",
            )
        }
        with patch.object(worker, "_admit",
                          side_effect=worker.Refusal("no owner approval")), \
             patch.object(worker, "_source_identity",
                          side_effect=AssertionError("source checked")), \
             patch.object(worker.hw, "_supervise",
                          side_effect=AssertionError("GPU subprocess launched")):
            with self.assertRaisesRegex(worker.Refusal, "no owner approval"):
                worker.run_approved(
                    paths, bundle_sha=research.BUNDLE_SHA, explicit_flag=True)
        with self.assertRaisesRegex(worker.Refusal, "explicit opt-in"):
            worker.run_approved(
                paths, bundle_sha=research.BUNDLE_SHA, explicit_flag=False)

    def test_parent_rejects_malformed_child_result_without_publishing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {
                "bundle_dir": root, "m9_report": root / "m9-report",
                "m9_weights": root / "m9-weights",
                "baseline_report": root / "baseline-report",
                "baseline_responses": root / "responses",
                "approval_file": root / "owner",
                "output_dir": root / "destination",
            }
            proof = {
                "m9WeightsSha256": "a"*64,
                "candidateSha256": "b"*64,
                "externalApprovalSha256": "c"*64,
                "scheduleSha256": "d"*64,
            }
            with patch.object(worker, "_admit", return_value=proof), \
                 patch.object(worker, "_source_identity", return_value={}), \
                 patch.object(worker.m8, "require_output",
                              return_value=paths["output_dir"]), \
                 patch.object(worker.hw, "_supervise",
                              return_value=(0, json.dumps({
                                  "protocol": worker.PROTOCOL,
                                  "status": "bounded-50-updates-complete-unreviewed",
                                  "optimizerUpdatesCompleted": 50,
                                  "identity": {},
                                  "originalM9WeightsSha256": "a"*64,
                                  "candidateSha256": "b"*64,
                                  "approvalSha256": "c"*64,
                                  "scheduleSha256": "d"*64,
                                  "finalHoldoutUsed": False,
                                  "historicalCheckpointWrites": False,
                                  "optimizerStateSaved": False,
                                  "snapshots": [],  # MUST be [10,25,50].
                              }), "", False)):
                with self.assertRaisesRegex(worker.Refusal, "progression"):
                    worker.run_approved(
                        paths, bundle_sha=research.BUNDLE_SHA, explicit_flag=True)
            self.assertFalse(paths["output_dir"].exists())
            self.assertEqual(list(root.glob(".destination.staging-*")), [])

    def test_parent_enforces_child_hard_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {name: root / name for name in (
                "bundle_dir", "m9_report", "m9_weights",
                "baseline_report", "baseline_responses",
                "approval_file", "output_dir",
            )}
            with patch.object(worker, "_admit", return_value={}), \
                 patch.object(worker, "_source_identity", return_value={}), \
                 patch.object(worker.m8, "require_output",
                              return_value=paths["output_dir"]), \
                 patch.object(worker.hw, "_supervise",
                              return_value=(-9, "", "", True)) as supervise:
                with self.assertRaisesRegex(worker.Refusal, "hard-timeout"):
                    worker.run_approved(
                        paths, bundle_sha=research.BUNDLE_SHA, explicit_flag=True)
                self.assertEqual(supervise.call_args.kwargs["timeout"], 3600)
            self.assertEqual(list(root.glob(".output_dir.staging-*")), [])

    def test_only_explicitly_admitted_child_imports_training_backend(self):
        source = Path(worker.__file__).read_text(encoding="utf-8")
        module = ast.parse(source)
        method = next(node for node in module.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_train_worker")
        admission_calls = [
            n for n in ast.walk(method) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "_admit"
        ]
        imports = [
            n for n in ast.walk(method) if isinstance(n, ast.Import)
            and any(alias.name == "torch" for alias in n.names)
        ]
        self.assertEqual(len(admission_calls), 1)
        self.assertEqual(len(imports), 1)
        self.assertLess(admission_calls[0].lineno, imports[0].lineno)
        self.assertNotIn("training/artifacts/plex600-m9-r3-first-eight/weights.pt",
                         source)

    def test_generation_helper_is_source_bound_not_early_stop_claim(self):
        self.assertEqual(worker.DECODERS, gate.DECODERS)
        self.assertEqual(worker.PROBE_ID, copy.WARMUP_RECORD_ID)
        self.assertEqual(worker.SEED, copy.SEED)
        self.assertEqual(worker.MAX_WEIGHTS, gate.CHECKPOINT_MAX_BYTES)


if __name__ == "__main__":
    unittest.main()
