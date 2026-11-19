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


def synthetic_checkpoint_responses(checkpoint_sha: str) -> tuple[str, dict]:
    """Test-only responses; no model generation and no training."""
    cases = copy.rows()
    result = []
    scored = {}
    for policy in worker.DECODERS:
        predictions = []
        for row in cases:
            predictions.append({
                "id": row["id"], "output": row["snippet"],
                "terminationReason": "eos",
            })
            result.append({
                "id": row["id"], "output": row["snippet"],
                "terminationReason": "eos", "generatedTokens": 1,
                "language": row["language"], "split": row["split"],
                "decodingPolicy": policy, "checkpointSha256": checkpoint_sha,
                "promptSha256": __import__("hashlib").sha256(
                    copy.prompt_for(row).encode("utf-8")).hexdigest(),
            })
        scored[policy] = copy.score_predictions(cases, predictions)
    return "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in result), scored


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
            for path_name, location in paths.items():
                if path_name not in ("bundle_dir", "output_dir"):
                    location.write_text("synthetic-path-only", encoding="utf-8")
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
            (root / "bundle_dir").mkdir()
            paths = {name: root / name for name in (
                "bundle_dir", "m9_report", "m9_weights",
                "baseline_report", "baseline_responses",
                "approval_file", "output_dir",
            )}
            for path_name, location in paths.items():
                if path_name not in ("bundle_dir", "output_dir"):
                    location.write_text("synthetic-path-only", encoding="utf-8")
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

    def test_timeout_preserves_only_verified_partial_and_never_labels_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / "stage"
            stage.mkdir()
            target = root / "output"
            first = stage / "step-10"
            first.mkdir()
            payload = first / "weights.pt"
            payload.write_bytes(b"cpu-test-only-not-a-model")
            digest = worker._sha(payload)
            responses = first / "responses.jsonl"
            response_text, scoring = synthetic_checkpoint_responses(digest)
            responses.write_bytes(response_text.encode("utf-8"))
            response_sha = research.digest_file(
                responses, limit=gate.RESPONSES_MAX_BYTES)
            proof = {"candidateSha256": "a"*64,
                     "m9WeightsSha256": "b"*64,
                     "externalApprovalSha256": "c"*64}
            report = {
                "optimizerUpdatesCompleted": 10,
                "checkpointSha256": digest,
                "candidateSha256": proof["candidateSha256"],
                "checkpoint": {"sha256": digest,
                               "bytes": payload.stat().st_size},
                "verifiedGeneration": True,
                "decodingPolicies": list(worker.DECODERS),
                "scores": scoring,
                "responsesSha256": response_sha,
            }
            (first / "report.json").write_text(
                json.dumps(report), encoding="utf-8")
            torn = stage / "step-25"
            torn.mkdir()
            (torn / "weights.pt").write_bytes(b"incomplete-copy")
            self.assertTrue(worker.preserve_verified_partial(
                stage, target, proof, failure_reason="hard-timeout"))
            interruption = json.loads((target / "interruption.json").read_text())
            self.assertEqual(interruption["lastVerifiedMilestone"], 10)
            self.assertFalse(interruption["trainingCompleted"])
            self.assertFalse(interruption["resumeAuthorized"])
            self.assertIn("not-training-success", interruption["status"])
            self.assertFalse(stage.exists())
            self.assertFalse((target / "step-25").exists())
            with self.assertRaisesRegex(worker.Refusal, "occupied"):
                worker.preserve_verified_partial(
                    target, target, proof, failure_reason="hard-timeout")

    def test_incomplete_partial_checkpoint_is_not_published_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / "stage"
            (stage / "step-10").mkdir(parents=True)
            (stage / "step-10" / "weights.pt").write_bytes(b"incomplete")
            proof = {"candidateSha256": "a"*64,
                     "m9WeightsSha256": "b"*64,
                     "externalApprovalSha256": "c"*64}
            self.assertFalse(worker.preserve_verified_partial(
                stage, root / "output", proof, failure_reason="worker-error"))
            self.assertFalse((root / "output").exists())

    def test_probe_separates_prompt_boundary_from_completed_answer(self):
        import torch
        from unittest.mock import Mock

        # One masked position, two content tokens, then EOS.
        packed = {"inputIds": [1, 2, 4, 5], "targetIds": [-100, 4, 5, 3],
                  "lossMaskPromptPositions": 1, "supervisedAnswerPositions": 3}
        logits = torch.zeros(1, 4, 6)
        logits[0, 1, 4] = 2
        logits[0, 1, 3] = 1
        logits[0, 3, 3] = 8
        model = Mock(return_value=(logits, torch.tensor(2.0)))
        with patch.object(torch, "autocast", return_value=torch.inference_mode()):
            result = worker.teacher_forced_probe(model, torch, packed, "cpu")
            self.assertEqual(result["firstPositionEosRank"], 2)
            self.assertEqual(result["eosTokenRank"], 1)
            self.assertFalse(result["firstPositionGreedyTokenIsEos"])
            self.assertLess(result["firstPositionEosProbability"], result["eosTokenProbability"])
            expected = -(logits[0, 1, :].log_softmax(0)[4]
                         + logits[0, 2, :].log_softmax(0)[5]) / 2
            self.assertAlmostEqual(result["teacherForcedAnswerOnlyNll"], expected.item())
            logits[0, 1, 3] = 3
            self.assertTrue(worker.teacher_forced_probe(
                model, torch, packed, "cpu")["firstPositionGreedyTokenIsEos"])

    def test_checkpoint_scores_are_recomputed_from_all_serialized_responses(self):
        with tempfile.TemporaryDirectory() as temp:
            responses = Path(temp) / "responses.jsonl"
            sha = "a" * 64
            response_text, scoring = synthetic_checkpoint_responses(sha)
            responses.write_bytes(response_text.encode("utf-8"))
            evidence = worker.verify_checkpoint_responses(responses, sha, scoring)
            self.assertEqual(evidence["responseCountVerified"], 36)
            self.assertEqual(evidence["sourceCheckpointSha256"], sha)
            broken_scores = json.loads(json.dumps(scoring))
            broken_scores["unmasked-v1"]["train"]["exact"] = 0
            with self.assertRaisesRegex(worker.Refusal, "diverge"):
                worker.verify_checkpoint_responses(responses, sha, broken_scores)
            lines = [json.loads(row) for row in response_text.splitlines()]
            lines[0]["checkpointSha256"] = "b" * 64
            responses.write_bytes(("".join(
                json.dumps(row) + "\n" for row in lines)).encode("utf-8"))
            with self.assertRaisesRegex(worker.Refusal, "mismatched"):
                worker.verify_checkpoint_responses(responses, sha, scoring)
            lines[0]["checkpointSha256"] = sha
            lines[-1]["id"] = lines[0]["id"]
            responses.write_bytes(("".join(
                json.dumps(row) + "\n" for row in lines)).encode("utf-8"))
            with self.assertRaises(worker.Refusal):
                worker.verify_checkpoint_responses(responses, sha, scoring)

    def test_uncommitted_training_worker_code_is_rejected(self):
        from types import SimpleNamespace
        with patch.object(worker.subprocess, "run",
                          return_value=SimpleNamespace(stdout=" M training/src/plex_training/plex600_m10_worker.py\n")):
            with self.assertRaisesRegex(worker.Refusal, "locally modified"):
                worker._source_identity()

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

    def test_warmup_comparison_tracks_actual_copy_progress_without_claiming_code_editing(self):
        rows = copy.rows()
        target = next(x for x in rows if x["id"] == worker.PROBE_ID)["snippet"]
        before = []
        after_scores = {}
        for policy in worker.DECODERS:
            before.append({"id": worker.PROBE_ID, "decodingPolicy": policy,
                           "output": " " * 96, "terminationReason": "max-new-tokens"})
            predictions = [
                {"id": r["id"], "output": r["snippet"],
                 "terminationReason": "eos"}
                for r in rows
            ]
            after_scores[policy] = copy.score_predictions(rows, predictions)
        result = worker.warmup_comparison(before, after_scores)
        self.assertEqual(set(result), set(worker.DECODERS))
        for val in result.values():
            self.assertFalse(val["beforeExactAndEos"])
            self.assertTrue(val["afterExactAndEos"])
            self.assertEqual(val["afterCharacterSimilarity"], 1.0)
            self.assertGreater(val["characterSimilarityDelta"], 0)
        bad = before[:-1]
        with self.assertRaisesRegex(worker.Refusal, "same exact warmup"):
            worker.warmup_comparison(bad, after_scores)

    def test_snapshot_reload_preserves_live_parameter_and_optimizer_continuity(self):
        import torch
        from plex_training.plex600 import Plex600Config, Plex600Model

        config = Plex600Config(
            vocab_size=260, context_length=12, width=16, layers=1,
            attention_heads=2, key_value_heads=1, feed_forward_width=32)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(41)
            model = Plex600Model(config)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=5e-5, eps=1e-8,
                weight_decay=0, foreach=False, fused=False)
            ids = torch.tensor([[4, 7, 8, 9, 4]], dtype=torch.long)
            targets = torch.tensor([[-100, -100, 7, 8, 3]], dtype=torch.long)
            model.train()
            _, loss = model(ids, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            original_params = tuple(model.parameters())
            original_param_ids = tuple(id(p) for p in original_params)
            original_values = [p.detach().clone() for p in original_params]
            embedding = model.token_embedding.weight
            initial_adam_step = optimizer.state[embedding]["step"].item()
            with tempfile.TemporaryDirectory() as temp:
                checkpoint = Path(temp) / "weights.pt"
                with patch.object(model, "to",
                                  side_effect=AssertionError("live model moved")):
                    receipt = worker._reload_parity(
                        model, torch, checkpoint, [4, 7, 8], config)
                self.assertTrue(checkpoint.is_file())
                self.assertEqual(receipt["sha256"], worker._sha(checkpoint))
                self.assertTrue(receipt["reloadedGreedyTokensMatch"])
                self.assertLessEqual(
                    receipt["maxReloadLogitAbsoluteDifference"], 0.0001)
                self.assertTrue(receipt["liveModelParameterDevicePreserved"])
                self.assertTrue(receipt["liveModelParameterIdentityPreserved"])
                self.assertEqual(
                    tuple(id(p) for p in model.parameters()), original_param_ids)
                self.assertTrue(all(torch.equal(p, previous)
                                    for p, previous in zip(model.parameters(),
                                                          original_values)))
                with self.assertRaisesRegex(worker.Refusal, "create-only"):
                    worker._reload_parity(
                        model, torch, checkpoint, [4, 7, 8], config)

            model.train()
            _, next_loss = model(ids, targets)
            next_loss.backward()
            optimizer.step()
            self.assertEqual(
                optimizer.state[embedding]["step"].item(), initial_adam_step + 1)
            self.assertFalse(torch.equal(
                embedding, original_values[0]))

    def test_generation_helper_is_source_bound_not_early_stop_claim(self):
        self.assertEqual(worker.DECODERS, gate.DECODERS)
        self.assertEqual(worker.PROBE_ID, copy.WARMUP_RECORD_ID)
        self.assertEqual(worker.SEED, copy.SEED)
        self.assertEqual(worker.MAX_WEIGHTS, gate.CHECKPOINT_MAX_BYTES)


if __name__ == "__main__":
    unittest.main()
