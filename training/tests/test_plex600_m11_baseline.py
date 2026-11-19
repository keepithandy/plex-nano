"""CPU-only M11 read-only M9 baseline checks, no model weights or CUDA."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout, redirect_stderr
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from plex_training import plex600_m11_baseline as baseline
from plex_training import plex600_m11_admission as admission
from plex_training import plex600_m11_source_conditioning as m11
from plex_training import plex600_m11_training_core as core
from plex_training import plex600_research as research


class TinyTokenizer:
    vocabulary_size = 1191

    def encode(self, value: str) -> list[int]:
        return [ord(ch) + 4 for ch in value]

    def decode(self, values: list[int]) -> str:
        return "".join(chr(t - 4) for t in values)


def synthetic_source() -> dict:
    return {
        "m9WeightsSha256": "a" * 64,
        "m9ReportSha256": "b" * 64,
        "bundleManifestSha256": research.BUNDLE_SHA,
        "tokenizerSha256": admission.m10_gate.TOKENIZER_SHA256,
        "candidateSha256": core.candidate_sha256(),
        "scheduleSha256": core.schedule_sha256(),
        "sourceHashes": {
            "modelSourceSha256": "c" * 64,
            "m11CandidateSourceSha256": "d" * 64,
            "m11TrainingCoreSourceSha256": "e" * 64,
            "m11AdmissionSourceSha256": "f" * 64,
        },
    }


def fake_observations(*, perfect: bool = True) -> list[dict]:
    data = []
    for policy in core.DECODERS:
        for case in m11.cases():
            data.append({
                "id": case["id"],
                "split": case["split"],
                "decodingPolicy": policy,
                "output": case["snippet"] if perfect else "= 0;\n",
                "terminationReason": "eos",
                "generatedTokens": 10,
                "promptSha256": hashlib.sha256(
                    m11.prompt_for(case).encode("utf-8")).hexdigest(),
                "checkpointSha256": "a" * 64,
            })
    return data


def audited_source(source: dict) -> dict:
    return {
        **source,
        "protocol": baseline.PROTOCOL,
        "status": "source-and-tokenizer-verified-no-model",
        "checkpointSha256": source["m9WeightsSha256"],
        "selectedCaseIds": [case["id"] for case in m11.cases()],
        "decodingPolicies": list(core.DECODERS),
        "responseCount": 8,
        "optimizerUpdates": 0,
        "trainingAuthorized": False,
        "modelLoaded": False,
        "checkpointWrites": False,
        "artifactWrites": False,
        "finalHoldoutUsed": False,
    }


class M11M9ReadOnlyBaselineTests(unittest.TestCase):
    def test_default_does_not_inspect_sources_or_load_model(self):
        with patch.object(baseline, "audit_inputs",
                          side_effect=AssertionError("touched source")), \
             patch.object(baseline, "run_baseline",
                          side_effect=AssertionError("started GPU")), \
             redirect_stdout(StringIO()) as out:
            self.assertEqual(baseline.main([]), 0)
        value = json.loads(out.getvalue())
        self.assertEqual(value["responseCount"], 8)
        self.assertEqual(value["decodingPolicies"], list(core.DECODERS))
        self.assertFalse(value["modelLoaded"])
        self.assertFalse(value["checkpointWrites"])
        self.assertFalse(value["trainingAuthorized"])
        self.assertEqual(value["optimizerUpdates"], 0)
        self.assertNotIn("torch", baseline.__dict__)

    def test_source_audit_uses_exact_original_m11_gate_and_space_prompts(self):
        source = synthetic_source()
        with patch.object(baseline.admission, "inspect_sources",
                          return_value=source) as inspect, \
             patch("plex_training.tokenizer.PlexTokenizer.load",
                   return_value=TinyTokenizer()):
            proof = baseline.audit_inputs(
                Path("bundle"), research.BUNDLE_SHA, Path("report"),
                Path("weights"))
        inspect.assert_called_once()
        self.assertEqual(proof["checkpointSha256"], "a" * 64)
        self.assertEqual(proof["responseCount"], 8)
        self.assertEqual(proof["selectedCaseIds"],
                         [row["id"] for row in m11.cases()])
        self.assertEqual([r["split"] for r in proof["promptBudget"]],
                         ["train", "train", "probe-only", "probe-only"])
        for item, case in zip(proof["promptBudget"], m11.cases()):
            self.assertTrue(m11.prompt_for(case).endswith("Copy: "))
            self.assertNotEqual(
                item["promptSha256"],
                hashlib.sha256(
                    baseline.admission.m10_copy.prompt_for(case).encode("utf-8")
                ).hexdigest())
        self.assertFalse(proof["modelLoaded"])
        self.assertFalse(proof["trainingAuthorized"])

    def test_sealing_eight_exact_responses_accepts_existing_m11_gate(self):
        with TemporaryDirectory() as tmp:
            src = audited_source(synthetic_source())
            result = baseline._write_receipts(
                Path(tmp), src, fake_observations(),
                peak_cuda_reserved=1234, elapsed_seconds=1.25)
            self.assertEqual(result["protocol"], admission.BASELINE_PROTOCOL)
            self.assertEqual(result["status"], admission.BASELINE_STATUS)
            self.assertEqual(result["optimizerUpdates"], 0)
            self.assertFalse(result["trainingAuthorized"])
            self.assertFalse(result["checkpointWrites"])
            self.assertFalse(result["finalHoldoutUsed"])
            for policy in core.DECODERS:
                self.assertTrue(result["scores"][policy]["trainingPairExactAndEos"])
                self.assertTrue(result["scores"][policy]["developmentPairExactAndEos"])
            receipt = admission.read_baseline(
                Path(tmp) / "report.json", Path(tmp) / "responses.jsonl",
                synthetic_source())
            self.assertTrue(receipt["baselineReceiptConsistencyVerified"])
            self.assertFalse(receipt["baselineInferenceIndependentlyReproduced"])
            self.assertEqual(len((Path(tmp) / "responses.jsonl").read_text(
                encoding="utf-8").splitlines()), 8)

    def test_constant_wrong_output_is_not_mistaken_for_source_learning(self):
        with TemporaryDirectory() as tmp:
            result = baseline._write_receipts(
                Path(tmp), audited_source(synthetic_source()),
                fake_observations(perfect=False), peak_cuda_reserved=0,
                elapsed_seconds=0.5)
            for policy in core.DECODERS:
                self.assertFalse(result["scores"][policy]["trainingPairExactAndEos"])
                self.assertFalse(result["scores"][policy]["developmentPairExactAndEos"])

    def test_reject_duplicates_missing_cases_and_wrong_checkpoint(self):
        source = audited_source(synthetic_source())
        observations = fake_observations()
        with self.assertRaisesRegex(baseline.Refusal, "exactly eight"):
            baseline._receipt_rows(source, observations[:-1])
        corrupted = [dict(x) for x in observations]
        corrupted[1]["id"] = corrupted[0]["id"]
        with self.assertRaisesRegex(baseline.Refusal, "duplicated"):
            baseline._receipt_rows(source, corrupted)
        corrupted = [dict(x) for x in observations]
        corrupted[0]["checkpointSha256"] = "0" * 64
        with self.assertRaisesRegex(baseline.Refusal, "checkpoint changed"):
            baseline._receipt_rows(source, corrupted)
        corrupted = [dict(x) for x in observations]
        corrupted[0]["split"] = "probe-only"
        with self.assertRaises(baseline.Refusal):
            baseline._receipt_rows(source, corrupted)

    def test_reject_newline_prompts_and_mismatched_hashes(self):
        source = audited_source(synthetic_source())
        corrupted = [dict(x) for x in fake_observations()]
        corrupted[0]["promptSha256"] = hashlib.sha256(
            baseline.admission.m10_copy.prompt_for(m11.cases()[0]).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(baseline.Refusal, "source/checkpoint"):
            baseline._receipt_rows(source, corrupted)

    def test_prevent_existing_output_before_any_original_checkpoint_hash(self):
        with patch.object(baseline.m8, "require_output",
                          side_effect=baseline.Refusal("occupied output")), \
             patch.object(baseline, "audit_inputs",
                          side_effect=AssertionError("opened checkpoint")), \
             self.assertRaisesRegex(baseline.Refusal, "occupied output"):
            baseline.run_baseline(Path("bundle"), research.BUNDLE_SHA,
                                  Path("report"), Path("weights"), Path("output"))

    def test_cpu_input_audit_no_output_and_training_flags_rejected(self):
        with redirect_stderr(StringIO()):
            self.assertEqual(baseline.main(["--audit-inputs"]), 2)
            self.assertEqual(baseline.main(["--run-read-only-baseline"]), 2)
            self.assertEqual(baseline.main(["--m9-weights", "weights.pt"]), 2)
            with self.assertRaises(SystemExit) as state:
                baseline.main(["--execute-approved-run"])
            self.assertEqual(state.exception.code, 2)
            with self.assertRaises(SystemExit) as state:
                baseline.main(["--audit-inputs", "--run-read-only-baseline"])
            self.assertEqual(state.exception.code, 2)

    def test_sealed_receipts_are_create_only(self):
        with TemporaryDirectory() as tmp:
            src = audited_source(synthetic_source())
            baseline._write_receipts(Path(tmp), src, fake_observations(),
                                     peak_cuda_reserved=0, elapsed_seconds=0.1)
            with self.assertRaises(FileExistsError):
                baseline._write_receipts(Path(tmp), src, fake_observations(),
                                         peak_cuda_reserved=0, elapsed_seconds=0.1)

    def test_source_has_no_training_or_checkpoint_save_calls(self):
        source = Path(baseline.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {ast.unparse(node.func) for node in ast.walk(tree)
                 if isinstance(node, ast.Call)}
        forbidden = {
            "torch.save", "optimizer.step", "model.train", "loss.backward",
            "torch.autograd.grad", "torch.enable_grad",
        }
        self.assertFalse(forbidden & calls)
        self.assertNotIn("--execute-approved-run", source)
        self.assertEqual(baseline.DECODERS, admission.DECODERS)


if __name__ == "__main__":
    unittest.main()
