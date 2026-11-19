"""CPU-only checks for Phase 1 checkpoint evidence and matched M10 receipts."""
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from plex_training import plex600_m10_copy_proposal as proposal
from plex_training import plex600_phase1_registry as phase1


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def setup_artifacts(root: Path) -> dict:
    files = {
        "checkpointPath": "training/artifacts/m9/weights.pt",
        "modelSourcePath": "training/src/plex_training/plex600.py",
        "tokenizerManifestPath": "training/artifacts/tokenizer/manifest.json",
        "trainingReportPath": "training/artifacts/m9/report.json",
        "datasetManifestPath": None,
        "evaluationReportPath": None,
        "evaluationResponsesPath": None,
    }
    for key, name in files.items():
        if name is not None:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((key + "\n").encode("ascii"))
    spec = {
        "schemaVersion": 1, "entryId": "m9-original",
        "experimentId": "m9-r3", "modelFamily": "plex-600m",
        "checkpointKind": "model-only", "sourceCommit": "a" * 40,
        "optimizerUpdates": 8, "parentEntryId": None,
        "parentCheckpointSha256": None,
        **files,
    }
    write_json(root / "training/artifacts/spec.json", spec)
    (root / "training/artifacts/registry").mkdir(parents=True, exist_ok=True)
    return spec


def create_receipt(directory: Path, *, baseline: bool, bad: bool = False) -> None:
    rows = proposal.rows()
    policies = phase1.DECODERS
    checkpoint = ("a" if baseline else "b") * 64
    results = []
    for policy in policies:
        for case in rows:
            output = case["snippet"]
            if bad and case["id"] == rows[0]["id"] and policy == policies[0]:
                output = "wrong\n"
            results.append({
                "id": case["id"], "output": output, "terminationReason": "eos",
                "generatedTokens": 4, "language": case["language"],
                "split": case["split"], "decodingPolicy": policy,
                "promptSha256": hashlib.sha256(
                    proposal.prompt_for(case).encode("utf-8")).hexdigest(),
                "m9WeightsSha256" if baseline else "checkpointSha256": checkpoint,
            })
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in results)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "responses.jsonl").write_bytes(raw.encode("utf-8"))
    report = {
        "responsesSha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "m9WeightsSha256" if baseline else "checkpointSha256": checkpoint,
    }
    if baseline:
        report["protocol"] = "plex600-m10-m9-read-only-copy-baseline-v1"
        report["selectedCaseIds"] = [row["id"] for row in rows]
    else:
        report["scores"] = {}
    write_json(directory / "report.json", report)


class Phase1RegistryTests(unittest.TestCase):
    def test_plan_is_inert_and_default_has_no_gpu_or_training(self):
        out = StringIO()
        with redirect_stdout(out):
            result = phase1.main([])
        self.assertEqual(result, 0)
        self.assertFalse(json.loads(out.getvalue())["trainingAuthorized"])
        self.assertNotIn("torch", phase1.__dict__)

    def test_capture_audit_and_inventory_preserve_original_weights(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_artifacts(root)
            weights = root / "training/artifacts/m9/weights.pt"
            original = weights.read_bytes()
            manifest = "training/artifacts/registry/m9.json"
            record = phase1.capture(root, "training/artifacts/spec.json", manifest)
            self.assertEqual(record["files"]["checkpoint"]["sha256"],
                             hashlib.sha256(original).hexdigest())
            self.assertFalse(record["inferenceVerified"])
            self.assertTrue(phase1.verify(root, manifest)["verifiedFileHashes"])
            inventory = phase1.inventory(root, "training/artifacts/registry")
            self.assertEqual(inventory["entries"], 1)
            self.assertEqual(inventory["missingParentEntries"], [])
            self.assertEqual(weights.read_bytes(), original)
            with self.assertRaisesRegex(phase1.Refusal, "immutable"):
                phase1.capture(root, "training/artifacts/spec.json", manifest)

    def test_modified_checkpoint_and_spec_are_detected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_artifacts(root)
            manifest = "training/artifacts/registry/m9.json"
            phase1.capture(root, "training/artifacts/spec.json", manifest)
            weights = root / "training/artifacts/m9/weights.pt"
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(phase1.Refusal, "changed"):
                phase1.verify(root, manifest)
            weights.write_bytes(b"checkpointPath\n")
            spec = root / "training/artifacts/spec.json"
            spec.write_text(spec.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(phase1.Refusal, "spec changed"):
                phase1.verify(root, manifest)

    def test_parent_identity_and_hash_must_be_coherent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = setup_artifacts(root)
            spec["parentEntryId"] = "m7"
            with self.assertRaisesRegex(phase1.Refusal, "Parent entry and hash"):
                phase1._validate_spec(spec)
            spec["parentCheckpointSha256"] = "c" * 64
            phase1._validate_spec(spec)
            write_json(root / "training/artifacts/spec.json", spec)
            phase1.capture(root, "training/artifacts/spec.json",
                           "training/artifacts/registry/m9.json")
            inv = phase1.inventory(root, "training/artifacts/registry")
            self.assertEqual(inv["missingParentEntries"], ["m7"])

    def test_rejects_path_traversal_and_symlink_evidence(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = setup_artifacts(root)
            spec["checkpointPath"] = "../outside"
            write_json(root / "training/artifacts/spec.json", spec)
            with self.assertRaisesRegex(phase1.Refusal, "relative"):
                phase1.capture(root, "training/artifacts/spec.json",
                               "training/artifacts/registry/m9.json")
            spec["checkpointPath"] = "training/artifacts/m9/weights.pt"
            write_json(root / "training/artifacts/spec.json", spec)
            target = root / spec["checkpointPath"]
            target.unlink()
            try:
                target.symlink_to(root / "training/artifacts/spec.json")
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks unavailable")
            with self.assertRaisesRegex(phase1.Refusal, "symlinks"):
                phase1.capture(root, "training/artifacts/spec.json",
                               "training/artifacts/registry/m9.json")

    def test_refuses_unbound_evaluation_receipts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = setup_artifacts(root)
            spec["evaluationReportPath"] = "training/artifacts/m9/eval/report.json"
            spec["evaluationResponsesPath"] = "training/artifacts/m9/eval/responses.jsonl"
            evidence = root / "training/artifacts/m9/eval"
            evidence.mkdir(parents=True)
            (evidence / "responses.jsonl").write_bytes(b"{}\n")
            write_json(evidence / "report.json", {
                "checkpointSha256": "b" * 64,
                "responsesSha256": "a" * 64,
            })
            write_json(root / "training/artifacts/spec.json", spec)
            with self.assertRaisesRegex(phase1.Refusal, "bind"):
                phase1.capture(root, "training/artifacts/spec.json",
                               "training/artifacts/registry/m9.json")

    def test_rejects_duplicate_entry_names(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_artifacts(root)
            phase1.capture(root, "training/artifacts/spec.json",
                           "training/artifacts/registry/one.json")
            phase1.capture(root, "training/artifacts/spec.json",
                           "training/artifacts/registry/two.json")
            with self.assertRaisesRegex(phase1.Refusal, "Duplicate"):
                phase1.inventory(root, "training/artifacts/registry")


class Phase1ComparisonTests(unittest.TestCase):
    def test_exact_m10_receipts_have_zero_delta(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_receipt(root / "baseline", baseline=True)
            create_receipt(root / "candidate", baseline=False)
            result = phase1.compare_m10(root / "baseline", root / "candidate")
            self.assertTrue(result["comparable"])
            self.assertFalse(result["inferenceIndependentlyProven"])
            for value in result["scores"].values():
                self.assertEqual(value["deltaExactAndEos"], 0)
                self.assertEqual(value["cases"], 12 if "train" in next(
                    k for k, v in result["scores"].items() if v is value) else 6)

    def test_detects_regression_without_training_claims(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_receipt(root / "baseline", baseline=True)
            create_receipt(root / "candidate", baseline=False, bad=True)
            result = phase1.compare_m10(root / "baseline", root / "candidate")
            self.assertEqual(sum(s["deltaExactAndEos"] for s in result["scores"].values()), -1)
            self.assertFalse(result["trainingAuthorized"])

    def test_refuses_tampered_responses_and_duplicate_case(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_receipt(root / "baseline", baseline=True)
            create_receipt(root / "candidate", baseline=False)
            response = root / "candidate/responses.jsonl"
            response.write_bytes(response.read_bytes() + b"{}\n")
            with self.assertRaisesRegex(phase1.Refusal, "hash"):
                phase1.compare_m10(root / "baseline", root / "candidate")
            report = root / "candidate/report.json"
            entry = json.loads(report.read_text(encoding="utf-8"))
            entry["responsesSha256"] = hashlib.sha256(response.read_bytes()).hexdigest()
            write_json(report, entry)
            with self.assertRaises(phase1.Refusal):
                phase1.compare_m10(root / "baseline", root / "candidate")

    def test_refuses_changed_source_prompt_even_if_rehashed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_receipt(root / "baseline", baseline=True)
            create_receipt(root / "candidate", baseline=False)
            response = root / "candidate/responses.jsonl"
            rows = [json.loads(s) for s in response.read_text(encoding="utf-8").splitlines()]
            rows[0]["promptSha256"] = "f" * 64
            raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
            response.write_bytes(raw)
            report = root / "candidate/report.json"
            data = json.loads(report.read_text(encoding="utf-8"))
            data["responsesSha256"] = hashlib.sha256(raw).hexdigest()
            write_json(report, data)
            with self.assertRaisesRegex(phase1.Refusal, "mismatched"):
                phase1.compare_m10(root / "baseline", root / "candidate")


if __name__ == "__main__":
    unittest.main()
