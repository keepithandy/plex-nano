"""Synthetic-only M12-A dataset audit checks. Tests never train or load model weights."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("plex_m12_data_audit.py")
spec = importlib.util.spec_from_file_location("plex_m12_data_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class M12DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="plex-m12-a-synthetic-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.data = {}
        for split, count in audit.EXPECTED_COUNTS.items():
            rows = []
            for i in range(count):
                base = ("fresh" if split == "development" else "familiar") + str(i).zfill(3)
                source = f"let {base} = {i};\n"
                rows.append({
                    "id": split + "-" + str(i),
                    "split": split,
                    "source": source,
                    "expected": source,
                    "instruction": "Copy this JavaScript exactly.",
                    "promptTemplate": ("copy-space" if split == "train" else
                                       ("copy-newline" if i % 2 == 0 else "console-general")),
                    "identifierFamily": "family-" + base,
                    "sourceTemplate": "source-" + base,
                    "provenanceReference": "synthetic-test-" + base,
                    "sourceOrigin": "owner-authored",  # synthetic tests exercise claimed fields only
                    "rightsClaim": "owner-claims-rights",
                    "sourceSha256": sha(source),
                    "expectedSha256": sha(source),
                    "trainingAuthorized": False,
                    "developmentSlices": (["unseen-identifier", "prompt-transfer"] if i < 16
                                         else ["unseen-identifier"] if i < 32 else [])
                    if split == "development" else [],
                })
            self.data[split] = rows
        self.write_all()

    def write_all(self):
        splits = {}
        for split in audit.EXPECTED_COUNTS:
            payload = "".join(json.dumps(row, ensure_ascii=True) + "\n"
                              for row in self.data[split]).encode("utf-8")
            (self.root / (split + ".jsonl")).write_bytes(payload)
            splits[split] = {"file": split + ".jsonl", "count": len(self.data[split]),
                             "sha256": hashlib.sha256(payload).hexdigest()}
        manifest = {
            "schema": audit.SCHEMA,
            "modelFamily": audit.MODEL,
            "status": "research-only-not-training-authorized",
            "trainingAuthorized": False,
            "rightsVerification": "author-claims-unverified",
            "splits": splits,
            "sealedHoldout": {
                "count": 64, "commitmentSha256": "0" * 64,
                "contentNotIncluded": True,
            },
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        self.manifest = manifest

    def test_synthetic_complete_manifest_passes_without_training(self):
        report, rows = audit.audit_manifest(self.root / "manifest.json")
        self.assertFalse(report["trainingAuthorized"])
        self.assertFalse(report["rightsIndependentlyVerified"])
        self.assertFalse(report["holdoutContentOpened"])
        self.assertFalse(report["holdoutExistenceIndependentlyVerified"])
        self.assertEqual(report["trainCount"], 256)
        self.assertEqual(report["developmentCount"], 64)
        self.assertEqual(report["developmentSliceClaims"]["prompt-transfer"], 16)
        self.assertTrue(audit.prompt_for(rows["train"][0]).endswith("Copy: "))
        self.assertTrue(audit.prompt_for(rows["development"][0]).endswith("Copy:\n"))
        self.assertTrue(audit.prompt_for(rows["development"][1]).endswith("Edited file:\n"))
        result = subprocess.run(
            [sys.executable, "-I", "-S", str(SCRIPT),
             "--manifest", str(self.root / "manifest.json")],
            capture_output=True, text=True, timeout=20, check=True,
        )
        self.assertIn("DATA STRUCTURE REVIEW ONLY", result.stdout)
        self.assertIn('"tokenizerAudit": "NOT RUN"', result.stdout)

    def test_data_change_detected_by_split_hash(self):
        path = self.root / "train.jsonl"
        with path.open("ab") as out:
            out.write(b" ")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            audit.audit_manifest(self.root / "manifest.json")

    def test_rehash_does_not_hide_cross_split_leakage(self):
        self.data["development"][0]["identifierFamily"] = self.data["train"][0]["identifierFamily"]
        self.write_all()
        with self.assertRaisesRegex(ValueError, "identifierFamily leakage"):
            audit.audit_manifest(self.root / "manifest.json")
        self.data["development"][0]["identifierFamily"] = "family-fresh000"
        self.data["development"][0]["source"] = self.data["train"][0]["source"]
        self.data["development"][0]["expected"] = self.data["train"][0]["expected"]
        self.data["development"][0]["sourceSha256"] = self.data["train"][0]["sourceSha256"]
        self.data["development"][0]["expectedSha256"] = self.data["train"][0]["expectedSha256"]
        self.write_all()
        with self.assertRaisesRegex(ValueError, "identical source leakage"):
            audit.audit_manifest(self.root / "manifest.json")

    def test_owner_rights_training_and_target_mismatch_refused(self):
        self.data["train"][0]["trainingAuthorized"] = True
        self.write_all()
        with self.assertRaisesRegex(ValueError, "no training approval"):
            audit.audit_manifest(self.root / "manifest.json")
        self.data["train"][0]["trainingAuthorized"] = False
        self.data["train"][0]["expected"] = "let malicious = 1;\n"
        self.write_all()
        with self.assertRaisesRegex(ValueError, "copy target"):
            audit.audit_manifest(self.root / "manifest.json")
        self.data["train"][0]["expected"] = self.data["train"][0]["source"]
        self.data["train"][0]["sourceOrigin"] = "assistant-authored"
        self.write_all()
        with self.assertRaisesRegex(ValueError, "owner-claimed provenance"):
            audit.audit_manifest(self.root / "manifest.json")

    def test_holdout_records_and_fake_approval_refused(self):
        self.manifest["trainingAuthorized"] = True
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cannot authorize training"):
            audit.audit_manifest(self.root / "manifest.json")
        self.manifest["trainingAuthorized"] = False
        self.manifest["sealedHoldout"]["file"] = "holdout.jsonl"
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sealed holdout commitment"):
            audit.audit_manifest(self.root / "manifest.json")

    def test_tokenizer_roundtrip_without_importing_torch(self):
        class ToyTokenizer:
            def encode(self, text):
                return [ord(ch) + 4 for ch in text]
            def decode(self, ids):
                return "".join(chr(v - 4) for v in ids)

        self.assertEqual(ToyTokenizer().decode(ToyTokenizer().encode("abc")), "abc")
        # Token budget is audited separately only when a real frozen tokenizer is supplied.
        # No CUDA/torch import, tokenizer fitting, or text generation occurs in these tests.


if __name__ == "__main__":
    unittest.main()
