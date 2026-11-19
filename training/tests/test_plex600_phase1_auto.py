"""Phase 1 safe path discovery / local history re-verification, no model imports."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from plex_training import plex600_phase1_auto as auto
from plex_training import plex600_phase1_historical as history
from plex_training import plex600_m10_worker as worker


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, sort_keys=True) + "\n").encode("utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def skeleton(root: Path) -> dict:
    artifacts = root / "training/artifacts"
    artifacts.mkdir(parents=True)
    backup = root / "m9-independent-backup"
    backup.mkdir()
    (backup / "weights.pt").write_bytes(b"fixture checkpoint, not real model weights\n")
    save_json(backup / "report.json", {"note": "fixture, not model report"})
    bundle = artifacts / "plex600-fresh-tokenizer"
    bundle.mkdir()
    baseline = artifacts / "plex600-m10-m9-full-baseline-fixture"
    baseline.mkdir()
    save_json(baseline / "report.json", {
        "status": "read-only-m9-generation-complete",
        "selectedCases": 18,
        "decodingPolicies": ["unmasked-v1", "fitted-vocab-v1"],
    })
    (baseline / "responses.jsonl").write_bytes(b"synthetic response only\n")
    run = artifacts / "plex600-m10-approved-fixture"
    run.mkdir()
    save_json(run / "report.json", {
        "protocol": worker.PROTOCOL,
        "status": "bounded-50-updates-complete-unreviewed",
        "baselineReportSha256": digest(baseline / "report.json"),
        "baselineResponsesSha256": digest(baseline / "responses.jsonl"),
    })
    return {
        "artifacts": artifacts, "backup": backup, "bundle": bundle,
        "baseline": baseline, "run": run,
    }


def example_history() -> dict:
    return {
        "protocol": history.PROTOCOL,
        "status": "owner-local-checkpoint-files-and-response-scores-reverified",
        "identities": {"candidateSha256": "a" * 64},
        "entries": [{"entryId": "m9-original"}, {"entryId": "m10-step-10"}],
        "comparisons": [{"step": 10, "caseComparisons": []}],
        "decisionLedger": [{"decision": "no-promotion-on-exact-copy"}],
        "verificationScope": "fixture-hash-test-only",
    }


class AutoRegistrationTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.fixtures = skeleton(self.root)

    def test_default_is_inert_without_paths_or_model(self):
        out = StringIO()
        with redirect_stdout(out):
            self.assertEqual(auto.main([]), 0)
        result = json.loads(out.getvalue())
        self.assertFalse(result["modelLoaded"])
        self.assertFalse(result["trainingExecuted"])
        self.assertNotIn("torch", auto.__dict__)

    def test_discover_matches_exact_report_and_response_hashes(self):
        with patch.object(auto.research, "ROOT", self.root):
            selected = auto.discover(self.fixtures["backup"])
        self.assertEqual(Path(selected["baselineDir"]).resolve(),
                         self.fixtures["baseline"].resolve())
        self.assertEqual(Path(selected["m10RunDir"]).resolve(),
                         self.fixtures["run"].resolve())
        self.assertEqual(
            selected["matchedBaselineResponsesSha256"],
            digest(self.fixtures["baseline"] / "responses.jsonl"))
        self.assertEqual(selected["status"],
                         "paths-and-baseline-receipt-hashes-matched-not-yet-full-audit")
        self.assertFalse(selected["modelLoaded"])

    def test_rejects_missing_baseline_and_tampered_baseline(self):
        base = self.fixtures["baseline"] / "responses.jsonl"
        base.write_bytes(base.read_bytes() + b"mutated")
        with patch.object(auto.research, "ROOT", self.root):
            with self.assertRaisesRegex(auto.Refusal, "Cannot uniquely match"):
                auto.discover(self.fixtures["backup"])

    def test_multiple_matching_baselines_are_not_chosen_arbitrarily(self):
        second = (self.fixtures["artifacts"]
                  / "plex600-m10-m9-full-baseline-duplicate")
        second.mkdir()
        for filename in ("report.json", "responses.jsonl"):
            (second / filename).write_bytes(
                (self.fixtures["baseline"] / filename).read_bytes())
        with patch.object(auto.research, "ROOT", self.root):
            with self.assertRaisesRegex(auto.Refusal, "Cannot uniquely match"):
                auto.discover(self.fixtures["backup"])

    def test_multiple_completed_runs_require_explicit_run(self):
        second = (self.fixtures["artifacts"]
                  / "plex600-m10-approved-second")
        second.mkdir()
        (second / "report.json").write_bytes(
            (self.fixtures["run"] / "report.json").read_bytes())
        with patch.object(auto.research, "ROOT", self.root):
            with self.assertRaisesRegex(auto.Refusal, "specify --m10-run-dir"):
                auto.discover(self.fixtures["backup"])
            selected = auto.discover(self.fixtures["backup"],
                                     run_dir=self.fixtures["run"])
        self.assertEqual(selected["m10RunDir"], str(self.fixtures["run"]))

    def test_external_run_directory_rejected(self):
        remote = self.root / "other-run"
        remote.mkdir()
        with patch.object(auto.research, "ROOT", self.root):
            with self.assertRaisesRegex(auto.Refusal, "direct child"):
                auto.discover(self.fixtures["backup"], run_dir=remote)

    def test_discovery_rejects_checkpoint_symlink(self):
        path = self.fixtures["backup"] / "weights.pt"
        path.unlink()
        try:
            path.symlink_to(self.fixtures["baseline"] / "report.json")
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks unavailable")
        with patch.object(auto.research, "ROOT", self.root):
            with self.assertRaisesRegex(auto.Refusal, "links"):
                auto.discover(self.fixtures["backup"])

    def test_verification_of_published_files_with_local_rehash(self):
        record = example_history()
        output = self.fixtures["artifacts"] / "phase1-registered-fixture"
        with patch.object(auto.research, "ROOT", self.root):
            result = history.register(output, record)
            self.assertEqual(result["checkpointCount"], 2)
            verification = auto.verify_registered(output, record)
            self.assertEqual(verification["checkpointEntries"], 2)
            self.assertEqual(len(verification["registeredFileSha256"]), 3)
            self.assertFalse(verification["historicalInferenceOriginIndependentlyProven"])

    def test_rejects_modified_registry_even_if_valid_json(self):
        record = example_history()
        output = self.fixtures["artifacts"] / "phase1-registered-fixture"
        with patch.object(auto.research, "ROOT", self.root):
            history.register(output, record)
            filename = output / "experiment-decisions.json"
            value = json.loads(filename.read_text(encoding="utf-8"))
            value["decisionLedger"][0]["decision"] = "promote"
            save_json(filename, value)
            with self.assertRaisesRegex(auto.Refusal, "disagrees with fresh audit"):
                auto.verify_registered(output, record)

    def test_no_implicit_local_approval_or_registration(self):
        with patch.object(auto.research, "ROOT", self.root):
            out = StringIO()
            err = StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc = auto.main(["--discover",
                                "--m9-backup-dir", str(self.fixtures["backup"]),
                                "--approval-file", "some-file.json"])
            self.assertEqual(rc, 2)
            self.assertIn("Discovery does not use approval", err.getvalue())
            self.assertFalse((self.fixtures["artifacts"] / "some-file.json").exists())

    def test_register_refuses_existing_output_without_large_audit(self):
        out = self.fixtures["artifacts"] / "history-existing"
        out.mkdir()
        with patch.object(auto.research, "ROOT", self.root), \
             patch.object(auto, "verified_history",
                          side_effect=AssertionError("must not audit")):
            stderr = StringIO()
            with redirect_stderr(stderr):
                status = auto.main([
                    "--register", "--m9-backup-dir", str(self.fixtures["backup"]),
                    "--output-dir", str(out),
                ])
            self.assertEqual(status, 2)
            self.assertIn("NEW direct child", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
