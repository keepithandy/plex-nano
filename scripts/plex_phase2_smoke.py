"""Phase 2 development smoke: fixed-case scoring and real console entry points."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PHASE2_PATH = Path(__file__).resolve().with_name("plex_phase2_eval.py")
_spec = importlib.util.spec_from_file_location("plex_phase2_eval", PHASE2_PATH)
phase2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phase2)

CONSOLE = Path(__file__).with_name("PlexLocal.py").resolve()


class Phase2Tests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="plex phase2 ")
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)

    def run_console(self, *commands):
        run = subprocess.run(
            [sys.executable, "-I", "-S", "-X", "utf8", str(CONSOLE)],
            cwd=self.folder, input="\n".join((*commands, "/exit", "")),
            capture_output=True, encoding="utf-8", timeout=20, check=True,
        )
        self.assertEqual(run.stderr, "")
        return run.stdout

    def test_frozen_development_suite(self):
        cases, digest = phase2.load_cases()
        self.assertEqual(len(cases), 33)
        self.assertEqual(len(digest), 64)
        result = phase2.score_editor(cases, phase2.editor_module())
        self.assertEqual(result, {
            "edits": [20, 20], "refusals": [13, 13], "failures": [],
        })

    def test_console_paraphrased_rename_diff_undo_save_and_preservation(self):
        original = self.folder / "source.js"
        before = b'let value = 1;\nconsole.log("value", value);\n'
        after = b'let count = 1;\nconsole.log("value", count);\n'
        original.write_bytes(before)
        transcript = self.run_console(
            '/open "source.js"', "Change the variable value to count",
            "/diff", "/save as edited.js", "/undo", "/save as restored.js",
        )
        self.assertIn("Understood (built-in): rename value to count.", transcript)
        self.assertIn("Built-in Python edit applied.", transcript)
        self.assertIn("Changed spans", transcript)
        self.assertIn('-let value = 1;', transcript)
        self.assertIn('+let count = 1;', transcript)
        self.assertNotIn("Generating locally with Plex 600M", transcript)
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual((self.folder / "edited.js").read_bytes(), after)
        self.assertEqual((self.folder / "restored.js").read_bytes(), before)

    def test_implicit_variable_requires_single_verifiable_target(self):
        original = self.folder / "two.js"
        before = b"let value = 1;\nlet score = 2;\n"
        original.write_bytes(before)
        transcript = self.run_console(
            "/open two.js", "Call this variable count instead",
            "/diff", "/save as untouched.js",
        )
        self.assertIn("Name the variable to rename", transcript)
        self.assertIn("No edits to show.", transcript)
        self.assertNotIn("Generating locally", transcript)
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual((self.folder / "untouched.js").read_bytes(), before)

    def test_natural_language_set_is_bounded(self):
        original = self.folder / "number.js"
        original.write_bytes(b"const value = 1;\r\n")
        transcript = self.run_console(
            "/open number.js", "Make value equal to 3", "/save as changed.js",
        )
        self.assertIn("Understood (built-in): set value to 3.", transcript)
        self.assertEqual(original.read_bytes(), b"const value = 1;\r\n")
        self.assertEqual((self.folder / "changed.js").read_bytes(), b"const value = 3;\r\n")

    def test_raw_model_empty_eos_cannot_be_misreported_as_refusal(self):
        cases, _ = phase2.load_cases()
        path = self.folder / "raw.jsonl"
        lines = [{"id": case["id"], "text": "", "stop": "eos", "elapsedMs": 12.5} for case in cases]
        path.write_text("".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8")
        score = phase2.score_model(cases, path)
        self.assertEqual(score["edits"], [0, 20])
        self.assertEqual(score["refusals"], [0, 13])
        self.assertEqual(score["emptyEos"], 33)
        self.assertEqual(score["avgMs"], 12.5)

    def test_raw_model_result_ids_cannot_be_incomplete_or_duplicated(self):
        cases, _ = phase2.load_cases()
        path = self.folder / "bad.jsonl"
        row = {"id": cases[0]["id"], "text": "", "stop": "eos"}
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must match the complete"):
            phase2.score_model(cases, path)
        path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate model output"):
            phase2.score_model(cases, path)

    def test_report_labels_model_as_not_run_without_results(self):
        completed = subprocess.run(
            [sys.executable, "-I", "-S", str(Path(phase2.__file__))],
            capture_output=True, text=True, timeout=20, check=True,
        )
        self.assertIn("Built-in: 20/20 exact edits; 13/13 correct refusals.", completed.stdout)
        self.assertIn("Raw 600M baseline: not run", completed.stdout)
        self.assertIn("Raw 600M alternative: not run", completed.stdout)


    def test_m11_comparison_requires_explicit_read_only_flag(self):
        script = Path(phase2.__file__).resolve().with_name("plex_phase2_infer.py")
        destination = self.folder / "raw-model-output"
        run = subprocess.run(
            [sys.executable, "-I", "-S", str(script), "--output-dir", str(destination)],
            capture_output=True, text=True, timeout=20,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Add --execute-read-only", run.stderr)
        self.assertFalse(destination.exists())

    def test_m11_alternative_prompt_keeps_supplied_file_explicit(self):
        script = Path(phase2.__file__).resolve().with_name("plex_phase2_infer.py")
        spec = importlib.util.spec_from_file_location("plex_phase2_infer", script)
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)
        prompt = compare.alternative_prompt("let value = 1;\n", "Rename value to count")
        self.assertIn("Source JavaScript:\nlet value = 1;\n", prompt)
        self.assertIn("Instruction: Rename value to count", prompt)
        self.assertIn("REFUSE:", prompt)



    def make_provenance_run(self):
        cases, suite_hash = phase2.load_cases()
        dest = self.folder / "raw-m11"
        dest.mkdir()
        digests = {}
        for name in ("baseline", "alternative"):
            rows = [{
                "id": case["id"], "text": "", "stop": "eos",
                "elapsedMs": 12.5, "generatedTokens": 0,
                "checkpointSha256": phase2.CHECKPOINT_SHA256,
            } for case in cases]
            raw = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
            (dest / f"{name}.jsonl").write_bytes(raw)
            digests[name] = hashlib.sha256(raw).hexdigest()
        receipt = {
            "protocol": phase2.RUN_PROTOCOL, "suiteSha256": suite_hash,
            "checkpointSha256": phase2.CHECKPOINT_SHA256,
            "tokenizerSha256": phase2.TOKENIZER_SHA256,
            "policy": "fitted-vocab-v1", "maxNewTokens": 96,
            "trainingPerformed": False, "prompts": ["baseline", "alternative"],
            "casesPerPrompt": len(cases), "responseSha256": digests,
        }
        (dest / "identity.json").write_text(json.dumps(receipt), encoding="utf-8")
        return cases, suite_hash, dest, receipt

    def test_signed_by_content_receipt_accepts_complete_synthetic_pair(self):
        cases, suite_hash, dest, _ = self.make_provenance_run()
        phase2.validate_raw_pair(
            cases, suite_hash, dest / "baseline.jsonl", dest / "alternative.jsonl"
        )
        score = phase2.score_model(cases, dest / "baseline.jsonl")
        self.assertEqual(score["emptyEos"], len(cases))
        self.assertEqual(score["edits"], [0, 20])
        self.assertEqual(score["refusals"], [0, 13])

    def test_raw_pair_refuses_mismatched_receipt_and_modified_responses(self):
        cases, suite_hash, dest, receipt = self.make_provenance_run()
        receipt["suiteSha256"] = "0" * 64
        (dest / "identity.json").write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            phase2.validate_raw_pair(
                cases, suite_hash, dest / "baseline.jsonl", dest / "alternative.jsonl"
            )
        receipt["suiteSha256"] = suite_hash
        (dest / "identity.json").write_text(json.dumps(receipt), encoding="utf-8")
        with (dest / "alternative.jsonl").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "differ from the saved receipt"):
            phase2.validate_raw_pair(
                cases, suite_hash, dest / "baseline.jsonl", dest / "alternative.jsonl"
            )

    def test_raw_pair_refuses_wrong_checkpoint_even_with_updated_response_digest(self):
        cases, suite_hash, dest, receipt = self.make_provenance_run()
        path = dest / "baseline.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["checkpointSha256"] = "0" * 64
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
        path.write_bytes(raw)
        receipt["responseSha256"]["baseline"] = hashlib.sha256(raw).hexdigest()
        (dest / "identity.json").write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid or unverified"):
            phase2.validate_raw_pair(
                cases, suite_hash, dest / "baseline.jsonl", dest / "alternative.jsonl"
            )

    def test_cli_requires_both_models_or_neither(self):
        run = subprocess.run([
            sys.executable, "-I", "-S", str(PHASE2_PATH),
            "--baseline-model-jsonl", str(self.folder / "incomplete.jsonl"),
        ], capture_output=True, text=True, timeout=20)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Provide both raw model files", run.stderr)

    def test_inference_refuses_output_outside_artifacts_before_model_load(self):
        script = PHASE2_PATH.with_name("plex_phase2_infer.py")
        destination = self.folder / "outside-artifacts"
        run = subprocess.run([
            sys.executable, "-I", "-S", str(script),
            "--execute-read-only", "--output-dir", str(destination),
        ], capture_output=True, text=True, timeout=20)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("NEW direct child of training/artifacts", run.stderr)
        self.assertNotIn("No module named", run.stderr)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
