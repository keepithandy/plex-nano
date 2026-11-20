"""Phase 2 development smoke: fixed-case scoring and real console entry points."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import plex_phase2_eval as phase2

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


if __name__ == "__main__":
    unittest.main()
