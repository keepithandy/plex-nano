"""End-to-end checks for the local console's supplied-file editing contract."""
import codecs
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CONSOLE = Path(__file__).resolve().with_name("PlexLocal.py")


class FileEditingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="plex phase1 ")
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)

    def run_console(self, *commands):
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-X", "utf8", str(CONSOLE)],
            cwd=self.folder, input="\n".join((*commands, "/exit", "")),
            encoding="utf-8", capture_output=True, timeout=20, check=True,
        )
        self.assertEqual(result.stderr, "")
        return result.stdout

    def test_rename_preserves_utf8_bom_crlf_quotes_comments_and_original(self):
        original = self.folder / "sample file.js"
        before = '// value café ☕\r\nconst value = 1;\r\nconsole.log("value", value); // value\r\n'
        after = '// value café ☕\r\nconst count = 1;\r\nconsole.log("value", count); // value\r\n'
        original.write_bytes(codecs.BOM_UTF8 + before.encode("utf-8"))
        transcript = self.run_console(
            '/open "sample file.js"', "copy exactly", '/save as "exact.js"',
            "rename value to count and keep its value at 1", "/diff",
            '/save as "edited.js"', "/undo", "/diff", '/save as "undone.js"',
        )
        self.assertIn("Current code (built-in command).", transcript)
        self.assertIn("Changed spans (line:column, end exclusive): 2:7-2:12, 3:22-3:27", transcript)
        self.assertIn('-console.log("value", value); // value', transcript)
        self.assertIn('+console.log("value", count); // value', transcript)
        self.assertIn(after.replace("\r\n", "\n"), transcript)
        self.assertIn("No edits to show.", transcript)
        for name in ("sample file.js", "exact.js", "undone.js"):
            self.assertEqual((self.folder / name).read_bytes(), codecs.BOM_UTF8 + before.encode("utf-8"))
        self.assertEqual((self.folder / "edited.js").read_bytes(), codecs.BOM_UTF8 + after.encode("utf-8"))

    def test_set_and_literal_replace_preserve_absent_final_newline(self):
        before = b"let value = -2;"
        (self.folder / "source.js").write_bytes(before)
        transcript = self.run_console(
            "/open source.js", "set value to 3", 'replace "value" with "count"', "/diff",
            "/save as edited.js", "/undo", "/save as prior.js",
        )
        self.assertIn("Changed spans (line:column, end exclusive): 1:13-1:15", transcript)
        self.assertIn(r"\ No newline at end of file", transcript)
        self.assertEqual((self.folder / "source.js").read_bytes(), before)
        self.assertEqual((self.folder / "edited.js").read_bytes(), b"let count = 3;")
        self.assertEqual((self.folder / "prior.js").read_bytes(), b"let value = 3;")

    def test_save_refuses_to_overwrite_source_or_existing_copy(self):
        original = self.folder / "source.js"
        target = self.folder / "already.js"
        original.write_bytes(b"let value = 1;\n")
        target.write_bytes(b"// keep this copy\n")
        transcript = self.run_console(
            "/open source.js", "set value to 2", "/save as ./source.js",
            "/save as already.js", "/save as new.js",
        )
        self.assertIn("The source file is protected.", transcript)
        self.assertIn("already.js already exists. Choose a new filename.", transcript)
        self.assertEqual(original.read_bytes(), b"let value = 1;\n")
        self.assertEqual(target.read_bytes(), b"// keep this copy\n")
        self.assertEqual((self.folder / "new.js").read_bytes(), b"let value = 2;\n")

    def test_pasting_starts_a_new_baseline_and_clears_the_old_bom_and_undo(self):
        original = self.folder / "opened.js"
        before = codecs.BOM_UTF8 + b"const value = 1;\r\n"
        original.write_bytes(before)
        transcript = self.run_console(
            "/open opened.js", "set value to 2", "/save as draft.js",
            "/paste", "const fresh = 3;", ".end",
            "/undo", "/diff", "/status", "/save as pasted.js",
        )
        self.assertIn("No edits to show.", transcript)
        self.assertIn("Source      pasted code", transcript)
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual((self.folder / "draft.js").read_bytes(), codecs.BOM_UTF8 + b"const value = 2;\r\n")
        self.assertEqual((self.folder / "pasted.js").read_bytes(), b"const fresh = 3;\n")

    def test_invalid_utf8_open_preserves_the_active_session(self):
        (self.folder / "invalid.js").write_bytes(b"\xff")
        transcript = self.run_console(
            "/paste", "let good = 2;", ".end", "/open invalid.js", "/save as saved.js",
        )
        self.assertIn("invalid.js is not a UTF-8 file.", transcript)
        self.assertEqual((self.folder / "saved.js").read_bytes(), b"let good = 2;\n")

    def test_ambiguous_and_unsupported_edits_keep_the_buffer_and_avoid_inference(self):
        cases = (
            ("let value = 1;\nfunction nested() { return value; }\n",
             "rename value to count", "Rename cannot verify syntax on line 2."),
            ("let value = 1;\nlet value = 2;\n",
             "rename value to count", "multiple declarations of value (line 2)."),
            ("let value = 1;\nlet count = 2;\n",
             "rename value to count", "Choose an unused JavaScript variable name other than count."),
            ("let value = 1;\nconsole.log(count, value);\n",
             "rename value to count", "Choose an unused JavaScript variable name other than count."),
            ("let value = 1;\n",
             "rename value to return", "Choose an unused JavaScript variable name other than return."),
            ("let value = 1;\n",
             "rename value to count and keep its value at 2", "The current value is 1."),
            ("let value = 1;\n",
             "rename missing to count", "No integer declaration named missing"),
            ('console.log("aaa");\n',
             'replace "aa" with "b"', "The original text must occur exactly once."),
            ('console.log("value", "value");\n',
             'replace "value" with "count"', "The original text must occur exactly once."),
            ("const value = 1;\nconsole.log(value + 1);\n",
             "set value to 2", "Set needs one integer declaration with that exact variable name."),
        )
        for index, (before, request, reason) in enumerate(cases):
            with self.subTest(source=before, request=request):
                original = self.folder / f"source-{index}.js"
                target = self.folder / f"saved-{index}.js"
                original.write_bytes(before.encode("utf-8"))
                transcript = self.run_console(
                    f"/open {original.name}", request, "/diff", f"/save as {target.name}",
                )
                self.assertIn(reason, transcript)
                self.assertIn("No edits to show.", transcript)
                self.assertNotIn("Generating locally", transcript)
                self.assertEqual(original.read_bytes(), before.encode("utf-8"))
                self.assertEqual(target.read_bytes(), original.read_bytes())


    def test_malformed_builtin_edits_refuse_without_model_fallback(self):
        before = b"let value = 1;\n"
        (self.folder / "source.js").write_bytes(before)
        transcript = self.run_console(
            "copy exactly",
            "/open source.js",
            "rename value to",
            "set value to many",
            "replace value with count",
            "/diff",
            "/save as unchanged.js",
        )
        self.assertIn("Paste a code snippet first.", transcript)
        self.assertEqual(transcript.count("Unsupported built-in edit syntax."), 3)
        self.assertNotIn("Generating locally", transcript)
        self.assertIn("No edits to show.", transcript)
        self.assertEqual((self.folder / "source.js").read_bytes(), before)
        self.assertEqual((self.folder / "unchanged.js").read_bytes(), before)

    def test_unicode_surrogate_replacement_refuses_and_console_stays_usable(self):
        before = b"let value = 1;\n"
        (self.folder / "source.js").write_bytes(before)
        transcript = self.run_console(
            "/open source.js",
            r'replace "value" with "\ud800"',
            "set value to 2",
            "/save as edited.js",
        )
        self.assertIn("Replacement contains an invalid Unicode surrogate.", transcript)
        self.assertIn("Built-in Python edit applied.", transcript)
        self.assertNotIn("Generating locally", transcript)
        self.assertEqual((self.folder / "source.js").read_bytes(), before)
        self.assertEqual((self.folder / "edited.js").read_bytes(), b"let value = 2;\n")

    def test_unsaved_edits_block_source_switch_until_save_or_discard(self):
        original = self.folder / "original.js"
        other = self.folder / "other.js"
        original.write_bytes(b"let value = 1;\n")
        other.write_bytes(b"let other = 9;\n")
        transcript = self.run_console(
            "/open original.js", "set value to 2", "/open other.js", "/diff",
            "/save as preserved.js", "/open other.js", "/status",
            "/open original.js", "set value to 3", "/paste",
            "/discard", "/paste", "let pasted = 4;", ".end",
            "/save as pasted.js",
        )
        self.assertEqual(transcript.count("Unsaved edits:"), 2)
        self.assertIn("Discarded in-memory edits.", transcript)
        self.assertIn("Source      other.js", transcript)
        self.assertEqual(original.read_bytes(), b"let value = 1;\n")
        self.assertEqual(other.read_bytes(), b"let other = 9;\n")
        self.assertEqual((self.folder / "preserved.js").read_bytes(), b"let value = 2;\n")
        self.assertEqual((self.folder / "pasted.js").read_bytes(), b"let pasted = 4;\n")



if __name__ == "__main__":
    unittest.main()
