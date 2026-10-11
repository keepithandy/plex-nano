"""U1.2 interactive CLI tests. No real weights or protected datasets."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
import unittest

from plex_training.interactive import (
    InteractiveModel, format_prompt, read_source, run_repl,
)
from plex_training.file_conditioned_contract import PROMPT_TEMPLATE
from plex_training.tokenizer import CODEC


class FakeTokenizer:
    vocabulary_size = 32

    def __init__(self):
        self.latest = None

    def encode(self, text):
        self.latest = text
        return [4, 5]

    def decode(self, ids):
        return self.latest + "RAW OUTPUT"


class InteractiveTests(unittest.TestCase):
    def test_plain_request_is_passed_unchanged(self):
        self.assertEqual(format_prompt("Write a CSS class."), "Write a CSS class.")

    def test_file_context_matches_existing_p3_template_and_does_not_edit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "index.html"
            source = '<button class="primary">Apply</button>'
            path.write_text(source, encoding="utf-8")
            selected = read_source(str(path))
            prompt = format_prompt("Change button text to Save.", selected)
            self.assertEqual(
                prompt,
                PROMPT_TEMPLATE.format(
                    language="html",
                    request="Change button text to Save.",
                    input_file=source,
                ),
            )
            self.assertEqual(path.read_text(encoding="utf-8"), source)

    def test_reject_unsupported_oversized_or_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                read_source(str(root / "missing.js"))
            unsupported = root / "code.py"
            unsupported.write_text("print('hi')", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Supported"):
                read_source(str(unsupported))
            large = root / "large.js"
            large.write_text("x" * 4097, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "4096"):
                read_source(str(large))

    def test_reject_overlong_combined_input(self):
        with self.assertRaisesRegex(ValueError, "4096"):
            format_prompt("a" * 4097)

    def test_checkpoint_and_tokenizer_loaded_only_once_for_many_requests(self):
        tokenizer = FakeTokenizer()
        metadata = {"tokenizerSha256": "test-hash"}
        payload = {
            "codec": CODEC,
            "tokenizerRecord": metadata,
            "step": 100,
            "datasetRecord": {},
            "initializationRecord": {"pretrainedCheckpointLoaded": False},
        }
        fake_model = SimpleNamespace(eval=Mock())
        fake_device = "cpu"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trained.pt"
            path.write_bytes(b"test-only fixture; never passed to torch.load")
            with (
                patch("plex_training.interactive._completion_tokenizer", return_value=(tokenizer, metadata)) as load_tokenizer,
                patch("plex_training.interactive.read_checkpoint", return_value=(fake_model, payload)) as load_weights,
                patch("plex_training.interactive.select_device", return_value=fake_device),
                patch("plex_training.interactive._generate_token_ids", return_value=([6, 7], False)) as decode,
            ):
                session = InteractiveModel(checkpoint=path, bundle_dir=Path(folder))
                self.assertEqual(session.complete("First request."), ("RAW OUTPUT", 2, False))
                self.assertEqual(session.complete("Second request."), ("RAW OUTPUT", 2, False))
                self.assertEqual(load_weights.call_count, 1)
                self.assertEqual(load_tokenizer.call_count, 1)
                self.assertEqual(decode.call_count, 2)
                self.assertEqual(decode.call_args.kwargs["seed"], 1337)
                fake_model.eval.assert_called_once_with()

    def test_reject_untrained_checkpoint(self):
        tokenizer = FakeTokenizer()
        metadata = {"tokenizerSha256": "test-hash"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stage.pt"
            path.write_bytes(b"fixture")
            payload = {"codec": CODEC, "tokenizerRecord": metadata, "step": 0,
                       "datasetRecord": {}, "initializationRecord": {
                           "pretrainedCheckpointLoaded": False,
                       }}
            with (
                patch("plex_training.interactive._completion_tokenizer", return_value=(tokenizer, metadata)),
                patch("plex_training.interactive.read_checkpoint", return_value=(Mock(), payload)),
                patch("plex_training.interactive.select_device", return_value="cpu"),
            ):
                with self.assertRaisesRegex(ValueError, "trained"):
                    InteractiveModel(checkpoint=path, bundle_dir=Path(folder))

    def test_many_prompts_new_session_and_exit_without_logging(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "test.css"
            file.write_text(".btn { color: red; }", encoding="utf-8")
            requests = iter([
                "/help", f'/file "{file}"',
                "Change red to navy.", "/new",
                "Write a CSS class.", "/exit",
            ])
            printed = []
            seen = []
            model = SimpleNamespace(step=100, device="cpu")

            def complete(prompt):
                seen.append(prompt)
                return "RAW RESPONSE", 7, False

            model.complete = complete
            result = run_repl(model, read=lambda _: next(requests), write=printed.append)
            self.assertEqual(result, 0)
            self.assertEqual(len(seen), 2)
            self.assertIn("Edited file:", seen[0])
            self.assertEqual(seen[1], "Write a CSS class.")
            self.assertEqual(printed.count("RAW RESPONSE"), 2)
            self.assertTrue(any("New prompt session" in line for line in printed))
            self.assertEqual(file.read_text(encoding="utf-8"), ".btn { color: red; }")

    def test_error_is_recoverable_and_no_output_is_fabricated(self):
        prompts = iter(["Wrong", "Good", "/exit"])
        lines = []
        model = SimpleNamespace(step=100, device="cpu")

        def complete(prompt):
            if prompt == "Wrong":
                raise ValueError("synthetic test error")
            return "", 0, True

        model.complete = complete
        result = run_repl(model, read=lambda _: next(prompts), write=lines.append)
        self.assertEqual(result, 0)
        self.assertTrue(any("synthetic test error" in line for line in lines))
        self.assertIn("(no text generated)", lines)


if __name__ == "__main__":
    unittest.main()
