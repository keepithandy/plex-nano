"""U1.2 read-only interactive terminal for a trained Plex Nano BPE checkpoint.

No training, no writes to the supplied file, and no automatic logging. Every
request is independent: this is NOT a conversation-memory model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Callable

from .checkpoint import read_checkpoint
from .completion import _completion_tokenizer, _generate_token_ids
from .file_conditioned_contract import PROMPT_TEMPLATE
from .telemetry import select_device
from .tokenizer import CODEC


@dataclass(frozen=True)
class SourceFile:
    path: Path
    language: str
    content: str


_EXTENSIONS = {
    ".htm": "html",
    ".html": "html",
    ".css": "css",
    ".js": "javascript",
}

_HELP = """Commands:
  /help        Show these commands
  /file PATH   Supply ONE local HTML, CSS or JavaScript file as context
  /clear       Stop supplying that file
  /new         Start a fresh prompt session (clears selected file)
  /exit        Close Plex

Anything else is sent to the existing model as a coding request.
Each request is INDEPENDENT. Plex does not remember earlier requests.
Supplied files are read-only: Plex never edits or saves them.
The output is the model's raw completion and may be wrong or incomplete.
"""


def read_source(raw_path: str) -> SourceFile:
    """Only read an explicitly selected, small local HTML/CSS/JS file."""
    path = Path(raw_path.strip().strip('"')).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("Choose an existing regular HTML, CSS or JavaScript file.")
    language = _EXTENSIONS.get(path.suffix.lower())
    if language is None:
        raise ValueError("Supported source files: .html, .htm, .css and .js.")
    if path.stat().st_size > 4096:
        raise ValueError("File exceeds the 4096-byte input limit; choose a smaller example.")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise ValueError("Supplied file must be UTF-8 encoded.") from exc
    if not content.strip():
        raise ValueError("Supplied file is empty.")
    return SourceFile(path=path.resolve(), language=language, content=content)


def format_prompt(request: str, source: SourceFile | None = None) -> str:
    """Use the P3-01 format when the user explicitly supplies code context."""
    if not request.strip():
        raise ValueError("Enter a nonempty coding request.")
    if source is None:
        prompt = request
    else:
        prompt = PROMPT_TEMPLATE.format(
            language=source.language, request=request, input_file=source.content
        )
    if len(prompt.encode("utf-8")) > 4096:
        raise ValueError("Request plus supplied file exceeds 4096 UTF-8 bytes.")
    return prompt


class InteractiveModel:
    """Load and verify weights/tokenizer exactly once per terminal session."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        bundle_dir: Path | None = None,
        device_name: str = "auto",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 1337,
    ) -> None:
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError("Checkpoint must be an existing regular trained .pt file.")
        tokenizer_directory = bundle_dir or checkpoint.parent / "tokenizer"
        tokenizer, record = _completion_tokenizer(tokenizer_directory)
        device = select_device(device_name)
        model, payload = read_checkpoint(checkpoint, device)
        if (
            payload.get("codec") != CODEC
            or payload.get("tokenizerRecord") != record
            or type(payload.get("step")) is not int
            or payload["step"] <= 0
            or not isinstance(payload.get("datasetRecord"), dict)
            or not isinstance(payload.get("initializationRecord"), dict)
            or payload["initializationRecord"].get("pretrainedCheckpointLoaded") is not False
        ):
            raise ValueError(
                "Checkpoint is not a trained Plex BPE checkpoint with matching tokenizer."
            )
        model.eval()
        self.checkpoint = checkpoint.resolve()
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.step = payload["step"]
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed

    def complete(self, prompt: str) -> tuple[str, int, bool]:
        """Return only actual model-generated text, not a fabricated assistant reply."""
        if not prompt or len(prompt.encode("utf-8")) > 4096:
            raise ValueError("Prompt must be nonempty and at most 4096 UTF-8 bytes.")
        prompt_ids = self.tokenizer.encode(prompt)
        generated, reached_eos = _generate_token_ids(
            self.model,
            prompt_ids,
            actual_vocab=self.tokenizer.vocabulary_size,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            seed=self.seed,
            device=self.device,
        )
        text = self.tokenizer.decode(prompt_ids + generated)
        if not text.startswith(prompt):
            raise RuntimeError("Tokenizer did not preserve the completion prompt.")
        return text[len(prompt):], len(generated), reached_eos


def run_repl(
    session: InteractiveModel,
    *,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> int:
    """Keep the same weights in memory until /exit; no prompt history is retained."""
    write("\nPlex Nano | interactive terminal (U1.2)")
    write(f"Loaded trained step {session.step} on {session.device}.")
    write("Type /help for commands or /exit to quit.")
    write("Warning: generations are experimental, not guaranteed correct code.\n")
    selected_file: SourceFile | None = None

    while True:
        try:
            request = read("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            write("\nPlex session closed.")
            return 0
        if not request:
            continue
        command = request.lower()
        if command in ("/exit", "/quit"):
            write("Plex session closed.")
            return 0
        if command == "/help":
            write(_HELP)
            continue
        if command in ("/new", "/clear"):
            selected_file = None
            if command == "/new":
                write("New prompt session. Previous prompts are not remembered.")
            else:
                write("Supplied file cleared.")
            continue
        if command == "/file":
            write("Use /file followed by the full path to a small .html, .css or .js file.")
            continue
        if command.startswith("/file "):
            try:
                new_file = read_source(request[6:])
            except (OSError, ValueError) as exc:
                write(f"File error: {exc}")
                continue
            selected_file = new_file
            write(f"Read-only file selected: {selected_file.path}")
            continue
        if request.startswith("/"):
            write("Unknown command. Type /help for available commands.")
            continue

        try:
            prompt = format_prompt(request, selected_file)
            result, token_count, reached_eos = session.complete(prompt)
        except (ValueError, RuntimeError, OSError) as exc:
            write(f"Plex could not complete that request: {exc}")
            write("Try a shorter request or smaller file. The session is still open.")
            continue
        except KeyboardInterrupt:
            write("\nGeneration interrupted. Type /exit to quit.")
            continue

        write("\nPlex >")
        write(result if result else "(no text generated)")
        if not reached_eos:
            write(f"[{token_count} tokens; stopped at token limit, response may be incomplete]")
        else:
            write(f"[{token_count} tokens; model ended its response]")
        write("")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="plex-interactive", description="Use a saved Plex model in a terminal (no training)."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    if not 1 <= args.max_new_tokens <= 256:
        parser.error("--max-new-tokens must be 1..256.")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 10:
        parser.error("--temperature must be between 0 and 10.")
    if not 0 <= args.seed <= 2**63 - 1:
        parser.error("--seed must be between 0 and 2^63-1.")

    try:
        session = InteractiveModel(
            checkpoint=args.checkpoint,
            bundle_dir=args.bundle_dir,
            device_name=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        return run_repl(session)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Plex interactive launch failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
