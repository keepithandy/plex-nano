"""Local editor with optional suggestions from Plex's scratch-trained 600M model."""
import codecs
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from bisect import bisect_right
from pathlib import Path

REPO = "/home/quali/plex-nano"
ARTIFACTS = Path("training/artifacts")
WEIGHTS = ARTIFACTS / "plex600-m11-eos1-f68483d42b-repair1/step-200/weights.pt"
DIGEST = "d7a7776de86230320f6e0e48b07dc76761c3d42ac80e4c330839e47c9ffc93dd"
DECL = re.compile(
    r"([ \t]*(?:let|var|const)[ \t]+)([A-Za-z_$][A-Za-z0-9_$]*)"
    r"([ \t]*=[ \t]*)(-?(?:0|[1-9][0-9]*))([ \t]*;?[ \t]*(?:\r?\n)?)"
)
IDENTIFIER = r"[A-Za-z_$][A-Za-z0-9_$]*"
LOG_ARG = re.compile(rf"""(?:{IDENTIFIER}|-?[0-9]+|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""")
LOG_LINE = re.compile(
    rf"[ \t]*console\.log\([ \t]*({LOG_ARG.pattern}(?:[ \t]*,[ \t]*{LOG_ARG.pattern})*)?"
    r"[ \t]*\)[ \t]*;?[ \t]*(?://.*)?"
)
RESERVED = set("""
await break case catch class const continue debugger default delete do else enum
export extends false finally for function if implements import in instanceof
interface let new null package private protected public return static super
switch this throw true try typeof var void while with yield arguments eval
""".split())


def rename_changes(source, old, new, expected_value):
    """Locate a binding and its references within the supported flat JavaScript subset."""
    if re.search(r"[\u2028\u2029]", source):
        raise ValueError("Rename cannot verify JavaScript with Unicode line separators.")
    changes = []
    declarations = set()
    names = set()
    initial = None
    for line_num, line in enumerate(re.finditer(r"[^\r\n]*(?:\r\n?|\n|$)", source), 1):
        start = line.start()
        code = line[0].rstrip("\r\n")
        if not code.strip(" \t") or code.lstrip(" \t").startswith("//"):
            continue
        declaration = DECL.fullmatch(code.partition("//")[0])
        if declaration:
            name = declaration[2]
            if name in declarations:
                raise ValueError(f"Rename cannot verify multiple declarations of {name} (line {line_num}).")
            if name in RESERVED or name == "console":
                raise ValueError(f"Rename cannot verify a declaration of {name} (line {line_num}).")
            declarations.add(name)
            names.add(name)
            if name == old:
                initial = declaration[4]
                changes.append((start + declaration.start(2), start + declaration.end(2), new))
            continue
        call = LOG_LINE.fullmatch(code)
        if not call:
            raise ValueError(
                f"Rename cannot verify syntax on line {line_num}. "
                "Use flat integer declarations and simple console.log calls."
            )
        names.add("console")
        for arg in LOG_ARG.finditer(call[1] or ""):
            if re.fullmatch(IDENTIFIER, arg[0]):
                names.add(arg[0])
                if arg[0] == old:
                    pos = start + call.start(1)
                    changes.append((pos + arg.start(), pos + arg.end(), new))
    if initial is None:
        raise ValueError(f"No integer declaration named {old} in this snippet.")
    if new in RESERVED or (old != new and new in names):
        raise ValueError(f"Choose an unused JavaScript variable name other than {new}.")
    if expected_value is not None and expected_value != initial:
        raise ValueError(f"The current value is {initial}.")
    return changes


COPY_REQUEST = re.compile(
    r"(?:return|copy) (?:the supplied|this) (?:javascript line|code|line) exactly as written\.?",
    re.IGNORECASE,
)


def verified_changes(source, request):
    command = request.strip()
    if command.lower() in {"copy", "copy exactly", "copy this code", "show"}:
        return []
    if COPY_REQUEST.fullmatch(" ".join(command.split())):
        return []
    rename = re.fullmatch(
        rf"(?i:rename)\s+({IDENTIFIER})\s+(?i:to)\s+({IDENTIFIER})"
        r"(?:\s+(?i:and keep its value)(?:\s+(?i:at)\s+(-?(?:0|[1-9][0-9]*)))?)?\.?",
        command,
    )
    if rename:
        return rename_changes(source, *rename.groups())
    change = re.fullmatch(r"(?i:set)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s+(?i:to)\s+(-?(?:0|[1-9][0-9]*))", command)
    if change:
        match = DECL.fullmatch(source)
        if not match or match[2] != change[1]:
            raise ValueError("Set needs one integer declaration with that exact variable name.")
        return [(match.start(4), match.end(4), change[2])]
    if command.lower().startswith('replace "'):
        literal = r'"(?:\\.|[^"\\])*"'
        parts = re.fullmatch(rf"(?i:replace)\s+({literal})\s+(?i:with)\s+({literal})", command)
        if not parts:
            raise ValueError('Use replace "old text" with "new text".')
        old, new = map(json.loads, parts.groups())
        if any(0xD800 <= ord(char) <= 0xDFFF for char in new):
            raise ValueError("Replacement contains an invalid Unicode surrogate.")
        start = source.find(old)
        if not old or start == -1 or source.find(old, start + 1) != -1:
            raise ValueError("The original text must occur exactly once.")
        return [(start, start + len(old), new)]
    if re.match(r"(?i)^(?:rename|set|replace)(?:\b|$)", command):
        raise ValueError(
            'Unsupported built-in edit syntax. Use "rename old to new", '
            '"set name to integer", or \'replace "old" with "new"\'.'
        )
    return None


# Phase 2 is a bounded, deterministic request interpreter, NOT a model result.
# Each inferred operation is rechecked by the existing Phase 1 verifier.
def phase2_plan(source, request):
    command = " ".join(request.strip().split())
    if not source:
        raise ValueError("Paste a code snippet first.")
    full = lambda pattern: re.fullmatch(pattern, command, flags=re.IGNORECASE)
    if full(rf"rename\s+(?:it|this|that)\s+to\s+{IDENTIFIER}\.?"):
        raise ValueError("I can't verify that edit. Name the exact variable instead of it, this, or that.")
    rename = (
        full(rf"(?:please\s+)?(?:change|rename)\s+(?:the\s+)?(?:variable|identifier)\s+({IDENTIFIER})\s+(?:to|into)\s+({IDENTIFIER})(?:\s+instead)?\.?")
        or full(rf"(?:please\s+)?(?:change|update)\s+(?:the\s+)?variable\s+name\s+({IDENTIFIER})\s+to\s+({IDENTIFIER})\.?")
        or full(rf"(?:please\s+)?rename\s+({IDENTIFIER})\s+as\s+({IDENTIFIER})\.?")
        or full(rf"(?:please\s+)?call\s+(?:the\s+)?variable\s+({IDENTIFIER})\s+({IDENTIFIER})\s+instead\.?")
    )
    if rename:
        old, new = rename.groups()
        plan = {"operation": "rename", "old": old, "new": new}
        return plan, verified_changes(source, f"rename {old} to {new}")

    implicit = full(rf"(?:please\s+)?call\s+this\s+variable\s+({IDENTIFIER})\s+instead\.?")
    if implicit:
        names = []
        for line in source.splitlines():
            line = line.partition("//")[0]
            match = DECL.fullmatch(line)
            if match:
                names.append(match[2])
        if len(names) != 1:
            raise ValueError("Name the variable to rename; this source does not have exactly one supported declaration.")
        old, new = names[0], implicit[1]
        plan = {"operation": "rename", "old": old, "new": new}
        return plan, verified_changes(source, f"rename {old} to {new}")

    integer = (
        full(rf"(?:please\s+)?(?:make|set)\s+({IDENTIFIER})\s+(?:equal\s+to|to\s+be)\s+(-?(?:0|[1-9][0-9]*))\.?")
        or full(rf"(?:please\s+)?change\s+(?:the\s+)?integer\s+({IDENTIFIER})\s+to\s+(-?(?:0|[1-9][0-9]*))\.?")
    )
    if integer:
        name, value = integer.groups()
        plan = {"operation": "set-integer", "name": name, "value": value}
        return plan, verified_changes(source, f"set {name} to {value}")

    if full(r"(?:please\s+)?copy\s+this\s+code\s+as\s+is\.?"):
        return {"operation": "copy"}, verified_changes(source, "copy exactly")

    # Only after trying bounded paraphrases should strict Phase 1 syntax
    # report its own errors. Otherwise "rename the variable X to Y" would
    # be mistaken for an invalid strict command before it could be parsed.
    changes = verified_changes(source, request)
    if changes is not None:
        return ({"operation": "verified-command", "request": request}, changes)

    # Explicit edit verbs must not silently become a model completion when
    # their targets or scopes cannot be verified.
    if re.match(r"(?i)^(?:please\s+)?(?:rename|change|update|call|set|make|replace)\b", command):
        raise ValueError(
            "I can't verify that edit. Name one variable and its new name, "
            "or use a supported set/replace command."
        )
    return None


def apply_changes(source, changes):
    parts = []
    offset = 0
    for start, end, replacement in changes:
        parts.extend((source[offset:start], replacement))
        offset = end
    parts.append(source[offset:])
    return "".join(parts)


def describe_changes(source, changes):
    """Describe the affected ranges in the buffer before the edit."""
    line_starts = [0, *(match.end() for match in re.finditer(r"\r\n?|\n", source))]

    def position(offset):
        line = bisect_right(line_starts, offset)
        return f"{line}:{offset - line_starts[line - 1] + 1}"

    spans = ", ".join(f"{position(start)}-{position(end)}" for start, end, _ in changes)
    return f"Changed spans (line:column, end exclusive): {spans}"


def show_source(source):
    display = source.replace("\r\n", "\n").replace("\r", "\n")
    print(display, end="" if display.endswith("\n") else "\n")


def load_model():
    import torch
    from plex_training.plex600 import Plex600Config, Plex600Model

    with WEIGHTS.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != DIGEST:
            raise ValueError("The weights differ from your measured step-200 checkpoint.")
    model = Plex600Model(Plex600Config())
    state = torch.load(WEIGHTS, mmap=True, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    del state
    return model.to("cuda")


def decode(model, tokenizer, tokens):
    import torch
    from plex_training.plex600_m8_eval import generate_one

    output, count, reason = generate_one(
        model, torch, tokenizer, tokens, device=torch.device("cuda"),
        max_new_tokens=96, decoding_policy="fitted-vocab-v1",
    )
    return {"text": output, "stop": reason, "generatedTokens": count, "checkpointSha256": DIGEST}


def generate(source, request):
    from plex_training.plex600_m8_eval import prompt_for
    from plex_training.tokenizer import PlexTokenizer

    tokenizer = PlexTokenizer.load(ARTIFACTS / "plex600-fresh-tokenizer")
    tokens = tokenizer.encode(prompt_for({"request": request, "inputFile": source}))
    if len(tokens) > 1024 - 96:
        raise ValueError("The supplied code and request exceed the model context. Use a smaller snippet.")
    return decode(load_model(), tokenizer, tokens)


def compare_copy_prompts():
    from plex_training import plex600_m11_source_conditioning as m11
    from plex_training import plex600_m8_eval as console
    from plex_training.tokenizer import PlexTokenizer

    request = "Return the supplied JavaScript line exactly as written."
    tokenizer = PlexTokenizer.load(ARTIFACTS / "plex600-fresh-tokenizer")
    model = load_model()
    for case in m11.cases():
        source = case["snippet"]
        prompts = {
            "console": console.prompt_for({"request": request, "inputFile": source}),
            "m11": m11.prompt_for(case),
        }
        for name, prompt in prompts.items():
            answer = decode(model, tokenizer, tokenizer.encode(prompt))
            print(json.dumps({
                "case": case["id"], "prompt": name, "expected": source, **answer,
                "exactWithEos": answer["text"] == source and answer["stop"] == "eos",
            }), flush=True)


def wsl_command(option):
    script = Path(__file__).resolve()
    linux_path = "/mnt/" + script.drive[0].lower() + script.as_posix()[2:]
    return ["wsl.exe", "-d", "Ubuntu", "--cd", REPO, "--",
           "/home/quali/.local/bin/uv", "run", "--project", "training", "--no-sync",
           "python", linux_path, option]


def suggest(source, request):
    if sys.platform != "win32":
        return generate(source, request)
    run = subprocess.run(wsl_command("--infer"), input=json.dumps([source, request]).encode(), capture_output=True)
    if run.returncode:
        detail = (run.stderr or run.stdout).replace(b"\0", b"").decode("utf-8", "replace").strip()
        raise RuntimeError(detail.splitlines()[-1] if detail else "WSL inference failed.")
    return json.loads(run.stdout)


def terminal_supports_ansi():
    if not sys.stdout.isatty() or os.environ.get("TERM") == "dumb":
        return False
    if os.name != "nt":
        return True
    import ctypes
    import msvcrt
    from ctypes import wintypes

    handle = wintypes.HANDLE(msvcrt.get_osfhandle(sys.stdout.fileno()))
    mode = wintypes.DWORD()
    kernel = ctypes.windll.kernel32
    return bool(kernel.GetConsoleMode(handle, ctypes.byref(mode))
                and kernel.SetConsoleMode(handle, mode.value | 0x0004))


class ConsoleView:
    palette = {
        "ice": "1;38;5;159", "cyan": "1;38;5;81", "blue": "38;5;75",
        "violet": "38;5;141", "muted": "38;5;246", "rule": "38;5;67",
        "added": "38;5;120", "removed": "38;5;210", "changed": "38;5;222",
    }
    logo = (
        "████▄   ██      █████   ██  ██",
        "██  ██  ██      ██       ████ ",
        "████▀   ██      ████      ██  ",
        "██      ██      ██       ████ ",
        "██      ██████  █████   ██  ██",
    )

    def __init__(self, ansi):
        self.interactive = sys.stdout.isatty()
        self.color = ansi and "NO_COLOR" not in os.environ
        try:
            "█▄▀─╭╰›…".encode(sys.stdout.encoding or "ascii")
            self.graphics = self.interactive
        except UnicodeEncodeError:
            self.graphics = False

    def ink(self, value, tone):
        return f"\033[{self.palette[tone]}m{value}\033[0m" if self.color else value

    def fit(self, value, limit):
        value = "".join(c if c.isprintable() else " " for c in str(value))
        if len(value) <= limit:
            return value
        dots = "…" if self.graphics else "..."
        return dots + value[-max(1, limit - len(dots)):]

    def dashboard(self):
        width = min(76, shutil.get_terminal_size().columns - 2)
        bar = "─" if self.graphics else "-"
        print()
        if self.graphics and width >= 56:
            side = ("", "PLEX LOCAL", "Coding console", "600M  /  M11 step 200", "")
            for row, tone, detail in zip(self.logo, ("ice", "cyan", "blue", "blue", "violet"), side):
                print("  " + self.ink(row, tone) + "    " +
                      self.ink(detail, "ice" if detail == "PLEX LOCAL" else "muted"))
        else:
            print("  " + self.ink("PLEX LOCAL", "cyan"))
            print("  " + self.ink("600M  /  M11 step 200", "muted"))
        print("  " + self.ink(bar * max(1, width - 1), "rule"))
        print("  " + self.ink("Folder   ", "muted") + self.fit(Path.cwd(), max(6, width - 10)))
        print("  " + self.ink("Model    ", "muted") + "Loads on request")
        print()
        shortcuts = (('/open "file.js"', "Open file"), ("/paste", "Paste code"), ("/help", "Commands"))
        if width >= 62:
            print("  " + "    ".join(self.ink(key, "cyan") + " " + hint for key, hint in shortcuts))
        else:
            for key, hint in shortcuts:
                print("  " + self.ink(f"{key:<16}", "cyan") + hint)
        print()

    def prompt(self, label, source, original, pending):
        if not self.interactive:
            return "plex> "
        if original is None:
            return "  " + self.ink("plex› " if self.graphics else "plex> ", "cyan")
        state = "edited" if source != original else "clean"
        if pending is not None:
            state += " / suggestion ready"
        corner, arrow = ("╭─ ", "╰─ ") if self.graphics else ("+ ", "> ")
        name = self.fit(Path(label).name, max(6, shutil.get_terminal_size().columns - len(state) - 12))
        return ("\n" + self.ink(corner, "rule") + name + "  " +
                self.ink(state, "muted" if state == "clean" else "changed") +
                "\n" + self.ink(arrow, "rule") + self.ink("plex> ", "cyan"))

    def diff_line(self, line):
        if line.startswith(("--- ", "+++ ", "@@")):
            return self.ink(line, "blue")
        if line.startswith("+"):
            return self.ink(line, "added")
        if line.startswith("-"):
            return self.ink(line, "removed")
        return line


def main():
    source = ""
    original = None
    source_path = None
    label = "pasted code"
    bom = b""
    history = []
    pending = None
    last_saved = None
    ansi = terminal_supports_ansi()
    view = ConsoleView(ansi)
    view.dashboard()
    while True:
        request = input(view.prompt(label, source, original, pending)).strip()
        if not request:
            continue
        if request.startswith("/"):
            parts = request[1:].split(maxsplit=1)
            if not parts:
                print("Type /help for commands.")
                continue
            command, *tail = parts
            command = command.lower()
            if command not in {"open", "save", "paste", "show", "diff", "undo", "discard", "accept",
                               "status", "help", "clear", "quit", "exit"}:
                print("Unknown command. Type /help.")
                continue
            request = " ".join([command, *tail])
            if command == "save" and not re.match(r"save\s+as(?:\s|$)", request, re.I):
                print('Use /save as "new.js".')
                continue
            if tail and command not in {"open", "save"}:
                print(f"/{command} takes no arguments.")
                continue
        cmd = request.lower()
        if cmd in {"quit", "exit"}:
            return
        if cmd == "clear":
            if ansi:
                print("\033[2J\033[H", end="")
            view.dashboard()
            continue
        if cmd == "discard":
            if original is None:
                print("Plex: No code is open or pasted.")
            else:
                source = original
                history.clear()
                pending = None
                last_saved = None
                print("Discarded in-memory edits. The original file was not changed.")
            continue
        if cmd == "status":
            print("\n  " + view.ink("SESSION", "cyan"))
            for key, value in (
                ("Folder", Path.cwd()), ("Source", label if original is not None else "(none)"),
                ("Buffer", f"{len(source.splitlines())} lines, {len(source)} characters"),
                ("Changes", "Edited" if original is not None and source != original else "None"),
                ("Suggestion", "Awaiting /accept" if pending is not None else "None"),
                ("Model", "Plex 600M / M11 step 200 (on demand)"),
            ):
                print("  " + view.ink(f"{key:<12}", "muted") + str(value))
            continue
        try:
            file_command = re.fullmatch(r"(open|save as)(?:\s+(.+))?", request, re.I)
            if file_command:
                action, name = file_command.groups()
                if not name:
                    raise ValueError(f'Use {action} "filename.js".')
                if name[0] in "\"'":
                    if len(name) < 2 or name[-1] != name[0]:
                        raise ValueError("Close the quote around the file path.")
                    name = name[1:-1]
                if not name:
                    raise ValueError("Supply a filename.")
                path = Path(name).expanduser()
                if action.lower() == "open":
                    if original is not None and source != original and source != last_saved:
                        raise ValueError('Unsaved edits: use /save as "copy.js" or /discard before opening another source.')
                    opened_path = path.resolve()
                    raw = opened_path.read_bytes()
                    try:
                        decoded = raw.decode("utf-8-sig")
                    except UnicodeDecodeError as exc:
                        raise ValueError(f"{path} is not a UTF-8 file.") from exc
                    source = original = decoded
                    source_path = opened_path
                    bom = codecs.BOM_UTF8 if raw.startswith(codecs.BOM_UTF8) else b""
                    label = str(path)
                    history.clear()
                    pending = None
                    last_saved = None
                    print(f"Opened {opened_path}. Type show to see the code.")
                else:
                    if original is None:
                        raise ValueError("Open or paste code before saving.")
                    if path.resolve() == source_path:
                        raise ValueError("Save as needs a new filename. The source file is protected.")
                    data = bom + source.encode("utf-8")
                    try:
                        with path.open("xb") as stream:
                            stream.write(data)
                    except FileExistsError as exc:
                        raise ValueError(f"{path} already exists. Choose a new filename.") from exc
                    last_saved = source
                    print(f"Saved {path.resolve()}")
                continue
            if cmd == "diff":
                if original == source or original is None:
                    print("No edits to show.")
                else:
                    for line in difflib.unified_diff(original.splitlines(keepends=True),
                                                     source.splitlines(keepends=True),
                                                     fromfile=label, tofile="current edit"):
                        print(view.diff_line(line.rstrip("\r\n")))
                        if not line.endswith(("\r", "\n")):
                            print(r"\ No newline at end of file")
                continue
            if cmd == "help":
                print('  /open "file.js"       Open a UTF-8 file\n'
                      '  /paste                Paste code, finishing with .end on its own line\n'
                      '  /show  /diff  /undo   Inspect the buffer, see changes, undo one edit\n'
                      '  /discard              Revert in-memory edits before opening new code\n'
                      '  /save as "new.js"     Write a new file\n'
                      '  /accept               Apply a complete model suggestion\n'
                      '  /status               Inspect the active session\n'
                      '  /clear  /exit         Redraw the dashboard or end the session\n\n'
                      '  Try: rename value to count, change the variable value to count,\n'
                      '       call this variable count instead, or make value equal to 3.\n'
                      '  Also: set count to 3, or replace "old" with "new".\n'
                      '  Verified rename supports flat integer declarations and simple console.log calls.\n'
                      '  Other edit requests use your local Plex 600M model.')
                continue
            if cmd == "paste":
                if original is not None and source != original and source != last_saved:
                    raise ValueError('Unsaved edits: use /save as "copy.js" or /discard before pasting new code.')
                print("Paste your code. End with .end on its own line.")
                lines = []
                while (line := input()) != ".end":
                    lines.append(line)
                source = original = "".join(line + "\n" for line in lines)
                source_path = None
                label = "pasted code"
                bom = b""
                history.clear()
                pending = None
                last_saved = None
                show_source(source)
                continue
            elif cmd == "undo":
                if history:
                    source = history.pop()
                    pending = None
                show_source(source)
                continue
            elif cmd == "accept":
                if pending is None:
                    raise ValueError("No complete model suggestion is pending.")
                result = pending
                print("Accepted suggestion from Plex 600M M11 step 200.")
            else:
                if original is None:
                    raise ValueError("Paste a code snippet first.")
                interpreted = phase2_plan(source, request)
                if interpreted is None:
                    pending = None
                    if not source:
                        raise ValueError("Paste a code snippet first.")
                    print("Generating locally with Plex 600M M11 step 200...", flush=True)
                    answer = suggest(source, request)
                    print(f'Checkpoint SHA-256: {answer["checkpointSha256"]}')
                    if not answer["text"].strip():
                        print(f'Plex 600M returned no code (stop: {answer["stop"]}).')
                        continue
                    print(f'Plex 600M model suggestion ({answer["stop"]}):\n{answer["text"]}')
                    if answer["stop"] == "eos":
                        if answer["text"] == source:
                            print("Model proposed no changes; nothing to accept.")
                            continue
                        print("Proposed diff (unverified model output):")
                        for line in difflib.unified_diff(
                            source.splitlines(keepends=True),
                            answer["text"].splitlines(keepends=True),
                            fromfile=label, tofile="Plex 600M suggestion",
                        ):
                            print(view.diff_line(line.rstrip("\r\n")))
                            if not line.endswith(("\r", "\n")):
                                print(r"\ No newline at end of file")
                        pending = answer["text"]
                        print("Review the diff. Type accept to use this unverified suggestion.")
                    else:
                        print("Incomplete model output; not available for acceptance.")
                    continue
                plan, changes = interpreted
                if plan["operation"] != "verified-command":
                    if plan["operation"] == "rename":
                        print(f'Understood (built-in): rename {plan["old"]} to {plan["new"]}.')
                    elif plan["operation"] == "set-integer":
                        print(f'Understood (built-in): set {plan["name"]} to {plan["value"]}.')
                    else:
                        print("Understood (built-in): copy unchanged.")
                result = apply_changes(source, changes)
                if result != source:
                    print("Built-in Python edit applied.")
                    print(describe_changes(source, changes))
                else:
                    print("Current code (built-in command).")
            if result != source:
                history.append(source)
                source, pending = result, None
            show_source(source)
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            print(view.ink("Plex: ", "removed") + str(exc))


if __name__ == "__main__":
    if sys.argv[1:] == ["--infer"]:
        print(json.dumps(generate(*json.load(sys.stdin))))
    elif sys.argv[1:] == ["--compare-copy-prompts"]:
        if sys.platform == "win32":
            sys.exit(subprocess.run(wsl_command("--compare-copy-prompts")).returncode)
        compare_copy_prompts()
    else:
        try:
            main()
        except (EOFError, KeyboardInterrupt):
            print()
