"""Local editor with optional suggestions from Plex's scratch-trained 600M model."""
import codecs
import difflib
import hashlib
import json
import re
import subprocess
import sys
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


def rename_variable(source, old, new, expected_value):
    """Rename a binding in a flat snippet of integer declarations and console.log calls."""
    spans = []
    names = set()
    initial = None
    offset = 0
    for line in source.splitlines(keepends=True):
        start = offset
        offset += len(line)
        code = line.rstrip("\r\n")
        if not code.strip() or code.lstrip().startswith("//"):
            continue
        declaration = DECL.fullmatch(code.partition("//")[0])
        if declaration:
            name = declaration[2]
            names.add(name)
            if name == old:
                if initial is not None:
                    raise ValueError(f"Multiple declarations of {old} need separate scopes.")
                initial = declaration[4]
                spans.append((start + declaration.start(2), start + declaration.end(2)))
            continue
        call = LOG_LINE.fullmatch(code)
        if not call:
            return None
        names.add("console")
        if old == "console":
            spans.append((start + code.index("console"), start + code.index("console") + 7))
        for arg in LOG_ARG.finditer(call[1] or ""):
            if re.fullmatch(IDENTIFIER, arg[0]):
                names.add(arg[0])
                if arg[0] == old:
                    pos = start + call.start(1)
                    spans.append((pos + arg.start(), pos + arg.end()))
    if initial is None:
        raise ValueError(f"No integer declaration named {old} in this snippet.")
    if new in RESERVED or (old != new and new in names):
        raise ValueError(f"Choose an unused JavaScript variable name other than {new}.")
    if expected_value is not None and expected_value != initial:
        raise ValueError(f"The current value is {initial}.")
    for start, end in reversed(spans):
        source = source[:start] + new + source[end:]
    return source


def verified_edit(source, request):
    command = request.strip()
    if command.lower() in {"copy", "copy exactly", "copy this code", "show"}:
        return source
    rename = re.fullmatch(
        rf"(?i:rename)\s+({IDENTIFIER})\s+(?i:to)\s+({IDENTIFIER})"
        r"(?:\s+(?i:and keep its value)(?:\s+(?i:at)\s+(-?(?:0|[1-9][0-9]*)))?)?\.?",
        command,
    )
    if rename:
        return rename_variable(source, *rename.groups())
    change = re.fullmatch(r"(?i:set)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s+(?i:to)\s+(-?(?:0|[1-9][0-9]*))", command)
    if change:
        match = DECL.fullmatch(source)
        if not match or match[2] != change[1]:
            raise ValueError("Set needs one integer declaration with that exact variable name.")
        return source[:match.start(4)] + change[2] + source[match.end(4):]
    if command.lower().startswith('replace "'):
        literal = r'"(?:\\.|[^"\\])*"'
        parts = re.fullmatch(rf"(?i:replace)\s+({literal})\s+(?i:with)\s+({literal})", command)
        if not parts:
            raise ValueError('Use replace "old text" with "new text".')
        old, new = map(json.loads, parts.groups())
        if not old or source.count(old) != 1:
            raise ValueError("The original text must occur exactly once.")
        return source.replace(old, new, 1)
    return None


def generate(source, request):
    import torch
    from plex_training.plex600 import Plex600Config, Plex600Model
    from plex_training.plex600_m8_eval import generate_one, prompt_for
    from plex_training.tokenizer import PlexTokenizer

    tokenizer = PlexTokenizer.load(ARTIFACTS / "plex600-fresh-tokenizer")
    tokens = tokenizer.encode(prompt_for({"request": request, "inputFile": source}))
    if len(tokens) > 1024 - 96:
        raise ValueError("The supplied code and request exceed the model context. Use a smaller snippet.")
    with WEIGHTS.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != DIGEST:
            raise ValueError("The weights differ from your measured step-200 checkpoint.")
    model = Plex600Model(Plex600Config())
    state = torch.load(WEIGHTS, mmap=True, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    del state
    model.to("cuda")
    output, _, reason = generate_one(model, torch, tokenizer, tokens, device=torch.device("cuda"),
                                     max_new_tokens=96, decoding_policy="fitted-vocab-v1")
    return {"text": output, "stop": reason}


def suggest(source, request):
    if sys.platform != "win32":
        return generate(source, request)
    script = Path(__file__).resolve()
    linux_path = "/mnt/" + script.drive[0].lower() + script.as_posix()[2:]
    cmd = ["wsl.exe", "-d", "Ubuntu", "--cd", REPO, "--",
           "/home/quali/.local/bin/uv", "run", "--project", "training", "--no-sync",
           "python", linux_path, "--infer"]
    run = subprocess.run(cmd, input=json.dumps([source, request]).encode(), capture_output=True)
    if run.returncode:
        detail = (run.stderr or run.stdout).replace(b"\0", b"").decode("utf-8", "replace").strip()
        raise RuntimeError(detail.splitlines()[-1] if detail else "WSL inference failed.")
    return json.loads(run.stdout)


def main():
    source = ""
    original = None
    label = "pasted code"
    bom = b""
    history = []
    pending = None
    print("Plex local | Plex 600M was trained from random weights. Model suggestions use M11 step 200.")
    print('Built-in commands run in Python. Paste code or open "file.js". Type help for commands.')
    while True:
        request = input("plex> ").strip()
        if not request:
            continue
        cmd = request.lower()
        if cmd == "quit":
            return
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
                path = Path(name).expanduser()
                if action.lower() == "open":
                    raw = path.read_bytes()
                    decoded = raw.decode("utf-8-sig")
                    source = original = decoded
                    bom = codecs.BOM_UTF8 if raw.startswith(codecs.BOM_UTF8) else b""
                    label = str(path)
                    history.clear()
                    pending = None
                    print(f"Opened {path.resolve()}. Type show to see the code.")
                else:
                    if original is None:
                        raise ValueError("Open or paste code before saving.")
                    data = bom + source.encode("utf-8")
                    with path.open("xb") as stream:
                        stream.write(data)
                    print(f"Saved {path.resolve()}")
                continue
            if cmd == "diff":
                if original == source or original is None:
                    print("No edits to show.")
                else:
                    for line in difflib.unified_diff(original.splitlines(keepends=True),
                                                     source.splitlines(keepends=True),
                                                     fromfile=label, tofile="current edit"):
                        print(line.rstrip("\r\n"))
                        if not line.endswith(("\r", "\n")):
                            print(r"\ No newline at end of file")
                continue
            if cmd == "help":
                print('Open "file.js" or paste code ending with .end. Then describe your edit.\n'
                      'show displays the code, diff displays changes, undo reverses the last edit.\n'
                      'Built-in edits include rename, set, and literal replace. Other edits may ask Plex 600M.\n'
                      'Model suggestions use the local scratch-trained M11 step-200 checkpoint.\n'
                      'accept uses a model suggestion. Save as "new.js" writes a new file. quit exits.')
                continue
            if cmd == "paste":
                print("Paste your code. End with .end on its own line.")
                lines = []
                while (line := input()) != ".end":
                    lines.append(line)
                result = "".join(line + "\n" for line in lines)
                if original is None:
                    original = result
            elif cmd == "undo":
                if history:
                    source = history.pop()
                    pending = None
                print(source, end="" if source.endswith("\n") else "\n")
                continue
            elif cmd == "accept":
                if pending is None:
                    raise ValueError("No complete model suggestion is pending.")
                result = pending
                print("Accepted suggestion from Plex 600M M11 step 200.")
            else:
                result = verified_edit(source, request)
                if result is None:
                    pending = None
                    if not source:
                        raise ValueError("Paste a code snippet first.")
                    print("Generating locally with Plex 600M M11 step 200...", flush=True)
                    answer = suggest(source, request)
                    if not answer["text"].strip():
                        print(f'Plex 600M returned no code (stop: {answer["stop"]}).')
                        continue
                    print(f'Plex 600M model suggestion ({answer["stop"]}):\n{answer["text"]}')
                    if answer["stop"] == "eos":
                        pending = answer["text"]
                        print("Type accept to use this suggestion.")
                    continue
                if result != source:
                    print("Built-in Python edit applied.")
                else:
                    print("Current code (built-in command).")
            if result != source:
                history.append(source)
                source, pending = result, None
            print(source, end="" if source.endswith("\n") else "\n")
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            print(f"Plex: {exc}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--infer"]:
        print(json.dumps(generate(*json.load(sys.stdin))))
    else:
        try:
            main()
        except (EOFError, KeyboardInterrupt):
            print()
