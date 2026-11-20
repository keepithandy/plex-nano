"""Run the documented Phase 0 example with an isolated, standard-library Python."""
import difflib
from pathlib import Path
import subprocess
import sys
import tempfile

CONSOLE = Path(__file__).resolve().with_name("PlexLocal.py")
BEFORE = 'let value = 1;\nconsole.log("value", value);\n'
AFTER = 'let count = 1;\nconsole.log("value", count);\n'


def run_console(directory, commands):
    return subprocess.run([sys.executable, "-I", "-S", str(CONSOLE)],
                          cwd=directory, input=commands, capture_output=True,
                          text=True, timeout=20, check=True).stdout


def main():
    with tempfile.TemporaryDirectory() as directory:
        commands = ("paste\n" + BEFORE + ".end\nrename value to count\ndiff\n"
                    + 'save as "result.js"\nquit\n')
        transcript = run_console(directory, commands)
        diff = "".join(difflib.unified_diff(BEFORE.splitlines(True), AFTER.splitlines(True),
                                            fromfile="pasted code", tofile="current edit"))
        for marker in ("Built-in Python edit applied.", diff, "M11 step 200"):
            if marker not in transcript:
                raise RuntimeError(f"Missing console output: {marker!r}\n{transcript}")
        if (Path(directory) / "result.js").read_bytes() != AFTER.encode():
            raise RuntimeError("Saved JavaScript differs from the frozen example.")
        request = "Return the supplied JavaScript line exactly as written."
        for name in ("score", "value"):
            for digit in (0, 1):
                line = f"let {name} = {digit};\n"
                copy_name = f"{name}-{digit}.js"
                commands = f'paste\n{line}.end\n{request}\nsave as "{copy_name}"\nquit\n'
                transcript = run_console(directory, commands)
                if f"Current code (built-in command).\n{line}" not in transcript:
                    raise RuntimeError(f"Copy was not handled by the editor:\n{transcript}")
                if (Path(directory) / copy_name).read_bytes() != line.encode():
                    raise RuntimeError(f"Copy mismatch for {copy_name}")
    print("PASS: rename and diff, plus four byte-exact built-in copies.")


if __name__ == "__main__":
    main()
