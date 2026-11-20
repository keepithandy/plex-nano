"""Run the documented Phase 0 example with an isolated, standard-library Python."""
import difflib
from pathlib import Path
import subprocess
import sys
import tempfile

CONSOLE = Path(__file__).resolve().with_name("PlexLocal.py")
BEFORE = 'let value = 1;\nconsole.log("value", value);\n'
AFTER = 'let count = 1;\nconsole.log("value", count);\n'


def main():
    with tempfile.TemporaryDirectory() as directory:
        commands = ("paste\n" + BEFORE + ".end\nrename value to count\ndiff\n"
                    + 'save as "result.js"\nquit\n')
        run = subprocess.run([sys.executable, "-I", "-S", str(CONSOLE)],
                             cwd=directory, input=commands, capture_output=True,
                             text=True, timeout=20, check=True)
        diff = "".join(difflib.unified_diff(BEFORE.splitlines(True), AFTER.splitlines(True),
                                            fromfile="pasted code", tofile="current edit"))
        for marker in ("Built-in Python edit applied.", diff, "M11 step 200"):
            if marker not in run.stdout:
                raise RuntimeError(f"Missing console output: {marker!r}\n{run.stdout}")
        if (Path(directory) / "result.js").read_bytes() != AFTER.encode():
            raise RuntimeError("Saved JavaScript differs from the frozen example.")
    print("PASS: built-in rename, diff and saved bytes (1 console case).")


if __name__ == "__main__":
    main()
