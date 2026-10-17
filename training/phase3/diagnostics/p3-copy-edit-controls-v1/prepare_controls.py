"""Prepare a bounded copy-versus-edit diagnostic; no model loading or inference.

This packet is separate from roadmap P3-03 (Insert and Delete). Only a new
candidate manifest is prepared. A model run requires separate authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
P302 = ROOT / "training/phase3/drafts/p3-02-exact-replacements-v1.jsonl"
P303 = ROOT / "training/phase3/diagnostics/p3-03-source-preservation-v1/p3-03-source-preservation-v1.json"
P302_SHA = "98d7ff276cf28d2509ad775c95c49ae0ffdb975de7e47f11e54430f3ca426403"
P303_SHA = "119eaa829825e3daa6c9156bd0ba84dfa32726dc2ebfa3d7cfd9ccf7612df578"
STEP100_SHA = "020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d"
TOKENIZER_SHA = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
OUTPUT = HERE / "copy-edit-controls-v1.json"
NAMES = ("cobalt", "spruce")
VALUES = {
    "html": ("Deploy", "Dismiss", "Restore"),
    "css": ("3px", "11px", "19px"),
    "javascript": ("4", "7", "13"),
}
SOURCE_LITERAL = {"html": "Open", "css": "8px", "javascript": "2"}
PROMPT = (
    "Edit the supplied file to satisfy the request.\n"
    "Language: {language}\nRequest: {request}\n"
    "File:\n{input_file}\nEdited file:"
)


def source_file(language: str, name: str, with_context: bool) -> str:
    if language == "html":
        target = f'<button id="{name}Button">Open</button>'
        return (f'<section data-owner="{name}">\n  {target}\n'
                f'  <p class="{name}-note">Untouched</p>\n</section>\n') if with_context else target
    if language == "css":
        return (f"/* Preserve {name} rules. */\n.{name}Grid {{\n"
                f"  display: grid;\n  gap: 8px;\n}}\n"
                f".{name}Note {{ color: #555; }}\n") if with_context else f".{name}Grid {{ gap: 8px; }}\n"
    if language == "javascript":
        target = f"const {name}Retries = 2;\n"
        return (f"// Preserve {name} settings.\n" + target +
                f"const {name}Enabled = true;\n") if with_context else target
    raise ValueError(f"Unsupported language: {language}")


def task_rows() -> list[dict[str, Any]]:
    tasks = []
    for language in ("html", "css", "javascript"):
        for namespace in NAMES:
            for condition in ("copy", "minimal-edit", "context-edit"):
                original = source_file(language, namespace, condition != "minimal-edit")
                old = SOURCE_LITERAL[language]
                if original.count(old) != 1:
                    raise ValueError("Target literal must be unique")
                start = original.index(old)
                end = start + len(old)
                for index, requested in enumerate((None,) if condition == "copy" else VALUES[language]):
                    if condition == "copy":
                        request = "Return the supplied file exactly unchanged. Make no edits."
                        target = original
                    else:
                        request = (f"Change the {'button text' if language == 'html' else 'gap' if language == 'css' else 'retry limit'} "
                                   f"to {requested}. Return the complete file. Preserve everything else exactly.")
                        target = original[:start] + requested + original[end:]
                    row = {
                        "taskId": f"p3-copy-{language}-{namespace}-{condition}-{index}",
                        "language": language,
                        "family": {"html": "button-text", "css": "gap", "javascript": "retry-limit"}[language],
                        "namespace": namespace,
                        "condition": condition,
                        "contrastGroupId": f"p3-copy-{language}-{namespace}-{condition}",
                        "contrastIndex": index,
                        "request": request,
                        "inputFile": original,
                        "expectedFile": target,
                        "prompt": PROMPT.format(language=language, request=request, input_file=original),
                        "target": {
                            "startCharacter": start, "endCharacter": end,
                            "sourceLiteral": old, "requestedLiteral": requested,
                        },
                    }
                    tasks.append(row)
    if len(tasks) != 42 or len({t["taskId"] for t in tasks}) != 42:
        raise ValueError("Copy/edit control requires 42 unique tasks")
    return tasks


def _checked_bytes(path: Path, expected: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or non-regular frozen source: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Frozen source hash mismatch: {path}")
    return raw


def _prior_file_sets() -> tuple[set[str], set[str]]:
    p302 = _checked_bytes(P302, P302_SHA)
    p303 = _checked_bytes(P303, P303_SHA)
    records = [json.loads(line) for line in p302.decode("utf-8").splitlines()]
    train = [r for r in records if r.get("candidateSplit") == "train"]
    prior = json.loads(p303.decode("utf-8"))["tasks"]
    if len(train) != 144 or len(prior) != 72:
        raise ValueError("Prior source record count mismatch")
    sources = {row["inputFile"] for row in train + prior}
    targets = {row["expectedFile"] for row in train + prior}
    return sources, targets


def make_manifest() -> dict[str, Any]:
    rows = task_rows()
    old_sources, old_targets = _prior_file_sets()
    inputs = {r["inputFile"] for r in rows}
    outputs = {r["expectedFile"] for r in rows}
    overlaps = {
        "inputsAgainstOldInputs": len(inputs & old_sources),
        "inputsAgainstOldTargets": len(inputs & old_targets),
        "targetsAgainstOldInputs": len(outputs & old_sources),
        "targetsAgainstOldTargets": len(outputs & old_targets),
    }
    if any(overlaps.values()):
        raise ValueError(f"Copy/edit control overlaps frozen prior files: {overlaps}")
    return {
        "schemaVersion": 1,
        "kind": "plex-p3-replacement-copy-versus-edit-controls-v1",
        "status": "prepared-no-model-execution-no-evaluation-authorization",
        "researchScope": "Replacement recovery; NOT roadmap P3-03 Insert and Delete",
        "model": {
            "parameters": 27566080,
            "contextLength": 512,
            "checkpointStep": 100,
            "checkpointSha256": STEP100_SHA,
            "tokenizerSha256": TOKENIZER_SHA,
        },
        "inferenceProposedNotAuthorized": {
            "greedy": True, "seed": 1337, "temperature": 0,
            "maximumNewTokens": 192, "checkpointReselectionAllowed": False,
        },
        "sourceIdentities": {"p302CandidateSha256": P302_SHA, "p303DiagnosticManifestSha256": P303_SHA},
        "overlapWithFrozenP302TrainAndP303Diagnostic": overlaps,
        "responseContract": "complete-file text; copy controls must return exact supplied bytes",
        "authorization": {
            "evaluationAuthorized": False, "trainingAuthorized": False,
            "modelLoaded": False, "optimizerUpdates": 0, "finalHoldoutOpened": False,
        },
        "conditionCounts": {"copy": 6, "minimal-edit": 18, "context-edit": 18},
        "passRule": "All 42 complete files must match expectedFile exactly; report each language and condition separately.",
        "tasks": rows,
    }


def score_one(task: dict[str, Any], actual: str) -> dict[str, Any]:
    """Model-free scoring for frozen single-literal tasks only.

    If a target cannot be isolated by preserved surrounding bytes, value
    selection is unknown rather than treated as independently measured.
    """
    if not isinstance(actual, str):
        raise ValueError("The complete decoded output must be a string")
    exact = actual == task["expectedFile"]
    if task["condition"] == "copy":
        return {"exact": exact, "unrelatedBytesPreserved": exact,
                "requestedValueCorrect": None, "targetExtractable": None}
    start = task["target"]["startCharacter"]
    end = task["target"]["endCharacter"]
    source = task["inputFile"]
    before, after = source[:start], source[end:]
    boundaries = actual.startswith(before) and actual.endswith(after)
    middle_end = len(actual) - len(after) if after else len(actual)
    middle = actual[len(before):middle_end] if boundaries else None
    return {
        "exact": exact,
        "unrelatedBytesPreserved": boundaries,
        "requestedValueCorrect": (middle == task["target"]["requestedLiteral"]) if boundaries else None,
        "targetExtractable": boundaries,
    }


def serialized_manifest() -> bytes:
    return (json.dumps(make_manifest(), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true", help="Compare bytes with previously frozen manifest")
    args = parser.parse_args()
    encoded = serialized_manifest()
    if args.verify:
        if args.output.read_bytes() != encoded:
            parser.error("Frozen manifest does not match deterministic builder")
    else:
        with args.output.open("xb") as f:
            f.write(encoded)
    print(json.dumps({"path": str(args.output), "sha256": hashlib.sha256(encoded).hexdigest(),
                      "taskCount": 42, "mode": "verify" if args.verify else "new-manifest"}))


if __name__ == "__main__":
    main()
