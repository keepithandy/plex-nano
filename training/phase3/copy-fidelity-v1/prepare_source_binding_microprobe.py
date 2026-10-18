"""P3-SB01: deterministic 48-case SOURCE-BINDING microprobe PREPARATION ONLY.

Never loads models, trains, runs inference, reads the final holdout, or selects
checkpoints. The committed candidate is a proposed/frozen fixture, NOT approval
to run it on the private model. Default action verifies its exact bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "source-binding-microprobe-v1.json"
PRIOR = {
    ROOT / "training/phase3/drafts/p3-02-exact-replacements-v1.jsonl":
        "98d7ff276cf28d2509ad775c95c49ae0ffdb975de7e47f11e54430f3ca426403",
    ROOT / "training/phase3/diagnostics/p3-03-source-preservation-v1/p3-03-source-preservation-v1.json":
        "119eaa829825e3daa6c9156bd0ba84dfa32726dc2ebfa3d7cfd9ccf7612df578",
    ROOT / "training/phase3/diagnostics/p3-copy-edit-controls-v1/copy-edit-controls-v1.json":
        "17834efb801ac963a4672ae616ca47e9eea40b2568f01951c3891c7e4997f01a",
    HERE / "copy-fidelity-candidate-v1.json":
        "9414d1da7672a386a920d1935723e9133d011c9c3a4029e5d5800f643835ca7a",
}
LANGUAGES = ("html", "css", "javascript")
CONTEXTS = ("minimal", "context")
# Independently checked against all four previous frozen candidate/control
# artifacts by FULL-BYTE matching plus a separate source/target overlap test.
NAMES = ("ulvar", "nyxen")
PROMPT = (
    "Edit the supplied file to satisfy the request.\n"
    "Language: {language}\nRequest: {request}\n"
    "File:\n{input_file}\nEdited file:"
)
MAX_NEW_TOKENS = 192


def _input_template(language: str, name: str, context: str) -> tuple[str, str, tuple[str, ...], str]:
    if language == "html":
        old, options, kind = "Open", ("Start", "Close", "Resume"), "button text"
        line = f'<button id="{name}-launch">Open</button>\n'
        source = (
            f'<section data-owner="{name}">\n  {line.strip()}\n'
            f'  <p class="{name}-hint">Keep this note.</p>\n</section>\n'
        ) if context == "context" else line
    elif language == "css":
        old, options, kind = "8px", ("3px", "11px", "19px"), "CSS gap"
        line = f".{name}Grid {{ gap: 8px; }}\n"
        source = (
            f"/* Preserve {name} rules. */\n.{name}Grid {{\n"
            f"  display: grid;\n  gap: 8px;\n}}\n"
            f".{name}Note {{ color: #555; }}\n"
        ) if context == "context" else line
    elif language == "javascript":
        old, options, kind = "2", ("4", "7", "13"), "retry count"
        line = f"const {name}Retries = 2;\n"
        source = (
            f"// Preserve {name} settings.\n"
            f"const {name}Retries = 2;\n"
            f"const {name}Enabled = true;\n"
        ) if context == "context" else line
    else:
        raise ValueError("Unsupported language")
    if not source.isascii() or source.count(old) != 1:
        raise ValueError("Source and unique literal contract violated")
    return source, old, options, kind


def _checked_prior() -> tuple[set[str], set[str]]:
    sources: set[str] = set()
    targets: set[str] = set()
    for path, digest in PRIOR.items():
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size > 4 * 1024 * 1024):
            raise ValueError("Frozen prior corpus is unavailable, linked or oversized")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Frozen prior corpus SHA changed")
        # Strong namespace disjointness: no source/expected file anywhere in
        # earlier experiments may contain either proposed microprobe namespace.
        for name in NAMES:
            if name.encode("ascii") in raw:
                raise ValueError(f"New source namespace collides with prior corpus: {name}")
        if path.suffix == ".jsonl":
            previous = [json.loads(x) for x in raw.splitlines() if x]
        else:
            data = json.loads(raw)
            previous = data["rows"] if "rows" in data else data["tasks"]
        for record in previous:
            sources.add(record["inputFile"])
            targets.add(record["expectedFile"])
    return sources, targets


def records() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for language in LANGUAGES:
        for name in NAMES:
            for context in CONTEXTS:
                source, old, options, kind = _input_template(language, name, context)
                start = source.index(old)
                for choice, value in enumerate((None,) + options):
                    condition = "copy" if choice == 0 else "edit"
                    request = (
                        "Return the supplied file exactly unchanged. Make no edits."
                        if choice == 0
                        else f"Change the {kind} to {value}. Return the complete file. Preserve everything else exactly."
                    )
                    expected = (source if value is None else
                                source[:start] + value + source[start + len(old):])
                    prompt = PROMPT.format(
                        language=language, request=request, input_file=source
                    )
                    ident = f"p3-sb01-{language}-{name}-{context}-{choice}"
                    if (not prompt.isascii()
                            or len(prompt.encode("utf-8")) + MAX_NEW_TOKENS + 1 > 512
                            or len(expected.encode("utf-8")) + 1 > MAX_NEW_TOKENS):
                        raise ValueError("Conservative 512/192 token budget exceeded")
                    entries.append({
                        "id": ident, "language": language, "sourceNamespace": name,
                        "context": context, "condition": condition,
                        "groupId": f"p3-sb01-{language}-{name}-{context}",
                        "contrastIndex": choice, "request": request,
                        "inputFile": source, "expectedFile": expected,
                        "prompt": prompt, "sourceLiteral": old,
                        "requestedLiteral": value, "targetStart": start,
                        "targetEnd": start + len(old),
                    })
    return entries


def make_manifest() -> dict[str, Any]:
    old_source, old_target = _checked_prior()
    entries = records()
    if (len(entries) != 48 or len({r["id"] for r in entries}) != 48
            or Counter(r["condition"] for r in entries) != {"copy": 12, "edit": 36}
            or Counter(r["language"] for r in entries) != {
                "html": 16, "css": 16, "javascript": 16,
            } or Counter(r["context"] for r in entries) != {
                "minimal": 24, "context": 24,
            }):
        raise ValueError("SB01 record shape changed")
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in entries:
        groups[r["groupId"]].append(r)
        if (r["sourceNamespace"] not in r["inputFile"]
                or r["sourceNamespace"] not in r["expectedFile"]
                or (r["expectedFile"] != (
                    r["inputFile"] if r["condition"] == "copy" else
                    r["inputFile"][:r["targetStart"]] + r["requestedLiteral"]
                    + r["inputFile"][r["targetEnd"]:]
                ))):
            raise ValueError("Input identifier or exact single literal edit changed")
    if (len(groups) != 12
            or any(len(rows) != 4
                   or {row["contrastIndex"] for row in rows} != {0, 1, 2, 3}
                   or len({row["inputFile"] for row in rows}) != 1
                   or len({row["expectedFile"] for row in rows}) != 4
                   for rows in groups.values())):
        raise ValueError("Four-case contrast groups changed")
    source_files = {r["inputFile"] for r in entries}
    target_files = {r["expectedFile"] for r in entries}
    overlap = {
        "newInputsInPreviousInputs": len(source_files & old_source),
        "newInputsInPreviousTargets": len(source_files & old_target),
        "newTargetsInPreviousInputs": len(target_files & old_source),
        "newTargetsInPreviousTargets": len(target_files & old_target),
    }
    if any(overlap.values()):
        raise ValueError("Microprobe source/target overlaps prior experiments")
    return {
        "schemaVersion": 1,
        "kind": "p3-source-binding-microprobe-preparation-v1",
        "status": "frozen-proposal-only-evaluation-not-authorized",
        "researchScope": "P3-SB01 new source-name copy and one-literal edit contrasts",
        "seed": 1337,
        "responseContract": "complete supplied source file; character-exact whitespace and identifiers",
        "generationProposalNotAuthorized": {
            "temperature": 0, "seed": 1337, "maximumNewTokens": MAX_NEW_TOKENS,
            "checkpointReselectionAllowed": False,
        },
        "authorization": {
            "modelLoaded": False, "trainingPerformed": False,
            "optimizerUpdates": 0, "evaluationAuthorized": False,
            "finalHoldoutOpened": False,
        },
        "originalResearchCheckpointSha256": (
            "c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230"
        ),
        "sourceNamespaceSplit": {
            "microprobe": list(NAMES),
            "priorTrain": ["cedar", "lunar", "prism", "amber", "zephyr", "meadow"],
            "priorValidation": ["granite", "saffron"],
            "priorDeveloper": ["cobalt", "spruce"],
        },
        "historicalSourceSha256": {
            path.relative_to(ROOT).as_posix(): digest
            for path, digest in PRIOR.items()
        },
        "counts": {
            "records": 48, "copies": 12, "edits": 36,
            "fourCaseGroups": 12, "casesPerLanguage": 16,
            "copiesPerLanguage": 4, "editsPerLanguage": 12,
            "overlapWithPreviousFiles": overlap,
            "maximumPromptBytes": max(len(r["prompt"].encode("utf-8")) for r in entries),
            "maximumAnswerBytes": max(len(r["expectedFile"].encode("utf-8")) for r in entries),
        },
        "proposedPassGates": {
            "copy": "12/12 exact",
            "edits": "at least 30/36 exact",
            "perLanguage": "at least 10/12 exact edits",
            "completeFourCaseGroups": "at least 9/12",
            "sourceNamespace": "preserved exactly in all fully correct outputs",
        },
        "rows": entries,
    }


def serialized() -> bytes:
    # Stable key sorting, compact JSON and final LF enable exact byte hashing.
    return (json.dumps(make_manifest(), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true",
                        help="Create candidate once only; NOT inference or training authorization")
    args = parser.parse_args()
    try:
        payload = serialized()
        if args.prepare:
            with OUTPUT.open("xb") as handle:
                handle.write(payload)
        elif OUTPUT.read_bytes() != payload:
            raise ValueError("Frozen candidate differs from deterministic builder")
        print(json.dumps({
            "status": "p3-source-binding-microprobe-prepared-no-inference"
            if args.prepare else "p3-source-binding-microprobe-byte-verified",
            "records": 48,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "trainingPerformed": False,
            "evaluationAuthorized": False,
            "finalHoldoutOpened": False,
        }, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"p3-sb01-microprobe: {exc}\n")


if __name__ == "__main__":
    main()
