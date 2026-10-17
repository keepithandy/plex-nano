"""Design/freeze a model-free P3 copy-fidelity curriculum. Never loads weights or trains.

Train/validation are split by fresh namespaces; the original development
diagnostics and final holdout remain excluded from both candidate splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "copy-fidelity-candidate-v1.json"
P302 = ROOT / "training/phase3/drafts/p3-02-exact-replacements-v1.jsonl"
P303 = ROOT / "training/phase3/diagnostics/p3-03-source-preservation-v1/p3-03-source-preservation-v1.json"
COPY_CONTROL = ROOT / "training/phase3/diagnostics/p3-copy-edit-controls-v1/copy-edit-controls-v1.json"
FROZEN_SOURCES = {
    P302: "98d7ff276cf28d2509ad775c95c49ae0ffdb975de7e47f11e54430f3ca426403",
    P303: "119eaa829825e3daa6c9156bd0ba84dfa32726dc2ebfa3d7cfd9ccf7612df578",
    COPY_CONTROL: "17834efb801ac963a4672ae616ca47e9eea40b2568f01951c3891c7e4997f01a",
}
TRAIN_NAMES = ("cedar", "lunar", "prism", "amber", "zephyr", "meadow")
VALIDATION_NAMES = ("granite", "saffron")
LANGUAGES = ("html", "css", "javascript")
SHAPES = ("a", "b")
CONDITIONS = ("copy", "edit")
CONTEXTS = ("minimal", "context")
PROMPT = ("Edit the supplied file to satisfy the request.\n"
          "Language: {language}\nRequest: {request}\n"
          "File:\n{input_file}\nEdited file:")
STEP100_SHA256 = "020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d"
TOKENIZER_SHA256 = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
MAX_NEW_TOKENS = 192


def _template(language: str, name: str, shape: str, context: str) -> tuple[str, str, tuple[str, ...], str]:
    """Return the source, unique old literal, three possible edits, target kind."""
    if language == "html":
        if shape == "a":
            line = f'<button id="{name}-launch">Open</button>'
            old, values = "Open", ("Start", "Close", "Resume")
        else:
            line = f'<button class="{name}Action">Ready</button>'
            old, values = "Ready", ("Publish", "Archive", "Review")
        source = (f'<article data-owner="{name}">\n  {line}\n'
                  f'  <p class="{name}-hint">Retain this.</p>\n</article>\n') if context == "context" else line + "\n"
        return source, old, values, "button text"
    if language == "css":
        if shape == "a":
            line, old, values = f".{name}Grid {{ gap: 8px; }}\n", "8px", ("3px", "11px", "19px")
        else:
            line, old, values = f".{name}Layout {{ gap: 6px; }}\n", "6px", ("4px", "10px", "14px")
        source = (f"/* Retain {name} colors. */\n" + line
                  + f".{name}Badge {{ color: #555; }}\n") if context == "context" else line
        return source, old, values, "gap"
    if language == "javascript":
        if shape == "a":
            line, old, values = f"const {name}Retries = 2;\n", "2", ("4", "7", "13")
        else:
            line, old, values = f"let {name}Attempts = 3;\n", "3", ("5", "9", "12")
        source = (f"// Retain {name} flag.\n" + line
                  + f"const {name}Enabled = true;\n") if context == "context" else line
        return source, old, values, "numeric limit"
    raise ValueError(f"Invalid language {language}")


def rows() -> list[dict[str, Any]]:
    result = []
    for split, names in (("train", TRAIN_NAMES), ("validation", VALIDATION_NAMES)):
        for language in LANGUAGES:
            for name in names:
                for shape in SHAPES:
                    for context in CONTEXTS:
                        source, old, values, target_kind = _template(language, name, shape, context)
                        if source.count(old) != 1 or not source.isascii():
                            raise ValueError("The source must be ASCII with a unique target literal")
                        start = source.index(old)
                        for index, new in enumerate((None,) + values):
                            condition = "copy" if index == 0 else "edit"
                            if condition == "copy":
                                request = ("Return the supplied file exactly unchanged. Make no edits."
                                           if context == "context" else
                                           "Copy the complete supplied file without changing any characters.")
                                expected = source
                            else:
                                request = (f"Change the {target_kind} to {new}. "
                                           "Return the complete file and preserve everything else exactly.")
                                expected = source[:start] + new + source[start + len(old):]
                            ident = f"p3-cf-{split}-{language}-{name}-{shape}-{context}-{index}"
                            prompt = PROMPT.format(language=language, request=request, input_file=source)
                            if (not prompt.isascii() or len(prompt.encode("utf-8")) + MAX_NEW_TOKENS + 1 > 512
                                or len(expected.encode("utf-8")) + 1 > MAX_NEW_TOKENS):
                                raise ValueError(f"Byte-conservative tokenizer/output budget failure: {ident}")
                            result.append({
                                "id": ident, "split": split, "language": language,
                                "namespace": name, "shape": shape, "context": context,
                                "condition": condition,
                                "groupId": f"p3-cf-{split}-{language}-{name}-{shape}-{context}",
                                "contrastIndex": index, "request": request, "inputFile": source,
                                "expectedFile": expected, "prompt": prompt,
                                "sourceLiteral": old, "requestedLiteral": new,
                                "targetStart": start, "targetEnd": start + len(old),
                            })
    return result


def _read_frozen(path: Path, digest: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular historical frozen input: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Historical frozen source hash changed: {path}")
    return raw


def check_rows(candidate: list[dict[str, Any]]) -> dict[str, Any]:
    if len(candidate) != 384 or len({r["id"] for r in candidate}) != 384:
        raise ValueError("Expected exactly 384 distinct candidate records")
    by_split = Counter(r["split"] for r in candidate)
    if by_split != {"train": 288, "validation": 96}:
        raise ValueError(f"Train/validation count mismatch: {by_split}")
    if set(TRAIN_NAMES) & set(VALIDATION_NAMES):
        raise ValueError("Train and validation namespaces overlap")
    per_group = defaultdict(list)
    by_field = {split: {field: set() for field in ("inputFile", "expectedFile", "prompt", "groupId")}
                for split in ("train", "validation")}
    for r in candidate:
        per_group[r["groupId"]].append(r)
        for field in by_field[r["split"]]:
            by_field[r["split"]][field].add(r[field])
        if r["expectedFile"] != (r["inputFile"] if r["condition"] == "copy" else
                                  r["inputFile"][:r["targetStart"]] + r["requestedLiteral"]
                                  + r["inputFile"][r["targetEnd"]:]):
            raise ValueError(f"Nonexact replacement: {r['id']}")
    if len(per_group) != 96 or any(len(group) != 4 or
           {r["contrastIndex"] for r in group} != {0, 1, 2, 3} or
           len({r["inputFile"] for r in group}) != 1 or
           len({r["split"] for r in group}) != 1 or
           len({r["expectedFile"] for r in group}) != 4
           for group in per_group.values()):
        raise ValueError("Split-source or four-case contrast group changed")
    split_overlap = {
        field: len(by_field["train"][field] & by_field["validation"][field])
        for field in ("inputFile", "expectedFile", "prompt", "groupId")
    }
    if any(split_overlap.values()):
        raise ValueError(f"Training-validation contamination: {split_overlap}")
    p302 = [json.loads(line) for line in _read_frozen(P302, FROZEN_SOURCES[P302]).splitlines()]
    p303 = json.loads(_read_frozen(P303, FROZEN_SOURCES[P303]))["tasks"]
    controls = json.loads(_read_frozen(COPY_CONTROL, FROZEN_SOURCES[COPY_CONTROL]))["tasks"]
    if len(p302) != 216 or len(p303) != 72 or len(controls) != 42:
        raise ValueError("Historical record counts changed")
    old = p302 + p303 + controls
    old_files = {r[field] for r in old for field in ("inputFile", "expectedFile")}
    candidate_files = {r[field] for r in candidate for field in ("inputFile", "expectedFile")}
    historical_overlap = len(old_files & candidate_files)
    if historical_overlap:
        raise ValueError(f"Candidate copies prior development/training files: {historical_overlap}")
    # Check namespace tokens, not just whole-file identity. Previously measured
    # controls must never enter this train/validation curriculum.
    historical_control_names = {r["namespace"] for r in controls}
    all_new_names = set(TRAIN_NAMES) | set(VALIDATION_NAMES)
    if all_new_names & historical_control_names:
        raise ValueError("Copied an existing frozen control namespace")
    return {
        "trainRecords": 288, "validationRecords": 96,
        "perLanguage": {lang: {"train": 96, "validation": 32} for lang in LANGUAGES},
        "perSplitCondition": {"train": {"copy": 72, "edit": 216},
                              "validation": {"copy": 24, "edit": 72}},
        "perSplitContext": {"train": {"minimal": 144, "context": 144},
                            "validation": {"minimal": 48, "context": 48}},
        "fourCaseGroups": 96, "splitOverlap": split_overlap,
        "historicalCompleteFileOverlap": historical_overlap,
        "frozenPriorRecords": {"p302": 216, "p303": 72, "copyControls": 42},
        "maximumPromptBytes": max(len(r["prompt"].encode("utf-8")) for r in candidate),
        "maximumAnswerBytes": max(len(r["expectedFile"].encode("utf-8")) for r in candidate),
    }


def manifest() -> dict[str, Any]:
    candidate = rows()
    counts = check_rows(candidate)
    return {
        "schemaVersion": 1, "kind": "p3-copy-fidelity-candidate-v1",
        "status": "frozen-preparation-only-not-training-authorized",
        "model": {"parameters": 27566080, "contextLength": 512,
                  "vocabSize": 16384, "tokenizerSha256": TOKENIZER_SHA256,
                  "startingCheckpointStep": 100, "startingCheckpointSha256": STEP100_SHA256},
        "promptTemplate": PROMPT,
        "proposedRunNotAuthorized": {
            "maximumOptimizerUpdates": 100, "maximumWallTimeSeconds": 600,
            "optimizer": "AdamW", "learningRate": 0.0001, "microBatch": 1,
            "gradientAccumulation": 16, "seed": 1337,
            "supervision": "answer-and-eos-only-next-token-v1",
            "validationSteps": [0, 25, 50, 75, 100],
            "checkpointReselectionAllowed": False, "resumeOrAutomaticContinuation": False,
            "generation": {"temperature": 0, "seed": 1337, "maximumNewTokens": MAX_NEW_TOKENS},
        },
        "authorization": {"dataPreparationOnly": True, "trainingAuthorized": False,
                          "evaluationAuthorized": False, "trainingPerformed": False,
                          "optimizerUpdatesPerformed": 0, "finalHoldoutOpened": False},
        "sourceSha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in FROZEN_SOURCES.items()},
        "namespaceSplits": {"train": list(TRAIN_NAMES), "validation": list(VALIDATION_NAMES)},
        "counts": counts, "rows": candidate,
    }


def serialized() -> bytes:
    return (json.dumps(manifest(), sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify immutable frozen manifest bytes")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    encoded = serialized()
    if args.verify:
        if args.output.read_bytes() != encoded:
            parser.error("Frozen curriculum differs from deterministic authored generator")
    else:
        with args.output.open("xb") as stream:
            stream.write(encoded)
    print(json.dumps({"records": 384, "status": "verified" if args.verify else "created",
                      "sha256": hashlib.sha256(encoded).hexdigest(),
                      "maximumPromptBytes": manifest()["counts"]["maximumPromptBytes"]}))


if __name__ == "__main__":
    main()
