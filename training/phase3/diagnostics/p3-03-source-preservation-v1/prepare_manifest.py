"""Build the frozen, unexecuted P3-03 diagnostic manifest.

This script only writes a prospective task manifest. It never loads a model,
checkpoint, tokenizer, saved response, or project holdout. Existing manifests
are never overwritten. It reads the frozen P3-02 authored candidate only to
verify its identity, compare tasks with its authored training split, and pin
the later training-target-overlap audit.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = HERE / "p3-03-source-preservation-v1.json"
P302_CANDIDATE = ROOT / "training/phase3/drafts/p3-02-exact-replacements-v1.jsonl"
P302_CANDIDATE_SHA256 = "98d7ff276cf28d2509ad775c95c49ae0ffdb975de7e47f11e54430f3ca426403"
P302_TRAIN_RECORDS = 144
CHECKPOINT_SHA256 = "020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d"
TOKENIZER_SHA256 = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
P302_BUNDLE_MANIFEST_SHA256 = "0cd6df3a6d8eec1d7da8c7a02af067b15299393d8b0e1756a071be17b9aca354"

PROMPT_TEMPLATE = (
    "Edit the supplied file to satisfy the request.\n"
    "Language: {language}\n"
    "Request: {request}\n"
    "File:\n"
    "{input_file}\n"
    "Edited file:"
)

NAMESPACES = ("harbor", "orbit")

FAMILIES: dict[str, list[dict[str, Any]]] = {
    "html": [
        {
            "family": "button-text",
            "requestTemplate": "Change the button text to {value}. Return the complete file. Preserve everything else exactly.",
            "inputTemplate": '<button id="{n}Action">Open</button>',
            "sourceLiteral": "Open",
            "values": ["Launch", "Inspect", "Reset"],
            "extraction": {"kind": "html-text", "tag": "button"},
        },
        {
            "family": "link-destination",
            "requestTemplate": "Change the link destination to {value}. Return the complete file. Preserve everything else exactly.",
            "inputTemplate": '<a class="{n}-link" href="/start">Go</a>',
            "sourceLiteral": "/start",
            "values": ["/alpha", "/beta", "/gamma"],
            "extraction": {"kind": "html-attribute", "tag": "a", "attribute": "href"},
        },
        {
            "family": "placeholder",
            "requestTemplate": "Change the input placeholder to {value}. Return the complete file. Preserve everything else exactly.",
            "inputTemplate": '<input id="{n}Query" placeholder="Search">',
            "sourceLiteral": "Search",
            "values": ["Find files", "Find pages", "Find people"],
            "extraction": {"kind": "html-attribute", "tag": "input", "attribute": "placeholder"},
        },
        {
            "family": "aria-label",
            "requestTemplate": "Change the navigation aria-label to {value}. Return the complete file. Preserve everything else exactly.",
            "inputTemplate": '<nav id="{n}-nav" aria-label="Main menu">Links</nav>',
            "sourceLiteral": "Main menu",
            "values": ["Quick links", "Site links", "Help links"],
            "extraction": {"kind": "html-attribute", "tag": "nav", "attribute": "aria-label"},
        },
    ],
    "css": [
        {
            "family": "gap",
            "requestTemplate": "Change the gap to {value}. Return the complete file. Preserve everything else exactly.",
            "inputTemplate": "/* Keep {n} rule. */\n.{n}-panel {{\n  display: grid;\n  gap: 8px;\n}}\n",
            "sourceLiteral": "8px",
            "values": ["4px", "12px", "20px"],
            "extraction": {"kind": "css-declaration", "property": "gap"},
        },
        {
            "family": "padding",
            "requestTemplate": "Change the padding to {value}. Return the complete file. Preserve everything else exactly.",
            "inputTemplate": "/* Keep {n} rule. */\n.{n}-panel {{\n  display: block;\n  padding: 8px;\n}}\n",
            "sourceLiteral": "8px",
            "values": ["6px", "14px", "22px"],
            "extraction": {"kind": "css-declaration", "property": "padding"},
        },
        {
            "family": "color",
            "requestTemplate": "Change the text color to {value}. Return the complete file. Preserve everything else exactly.",
            "inputTemplate": "/* Keep {n} rule. */\n.{n}-label {{\n  font-weight: 600;\n  color: #111111;\n}}\n",
            "sourceLiteral": "#111111",
            "values": ["#224466", "#663322", "#226633"],
            "extraction": {"kind": "css-declaration", "property": "color"},
        },
        {
            "family": "radius",
            "requestTemplate": "Change the border radius to {value}. Return the complete file. Preserve everything else exactly.",
            "inputTemplate": "/* Keep {n} rule. */\n.{n}-card {{\n  border: 1px solid #ccc;\n  border-radius: 4px;\n}}\n",
            "sourceLiteral": "4px",
            "values": ["2px", "10px", "18px"],
            "extraction": {"kind": "css-declaration", "property": "border-radius"},
        },
    ],
    "javascript": [
        {
            "family": "retry-limit",
            "requestTemplate": "Change the retry limit to {value}. Return the complete file. Preserve everything else exactly.",
            "inputTemplate": "// Keep this note for {n}.\nconst {n}Retries = 2;\nconst {n}Enabled = true;\n",
            "sourceLiteral": "2",
            "values": ["3", "6", "9"],
            "extraction": {"kind": "js-number", "identifierSuffix": "Retries"},
        },
        {
            "family": "message",
            "requestTemplate": "Change the message to {value}. Return the complete file. Preserve everything else exactly.",
            "inputTemplate": '// Keep this note for {n}.\nconst {n}Message = "Ready";\nconst {n}Enabled = true;\n',
            "sourceLiteral": '"Ready"',
            "values": ['"Queued"', '"Paused"', '"Done"'],
            "extraction": {"kind": "js-string", "identifierSuffix": "Message"},
        },
        {
            "family": "selector",
            "requestTemplate": "Change the selector to {value}. Return the complete file. Preserve everything else exactly.",
            "inputTemplate": '// Keep this note for {n}.\nconst {n}Target = document.querySelector("#old");\nconst {n}Enabled = true;\n',
            "sourceLiteral": '"#old"',
            "values": ['"#one"', '"#two"', '"#three"'],
            "extraction": {"kind": "js-selector", "identifierSuffix": "Target"},
        },
        {
            "family": "default-size",
            "requestTemplate": "Change the default size to {value}. Return the complete file. Preserve everything else exactly.",
            "inputTemplate": "// Keep this note for {n}.\nconst {n}Size = 10;\nconst {n}Enabled = true;\n",
            "sourceLiteral": "10",
            "values": ["12", "16", "24"],
            "extraction": {"kind": "js-number", "identifierSuffix": "Size"},
        },
    ],
}


def _preservation_checks(language: str, family: str, namespace: str) -> list[dict[str, str]]:
    if language == "html":
        choices = {
            "button-text": ("button id attribute", "button", "id", f"{namespace}Action", "exact"),
            "link-destination": ("anchor class name", "a", "class", f"{namespace}-link", "class-token"),
            "placeholder": ("input id attribute", "input", "id", f"{namespace}Query", "exact"),
            "aria-label": ("navigation id attribute", "nav", "id", f"{namespace}-nav", "exact"),
        }
        role, tag, attribute, expected, match = choices[family]
        return [{"role": role, "kind": "html-attribute", "tag": tag,
                 "attribute": attribute, "expected": expected, "match": match}]
    if language == "css":
        selector = {
            "gap": f".{namespace}-panel",
            "padding": f".{namespace}-panel",
            "color": f".{namespace}-label",
            "radius": f".{namespace}-card",
        }[family]
        return [
            {"role": "first rule selector", "kind": "css-first-selector", "expected": selector},
            {"role": "preserved namespace comment", "kind": "css-comment", "expected": f"/* Keep {namespace} rule. */"},
        ]
    target_name = {
        "retry-limit": f"{namespace}Retries",
        "message": f"{namespace}Message",
        "selector": f"{namespace}Target",
        "default-size": f"{namespace}Size",
    }[family]
    return [
        {"role": "preserved namespace comment", "kind": "js-first-line", "expected": f"// Keep this note for {namespace}."},
        {"role": "editable declaration identifier", "kind": "js-identifier", "expected": target_name},
        {"role": "unrelated declaration identifier", "kind": "js-identifier", "expected": f"{namespace}Enabled"},
    ]


def _load_training_split() -> tuple[list[dict[str, Any]], set[str], set[str]]:
    if P302_CANDIDATE.is_symlink() or not P302_CANDIDATE.is_file():
        raise ValueError("Frozen P3-02 authored candidate is unavailable or not a regular file")
    raw = P302_CANDIDATE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != P302_CANDIDATE_SHA256:
        raise ValueError(f"P3-02 candidate identity mismatch: {digest}")
    train: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith('{"candidateSplit":"train",'):
            train.append(json.loads(line))
    if len(train) != P302_TRAIN_RECORDS:
        raise ValueError(f"Expected {P302_TRAIN_RECORDS} P3-02 training rows; found {len(train)}")
    if any(type(row.get(key)) is not str for row in train for key in ("inputFile", "expectedFile")):
        raise ValueError("P3-02 training row is missing an input or target file")
    return train, {row["inputFile"] for row in train}, {row["expectedFile"] for row in train}


def _task_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    tasks: list[dict[str, Any]] = []
    templates: list[dict[str, str]] = []
    for language in ("html", "css", "javascript"):
        for spec in FAMILIES[language]:
            family = spec["family"]
            templates.append({"language": language, "family": family,
                              "template": spec["requestTemplate"]})
            for namespace in NAMESPACES:
                group_id = f"p303-{language}-{family}-{namespace}"
                input_file = spec["inputTemplate"].format(n=namespace)
                source_literal = spec["sourceLiteral"]
                if input_file.count(source_literal) != 1:
                    raise ValueError(f"Expected one authorized source literal in {group_id}")
                start = input_file.index(source_literal)
                end = start + len(source_literal)
                prefix, suffix = input_file[:start], input_file[end:]
                for index, requested_literal in enumerate(spec["values"]):
                    request = spec["requestTemplate"].format(value=requested_literal)
                    expected_file = prefix + requested_literal + suffix
                    task_id = f"{group_id}-{index}"
                    prompt = PROMPT_TEMPLATE.format(language=language, request=request, input_file=input_file)
                    tasks.append({
                        "taskId": task_id,
                        "language": language,
                        "family": family,
                        "sourceNamespace": namespace,
                        "contrastGroupId": group_id,
                        "contrastIndex": index,
                        "request": request,
                        "inputFile": input_file,
                        "expectedFile": expected_file,
                        "prompt": prompt,
                        "target": {
                            "sourceLiteral": source_literal,
                            "requestedLiteral": requested_literal,
                            "startCharacter": start,
                            "endCharacter": end,
                            "extraction": spec["extraction"],
                        },
                        "sourceNameRoles": _preservation_checks(language, family, namespace),
                    })
    if len(tasks) != 72 or len({t["contrastGroupId"] for t in tasks}) != 24:
        raise ValueError("P3-03 requires exactly 72 tasks in 24 contrast groups")
    language_counts = {language: sum(task["language"] == language for task in tasks)
                       for language in ("html", "css", "javascript")}
    if len(templates) != 12 or language_counts != {"html": 24, "css": 24, "javascript": 24}:
        raise ValueError("P3-03 requires 12 family templates and 24 tasks per language")
    for group_id in sorted({t["contrastGroupId"] for t in tasks}):
        group = [t for t in tasks if t["contrastGroupId"] == group_id]
        if (len(group) != 3 or len({t["inputFile"] for t in group}) != 1
                or len({t["expectedFile"] for t in group}) != 3
                or len({t["request"] for t in group}) != 3):
            raise ValueError(f"Invalid three-value contrast group: {group_id}")
    return tasks, templates


def _overlap_summary(tasks: list[dict[str, Any]], train_inputs: set[str], train_targets: set[str]) -> dict[str, int]:
    prospective_inputs = [task["inputFile"] for task in tasks]
    prospective_targets = [task["expectedFile"] for task in tasks]
    return {
        "prospectiveInputsMatchingTrainingInputs": sum(value in train_inputs for value in prospective_inputs),
        "prospectiveInputsMatchingTrainingTargets": sum(value in train_targets for value in prospective_inputs),
        "prospectiveTargetsMatchingTrainingInputs": sum(value in train_inputs for value in prospective_targets),
        "prospectiveTargetsMatchingTrainingTargets": sum(value in train_targets for value in prospective_targets),
    }


def prepare_manifest() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {OUTPUT}")
    train, train_inputs, train_targets = _load_training_split()
    tasks, request_templates = _task_rows()
    overlaps = _overlap_summary(tasks, train_inputs, train_targets)
    if any(overlaps.values()):
        print(json.dumps({"status": "stopped-training-overlap-found", "overlap": overlaps}, indent=2))
        raise SystemExit("P3-03 prospective tasks overlap the frozen P3-02 training split; no substitutions were made")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    manifest = {
        "schemaVersion": 1,
        "milestone": "P3-03",
        "kind": "plex-p3-03-source-preservation-request-value-diagnostic-v1",
        "status": "prepared-unexecuted-evaluation-not-authorized",
        "preparedDate": "2026-10-17",
        "preparedAgainstGitCommit": commit,
        "model": {
            "name": "Plex Nano",
            "parameters": 27566080,
            "contextLength": 512,
            "vocabularySize": 16384,
            "checkpointStep": 100,
            "checkpointSha256": CHECKPOINT_SHA256,
            "checkpointHashReverifiedDuringPreparation": False,
            "tokenizerSha256": TOKENIZER_SHA256,
            "p302BundleManifestSha256": P302_BUNDLE_MANIFEST_SHA256,
        },
        "inferenceConfigurationFromP302": {
            "policy": "greedy-unreselected-fixed-endpoint-v1",
            "seed": 1337,
            "temperature": 0,
            "maximumNewTokens": 192,
            "checkpointReselectionAllowed": False,
            "configurationSource": "training/pretraining/p3-02-development-gate-result-summary.json",
        },
        "serialization": {
            "template": PROMPT_TEMPLATE,
            "encoding": "UTF-8",
            "responseContract": "complete edited file text only",
        },
        "p302TrainingSplit": {
            "candidatePath": "training/phase3/drafts/p3-02-exact-replacements-v1.jsonl",
            "candidateSha256": P302_CANDIDATE_SHA256,
            "recordCount": len(train),
            "trainingTargetCount": len(train_targets),
            "prospectiveCompleteFileOverlap": overlaps,
            "overlapFound": False,
        },
        "authorizationAndExecution": {
            "modelResponsesAuthorized": False,
            "checkpointLoaded": False,
            "modelResponsesGenerated": False,
            "trainingPerformed": False,
            "optimizerUpdates": 0,
            "officialP302ScorerModified": False,
            "checkpointReselected": False,
            "finalProjectHoldoutOpened": False,
        },
        "requestTemplates": request_templates,
        "taskCount": len(tasks),
        "contrastGroupCount": 24,
        "tasks": tasks,
        "frozenDecisionRules": {
            "fullPass": {
                "sourceNamePreservation": 72,
                "requestedValueSelection": 72,
                "unrelatedContentPreservation": 72,
                "exactFiles": 72,
                "fullyCorrectContrastGroups": 24,
            },
            "malformedMissingDuplicatedOrAmbiguousTarget": "requested-value-selection-failure",
            "doNotModifyTasksAfterAnyNewModelOutput": True,
            "diagnosticDoesNotReplaceP302Gate": True,
            "inferNoInternalCauseFromOutputCategories": True,
        },
        "measurementDefinitions": {
            "sourceNamePreservation": "Every listed sourceNameRole must remain in its expected structural role; all roles must pass for a task pass.",
            "requestedValueSelection": "Extract only the designated target role, independent of supplied source names; missing, malformed, duplicated, or ambiguous targets fail.",
            "unrelatedContentPreservation": "UTF-8 bytes before and after the authorized target span must match the supplied input exactly.",
            "exactFileCorrectness": "Decoded response must equal expectedFile byte-for-byte when encoded as UTF-8.",
            "contrastSensitivity": "Per group report distinct complete outputs, distinct successfully extracted target literals, correct requested literals, and all-three exact status.",
            "trainingTargetOverlap": "Count response strings exactly equal to any expectedFile in the frozen P3-02 authored training split; overlap is not proof of memorization.",
        },
    }
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with OUTPUT.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps({
        "status": manifest["status"],
        "manifestPath": str(OUTPUT),
        "manifestSha256": hashlib.sha256(encoded).hexdigest(),
        "taskCount": len(tasks),
        "contrastGroupCount": 24,
        "p302TrainingRowsRead": len(train),
        "overlap": overlaps,
        "preparedAgainstGitCommit": commit,
    }, indent=2))
    return manifest


if __name__ == "__main__":
    prepare_manifest()
