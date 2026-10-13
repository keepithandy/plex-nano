"""P3-02 exact-replacement candidate preparation; no model or optimizer work."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from .completion import _completion_tokenizer
from .file_conditioned_contract import (
    EXPECTED_TOKENIZER_SHA256, PROMPT_TEMPLATE, ROOT, TOKENIZER_BUNDLE_SHA256,
    _canonical_bytes, _publish_p3_pair, _read_json, _require_separate_p3_outputs,
    build_candidate_rows,
)

MILESTONE = "P3-02"
GENERATOR_VERSION = "p3-02-exact-replacements-generator-v1"
DEFAULT_CONTRACT = ROOT / "training/pretraining/p3-02-exact-replacements-preparation-contract.json"
DEFAULT_CANDIDATE = ROOT / "training/phase3/drafts/p3-02-exact-replacements-v1.jsonl"
DEFAULT_REVIEW = DEFAULT_CANDIDATE.with_suffix(".review.json")
DEFAULT_BUNDLE = ROOT / "training/artifacts/request-grounded/p2-48-training-bundle"
EXPECTED_SHA256 = "98d7ff276cf28d2509ad775c95c49ae0ffdb975de7e47f11e54430f3ca426403"
EXPECTED_BYTES = 128214
CONTEXT_NAMES = ("cart", "profile", "search", "settings", "gallery", "account")
SPLITS = ("train", "train", "train", "train", "validation", "development")
SAFE_FLAGS = {
    "checkpointStagingAuthorized": False, "optimizerCreationAuthorized": False,
    "modelTrainingAuthorized": False, "trainingPerformed": False,
    "researchOptimizerUpdates": 0, "finalHoldoutOpened": False,
    "automaticContinuation": False,
}
PROPOSED_RUN = {
    "sourceCheckpointSha256": "fd86d11375e547f05b2fab7a36188ca30a7ecd0b0f38198de62bcddbfba489b7",
    "modelParameters": 27566080, "contextLength": 512, "vocabularySize": 16384,
    "seed": 1337, "maximumOptimizerUpdates": 100, "maximumWallTimeSeconds": 600,
    "microBatch": 1, "gradientAccumulation": 16, "learningRate": 0.0001,
    "objective": "answer-eos-only-complete-record-v1",
    "validationSteps": [0, 25, 50, 75, 100], "endpointStep": 100,
    "checkpointReselectionAllowed": False, "maxNewTokens": 192,
    "developmentGate": {"minimumExact": 30, "tasks": 36, "minimumExactPerLanguage": 8,
                        "tasksPerLanguage": 12, "minimumPreserved": 34,
                        "minimumAllCorrectContrastGroups": 10, "contrastGroups": 12,
                        "requireFullPromptVisible": True},
}


def _cases(language: str, name: str):
    if language == "html":
        return (
            ("button-text", f'<button id="{name}-save">Cancel</button>', "Cancel",
             ("Confirm", "Save", "Continue"), f'For the button with id {name}-save, change its text to {{value}}.'),
            ("link-href", f'<a id="{name}-link" href="/start">Details</a>', "/start",
             ("/help", "/orders", "/contact"), f'For the link with id {name}-link, set href to {{value}}.'),
            ("input-placeholder", f'<input id="{name}-input" placeholder="Enter text">', "Enter text",
             ("Your name", "Search here", "Email address"), f'For the input with id {name}-input, set the placeholder to {{value}}.'),
            ("aria-label", f'<nav id="{name}-nav" aria-label="Navigation"></nav>', "Navigation",
             ("Main menu", "Quick links", "Account menu"), f'For the nav with id {name}-nav, set aria-label to {{value}}.'),
        )
    if language == "css":
        return tuple(
            (family, f'.{name}-{family} {{\n  {prop}: {old};\n  display: block;\n}}', old,
             values, f'In .{name}-{family}, change {prop} to {{value}}.')
            for family, prop, old, values in (
                ("gap", "gap", "6px", ("12px", "16px", "24px")),
                ("padding", "padding", "7px", ("10px", "18px", "22px")),
                ("color", "color", "black", ("navy", "teal", "purple")),
                ("radius", "border-radius", "3px", ("4px", "8px", "12px")),
            )
        )
    return (
        ("retry-limit", f'const {name}Retries = 2;', "2", ("4", "5", "7"),
         f'Set {name}Retries to {{value}}.'),
        ("message", f'const {name}Message = "Pending";', "Pending", ("Saved", "Ready", "Complete"),
         f'Set the string in {name}Message to {{value}}.'),
        ("selector", f'const {name}Button = document.querySelector(".old-action");', ".old-action",
         (".save-action", ".close-action", ".next-action"), f'Change the selector used by {name}Button to {{value}}.'),
        ("default-size", f'function {name}Size(size = 9) {{\n  return size;\n}}', "9", ("15", "20", "30"),
         f'Change the default size argument in {name}Size to {{value}}.'),
    )


def build_replacement_rows() -> list[dict[str, Any]]:
    rows = []
    for language in ("html", "css", "javascript"):
        for index, (name, split) in enumerate(zip(CONTEXT_NAMES, SPLITS)):
            for family, snippet, old, values, request in _cases(language, name):
                if language == "html":
                    source = f'<section class="{name}">\n  <p>Keep this note.</p>\n  {snippet}\n</section>'
                elif language == "css":
                    source = f'/* Keep this note for {name}. */\n{snippet}\n.{name}-aside {{ opacity: 0.8; }}'
                else:
                    source = f'// Keep this note for {name}.\n{snippet}\nconst {name}Enabled = true;'
                # Split by supplied file, not by alternate requests for that file.
                group = f"p302-{language}-{family}-{index:02d}"
                for variant, new in enumerate(values):
                    rows.append({
                        "schemaVersion": 1, "id": f"{group}-{variant}",
                        "candidateSplit": split, "contrastGroupId": group,
                        "language": language, "editFamily": family,
                        "request": request.format(value=new), "inputFile": source,
                        "expectedFile": source.replace(old, new, 1),
                        "oldText": old, "newText": new,
                        "provenance": "project-authored-p3-02-exact-replacements-v1",
                    })
    return rows


def render_prompt(row: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(language=row["language"], request=row["request"], input_file=row["inputFile"])


def validate_replacement_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(rows, list) or len(rows) != 216:
        raise ValueError("P3-02 requires 216 authored replacement records")
    approved = {row["id"]: row for row in build_replacement_rows()}
    groups = defaultdict(list)
    seen = set()
    split_sets = defaultdict(lambda: defaultdict(set))
    counts = Counter()
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("schemaVersion")) is not int
                or not isinstance(row.get("id"), str) or row != approved.get(row["id"])):
            raise ValueError("P3-02 row differs from the authored request/file/replacement")
        if row["id"] in seen:
            raise ValueError("Duplicate P3-02 record")
        seen.add(row["id"])
        if row["inputFile"].count(row["oldText"]) != 1 or row["oldText"] == row["newText"]:
            raise ValueError("P3-02 must identify one unambiguous changed region")
        if row["inputFile"].replace(row["oldText"], row["newText"], 1) != row["expectedFile"]:
            raise ValueError("P3-02 target changes unrelated text")
        groups[row["contrastGroupId"]].append(row)
        split = row["candidateSplit"]
        counts[split] += 1
        for field in ("contrastGroupId", "request", "inputFile", "expectedFile"):
            split_sets[split][field].add(row[field])
        split_sets[split]["prompt"].add(render_prompt(row))
    for group in groups.values():
        if (len(group) != 3 or len({r["candidateSplit"] for r in group}) != 1
                or len({r["inputFile"] for r in group}) != 1
                or len({r["request"] for r in group}) != 3
                or len({r["expectedFile"] for r in group}) != 3):
            raise ValueError("P3-02 contrast groups must stay intact with three distinct edits")
    overlaps = {}
    for first, second in combinations(("train", "validation", "development"), 2):
        overlaps[f"{first}:{second}"] = {
            field: len(split_sets[first][field] & split_sets[second][field])
            for field in ("contrastGroupId", "request", "inputFile", "expectedFile", "prompt")
        }
        if any(overlaps[f"{first}:{second}"].values()):
            raise ValueError("P3-02 splits contain exact overlapping identities")
    p301 = build_candidate_rows()
    contract_overlap = {
        field: len({row[field] for row in rows} & {row[field] for row in p301})
        for field in ("request", "inputFile", "expectedFile")
    }
    if any(contract_overlap.values()):
        raise ValueError("P3-02 duplicates a frozen P3-01 contract example")
    return {"records": len(rows), "contrastGroups": len(groups),
            "splitCounts": dict(sorted(counts.items())),
            "perLanguage": dict(Counter(row["language"] for row in rows)),
            "splitOverlap": overlaps, "p301ExactOverlap": contract_overlap}


def _identity(raw: bytes) -> dict[str, Any]:
    return {"candidateSha256": hashlib.sha256(raw).hexdigest(), "byteCount": len(raw)}


def _contract(path: Path) -> dict[str, Any]:
    contract = _read_json(path)
    required = {"schemaVersion": 1, "milestone": MILESTONE,
                "kind": "plex-p3-02-exact-replacements-preparation-v1",
                "dataPreparationAuthorized": True, "contractReviewAuthorized": True,
                "tokenizerContextReviewAuthorized": True, **SAFE_FLAGS}
    for field, expected in required.items():
        if type(contract.get(field)) is not type(expected) or contract[field] != expected:
            raise ValueError(f"P3-02 preparation boundary differs: {field}")
    if json.dumps(contract.get("candidateIdentity"), sort_keys=True) != json.dumps(
            {"sha256": EXPECTED_SHA256, "bytes": EXPECTED_BYTES}, sort_keys=True):
        raise ValueError("P3-02 contract candidate identity differs")
    if json.dumps(contract.get("proposedRun"), sort_keys=True) != json.dumps(PROPOSED_RUN, sort_keys=True):
        raise ValueError("P3-02 proposed experiment differs from the preparation contract")
    return contract


def generate_replacement_candidate(*, candidate_path: Path = DEFAULT_CANDIDATE,
                                   review_path: Path = DEFAULT_REVIEW,
                                   contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    _contract(contract_path)
    rows = build_replacement_rows()
    summary = validate_replacement_rows(rows)
    raw = _canonical_bytes(rows)
    if _identity(raw) != {"candidateSha256": EXPECTED_SHA256, "byteCount": EXPECTED_BYTES}:
        raise ValueError("P3-02 generator differs from the frozen candidate")
    report = {"schemaVersion": 1, "milestone": MILESTONE, "generatorVersion": GENERATOR_VERSION,
              "status": "candidate-generated-pending-review", **_identity(raw), **summary, **SAFE_FLAGS}
    _publish_p3_pair(candidate_path=candidate_path, candidate_bytes=raw,
                    review_path=review_path, review_bytes=(json.dumps(report, indent=2) + "\n").encode(),
                    contract_path=contract_path)
    return report


def review_replacement_candidate(*, candidate_path: Path = DEFAULT_CANDIDATE,
                                 review_path: Path = DEFAULT_REVIEW,
                                 contract_path: Path = DEFAULT_CONTRACT,
                                 bundle_dir: Path = DEFAULT_BUNDLE,
                                 report_path: Path | None = None) -> dict[str, Any]:
    if report_path is not None:
        _require_separate_p3_outputs(outputs={"review report": report_path}, protected_inputs={
            "candidate": candidate_path, "draft review": review_path, "contract": contract_path,
            **{name: bundle_dir / name for name in
               ("tokenizer.json", "tokenizer-config.json", "model-config.json", "manifest.json")},
        })
    _contract(contract_path)
    if candidate_path.is_symlink() or not candidate_path.is_file() or candidate_path.stat().st_size > 1024 * 1024:
        raise ValueError("P3-02 candidate must be a bounded regular file")
    with candidate_path.open("rb") as stream:
        raw = stream.read(EXPECTED_BYTES + 1)
    if _identity(raw) != {"candidateSha256": EXPECTED_SHA256, "byteCount": EXPECTED_BYTES}:
        raise ValueError("P3-02 frozen candidate identity changed")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    summary = validate_replacement_rows(rows)
    if raw != _canonical_bytes(rows):
        raise ValueError("P3-02 candidate serialization is not canonical")
    metadata = _read_json(review_path)
    expected = {"schemaVersion": 1, "milestone": MILESTONE, "generatorVersion": GENERATOR_VERSION,
                "status": "candidate-generated-pending-review", **_identity(raw), **summary, **SAFE_FLAGS}
    if json.dumps(metadata, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("P3-02 draft review differs from generated evidence")
    tokenizer, identity = _completion_tokenizer(bundle_dir)
    if (identity["tokenizerSha256"] != EXPECTED_TOKENIZER_SHA256
            or identity["bundleManifestSha256"] != TOKENIZER_BUNDLE_SHA256
            or tokenizer.vocabulary_size != 16384):
        raise ValueError("P3-02 requires the frozen P2-48 tokenizer")
    lengths, prompts, targets = [], [], []
    split_tokens = Counter()
    for row in rows:
        prompt, target = render_prompt(row), row["expectedFile"]
        prompt_ids, target_ids = tokenizer.encode(prompt), tokenizer.encode(target)
        if tokenizer.decode(prompt_ids) != prompt or tokenizer.decode(target_ids) != target:
            raise ValueError("P3-02 tokenizer roundtrip failed")
        if any(token < 4 for token in prompt_ids + target_ids):
            raise ValueError("P3-02 text contains a control/unknown token")
        length = len(prompt_ids) + len(target_ids) + 1
        if length > 512 or len(target_ids) + 1 > 192 or len(prompt_ids) + 192 > 512:
            raise ValueError("P3-02 record or full-prompt decoding budget exceeds context")
        lengths.append(length); prompts.append(len(prompt_ids)); targets.append(len(target_ids) + 1)
        split_tokens[row["candidateSplit"]] += length
    report = {**expected, "status": "candidate-review-passed",
              "tokenizerPreflight": {"checked": True, **identity,
                  "encoding": "encode(prompt) + encode(target) + [EOS=3]",
                  "minimumRecordTokensIncludingEos": min(lengths),
                  "maximumRecordTokensIncludingEos": max(lengths),
                  "maximumPromptTokens": max(prompts), "maximumTargetTokensIncludingEos": max(targets),
                  "splitTokenCounts": dict(sorted(split_tokens.items())),
                  "allFit512Tokens": True, "maxNewTokensRequested": 192,
                  "fullPromptRetainedForMaximumGenerationBudget": True}}
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    return report


def score_replacement(row: dict[str, Any], actual: str) -> dict[str, Any]:
    approved = {r["id"]: r for r in build_replacement_rows()}
    if (not isinstance(row, dict) or type(row.get("schemaVersion")) is not int
            or not isinstance(row.get("id"), str) or row != approved.get(row["id"])):
        raise ValueError("Scoring requires an exact frozen P3-02 record")
    if not isinstance(actual, str):
        raise ValueError("Replacement output must be text")
    actual = actual.replace("\r\n", "\n").replace("\r", "\n")
    prefix, suffix = row["inputFile"].split(row["oldText"])
    preserved = len(actual) >= len(prefix) + len(suffix) and actual.startswith(prefix) and actual.endswith(suffix)
    exact = actual == row["expectedFile"]
    return {"exactMatch": exact, "requestedEditCorrect": exact,
            "unrelatedCodePreserved": preserved, "unnecessaryEdits": not preserved,
            "syntaxCheckability": "known-valid-target" if exact else "not-checked"}
