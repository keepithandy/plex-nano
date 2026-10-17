"""Read-only P3-03 scorer for a separately authorized prediction file.

This module has no model, checkpoint, tokenizer, optimizer, network, or holdout
imports. It reads the frozen manifest, the P3-02 authored training candidate,
and a prediction JSON supplied after separately authorized inference. It writes
a new Markdown report and JSON audit record without overwriting existing files.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEFAULT_MANIFEST = HERE / "p3-03-source-preservation-v1.json"
EXPECTED_MANIFEST_SHA256 = "119eaa829825e3daa6c9156bd0ba84dfa32726dc2ebfa3d7cfd9ccf7612df578"
P302_CANDIDATE = ROOT / "training/phase3/drafts/p3-02-exact-replacements-v1.jsonl"
P302_CANDIDATE_SHA256 = "98d7ff276cf28d2509ad775c95c49ae0ffdb975de7e47f11e54430f3ca426403"
P302_TRAIN_RECORDS = 144
MAX_INPUT_BYTES = 16 * 1024 * 1024


class _TargetHTMLParser(HTMLParser):
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, target_tag: str):
        super().__init__(convert_charrefs=True)
        self.target_tag = target_tag
        self.nodes: list[dict[str, Any]] = []
        self.stack: list[tuple[str, dict[str, Any] | None]] = []
        self.line_offsets = [0]

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.line_offsets[min(line - 1, len(self.line_offsets) - 1)] + column

    def _start(self, tag: str, attrs: list[tuple[str, str | None]], self_closing: bool) -> None:
        node: dict[str, Any] | None = None
        for _, open_node in self.stack:
            if open_node is not None:
                open_node["nested"] = True
        if tag == self.target_tag:
            raw_start_tag = self.get_starttag_text() or ""
            start_offset = self._offset()
            node = {"tag": tag, "attrs": attrs, "text": [], "closed": False,
                    "nested": False, "malformed": False, "nonTextContent": False,
                    "startTagText": raw_start_tag, "startOffset": start_offset,
                    "contentStart": start_offset + len(raw_start_tag), "contentEnd": None}
            self.nodes.append(node)
        if self_closing or tag in self.VOID_TAGS:
            if node is not None:
                node["closed"] = True
            return
        self.stack.append((tag, node))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag, attrs, False)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag, attrs, True)

    def handle_endtag(self, tag: str) -> None:
        end_offset = self._offset()
        match = next((i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i][0] == tag), None)
        if match is None:
            for _, node in self.stack:
                if node is not None:
                    node["malformed"] = True
            return
        while len(self.stack) - 1 > match:
            _, unclosed = self.stack.pop()
            if unclosed is not None:
                unclosed["malformed"] = True
        _, node = self.stack.pop()
        if node is not None:
            node["closed"] = True
            node["contentEnd"] = end_offset

    def handle_data(self, data: str) -> None:
        for _, node in self.stack:
            if node is not None:
                node["text"].append(data)

    def handle_comment(self, data: str) -> None:
        for _, node in self.stack:
            if node is not None:
                node["nonTextContent"] = True

    def feed(self, data: str) -> None:
        self.line_offsets = [0]
        self.line_offsets.extend(index + 1 for index, character in enumerate(data) if character == "\n")
        super().feed(data)

    def finish(self) -> list[dict[str, Any]]:
        self.close()
        for _, node in self.stack:
            if node is not None:
                node["malformed"] = True
        return self.nodes


def _html_nodes(text: str, tag: str) -> list[dict[str, Any]]:
    parser = _TargetHTMLParser(tag)
    try:
        parser.feed(text)
        return parser.finish()
    except Exception:
        return []


def _one_attribute(node: dict[str, Any], name: str) -> tuple[bool, str | None]:
    values = [value for attr, value in node["attrs"] if attr == name]
    if len(values) != 1 or not isinstance(values[0], str):
        return False, None
    return True, values[0]


def _html_attribute_span(node: dict[str, Any], name: str) -> tuple[int, int] | None:
    pattern = re.compile(
        rf"(?:^|\s){re.escape(name)}\s*=\s*(?P<value>\"[^\"]*\"|'[^']*'|[^\s/>]+)",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(node["startTagText"]))
    if len(matches) != 1:
        return None
    match = matches[0]
    value = match.group("value")
    if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
        left, right = match.start("value") + 1, match.end("value") - 1
    else:
        left, right = match.start("value"), match.end("value")
    base = node["startOffset"]
    return base + left, base + right


def _css_parts(text: str) -> tuple[list[str], str | None, str | None, str | None, int | None, int | None]:
    """Return comments, first selector, first rule body, and a parse error."""
    comments: list[str] = []
    kept: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                return comments, None, None, "unclosed comment", None, None
            comments.append(text[i:end + 2])
            kept.extend(" " * (end + 2 - i))
            i = end + 2
        else:
            kept.append(text[i])
            i += 1
    clean = "".join(kept)
    opening = clean.find("{")
    if opening < 0:
        return comments, None, None, "missing first rule", None, None
    closing = clean.find("}", opening + 1)
    if closing < 0 or "{" in clean[opening + 1:closing]:
        return comments, None, None, "malformed first rule", None, None
    if "}" in clean[:opening]:
        selector = clean[:opening].rsplit("}", 1)[-1].strip()
    else:
        selector = clean[:opening].strip()
    return comments, selector, clean[opening + 1:closing], None, opening, closing


def _css_declaration_values(body: str, target: str, body_offset: int) -> tuple[list[str], list[str], list[tuple[int, int]]]:
    values: list[str] = []
    malformed: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for declaration in body.split(";"):
        item = declaration.strip()
        if not item:
            cursor += len(declaration) + 1
            continue
        prop, sep, value = item.partition(":")
        if not sep or not prop.strip() or not value.strip():
            malformed.append(item)
            cursor += len(declaration) + 1
            continue
        if prop.strip().lower() == target.lower():
            values.append(value.strip())
            colon_offset = declaration.find(":")
            raw_value = declaration[colon_offset + 1:]
            leading = len(raw_value) - len(raw_value.lstrip())
            trimmed = raw_value.strip()
            left = cursor + colon_offset + 1 + leading
            spans.append((body_offset + left, body_offset + left + len(trimmed)))
        cursor += len(declaration) + 1
    return values, malformed, spans


_JS_CONST = re.compile(r"^\s*const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(.*?)\s*;\s*$")
_JS_NUMBER = re.compile(r"^(?:0|[1-9][0-9]*)$")
_JS_DOUBLE_STRING = re.compile(r'^"(?:[^"\\]|\\.)*"$')
_JS_SELECTOR = re.compile(r'^document\.querySelector\(\s*("(?:[^"\\]|\\.)*")\s*\)$')


def _js_declarations(text: str) -> list[dict[str, Any]]:
    declarations = []
    offset = 0
    for line_no, raw_line in enumerate(text.splitlines(keepends=True)):
        line = raw_line.rstrip("\r\n")
        match = _JS_CONST.fullmatch(line)
        if match:
            start, end = match.span(2)
            declarations.append({"name": match.group(1), "initializer": match.group(2), "line": line_no,
                                 "valueStart": offset + start, "valueEnd": offset + end})
        offset += len(raw_line)
    return declarations


def _extract_target(task: dict[str, Any], actual: str) -> dict[str, Any]:
    extraction = task["target"]["extraction"]
    kind = extraction["kind"]
    if kind in ("html-text", "html-attribute"):
        nodes = _html_nodes(actual, extraction["tag"])
        if len(nodes) != 1:
            return {"valid": False, "value": None, "reason": "missing-or-ambiguous-element", "structurallyValid": False}
        node = nodes[0]
        if not node["closed"] or node["malformed"]:
            return {"valid": False, "value": None, "reason": "malformed-element", "structurallyValid": False}
        if kind == "html-text":
            if node["nested"] or node["nonTextContent"] or node["contentEnd"] is None:
                return {"valid": False, "value": None, "reason": "nested-target-content", "structurallyValid": False}
            value = "".join(node["text"])
            start, end = node["contentStart"], node["contentEnd"]
        else:
            valid, value = _one_attribute(node, extraction["attribute"])
            if not valid:
                return {"valid": False, "value": None, "reason": "missing-duplicate-or-nonliteral-attribute", "structurallyValid": False}
            span = _html_attribute_span(node, extraction["attribute"])
            if span is None:
                return {"valid": False, "value": None, "reason": "attribute-span-ambiguous", "structurallyValid": False}
            start, end = span
        return {"valid": True, "value": value, "reason": None, "structurallyValid": True,
                "startCharacter": start, "endCharacter": end}
    if kind == "css-declaration":
        _, _, body, error, opening, _ = _css_parts(actual)
        if error or body is None:
            return {"valid": False, "value": None, "reason": error or "malformed-first-rule", "structurallyValid": False}
        values, _, spans = _css_declaration_values(body, extraction["property"], (opening or 0) + 1)
        if len(values) != 1:
            reason = "target-property-missing" if not values else "target-property-duplicated-or-ambiguous"
            return {"valid": False, "value": None, "reason": reason, "structurallyValid": False}
        return {"valid": True, "value": values[0], "reason": None, "structurallyValid": True,
                "startCharacter": spans[0][0], "endCharacter": spans[0][1]}
    if kind not in ("js-number", "js-string", "js-selector"):
        return {"valid": False, "value": None, "reason": "unknown-extraction-kind", "structurallyValid": False}
    declarations = _js_declarations(actual)
    role_declarations = [d for d in declarations if d["name"].endswith(extraction["identifierSuffix"])]
    if len(role_declarations) != 1:
        reason = "target-declaration-missing" if not role_declarations else "target-declaration-duplicated-or-ambiguous"
        return {"valid": False, "value": None, "reason": reason, "structurallyValid": False}
    candidate = role_declarations[0]
    if kind == "js-number":
        if not _JS_NUMBER.fullmatch(candidate["initializer"]):
            return {"valid": False, "value": None, "reason": "target-number-malformed", "structurallyValid": False}
        value = candidate["initializer"]
    elif kind == "js-string":
        if not _JS_DOUBLE_STRING.fullmatch(candidate["initializer"]):
            return {"valid": False, "value": None, "reason": "target-string-malformed", "structurallyValid": False}
        try:
            decoded_string = json.loads(candidate["initializer"])
        except json.JSONDecodeError:
            return {"valid": False, "value": None, "reason": "target-string-malformed", "structurallyValid": False}
        if type(decoded_string) is not str:
            return {"valid": False, "value": None, "reason": "target-string-malformed", "structurallyValid": False}
        value = candidate["initializer"]
    else:
        selector_match = _JS_SELECTOR.fullmatch(candidate["initializer"])
        if selector_match is None:
            return {"valid": False, "value": None, "reason": "target-selector-malformed", "structurallyValid": False}
        try:
            decoded_selector = json.loads(selector_match.group(1))
        except json.JSONDecodeError:
            return {"valid": False, "value": None, "reason": "target-selector-malformed", "structurallyValid": False}
        if type(decoded_selector) is not str:
            return {"valid": False, "value": None, "reason": "target-selector-malformed", "structurallyValid": False}
        value = selector_match.group(1)
    if kind == "js-selector":
        selector_match = _JS_SELECTOR.fullmatch(candidate["initializer"])
        span_start = candidate["valueStart"] + selector_match.start(1)
        span_end = candidate["valueStart"] + selector_match.end(1)
    else:
        span_start, span_end = candidate["valueStart"], candidate["valueEnd"]
    return {"valid": True, "value": value, "reason": None, "structurallyValid": True,
            "startCharacter": span_start, "endCharacter": span_end}


def _preservation_observations(task: dict[str, Any], actual: str) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for check in task["sourceNameRoles"]:
        kind = check["kind"]
        passed = False
        observed: Any = None
        if kind == "html-attribute":
            nodes = _html_nodes(actual, check["tag"])
            if len(nodes) == 1 and nodes[0]["closed"] and not nodes[0]["malformed"]:
                valid, observed = _one_attribute(nodes[0], check["attribute"])
                if valid:
                    if check.get("match") == "class-token":
                        passed = check["expected"] in observed.split()
                    else:
                        passed = observed == check["expected"]
        elif kind == "css-first-selector":
            _, observed, _, error, _, _ = _css_parts(actual)
            passed = error is None and observed == check["expected"]
        elif kind == "css-comment":
            comments, _, _, _, _, _ = _css_parts(actual)
            observed = [comment for comment in comments if check["expected"] in comment]
            passed = observed == [check["expected"]]
        elif kind == "js-first-line":
            first = actual.splitlines()[0] if actual.splitlines() else None
            observed = first
            passed = first == check["expected"]
        elif kind == "js-identifier":
            declarations = _js_declarations(actual)
            observed = [d["name"] for d in declarations if d["name"] == check["expected"]]
            passed = len(observed) == 1
        observations.append({"role": check["role"], "expected": check["expected"],
                             "observed": observed, "pass": bool(passed)})
    return observations


def _region_diff(expected: bytes, observed: bytes, region: str) -> list[dict[str, Any]]:
    hunks = []
    matcher = difflib.SequenceMatcher(a=expected, b=observed, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        hunks.append({
            "region": region,
            "category": f"{region}-{tag}",
            "expectedByteRange": [i1, i2],
            "observedByteRange": [j1, j2],
            "expectedUtf8": expected[i1:i2].decode("utf-8", errors="replace"),
            "observedUtf8": observed[j1:j2].decode("utf-8", errors="replace"),
        })
    return hunks


def _unrelated_preservation(task: dict[str, Any], actual: str,
                            extracted: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    source = task["inputFile"].encode("utf-8")
    start = task["target"]["startCharacter"]
    end = task["target"]["endCharacter"]
    prefix, suffix = source[:start], source[end:]
    if (not extracted.get("valid") or type(extracted.get("startCharacter")) is not int
            or type(extracted.get("endCharacter")) is not int):
        return False, [{"region": "target-boundary", "category": "authorized-target-span-unavailable"}]
    actual_start, actual_end = extracted["startCharacter"], extracted["endCharacter"]
    if actual_start < 0 or actual_end < actual_start or actual_end > len(actual):
        return False, [{"region": "target-boundary", "category": "invalid-authorized-target-span"}]
    actual_prefix = actual[:actual_start].encode("utf-8")
    actual_suffix = actual[actual_end:].encode("utf-8")
    prefix_ok = actual_prefix == prefix
    suffix_ok = actual_suffix == suffix
    changes: list[dict[str, Any]] = []
    if not prefix_ok:
        changes.extend(_region_diff(prefix, actual_prefix, "before-target"))
    if not suffix_ok:
        changes.extend(_region_diff(suffix, actual_suffix, "after-target"))
    return prefix_ok and suffix_ok, changes


def _load_p302_training_targets() -> tuple[list[dict[str, Any]], set[str], str]:
    if P302_CANDIDATE.is_symlink() or not P302_CANDIDATE.is_file() or P302_CANDIDATE.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Frozen P3-02 candidate is missing, linked, or too large")
    raw = P302_CANDIDATE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != P302_CANDIDATE_SHA256:
        raise ValueError(f"P3-02 candidate hash mismatch: {digest}")
    train = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith('{"candidateSplit":"train",'):
            row = json.loads(line)
            if type(row.get("expectedFile")) is not str:
                raise ValueError("P3-02 training target is not a string")
            train.append(row)
    if len(train) != P302_TRAIN_RECORDS:
        raise ValueError(f"Expected {P302_TRAIN_RECORDS} P3-02 training targets; found {len(train)}")
    return train, {row["expectedFile"] for row in train}, digest


def _validate_execution(manifest: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
    auth = envelope.get("evaluationAuthorization")
    if (not isinstance(auth, dict) or auth.get("status") != "separately-authorized"
            or not isinstance(auth.get("evidenceReference"), str) or not auth["evidenceReference"].strip()):
        raise ValueError("Prediction file lacks evidence of separate P3-03 evaluation authorization")
    execution = envelope.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("Prediction file lacks execution provenance")
    model = manifest["model"]
    config = manifest["inferenceConfigurationFromP302"]
    expected = {
        "checkpointStep": model["checkpointStep"],
        "checkpointSha256": model["checkpointSha256"],
        "checkpointHashVerified": True,
        "tokenizerSha256": model["tokenizerSha256"],
        "p302BundleManifestSha256": model["p302BundleManifestSha256"],
        "policy": config["policy"],
        "seed": config["seed"],
        "temperature": config["temperature"],
        "maximumNewTokens": config["maximumNewTokens"],
        "checkpointReselectionAllowed": False,
        "trainingPerformedDuringDiagnostic": False,
        "optimizerUpdatesDuringDiagnostic": 0,
        "officialP302ScorerModified": False,
        "finalProjectHoldoutOpened": False,
        "promptTemplateSha256": hashlib.sha256(manifest["serialization"]["template"].encode("utf-8")).hexdigest(),
    }
    for key, value in expected.items():
        if execution.get(key) != value or type(execution.get(key)) is not type(value):
            raise ValueError(f"Prediction execution provenance differs from the frozen contract: {key}")
    if not isinstance(execution.get("codeRevision"), str) or not execution["codeRevision"].strip():
        raise ValueError("Prediction file must record the inference code revision")
    return {**execution, "evaluationAuthorizationEvidenceReference": auth["evidenceReference"]}


def _read_json(path: Path) -> tuple[bytes, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"Input must be a bounded regular file: {path}")
    raw = path.read_bytes()
    return raw, json.loads(raw.decode("utf-8"))


def _score(manifest: dict[str, Any], envelope: dict[str, Any], train_targets: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    tasks = manifest.get("tasks")
    responses = envelope.get("responses")
    if not isinstance(tasks, list) or len(tasks) != 72 or not isinstance(responses, list) or len(responses) != 72:
        raise ValueError("P3-03 analysis requires the frozen 72-task manifest and exactly 72 responses")
    expected_ids = {task["taskId"] for task in tasks}
    by_id: dict[str, str] = {}
    for response in responses:
        if not isinstance(response, dict) or type(response.get("actual")) is not str:
            raise ValueError("Each response must contain a string taskId and decoded actual output")
        ident = response.get("taskId")
        if type(ident) is not str or ident not in expected_ids or ident in by_id:
            raise ValueError("Response task ID is missing, unexpected, or duplicated")
        by_id[ident] = response["actual"]
    if set(by_id) != expected_ids:
        raise ValueError("Response set does not exactly match the frozen 72 tasks")

    audits: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        actual = by_id[task["taskId"]]
        extracted = _extract_target(task, actual)
        extracted["pass"] = bool(extracted["valid"] and extracted["value"] == task["target"]["requestedLiteral"])
        names = _preservation_observations(task, actual)
        names_pass = all(item["pass"] for item in names)
        unrelated_pass, modifications = _unrelated_preservation(task, actual, extracted)
        unrelated_scorable = not any(
            change["category"] in ("authorized-target-span-unavailable", "invalid-authorized-target-span")
            for change in modifications
        )
        exact = actual.encode("utf-8") == task["expectedFile"].encode("utf-8")
        audit = {
            "taskId": task["taskId"], "language": task["language"], "family": task["family"],
            "sourceNamespace": task["sourceNamespace"], "contrastGroupId": task["contrastGroupId"],
            "request": task["request"], "requestedLiteral": task["target"]["requestedLiteral"],
            "inputFile": task["inputFile"], "expectedFile": task["expectedFile"],
            "actual": actual, "sourceNamePreservation": names_pass,
            "sourceNameRoleObservations": names, "requestedValueSelection": extracted,
            "unrelatedContentPreservation": unrelated_pass,
            "unrelatedContentScorable": unrelated_scorable,
            "unrelatedModifications": modifications,
            "exactFileCorrectness": exact,
            "matchesP302TrainingTarget": actual in train_targets,
        }
        audits.append(audit)
        groups[task["contrastGroupId"]].append(audit)

    counts = {
        "tasks": len(audits),
        "sourceNamePreservationPasses": sum(a["sourceNamePreservation"] for a in audits),
        "requestedValueSelectionPasses": sum(a["requestedValueSelection"]["pass"] for a in audits),
        "unrelatedContentPreservationPasses": sum(a["unrelatedContentPreservation"] for a in audits),
        "exactFileCorrect": sum(a["exactFileCorrectness"] for a in audits),
        "trainingTargetOutputMatches": sum(a["matchesP302TrainingTarget"] for a in audits),
    }
    unscorable_preservation = sum(not a["unrelatedContentScorable"] for a in audits)
    proven_preservation_failure = any(
        not a["sourceNamePreservation"] or (
            a["unrelatedContentScorable"] and not a["unrelatedContentPreservation"]
        ) for a in audits
    )
    matrix = {
        "namesPass_valuePass": sum(a["sourceNamePreservation"] and a["requestedValueSelection"]["pass"] for a in audits),
        "namesFail_valuePass": sum((not a["sourceNamePreservation"]) and a["requestedValueSelection"]["pass"] for a in audits),
        "namesPass_valueFail": sum(a["sourceNamePreservation"] and (not a["requestedValueSelection"]["pass"]) for a in audits),
        "namesFail_valueFail": sum((not a["sourceNamePreservation"]) and (not a["requestedValueSelection"]["pass"]) for a in audits),
    }
    group_results = []
    for group_id in sorted(groups):
        members = groups[group_id]
        extracted_values = {a["requestedValueSelection"]["value"] for a in members
                            if a["requestedValueSelection"]["valid"]}
        group_results.append({
            "contrastGroupId": group_id,
            "distinctCompleteOutputs": len({a["actual"] for a in members}),
            "distinctExtractedTargetValues": len(extracted_values),
            "unextractableTaskCount": sum(not a["requestedValueSelection"]["valid"] for a in members),
            "individuallyCorrectRequestedValues": sum(a["requestedValueSelection"]["pass"] for a in members),
            "fullyCorrect": len(members) == 3 and all(a["exactFileCorrectness"] for a in members),
        })
    if len(group_results) != 24 or any(len(members) != 3 for members in groups.values()):
        raise ValueError("Frozen contrast groups are incomplete")
    names_pass = counts["sourceNamePreservationPasses"] == 72
    unrelated_pass = counts["unrelatedContentPreservationPasses"] == 72
    values_pass = counts["requestedValueSelectionPasses"] == 72 and all(
        group["individuallyCorrectRequestedValues"] == 3 and
        group["distinctExtractedTargetValues"] == 3 for group in group_results
    )
    preservation_pass = names_pass and unrelated_pass
    if unscorable_preservation and not proven_preservation_failure:
        outcome = "scoring-ambiguous"
        next_proposal = "Review target-span extraction and validate preservation scoring before interpreting the model output."
    elif not preservation_pass and values_pass:
        outcome = "preservation-fails-value-passes"
        next_proposal = "Propose a bounded experiment on retaining supplied names and unrelated code."
    elif preservation_pass and not values_pass:
        outcome = "preservation-passes-value-fails"
        next_proposal = "Propose a bounded request-contrast experiment selecting different replacement values for the same supplied file."
    elif not preservation_pass and not values_pass:
        outcome = "both-fail"
        next_proposal = "Propose a simpler controlled task that isolates copying supplied code from applying the requested change."
    else:
        outcome = "both-pass"
        next_proposal = "Investigate the difference from the failed 36-task gate and confirm on independent replacement tasks before advancing."
    decision = {
        "fullDiagnosticPass": (
            counts["sourceNamePreservationPasses"] == 72
            and counts["requestedValueSelectionPasses"] == 72
            and counts["unrelatedContentPreservationPasses"] == 72
            and counts["exactFileCorrect"] == 72
            and sum(group["fullyCorrect"] for group in group_results) == 24
        ),
        "preservationFailureObserved": proven_preservation_failure,
        "valueSelectionFailureObserved": not values_pass,
        "unscorablePreservationTasks": unscorable_preservation,
        "outcome": outcome,
        "nextProposal": next_proposal,
        "mixedNameAndValueFailuresObserved": matrix["namesFail_valueFail"] > 0,
        "officialP302GateModified": False,
        "internalCauseInferred": False,
    }
    summary = {"counts": counts, "crossCategoryFailureMatrix": matrix,
               "contrastGroups": group_results, "decision": decision}
    return summary, {"schemaVersion": 1, "milestone": "P3-03", "tasks": audits}


def _markdown_report(manifest: dict[str, Any], manifest_sha: str, execution: dict[str, Any],
                     training_hash: str, analysis_sha: str,
                     summary: dict[str, Any], audit: dict[str, Any]) -> str:
    counts = summary["counts"]
    audit_records = audit["tasks"]
    per_language: dict[str, dict[str, int]] = {}
    for language in ("html", "css", "javascript"):
        selected = [record for record in audit_records if record["language"] == language]
        per_language[language] = {
            "tasks": len(selected),
            "names": sum(record["sourceNamePreservation"] for record in selected),
            "values": sum(record["requestedValueSelection"]["pass"] for record in selected),
            "unrelated": sum(record["unrelatedContentPreservation"] for record in selected),
            "exact": sum(record["exactFileCorrectness"] for record in selected),
            "unscorable": sum(not record["unrelatedContentScorable"] for record in selected),
        }
    per_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in audit_records:
        per_family[(record["language"], record["family"])].append(record)
    modification_tasks: dict[str, set[str]] = defaultdict(set)
    for record in audit_records:
        for change in record["unrelatedModifications"]:
            modification_tasks[change["category"]].add(record["taskId"])
    name_failures = [record for record in audit_records if not record["sourceNamePreservation"]]
    value_failures = [record for record in audit_records if not record["requestedValueSelection"]["pass"]]
    lines = [
        "# P3-03 — Source Preservation and Request-Value Diagnostic",
        "",
        "## 1. Baseline and artifact verification",
        "",
        "Plex Nano remains the 27,566,080-parameter coding model. The frozen P3-02 development gate remains failed: 0/36 exact files, 0/36 preservation passes, and 0/12 fully correct contrast groups. The October 17 saved-response review also recorded 36/36 outputs losing `account`, 25/36 outputs matching a P3-02 training target, and 8/12 groups with byte-identical outputs for all three requests. Those historical measurements remain unchanged. This report adds separate diagnostic measurements and does not replace that gate.",
        "",
        f"- Fixed checkpoint: step {execution['checkpointStep']}, SHA-256 `{execution['checkpointSha256']}` (execution provenance reports the hash was verified).",
        f"- Tokenizer SHA-256: `{execution['tokenizerSha256']}`.",
        f"- P3-02 bundle manifest SHA-256: `{execution['p302BundleManifestSha256']}`.",
        f"- Inference: `{execution['policy']}`, seed {execution['seed']}, temperature {execution['temperature']}, maximum {execution['maximumNewTokens']} new tokens.",
        f"- Inference code revision: `{execution['codeRevision']}`.",
        f"- Read-only analysis implementation SHA-256: `{analysis_sha}`.",
        f"- Separate evaluation authorization reference: `{execution['evaluationAuthorizationEvidenceReference']}`.",
        "- Final project holdout opened: no.",
        "",
        "## 2. Frozen task manifest and SHA-256",
        "",
        f"- Manifest: `{manifest_sha}`.",
        f"- Frozen tasks: {manifest['taskCount']}; contrast groups: {manifest['contrastGroupCount']}.",
        "- P3-02 candidate hash: `" + training_hash + "`; only the 144 authored training records were used for overlap checks.",
        "- Complete-file overlap preflight counts: `" + json.dumps(manifest['p302TrainingSplit']['prospectiveCompleteFileOverlap'], sort_keys=True) + "`.",
        "",
        "## 3. Name-preservation results",
        "",
        f"{counts['sourceNamePreservationPasses']}/72 tasks passed all source-name roles.",
        "",
        "| Language | Role-complete passes | Tasks |",
        "|---|---:|---:|",
        *[f"| {language} | {per_language[language]['names']} | {per_language[language]['tasks']} |" for language in ("html", "css", "javascript")],
        "",
        "Role-level expected and observed values for every task are in the accompanying JSON audit.",
        "",
        "Name-preservation failures:",
        "",
        *([f"- `{record['taskId']}` ({record['language']}/{record['family']}): failed roles {', '.join(item['role'] for item in record['sourceNameRoleObservations'] if not item['pass'])}." for record in name_failures] or ["- None observed."]),
        "",
        "## 4. Requested-value selection results",
        "",
        f"{counts['requestedValueSelectionPasses']}/72 requested values were selected correctly. Missing, malformed, duplicated, or ambiguous target structures count as failures.",
        "",
        "| Language | Correct selected values | Tasks |",
        "|---|---:|---:|",
        *[f"| {language} | {per_language[language]['values']} | {per_language[language]['tasks']} |" for language in ("html", "css", "javascript")],
        "",
        "Each extracted literal, structural-validity result, and failure reason is in the accompanying JSON audit.",
        "",
        "Requested-value failures:",
        "",
        *([f"- `{record['taskId']}` requested `{record['requestedLiteral']}`; extracted `{record['requestedValueSelection']['value']}`; structural validity `{record['requestedValueSelection']['structurallyValid']}`; reason `{record['requestedValueSelection']['reason']}`; source-name preservation `{record['sourceNamePreservation']}`." for record in value_failures] or ["- None observed."]),
        "",
        "## 5. Cross-category failure matrix",
        "",
        "| Source names | Requested value | Tasks |",
        "|---|---|---:|",
        f"| Pass | Pass | {summary['crossCategoryFailureMatrix']['namesPass_valuePass']} |",
        f"| Fail | Pass | {summary['crossCategoryFailureMatrix']['namesFail_valuePass']} |",
        f"| Pass | Fail | {summary['crossCategoryFailureMatrix']['namesPass_valueFail']} |",
        f"| Fail | Fail | {summary['crossCategoryFailureMatrix']['namesFail_valueFail']} |",
        "",
        "## 6. Unrelated-file modifications",
        "",
        f"{counts['unrelatedContentPreservationPasses']}/72 tasks preserved every byte outside the authorized target span.",
        "",
        "| Modification category | Tasks |",
        "|---|---:|",
        *([f"| {category} | {len(task_ids)} |" for category, task_ids in sorted(modification_tasks.items())] or ["| None observed | 0 |"]),
        "",
        *(["Affected task IDs:", ""] + [f"- `{category}`: {', '.join(f'`{task_id}`' for task_id in sorted(task_ids))}" for category, task_ids in sorted(modification_tasks.items())] if modification_tasks else []),
        "The JSON audit includes every differing prefix or suffix hunk, categorized as insertion, deletion, or replacement, with byte ranges and expected/observed text.",
        "",
        "## 7. Exact-file correctness",
        "",
        f"{counts['exactFileCorrect']}/72 decoded responses exactly matched their frozen expected files as UTF-8 bytes.",
        "",
        "## 8. Contrast-group results",
        "",
        f"Fully correct groups: {sum(group['fullyCorrect'] for group in summary['contrastGroups'])}/24. Per-group output diversity, extracted-value diversity, correct values, and exact correctness are in the audit record.",
        "",
        "| Language | Name roles | Requested values | Unrelated bytes | Exact files | Unscorable boundaries | Tasks |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {language} | {per_language[language]['names']} | {per_language[language]['values']} | "
            f"{per_language[language]['unrelated']} | {per_language[language]['exact']} | "
            f"{per_language[language]['unscorable']} | {per_language[language]['tasks']} |"
            for language in ("html", "css", "javascript")
        ],
        "",
        "| Language / family | Name roles | Requested values | Unrelated bytes | Exact files | Unscorable boundaries | Tasks |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {language} / {family} | {sum(item['sourceNamePreservation'] for item in selected)} | "
            f"{sum(item['requestedValueSelection']['pass'] for item in selected)} | "
            f"{sum(item['unrelatedContentPreservation'] for item in selected)} | "
            f"{sum(item['exactFileCorrectness'] for item in selected)} | "
            f"{sum(not item['unrelatedContentScorable'] for item in selected)} | {len(selected)} |"
            for (language, family), selected in sorted(per_family.items())
        ],
        "",
        "## 9. Training-target overlap observations",
        "",
        f"{counts['trainingTargetOutputMatches']}/72 outputs exactly matched an expected file in the frozen P3-02 authored training split. This is output overlap evidence only; it does not prove memorization or an internal cause.",
        "",
        "## 10. Limitations and proposed next investigation",
        "",
        "These measurements classify observable output behavior. They do not establish an internal architectural or training cause and do not establish a general model-quality claim. The full-pass result, if any, supports only a proposal for further independent evaluation. The P3-02 gate remains failed and no training is authorized by this diagnostic.",
        "",
        "## Preregistered decision",
        "",
        f"- Full diagnostic pass: **{'yes' if summary['decision']['fullDiagnosticPass'] else 'no'}**.",
        f"- Preservation failure observed: **{'yes' if summary['decision']['preservationFailureObserved'] else 'no'}**.",
        f"- Requested-value selection failure observed: **{'yes' if summary['decision']['valueSelectionFailureObserved'] else 'no'}**.",
        f"- Tasks with an unscorable unrelated-content boundary: **{summary['decision']['unscorablePreservationTasks']}**.",
        f"- Decision outcome: **{summary['decision']['outcome']}**.",
        f"- Next proposal: {summary['decision']['nextProposal']}",
        "- Internal cause inferred: **no**.",
        "",
    ]
    return "\n".join(lines)


def analyze(manifest_path: Path, response_path: Path, report_path: Path, audit_path: Path) -> None:
    manifest_raw, manifest = _read_json(manifest_path)
    response_raw, envelope = _read_json(response_path)
    if not isinstance(manifest, dict) or not isinstance(envelope, dict):
        raise ValueError("Manifest and prediction envelope must be JSON objects")
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Frozen P3-03 manifest hash mismatch: {manifest_sha}")
    if envelope.get("manifestSha256") != manifest_sha:
        raise ValueError("Prediction envelope manifest SHA-256 does not match the frozen manifest")
    if type(envelope.get("schemaVersion")) is not int or envelope["schemaVersion"] != 1:
        raise ValueError("Prediction envelope schema version must be integer 1")
    if manifest.get("status") != "prepared-unexecuted-evaluation-not-authorized":
        raise ValueError("Unexpected manifest status")
    if manifest.get("taskCount") != 72 or manifest.get("contrastGroupCount") != 24:
        raise ValueError("Manifest counts differ from the preregistered design")
    execution = _validate_execution(manifest, envelope)
    training_rows, train_targets, training_hash = _load_p302_training_targets()
    if len(training_rows) != P302_TRAIN_RECORDS:
        raise ValueError("Unexpected P3-02 training split count")
    summary, audit = _score(manifest, envelope, train_targets)
    analysis_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    audit["manifestSha256"] = manifest_sha
    audit["predictionEnvelopeSha256"] = hashlib.sha256(response_raw).hexdigest()
    audit["p302TrainingCandidateSha256"] = training_hash
    audit["analysisCodeSha256"] = analysis_sha
    audit["execution"] = execution
    audit["summary"] = summary
    report_text = _markdown_report(manifest, manifest_sha, execution, training_hash, analysis_sha, summary, audit)
    audit_text = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    destinations = [report_path, audit_path]
    protected = [manifest_path, response_path, P302_CANDIDATE]
    resolved_destinations = [path.resolve() for path in destinations]
    if resolved_destinations[0] == resolved_destinations[1] or any(
        destination == protected_path.resolve() for destination in resolved_destinations for protected_path in protected
    ):
        raise ValueError("Report and audit paths must be separate from each other and all inputs")
    for path in destinations:
        if path.parent.is_symlink() or not path.parent.is_dir() or path.exists() or path.is_symlink():
            raise ValueError(f"Output parent must exist and output must be a new regular path: {path}")
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report_text)
    try:
        with audit_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(audit_text)
    except Exception:
        report_path.unlink(missing_ok=True)
        raise
    print(json.dumps({"status": "analysis-complete", "report": str(report_path),
                      "audit": str(audit_path), "counts": summary["counts"],
                      "fullDiagnosticPass": summary["decision"]["fullDiagnosticPass"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--responses", required=True, type=Path,
                        help="Separately authorized prediction envelope JSON; this tool never generates responses")
    parser.add_argument("--report", required=True, type=Path, help="New Markdown report path; existing files are refused")
    parser.add_argument("--audit", required=True, type=Path, help="New JSON audit path; existing files are refused")
    args = parser.parse_args()
    analyze(args.manifest, args.responses, args.report, args.audit)


if __name__ == "__main__":
    main()
