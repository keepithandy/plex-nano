"""Score a separately authorized P3 copy-versus-edit response envelope; no model I/O.

All checks apply to frozen complete-file outputs. A changed source boundary
makes requested-value correctness *unscorable*, not independently false.
No existing result is overwritten. No training, checkpoint or tokenizer loads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "copy-edit-controls-v1.json"
EXPECTED_MANIFEST_SHA256 = "17834efb801ac963a4672ae616ca47e9eea40b2568f01951c3891c7e4997f01a"
CHECKPOINT_SHA256 = "020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d"
TOKENIZER_SHA256 = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
BUNDLE_SHA256 = "0cd6df3a6d8eec1d7da8c7a02af067b15299393d8b0e1756a071be17b9aca354"
MAX_ENVELOPE_BYTES = 2_000_000
PREP_SPEC = importlib.util.spec_from_file_location("copy_edit_prep", HERE / "prepare_controls.py")
assert PREP_SPEC is not None and PREP_SPEC.loader is not None
PREP = importlib.util.module_from_spec(PREP_SPEC)
PREP_SPEC.loader.exec_module(PREP)


def read_manifest() -> dict[str, Any]:
    raw = MANIFEST_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen 42-task manifest identity mismatch")
    if raw != PREP.serialized_manifest():
        raise ValueError("Frozen task manifest no longer matches its source builder")
    manifest = json.loads(raw)
    if len(manifest.get("tasks", [])) != 42:
        raise ValueError("Frozen diagnostic task count changed")
    return manifest


def _check_envelope(manifest: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
    if type(envelope) is not dict or type(envelope.get("schemaVersion")) is not int or envelope["schemaVersion"] != 1:
        raise ValueError("Invalid response-envelope schema")
    if envelope.get("manifestSha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Response envelope manifest hash mismatch")
    auth = envelope.get("evaluationAuthorization")
    if (type(auth) is not dict or auth.get("status") != "separately-authorized"
        or type(auth.get("evidenceReference")) is not str
        or not auth["evidenceReference"].strip()):
        raise ValueError("Response envelope lacks separate evaluation authorization evidence")
    meta = envelope.get("execution")
    if type(meta) is not dict:
        raise ValueError("Missing execution provenance")
    fixed = {
        "checkpointStep": 100, "checkpointSha256": CHECKPOINT_SHA256,
        "checkpointHashVerified": True, "tokenizerSha256": TOKENIZER_SHA256,
        "p302BundleManifestSha256": BUNDLE_SHA256,
        "policy": "greedy-unreselected-fixed-endpoint-v1",
        "seed": 1337, "temperature": 0,
        "maximumNewTokens": 192, "checkpointReselectionAllowed": False,
        "trainingPerformedDuringDiagnostic": False, "optimizerUpdatesDuringDiagnostic": 0,
        "finalProjectHoldoutOpened": False,
        "promptTemplateSha256": hashlib.sha256(PREP.PROMPT.encode("utf-8")).hexdigest(),
    }
    for key, expected in fixed.items():
        observed = meta.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(f"Execution provenance disagrees with fixed {key}")
    for key in ("codeRevision", "runnerSha256", "device"):
        if type(meta.get(key)) is not str or not meta[key].strip():
            raise ValueError(f"Missing execution provenance: {key}")
    rows, generation = envelope.get("responses"), envelope.get("generationAudit")
    tasks = manifest["tasks"]
    if (type(rows) is not list or type(generation) is not list
        or len(rows) != len(tasks) or len(generation) != len(tasks)):
        raise ValueError("Response/audit arrays must contain all 42 frozen tasks")
    seen = set()
    for expected, row, audit in zip(tasks, rows, generation, strict=True):
        if (type(row) is not dict or type(audit) is not dict
            or row.get("taskId") != expected["taskId"]
            or audit.get("taskId") != expected["taskId"]):
            raise ValueError("Response/audit task order or identity differs from frozen manifest")
        if row["taskId"] in seen or type(row.get("actual")) is not str:
            raise ValueError("Duplicate task ID or invalid response text")
        seen.add(row["taskId"])
        if (type(audit.get("generatedTokenCount")) is not int
            or not 0 <= audit["generatedTokenCount"] <= 192
            or type(audit.get("stoppedAtEos")) is not bool
            or audit.get("fullPromptVisibleForEveryGenerationStep") is not True):
            raise ValueError(f"Invalid generation status for {row['taskId']}")
    return meta


def score(manifest: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
    execution = _check_envelope(manifest, envelope)
    records = []
    for task, row, gen in zip(manifest["tasks"], envelope["responses"], envelope["generationAudit"], strict=True):
        observed = PREP.score_one(task, row["actual"])
        # A non-EOS result is an incomplete output even if its visible prefix
        # happens to equal the expected full file.
        completed = gen["stoppedAtEos"] and gen["fullPromptVisibleForEveryGenerationStep"]
        records.append({
            "taskId": task["taskId"],
            "language": task["language"],
            "condition": task["condition"],
            "namespace": task["namespace"],
            "contrastGroupId": task["contrastGroupId"],
            "exact": bool(observed["exact"] and completed),
            "rawTextExact": observed["exact"],
            "unrelatedBytesPreserved": observed["unrelatedBytesPreserved"],
            "requestedValueCorrect": observed["requestedValueCorrect"],
            "targetExtractable": observed["targetExtractable"],
            "stoppedAtEos": gen["stoppedAtEos"],
            "generatedTokenCount": gen["generatedTokenCount"],
        })
    by_group = defaultdict(list)
    for result in records:
        by_group[(result["language"], result["condition"])].append(result)
    summary = {}
    for (language, condition), group in sorted(by_group.items()):
        summary[f"{language}/{condition}"] = {
            "total": len(group),
            "exact": sum(t["exact"] for t in group),
            "preservation": sum(t["unrelatedBytesPreserved"] for t in group),
            "requestedValueCorrect": sum(t["requestedValueCorrect"] is True for t in group),
            "requestedValueIncorrect": sum(t["requestedValueCorrect"] is False for t in group),
            "requestedValueUnscorable": sum(t["requestedValueCorrect"] is None for t in group
                                             if condition != "copy"),
            "eosTerminated": sum(t["stoppedAtEos"] for t in group),
        }
    count_by_condition = {
        condition: sum(t["exact"] for t in records if t["condition"] == condition)
        for condition in ("copy", "minimal-edit", "context-edit")
    }
    if count_by_condition["copy"] != 6:
        decision = "copy-control-failed-review-source-copying"
    elif count_by_condition["minimal-edit"] != 18:
        decision = "minimal-edit-failed-review-target-and-request-binding"
    elif count_by_condition["context-edit"] != 18:
        decision = "context-edit-failed-review-unrelated-code-preservation"
    else:
        decision = "all-controls-passed-require-fresh-independent-confirmation"
    # 12 edit contrast groups, 3 values each: track complete groups only.
    contrast = defaultdict(list)
    for record in records:
        if record["condition"] != "copy":
            contrast[record["contrastGroupId"]].append(record)
    if len(contrast) != 12 or any(len(group) != 3 for group in contrast.values()):
        raise ValueError("The fixed 12 three-request contrast groups changed")
    fully_correct_groups = sum(all(r["exact"] for r in group) for group in contrast.values())
    return {
        "schemaVersion": 1,
        "kind": "p3-copy-versus-edit-42-task-read-only-scoring-v1",
        "manifestSha256": EXPECTED_MANIFEST_SHA256,
        "execution": execution,
        "evaluationAuthorization": envelope["evaluationAuthorization"],
        "total": 42,
        "exact": sum(t["exact"] for t in records),
        "byConditionExact": count_by_condition,
        "fullyCorrectEditContrastGroups": fully_correct_groups,
        "totalEditContrastGroups": 12,
        "byLanguageAndCondition": summary,
        "decision": decision,
        "trainingPerformed": False,
        "optimizerUpdates": 0,
        "finalProjectHoldoutOpened": False,
        "perTask": records,
    }


def report_md(summary: dict[str, Any]) -> str:
    lines = [
        "# P3 copy-versus-edit — fixed 42-task diagnostic",
        "",
        f"Manifest SHA-256: \x60{summary['manifestSha256']}\x60",
        "",
        f"Evaluation: {summary['exact']}/{summary['total']} complete exact files; "
        f"{summary['fullyCorrectEditContrastGroups']}/12 correct three-request edit groups.",
        "",
        "| Language / condition | Tasks | Exact | Preserved context | Value right | Value wrong | Unscorable | EOS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summary["byLanguageAndCondition"].items():
        lines.append(
            f"| {key} | {row['total']} | {row['exact']} | {row['preservation']} | "
            f"{row['requestedValueCorrect']} | {row['requestedValueIncorrect']} | "
            f"{row['requestedValueUnscorable']} | {row['eosTerminated']} |"
        )
    lines += [
        "", f"Decision branch: **{summary['decision']}**", "",
        "This is a read-only, fixed-endpoint development diagnostic. Unknown target value "
        "due to changed boundaries is unscorable, not proof of an independent value error.",
        "Successful controls do not replace P3-02's failed 0/36 development gate or "
        "authorize training. The final holdout stays closed.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            raise FileExistsError("Output directory exists; refusing overwrite")
        if args.responses.stat().st_size > MAX_ENVELOPE_BYTES:
            raise ValueError("Response envelope exceeds maximum accepted size")
        manifest = read_manifest()
        envelope = json.loads(args.responses.read_text(encoding="utf-8"))
        summary = score(manifest, envelope)
        result_text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        result_md = report_md(summary)
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        args.output_dir.mkdir(exist_ok=False)
        (args.output_dir / "audit.json").write_text(result_text, encoding="utf-8", newline="\n")
        (args.output_dir / "report.md").write_text(result_md, encoding="utf-8", newline="\n")
        print(json.dumps({"decision": summary["decision"], "exact": summary["exact"],
                          "report": str(args.output_dir / "report.md")}))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"p3-copy-edit-score: {exc}\n")


if __name__ == "__main__":
    main()
