"""Plex Phase 1: local, byte-bound checkpoint inventory and M10 receipt comparison.

Pure standard library. Never imports torch, loads checkpoints, trains, infers,
downloads content, or changes an existing artifact. Evidence of file identity
does not prove that claimed training or inference occurred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any

SCHEMA = "plex-phase1-local-checkpoint-entry-v1"
SPEC_KEYS = {
    "schemaVersion", "entryId", "experimentId", "modelFamily",
    "checkpointKind", "checkpointPath", "modelSourcePath",
    "tokenizerManifestPath", "trainingReportPath", "datasetManifestPath",
    "sourceCommit", "optimizerUpdates", "parentEntryId",
    "parentCheckpointSha256", "evaluationReportPath", "evaluationResponsesPath",
}
FILE_FIELDS = {
    "checkpoint": ("checkpointPath", 4 * 1024**3),
    "modelSource": ("modelSourcePath", 2 * 1024**2),
    "tokenizerManifest": ("tokenizerManifestPath", 16 * 1024**2),
    "trainingReport": ("trainingReportPath", 16 * 1024**2),
    "datasetManifest": ("datasetManifestPath", 16 * 1024**2),
    "evaluationReport": ("evaluationReportPath", 16 * 1024**2),
    "evaluationResponses": ("evaluationResponsesPath", 16 * 1024**2),
}
MAX_JSON = 2 * 1024**2
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}\Z")
DECODERS = ("unmasked-v1", "fitted-vocab-v1")
TERMINATIONS = {
    "eos", "max-new-tokens", "invalid-control-token",
    "outside-fitted-tokenizer",
}


class Refusal(ValueError):
    pass


def _regular(path: Path, maximum: int) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise Refusal("Missing or inaccessible evidence file") from exc
    if (not stat.S_ISREG(info.st_mode)
            or bool(info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
            if hasattr(info, "st_file_attributes") else not stat.S_ISREG(info.st_mode)):
        raise Refusal("Evidence must be a regular file without reparse points")
    if info.st_size > maximum:
        raise Refusal("Evidence exceeds bounded file limit")


def _read_json(path: Path) -> dict[str, Any]:
    _regular(path, MAX_JSON)
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Refusal("Evidence JSON cannot be read") from exc
    if type(data) is not dict:
        raise Refusal("Evidence must be a JSON object")
    return data


def _file_hash(path: Path, maximum: int) -> dict[str, Any]:
    _regular(path, maximum)
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > maximum:
                raise Refusal("Evidence exceeded bounded read size")
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": total}


def _root(root: Path) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise Refusal("Repository root must be a real directory")
    return root.resolve(strict=True)


def _relative(root: Path, name: str, maximum: int) -> tuple[Path, dict[str, Any]]:
    if (type(name) is not str or not name or "\\" in name
            or Path(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise Refusal("Evidence paths must be relative POSIX-style paths under root")
    cursor = root
    for piece in name.split("/"):
        cursor = cursor / piece
        if cursor.is_symlink():
            raise Refusal("Evidence paths cannot traverse symlinks")
    try:
        cursor.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise Refusal("Evidence path escapes or does not exist under root") from exc
    return cursor, {"path": name, **_file_hash(cursor, maximum)}


def _optional_sha(value: Any) -> bool:
    return value is None or (type(value) is str and HEX64.fullmatch(value) is not None)


def _validate_spec(spec: dict[str, Any]) -> None:
    if set(spec) != SPEC_KEYS or spec.get("schemaVersion") != 1:
        raise Refusal("Checkpoint spec requires the exact v1 field set")
    if (type(spec["entryId"]) is not str or not IDENTIFIER.fullmatch(spec["entryId"])
            or type(spec["experimentId"]) is not str
            or not IDENTIFIER.fullmatch(spec["experimentId"])
            or spec["modelFamily"] not in ("plex-27m", "plex-600m")
            or spec["checkpointKind"] not in ("model-only", "training-state")
            or type(spec["sourceCommit"]) is not str
            or not HEX40.fullmatch(spec["sourceCommit"])
            or type(spec["optimizerUpdates"]) is not int
            or spec["optimizerUpdates"] < 0):
        raise Refusal("Checkpoint spec has invalid identity or training metadata")
    parent = spec["parentEntryId"]
    if (parent is not None
            and (type(parent) is not str or not IDENTIFIER.fullmatch(parent)
                 or parent == spec["entryId"])):
        raise Refusal("Invalid parent entry")
    if not _optional_sha(spec["parentCheckpointSha256"]):
        raise Refusal("Invalid parent checkpoint hash")
    if (parent is None) != (spec["parentCheckpointSha256"] is None):
        raise Refusal("Parent entry and hash must be supplied together")
    for name, (key, _) in FILE_FIELDS.items():
        val = spec[key]
        if name in ("datasetManifest", "evaluationReport", "evaluationResponses"):
            if val is None:
                continue
        if type(val) is not str or not val:
            raise Refusal("Required checkpoint evidence path missing")
    if (spec["evaluationReportPath"] is None) != (spec["evaluationResponsesPath"] is None):
        raise Refusal("Evaluation report and response paths must be supplied together")


def plan() -> dict[str, Any]:
    return {
        "schema": SCHEMA, "status": "plan-only",
        "operations": ["capture-create-only", "verify-read-only",
                       "inventory-read-only", "compare-m10-read-only"],
        "checkpointLoaded": False, "trainingAuthorized": False,
        "optimizerUpdatesExecuted": 0, "finalHoldoutUsed": False,
        "writesExistingArtifacts": False,
    }


def capture(root: Path, spec_path: str, out_path: str) -> dict[str, Any]:
    base = _root(root)
    source, spec_evidence = _relative(base, spec_path, MAX_JSON)
    spec = _read_json(source)
    _validate_spec(spec)
    evidence: dict[str, Any] = {}
    for name, (key, max_bytes) in FILE_FIELDS.items():
        value = spec[key]
        evidence[name] = None if value is None else _relative(base, value, max_bytes)[1]
    if evidence["evaluationReport"] is not None:
        report = _read_json(base / evidence["evaluationReport"]["path"])
        digest = evidence["evaluationResponses"]["sha256"]
        if (report.get("responsesSha256") != digest
                or report.get("checkpointSha256", report.get("m9WeightsSha256"))
                != evidence["checkpoint"]["sha256"]):
            raise Refusal("Evaluation receipts do not bind to checkpoint and raw responses")
    record = {
        "schema": SCHEMA, "entryId": spec["entryId"],
        "experimentId": spec["experimentId"],
        "modelFamily": spec["modelFamily"],
        "checkpointKind": spec["checkpointKind"],
        "sourceCommit": spec["sourceCommit"],
        "optimizerUpdatesDeclared": spec["optimizerUpdates"],
        "parentEntryId": spec["parentEntryId"],
        "parentCheckpointSha256": spec["parentCheckpointSha256"],
        "spec": spec_evidence,
        "files": evidence,
        "provenanceStatus": "local-bytes-verified-metadata-owner-declared",
        "inferenceVerified": False,
        "trainingAuthorized": False,
    }
    destination = base
    parts = out_path.split("/") if type(out_path) is str else []
    if (len(parts) < 2 or any(x in ("", ".", "..") for x in parts)
            or Path(out_path).is_absolute() or "\\" in out_path
            or not out_path.endswith(".json")):
        raise Refusal("Registry output must be a new relative JSON file in an existing directory")
    for piece in parts[:-1]:
        destination = destination / piece
        if destination.is_symlink() or not destination.is_dir():
            raise Refusal("Registry output must use a real preexisting directory")
    destination = destination / parts[-1]
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
            stream.write("\n")
    except FileExistsError as exc:
        raise Refusal("Registry is immutable: destination already exists") from exc
    return record


def verify(root: Path, manifest_path: str) -> dict[str, Any]:
    base = _root(root)
    file_path, _ = _relative(base, manifest_path, MAX_JSON)
    record = _read_json(file_path)
    if (record.get("schema") != SCHEMA or record.get("inferenceVerified") is not False
            or record.get("trainingAuthorized") is not False
            or record.get("provenanceStatus")
            != "local-bytes-verified-metadata-owner-declared"
            or set(record.get("files", {})) != set(FILE_FIELDS)):
        raise Refusal("Unsupported or malformed checkpoint registry entry")
    spec_record = record.get("spec")
    if type(spec_record) is not dict or set(spec_record) != {"path", "sha256", "bytes"}:
        raise Refusal("Registry does not bind its original spec")
    _, actual_spec = _relative(base, spec_record["path"], MAX_JSON)
    if actual_spec != spec_record:
        raise Refusal("Checkpoint metadata spec changed")
    spec = _read_json(base / spec_record["path"])
    _validate_spec(spec)
    expected = {
        "entryId": spec["entryId"], "experimentId": spec["experimentId"],
        "modelFamily": spec["modelFamily"], "checkpointKind": spec["checkpointKind"],
        "sourceCommit": spec["sourceCommit"],
        "optimizerUpdatesDeclared": spec["optimizerUpdates"],
        "parentEntryId": spec["parentEntryId"],
        "parentCheckpointSha256": spec["parentCheckpointSha256"],
    }
    if any(record.get(k) != v for k, v in expected.items()):
        raise Refusal("Registry identity differs from original spec")
    for name, (key, max_bytes) in FILE_FIELDS.items():
        value = spec[key]
        observed = record["files"][name]
        if value is None:
            if observed is not None:
                raise Refusal("Unexpected optional artifact evidence")
        elif (type(observed) is not dict or set(observed) != {"path", "sha256", "bytes"}
              or _relative(base, value, max_bytes)[1] != observed):
            raise Refusal("Bound checkpoint or evidence file changed")
    if record["files"]["evaluationReport"] is not None:
        report = _read_json(base / spec["evaluationReportPath"])
        if (report.get("responsesSha256")
                != record["files"]["evaluationResponses"]["sha256"]
                or report.get("checkpointSha256", report.get("m9WeightsSha256"))
                != record["files"]["checkpoint"]["sha256"]):
            raise Refusal("Evaluation report no longer binds to checkpoint evidence")
    return {
        "entryId": record["entryId"], "checkpointSha256": record["files"]["checkpoint"]["sha256"],
        "parentEntryId": record["parentEntryId"],
        "parentCheckpointSha256": record["parentCheckpointSha256"],
        "verifiedFileHashes": True, "modelLoaded": False,
        "inferenceVerified": False, "trainingAuthorized": False,
    }


def inventory(root: Path, directory: str) -> dict[str, Any]:
    base = _root(root)
    if (type(directory) is not str or not directory
            or any(x in ("", ".", "..") for x in directory.split("/"))
            or Path(directory).is_absolute() or "\\" in directory):
        raise Refusal("Inventory directory must be relative to root")
    folder = base
    for piece in directory.split("/"):
        folder /= piece
        if folder.is_symlink() or not folder.is_dir():
            raise Refusal("Inventory requires a real directory without links")
    entries = []
    for file in sorted(folder.iterdir()):
        if file.suffix != ".json" or not file.is_file() or file.is_symlink():
            raise Refusal("Inventory contains unexpected file or link")
        entries.append(verify(base, directory + "/" + file.name))
    if len(entries) > 1024:
        raise Refusal("Inventory entry limit exceeded")
    by_id = {r["entryId"]: r for r in entries}
    if len(by_id) != len(entries):
        raise Refusal("Duplicate checkpoint entries")
    for entry in entries:
        parent_id = entry["parentEntryId"]
        if parent_id is not None and parent_id in by_id:
            if (entry["parentCheckpointSha256"]
                    != by_id[parent_id]["checkpointSha256"]):
                raise Refusal("Parent checkpoint lineage hash mismatch")
    for entry in entries:
        seen = set()
        parent_id = entry["entryId"]
        while parent_id in by_id:
            if parent_id in seen:
                raise Refusal("Checkpoint lineage contains a cycle")
            seen.add(parent_id)
            parent_id = by_id[parent_id]["parentEntryId"]
            if parent_id is None:
                break
    return {
        "schema": SCHEMA, "entries": len(entries),
        "verifiedLocalFileHashes": True,
        "missingParentEntries": sorted({
            e["parentEntryId"] for e in entries
            if e["parentEntryId"] is not None and e["parentEntryId"] not in by_id
        }),
        "checkpointLineage": entries, "modelLoaded": False,
        "trainingAuthorized": False,
    }


def _receipt(directory: Path, baseline: bool) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate frozen M10 prompt identities, file hashes and decoder coverage."""
    from . import plex600_m10_copy_proposal as proposal
    if directory.is_symlink() or not directory.is_dir():
        raise Refusal("Comparison input must be a real directory")
    report_path, response_path = directory / "report.json", directory / "responses.jsonl"
    report = _read_json(report_path)
    raw_hash = _file_hash(response_path, 16 * 1024**2)["sha256"]
    if report.get("responsesSha256") != raw_hash:
        raise Refusal("Comparison response bytes differ from report hash")
    checkpoint = report.get("m9WeightsSha256") if baseline else report.get("checkpointSha256")
    if type(checkpoint) is not str or not HEX64.fullmatch(checkpoint):
        raise Refusal("Comparison missing checkpoint SHA-256")
    if baseline and report.get("protocol") != "plex600-m10-m9-read-only-copy-baseline-v1":
        raise Refusal("Comparison requires an M10 read-only M9 baseline")
    rows = proposal.rows()
    cases = {row["id"]: row for row in rows}
    if len(cases) != 18:
        raise Refusal("M10 proposal has unexpected case inventory")
    if baseline and set(report.get("selectedCaseIds", [])) != set(cases):
        raise Refusal("M10 baseline must contain every case")
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    with response_path.open("rb") as stream:
        for line in stream:
            if not line or not line.strip():
                raise Refusal("Unexpected blank JSONL record")
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise Refusal("Malformed comparison JSONL") from exc
            required = {
                "id", "output", "terminationReason", "generatedTokens",
                "language", "split", "decodingPolicy", "promptSha256",
                "m9WeightsSha256" if baseline else "checkpointSha256",
            }
            if type(row) is not dict or set(row) != required:
                raise Refusal("Unexpected response structure")
            case = cases.get(row["id"])
            key = (row["id"], row["decodingPolicy"])
            if (case is None or row["decodingPolicy"] not in DECODERS
                    or key in observed
                    or row["split"] != case["split"]
                    or row["language"] != case["language"]
                    or row["promptSha256"] != hashlib.sha256(
                        proposal.prompt_for(case).encode("utf-8")).hexdigest()
                    or row["m9WeightsSha256" if baseline else "checkpointSha256"] != checkpoint
                    or type(row["output"]) is not str or len(row["output"]) > 8192
                    or row["terminationReason"] not in TERMINATIONS
                    or type(row["generatedTokens"]) is not int
                    or not 0 <= row["generatedTokens"] <= 96):
                raise Refusal("Comparison has duplicate, mismatched, or unsafe responses")
            observed[key] = {
                "id": row["id"], "policy": row["decodingPolicy"], "split": row["split"],
                "exactAndEos": row["output"] == case["snippet"]
                and row["terminationReason"] == "eos",
                "eos": row["terminationReason"] == "eos",
                "checkpointSha256": checkpoint, "promptSha256": row["promptSha256"],
            }
    if len(observed) != 18 * len(DECODERS):
        raise Refusal("Comparison requires all 18 cases under both decoders")
    return observed


def compare_m10(baseline: Path, candidate: Path) -> dict[str, Any]:
    left = _receipt(baseline, True)
    right = _receipt(candidate, False)
    if (set(left) != set(right)
            or any(left[key]["promptSha256"] != right[key]["promptSha256"]
                   for key in left)):
        raise Refusal("Comparison prompts and decoding policies must match exactly")
    scores = {}
    for policy in DECODERS:
        for split in ("train", "heldout"):
            keys = [key for key in left if key[1] == policy and left[key]["split"] == split]
            b = sum(left[key]["exactAndEos"] for key in keys)
            c = sum(right[key]["exactAndEos"] for key in keys)
            scores[policy + "/" + split] = {
                "cases": len(keys), "baselineExactAndEos": b,
                "candidateExactAndEos": c, "deltaExactAndEos": c - b,
            }
    return {
        "protocol": "plex-phase1-m10-receipt-contrast-v1",
        "status": "file-hashes-and-cases-checked-inference-origin-not-proven",
        "baselineCheckpointSha256": next(iter(left.values()))["checkpointSha256"],
        "candidateCheckpointSha256": next(iter(right.values()))["checkpointSha256"],
        "scores": scores, "comparable": True, "trainingAuthorized": False,
        "inferenceIndependentlyProven": False, "finalHoldoutUsed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--capture", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--inventory", action="store_true")
    group.add_argument("--compare-m10", action="store_true")
    for option in ("root", "spec", "output", "manifest", "registry-dir",
                   "baseline-dir", "candidate-dir"):
        parser.add_argument("--" + option, type=str)
    args = parser.parse_args(argv)
    try:
        if args.capture:
            if not all((args.root, args.spec, args.output)):
                raise Refusal("Capture requires --root --spec --output")
            result = capture(Path(args.root), args.spec, args.output)
        elif args.verify:
            if not all((args.root, args.manifest)):
                raise Refusal("Verify requires --root --manifest")
            result = verify(Path(args.root), args.manifest)
        elif args.inventory:
            if not all((args.root, args.registry_dir)):
                raise Refusal("Inventory requires --root --registry-dir")
            result = inventory(Path(args.root), args.registry_dir)
        elif args.compare_m10:
            if not all((args.baseline_dir, args.candidate_dir)):
                raise Refusal("Compare requires both read-only receipt directories")
            result = compare_m10(Path(args.baseline_dir), Path(args.candidate_dir))
        else:
            if any(v is not None for v in vars(args).values()):
                raise Refusal("Default mode cannot receive file arguments")
            result = plan()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (Refusal, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
