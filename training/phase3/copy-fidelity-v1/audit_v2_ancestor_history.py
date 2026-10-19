"""SB03 V2: stream-verify EXPLICIT non-holdout training ancestors against 48 names.

This is a model-free, read-only, fail-closed ancestry-collision auditor.
It never loads checkpoints, imports torch, launches training/inference, creates
files, enumerates directories, or opens the project's sealed final holdout.

A checksum-clean user-supplied file list is evidence ONLY for those listed
files. It is NOT proof that every ancestor corpus has been inventoried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROPOSAL = Path.home() / (
    "PlexArtifacts/file-edits/p3-sb03-v2-feasibility/"
    "v2-proposal-unreviewed.json"
)
EXPECTED_V2_SHA256 = "5f957cc30b1c58e7e22a8e2565efbc143303431526a61ecc2b65ef0e741ebb21"
ORIGINAL_SHA256 = "9414d1da7672a386a920d1935723e9133d011c9c3a4029e5d5800f643835ca7a"
SAMPLER_SHA256 = "9ddf29a3a2972cee0c8c2f79db97d7ef3f823065a2db6ee0840e89bb01760c13"
CHECKPOINT_SHA256 = "c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230"
TOKENIZER_SHA256 = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
SCOPE = "p3-sb03-v2-non-holdout-ancestor-training-text-inventory-v1"
MAX_ENTRIES = 1000
MAX_FILE_BYTES = 200 * 1024**3
CHUNK = 1024 * 1024
ALLOWED = {".json", ".jsonl", ".txt", ".md", ".html", ".css", ".js", ".ts",
           ".py", ".csv", ".yaml", ".yml", ".xml", ".htm"}
SHA_CHARS = frozenset("0123456789abcdef")


def _regular_file(path: Path, *, max_bytes: int) -> None:
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or not 0 < path.stat().st_size <= max_bytes):
        raise ValueError("Input must be a nonempty, existing regular file within size cap")
    # Prevent the path and its real resolution from aliasing the protected data.
    canonical = path.resolve(strict=True)
    for item in (path, canonical):
        if any("holdout" in part.casefold() for part in item.parts):
            raise ValueError("Protected final-holdout path is forbidden")


def _read_proposal(path: Path) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    _regular_file(path, max_bytes=1024 * 1024)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_V2_SHA256:
        raise ValueError("V2 proposal SHA-256 differs from owner-confirmed identity")
    candidate = json.loads(raw)
    if (not isinstance(candidate, dict)
            or candidate.get("kind") != "p3-sb03-v2-proposed-token-count-matched-namespace-diversity"
            or candidate.get("status") != "local-proposal-only-global-history-unverified-no-training"
            or candidate.get("startingCheckpointSha256") != CHECKPOINT_SHA256
            or candidate.get("frozenTokenizerSha256") != TOKENIZER_SHA256
            or candidate.get("baseCandidateSha256") != ORIGINAL_SHA256
            or candidate.get("sampler", {}).get("drawIndexCanonicalSha256") != SAMPLER_SHA256
            or candidate.get("selection", {}).get("matchedAllOriginalTokenCounts") is not True
            or candidate.get("selection", {}).get("completeHistoricalTrainingInventoryVerified") is not False
            or candidate.get("authorization", {}).get("trainingAuthorized") is not False
            or candidate.get("authorization", {}).get("inferenceAuthorized") is not False
            or candidate.get("authorization", {}).get("finalHoldoutOpened") is not False
            or candidate.get("authorization", {}).get("optimizerUpdates") != 0):
        raise ValueError("V2 proposal provenance, fixed sampler or permissions mismatch")
    rows = candidate.get("rows")
    names = candidate.get("selectedTrainNamespaces")
    if (not isinstance(rows, list) or len(rows) != 384
            or not isinstance(names, dict)
            or set(names) != {"cedar", "lunar", "prism", "amber", "zephyr", "meadow"}
            or any(not isinstance(vals, list) or len(vals) != 8 for vals in names.values())):
        raise ValueError("Expected original 384-case V2 comparison and 48 train slots")
    flat = [word for vals in names.values() for word in vals]
    if (len(flat) != len(set(flat)) or len(flat) != 48
            or any(type(name) is not str or not name.isascii()
                   or not name.isalpha() or not name.islower()
                   or len(name) not in (5, 6) for name in flat)):
        raise ValueError("V2 selected names are invalid or duplicate")
    if sum(r["split"] == "train" for r in rows) != 288:
        raise ValueError("V2 training-record split changed")
    if len({r["namespace"] for r in rows if r["split"] == "train"}) != 48:
        raise ValueError("V2 train row namespace identities changed")
    return candidate, tuple(name.encode("ascii") for name in flat)


def _inventory(path: Path) -> list[dict[str, str]]:
    _regular_file(path, max_bytes=2 * 1024 * 1024)
    manifest = json.loads(path.read_bytes())
    if (not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1
            or manifest.get("scope") != SCOPE
            or not isinstance(manifest.get("files"), list)
            or not 1 <= len(manifest["files"]) <= MAX_ENTRIES):
        raise ValueError("Explicit non-holdout inventory must contain 1..1000 files")
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for entry in manifest["files"]:
        if (not isinstance(entry, dict) or set(entry) != {"path", "sha256", "role"}
                or not all(type(entry[k]) is str and entry[k] for k in entry)
                or entry["role"] not in {"ancestor-training-text", "ancestor-curated-source"}
                or len(entry["sha256"]) != 64
                or any(letter not in SHA_CHARS for letter in entry["sha256"])):
            raise ValueError("Invalid inventory entry or missing SHA-256/role")
        source = Path(entry["path"]).expanduser()
        _regular_file(source, max_bytes=MAX_FILE_BYTES)
        if source.suffix.casefold() not in ALLOWED:
            raise ValueError("Only declared plain-text training source files are supported")
        canonical = str(source.resolve(strict=True))
        if canonical in seen:
            raise ValueError("Duplicate historical training source path")
        seen.add(canonical)
        out.append({"path": canonical, "sha256": entry["sha256"],
                    "role": entry["role"]})
    return out


def _stream_sha_and_matches(
    stream: BinaryIO, words: tuple[bytes, ...], *, buffer_size: int = CHUNK,
) -> tuple[str, int, set[str]]:
    """Chunk-boundary-safe, case-insensitive ASCII substring scanner."""
    if not 1 <= buffer_size <= 8 * 1024 * 1024:
        raise ValueError("Unsupported bounded buffer size")
    digest = hashlib.sha256()
    total = 0
    found: set[str] = set()
    previous_tail = b""
    tail_bytes = max(map(len, words)) - 1
    while part := stream.read(buffer_size):
        if not isinstance(part, bytes):
            raise TypeError("History stream did not return bytes")
        total += len(part)
        if total > MAX_FILE_BYTES:
            raise ValueError("Input exceeded maximum permitted historical file bytes")
        digest.update(part)
        checked = (previous_tail + part).lower()
        for word in words:
            if word.decode("ascii") not in found and word in checked:
                found.add(word.decode("ascii"))
        previous_tail = checked[-tail_bytes:]
    return digest.hexdigest(), total, found


def audit(*, proposal: Path, inventory: Path) -> dict[str, Any]:
    _, names = _read_proposal(proposal)
    files = _inventory(inventory)
    conflicts = set()
    scanned_bytes = 0
    file_hits = 0
    detail = []
    for i, entry in enumerate(files):
        source = Path(entry["path"])
        with source.open("rb") as stream:
            digest, length, hits = _stream_sha_and_matches(stream, names)
        if digest != entry["sha256"]:
            raise ValueError(f"Non-holdout source {i + 1} SHA-256 mismatch; report withheld")
        scanned_bytes += length
        conflicts.update(hits)
        file_hits += bool(hits)
        detail.append({
            "inventoryIndex": i + 1, "role": entry["role"],
            "sha256": digest, "bytes": length,
            "collidingNames": sorted(hits),
        })
    return {
        "kind": "p3-sb03-v2-streaming-ancestor-screen-v1",
        "status": ("listed-sources-include-training-name-collisions"
                   if conflicts else "listed-sources-clear-completeness-unverified"),
        "v2ProposalSha256": EXPECTED_V2_SHA256,
        "selectedNames": 48,
        "inventoryFileCount": len(files),
        "verifiedSourceFiles": len(detail),
        "verifiedSourceBytes": scanned_bytes,
        "filesWithNameCollisions": file_hits,
        "collidingNames": sorted(conflicts),
        "fileResults": detail,
        "caseInsensitive": True,
        "historicalTrainingLineageComplete": False,
        "historyCoverageOnlyListedFiles": True,
        "readyForTraining": False,
        "modelLoaded": False,
        "optimizerUpdates": 0,
        "trainingAuthorized": False,
        "newInferenceAuthorized": False,
        "finalHoldoutOpened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=PROPOSAL)
    parser.add_argument("--inventory", type=Path, required=True,
                        help="Explicit complete path and SHA for each non-holdout ancestor file")
    args = parser.parse_args()
    try:
        print(json.dumps(audit(proposal=args.proposal, inventory=args.inventory),
                         indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"p3-sb03-v2-history-screen: {exc}\n")


if __name__ == "__main__":
    main()
