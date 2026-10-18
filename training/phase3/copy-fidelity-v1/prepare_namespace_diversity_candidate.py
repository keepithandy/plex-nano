"""P3-SB03: deterministic namespace-diversity candidate; NO model training/inference.

This freezes a *proposed* 288/96 intervention and 288/96 matched comparator
with every task except TRAIN source namespace left unchanged. It never touches
PlexArtifacts checkpoints, bundles, optimizer state, or sealed final holdout.

There are two independent blockers before any model experiment:
- historical training data held outside this repository has NOT been exhausted
  and must be SHA-inventoried/reviewed locally, without opening final holdout;
- frozen-tokenizer per-record token-count parity is NOT yet demonstrated.
A candidate artifact is NOT optimizer authorization, staging permission, or a
claim of clean randomized-name causal effects.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREP_PATH = HERE / "prepare_copy_fidelity.py"
FROZEN = HERE / "copy-fidelity-candidate-v1.json"
MICROPROBE = HERE / "source-binding-microprobe-v1.json"
OUTPUT = HERE / "sb03-namespace-diversity-proposal-v1.json"
OLD_NAMES = ("cedar", "lunar", "prism", "amber", "zephyr", "meadow")
VAL_NAMES = ("granite", "saffron")
MICRO_NAMES = ("ulvar", "nyxen")
SLOTS_PER_NAME = 8
SEED = 1337
DRAW_COUNT = 1600
EXPECTED_DRAW_SHA = "d496d160c5e579037061495f5cd3d64895b3d2209b2718d86d226677eadd818d"
SOURCE_SHA = "9414d1da7672a386a920d1935723e9133d011c9c3a4029e5d5800f643835ca7a"
MICRO_SHA = "0a30df999f8e1c0f12cf0f6e4a71e0eab814bdd99b3bd83d0e2184637ad31fda"
TOKENIZER_SHA = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
ENDPOINT_SHA = "c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230"
HISTORICAL = {
    ROOT / "training/phase3/drafts/p3-02-exact-replacements-v1.jsonl":
        "98d7ff276cf28d2509ad775c95c49ae0ffdb975de7e47f11e54430f3ca426403",
    ROOT / "training/phase3/diagnostics/p3-03-source-preservation-v1/p3-03-source-preservation-v1.json":
        "119eaa829825e3daa6c9156bd0ba84dfa32726dc2ebfa3d7cfd9ccf7612df578",
    ROOT / "training/phase3/diagnostics/p3-copy-edit-controls-v1/copy-edit-controls-v1.json":
        "17834efb801ac963a4672ae616ca47e9eea40b2568f01951c3891c7e4997f01a",
    FROZEN: SOURCE_SHA,
    MICROPROBE: MICRO_SHA,
}


def _checked(path: Path, digest: str, max_bytes: int = 1024 * 1024) -> bytes:
    if (path.is_symlink() or not path.is_file()
            or not 0 < path.stat().st_size <= max_bytes):
        raise ValueError(f"Required regular frozen source missing/oversized: {path.name}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Frozen source SHA mismatch: {path.name}")
    return raw


def _canonical(item: Any) -> bytes:
    return (json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n").encode("ascii")


def _historical() -> tuple[dict[str, bytes], set[str]]:
    raw = {}
    prior_files: set[str] = set()
    for p, sha in HISTORICAL.items():
        data = _checked(p, sha)
        raw[p.relative_to(ROOT).as_posix()] = data
        if p.suffix == ".jsonl":
            rows = [json.loads(line) for line in data.splitlines() if line]
        else:
            object_ = json.loads(data)
            rows = object_["rows"] if "rows" in object_ else object_["tasks"]
        for r in rows:
            prior_files.add(r["inputFile"])
            prior_files.add(r["expectedFile"])
    return raw, prior_files


def _names(previous: dict[str, bytes]) -> dict[str, list[str]]:
    forbidden = b"\n".join(previous.values())
    names: dict[str, list[str]] = {}
    used = set(OLD_NAMES + VAL_NAMES + MICRO_NAMES)
    for old in OLD_NAMES:
        group = []
        for slot in range(SLOTS_PER_NAME):
            for attempt in range(1000):
                digest = hashlib.sha256(
                    f"p3-sb03|{old}|{slot}|{attempt}".encode("ascii")
                ).digest()
                candidate = "".join(chr(97 + byte % 26) for byte in digest[:len(old)])
                if candidate not in used and candidate.encode("ascii") not in forbidden:
                    break
            else:
                raise ValueError("No usable disjoint source namespace available")
            if len(candidate) != len(old):
                raise ValueError("Namespace byte length drifted from comparator")
            group.append(candidate)
            used.add(candidate)
        names[old] = group
    if sum(map(len, names.values())) != 48 or len(used) != 48 + 10:
        raise ValueError("48 fresh namespace identities were not established")
    return names


def _draws() -> list[int]:
    """Frozen full-coverage shuffled epochs, not randrange with uncovered rows."""
    rng = random.Random(SEED)
    draw = []
    while len(draw) < DRAW_COUNT:
        epoch = list(range(288))
        rng.shuffle(epoch)
        draw.extend(epoch[:DRAW_COUNT - len(draw)])
    counters = Counter(draw)
    if (len(draw) != DRAW_COUNT or len(counters) != 288
            or min(counters.values()) < 5 or max(counters.values()) > 6
            or hashlib.sha256(_canonical(draw)).hexdigest() != EXPECTED_DRAW_SHA):
        raise ValueError("Frozen balanced sampler identity/coverage drift")
    return draw


def make() -> dict[str, Any]:
    previous, old_files = _historical()
    source = json.loads(previous[FROZEN.relative_to(ROOT).as_posix()])
    if (len(source["rows"]) != 384
            or Counter(r["split"] for r in source["rows"]) != {
                "train": 288, "validation": 96,
            }
            or source.get("namespaceSplits") != {
                "train": list(OLD_NAMES), "validation": list(VAL_NAMES),
            }):
        raise ValueError("Source curriculum topology changed")
    names = _names(previous)
    slots: dict[str, list[str]] = defaultdict(list)
    transformed = []
    seen_groups: dict[str, dict[str, str]] = defaultdict(dict)
    for source_row in source["rows"]:
        row = dict(source_row)
        if row["split"] == "train":
            old = row["namespace"]
            if old not in OLD_NAMES or row["split"] != "train":
                raise ValueError("Only training namespaces may be edited")
            group = row["groupId"]
            if group not in seen_groups[old]:
                index = len(seen_groups[old]) % SLOTS_PER_NAME
                seen_groups[old][group] = names[old][index]
            new = seen_groups[old][group]
            for field in ("id", "groupId", "prompt", "inputFile", "expectedFile"):
                if old not in row[field]:
                    raise ValueError(f"Source namespace absent from original field: {field}")
                row[field] = row[field].replace(old, new)
            row["namespace"] = new
            slots[old].append(new)
        else:
            if row != source_row:
                raise ValueError("Validation row may not be modified")
        if (len(row["inputFile"]) != len(source_row["inputFile"])
                or len(row["expectedFile"]) != len(source_row["expectedFile"])
                or len(row["prompt"]) != len(source_row["prompt"])
                or row["request"] != source_row["request"]
                or row["targetStart"] != source_row["targetStart"]
                or row["targetEnd"] != source_row["targetEnd"]
                or row["condition"] != source_row["condition"]
                or row["contrastIndex"] != source_row["contrastIndex"]
                or row["language"] != source_row["language"]
                or row["shape"] != source_row["shape"]
                or row["context"] != source_row["context"]):
            raise ValueError("Namespace is not the sole intervention variable")
        expected = (
            row["inputFile"] if row["condition"] == "copy" else
            row["inputFile"][:row["targetStart"]] + row["requestedLiteral"]
            + row["inputFile"][row["targetEnd"]:]
        )
        if (row["expectedFile"] != expected
                or row["prompt"] != (
                    "Edit the supplied file to satisfy the request.\n"
                    f"Language: {row['language']}\nRequest: {row['request']}\n"
                    f"File:\n{row['inputFile']}\nEdited file:"
                )
                or len(row["prompt"].encode("ascii")) + 193 > 512
                or len(expected.encode("ascii")) + 1 > 192):
            raise ValueError(f"Full-file output or conservative context invalid: {row['id']}")
        transformed.append(row)
    if any(len(seen_groups[name]) != 12 for name in OLD_NAMES):
        raise ValueError("Expected exactly 12 training groups per original namespace")
    used = {row["namespace"] for row in transformed if row["split"] == "train"}
    if used != {name for pool in names.values() for name in pool}:
        raise ValueError("All 48 new namespaces must be used")
    if any(r["namespace"] in OLD_NAMES + MICRO_NAMES + VAL_NAMES
           for r in transformed if r["split"] == "train"):
        raise ValueError("Training/development namespace contamination")
    new_train_files = {r[field] for r in transformed if r["split"] == "train"
                       for field in ("inputFile", "expectedFile")}
    # The base train files are expected to differ, and validation files must
    # remain byte-identical; the new training files must not overlap ANY prior.
    if new_train_files & old_files:
        raise ValueError("New training sources/targets overlap historical files")
    groups = defaultdict(list)
    for r in transformed:
        groups[r["groupId"]].append(r)
    if (len(groups) != 96
            or len({r["id"] for r in transformed}) != 384
            or any(len(rows) != 4
                   or {r["contrastIndex"] for r in rows} != {0, 1, 2, 3}
                   or len({r["inputFile"] for r in rows}) != 1
                   or len({r["expectedFile"] for r in rows}) != 4
                   for rows in groups.values())):
        raise ValueError("Frozen four-case groups or row identities changed")
    draws = _draws()
    return {
        "schemaVersion": 1,
        "kind": "p3-sb03-proposed-namespace-diversity-matched-training-v1",
        "status": "frozen-proposal-not-token-matched-not-authorized",
        "hypothesis": "A: fixed-name template fallback versus diversified train names",
        "comparison": {
            "comparator": "unchanged SHA-pinned copy-fidelity-v1 288/96 candidate",
            "intervention": "exactly 48 deterministic new train namespaces across 72 four-case train groups",
            "topology": "original ordered 288 training and byte-identical 96 validation records",
            "independentVariable": "training source namespace diversity",
            "matchingStatus": "ASCII byte lengths matched; exact tokenizer per-record parity NOT measured",
            "globalHistoryStatus": "five repository-frozen corpora screened; complete local training history NOT inventoried",
            "comparatorRequiresSeparateOwnerAuthorization": True,
            "interventionRequiresSeparateOwnerAuthorization": True,
        },
        "fixedStartingCheckpointSha256": ENDPOINT_SHA,
        "tokenizerSha256": TOKENIZER_SHA,
        "historicalSourceSha256": {p.relative_to(ROOT).as_posix(): v
                                    for p, v in HISTORICAL.items()},
        "trainNamespacePool": names,
        "validationNamespaces": list(VAL_NAMES),
        "microprobeExcludedNamespaces": list(MICRO_NAMES),
        "sampler": {
            "policy": "seed1337-full-coverage-shuffled-epochs-1600-positions",
            "seed": SEED,
            "draws": DRAW_COUNT, "uniqueTrainRecords": len(set(draws)),
            "minimumRecordDraws": min(Counter(draws).values()),
            "maximumRecordDraws": max(Counter(draws).values()),
            "drawIndexCanonicalSha256": EXPECTED_DRAW_SHA,
            "drawIndexArray": draws,
        },
        "counts": {
            "train": 288, "validation": 96,
            "copyTrain": 72, "editTrain": 216,
            "copyValidation": 24, "editValidation": 72,
            "trainGroups": 72, "validationGroups": 24,
            "trainNewNamespaceCount": 48,
            "originalTrainGroupsPerNamespace": 12,
            "newTrainFileOverlapWithHistorical": 0,
            "maximumPromptBytes": max(len(r["prompt"].encode("ascii")) for r in transformed),
            "maximumAnswerBytes": max(len(r["expectedFile"].encode("ascii")) for r in transformed),
        },
        "authorization": {
            "candidatePreparationApproved": True,
            "tokenizerParityVerified": False,
            "entireHistoricalTrainingInventoryVerified": False,
            "modelLoaded": False, "optimizerUpdates": 0,
            "stageCreated": False, "bundleCreated": False,
            "trainingAuthorized": False, "newGenerationAuthorized": False,
            "finalHoldoutOpened": False,
        },
        "rows": transformed,
    }


def serialized() -> bytes:
    return _canonical(make())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true",
                        help="Create once for code review; cannot train or stage")
    parser.add_argument("--show-sampler", action="store_true",
                        help="Show frozen sampler and coverage only, no model")
    args = parser.parse_args()
    try:
        manifest = make()
        output = _canonical(manifest)
        if args.prepare:
            with OUTPUT.open("xb") as handle:
                handle.write(output)
        elif OUTPUT.read_bytes() != output:
            raise ValueError("Frozen candidate bytes are not deterministic")
        info = {
            "status": manifest["status"],
            "candidateSha256": hashlib.sha256(output).hexdigest(),
            "candidateRows": len(manifest["rows"]),
            "newTrainNamespaces": 48,
            "sampler": {k: v for k, v in manifest["sampler"].items()
                        if k != "drawIndexArray"},
            "trainingAuthorized": False,
            "modelLoaded": False,
            "optimizerUpdates": 0,
            "finalHoldoutOpened": False,
        }
        print(json.dumps(info, sort_keys=True, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"p3-sb03: {exc}\n")


if __name__ == "__main__":
    main()
