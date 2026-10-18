"""P3-SB03 V2: frozen-tokenizer source-name parity FEASIBILITY, NO MODEL RUN.

Screen every proposed replacement by the real SHA-pinned BPE tokenizer against
all its allocated four-case train groups. A name is accepted ONLY if each
original full prompt and each answer-plus-EOS token count matches exactly.

Default: pure read-only audit and JSON stdout. --prepare-local writes a *fresh*
proposal-only JSON file; never modifies V1, checkpoint, tokenizer, bundles,
history files or sealed final holdout. The code does NOT authorize inference,
training, stage creation, checkpoint selection, or a causal success claim.

Training ancestry outside the five SHA-pinned repository P3 corpora must be
separately audited; --history-inventory can screen additional explicitly listed
non-holdout files, but cannot attest that the inventory is complete.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location(
    "p3_sb03_v2_original_builder", HERE / "prepare_namespace_diversity_candidate.py"
)
assert SPEC and SPEC.loader
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)

V1_PROPOSAL_SHA = "4bbdc55782d9447271323378367831656dc1f05cb5bb174138738a6a7e5e7dea"
TOKENIZER = Path.home() / "PlexArtifacts/file-edits/p3-copy-fidelity-v1-bundle/tokenizer.json"
DEFAULT_OUTPUT = (
    Path.home() / "PlexArtifacts/file-edits/p3-sb03-v2-feasibility/"
    "v2-proposal-unreviewed.json"
)
SEED_TAG = "p3-sb03-v2-tokenizer-namespace-search-v1"
MAX_CANDIDATES_CAP = 20_000
DEFAULT_ATTEMPTS = 2500
MAX_LOCAL_INVENTORY_BYTES = 128 * 1024 * 1024


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _signature(tokenizer: Any, row: dict) -> tuple[int, int]:
    """Prompt and answer+EOS counts; never truncate or drop the source."""
    pids = tokenizer.encode(row["prompt"])
    aids = tokenizer.encode(row["expectedFile"])
    if (not pids or not aids
            or any(type(t) is not int or not 4 <= t < 16384 for t in pids + aids)):
        raise ValueError("Tokenizer emitted invalid ordinary-token IDs")
    if (len(pids) + 193 > 512 or len(aids) + 1 > 192
            or len(pids) + len(aids) + 1 > 512):
        raise ValueError("Full context/response budget exceeded")
    return len(pids), len(aids) + 1


def _roundtrip(tokenizer: Any, row: dict) -> None:
    for k in ("prompt", "expectedFile"):
        ids = tokenizer.encode(row[k])
        if tokenizer.decode(ids) != row[k]:
            raise ValueError(f"Tokenizer does not roundtrip {k}")


def _candidate_row(original: dict, new: str) -> dict:
    """Transform only train source namespace; never change requests or targets."""
    old = original["namespace"]
    if (original["split"] != "train"
            or old not in V1.OLD_NAMES or len(new) != len(old)
            or not new.isascii() or not new.islower() or not new.isalpha()):
        raise ValueError("Invalid scoped namespace replacement")
    row = dict(original)
    for k in ("id", "groupId", "prompt", "inputFile", "expectedFile"):
        if old not in row[k]:
            raise ValueError(f"Missing source name in {k}")
        row[k] = row[k].replace(old, new)
    row["namespace"] = new
    for k in ("prompt", "inputFile", "expectedFile"):
        if len(original[k]) != len(row[k]):
            raise ValueError("Source byte length not preserved")
    if any(row[k] != original[k] for k in (
        "language", "request", "targetStart", "targetEnd",
        "condition", "contrastIndex", "shape", "context",
        "requestedLiteral", "sourceLiteral",
    )):
        raise ValueError("The intervention changed an unrelated contract field")
    if row["condition"] == "copy":
        expected = row["inputFile"]
    else:
        expected = (
            row["inputFile"][:row["targetStart"]]
            + row["requestedLiteral"]
            + row["inputFile"][row["targetEnd"]:]
        )
    if (expected != row["expectedFile"]
            or row["prompt"] != (
                "Edit the supplied file to satisfy the request.\n"
                f"Language: {row['language']}\nRequest: {row['request']}\n"
                f"File:\n{row['inputFile']}\nEdited file:"
            )):
        raise ValueError("Expected full-file or prompt contract changed")
    return row


def _slot_rows(base: list[dict]) -> dict[tuple[str, int], list[tuple[int, dict]]]:
    """Allocate 12 original groups/name to the 8 stable slots of the V1 plan."""
    grouped: dict[str, dict[str, list[tuple[int, dict]]]] = defaultdict(dict)
    for i, row in enumerate(base):
        if row["split"] != "train":
            continue
        old = row["namespace"]
        grouped[old].setdefault(row["groupId"], []).append((i, row))
    if set(grouped) != set(V1.OLD_NAMES):
        raise ValueError("Original train namespace composition changed")
    result: dict[tuple[str, int], list[tuple[int, dict]]] = {}
    for name in V1.OLD_NAMES:
        groups = grouped[name]
        if len(groups) != 12:
            raise ValueError("Original 12 four-case groups per name changed")
        for index, rows in enumerate(groups.values()):
            if (len(rows) != 4
                    or {r["contrastIndex"] for _, r in rows} != {0, 1, 2, 3}):
                raise ValueError("Original four-case contrast group changed")
            result.setdefault((name, index % V1.SLOTS_PER_NAME), []).extend(rows)
    if (len(result) != 48 or sum(len(v) for v in result.values()) != 288
            or any(len(v) not in (4, 8) for v in result.values())):
        raise ValueError("Original slot allocation changed")
    return result


def _vocabulary_words(tokenizer: Any, length: int, old: str, slot: int) -> list[str]:
    backend = getattr(tokenizer, "backend", None)
    vocab = backend.get_vocab() if backend is not None else {}
    if not isinstance(vocab, dict):
        raise ValueError("Frozen tokenizer vocabulary cannot be inspected safely")
    words = {
        w for w, token_id in vocab.items()
        if (type(w) is str and type(token_id) is int and 4 <= token_id < 16384
            and len(w) == length and w.isascii()
            and w.isalpha() and w.islower())
    }
    # Candidate priority is frozen, not selected after a model result.
    return sorted(
        words,
        key=lambda w: (
            _sha(f"{SEED_TAG}|vocab|{old}|{slot}|{w}".encode("ascii")), w,
        ),
    )


def _candidate_stream(tokenizer: Any, old: str, slot: int) -> Iterator[str]:
    seen = set()
    for word in _vocabulary_words(tokenizer, len(old), old, slot):
        if word not in seen:
            seen.add(word)
            yield word
    attempt = 0
    while True:
        digest = hashlib.sha256(
            f"{SEED_TAG}|hash|{old}|{slot}|{attempt}".encode("ascii")
        ).digest()
        candidate = "".join(chr(97 + b % 26) for b in digest[:len(old)])
        if candidate not in seen:
            seen.add(candidate)
            yield candidate
        attempt += 1


def _optional_history(path: Path | None) -> tuple[list[bytes], dict[str, str]]:
    """Only explicit non-holdout paths; no directory walking or holdout reads."""
    if path is None:
        return [], {}
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Additional history inventory missing, linked or oversized")
    manifest = json.loads(path.read_bytes())
    if (not isinstance(manifest, dict)
            or manifest.get("scope") != "non-final-holdout-training-corpora"
            or type(manifest.get("files")) is not list
            or not manifest["files"] or len(manifest["files"]) > 100):
        raise ValueError("Expected explicit non-final-holdout history inventory")
    raw_files, sha_map = [], {}
    total_history_bytes = 0
    for item in manifest["files"]:
        if (not isinstance(item, dict)
                or set(item) != {"path", "sha256"}
                or type(item["path"]) is not str
                or type(item["sha256"]) is not str
                or len(item["sha256"]) != 64):
            raise ValueError("Invalid history file identity")
        source = Path(item["path"]).expanduser()
        if (not source.is_absolute() or source.is_symlink()
                or any("holdout" in part.lower() for part in source.parts)
                or not source.is_file()
                or not 0 < source.stat().st_size <= MAX_LOCAL_INVENTORY_BYTES):
            raise ValueError("Unsafe, oversized, or holdout history path refused")
        canonical = source.resolve()
        if (any("holdout" in part.lower() for part in canonical.parts)
                or str(canonical) in sha_map):
            raise ValueError("Holdout alias or duplicate history path refused")
        total_history_bytes += source.stat().st_size
        if total_history_bytes > MAX_LOCAL_INVENTORY_BYTES:
            raise ValueError("Aggregate history input exceeds memory limit")
        raw = source.read_bytes()
        if _sha(raw) != item["sha256"]:
            raise ValueError("History file SHA mismatch")
        sha_map[str(canonical)] = item["sha256"]
        raw_files.append(raw)
    return raw_files, sha_map


def _check_inputs() -> tuple[list[dict], list[bytes], set[str], dict[str, str]]:
    V1._checked(V1.OUTPUT, V1_PROPOSAL_SHA)
    old_bytes, prior_files = V1._historical()  # verifies all five frozen P3 files
    original = json.loads(V1._checked(V1.FROZEN, V1.SOURCE_SHA))
    rows = original["rows"]
    if (len(rows) != 384
            or Counter(r["split"] for r in rows) != {"train": 288, "validation": 96}):
        raise ValueError("Original 288/96 frozen split changed")
    return rows, list(old_bytes.values()), prior_files, {
        k: _sha(v) for k, v in sorted(old_bytes.items())
    }


def search(
    tokenizer: Any, *, attempts_per_slot: int = DEFAULT_ATTEMPTS,
    extra_history: list[bytes] | None = None,
    extra_history_shas: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return bounded feasibility receipt and optional UNAPPROVED v2 proposal."""
    if (type(attempts_per_slot) is not int
            or not 1 <= attempts_per_slot <= MAX_CANDIDATES_CAP):
        raise ValueError("Invalid bounded candidate-search limit")
    if type(tokenizer.vocabulary_size) is not int or tokenizer.vocabulary_size != 16384:
        raise ValueError("Expected frozen 16384 vocabulary")
    original, previous, prior_files, source_shas = _check_inputs()
    extra_history = extra_history or []
    extra_history_shas = extra_history_shas or {}
    if any(type(data) is not bytes for data in extra_history):
        raise ValueError("Invalid additional history bytes")
    forbidden = previous + extra_history
    assigned = _slot_rows(original)
    signatures = [_signature(tokenizer, r) for r in original]
    for row in original:
        _roundtrip(tokenizer, row)
    excluded = set(V1.OLD_NAMES + V1.VAL_NAMES + V1.MICRO_NAMES)
    used_names = set(excluded)
    selection: dict[str, list[str]] = {old: [] for old in V1.OLD_NAMES}
    slot_records: dict[str, Any] = {}
    global_tried = 0
    new_rows = list(original)
    for (old, slot), rows in assigned.items():
        tested = 0
        eligible = None
        for candidate in _candidate_stream(tokenizer, old, slot):
            if tested >= attempts_per_slot:
                break
            tested += 1
            global_tried += 1
            if (candidate in used_names or any(
                candidate.encode("ascii") in raw for raw in forbidden
            )):
                continue
            replacement = []
            for index, row in rows:
                edited = _candidate_row(row, candidate)
                if _signature(tokenizer, edited) != signatures[index]:
                    break
                replacement.append((index, edited))
            else:
                if (any(edited["inputFile"] in prior_files
                        or edited["expectedFile"] in prior_files
                        for _, edited in replacement)):
                    continue
                for _, edited in replacement:
                    _roundtrip(tokenizer, edited)
                eligible = candidate
                for index, edited in replacement:
                    new_rows[index] = edited
                selection[old].append(candidate)
                used_names.add(candidate)
                break
        slot_records[f"{old}.{slot}"] = {
            "slot": slot, "priorSourceNamespace": old,
            "assignedCases": len(rows), "attempted": tested,
            "selectedNamespace": eligible,
            "matched": eligible is not None,
        }
        if eligible is None:
            # No half-complete V2 manifest is publishable. Stop on first
            # failed slot to prevent implying that partial coverage suffices.
            break
    full_match = (
        len(slot_records) == 48
        and all(item["matched"] for item in slot_records.values())
    )
    receipt: dict[str, Any] = {
        "kind": "p3-sb03-v2-namespace-token-feasibility-v1",
        "status": ("token-parity-feasible-for-five-p3-corpora"
                   if full_match else "token-parity-infeasible-within-search-budget"),
        "frozenV1Sha256": V1_PROPOSAL_SHA,
        "baseCurriculumSha256": V1.SOURCE_SHA,
        "tokenizerSha256": V1.TOKENIZER_SHA,
        "sampledSlots": len(slot_records), "requiredSlots": 48,
        "testedCandidates": global_tried, "attemptsPerSlot": attempts_per_slot,
        "slotResults": slot_records,
        "newNamesSelected": sum(len(v) for v in selection.values()),
        "fiveP3CorpPriorityOnly": True,
        "additionalHistoryFilesScreened": len(extra_history),
        "additionalHistorySha256": extra_history_shas,
        "entireHistoricalTrainingInventoryVerified": False,
        "matchedRows": 384 if full_match else None,
        "matchedValidationRows": 96 if full_match else None,
        "trainingReady": False,
        "trainingAuthorized": False, "newInferenceAuthorized": False,
        "modelLoaded": False, "optimizerUpdates": 0,
        "finalHoldoutOpened": False,
    }
    if not full_match:
        return receipt, None
    if (len(set(r["id"] for r in new_rows)) != 384
            or len({r["namespace"] for r in new_rows if r["split"] == "train"}) != 48
            or any(r != original[i] for i, r in enumerate(new_rows)
                   if r["split"] == "validation")
            or any(_signature(tokenizer, row) != sig
                   for row, sig in zip(new_rows, signatures, strict=True))):
        raise ValueError("Final V2 identity or tokenizer parity verification failed")
    groups = defaultdict(list)
    for row in new_rows:
        groups[row["groupId"]].append(row)
    if (len(groups) != 96
            or any(len(g) != 4
                   or {r["contrastIndex"] for r in g} != {0, 1, 2, 3}
                   for g in groups.values())):
        raise ValueError("V2 contrast group coherence changed")
    sampler = V1._draws()
    v1 = json.loads(V1._checked(V1.OUTPUT, V1_PROPOSAL_SHA))
    if sampler != v1["sampler"]["drawIndexArray"]:
        raise ValueError("Original frozen full-coverage sampler drift")
    # Only exact parity + known P3 screens. Other historical sources are NOT
    # proven absent, and no optimizer-stage or run may be constructed.
    draft = {
        "schemaVersion": 2,
        "kind": "p3-sb03-v2-proposed-token-count-matched-namespace-diversity",
        "status": "local-proposal-only-global-history-unverified-no-training",
        "startingCheckpointSha256": V1.ENDPOINT_SHA,
        "frozenTokenizerSha256": V1.TOKENIZER_SHA,
        "v1UnchangedProposalSha256": V1_PROPOSAL_SHA,
        "baseCandidateSha256": V1.SOURCE_SHA,
        "inputCorpusSha256": source_shas,
        "additionalHistorySha256": extra_history_shas,
        "selectedTrainNamespaces": selection,
        "selection": {
            "algorithm": SEED_TAG, "limitPerSlot": attempts_per_slot,
            "slotResults": slot_records,
            "vocabCandidatesExaminedFirst": True,
            "matchedAllOriginalTokenCounts": True,
            "completeHistoricalTrainingInventoryVerified": False,
        },
        "sampler": v1["sampler"],
        "counts": {
            "train": 288, "validation": 96,
            "trainNewNames": 48, "trainGroups": 72, "validationGroups": 24,
            "sourceTokenCountParityCases": 384, "exactPromptTokenParity": True,
            "exactAnswerAndEosTokenParity": True,
        },
        "authorization": {
            "modelLoaded": False, "optimizerUpdates": 0,
            "trainingAuthorized": False, "inferenceAuthorized": False,
            "stageCreated": False, "bundleCreated": False,
            "finalHoldoutOpened": False,
        },
        "rows": new_rows,
    }
    data = V1._canonical(draft)
    receipt["unapprovedV2ProposalSha256"] = _sha(data)
    receipt["unapprovedV2ProposalBytes"] = len(data)
    return receipt, draft


def tokenizer_from_file(path: Path = TOKENIZER) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Frozen tokenizer missing, linked or oversized")
    raw = path.read_bytes()
    if _sha(raw) != V1.TOKENIZER_SHA:
        raise ValueError("Frozen tokenizer byte SHA mismatch")
    from tokenizers import Tokenizer
    from plex_training.tokenizer import PlexTokenizer

    # Load from the already verified bytes rather than reopening mutable path.
    return PlexTokenizer(Tokenizer.from_str(raw.decode("utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--attempts-per-slot", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--history-inventory", type=Path,
                        help="Explicit non-holdout corpus file+SHA list; no automatic scan")
    parser.add_argument("--prepare-local", action="store_true",
                        help="Save a successful UNAPPROVED V2 proposal to a fresh local path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        extra, shas = _optional_history(args.history_inventory)
        receipt, proposal = search(
            tokenizer_from_file(args.tokenizer),
            attempts_per_slot=args.attempts_per_slot,
            extra_history=extra, extra_history_shas=shas,
        )
        if args.prepare_local:
            if proposal is None:
                raise ValueError("No complete 48-slot token-matched V2 candidate; nothing written")
            target = args.output.resolve(strict=False)
            root = (Path.home() / "PlexArtifacts").resolve()
            if (target == root or root not in target.parents
                    or any("holdout" in p.lower() or "checkpoint" in p.lower()
                           for p in target.parts)):
                raise ValueError("Output must be a fresh non-checkpoint PlexArtifacts path")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as fp:
                fp.write(V1._canonical(proposal))
            receipt["newLocalUnapprovedProposalFile"] = str(target)
        print(json.dumps(receipt, sort_keys=True, indent=2))
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.exit(2, f"p3-sb03-v2-feasibility: {exc}\n")


if __name__ == "__main__":
    main()
