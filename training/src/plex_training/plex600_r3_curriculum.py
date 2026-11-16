"""Plex 600M R3: quarantined source-conditioned file-edit candidate curriculum.

All 111 proposed rows are assistant-generated, not automatically owner-approved.
This module audits provenance, strict edit preservation, family disjointness,
evaluation overlap and *answer-only* causal training masks. It never trains,
approves a source, touches M7 checkpoints, or opens the final holdout.

The optional pack command materializes a create-only, explicitly quarantined
tokenized proposal using the exact frozen M7 tokenizer. NO optimizer can be
executed from this module and the output is not a training authorization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from . import plex600_m7 as m7
from . import plex600_m8_eval as m8
from . import plex600_owner_pilot as owner
from . import plex600_research as research
from . import plex600_setup as setup
from .tokenizer import PlexTokenizer, train_tokenizer

ROOT = research.ROOT
CANDIDATE = ROOT / "training/phase3/candidates/plex600-r3-source-conditioned-v1.jsonl"
PROTOCOL = "plex600-r3-source-conditioned-proposal-v1"
SOURCE_ORIGIN = "assistant-generated-new-candidate-proposed-for-owner-review"
SOURCE_SPLIT = "quarantined-candidate-no-training-approval"
RIGHTS_SCOPE = "candidate-preparation-only-owner-training-approval-required"
FAMILIES = 37
ROWS = FAMILIES * 3
TRAIN_FAMILIES = 28
VALIDATION_FAMILIES = 9
MAX_JSONL = 1024 * 1024


class Refusal(ValueError):
    pass


def source_sha256() -> str:
    return research.digest_file(CANDIDATE, MAX_JSONL)


def rows() -> list[dict[str, Any]]:
    items = research.read_jsonl(CANDIDATE, limit=MAX_JSONL)
    if len(items) != ROWS:
        raise Refusal("R3 candidate row count changed")
    expected_keys = {
        "schemaVersion", "id", "language", "family", "variant",
        "request", "inputFile", "expectedFile", "oldText", "newText",
        "sourceOrigin", "split", "trainingAuthorized", "rightsScope"
    }
    ids: set[str] = set()
    inputs: set[str] = set()
    groups: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for row in items:
        if (set(row) != expected_keys or type(row["schemaVersion"]) is not int
                or row["schemaVersion"] != 1
                or row["sourceOrigin"] != SOURCE_ORIGIN
                or row["split"] != SOURCE_SPLIT
                or row["rightsScope"] != RIGHTS_SCOPE
                or row["trainingAuthorized"] is not False
                or row["language"] not in research.LANGUAGES
                or row["variant"] not in ("base", "header", "footer")
                or not all(type(row[k]) is str and row[k] for k in
                           ("id", "family", "request", "inputFile",
                            "expectedFile", "oldText", "newText"))
                or row["id"] in ids
                or row["inputFile"] in inputs
                or row["id"] != f"{row['family']}-{row['variant']}"
                or row["family"] !=
                    f"r3-{row['language']}-{row['family'].removeprefix('r3-' + row['language'] + '-')}"
                or not research.exact_patch(
                    row, row["inputFile"], row["expectedFile"])
                or row["inputFile"] == row["expectedFile"]):
            raise Refusal("R3 candidate provenance, identity or exact edit invalid")
        ids.add(row["id"])
        inputs.add(row["inputFile"])
        counts[row["language"]] += 1
        family = groups.setdefault(row["family"], {
            "language": row["language"], "variants": set(),
            "operation": (row["oldText"], row["newText"], row["request"])
        })
        if (family["language"] != row["language"]
                or family["operation"] !=
                    (row["oldText"], row["newText"], row["request"])
                or row["variant"] in family["variants"]):
            raise Refusal("R3 family variants changed edit semantics")
        family["variants"].add(row["variant"])
    if (len(groups) != FAMILIES
            or any(v["variants"] != {"base", "header", "footer"}
                   for v in groups.values())
            or dict(counts) != {"html": 36, "css": 39, "javascript": 36}):
        raise Refusal("R3 language/family/variation counts changed")

    # M8 is the frozen development benchmark; never use it as training data.
    benchmark = research.benchmark()
    benchmark_ops = {(c["oldText"], c["newText"]) for c in benchmark}
    benchmark_input = {c["inputFile"] for c in benchmark}
    benchmark_answer = {c["expectedFile"] for c in benchmark}
    if any((c["oldText"], c["newText"]) in benchmark_ops
           or c["inputFile"] in benchmark_input
           or c["expectedFile"] in benchmark_answer for c in items):
        raise Refusal("R3 candidate overlaps frozen M8 exact edit/input/output")
    return items


def family_splits(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = defaultdict(list)
    for row in items:
        families[row["language"]].append(row["family"])
    splits = {"train": [], "validation": []}
    for language in sorted(research.LANGUAGES):
        unique = set(families[language])
        ranked = sorted(unique, key=lambda family: (
            hashlib.sha256(("r3-validation-v1:" + family).encode()).hexdigest(),
            family))
        if len(unique) < 6:
            raise Refusal("R3 training split lacks source families")
        splits["validation"].extend(ranked[:3])
        splits["train"].extend(ranked[3:])
    if (len(splits["train"]) != TRAIN_FAMILIES
            or len(splits["validation"]) != VALIDATION_FAMILIES
            or set(splits["train"]) & set(splits["validation"])):
        raise Refusal("R3 train/validation family isolation failed")
    return splits


def masked_record(prompt_ids: list[int], answer_ids: list[int], *,
                  fitted_vocab_size: int) -> dict[str, Any]:
    """Causal completion loss, never supervising prompt or source tokens."""
    if (type(fitted_vocab_size) is not int or fitted_vocab_size < 260
            or not prompt_ids or not answer_ids
            or answer_ids[-1] != 3
            or len(prompt_ids) + len(answer_ids) - 1 > 1024
            or any(type(token) is not int or token < 0
                   or token >= fitted_vocab_size
                   for token in prompt_ids + answer_ids)
            or any(token < 4 for token in answer_ids[:-1])):
        raise Refusal("R3 masked record is malformed or exceeds context")
    # Predictions at position prompt[-1] are trained on answer[0];
    # the EOS target is scored at the end. No BOS/PAD or prompt loss.
    inputs = prompt_ids + answer_ids[:-1]
    targets = ([-100] * (len(prompt_ids) - 1)) + answer_ids
    if (len(inputs) != len(targets)
            or targets.count(-100) != len(prompt_ids) - 1
            or targets[-1] != 3
            or sum(x != -100 for x in targets) != len(answer_ids)):
        raise Refusal("R3 causal mask alignment invalid")
    return {
        "inputIds": inputs, "targetIds": targets,
        "promptTokens": len(prompt_ids),
        "answerTokensIncludingEos": len(answer_ids),
        "lossMaskPromptPositions": len(prompt_ids) - 1,
        "supervisedAnswerPositions": len(answer_ids),
    }


def audit(tokenizer: Any | None = None) -> dict[str, Any]:
    items = rows()
    splits = family_splits(items)
    source_digest = source_sha256()
    selected = {family: name for name, families in splits.items()
                for family in families}
    by_split = Counter(selected[row["family"]] for row in items)
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "quarantined-candidate-audited-no-training-permission",
        "candidateSha256": source_digest,
        "benchmarkSha256": research.digest_file(research.BENCHMARK),
        "sourceOrigin": SOURCE_ORIGIN,
        "candidateRows": len(items),
        "distinctEditFamilies": FAMILIES,
        "variationsPerFamily": 3,
        "uniqueEditOperations": len({
            (row["oldText"], row["newText"]) for row in items
        }),
        "languages": dict(Counter(row["language"] for row in items)),
        "trainCandidateRows": by_split["train"],
        "validationCandidateRows": by_split["validation"],
        "trainCandidateFamilies": TRAIN_FAMILIES,
        "validationCandidateFamilies": VALIDATION_FAMILIES,
        "familySplitDisjoint": True,
        "exactM8EditOverlap": 0,
        "benchmarkNeverTrainingInput": True,
        "modelLoaded": False, "optimizerUpdates": 0,
        "trainingAuthorized": False, "sourceApprovedForTraining": False,
        "finalHoldoutUsed": False,
    }
    if tokenizer is None:
        return result
    all_packed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for case in items:
        prompt = m8.prompt_for(case)
        prefix = tokenizer.encode(prompt)
        gold = tokenizer.encode(case["expectedFile"]) + [3]
        if tokenizer.decode(prefix) != prompt or tokenizer.decode(gold[:-1]) != case["expectedFile"]:
            raise Refusal("R3 exact tokenizer roundtrip failed")
        packed = masked_record(
            prefix, gold, fitted_vocab_size=tokenizer.vocabulary_size)
        all_packed.append((case, packed))
    result.update({
        "tokenizerSha256": m7.TOKENIZER_SHA,
        "fittedVocabularySize": tokenizer.vocabulary_size,
        "maxInputPositions": max(p["promptTokens"] + p["answerTokensIncludingEos"] - 1
                                 for _, p in all_packed),
        "maxAnswerTokensIncludingEos": max(
            p["answerTokensIncludingEos"] for _, p in all_packed),
        "supervisedAnswerPositions": sum(
            p["supervisedAnswerPositions"] for _, p in all_packed),
        "maskedPromptPositions": sum(
            p["lossMaskPromptPositions"] for _, p in all_packed),
        "packedPreviewNoModel": True,
    })
    return result


def with_frozen_tokenizer(callback: Any) -> Any:
    # Fit only to the original preparation-authorized M7 training split, never
    # the R3 candidates; reconstruct the exact pinned model-embedding IDs.
    if owner.verify(owner.DATASET)["status"] != "verified-owner-authored-tokenizer-ready":
        raise Refusal("M7 source audit failed")
    if research.digest_file(owner.DATASET / "manifest.json", 1024 * 1024) != m7.SOURCE_SHA:
        raise Refusal("Original M7 dataset manifest differs")
    with tempfile.TemporaryDirectory(prefix="plex600-r3-tokenizer-") as temp:
        bundle = Path(temp) / "bundle"
        fitted = train_tokenizer(
            owner.DATASET, bundle, vocab_size=32768, min_frequency=2,
            model_family="plex-600m-v1", storage_limit_bytes=16 * 1024 * 1024)
        sha = research.digest_file(bundle / "manifest.json", 1024 * 1024)
        if (sha != research.BUNDLE_SHA
                or fitted["tokenizerSha256"] != m7.TOKENIZER_SHA
                or fitted["actualVocabularySize"] != 1191):
            raise Refusal("Reconstructed tokenizer not identical to frozen M7")
        bound = setup.check_source_binding(
            bundle, owner.DATASET, bundle_manifest_sha256=sha,
            source_manifest_sha256=m7.SOURCE_SHA)
        if bound["sourceBindingVerified"] is not True:
            raise Refusal("M7 tokenizer source binding invalid")
        return callback(PlexTokenizer.load(bundle))


def materialize(output: Path, tokenizer: Any) -> dict[str, Any]:
    """Create-only tokenized proposal; contains NO usable training approval."""
    out_root = (ROOT / "training/artifacts").resolve(strict=True)
    target = output.resolve(strict=False)
    if (output.is_symlink() or target.exists()
            or target.parent != out_root or not target.name.startswith("plex600-r3-")):
        raise Refusal("R3 candidate bundle must be new under training/artifacts/plex600-r3-*")
    inspected = audit(tokenizer)
    entries = rows()
    assignment = family_splits(entries)
    split_for = {f: split for split, families in assignment.items() for f in families}
    staged = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=out_root))
    try:
        files: dict[str, str] = {}
        for split in ("train", "validation"):
            prepared = []
            for case in entries:
                if split_for[case["family"]] != split:
                    continue
                sample = masked_record(
                    tokenizer.encode(m8.prompt_for(case)),
                    tokenizer.encode(case["expectedFile"]) + [3],
                    fitted_vocab_size=tokenizer.vocabulary_size)
                prepared.append({
                    "id": case["id"], "family": case["family"],
                    "language": case["language"], **sample})
            raw = ("\n".join(json.dumps(x, sort_keys=True, separators=(",", ":"))
                             for x in prepared) + "\n").encode("utf-8")
            if len(raw) > 4 * 1024 * 1024:
                raise Refusal("R3 packed proposal exceeds data budget")
            (staged / (split + ".jsonl")).write_bytes(raw)
            files[split + ".jsonl"] = hashlib.sha256(raw).hexdigest()
        manifest = {
            **inspected, "status": "packed-quarantined-candidate-not-authorized",
            "bundleSha256": research.BUNDLE_SHA,
            "filesSha256": files,
            "authorizingApprovalRecord": None,
        }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8", newline="\n")
        if target.exists():
            raise Refusal("R3 candidate destination appeared during packaging")
        os.rename(staged, target)
        staged = None
        return manifest
    finally:
        if staged is not None:
            shutil.rmtree(staged, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-frozen-tokenizer", action="store_true")
    parser.add_argument("--pack-quarantined-candidate", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.pack_quarantined_candidate is not None:
            result = with_frozen_tokenizer(
                lambda tokenizer: materialize(args.pack_quarantined_candidate, tokenizer))
        elif args.measure_frozen_tokenizer:
            result = with_frozen_tokenizer(audit)
        else:
            result = audit()
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print("Plex R3 preparation refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
