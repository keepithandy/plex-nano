"""Bounded, read-only heuristics over two hash-pinned historical corpora.

This is a review aid, not a contamination or rights clearance. No source
program, tokenizer, model, network or final-holdout input is executed/opened.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "training/pretraining/plex600-m3-source-subset-review-2026-10-31.json"
RECEIPT_SHA = "057f6aeb94df2cd8cbb9667964d6c007e0de5386d2be9382e8428905e9da2566"
SHINGLE_TOKENS = 12
MIN_SHINGLES = 24
MAX_SECONDS = 120
TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PATTERNS = {
    "copyright": re.compile(r"\bcopyright\b|©", re.I),
    "license": re.compile(r"\blicen[sc]e(?:d|s)?\b|SPDX-License-Identifier", re.I),
    "terms": re.compile(r"all rights reserved|permission is hereby|redistribution|creative commons", re.I),
    "attribution": re.compile(r"\battribution\b|adapted from|copied from|based on|derived from", re.I),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bounded_read(path, limit):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected regular local review input")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Review input too large")
    return raw


def features(text):
    tokens = TOKEN.findall(text)
    return {hashlib.sha256(json.dumps(tokens[i:i + SHINGLE_TOKENS],
                ensure_ascii=False, separators=(",", ":")).encode()).digest()[:16]
            for i in range(max(0, len(tokens) - SHINGLE_TOKENS + 1))}


def similarity(a, b):
    if min(len(a), len(b)) < MIN_SHINGLES:
        return None
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    # Integer comparisons keep the reporting threshold exact.
    jaccard = intersection * 5 >= union * 4
    contained = intersection * 10 >= min(len(a), len(b)) * 9
    if not (jaccard or contained):
        return None
    return {"sharedShingles": intersection, "jaccard": intersection / union,
            "smallerFileContainment": intersection / min(len(a), len(b)),
            "jaccardFlag": jaccard, "containmentFlag": contained}


def notice_flags(text):
    return [{"line": i, "categories": hits}
            for i, line in enumerate(text.splitlines(), 1)
            if (hits := [name for name, pattern in PATTERNS.items() if pattern.search(line)])]


def audit():
    started = time.monotonic()
    raw = bounded_read(RECEIPT, 1024**2)
    if digest(raw) != RECEIPT_SHA:
        raise ValueError("Historical source-review receipt changed")
    reviewed = json.loads(raw)
    records, sets, notices = [], [], []
    total_bytes = total_shingles = 0
    for corpus in reviewed["corpora"]:
        directory = ROOT / "training/artifacts/datasets" / corpus["id"]
        raw_manifest = bounded_read(directory / "manifest.json", 1024**2)
        if digest(raw_manifest) != corpus["manifestSha256"]:
            raise ValueError("Historical manifest changed")
        manifest = json.loads(raw_manifest)
        identities = []
        for split in ("train", "validation"):
            raw_split = bounded_read(directory / (split + ".jsonl"), 16 * 1024**2)
            if digest(raw_split) != manifest["summary"][split + "JsonlSha256"]:
                raise ValueError("Historical split changed")
            rows = [json.loads(line) for line in raw_split.splitlines()]
            if len(rows) != manifest["summary"][split + "Records"]:
                raise ValueError("Historical count mismatch")
            for row in rows:
                text = row["text"]
                if digest(text.encode()) != row["contentSha256"]:
                    raise ValueError("Record content hash mismatch")
                identity = {"corpus": corpus["id"], "split": split,
                    **{k: row[k] for k in ("sourceId", "path", "recordId", "contentSha256")}}
                identities.append(identity)
                records.append(dict(identity, groupId=row["groupId"]))
                sets.append(features(text))
                total_bytes += len(text.encode())
                total_shingles += len(sets[-1])
                flags = notice_flags(text)
                if flags:
                    notices.append({"record": len(records) - 1, "lines": flags})
                if len(records) > 4000 or total_bytes > 16 * 1024**2 or total_shingles > 2_000_000:
                    raise ValueError("Review record/byte/shingle bound exceeded")
                if time.monotonic() - started > MAX_SECONDS:
                    raise ValueError("Review time bound exceeded")
        if digest(json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()) != corpus["orderedRecordIdentitySha256"]:
            raise ValueError("Ordered source identities differ from review receipt")
    pairs, compared = [], 0
    for i, record in enumerate(records):
        for j in range(i):
            if record["groupId"] == records[j]["groupId"]:
                continue
            compared += 1
            score = similarity(sets[i], sets[j])
            if score is not None:
                pairs.append({"left": j, "right": i, **score,
                              "crossSplit": record["split"] != records[j]["split"]})
            if len(pairs) > 5000 or compared > 8_000_000:
                raise ValueError("Review comparison/output bound exceeded")
        if time.monotonic() - started > MAX_SECONDS:
            raise ValueError("Review time bound exceeded")
    return {"kind": "plex600-m3-overlap-notice-review-v1", "sourceReviewSha256": RECEIPT_SHA,
            "driverSha256": digest(bounded_read(Path(__file__), 1024**2)),
            "protocol": {"shingleTokens": SHINGLE_TOKENS, "minimumUniqueShingles": MIN_SHINGLES,
                "jaccardThreshold": "4/5", "smallerFileContainmentThreshold": "9/10",
                "normalization": "case-sensitive Unicode word/punctuation tokens; whitespace ignored",
                "sameFamilyCompared": False, "timeLimitSeconds": MAX_SECONDS},
            "records": records, "comparedCrossFamilyPairs": compared,
            "recordsBelowShingleMinimum": sum(len(s) < MIN_SHINGLES for s in sets),
            "similarityFlags": pairs, "noticeFlags": notices,
            "noticeCategories": dict(Counter(k for n in notices for line in n["lines"] for k in line["categories"])),
            "modelExecuted": False, "finalHoldoutAccessed": False, "newCorpusPrepared": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing review receipt")
    result = audit()
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"records": len(result["records"]), "pairsCompared": result["comparedCrossFamilyPairs"],
                     "similarityFlags": len(result["similarityFlags"]), "noticeRecords": len(result["noticeFlags"])}))
