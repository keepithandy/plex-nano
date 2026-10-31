"""Fail-closed, read-only candidate scope audit for untrained Plex 600M.

Never reads raw data, constructs a corpus, fits a tokenizer or updates weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

PRE = Path(__file__).resolve().parents[3] / "training" / "pretraining"
REVIEW = ("plex600-m3-source-subset-review-2026-10-31.json",
          "057f6aeb94df2cd8cbb9667964d6c007e0de5386d2be9382e8428905e9da2566")
HOLDS = ("plex600-m3-proposed-source-holds-v1.json",
         "020c2f331c3a1a9982e536b9cf7b3524616f96247bb8467a5429f5c2b9d115e5")
ROLES = {
    "microsoft-web-dev": ("microsoft-web-dev-for-beginners", "train"),
    "mdn-css-examples": ("source-family-mdn-web-examples", "train"),
    "mdn-dom-examples": ("source-family-mdn-web-examples", "train"),
    "mdn-interactive-examples": ("source-family-mdn-web-examples", "train"),
    "mdn-learning-area": ("source-family-mdn-web-examples", "train"),
    "soyaine-javascript30": ("github-soyaine-javascript30", "train"),
    "traversy-web-projects": ("bradtraversy-50projects50days", "validation"),
    "bulma-templates": ("github-bulmatemplates-bulma-templates", "validation"),
}
FIELDS = ("corpus", "split", "sourceId", "path", "recordId", "contentSha256", "groupId")


def read_json(name, expected_sha=None):
    path = PRE / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError("Missing, linked, or oversized metadata")
    raw = path.read_bytes()
    if expected_sha and hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Metadata SHA-256 mismatch")
    return json.loads(raw)


def verify(review, holds, plan):
    if review.get("kind") != "plex600-m3-historical-subset-review-v1":
        raise ValueError("Unrecognized review")
    if holds.get("kind") != "plex600-m3-proposed-source-holds-v1" or holds.get("status") != "review-proposal-only":
        raise ValueError("Unrecognized holds")
    if (plan.get("kind") != "plex600-m3-candidate-scope-v1"
            or plan.get("status") != "review-only"
            or any(plan.get(k) is not False for k in
                   ("rightsApproved", "preparationAuthorized", "trainingAuthorized", "holdoutOpened"))):
        raise ValueError("Scope cannot authorize work")
    if plan.get("quarantine") != "all-proposed-omissions-and-all-notice-flags":
        raise ValueError("Quarantine policy must remain fail-closed")
    if (type(plan.get("maxCandidateRecords")) is not int
            or not 1 <= plan["maxCandidateRecords"] <= 3000
            or plan.get("maxOutputBytes") != 16 * 1024**2):
        raise ValueError("Invalid bounded preparation target")
    sources = {}
    for corpus in review["corpora"]:
        if corpus["id"] not in ("p1-14-starter-v1", "p2-26-web-batch-2-v2"):
            raise ValueError("Unknown historical corpus")
        for source in corpus["sources"]:
            name = source["id"]
            if name not in ROLES or name in sources:
                raise ValueError("Unknown or duplicate source")
            group, split = ROLES[name]
            paths = source["acceptedPaths"]
            if (source["groupId"] != group or source["splitCounts"] != {split: len(paths)}
                    or len(paths) != len(set(paths))):
                raise ValueError("Changed family, split, or accepted paths")
            sources[name] = (corpus["id"], group, split, set(paths))
    if set(sources) != set(ROLES):
        raise ValueError("Incomplete source inventory")
    roles = plan.get("sourceRoles")
    expected_roles = {k: ("candidate-train" if s == "train" else "historical-regression-only")
                      for k, (_, s) in ROLES.items()}
    if roles != expected_roles:
        raise ValueError("Historical split cannot become fresh evaluation or training")

    def check(records, expected_count):
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError("Changed quarantine count")
        found = {}
        for record in records:
            if not isinstance(record, dict) or set(record) != set(FIELDS):
                raise ValueError("Incomplete held record identity")
            name, path = record["sourceId"], record["path"]
            if name not in sources:
                raise ValueError("Unknown held source")
            corpus, group, split, accepted = sources[name]
            if (path not in accepted or record["corpus"] != corpus or
                    record["groupId"] != group or record["split"] != split):
                raise ValueError("Held record outside reviewed source and split")
            if any(not isinstance(record[k], str) or not record[k] for k in FIELDS):
                raise ValueError("Invalid held identity")
            if any(len(record[k]) != 64 or
                   any(ch not in "0123456789abcdef" for ch in record[k])
                   for k in ("recordId", "contentSha256")):
                raise ValueError("Invalid held hash")
            key = (name, path)
            if key in found:
                raise ValueError("Duplicate held record")
            found[key] = tuple(record[k] for k in FIELDS)
        return found

    omitted = check(holds["proposedOmissions"], 69)
    flagged = check(holds["noticeTriageRecords"], 101)
    for key in omitted.keys() & flagged.keys():
        if omitted[key] != flagged[key]:
            raise ValueError("Conflicting held record identity")
    quarantined = omitted.keys() | flagged.keys()
    if len(quarantined) != 145:
        raise ValueError("Unexpected quarantine overlap")
    summary = []
    candidate = held_train = 0
    for name in sorted(sources):
        _, group, split, paths = sources[name]
        held = sum(k[0] == name for k in quarantined)
        count = len(paths) - held
        if split == "train":
            held_train += held
            candidate += count
        summary.append({"sourceId": name, "group": group, "role": expected_roles[name],
                        "historical": len(paths), "quarantined": held,
                        "candidateForReview": count if split == "train" else 0})
    if candidate > plan["maxCandidateRecords"]:
        raise ValueError("Candidate exceeds preparation target")
    return {"status": "review-only", "historicalRecords": sum(x["historical"] for x in summary),
            "quarantinedRecords": len(quarantined), "trainQuarantined": held_train,
            "candidateTrainRecords": candidate, "sources": summary,
            "rawDataRead": False, "newCorpusPrepared": False, "trainingAuthorized": False,
            "optimizerUpdates": 0, "finalHoldoutOpened": False}


def inspect():
    return verify(read_json(*REVIEW), read_json(*HOLDS),
                  read_json("plex600-m3-candidate-scope-v1.json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("Use --check: this tool never builds a dataset")
    print(json.dumps(inspect(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
