"""Read-only audit of two explicit historical corpora; writes a create-only receipt.

No downloads, source execution, tokenizer/model work, or final-holdout reads.
Run from any directory with --output pointing to a new review-receipt file.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
CORPORA = (
    ("p1-14-starter-v1", "training/dataset-sources.starter-v1.json",
     "58845cbc9251c0c33d5f2980d8b47d28dccb05638d68ac572a5841cdf96c8fcc"),
    ("p2-26-web-batch-2-v2", "training/pretraining/sources.p2-26-batch-2.local.json",
     "2f02f199d101050c9939df6d389851e46bbd858157cb68d7cb8cc75863178a91"),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, limit=1024 * 1024):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular local input: " + str(path))
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Review input exceeds its byte limit")
    return raw


def contained(root, relative):
    p = root / relative
    p.resolve(strict=True).relative_to(root.resolve(strict=True))
    return p


def audit():
    report = {"kind": "plex600-m3-historical-subset-review-v1",
              "scope": "local historical accepted files and notices only",
              "driverSha256": sha(read(Path(__file__))), "corpora": [],
              "upstreamRevisionReauthenticated": False, "modelExecuted": False,
              "newCorpusPrepared": False, "finalHoldoutAccessed": False}
    all_content = defaultdict(list)
    lock = json.loads(read(ROOT / "training/dataset-source-lock.json"))
    locked = {s["id"]: s for s in lock["sources"]}
    for name, catalog_name, expected in CORPORA:
        directory = ROOT / "training/artifacts/datasets" / name
        manifest_raw = read(directory / "manifest.json")
        if sha(manifest_raw) != expected:
            raise ValueError("Historical manifest hash mismatch")
        manifest = json.loads(manifest_raw)
        catalog_path = ROOT / catalog_name
        catalog_raw = read(catalog_path)
        if sha(catalog_raw) != manifest["sourceCatalogSha256"]:
            raise ValueError("Historical source catalog hash mismatch")
        catalogs = {s["id"]: s for s in json.loads(catalog_raw)["sources"]}
        sources = {}
        raw_roots = {}
        for s in manifest["sources"]:
            c = catalogs[s["id"]]
            for key in ("revision", "origin", "groupId", "licenseId"):
                if c[key] != s[key]:
                    raise ValueError("Catalog/source identity mismatch")
            raw_root = contained(catalog_path.parent, c["localPath"])
            notice = read(contained(directory, s["licenseNoticeFile"]))
            original_notice = read(contained(raw_root, c["licenseEvidence"]))
            if sha(notice) != s["licenseNoticeSha256"] or notice != original_notice:
                raise ValueError("Packaged notice differs from source evidence")
            result = {"id": s["id"], "revision": s["revision"], "groupId": s["groupId"],
                      "recordedLicenseId": s["licenseId"], "noticeSha256": sha(notice),
                      "noticeBytes": len(notice), "packagedNoticeMatchesLocalSource": True,
                      "splitCounts": Counter(), "extensions": Counter(), "acceptedPaths": []}
            if name == "p1-14-starter-v1":
                for item in locked[s["id"]]["files"]:
                    raw = read(contained(raw_root, item["path"]))
                    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                    if len(raw) != item["bytes"] or blob != item["gitBlobSha1"]:
                        raise ValueError("Starter source differs from upstream file lock")
                result["lockedFilesVerified"] = len(locked[s["id"]]["files"])
            sources[s["id"]], raw_roots[s["id"]] = result, raw_root
        identity_rows = []
        for split in ("train", "validation"):
            digest, count = hashlib.sha256(), 0
            with (directory / (split + ".jsonl")).open("rb") as stream:
                while line := stream.readline(8 * 1024 * 1024 + 1):
                    if len(line) > 8 * 1024 * 1024:
                        raise ValueError("Oversize JSONL record")
                    digest.update(line)
                    row = json.loads(line)
                    s = sources[row["sourceId"]]
                    if row["groupId"] != s["groupId"]:
                        raise ValueError("Record/source group mismatch")
                    raw = read(contained(raw_roots[s["id"]], row["path"]))
                    normalized = unicodedata.normalize("NFC", raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n"))
                    if normalized != row["text"] or sha(normalized.encode()) != row["contentSha256"]:
                        raise ValueError("Accepted source does not reproduce corpus text")
                    if name == "p1-14-starter-v1" and row["path"] not in {x["path"] for x in locked[s["id"]]["files"]}:
                        raise ValueError("Accepted starter path is absent from source lock")
                    s["splitCounts"][split] += 1
                    s["extensions"][Path(row["path"]).suffix.lower()] += 1
                    s["acceptedPaths"].append(row["path"])
                    identity = {"corpus": name, "split": split, "sourceId": s["id"],
                                "path": row["path"], "recordId": row["recordId"],
                                "contentSha256": row["contentSha256"]}
                    identity_rows.append(identity)
                    all_content[row["contentSha256"]].append(identity)
                    count += 1
            if digest.hexdigest() != manifest["summary"][split + "JsonlSha256"] or count != manifest["summary"][split + "Records"]:
                raise ValueError("Historical split hash/count mismatch")
        for s in manifest["sources"]:
            if len(sources[s["id"]]["acceptedPaths"]) != s["acceptedFiles"]:
                raise ValueError("Source accepted-file count mismatch")
            sources[s["id"]]["acceptedPaths"].sort()
        report["corpora"].append({"id": name, "manifestSha256": expected,
            "catalogSha256": sha(catalog_raw), "orderedRecordIdentitySha256": sha(
                json.dumps(identity_rows, sort_keys=True, separators=(",", ":")).encode()),
            "recordsVerified": len(identity_rows), "sources": list(sources.values())})
    report["exactDuplicateGroups"] = [rows for rows in all_content.values() if len(rows) > 1]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit()
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"recordsVerified": sum(c["recordsVerified"] for c in receipt["corpora"]),
                      "sourcesVerified": sum(len(c["sources"]) for c in receipt["corpora"]),
                      "exactDuplicateGroups": len(receipt["exactDuplicateGroups"])}))
