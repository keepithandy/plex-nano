"""Synthetic end-to-end M4 checks: local corpus only, no model or tokenizer."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from plex_training import plex600_m4 as m4
from plex_training import plex600_m3_scope as m3


def record(source, group, path, text):
    return {"sourceId": source, "groupId": group, "path": path,
            "recordId": m4.sha(path.encode("utf-8")),
            "contentSha256": m4.sha(text.encode("utf-8")), "text": text}


def input_fixture(root: Path, *, two_train=False):
    corpus = root / "historical" / "p1-14-starter-v1"
    (corpus / "licenses").mkdir(parents=True)
    names = {
        "microsoft-web-dev": ("microsoft-web-dev-for-beginners", "train"),
        "traversy-web-projects": ("bradtraversy-50projects50days", "validation"),
    }
    train = [record("microsoft-web-dev", names["microsoft-web-dev"][0],
                    "main.js", "function double(n) { return n * 2; }\n")]
    if two_train:
        train.append(record("microsoft-web-dev", names["microsoft-web-dev"][0],
                            "extra.js", "function triple(n) { return n * 3; }\n"))
    validation = [record("traversy-web-projects", names["traversy-web-projects"][0],
                         "style.css", ".card { color: red; }\n")]
    split_raw = {"train": b"".join(m4.json_bytes(row) for row in train),
                 "validation": b"".join(m4.json_bytes(row) for row in validation)}
    for split, raw in split_raw.items():
        (corpus / (split + ".jsonl")).write_bytes(raw)
    prior, sources = [], []
    for name, (group, split) in names.items():
        lic = ("License for " + name + "\n").encode()
        (corpus / "licenses" / (name + ".txt")).write_bytes(lic)
        paths = sorted(r["path"] for r in (train if split == "train" else validation))
        prior.append({"id": name, "groupId": group, "revision": "a" * 40,
                      "noticeSha256": m4.sha(lic), "acceptedPaths": paths})
        sources.append({"id": name, "groupId": group, "revision": "a" * 40,
                        "licenseNoticeFile": f"licenses/{name}.txt",
                        "licenseNoticeSha256": m4.sha(lic)})
    catalog_sha = "b" * 64
    manifest = {"sourceCatalogSha256": catalog_sha, "sources": sources,
                "summary": {f"{split}JsonlSha256": m4.sha(raw)
                            for split, raw in split_raw.items()}}
    manifest["summary"].update({f"{split}Records": len(records)
                               for split, records in (("train", train), ("validation", validation))})
    manifest_raw = m4.json_bytes(manifest)
    (corpus / "manifest.json").write_bytes(manifest_raw)
    identities = []
    for split, entries in (("train", train), ("validation", validation)):
        for r in entries:
            identities.append({"corpus": "p1-14-starter-v1", "split": split,
                               **{key: r[key] for key in ("sourceId", "path",
                                                          "recordId", "contentSha256")}})
    identity_sha = m4.sha(json.dumps(identities, sort_keys=True, separators=(",", ":")).encode())
    review = {"corpora": [{
        "id": "p1-14-starter-v1", "sources": prior,
        "manifestSha256": m4.sha(manifest_raw), "catalogSha256": catalog_sha,
        "orderedRecordIdentitySha256": identity_sha,
    }]}
    return names, review, train, validation, corpus


@contextmanager
def synthetic_build(root: Path, *, two_train=False, holds=None, budget=16 * 1024**2):
    roles, review, train, validation, corpus = input_fixture(root, two_train=two_train)
    holds = holds or {"proposedOmissions": [], "noticeTriageRecords": []}
    held = {(r["sourceId"], r["path"]) for r in holds["proposedOmissions"] + holds["noticeTriageRecords"]}
    scope = {"candidateTrainRecords": len(train) - len(held), "trainQuarantined": len(held)}
    plan = {"maxCandidateRecords": 3000, "maxOutputBytes": budget}
    policy_sha = m4.sha(m4.read_bounded(
        m3.PRE / "plex600-m3-candidate-scope-v1.json", m4.MAX_METADATA))
    decision = {"kind": "plex600-m4-rights-and-preparation-v1",
                "sourceReviewSha256": m3.REVIEW[1], "holdsSha256": m3.HOLDS[1],
                "scopeSha256": policy_sha,
                "m4PreparationApproved": True, "trainingAuthorized": False,
                "finalHoldoutAccessed": False,
                "sources": [{"sourceId": "microsoft-web-dev",
                             "rightsStatus": "approved", "reviewer": "synthetic-test",
                             "evidence": "test fixture", "approvedScope": "synthetic fixture only"}]}
    decision_file = root / "synthetic-decision.json"
    decision_file.write_bytes(m4.json_bytes(decision))
    values = {m3.REVIEW[0]: review, m3.HOLDS[0]: holds,
              "plex600-m3-candidate-scope-v1.json": plan}
    with patch.dict(m3.ROLES, roles, clear=True), patch.object(m3, "inspect", return_value=scope), \
            patch.object(m3, "read_json", side_effect=lambda key, *_: values[key]):
        yield root / "historical", decision_file, corpus, train, validation


class Plex600M4Tests(unittest.TestCase):
    def test_synthetic_build_is_deterministic_and_never_creates_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with synthetic_build(root) as (inputs, decision, _, train, _):
                first = m4.build(inputs, root / "one", decision)
                second = m4.build(inputs, root / "two", decision)
            self.assertEqual(first["manifestSha256"], second["manifestSha256"])
            self.assertEqual((root / "one/train.jsonl").read_bytes(),
                             (root / "two/train.jsonl").read_bytes())
            self.assertEqual(first["candidateTrainRecords"], 1)
            self.assertFalse((root / "one/validation.jsonl").exists())
            self.assertEqual(m4.verify_output(root / "one")["trainRecords"], 1)

    def test_corrupted_historical_split_refuses_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with synthetic_build(root) as (inputs, decision, corpus, *_):
                (corpus / "train.jsonl").write_bytes(b'{"malicious":true}\n')
                with self.assertRaisesRegex(ValueError, "split hash"):
                    m4.build(inputs, root / "bad", decision)
            self.assertFalse((root / "bad").exists())

    def test_rights_gate_precedes_corpus_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with synthetic_build(root) as (inputs, decision, *_):
                value = json.loads(decision.read_text())
                value["sources"][0]["rightsStatus"] = "pending"
                decision.write_bytes(m4.json_bytes(value))
                with patch.object(m4, "historical_records", side_effect=AssertionError("raw access")):
                    with self.assertRaisesRegex(ValueError, "unapproved rights"):
                        m4.build(inputs, root / "bad", decision)

    def test_approved_m4_does_not_imply_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with synthetic_build(root) as (inputs, decision, *_):
                value = json.loads(decision.read_text())
                value["trainingAuthorized"] = True
                decision.write_bytes(m4.json_bytes(value))
                with self.assertRaisesRegex(ValueError, "independent rights"):
                    m4.build(inputs, root / "bad", decision)
            self.assertFalse((root / "bad").exists())

    def test_quarantined_source_is_excluded_and_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record_id = m4.sha(b"extra.js")
            hold = {"sourceId": "microsoft-web-dev", "path": "extra.js",
                    "recordId": record_id}
            with synthetic_build(root, two_train=True, holds={
                "proposedOmissions": [hold], "noticeTriageRecords": [hold]
            }) as (inputs, decision, *_):
                m4.build(inputs, root / "candidate", decision)
            lines = (root / "candidate/train.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["path"], "main.js")
            skipped = json.loads((root / "candidate/skipped-audit.json").read_text())["skipped"]
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]["reasons"],
                             ["embedded-library-hold", "unclassified-notice"])

    def test_budget_refuses_partial_publishing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with synthetic_build(root, budget=40) as (inputs, decision, *_):
                with self.assertRaisesRegex(ValueError, "16 MiB"):
                    m4.build(inputs, root / "bad", decision)
            self.assertFalse((root / "bad").exists())

    def test_changed_output_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with synthetic_build(root) as (inputs, decision, *_):
                m4.build(inputs, root / "candidate", decision)
            (root / "candidate/train.jsonl").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                m4.verify_output(root / "candidate")

    def test_candidate_secret_is_skipped(self):
        row = record("microsoft-web-dev", "microsoft-web-dev-for-beginners",
                     "credentials.js", 'const x = "-----BEGIN PRIVATE KEY-----";\n')
        row["_historicalSplit"] = "train"
        output, audit, _ = m4.select_candidate(
            [row], {"proposedOmissions": [], "noticeTriageRecords": []},
            {"candidateTrainRecords": 1, "trainQuarantined": 0},
            {"maxCandidateRecords": 3000})
        self.assertEqual(output, [])
        self.assertIn("secret-pattern", audit["skipped"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
