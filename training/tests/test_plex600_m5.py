"""Synthetic M5-to-600M-tokenizer pipeline tests; no source downloads or GPU."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from plex_training import plex600_m5 as m5
from plex_training import plex600_m4 as m4
from plex_training import plex600_m3_scope as m3
from plex_training.plex600_setup import check_source_binding, check_tokenizer_bundle
from plex_training.tokenizer import train_tokenizer


class Plex600M5Tests(unittest.TestCase):
    def fixture(self, root: Path):
        candidate = root / "candidate"
        (candidate / "licenses").mkdir(parents=True)
        items = []
        decision_sources = []
        licenses = {}
        for source_id, (family, role) in sorted(m3.ROLES.items()):
            if role != "train":
                continue
            decision_sources.append({
                "sourceId": source_id, "rightsStatus": "approved",
                "reviewer": "test-only", "evidence": "synthetic fixture",
                "approvedScope": "test data only"})
            for index in range(3):
                code = (f"// {source_id} / {index}\n"
                        f"function a_{index}(value) {{ return value + {index}; }}\n")
                row = {"recordId": hashlib.sha256(f"{source_id}/{index}".encode()).hexdigest(),
                       "sourceId": source_id, "path": f"folder{index}/sample{index}.js",
                       "groupId": family, "text": code,
                       "contentSha256": m5._sha(code.encode())}
                items.append(row)
            notice = (source_id + " synthetic test license\n").encode()
            relative = "licenses/" + source_id + ".txt"
            (candidate / relative).write_bytes(notice)
            licenses[relative] = m5._sha(notice)
        scope_sha = m5._sha(m4.read_bounded(
            m3.PRE / "plex600-m3-candidate-scope-v1.json", m4.MAX_METADATA))
        decision = {"kind": "plex600-m4-rights-and-preparation-v1",
                    "m4PreparationApproved": True, "trainingAuthorized": False,
                    "finalHoldoutAccessed": False, "sourceReviewSha256": m3.REVIEW[1],
                    "holdsSha256": m3.HOLDS[1], "scopeSha256": scope_sha,
                    "sources": decision_sources}
        decision_path = root / "rights.json"
        decision_raw = m5._json(decision)
        decision_path.write_bytes(decision_raw)
        train_raw = b"".join(m5._json(row) for row in items)
        (candidate / "train.jsonl").write_bytes(train_raw)
        audit_raw = m5._json({"summary": {"skippedTrainRecords": 0}, "skipped": []})
        (candidate / "skipped-audit.json").write_bytes(audit_raw)
        manifest = {"pipelineVersion": "plex600-m4-candidate-v1",
                    "status": "review-only-not-ready-for-tokenizer",
                    "sourceReviewSha256": m3.REVIEW[1],
                    "holdsSha256": m3.HOLDS[1], "scopeSha256": scope_sha,
                    "rightsDecisionSha256": m5._sha(decision_raw),
                    "noticeSha256": licenses, "trainRecords": len(items),
                    "trainBytes": len(train_raw),
                    "trainJsonlSha256": m5._sha(train_raw),
                    "auditSha256": m5._sha(audit_raw),
                    "trainingAuthorized": False, "freshEvaluationCreated": False}
        (candidate / "manifest.json").write_bytes(m5._json(manifest))
        return candidate, decision_path, items

    def test_split_and_tokenizer_and_source_bind_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, decision, inputs = self.fixture(root)
            one = m5.build(candidate, root / "prepared1", decision)
            two = m5.build(candidate, root / "prepared2", decision)
            self.assertEqual(one["manifestSha256"], two["manifestSha256"])
            self.assertEqual(one["trainRecords"] + one["validationRecords"], len(inputs))
            self.assertGreater(one["sourceFamilies"], 1)
            for filename in ("train.jsonl", "validation.jsonl", "manifest.json"):
                self.assertEqual((root / "prepared1" / filename).read_bytes(),
                                 (root / "prepared2" / filename).read_bytes())
            train = [json.loads(line) for line in (root / "prepared1/train.jsonl").read_text().splitlines()]
            validation = [json.loads(line) for line in (root / "prepared1/validation.jsonl").read_text().splitlines()]
            self.assertFalse({r["contentSha256"] for r in train} & {r["contentSha256"] for r in validation})
            self.assertFalse({(r["sourceFamilyId"], r["splitGroupId"]) for r in train} &
                             {(r["sourceFamilyId"], r["splitGroupId"]) for r in validation})
            self.assertEqual({r["sourceFamilyId"] for r in train},
                             {r["sourceFamilyId"] for r in validation})
            dataset = root / "prepared1"
            output = root / "bundle"
            fit = train_tokenizer(dataset, output, vocab_size=300, model_family="plex-600m-v1")
            self.assertGreater(fit["train"]["tokens"] if "tokens" in fit["train"] else fit["train"]["tokenCount"], 0)
            self.assertTrue(check_tokenizer_bundle(output)["verifiedSplits"])
            packed_manifest_hash = m5._sha((output / "manifest.json").read_bytes())
            source_sha = m5._sha((dataset / "manifest.json").read_bytes())
            self.assertTrue(check_source_binding(output, dataset,
                bundle_manifest_sha256=packed_manifest_hash,
                source_manifest_sha256=source_sha)["sourceBindingVerified"])

    def test_modified_candidate_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, decision, _ = self.fixture(root)
            (candidate / "train.jsonl").write_bytes(b'{"tampered":true}\n')
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                m5.build(candidate, root / "bad", decision)
            self.assertFalse((root / "bad").exists())

    def test_rights_decision_must_match_m4_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, decision, _ = self.fixture(root)
            raw = json.loads(decision.read_text())
            raw["sources"][0]["rightsStatus"] = "pending"
            decision.write_bytes(m5._json(raw))
            with self.assertRaises(ValueError):
                m5.build(candidate, root / "bad", decision)
            self.assertFalse((root / "bad").exists())

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, decision, _ = self.fixture(root)
            target = root / "prepared"
            target.mkdir()
            (target / "important.txt").write_text("keep")
            with self.assertRaisesRegex(ValueError, "new and separate"):
                m5.build(candidate, target, decision)
            self.assertEqual((target / "important.txt").read_text(), "keep")

    def test_family_must_have_multiple_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, rows = self.fixture(root)
            only = [r for r in rows if r["groupId"] == "microsoft-web-dev-for-beginners"]
            with self.assertRaisesRegex(ValueError, "two source families"):
                m5._split(only, 1337, 10)

    def test_existing_family_group_never_crosses_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, rows = self.fixture(root)
            mdn = [r for r in rows if r["sourceId"].startswith("mdn-")]
            self.assertGreaterEqual(len(mdn), 2)
            # Same historical project name from two MDN sources must remain
            # inseparable despite different source IDs.
            mdn[0]["splitGroupId"] = "shared-project"
            mdn[1]["splitGroupId"] = "shared-project"
            split, _ = m5._split(rows, 1337, 20)
            self.assertEqual(
                len({kind for kind, records in split.items()
                     for r in records if r["splitGroupId"].endswith(":shared-project")}), 1)

    def test_bad_seed_and_validation_percent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, rows = self.fixture(root)
            for seed, pct in ((True, 10), (-1, 10), (1337, 0), (1337, 80)):
                with self.subTest(seed=seed, pct=pct), self.assertRaises(ValueError):
                    m5._split(rows, seed, pct)

    def test_zero_split_groups_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, rows = self.fixture(root)
            family = rows[0]["groupId"]
            for row in rows:
                if row["groupId"] == family:
                    row["splitGroupId"] = "one-original-project"
            with self.assertRaisesRegex(ValueError, "cannot be present in both"):
                m5._split(rows, 1337, 10)


if __name__ == "__main__":
    unittest.main()
