"""Review-only tests for the 600M scratch tokenizer and safe inspection CLI."""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from plex_training.plex600_setup import (
    MODEL_CAPACITY,
    MODEL_FAMILY,
    check_tokenizer_bundle,
    inspect,
    main,
)
from plex_training.tokenizer import PlexTokenizer, ROUNDTRIP_SAMPLES, train_tokenizer


class Plex600SetupTests(unittest.TestCase):
    @staticmethod
    def dataset(root: Path, *, validation: str = "Reserved example unseen by tokenizer 🚀\n") -> Path:
        dataset = root / "dataset"
        dataset.mkdir(exist_ok=True)
        (dataset / "licenses").mkdir(exist_ok=True)
        notice = b"Local test fixture, approved by its author.\n"
        (dataset / "licenses/notice.txt").write_bytes(notice)
        sources = []
        summary = {}
        for split, body in (
            ("train", "function add(left, right) { return left + right; }\n" * 20),
            ("validation", validation),
        ):
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            row = {
                "recordId": split, "sourceId": split, "groupId": split,
                "contentSha256": digest, "text": body, "path": "fixture.js",
            }
            raw = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            (dataset / (split + ".jsonl")).write_bytes(raw)
            summary[split + "Records"] = 1
            summary[split + "JsonlSha256"] = hashlib.sha256(raw).hexdigest()
            sources.append({
                "id": split, "groupId": split, "rightsReviewStatus": "approved",
                "licenseNoticeFile": "licenses/notice.txt",
                "licenseNoticeSha256": hashlib.sha256(notice).hexdigest(),
            })
        (dataset / "manifest.json").write_text(
            json.dumps({"schemaVersion": 1, "pipelineVersion": "p1-14.2",
                        "sources": sources, "summary": summary}), encoding="utf-8"
        )
        return dataset

    def test_inspection_does_not_instantiate_full_model(self) -> None:
        report = inspect()
        self.assertEqual(report["parameterCount"], 600_372_480)
        self.assertEqual(report["modelFamily"], MODEL_FAMILY)
        self.assertFalse(report["initializedWeights"])
        self.assertFalse(report["trainingAuthorized"])
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(main(["inspect"]), 0)
        self.assertEqual(json.loads(output.getvalue()), report)

    def test_fit_reuses_verified_dataset_and_writes_600m_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self.dataset(root)
            result = train_tokenizer(dataset, root / "bundle", vocab_size=320,
                                     model_family=MODEL_FAMILY)
            self.assertEqual(result["modelVocabularyCapacity"], MODEL_CAPACITY)
            settings = json.loads((root / "bundle/tokenizer-config.json").read_text())
            self.assertEqual(settings["modelFamily"], MODEL_FAMILY)
            self.assertEqual(settings["modelVocabularyCapacity"], MODEL_CAPACITY)
            report = check_tokenizer_bundle(root / "bundle")
            self.assertEqual(report["actualVocabularySize"], result["actualVocabularySize"])
            self.assertEqual(report["verifiedSplits"]["train"]["records"], 1)
            self.assertEqual(report["verifiedSplits"]["validation"]["records"], 1)
            self.assertEqual(report["optimizerUpdates"], 0)
            tokenizer = PlexTokenizer.load(root / "bundle")
            for sample in ROUNDTRIP_SAMPLES:
                self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_cli_fit_check_and_refusal_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self.dataset(root)
            args = ["tokenizer-train", "--dataset-dir", str(dataset),
                    "--output-dir", str(root / "bundle"), "--vocab-size", "310"]
            with redirect_stdout(StringIO()) as output:
                self.assertEqual(main(args), 0)
            self.assertIn("verifiedBundle", json.loads(output.getvalue()))
            with redirect_stdout(StringIO()) as output:
                self.assertEqual(main(["tokenizer-check", "--bundle-dir", str(root / "bundle")]), 0)
            self.assertEqual(json.loads(output.getvalue())["modelFamily"], MODEL_FAMILY)
            with redirect_stderr(StringIO()) as error:
                self.assertEqual(main(args), 2)
            self.assertIn("overwrite", error.getvalue())

    def test_nano_defaults_are_not_relabelled_as_600m(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self.dataset(root)
            nano = train_tokenizer(dataset, root / "nano", vocab_size=300)
            self.assertEqual(nano["modelVocabularyCapacity"], 16_384)
            with self.assertRaisesRegex(ValueError, "not configured for Plex 600M"):
                check_tokenizer_bundle(root / "nano")
            with self.assertRaisesRegex(ValueError, "capacity"):
                train_tokenizer(dataset, root / "wrong", vocab_size=32_768)
            self.assertFalse((root / "wrong").exists())
            with self.assertRaisesRegex(ValueError, "Unrecognized model family"):
                train_tokenizer(dataset, root / "wrong", model_family="untrusted")

    def test_hash_tampering_and_index_framing_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self.dataset(root)
            train_tokenizer(dataset, root / "bundle", vocab_size=300,
                            model_family=MODEL_FAMILY)
            path = root / "bundle"
            indexed = path / "train.index.json"
            original = indexed.read_bytes()
            rows = json.loads(original)
            rows[0]["startToken"] = 1
            indexed.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "contiguous"):
                check_tokenizer_bundle(path)
            indexed.write_bytes(original)
            self.assertTrue(check_tokenizer_bundle(path)["verifiedSplits"])
            packed = path / "train.tokens.u16le"
            packed.write_bytes(packed.read_bytes() + b"\x00\x00")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                check_tokenizer_bundle(path)

    def test_validation_text_does_not_affect_tokenizer_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self.dataset(root)
            first = train_tokenizer(dataset, root / "first", vocab_size=320,
                                    model_family=MODEL_FAMILY)
            self.dataset(root, validation="Different unseen validation text 🦕Ω\n")
            second = train_tokenizer(dataset, root / "second", vocab_size=320,
                                     model_family=MODEL_FAMILY)
            self.assertEqual(first["tokenizerSha256"], second["tokenizerSha256"])
            self.assertNotEqual(first["validation"]["sha256"], second["validation"]["sha256"])
            check_tokenizer_bundle(root / "first")
            check_tokenizer_bundle(root / "second")


if __name__ == "__main__":
    unittest.main()
