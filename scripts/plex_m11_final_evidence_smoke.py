"""CPU-only M11 final evidence audit tests: no torch, model weights, or training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "plex_m11_final_evidence.py"
spec = importlib.util.spec_from_file_location("plex_m11_final_evidence", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class FinalM11EvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="plex-m11-final-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.eos = self.root / "eos"
        self.edits = self.root / "edit"
        self.eos.mkdir()
        self.edits.mkdir()

    def make_eos(self):
        rows = []
        summary = {}
        for name in audit.TEMPLATES:
            summary[name] = {
                "exactAndEos": 0, "trainingExact": 0, "probeExact": 0,
                "immediateEos": 0, "firstTokenFavorsEos": 0,
            }
            for case_id, (split, expected) in audit.EXPECTED_LINES.items():
                is_copy = name == "m11-trained"
                actual = expected if split == "train" and is_copy else (
                    expected.replace("value", "score") if is_copy else ""
                )
                exact = actual == expected
                immediate = not is_copy
                row = {
                    "id": case_id, "split": split, "template": name,
                    "output": actual, "stop": "eos",
                    "checkpointSha256": audit.WEIGHTS_SHA,
                    "generatedTokens": 0 if immediate else 6,
                    "exactAndEos": exact,
                    "firstToken": {
                        "fittedFirstTokenId": 3 if immediate else 4,
                        "eosFittedRank": 1 if immediate else 2,
                        "eosMinusBestCodeLogit": 1.2 if immediate else -1.2,
                        "goldMinusEosLogit": -1.2 if immediate else 1.2,
                    },
                }
                rows.append(row)
                summary[name]["exactAndEos"] += int(exact)
                summary[name]["trainingExact"] += int(exact and split == "train")
                summary[name]["probeExact"] += int(exact and split == "probe-only")
                summary[name]["immediateEos"] += int(immediate)
                summary[name]["firstTokenFavorsEos"] += int(immediate)
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
        (self.eos / "probes.jsonl").write_bytes(raw)
        receipt = {
            "protocol": audit.EOS_PROTOCOL, "checkpointSha256": audit.WEIGHTS_SHA,
            "tokenizerSha256": audit.TOKENIZER_SHA,
            "cases": 4, "templates": list(audit.TEMPLATES),
            "generations": 12, "maxNewTokens": 96,
            "trainingPerformed": False, "finalHoldoutAccessed": False,
            "policy": "fitted-vocab-v1",
            "probeResponsesSha256": hashlib.sha256(raw).hexdigest(),
            "summary": summary,
        }
        (self.eos / "report.json").write_text(json.dumps(receipt), encoding="utf-8")
        return receipt

    def make_edits(self):
        cases = json.loads(audit.FIXTURE.read_text(encoding="utf-8"))["cases"]
        sha = {}
        for name in ("baseline", "alternative"):
            raw = "".join(json.dumps({
                "id": case["id"], "text": "", "stop": "eos",
                "generatedTokens": 0, "checkpointSha256": audit.WEIGHTS_SHA,
            }) + "\n" for case in cases).encode("utf-8")
            (self.edits / (name + ".jsonl")).write_bytes(raw)
            sha[name] = hashlib.sha256(raw).hexdigest()
        receipt = {
            "protocol": audit.EDIT_PROTOCOL,
            "checkpointSha256": audit.WEIGHTS_SHA,
            "tokenizerSha256": audit.TOKENIZER_SHA,
            "suiteSha256": hashlib.sha256(audit.FIXTURE.read_bytes()).hexdigest(),
            "casesPerPrompt": 33, "trainingPerformed": False,
            "policy": "fitted-vocab-v1", "maxNewTokens": 96,
            "prompts": ["baseline", "alternative"], "responseSha256": sha,
        }
        (self.edits / "identity.json").write_text(json.dumps(receipt), encoding="utf-8")

    def test_reproduces_user_facing_results_without_a_model(self):
        self.make_eos()
        self.make_edits()
        eos = audit.audit_eos(self.eos)
        edits = audit.audit_edits(self.edits)
        self.assertEqual(eos["m11-trained"]["trainExact"], 2)
        self.assertEqual(eos["m11-trained"]["probeExact"], 0)
        self.assertEqual(eos["m11-newline"]["immediateEos"], 4)
        self.assertEqual(eos["console-general"]["immediateEos"], 4)
        self.assertEqual(eos["m11-trained"]["meanEosMinusBestCodeLogit"], -1.2)
        self.assertEqual(eos["console-general"]["meanEosMinusBestCodeLogit"], 1.2)
        self.assertEqual(edits["baseline"]["exactEdits"], 0)
        self.assertEqual(edits["baseline"]["emptyEos"], 33)
        result = subprocess.run([
            sys.executable, "-I", "-S", str(SCRIPT),
            "--eos-dir", str(self.eos), "--phase2-dir", str(self.edits),
        ], capture_output=True, text=True, timeout=20, check=True)
        self.assertIn("Familiar train: 2/2; unseen development: 0/2.", result.stdout)
        self.assertIn("Mean logit gaps: EOS minus best code", result.stdout)
        self.assertIn("0/20 model edits", result.stdout)
        self.assertIn("Training remains paused.", result.stdout)

    def test_hash_mismatch_refuses(self):
        self.make_eos()
        self.make_edits()
        with (self.eos / "probes.jsonl").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError, "probeResponsesSha256"):
            audit.audit_eos(self.eos)
        with (self.edits / "baseline.jsonl").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError, "Altered or partial"):
            audit.audit_edits(self.edits)

    def test_wrong_model_and_training_metadata_refuse(self):
        receipt = self.make_eos()
        self.make_edits()
        receipt["trainingPerformed"] = True
        (self.eos / "report.json").write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "trainingPerformed"):
            audit.audit_eos(self.eos)
        receipt["trainingPerformed"] = False
        receipt["checkpointSha256"] = "0" * 64
        (self.eos / "report.json").write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checkpointSha256"):
            audit.audit_eos(self.eos)
        identity = json.loads((self.edits / "identity.json").read_text(encoding="utf-8"))
        identity["trainingPerformed"] = True
        (self.edits / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "trainingPerformed"):
            audit.audit_edits(self.edits)

    def test_misreported_result_and_first_token_refuse(self):
        receipt = self.make_eos()
        path = self.eos / "probes.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["exactAndEos"] = False
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
        path.write_bytes(raw)
        receipt["probeResponsesSha256"] = hashlib.sha256(raw).hexdigest()
        (self.eos / "report.json").write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactness flag"):
            audit.audit_eos(self.eos)
        rows[0]["exactAndEos"] = True
        rows[0]["firstToken"]["fittedFirstTokenId"] = 3
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
        path.write_bytes(raw)
        receipt["probeResponsesSha256"] = hashlib.sha256(raw).hexdigest()
        (self.eos / "report.json").write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "EOS first token"):
            audit.audit_eos(self.eos)


if __name__ == "__main__":
    unittest.main()
