"""Read-only final M11 evidence audit. No model loading, training or file writes.

Consumes the TWO existing local experiment directories, verifies hashes and run
identity, and produces a concise user-facing decision. Never reads a final holdout.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures/plex_phase2_requests_v1.json"
WEIGHTS_SHA = "d7a7776de86230320f6e0e48b07dc76761c3d42ac80e4c330839e47c9ffc93dd"
TOKENIZER_SHA = "9feef87bdf5fc28b3315df589e4e0b8ffcf5f40b18a4c357c91759182340a2e2"
EOS_PROTOCOL = "plex-phase2-m11-eos-diagnostic-v1"
EDIT_PROTOCOL = "plex-phase2-m11-inference-v1"
TEMPLATES = ("m11-trained", "m11-newline", "console-general")
EXPECTED_LINES = {
    "m11-train-score-0": ("train", "let score = 0;\n"),
    "m11-train-score-1": ("train", "let score = 1;\n"),
    "m11-probe-only-value-0": ("probe-only", "let value = 0;\n"),
    "m11-probe-only-value-1": ("probe-only", "let value = 1;\n"),
}
STOP_REASONS = {"eos", "max-new-tokens", "invalid-control-token", "outside-fitted-tokenizer"}


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"Invalid JSON object: {path.name}")
    return value


def read_rows(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if any(type(row) is not dict for row in rows):
        raise ValueError("Expected JSON objects in " + path.name)
    return rows, hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_eos(directory):
    receipt = read_json(directory / "report.json")
    rows, raw_sha = read_rows(directory / "probes.jsonl")
    for key, expected in {
        "protocol": EOS_PROTOCOL, "checkpointSha256": WEIGHTS_SHA,
        "tokenizerSha256": TOKENIZER_SHA, "cases": 4,
        "generations": 12, "templates": list(TEMPLATES),
        "trainingPerformed": False, "finalHoldoutAccessed": False,
        "policy": "fitted-vocab-v1", "maxNewTokens": 96,
        "probeResponsesSha256": raw_sha,
    }.items():
        require(type(receipt.get(key)) is type(expected) and receipt[key] == expected,
                "EOS evidence receipt mismatch: " + key)
    require(len(rows) == 12, "EOS probe has missing or extra rows")
    seen = set()
    summaries = {}
    for name in TEMPLATES:
        summaries[name] = {"exact": 0, "trainExact": 0, "probeExact": 0,
                           "immediateEos": 0, "eosFirst": 0, "total": 0}
    for row in rows:
        case = EXPECTED_LINES.get(row.get("id"))
        template = row.get("template")
        require(case is not None and template in TEMPLATES, "Unknown M11 case or prompt")
        require((row["id"], template) not in seen, "Repeated EOS probe case")
        seen.add((row["id"], template))
        split, expected_line = case
        require(row.get("split") == split and row.get("checkpointSha256") == WEIGHTS_SHA,
                "EOS row identity mismatch")
        require(type(row.get("output")) is str and row.get("stop") in STOP_REASONS
                and type(row.get("generatedTokens")) is int
                and 0 <= row["generatedTokens"] <= 96,
                "EOS row generation metadata invalid")
        exact = row["output"] == expected_line and row["stop"] == "eos"
        require(type(row.get("exactAndEos")) is bool and row["exactAndEos"] == exact,
                "EOS exactness flag does not match generated code")
        metrics = row.get("firstToken")
        require(type(metrics) is dict, "EOS first-token measurements missing")
        selected = metrics.get("fittedFirstTokenId")
        rank = metrics.get("eosFittedRank")
        require(type(selected) is int and 3 <= selected < 1191
                and type(rank) is int and 1 <= rank < 1191,
                "Invalid first-token EOS rank")
        for key in ("eosMinusBestCodeLogit", "goldMinusEosLogit"):
            require(type(metrics.get(key)) in (int, float)
                    and math.isfinite(metrics[key]), "Invalid EOS margin")
        immediate = row["stop"] == "eos" and row["generatedTokens"] == 0
        require(immediate == (selected == 3), "EOS first token conflicts with greedy output")
        group = summaries[template]
        group["total"] += 1
        group["exact"] += int(exact)
        group["trainExact"] += int(exact and split == "train")
        group["probeExact"] += int(exact and split == "probe-only")
        group["immediateEos"] += int(immediate)
        group["eosFirst"] += int(selected == 3)
    require(len(seen) == 12 and all(group["total"] == 4 for group in summaries.values()),
            "Incomplete M11 prompt comparison")
    for name, group in summaries.items():
        published = receipt.get("summary", {}).get(name, {})
        for key, pub_key in (
            ("exact", "exactAndEos"), ("trainExact", "trainingExact"),
            ("probeExact", "probeExact"), ("immediateEos", "immediateEos"),
            ("eosFirst", "firstTokenFavorsEos"),
        ):
            require(type(published.get(pub_key)) is int
                    and published[pub_key] == group[key],
                    "EOS report summary does not match raw data: " + name)
    return summaries


def audit_edits(directory, fixture=FIXTURE):
    cases = json.loads(fixture.read_text(encoding="utf-8"))["cases"]
    require(len(cases) == 33 and len({c["id"] for c in cases}) == 33,
            "Changed Phase 2 development suite")
    receipt = read_json(directory / "identity.json")
    expected = {
        "protocol": EDIT_PROTOCOL,
        "checkpointSha256": WEIGHTS_SHA,
        "tokenizerSha256": TOKENIZER_SHA,
        "suiteSha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "casesPerPrompt": 33, "trainingPerformed": False,
        "policy": "fitted-vocab-v1", "maxNewTokens": 96,
        "prompts": ["baseline", "alternative"],
    }
    for key, value in expected.items():
        require(type(receipt.get(key)) is type(value) and receipt[key] == value,
                "Phase 2 raw-eval receipt mismatch: " + key)
    hashes = receipt.get("responseSha256")
    require(type(hashes) is dict and set(hashes) == {"baseline", "alternative"},
            "Missing paired model file hashes")
    by_id = {case["id"]: case for case in cases}
    groups = {}
    for name in ("baseline", "alternative"):
        rows, sha = read_rows(directory / (name + ".jsonl"))
        require(hashes[name] == sha and len(rows) == 33, "Altered or partial " + name + " outputs")
        require({r.get("id") for r in rows} == set(by_id), "Missing or duplicate model case IDs")
        result = {"exactEdits": 0, "totalEdits": 0, "refusals": 0, "totalRefusals": 0, "emptyEos": 0}
        for row in rows:
            case = by_id[row["id"]]
            require(row.get("checkpointSha256") == WEIGHTS_SHA
                    and type(row.get("text")) is str
                    and row.get("stop") in STOP_REASONS
                    and type(row.get("generatedTokens")) is int
                    and 0 <= row["generatedTokens"] <= 96,
                    "Invalid raw M11 generation metadata")
            text = row["text"]
            result["emptyEos"] += int(text == "" and row["stop"] == "eos")
            if "expected" in case:
                result["totalEdits"] += 1
                result["exactEdits"] += int(text == case["expected"] and row["stop"] == "eos")
            else:
                result["totalRefusals"] += 1
                result["refusals"] += int(
                    row["stop"] == "eos" and text.startswith("REFUSE:")
                    and "\n" not in text.strip()
                )
        groups[name] = result
    return groups


def print_report(eos, edits):
    print("PLEX M11 - FINAL EVIDENCE CHECK (READ ONLY)")
    for label, key in (("Original M11 prompt", "m11-trained"),
                       ("Changed delimiter", "m11-newline"),
                       ("General console prompt", "console-general")):
        item = eos[key]
        print(f"{label}: {item['exact']}/4 exact; {item['immediateEos']}/4 immediate EOS.")
    print(f"Familiar train: {eos['m11-trained']['trainExact']}/2; "
          f"unseen development: {eos['m11-trained']['probeExact']}/2.")
    for name in ("baseline", "alternative"):
        result = edits[name]
        print(f"Phase 2 {name}: {result['exactEdits']}/{result['totalEdits']} model edits; "
              f"{result['refusals']}/{result['totalRefusals']} refusals; "
              f"{result['emptyEos']}/33 empty EOS.")
    print("Decision: M11 has NOT demonstrated unseen copying or working code edits.")
    print("Next: review M12 data/evaluation design. Training remains paused.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eos-dir", type=Path, required=True)
    parser.add_argument("--phase2-dir", type=Path, required=True)
    args = parser.parse_args()
    eos = audit_eos(args.eos_dir)
    edits = audit_edits(args.phase2_dir)
    print_report(eos, edits)


if __name__ == "__main__":
    main()
