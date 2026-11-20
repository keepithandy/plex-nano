"""Read-only Phase 2 development scorer. Never loads model weights or trains."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent / "fixtures" / "plex_phase2_requests_v1.json"


def editor_module():
    spec = importlib.util.spec_from_file_location("plex_phase2_console", HERE / "PlexLocal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cases(path=SUITE):
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    cases = data["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)) or not cases:
        raise ValueError("Development cases must have unique IDs.")
    for item in cases:
        if ("expected" in item) == ("refusalContains" in item):
            raise ValueError("Each case must declare exactly one expected edit or refusal.")
    return cases, hashlib.sha256(raw).hexdigest()


def score_editor(cases, module):
    correct_edits = correct_refusals = 0
    edit_total = refusal_total = 0
    failures = []
    for item in cases:
        is_refusal = "refusalContains" in item
        if is_refusal:
            refusal_total += 1
        else:
            edit_total += 1
        try:
            plan = module.phase2_plan(item["source"], item["request"])
            if plan is None:
                result = "would route to model instead of a verified edit/refusal"
            elif is_refusal:
                result = "accepted a request that must be refused"
            else:
                _, changes = plan
                edited = module.apply_changes(item["source"], changes)
                if edited == item["expected"]:
                    correct_edits += 1
                    continue
                result = "edited file did not match expected bytes"
        except ValueError as exc:
            if is_refusal and item["refusalContains"] in str(exc):
                correct_refusals += 1
                continue
            result = "wrong or unexpected refusal: " + str(exc)
        failures.append((item["id"], result))
    return {
        "edits": [correct_edits, edit_total],
        "refusals": [correct_refusals, refusal_total],
        "failures": failures,
    }


def score_model(cases, path):
    """Score owner-supplied raw generations, not built-in editor output.

    Every case must have one unmodified raw result with id, text, stop and
    elapsedMs. REFUSE: is an explicit non-code refusal protocol.
    Empty EOS is never counted as a correct refusal.
    """
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in rows:
            raise ValueError(f"Duplicate model output: {row['id']}")
        if not isinstance(row.get("text"), str) or not isinstance(row.get("stop"), str):
            raise ValueError("Every model result needs text and stop strings.")
        rows[row["id"]] = row
    expected_ids = {item["id"] for item in cases}
    if set(rows) != expected_ids:
        raise ValueError("Model result IDs must match the complete frozen development set.")
    exact = refusals = empty_eos = unintended = 0
    edit_count = sum("expected" in item for item in cases)
    times = []
    for item in cases:
        result = rows[item["id"]]
        text, stop = result["text"], result["stop"]
        if text.strip() == "" and stop == "eos":
            empty_eos += 1
        ms = result.get("elapsedMs")
        if isinstance(ms, (float, int)) and not isinstance(ms, bool) and 0 <= ms < float("inf"):
            times.append(ms)
        if "expected" in item:
            if text == item["expected"] and stop == "eos":
                exact += 1
            elif text and text != item["source"]:
                unintended += 1
        elif stop == "eos" and text.startswith("REFUSE:") and "\n" not in text.strip():
            refusals += 1
    return {
        "edits": [exact, edit_count],
        "refusals": [refusals, len(cases) - edit_count],
        "emptyEos": empty_eos,
        "incorrectNonemptyEdits": unintended,
        "avgMs": round(sum(times) / len(times), 1) if len(times) == len(cases) else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Score Phase 2 built-in and separately supplied raw model outputs.")
    parser.add_argument("--baseline-model-jsonl", type=Path)
    parser.add_argument("--alternative-model-jsonl", type=Path)
    args = parser.parse_args()
    cases, digest = load_cases()
    print(f"Phase 2 development: {len(cases)} fixed cases (SHA-256 {digest[:12]})")
    editor = score_editor(cases, editor_module())
    print(f"Built-in: {editor['edits'][0]}/{editor['edits'][1]} exact edits; "
          f"{editor['refusals'][0]}/{editor['refusals'][1]} correct refusals.")
    for name, path in (("Raw 600M baseline", args.baseline_model_jsonl),
                       ("Raw 600M alternative", args.alternative_model_jsonl)):
        if path is None:
            print(f"{name}: not run (no model results provided).")
            continue
        score = score_model(cases, path)
        print(f"{name}: {score['edits'][0]}/{score['edits'][1]} exact; "
              f"{score['refusals'][0]}/{score['refusals'][1]} explicit refusals; "
              f"empty EOS {score['emptyEos']}; other incorrect nonempty edits "
              f"{score['incorrectNonemptyEdits']}; mean latency "
              f"{score['avgMs'] if score['avgMs'] is not None else 'unavailable'} ms.")
    for case_id, reason in editor["failures"]:
        print(f"FAIL {case_id}: {reason}")
    if editor["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
