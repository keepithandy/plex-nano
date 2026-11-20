"""Diagnose immediate-EOS behavior of the pinned Plex M11 checkpoint.

Inference-only. The original decoder is reused unchanged, and the first token
logits are inspected using exactly its fitted-vocab BF16 forward path.
No training, optimizer, model checkpoint write, or final holdout access.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "training/artifacts"
PROTOCOL = "plex-phase2-m11-eos-diagnostic-v1"
EXPECTED_TOKENIZER_SHA256 = "9feef87bdf5fc28b3315df589e4e0b8ffcf5f40b18a4c357c91759182340a2e2"
MAX_NEW_TOKENS = 96
TEMPLATES = ("m11-trained", "m11-newline", "console-general")


def code_console():
    script = Path(__file__).with_name("PlexLocal.py")
    spec = importlib.util.spec_from_file_location("plex_eos_console", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_prompts(cases, historical_prompt, console_prompt):
    """One frozen set of four familiar lines, compared under three prompts."""
    planned = []
    for case in cases:
        source = case["snippet"]
        trained = historical_prompt(case)
        if not trained.endswith("Copy: "):
            raise ValueError("Frozen M11 training prompt must end in 'Copy: '")
        variations = {
            "m11-trained": trained,
            "m11-newline": trained[:-1] + "\n",
            "console-general": console_prompt({
                "request": "Return the supplied JavaScript line exactly as written.",
                "inputFile": source,
            }),
        }
        for template in TEMPLATES:
            planned.append({
                "id": case["id"], "split": case["split"],
                "template": template, "prompt": variations[template],
                "expected": source,
            })
    return planned


def first_token_metrics(logits, vocab_size, expected_token_id):
    """Pure scoring logic, independently testable without torch/CUDA.

    EOS is token 3; fitted-vocab greedy decoding picks from IDs 3..V-1.
    """
    if (type(vocab_size) is not int or vocab_size < 5
            or type(expected_token_id) is not int
            or not 4 <= expected_token_id < vocab_size
            or len(logits) < vocab_size
            or not all(math.isfinite(float(x)) for x in logits)):
        raise ValueError("Invalid next-token logits, expected token, or fitted vocabulary.")
    all_best = max(range(len(logits)), key=lambda t: float(logits[t]))
    fitted_best = max(range(3, vocab_size), key=lambda t: float(logits[t]))
    non_eos = max(range(4, vocab_size), key=lambda t: float(logits[t]))
    eos = float(logits[3])
    return {
        "fittedFirstTokenId": fitted_best,
        "unmaskedFirstTokenId": all_best,
        "eosFittedRank": 1 + sum(float(logits[t]) > eos for t in range(4, vocab_size)),
        "eosMinusBestCodeLogit": round(eos - float(logits[non_eos]), 4),
        "goldMinusEosLogit": round(float(logits[expected_token_id]) - eos, 4),
        "bestCodeTokenId": non_eos,
    }


def summarize(rows):
    """Short, user-facing outcomes; evidence rows retain all actual details."""
    expected = len(TEMPLATES) * 4
    if len(rows) != expected or len({(r["id"], r["template"]) for r in rows}) != expected:
        raise ValueError("EOS diagnostic must contain all 12 unique, scored probes.")
    summary = {}
    for name in TEMPLATES:
        group = [r for r in rows if r["template"] == name]
        summary[name] = {
            "exactAndEos": sum(r["exactAndEos"] for r in group),
            "total": len(group),
            "trainingExact": sum(r["exactAndEos"] for r in group if r["split"] == "train"),
            "trainingTotal": sum(r["split"] == "train" for r in group),
            "probeExact": sum(r["exactAndEos"] for r in group if r["split"] == "probe-only"),
            "probeTotal": sum(r["split"] == "probe-only" for r in group),
            "immediateEos": sum(r["stop"] == "eos" and r["generatedTokens"] == 0 for r in group),
            "firstTokenFavorsEos": sum(r["firstToken"]["fittedFirstTokenId"] == 3 for r in group),
            "stops": dict(Counter(r["stop"] for r in group)),
        }
    return summary


def report(summary):
    print("PLEX M11 - EOS DIAGNOSTIC")
    for name in TEMPLATES:
        row = summary[name]
        print(f"{name}: {row['exactAndEos']}/{row['total']} exact; "
              f"{row['immediateEos']}/{row['total']} immediate EOS.")
        if name == "m11-trained":
            print(f"  Familiar training: {row['trainingExact']}/{row['trainingTotal']}; "
                  f"development copies: {row['probeExact']}/{row['probeTotal']}.")
    first = summary["m11-trained"]
    general = summary["console-general"]
    if first["trainingExact"] == first["trainingTotal"] and general["immediateEos"] == general["total"]:
        print("Finding: M11 can copy familiar training lines under its original prompt, "
              "but the current console prompt triggers immediate EOS. "
              "This is a prompt-conditioning failure, not proof of general coding ability.")
    elif first["trainingExact"] < first["trainingTotal"]:
        print("Finding: familiar training prompts did not fully reproduce the recorded behavior. "
              "Check checkpoint/tokenizer/decoder parity before changing prompting or training.")
    else:
        print("Finding: inspect saved per-case logits and outputs. "
              "The result does not establish reliable editing.")
    print("Training: paused. Raw generations are not built-in editor successes.")


def output_destination(path):
    root = ARTIFACTS.resolve(strict=True)
    dest = path.expanduser().resolve()
    if (dest.exists() or path.is_symlink() or dest.parent != root
            or path.parent.is_symlink()):
        raise ValueError("Choose a NEW output directory directly under training/artifacts.")
    return dest, root


def save_results(dest, root, rows, summary, console, sha):
    """Create-only and staged output; do not overwrite previous experiments."""
    stage = Path(tempfile.mkdtemp(prefix=f".{dest.name}.staging-", dir=root))
    try:
        raw = "".join(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n" for item in rows).encode("utf-8")
        (stage / "probes.jsonl").write_bytes(raw)
        receipt = {
            "protocol": PROTOCOL, "checkpointSha256": console.DIGEST,
            "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
            "cases": 4, "templates": list(TEMPLATES), "generations": len(rows),
            "maxNewTokens": MAX_NEW_TOKENS, "policy": "fitted-vocab-v1",
            "trainingPerformed": False, "finalHoldoutAccessed": False,
            "probeResponsesSha256": hashlib.sha256(raw).hexdigest(),
            "m11FrozenCasePromptSha256": sha,
            "summary": summary,
        }
        (stage / "report.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if dest.exists():
            raise FileExistsError("Refusing to replace an existing EOS report.")
        os.rename(stage, dest)
        stage = None
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute-read-only", action="store_true")
    args = parser.parse_args()
    if not args.execute_read_only:
        parser.error("Add --execute-read-only for inference; no model is loaded by default.")
    dest, root = output_destination(args.output_dir)

    console = code_console()
    from plex_training import plex600_m11_source_conditioning as m11
    from plex_training.plex600_m8_eval import prompt_for as console_prompt
    from plex_training.tokenizer import PlexTokenizer

    cases = m11.cases()  # the four frozen M11 training/probe-only cases; NOT the holdout
    if len(cases) != 4 or len({c["id"] for c in cases}) != 4:
        raise ValueError("Unexpected M11 frozen development case identity.")
    planned = make_prompts(cases, m11.prompt_for, console_prompt)
    frozen_sha = hashlib.sha256(
        json.dumps(planned, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    bundle = console.ARTIFACTS / "plex600-fresh-tokenizer"
    cfg = json.loads((bundle / "tokenizer-config.json").read_text(encoding="utf-8"))
    if cfg.get("tokenizerSha256") != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("Wrong local M11 tokenizer; refusing the comparison.")
    tokenizer = PlexTokenizer.load(bundle)
    prepared = []
    for case in planned:
        ids = tokenizer.encode(case["prompt"])
        gold = tokenizer.encode(case["expected"])
        if (not ids or not gold or len(ids) + MAX_NEW_TOKENS > 1024
                or tokenizer.decode(ids) != case["prompt"]
                or tokenizer.decode(gold) != case["expected"]):
            raise ValueError(f"Input or gold token budget/roundtrip failed for {case['id']}")
        prepared.append((case, ids, gold[0]))

    import torch
    from plex_training.plex600_m8_eval import generate_one

    model = console.load_model()  # check exact step-200 SHA before use, no optimizer
    model.eval()
    rows = []
    for case, ids, gold_id in prepared:
        start = time.perf_counter()
        with torch.inference_mode():
            inputs = torch.tensor([ids], dtype=torch.long, device=torch.device("cuda"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(inputs)
            metrics = first_token_metrics(
                logits[0, -1, :].float().cpu().tolist(), tokenizer.vocabulary_size, gold_id
            )
        generated, count, stop = generate_one(
            model, torch, tokenizer, ids, device=torch.device("cuda"),
            max_new_tokens=MAX_NEW_TOKENS, decoding_policy="fitted-vocab-v1",
        )
        if (stop == "eos" and count == 0) != (metrics["fittedFirstTokenId"] == 3):
            raise ValueError("First-token logits do not match the frozen decoder's actual first prediction.")
        rows.append({
            "id": case["id"], "split": case["split"], "template": case["template"],
            "promptTokens": len(ids), "goldInitialTokenId": gold_id,
            "firstToken": metrics, "output": generated,
            "stop": stop, "generatedTokens": count,
            "exactAndEos": generated == case["expected"] and stop == "eos",
            "elapsedMs": round((time.perf_counter() - start) * 1000, 1),
            "checkpointSha256": console.DIGEST,
        })
    summary = summarize(rows)
    save_results(dest, root, rows, summary, console, frozen_sha)
    report(summary)
    print(f"Evidence: {dest}")


if __name__ == "__main__":
    main()
