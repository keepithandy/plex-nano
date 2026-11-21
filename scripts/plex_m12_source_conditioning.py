"""M12-A read-only M11 teacher-forced source-conditioning diagnostic.

Measures fitted-vocabulary next-token ranks on the four FROZEN M11
training/development examples. Contrasts the first identifier token under
score versus value sources and original versus general-console prompts.
NEVER claims teacher-forced token matches are autonomous generations.
No optimizer, gradient, checkpoint writes, final holdout or training.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "training/artifacts"
PROTOCOL = "plex-m12-a-m11-teacher-forcing-v1"
WEIGHTS_SHA = "d7a7776de86230320f6e0e48b07dc76761c3d42ac80e4c330839e47c9ffc93dd"
TOKENIZER_SHA = "9feef87bdf5fc28b3315df589e4e0b8ffcf5f40b18a4c357c91759182340a2e2"
TEMPLATES = ("m11-trained", "console-general")
EOS_ID = 3
MAX_TARGET_WITH_EOS = 96
MAX_CONTEXT = 1024


def console_module():
    spec = importlib.util.spec_from_file_location("plex_m12_console", ROOT / "scripts/PlexLocal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def planned_cases(cases, trained_prompt, console_prompt):
    if (len(cases) != 4 or
            {(case["split"], case["snippet"]) for case in cases} != {
                ("train", "let score = 0;\n"), ("train", "let score = 1;\n"),
                ("probe-only", "let value = 0;\n"), ("probe-only", "let value = 1;\n")}):
        raise ValueError("Only four frozen M11 research cases are admitted")
    result = []
    for case in cases:
        trained = trained_prompt(case)
        if not trained.endswith("Copy: "):
            raise ValueError("Original M11 prompt delimiter changed")
        general = console_prompt({
            "request": "Return the supplied JavaScript line exactly as written.",
            "inputFile": case["snippet"],
        })
        if not general.endswith("Edited file:\n"):
            raise ValueError("General console prompt delimiter changed")
        for template, prompt in (("m11-trained", trained), ("console-general", general)):
            result.append({
                "id": case["id"], "split": case["split"], "template": template,
                "prompt": prompt, "expected": case["snippet"],
                "digit": "0" if case["snippet"].endswith("0;\n") else "1",
            })
    return result


def first_difference(a, b):
    """Earliest target token that differs. Both prefixes must be nonempty."""
    if not a or not b:
        raise ValueError("Nonempty token targets required")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            if i == 0:
                raise ValueError("No common answer prefix to isolate the identifier")
            return i, x, y
    raise ValueError("No early identifier divergence found between targets")


def score_fitted_logits(fitted_logits, expected_id):
    """Pure CPU scoring with logits indexed from EOS ID 3, not from token zero."""
    if (not isinstance(fitted_logits, (list, tuple)) or len(fitted_logits) < 2
            or type(expected_id) is not int
            or not 3 <= expected_id < 3 + len(fitted_logits)
            or any(type(value) not in (float, int) or not math.isfinite(value)
                   for value in fitted_logits)):
        raise ValueError("Invalid fitted-vocabulary logits or expected ID")
    best_index = max(range(len(fitted_logits)), key=lambda i: fitted_logits[i])
    best = fitted_logits[best_index]
    gold = fitted_logits[expected_id - 3]
    scale = best + math.log(sum(math.exp(value - best) for value in fitted_logits))
    best_code = max(fitted_logits[1:])
    return {
        "goldId": expected_id,
        "fittedGreedyId": best_index + 3,
        "goldRank": 1 + sum(value > gold for value in fitted_logits),
        "goldNegativeLogProbability": round(scale - gold, 5),
        "goldMinusEosLogit": round(gold - fitted_logits[0], 4),
        "eosMinusBestCodeLogit": round(fitted_logits[0] - best_code, 4),
    }


def score_target(logits_by_step, gold_ids):
    """Score target and terminal EOS; never generate or accept code."""
    if len(gold_ids) < 2 or gold_ids[-1] != EOS_ID or len(logits_by_step) != len(gold_ids):
        raise ValueError("Teacher-forced logits must include the gold EOS step")
    scored = [score_fitted_logits(logits, gold) for logits, gold in
              zip(logits_by_step, gold_ids)]
    return {
        "tokens": scored,
        "targetTokensPlusEos": len(gold_ids),
        "goldGreedyMatches": sum(row["fittedGreedyId"] == row["goldId"] for row in scored),
        "meanFittedGoldNll": round(sum(row["goldNegativeLogProbability"]
                                      for row in scored) / len(scored), 5),
        "firstTargetTokenRank": scored[0]["goldRank"],
        "eosGoldRankAtGoldEnd": scored[-1]["goldRank"],
        "prematureEosSteps": sum(row["fittedGreedyId"] == EOS_ID for row in scored[:-1]),
        "teacherForced": True,
        "autonomousGenerationPerformed": False,
    }


def contrast_logit(fitted_logits, score_id, value_id):
    if score_id == value_id or not (3 <= score_id < len(fitted_logits) + 3
                                     and 3 <= value_id < len(fitted_logits) + 3):
        raise ValueError("Source-contrast IDs are missing or identical")
    return round(fitted_logits[value_id - 3] - fitted_logits[score_id - 3], 4)


def summarize(rows):
    if len(rows) != 8 or len({(r["id"], r["template"]) for r in rows}) != 8:
        raise ValueError("Expected eight unique teacher-forced M11 cases")
    groups = {}
    for template in TEMPLATES:
        for split in ("train", "probe-only"):
            cases = [r for r in rows if r["template"] == template and r["split"] == split]
            if len(cases) != 2:
                raise ValueError("Incomplete teacher-forced group")
            name = f"{template}/{split}"
            groups[name] = {
                "cases": len(cases),
                "meanFittedGoldNll": round(sum(r["scores"]["meanFittedGoldNll"]
                                               for r in cases) / 2, 4),
                "firstTokenEosCases": sum(
                    r["scores"]["tokens"][0]["fittedGreedyId"] == EOS_ID for r in cases),
                "allGoldGreedyTokens": sum(
                    r["scores"]["goldGreedyMatches"] == r["scores"]["targetTokensPlusEos"]
                    for r in cases),
            }
    contrasts = {}
    for template in TEMPLATES:
        count = 0
        for digit in ("0", "1"):
            train = next((r for r in rows if r["split"] == "train"
                          and r["template"] == template and r["digit"] == digit), None)
            probe = next((r for r in rows if r["split"] == "probe-only"
                          and r["template"] == template and r["digit"] == digit), None)
            if train is None or probe is None:
                raise ValueError("M12-A missing matched source-contrast cases")
            count += int(train["valueMinusScoreLogit"] < 0 < probe["valueMinusScoreLogit"])
        contrasts[template] = {"switchedPreferencePairs": count, "pairs": 2}
    return {"groups": groups, "sourceContrasts": contrasts,
            "trainingPerformed": False, "modelImprovementClaimed": False}


def output_destination(path):
    root = ARTIFACTS.resolve(strict=True)
    dest = path.expanduser().resolve()
    if dest.exists() or path.is_symlink() or dest.parent != root or path.parent.is_symlink():
        raise ValueError("M12-A output must be a NEW direct child of training/artifacts")
    return dest, root


def save_report(dest, root, rows, summary, frozen_hash):
    stage = Path(tempfile.mkdtemp(prefix=f".{dest.name}.staging-", dir=root))
    try:
        raw = "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
                      for row in rows).encode("utf-8")
        (stage / "rows.jsonl").write_bytes(raw)
        report = {
            "protocol": PROTOCOL, "checkpointSha256": WEIGHTS_SHA,
            "tokenizerSha256": TOKENIZER_SHA, "m11FrozenPromptSha256": frozen_hash,
            "responsesSha256": hashlib.sha256(raw).hexdigest(),
            "cases": 4, "prompts": list(TEMPLATES), "scoredContexts": 8,
            "scoring": "gold-prefix-fitted-vocabulary-next-token-logit-v1",
            "teacherForced": True, "goldTargetSuppliedToModel": True,
            "autonomousGenerationPerformed": False,
            "trainingPerformed": False, "optimizerCreated": False,
            "checkpointWrites": False, "finalHoldoutAccessed": False,
            "summary": summary,
        }
        (stage / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if dest.exists():
            raise FileExistsError("Refuse to replace existing M12-A evidence")
        os.rename(stage, dest)
        stage = None
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def print_summary(summary):
    print("PLEX M12-A - SOURCE CONDITIONING (TEACHER FORCED)")
    for label, group in summary["groups"].items():
        print(f"{label}: mean fitted gold NLL={group['meanFittedGoldNll']:.4f}; "
              f"first-token EOS={group['firstTokenEosCases']}/2; "
              f"all-gold greedy={group['allGoldGreedyTokens']}/2.")
    for template, result in summary["sourceContrasts"].items():
        print(f"{template}: identifier preference switched correctly in "
              f"{result['switchedPreferencePairs']}/2 matched source pairs.")
    print("These are gold-prefix diagnostics, NOT autonomous coding or training.")
    print("Training: paused. No weights, tokenizer, or holdout were changed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute-read-only", action="store_true")
    args = parser.parse_args()
    if not args.execute_read_only:
        parser.error("Add --execute-read-only to score existing weights; training is not permitted")
    dest, root = output_destination(args.output_dir)

    console = console_module()
    if console.DIGEST != WEIGHTS_SHA:
        raise ValueError("Wrong frozen M11 checkpoint SHA")
    from plex_training import plex600_m11_source_conditioning as m11
    from plex_training.plex600_m8_eval import prompt_for as console_prompt
    from plex_training.tokenizer import PlexTokenizer
    bundle = console.ARTIFACTS / "plex600-fresh-tokenizer"
    settings = json.loads((bundle / "tokenizer-config.json").read_text(encoding="utf-8"))
    if settings.get("tokenizerSha256") != TOKENIZER_SHA:
        raise ValueError("Wrong M11 tokenizer identity")
    tokenizer = PlexTokenizer.load(bundle)
    cases = planned_cases(m11.cases(), m11.prompt_for, console_prompt)
    frozen_hash = hashlib.sha256(json.dumps(
        cases, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()

    prepared = []
    for row in cases:
        prompt_ids = tokenizer.encode(row["prompt"])
        target_ids = tokenizer.encode(row["expected"])
        if (not prompt_ids or not target_ids
                or len(target_ids) + 1 > MAX_TARGET_WITH_EOS
                or len(prompt_ids) + len(target_ids) > MAX_CONTEXT
                or tokenizer.decode(prompt_ids) != row["prompt"]
                or tokenizer.decode(target_ids) != row["expected"]):
            raise ValueError("M12-A prompt/target tokenizer or length mismatch")
        prepared.append((row, prompt_ids, target_ids))

    # The score and value outputs differ at an identifier token under the SAME digit.
    contrasts = {}
    for digit in ("0", "1"):
        score_ids = next(ids for row, _, ids in prepared
                         if row["split"] == "train" and row["digit"] == digit)
        value_ids = next(ids for row, _, ids in prepared
                         if row["split"] == "probe-only" and row["digit"] == digit)
        contrasts[digit] = first_difference(score_ids, value_ids)

    import torch
    model = console.load_model()  # verifies SHA before CUDA load
    model.eval()
    rows = []
    with torch.inference_mode():
        for row, prompt_ids, target_ids in prepared:
            inputs = torch.tensor(
                [prompt_ids + target_ids], dtype=torch.long, device=torch.device("cuda"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(inputs)
            start = len(prompt_ids) - 1
            gold_ids = target_ids + [EOS_ID]
            fitted_steps = (logits[0, start:start + len(gold_ids), 3:tokenizer.vocabulary_size]
                            .float().cpu().tolist())
            scores = score_target(fitted_steps, gold_ids)
            difference_idx, score_id, value_id = contrasts[row["digit"]]
            if difference_idx >= len(target_ids):
                raise ValueError("Source-contrast target token missing")
            source_contrast = contrast_logit(
                fitted_steps[difference_idx], score_id, value_id)
            rows.append({
                "id": row["id"], "split": row["split"],
                "digit": row["digit"], "template": row["template"],
                "promptTokens": len(prompt_ids), "answerTokens": len(target_ids),
                "sourceContrastPosition": difference_idx,
                "sourceContrastTokenIds": {"score": score_id, "value": value_id},
                "valueMinusScoreLogit": source_contrast,
                "scores": scores, "checkpointSha256": WEIGHTS_SHA,
            })
    summary = summarize(rows)
    save_report(dest, root, rows, summary, frozen_hash)
    print_summary(summary)
    print(f"Evidence: {dest}")


if __name__ == "__main__":
    main()
