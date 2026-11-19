"""Forward-only controls for the completed November 18 M10 copying run."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

from . import plex600_m10_admission as gate
from . import plex600_m10_copy_proposal as copy
from . import plex600_m10_worker as worker
from . import plex600_m8_eval as m8
from . import plex600_research as research


RUN_SHA = "0de2e3e370baac7ed97c482c754e1a2476b46c7fd0911b0c47e813c66036716a"

# Lexical anchors chosen from the frozen cases before scoring the alternatives.
# Each receives suffix 1 or 2, except the existing 0 literals, which use 0 or 1.
SOURCE_ANCHORS = {
    "m10-html-paragraph": "note", "m10-html-button": "Save",
    "m10-html-link": "Help", "m10-html-label": "email",
    "m10-html-output": "0", "m10-html-emphasis": "Done",
    "m10-css-padding": "card", "m10-css-color": "note",
    "m10-css-display": "pane", "m10-css-border": "tile",
    "m10-css-weight": "badge", "m10-css-margin": "icon",
    "m10-js-boolean": "ready", "m10-js-counter": "0",
    "m10-js-function": "add", "m10-js-array": "colors",
    "m10-js-object": "user", "m10-js-loop": "nums",
}


def verified_inputs(args):
    from .tokenizer import PlexTokenizer

    path = args.run_dir / "report.json"
    if research.digest_file(path) != RUN_SHA:
        raise ValueError("Expected the completed November 18 M10 report")
    report = json.loads(path.read_bytes())
    if research.digest_file(Path(__file__).with_name("plex600.py")) != report["identity"]["modelSha256"]:
        raise ValueError("Model source differs from the approved run")
    backup = copy.audit_m9_backup(args.m9_backup / "report.json", args.m9_backup / "weights.pt")
    m8.verified_bundle(args.bundle, research.BUNDLE_SHA)
    tokenizer = PlexTokenizer.load(args.bundle)
    candidate_sha = copy.audit(tokenizer)["candidateSha256"]
    proof = gate.read_baseline_receipt(
        args.baseline / "report.json", args.baseline / "responses.jsonl",
        m9_weights_sha=backup["m9WeightsSha256"], m9_report_sha=backup["m9ReportSha256"],
        candidate_sha=candidate_sha)
    if (candidate_sha != report["candidateSha256"]
            or backup["m9WeightsSha256"] != report["originalM9WeightsSha256"]
            or backup["m9ReportSha256"] != report["originalM9ReportSha256"]
            or any(proof[k] != report[k] for k in ("baselineReportSha256", "baselineResponsesSha256"))):
        raise ValueError("M9 source, prompt set, or baseline receipt mismatch")
    checkpoints = [("M9", args.m9_backup, args.baseline, backup["m9WeightsSha256"])]
    for snap in report["snapshots"]:
        step = snap["optimizerUpdatesCompleted"]
        folder = args.run_dir / f"step-{step}"
        digest = research.digest_file(folder / "weights.pt", gate.CHECKPOINT_MAX_BYTES)
        if (digest != snap["checkpointSha256"] or
                research.digest_file(folder / "responses.jsonl") != snap["responsesSha256"]):
            raise ValueError(f"Invalid snapshot at update {step}")
        worker.verify_checkpoint_responses(folder / "responses.jsonl", digest, snap["scores"])
        checkpoints.append((f"M10-{step}", folder, folder, digest))
    return tokenizer, checkpoints


def contexts(prompt, answer, newline, space):
    """Token-level controls. Replacement keeps position and all preceding IDs fixed."""
    if prompt[-1] != newline or answer[-1] != newline or not answer[:-1]:
        raise ValueError("Expected separate final newline in both prompt and answer")
    complete = prompt + answer
    return {
        "prompt": prompt, "promptSpace": prompt[:-1] + [space],
        "promptBeforeNewline": prompt[:-1],
        "gold": complete, "goldSpace": complete[:-1] + [space],
        "goldBeforeNewline": complete[:-1],
    }


def metrics(logits, first):
    scores = logits[0, -1].float()
    if not scores.isfinite().all().item():
        raise ValueError("Nonfinite prompt scores")
    probs = scores.softmax(-1)
    return {
        "eosProbability": probs[3].item(), "correctFirstProbability": probs[first].item(),
        "eosMinusFirstLogit": (scores[3] - scores[first]).item(),
        "eosRank": int((scores > scores[3]).sum()) + 1, "greedy": int(scores.argmax()),
    }


def native_states(model, inputs):
    """Capture the full sequence after embedding and after each residual block."""
    states = []
    with ExitStack() as hooks:
        for module in (model.embedding_dropout, *model.blocks):
            hook = module.register_forward_hook(lambda _m, _args, result: states.append(result))
            hooks.callback(hook.remove)
        logits, _ = model(inputs)
    return states, logits


def finish(torch, receiver, hidden, cut, rotary):
    for index in range(cut, len(receiver.blocks)):
        hidden = receiver.blocks[index](hidden, *rotary)
    return torch.nn.functional.linear(receiver.final_norm(hidden), receiver.token_embedding.weight)


def sweep(torch, models, inputs, first):
    """Exchange whole-sequence states in both directions. At selected cuts also
    exchange the final position or its preceding context independently.
    """
    states, native = {}, {}
    for name, model in models.items():
        states[name], native[name] = native_states(model, inputs)
    results = []
    for receiver, donor in (("M9", "M10-10"), ("M10-10", "M9")):
        model = models[receiver]
        rotary = model.rotary(inputs.shape[1], inputs.device)
        depth = len(model.blocks)
        for cut in range(depth + 1):
            old, new = states[receiver][cut], states[donor][cut]
            variants = {"all": new}
            if cut in {0, 1, 2, 4, depth // 2, depth}:
                sham = finish(torch, model, old, cut, rotary)
                if not torch.equal(sham[:, -1], native[receiver][:, -1]):
                    raise ValueError(f"Same-checkpoint control failed at cut {cut}")
                variants["last"] = torch.cat((old[:, :-1], new[:, -1:]), dim=1)
                variants["preceding"] = torch.cat((new[:, :-1], old[:, -1:]), dim=1)
            for positions, state in variants.items():
                logits = finish(torch, model, state, cut, rotary)
                results.append({"receiver": receiver, "donor": donor, "cut": cut,
                                "positions": positions, **metrics(logits, first)})
    return results


def gold_prefix_scores(torch, logits, prompt_length, answer):
    """Predict each content token and terminal EOS with its exact gold prefix."""
    scores = logits[0, prompt_length - 1:].float()
    target = torch.tensor(answer + [3], device=scores.device)
    if scores.shape[0] != len(target):
        raise ValueError("Gold prefix/target length mismatch")
    greedy = scores.argmax(-1)
    gold = scores.gather(1, target[:, None]).squeeze(1)
    return {
        "targetIds": target.tolist(), "greedyIds": greedy.tolist(),
        "eosMinusTargetLogits": (scores[:, 3] - gold).tolist(),
    }


def continuations(torch, model, tokenizer, prompt, answer, logits, historical):
    """Intervene once, then apply the existing fitted greedy decoder for at most 95 tokens."""
    vocab = tokenizer.vocabulary_size
    fitted_first = m8.select_next_id(logits, torch, vocab, "fitted-vocab-v1")
    best_content = int(logits[4:vocab].argmax()) + 4
    generated = {}
    rows = []
    for label, seed in (("bestContentFirst", best_content), ("referenceFirst", answer[0])):
        if seed not in generated:
            prefix = tokenizer.decode([seed])
            if "\ufffd" in prefix:
                raise ValueError("First token cannot be decoded independently")
            suffix, count, reason = m8.generate_one(
                model, torch, tokenizer, prompt + [seed], device=logits.device,
                max_new_tokens=copy.MAX_GENERATION - 1, decoding_policy="fitted-vocab-v1")
            generated[seed] = (prefix + suffix, 1 + count, reason)
        result, length, stop = generated[seed]
        if seed == fitted_first and (result, length, stop) != (
                historical["output"], historical["generatedTokens"], historical["terminationReason"]):
            raise ValueError("Unchanged first choice did not reproduce the fitted generation")
        rows.append({"condition": label, "seedId": seed, "seedCharacters": len(tokenizer.decode([seed])),
                     "correctFirst": seed == answer[0], "output": result,
                     "generatedTokensIncludingSeed": length, "terminationReason": stop})
    return rows


def one_difference(left, right):
    differences = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    if len(left) != len(right) or len(differences) != 1:
        raise ValueError("Matched source test requires exactly one changed token")
    return differences[0]


def source_pair(tokenizer, case):
    anchor = SOURCE_ANCHORS[case["id"]]
    if case["snippet"].count(anchor) != 1:
        raise ValueError("Expected a unique source anchor")
    choices = ("0", "1") if anchor == "0" else (anchor + "1", anchor + "2")
    prompts, answers = [], []
    for choice in choices:
        snippet = case["snippet"].replace(anchor, choice)
        prompts.append(tokenizer.encode(copy.prompt_for({**case, "snippet": snippet})))
        answers.append(tokenizer.encode(snippet))
    source_index = one_difference(*prompts)
    answer_index = one_difference(*answers)
    pair = [answer[answer_index] for answer in answers]
    if [prompt[source_index] for prompt in prompts] != pair:
        raise ValueError("Source and answer probes must compare the same token pair")
    return ([prompt + answers[0][:answer_index] for prompt in prompts], pair,
            {"choices": choices, "sourcePosition": source_index, "answerPosition": answer_index})


def probe_source(torch, model, tokenizer, case):
    inputs, pair, metadata = source_pair(tokenizer, case)
    margins, probabilities, winners = [], [], []
    for ids in inputs:
        logits, _ = model(torch.tensor([ids], device=next(model.parameters()).device))
        scores = logits[0, -1].float()
        margins.append((scores[pair[1]] - scores[pair[0]]).item())
        probabilities.append(scores.softmax(-1)[pair].tolist())
        winners.append(int(scores.argmax()))
    return {**metadata, "tokenIds": pair, "greedyIds": winners, "probabilities": probabilities,
            "bMinusALogits": margins, "pairedChange": margins[1] - margins[0],
            "prefersMatchingTokenBothWays": margins[0] < 0 < margins[1]}


def run(args):
    tokenizer, checkpoints = verified_inputs(args)
    import torch
    from .plex600 import Plex600Config, Plex600Model

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Native CUDA BF16 is required to match the run")
    newline, = tokenizer.encode("\n")
    space, = tokenizer.encode(" ")
    cases = copy.rows()
    prepared = {}
    for case in cases:
        answer = tokenizer.encode(case["snippet"])
        prompt = tokenizer.encode(copy.prompt_for(case))
        prepared[case["id"]] = (answer[0], contexts(prompt, answer, newline, space))
    controls, exchanges, completions, gold_scores, source, models = [], [], [], [], [], {}
    with torch.inference_mode():
        for name, folder, response_dir, _ in checkpoints:
            print(f"Evaluating {name}", file=sys.stderr, flush=True)
            historical = {(row["decodingPolicy"], row["id"]): row
                          for row in research.read_jsonl(response_dir / "responses.jsonl")}
            model = Plex600Model(Plex600Config()).eval().requires_grad_(False)
            state = torch.load(folder / "weights.pt", weights_only=True, mmap=True, map_location="cpu")
            model.load_state_dict(state, strict=True)
            del state
            model.to("cuda")
            if args.study == "source-copy":
                if torch.get_float32_matmul_precision() != "highest":
                    raise ValueError("Source-copy FP32 control requires highest matmul precision")
                for precision in ("bf16", "fp32"):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                        for case in cases:
                            source.append({"checkpoint": name, "case": case["id"], "split": case["split"],
                                           "precision": precision,
                                           **probe_source(torch, model, tokenizer, case)})
                del model
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for case in cases:
                    first, variants = prepared[case["id"]]
                    answer = tokenizer.encode(case["snippet"])
                    for label, ids in variants.items():
                        logits, _ = model(torch.tensor([ids], device="cuda"))
                        values = metrics(logits, first)
                        if label == "prompt":
                            past = historical["unmasked-v1", case["id"]]
                            immediate = past["generatedTokens"] == 0 and past["terminationReason"] == "eos"
                            if (values["greedy"] == 3) != immediate:
                                raise ValueError(f"Historical immediate EOS differs on {case['id']}")
                            for row in continuations(torch, model, tokenizer, ids, answer,
                                                     logits[0, -1], historical["fitted-vocab-v1", case["id"]]):
                                completions.append({"checkpoint": name, "case": case["id"],
                                                    "split": case["split"],
                                                    "exact": row["output"] == case["snippet"],
                                                    "characterSimilarity": copy.character_similarity(
                                                        case["snippet"], row["output"]), **row})
                        if label == "gold":
                            gold_scores.append({"checkpoint": name, "case": case["id"], **gold_prefix_scores(
                                torch, logits, len(variants["prompt"]), answer)})
                        controls.append({"checkpoint": name, "case": case["id"], "split": case["split"],
                                         "context": label, "tokenCount": len(ids), **values})
            if name == "M9":
                models[name] = model
            if name == "M10-10":
                models[name] = model
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for case in cases:
                        first, variants = prepared[case["id"]]
                        exchanges.extend({"case": case["id"], **result} for result in sweep(
                            torch, models, torch.tensor([variants["prompt"]], device="cuda"), first))
                models.clear()
            del model
    return {"protocol": "plex600-m10-warmup-probes-v1", "study": args.study, "runReportSha256": RUN_SHA,
            "checkpointSha256": {name: digest for name, _, _, digest in checkpoints},
            "torchVersion": torch.__version__, "device": torch.cuda.get_device_name(),
            "newlineTokenId": newline, "spaceTokenId": space, "optimizerUpdates": 0,
            "contexts": controls, "exchanges": exchanges,
            "continuations": completions, "goldPrefixes": gold_scores, "sourcePairs": source}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--study", choices=("warmup", "source-copy"), default="warmup")
    for option in ("run-dir", "m9-backup", "baseline", "bundle"):
        parser.add_argument("--" + option, type=Path)
    args = parser.parse_args(argv)
    if not args.run_inference:
        print(json.dumps({"mode": "plan", "checkpoints": ["M9", 10, 25, 50], "cases": 18,
                          "optimizerUpdates": 0, "modelLoaded": False}))
        return
    if not all((args.run_dir, args.m9_backup, args.baseline, args.bundle)):
        parser.error("--run-inference requires --run-dir, --m9-backup, --baseline and --bundle")
    print(json.dumps(run(args), allow_nan=False))


if __name__ == "__main__":
    main()
