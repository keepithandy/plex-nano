"""Explicit, inference-only Phase 2 M11 comparison on the owner's CUDA setup.

Does not call a training worker, optimizer, or experiment approval flow.
Results are written to a fresh directory and scored separately from the editor.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUITE = ROOT / "fixtures" / "plex_phase2_requests_v1.json"


def load_console():
    spec = importlib.util.spec_from_file_location("plex_phase2_console", HERE / "PlexLocal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def alternative_prompt(source, request):
    return (
        "You are editing only the supplied JavaScript. Return the entire edited file, "
        "not a description. If the request is ambiguous or cannot be safely completed, "
        "return a single line starting REFUSE:.\n"
        f"Instruction: {request}\n"
        f"Source JavaScript:\n{source}\n"
        "Edited JavaScript:\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Inference only: compare M11 prompt formats. Does not train.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing paths are refused.")
    parser.add_argument("--execute-read-only", action="store_true",
                        help="Explicitly run both prompt formats with the current checkpoint on CUDA.")
    args = parser.parse_args()
    if not args.execute_read_only:
        parser.error("Add --execute-read-only to load the existing checkpoint; no training will occur.")
    cases = json.loads(SUITE.read_text(encoding="utf-8"))["cases"]
    with SUITE.open("rb") as stream:
        suite_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        parser.error(f"Output directory already exists: {output_dir}")
    console = load_console()
    from plex_training.plex600_m8_eval import prompt_for
    from plex_training.tokenizer import PlexTokenizer

    tokenizer = PlexTokenizer.load(console.ARTIFACTS / "plex600-fresh-tokenizer")
    prompts = []
    for case in cases:
        for name, prompt in (
            ("baseline", prompt_for({"request": case["request"], "inputFile": case["source"]})),
            ("alternative", alternative_prompt(case["source"], case["request"])),
        ):
            tokens = tokenizer.encode(prompt)
            if len(tokens) > 1024 - 96:
                raise ValueError(f"Prompt exceeds context for {case['id']} ({name})")
            prompts.append((case["id"], name, tokens))
    model = console.load_model()  # SHA-256 verified; inference only.
    results = {"baseline": [], "alternative": []}
    for case_id, name, tokens in prompts:
        start = time.perf_counter()
        answer = console.decode(model, tokenizer, tokens)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        results[name].append({
            "id": case_id, "text": answer["text"], "stop": answer["stop"],
            "elapsedMs": elapsed_ms, "generatedTokens": answer["generatedTokens"],
            "checkpointSha256": answer["checkpointSha256"],
        })
        print(f"{name}: {case_id} -> {answer['stop']} ({answer['generatedTokens']} tokens)", flush=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("baseline", "alternative"):
        with (output_dir / f"{name}.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in results[name]:
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    with (output_dir / "identity.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "suiteSha256": suite_sha, "checkpointSha256": console.DIGEST,
            "policy": "fitted-vocab-v1", "maxNewTokens": 96,
            "trainingPerformed": False, "prompts": ["baseline", "alternative"],
            "casesPerPrompt": len(cases),
        }, stream, indent=2)
        stream.write("\n")
    print(f"Read-only M11 outputs saved in new directory: {output_dir}")
    print("Score with scripts/plex_phase2_eval.py --baseline-model-jsonl "
          "<dir>/baseline.jsonl --alternative-model-jsonl <dir>/alternative.jsonl")


if __name__ == "__main__":
    main()
