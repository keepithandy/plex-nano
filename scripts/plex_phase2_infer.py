"""Explicit, inference-only Phase 2 M11 comparison on the owner's CUDA setup.

Does not call a training worker, optimizer, or experiment approval flow.
Results are written to a fresh directory and scored separately from the editor.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUITE = ROOT / "fixtures" / "plex_phase2_requests_v1.json"
EXPECTED_TOKENIZER_SHA256 = "9feef87bdf5fc28b3315df589e4e0b8ffcf5f40b18a4c357c91759182340a2e2"
PROTOCOL = "plex-phase2-m11-inference-v1"


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
    artifact_root = (ROOT / "training/artifacts").resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if (output_dir.exists() or args.output_dir.is_symlink() or
            output_dir.parent != artifact_root or
            args.output_dir.parent.is_symlink()):
        parser.error("Output must be a NEW direct child of training/artifacts; no existing files are overwritten.")
    if len(cases) != 33 or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Phase 2 expects the complete 33-case development suite.")
    console = load_console()
    from plex_training.plex600_m8_eval import prompt_for
    from plex_training.tokenizer import PlexTokenizer

    bundle = console.ARTIFACTS / "plex600-fresh-tokenizer"
    tokenizer_config = json.loads((bundle / "tokenizer-config.json").read_text(encoding="utf-8"))
    if tokenizer_config.get("tokenizerSha256") != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("Local tokenizer does not match the frozen M11 experiment identity.")
    tokenizer = PlexTokenizer.load(bundle)
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
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=artifact_root))
    try:
        response_hashes = {}
        for name in ("baseline", "alternative"):
            raw = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in results[name]).encode("utf-8")
            with (stage / f"{name}.jsonl").open("xb") as stream:
                stream.write(raw)
            response_hashes[name] = hashlib.sha256(raw).hexdigest()
        with (stage / "identity.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump({
                "protocol": PROTOCOL, "suiteSha256": suite_sha,
                "checkpointSha256": console.DIGEST,
                "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
                "policy": "fitted-vocab-v1", "maxNewTokens": 96,
                "trainingPerformed": False, "prompts": ["baseline", "alternative"],
                "casesPerPrompt": len(cases), "responseSha256": response_hashes,
            }, stream, indent=2)
            stream.write("\n")
        if output_dir.exists():
            raise FileExistsError(f"Output directory appeared during generation: {output_dir}")
        os.rename(stage, output_dir)
        stage = None
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
    print(f"Read-only M11 outputs saved in new directory: {output_dir}")
    print("Score with scripts/plex_phase2_eval.py --baseline-model-jsonl "
          "<dir>/baseline.jsonl --alternative-model-jsonl <dir>/alternative.jsonl")


if __name__ == "__main__":
    main()
