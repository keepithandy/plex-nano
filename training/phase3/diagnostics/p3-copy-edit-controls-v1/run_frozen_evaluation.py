"""Preflight or run the single authorized 42-task, fixed-step-100 P3 diagnostic.

No checkpoint selection, optimizer, training, automatic retry, or holdout access.
Default invocation performs *preflight only*. --run is an explicit opt-in and
refuses any existing result directory. Requires the owner's local artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from plex_training.checkpoint import read_checkpoint
from plex_training.completion import _completion_tokenizer, _generate_token_ids
from plex_training.file_edit_replacements_bundle import inspect_replacement_bundle
from plex_training.file_edit_replacements_development import (
    DEFAULT_FROZEN_REPORT, _frozen_run_report,
)
from plex_training.file_edit_replacements_stage import BUNDLE_SHA256
from plex_training.model import parameter_count
from plex_training.tokenizer import CODEC, sha256_file


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MANIFEST = HERE / "copy-edit-controls-v1.json"
MANIFEST_SHA256 = "17834efb801ac963a4672ae616ca47e9eea40b2568f01951c3891c7e4997f01a"
P302_STEP100_SHA256 = "020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d"
TOKENIZER_SHA256 = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
LOCAL_RUN = Path.home() / "PlexArtifacts/file-edits/p3-02-first-run"
CHECKPOINT = LOCAL_RUN / "checkpoints/step-0100.pt"
BUNDLE = Path.home() / "PlexArtifacts/p3-02-training-bundle"
OUTPUT = Path.home() / "PlexArtifacts/file-edits/p3-copy-edit-controls-v1-evaluation"
MAX_NEW_TOKENS = 192
SEED = 1337

PREP_SPEC = importlib.util.spec_from_file_location("p3_copy_edit_prep", HERE / "prepare_controls.py")
assert PREP_SPEC is not None and PREP_SPEC.loader is not None
PREP = importlib.util.module_from_spec(PREP_SPEC)
PREP_SPEC.loader.exec_module(PREP)


def _manifest() -> dict[str, Any]:
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("Fixed copy/edit manifest SHA-256 mismatch")
    if MANIFEST.read_bytes() != PREP.serialized_manifest():
        raise ValueError("The frozen manifest differs from its deterministic builder")
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if value.get("schemaVersion") != 1 or len(value.get("tasks", [])) != 42:
        raise ValueError("Frozen 42-task manifest contract invalid")
    if value.get("authorization", {}).get("trainingAuthorized") is not False:
        raise ValueError("This experiment cannot authorize training")
    return value


def preflight() -> tuple[dict[str, Any], Any, list[list[int]]]:
    """No model inference, optimizer steps, or artifact writes."""
    manifest = _manifest()
    _frozen_run_report(
        local_result=LOCAL_RUN / "result.json",
        frozen_report=DEFAULT_FROZEN_REPORT,
        checkpoint=CHECKPOINT,
    )
    if sha256_file(CHECKPOINT) != P302_STEP100_SHA256:
        raise ValueError("Fixed step-100 checkpoint bytes changed")
    inspected = inspect_replacement_bundle(BUNDLE)
    if inspected["bundleManifestSha256"] != BUNDLE_SHA256:
        raise ValueError("Frozen P3-02 bundle identity changed")
    tokenizer, record = _completion_tokenizer(BUNDLE)
    if record["tokenizerSha256"] != TOKENIZER_SHA256 or tokenizer.vocabulary_size != 16384:
        raise ValueError("Frozen tokenizer identity changed")
    prompts: list[list[int]] = []
    for task in manifest["tasks"]:
        correct = PREP.PROMPT.format(
            language=task["language"], request=task["request"], input_file=task["inputFile"],
        )
        if task["prompt"] != correct:
            raise ValueError(f"Prompt contract altered: {task['taskId']}")
        tokens = tokenizer.encode(correct)
        if not tokens or len(tokens) + MAX_NEW_TOKENS + 1 > 512:
            raise ValueError(f"Prompt may exceed fixed 512-token budget: {task['taskId']}")
        if tokenizer.decode(tokens) != correct:
            raise ValueError(f"Tokenizer failed prompt roundtrip: {task['taskId']}")
        prompts.append(tokens)
    return manifest, tokenizer, prompts


def run(authorization_reference: str) -> Path:
    if not authorization_reference.strip():
        raise ValueError("A separately recorded owner evaluation authorization reference is required")
    if OUTPUT.exists():
        raise FileExistsError("Existing evaluation output; no overwrite, retry, or reselection")
    manifest, tokenizer, prompts = preflight()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                      capture_output=True, text=True, check=True).stdout.strip():
        raise ValueError("Commit the fixed runner before evaluating the model")
    if not torch.cuda.is_available():
        raise ValueError("Local CUDA is required for this fixed-endpoint diagnostic")
    device = torch.device("cuda")
    model, payload = read_checkpoint(CHECKPOINT, torch.device("cpu"))
    if (payload.get("codec") != CODEC
        or type(payload.get("step")) is not int or payload["step"] != 100
        or parameter_count(model) != 27566080
        or model.config.context_length != 512 or model.config.vocab_size != 16384
        or not isinstance(payload.get("tokenizerRecord"), dict)
        or payload["tokenizerRecord"].get("tokenizerSha256") != TOKENIZER_SHA256
        or not isinstance(payload.get("datasetRecord"), dict)
        or payload["datasetRecord"].get("bundleManifestSha256") != BUNDLE_SHA256):
        raise ValueError("Checkpoint internal identities do not match the frozen endpoint")
    model = model.to(device)
    model.eval()
    responses, generation_audit = [], []
    for task, prompt_ids in zip(manifest["tasks"], prompts, strict=True):
        generated, stopped, context = _generate_token_ids(
            model, prompt_ids, actual_vocab=tokenizer.vocabulary_size,
            max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, seed=SEED, device=device,
        )
        if not context["fullPromptVisibleForEveryGenerationStep"]:
            raise ValueError(f"Source prompt evicted for {task['taskId']}; aborting without results")
        full = tokenizer.decode(prompt_ids + generated)
        if not full.startswith(task["prompt"]):
            raise ValueError(f"Decode failed to retain original prompt: {task['taskId']}")
        responses.append({"taskId": task["taskId"], "actual": full[len(task["prompt"]):]})
        generation_audit.append({
            "taskId": task["taskId"],
            "generatedTokenCount": len(generated),
            "stoppedAtEos": stopped,
            "fullPromptVisibleForEveryGenerationStep": True,
        })
    envelope = {
        "schemaVersion": 1,
        "manifestSha256": MANIFEST_SHA256,
        "evaluationAuthorization": {
            "status": "separately-authorized",
            "evidenceReference": authorization_reference,
        },
        "execution": {
            "checkpointStep": 100, "checkpointSha256": P302_STEP100_SHA256,
            "checkpointHashVerified": True,
            "tokenizerSha256": TOKENIZER_SHA256,
            "p302BundleManifestSha256": BUNDLE_SHA256,
            "policy": "greedy-unreselected-fixed-endpoint-v1",
            "seed": SEED, "temperature": 0,
            "maximumNewTokens": MAX_NEW_TOKENS,
            "checkpointReselectionAllowed": False,
            "trainingPerformedDuringDiagnostic": False,
            "optimizerUpdatesDuringDiagnostic": 0,
            "finalProjectHoldoutOpened": False,
            "promptTemplateSha256": hashlib.sha256(PREP.PROMPT.encode("utf-8")).hexdigest(),
            "codeRevision": revision,
            "runnerSha256": sha256_file(Path(__file__)),
            "device": str(device),
        },
        "responses": responses,
        "generationAudit": generation_audit,
    }
    # Publication is the last step, after all generations and their checks.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(exist_ok=False)
    path = OUTPUT / "responses.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(envelope, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute one authorized 42-task evaluation")
    parser.add_argument("--authorization-reference", default="", help="Owner's separate evaluation approval, recorded verbatim")
    args = parser.parse_args()
    try:
        if args.run:
            print(json.dumps({"result": str(run(args.authorization_reference)),
                              "evaluation": "performed", "trainingPerformed": False}))
        else:
            manifest, _tokenizer, prompts = preflight()
            print(json.dumps({
                "status": "local-preflight-passed-no-generation",
                "manifestSha256": MANIFEST_SHA256, "taskCount": len(manifest["tasks"]),
                "longestPromptTokens": max(map(len, prompts)),
                "maximumNewTokens": MAX_NEW_TOKENS,
                "checkpointVerified": True, "tokenizerVerified": True,
                "optimizerUpdates": 0,
            }, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"p3-copy-edit: {exc}\n")


if __name__ == "__main__":
    main()
