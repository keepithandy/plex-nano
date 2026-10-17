"""Generate the 72 frozen supplied-file diagnostic responses without training.

This runner is restricted to the original P3-02 step-100 checkpoint, its
verified tokenizer bundle, the frozen P3-03 manifest, and one fresh local
artifact directory. It never reads the final project holdout.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from plex_training.checkpoint import read_checkpoint
from plex_training.completion import _completion_tokenizer, _generate_token_ids
from plex_training.file_edit_replacements_bundle import inspect_replacement_bundle
from plex_training.file_edit_replacements_development import (
    DEFAULT_FROZEN_REPORT, EXPECTED_ENDPOINT_SHA256, EXPECTED_TOKENIZER_SHA256,
    _frozen_run_report,
)
from plex_training.file_edit_replacements_stage import BUNDLE_SHA256
from plex_training.model import parameter_count
from plex_training.tokenizer import CODEC, sha256_file


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MANIFEST = HERE / "p3-03-source-preservation-v1.json"
MANIFEST_SHA256 = "119eaa829825e3daa6c9156bd0ba84dfa32726dc2ebfa3d7cfd9ccf7612df578"
LOCAL_RUN = Path.home() / "PlexArtifacts/file-edits/p3-02-first-run"
CHECKPOINT = LOCAL_RUN / "checkpoints/step-0100.pt"
BUNDLE = Path.home() / "PlexArtifacts/p3-02-training-bundle"
OUTPUT = ROOT / "training/artifacts/file-edits/p3-03-source-preservation-v1"
AUTHORIZATION_REFERENCE = (
    'Owner message on 2026-10-17: "continue the work from here now" '
    "following the stated decision whether to authorize the bounded 72-task evaluation"
)


def _frozen_inputs() -> tuple[dict[str, Any], Any, list[list[int]]]:
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("Frozen 72-task manifest SHA-256 changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("taskCount") != 72 or manifest.get("contrastGroupCount") != 24:
        raise ValueError("Frozen 72-task manifest shape changed")
    _frozen_run_report(
        local_result=LOCAL_RUN / "result.json",
        frozen_report=DEFAULT_FROZEN_REPORT,
        checkpoint=CHECKPOINT,
    )
    inspected = inspect_replacement_bundle(BUNDLE)
    if inspected["bundleManifestSha256"] != BUNDLE_SHA256:
        raise ValueError("Frozen P3-02 bundle SHA-256 changed")
    tokenizer, record = _completion_tokenizer(BUNDLE)
    if record["tokenizerSha256"] != EXPECTED_TOKENIZER_SHA256 or tokenizer.vocabulary_size != 16384:
        raise ValueError("Frozen tokenizer identity changed")

    prompts: list[list[int]] = []
    for task in manifest["tasks"]:
        expected_prompt = manifest["serialization"]["template"].format(
            language=task["language"], request=task["request"], input_file=task["inputFile"],
        )
        if task["prompt"] != expected_prompt:
            raise ValueError("Frozen task prompt differs from serialization contract")
        prompt_ids = tokenizer.encode(expected_prompt)
        if not prompt_ids or len(prompt_ids) + 192 > 512:
            raise ValueError("Frozen prompt plus full output budget exceeds context")
        prompts.append(prompt_ids)
    return manifest, tokenizer, prompts


def run() -> Path:
    if OUTPUT.exists():
        raise FileExistsError("Diagnostic output already exists; refusing to rerun or overwrite")
    manifest, tokenizer, prompt_ids_by_task = _frozen_inputs()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                      capture_output=True, text=True).stdout.strip():
        raise ValueError("Commit the fixed inference runner before generating model responses")
    if not torch.cuda.is_available():
        raise ValueError("The bounded diagnostic requires the available local CUDA device")
    device = torch.device("cuda")
    model, payload = read_checkpoint(CHECKPOINT, torch.device("cpu"))
    if (payload.get("codec") != CODEC or type(payload.get("step")) is not int
            or payload["step"] != 100 or parameter_count(model) != 27566080
            or model.config.context_length != 512 or model.config.vocab_size != 16384
            or not isinstance(payload.get("tokenizerRecord"), dict)
            or payload["tokenizerRecord"].get("tokenizerSha256") != EXPECTED_TOKENIZER_SHA256
            or not isinstance(payload.get("datasetRecord"), dict)
            or payload["datasetRecord"].get("bundleManifestSha256") != BUNDLE_SHA256):
        raise ValueError("Checkpoint payload differs from frozen model/tokenizer/bundle identities")
    model = model.to(device)
    model.eval()
    responses: list[dict[str, str]] = []
    generation_audit: list[dict[str, Any]] = []
    for index, (task, prompt_ids) in enumerate(zip(manifest["tasks"], prompt_ids_by_task, strict=True), 1):
        generated, stopped, context = _generate_token_ids(
            model, prompt_ids, actual_vocab=tokenizer.vocabulary_size,
            max_new_tokens=192, temperature=0.0, seed=1337, device=device,
        )
        if not context["fullPromptVisibleForEveryGenerationStep"]:
            raise ValueError("Prompt eviction occurred; refusing this diagnostic run")
        entire = tokenizer.decode(prompt_ids + generated)
        if not entire.startswith(task["prompt"]):
            raise ValueError("Decoded generation does not preserve the supplied prompt")
        responses.append({"taskId": task["taskId"], "actual": entire[len(task["prompt"]):]})
        generation_audit.append({
            "taskId": task["taskId"], "generatedTokenCount": len(generated),
            "stoppedAtEos": stopped,
            "fullPromptVisibleForEveryGenerationStep": True,
        })
        if index % 12 == 0:
            print(f"Generated {index}/72 frozen responses", flush=True)

    envelope = {
        "schemaVersion": 1,
        "manifestSha256": MANIFEST_SHA256,
        "evaluationAuthorization": {
            "status": "separately-authorized",
            "evidenceReference": AUTHORIZATION_REFERENCE,
        },
        "execution": {
            "checkpointStep": 100,
            "checkpointSha256": EXPECTED_ENDPOINT_SHA256,
            "checkpointHashVerified": True,
            "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
            "p302BundleManifestSha256": BUNDLE_SHA256,
            "policy": "greedy-unreselected-fixed-endpoint-v1",
            "seed": 1337,
            "temperature": 0,
            "maximumNewTokens": 192,
            "checkpointReselectionAllowed": False,
            "trainingPerformedDuringDiagnostic": False,
            "optimizerUpdatesDuringDiagnostic": 0,
            "officialP302ScorerModified": False,
            "finalProjectHoldoutOpened": False,
            "promptTemplateSha256": hashlib.sha256(
                manifest["serialization"]["template"].encode("utf-8")
            ).hexdigest(),
            "codeRevision": revision,
            "runnerSha256": sha256_file(Path(__file__)),
            "device": str(device),
        },
        "responses": responses,
        "generationAudit": generation_audit,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(exist_ok=False)
    destination = OUTPUT / "responses.json"
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(envelope, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return destination


if __name__ == "__main__":
    print(run())
