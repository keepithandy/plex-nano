"""Zero-update P3 copy-fidelity checkpoint stage, separated from training approval.

Default CLI verifies source and bundle without writing. --authorize-zero-stage
only serializes original weights and an EMPTY optimizer into a fresh stage
directory, then verifies tensor-by-tensor identity. Never optimizer.step().
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
from pathlib import Path
from typing import Any

import torch

from plex_training.artifacts import enforce_storage_limit, path_within_root
from plex_training.checkpoint import read_checkpoint, save_checkpoint
from plex_training.file_edit_replacements_development import DEFAULT_FROZEN_REPORT, _frozen_run_report
from plex_training.file_edit_replacements_stage import _check_stage_payload as _check_prior_stage
from plex_training.model import parameter_count
from plex_training.tokenizer import CODEC, sha256_file

HERE = Path(__file__).resolve().parent
ARTIFACT_ROOT = Path.home() / "PlexArtifacts"
SOURCE_RUN = ARTIFACT_ROOT / "file-edits/p3-02-first-run"
SOURCE_CHECKPOINT = SOURCE_RUN / "checkpoints/step-0100.pt"
BUNDLE_DIR = ARTIFACT_ROOT / "file-edits/p3-copy-fidelity-v1-bundle"
STAGE_DIR = ARTIFACT_ROOT / "file-edits/p3-copy-fidelity-v1-stage0"
STAGE_NAME = "stage-checkpoint.pt"
SOURCE_SHA256 = "020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d"
SOURCE_SIZE = 330899611
ORIGINAL_P302_BUNDLE_SHA = "0cd6df3a6d8eec1d7da8c7a02af067b15299393d8b0e1756a071be17b9aca354"
TOKENIZER_SHA = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
CANDIDATE_SHA = "9414d1da7672a386a920d1935723e9133d011c9c3a4029e5d5800f643835ca7a"

SPEC = importlib.util.spec_from_file_location("copy_fidelity_pack_stage", HERE / "pack_bundle.py")
assert SPEC and SPEC.loader
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)


def _transition(bundle_sha: str) -> dict[str, Any]:
    return {
        "kind": "p3-copy-fidelity-stage-transition-v1",
        "baseMilestone": "P3-02", "baseStep": 100,
        "baseCheckpointSha256": SOURCE_SHA256,
        "candidateSha256": CANDIDATE_SHA, "bundleManifestSha256": bundle_sha,
        "baseModelWeightsPreserved": True, "baseOptimizerStateReused": False,
        "baseSamplerStateReused": False, "baseStepReused": False,
        "stageStep": 0, "stageTokensProcessed": 0,
        "optimizerUpdatesPerformed": 0, "trainingPerformed": False,
        "finalHoldoutOpened": False,
    }


def _settings() -> dict[str, Any]:
    return {
        "kind": "p3-copy-fidelity-zero-stage-serialization-only-v1",
        "purpose": "Serialize original weights and a fresh empty optimizer; no training",
        "trainingAuthorized": False, "evaluationAuthorized": False,
        "optimizerUpdatesAuthorized": 0, "optimizerKind": "AdamW",
        "proposedLearningRate": 0.0001, "microBatch": 1,
        "gradientAccumulation": 16, "seed": 1337,
        "automaticContinuation": False, "resumeAuthorized": False,
    }


def source_preflight(*, source_checkpoint: Path = SOURCE_CHECKPOINT,
                     source_result: Path = SOURCE_RUN / "result.json",
                     bundle_dir: Path = BUNDLE_DIR) -> dict[str, Any]:
    dataset = PACK.inspect(bundle_dir)
    if dataset["candidateSha256"] != CANDIDATE_SHA or not dataset["bundleVerified"]:
        raise ValueError("Copy-fidelity packed bundle failed identity verification")
    _frozen_run_report(
        local_result=source_result, frozen_report=DEFAULT_FROZEN_REPORT,
        checkpoint=source_checkpoint,
    )
    if (source_checkpoint.is_symlink() or not source_checkpoint.is_file()
            or source_checkpoint.stat().st_size != SOURCE_SIZE
            or sha256_file(source_checkpoint) != SOURCE_SHA256):
        raise ValueError("Original P3-02 step-100 checkpoint bytes changed")
    model, payload = read_checkpoint(source_checkpoint, torch.device("cpu"))
    if (parameter_count(model) != 27566080
            or model.config.context_length != 512 or model.config.vocab_size != 16384
            or type(payload.get("step")) is not int or payload["step"] != 100
            or payload.get("codec") != CODEC
            or not isinstance(payload.get("tokenizerRecord"), dict)
            or payload["tokenizerRecord"].get("tokenizerSha256") != TOKENIZER_SHA
            or payload["tokenizerRecord"].get("bundleManifestSha256") != ORIGINAL_P302_BUNDLE_SHA
            or not isinstance(payload.get("initializationRecord"), dict)
            or not isinstance(payload.get("datasetRecord"), dict)
            or payload["datasetRecord"].get("bundleManifestSha256") != ORIGINAL_P302_BUNDLE_SHA):
        raise ValueError("Source model/payload is not original P3-02 fixed endpoint")
    return {"status": "copy-fidelity-source-stage-read-only-preflight-passed",
            "sourceCheckpointSha256": SOURCE_SHA256,
            "bundleManifestSha256": dataset["bundleManifestSha256"],
            "candidateSha256": CANDIDATE_SHA, "sourceWeightsVerified": True,
            "optimizerCreated": False, "trainingPerformed": False,
            "optimizerUpdates": 0, "finalHoldoutOpened": False}


def _dataset_record(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "p3-copy-fidelity-frozen-train-validation-v1",
        "candidateSha256": CANDIDATE_SHA,
        "bundleManifestSha256": dataset["bundleManifestSha256"],
        "trainRecords": dataset["splits"]["train"]["records"],
        "validationRecords": dataset["splits"]["validation"]["records"],
        "trainAnswerAndEosTargets": dataset["splits"]["train"]["answerAndEosTargetPositions"],
        "validationAnswerAndEosTargets": dataset["splits"]["validation"]["answerAndEosTargetPositions"],
        "developmentInTrainingBundle": False, "finalHoldoutOpened": False,
    }


def verify_stage(*, checkpoint: Path,
                 source_checkpoint: Path = SOURCE_CHECKPOINT,
                 source_result: Path = SOURCE_RUN / "result.json",
                 bundle_dir: Path = BUNDLE_DIR) -> dict[str, Any]:
    proof = source_preflight(
        source_checkpoint=source_checkpoint, source_result=source_result,
        bundle_dir=bundle_dir,
    )
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("Stage checkpoint must be a regular file")
    model, _ = read_checkpoint(source_checkpoint, torch.device("cpu"))
    staged, payload = read_checkpoint(checkpoint, torch.device("cpu"))
    dataset = PACK.inspect(bundle_dir)
    optimizer = payload.get("optimizerStateDict")
    if (type(payload.get("step")) is not int or payload["step"] != 0
            or type(payload.get("tokensProcessedTotal")) is not int
            or payload["tokensProcessedTotal"] != 0
            or payload.get("codec") != CODEC or payload.get("seed") != 1337
            or payload.get("trainingSettings") != _settings()
            or payload.get("stageTransitionRecord") != _transition(proof["bundleManifestSha256"])
            or payload.get("datasetRecord") != _dataset_record(dataset)
            or payload.get("scheduleState") is not None
            or not isinstance(optimizer, dict) or optimizer.get("state") != {}
            or not isinstance(optimizer.get("param_groups"), list)
            or len(optimizer["param_groups"]) != 1
            or payload.get("tokenizerRecord", {}).get("tokenizerSha256") != TOKENIZER_SHA):
        raise ValueError("Zero-stage provenance or empty-optimizer invariant failed")
    source_state, stage_state = model.state_dict(), staged.state_dict()
    if (model.config != staged.config
            or parameter_count(staged) != 27566080
            or source_state.keys() != stage_state.keys()
            or any(not torch.equal(source_state[key], stage_state[key]) for key in source_state)):
        raise ValueError("Staged tensor weights differ from the original frozen endpoint")
    return {"status": "copy-fidelity-zero-stage-verified",
            "stageCheckpointSha256": sha256_file(checkpoint),
            **proof, "baseWeightsPreservedExactly": True,
            "stageStep": 0, "stageTokensProcessed": 0,
            "optimizerStateEmpty": True, "optimizerCreatedForSerializationOnly": True}


def create_stage(*, authorize_zero_stage: bool = False,
                 source_checkpoint: Path = SOURCE_CHECKPOINT,
                 source_result: Path = SOURCE_RUN / "result.json",
                 bundle_dir: Path = BUNDLE_DIR, output_dir: Path = STAGE_DIR,
                 artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    if authorize_zero_stage is not True:
        raise ValueError("Staging requires explicit --authorize-zero-stage; no training permission")
    root = artifact_root.resolve()
    output = path_within_root(output_dir, root)
    bundle = bundle_dir.resolve(strict=True)
    base = source_checkpoint.resolve(strict=True)
    if (output == root or output.exists() or output == bundle
            or output in bundle.parents or bundle in output.parents
            or output == base or output in base.parents
            or output == source_result.resolve(strict=False)
            or output in source_result.resolve(strict=False).parents):
        raise ValueError("Zero-stage output must be a fresh directory separate from source artifacts")
    proof = source_preflight(
        source_checkpoint=source_checkpoint, source_result=source_result,
        bundle_dir=bundle_dir,
    )
    dataset = PACK.inspect(bundle_dir)
    model, payload = read_checkpoint(source_checkpoint, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    if optimizer.state:
        raise ValueError("Fresh zero-stage optimizer must have no moments/updates")
    output.parent.mkdir(parents=True, exist_ok=True)
    enforce_storage_limit(root, additional_bytes=2 * SOURCE_SIZE, limit_bytes=PACK.MAX_STORAGE)
    output.mkdir(exist_ok=False)
    checkpoint = output / STAGE_NAME
    try:
        save_checkpoint(
            model, optimizer, step=0, seed=1337, codec=CODEC,
            sampling_rng=random.Random(1337), device=torch.device("cpu"),
            destination=checkpoint, artifact_root=root,
            initialization_record=payload["initializationRecord"],
            stage_transition_record=_transition(proof["bundleManifestSha256"]),
            tokenizer_record=payload["tokenizerRecord"],
            dataset_record=_dataset_record(dataset), training_settings=_settings(),
            tokens_processed_total=0,
        )
        result = verify_stage(
            checkpoint=checkpoint, source_checkpoint=source_checkpoint,
            source_result=source_result, bundle_dir=bundle_dir,
        )
        if not result["baseWeightsPreservedExactly"]:
            raise ValueError("Stage weight preservation could not be proved")
    except Exception:
        shutil.rmtree(output)
        raise
    return {"status": "copy-fidelity-zero-stage-created-and-verified",
            "stageDirectory": str(output), "zeroStageExplicitlyAuthorized": True,
            **result, "trainingPerformed": False, "optimizerUpdates": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--authorize-zero-stage", action="store_true",
                       help="Create one fresh checkpoint without optimizer updates")
    flags.add_argument("--verify-stage", action="store_true",
                       help="Verify existing staged tensors/provenance")
    parser.add_argument("--stage-dir", type=Path, default=STAGE_DIR)
    args = parser.parse_args()
    try:
        if args.authorize_zero_stage:
            result = create_stage(authorize_zero_stage=True, output_dir=args.stage_dir)
        elif args.verify_stage:
            result = verify_stage(checkpoint=args.stage_dir / STAGE_NAME)
        else:
            result = source_preflight()
        print(json.dumps(result, indent=2))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"p3-copy-fidelity-stage: {exc}\n")


if __name__ == "__main__":
    main()
