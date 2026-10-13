"""P3-02 exact-replacement train/validation bundle and read-only P2-48 source checks.

No optimizer, stage checkpoint, trainer, or final-development reader lives here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

import torch

from .artifacts import enforce_storage_limit, path_within_root
from .checkpoint import read_checkpoint
from .completion import _completion_tokenizer
from .config import DEFAULT_CONFIG
from .file_edit_replacements import (
    DEFAULT_BUNDLE as DEFAULT_SOURCE_TOKENIZER,
    DEFAULT_CANDIDATE, DEFAULT_CONTRACT, DEFAULT_REVIEW,
    EXPECTED_SHA256, EXPECTED_TOKENIZER_SHA256, MILESTONE, PROPOSED_RUN,
    SAFE_FLAGS, TOKENIZER_BUNDLE_SHA256, ROOT, _contract,
    answer_only_target_mask, build_replacement_rows, plan_replacement_supervision,
    preflight_replacement_supervision, render_prompt,
)
from .model import parameter_count
from .tokenizer import CODEC, sha256_file

PIPELINE_VERSION = "p3-02-answer-eos-bundle-v1"
DEFAULT_OUTPUT = ROOT / "training/artifacts/file-edits/p3-02-training-bundle"
DEFAULT_CHECKPOINT = ROOT / "training/artifacts/request-grounded/p2-48-first-run/checkpoints/step-0100.pt"
DEFAULT_P248_RESULT = ROOT / "training/pretraining/p2-48-final-run-result.json"
SOURCE_CHECKPOINT_SHA256 = PROPOSED_RUN["sourceCheckpointSha256"]
SOURCE_CHECKPOINT_BYTES = 330900379
SPLITS = ("train", "validation")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer-config.json", "model-config.json")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _safe_file(path: Path, maximum: int = 2 * 1024 * 1024) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("P3-02 bundle file missing, linked, or oversized")
    return path.read_bytes()


def _split_files(tokenizer: Any) -> tuple[dict[str, dict[str, bytes]], dict[str, Any]]:
    """Encode only the frozen train/validation rows with separate prompt/answer BPE."""
    rows = build_replacement_rows()
    plan = plan_replacement_supervision(
        rows, tokenizer,
        seed=PROPOSED_RUN["seed"],
        examples=PROPOSED_RUN["maximumOptimizerUpdates"] *
                 PROPOSED_RUN["microBatch"] * PROPOSED_RUN["gradientAccumulation"],
    )
    files: dict[str, dict[str, bytes]] = {}
    for split in SPLITS:
        jsonl = bytearray()
        tokens = bytearray()
        weights = bytearray()
        index = []
        position = target_position = 0
        for row in rows:
            if row["candidateSplit"] != split:
                continue
            prompt = render_prompt(row)
            target = row["expectedFile"]
            prompt_ids = tokenizer.encode(prompt)
            answer_ids = tokenizer.encode(target)
            packed = prompt_ids + answer_ids + [3]
            mask = bytes(answer_only_target_mask(len(prompt_ids), len(answer_ids)))
            assert len(mask) == len(packed) - 1
            jsonl += _canonical({
                "recordId": row["id"], "text": prompt + target,
                "promptTokens": len(prompt_ids),
                "answerTokensIncludingEos": len(answer_ids) + 1,
            })
            index.append({
                "recordId": row["id"], "startToken": position,
                "tokenCount": len(packed), "startTarget": target_position,
                "targetCount": len(mask), "promptTokens": len(prompt_ids),
                "answerTokensIncludingEos": len(answer_ids) + 1,
            })
            tokens += struct.pack("<" + "H" * len(packed), *packed)
            weights += mask
            position += len(packed)
            target_position += len(mask)
        files[split] = {
            "jsonl": bytes(jsonl),
            "tokens.u16le": bytes(tokens),
            "weights.u8": bytes(weights),
            "index.json": (json.dumps(index, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        }
        if hashlib.sha256(files[split]["tokens.u16le"]).hexdigest() != plan["packedTokenSha256"][split]:
            raise ValueError("P3-02 bundle token bytes disagree with read-only sampler preflight")
        if (len(tokens) // 2 != plan["recordTokens"][split]
                or sum(weights) != plan["answerAndEosTargetPositions"][split]
                or len(index) != plan["records"][split]):
            raise ValueError("P3-02 bundle target accounting differs from reviewed proposal")
    return files, plan


def _manifest(files: dict[str, dict[str, bytes]], plan: dict[str, Any],
              model_config_sha256: str) -> dict[str, Any]:
    splits = {
        split: {
            "records": plan["records"][split],
            "recordTokens": plan["recordTokens"][split],
            "answerAndEosTargetPositions": plan["answerAndEosTargetPositions"][split],
            **{name + "Sha256": hashlib.sha256(raw).hexdigest()
               for name, raw in files[split].items()},
        }
        for split in SPLITS
    }
    return {
        "schemaVersion": 1, "milestone": MILESTONE,
        "pipelineVersion": PIPELINE_VERSION,
        "codec": CODEC, "actualVocabularySize": 16384,
        "modelVocabularyCapacity": DEFAULT_CONFIG.vocab_size,
        "modelConfigSha256": model_config_sha256,
        "contextLength": DEFAULT_CONFIG.context_length,
        "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
        "sourceTokenizerBundleManifestSha256": TOKENIZER_BUNDLE_SHA256,
        "candidateSha256": EXPECTED_SHA256,
        "supervision": plan["supervision"],
        "maskFormat": "one uint8 weight per next-token target; prompt zero; answer/EOS one",
        "proposalSampler": plan["sampler"],
        "splits": splits,
        **SAFE_FLAGS,
    }


def prepare_replacement_bundle(
    *, output_dir: Path = DEFAULT_OUTPUT,
    source_bundle_dir: Path = DEFAULT_SOURCE_TOKENIZER,
    candidate_path: Path = DEFAULT_CANDIDATE,
    review_path: Path = DEFAULT_REVIEW,
    contract_path: Path = DEFAULT_CONTRACT,
    artifact_root: Path = ROOT / "training/artifacts",
    storage_limit_bytes: int = 200 * 1024**3,
) -> dict[str, Any]:
    """Write one atomic, verified P3-02 training bundle. Zero optimizer updates."""
    if type(storage_limit_bytes) is not int or storage_limit_bytes < 1:
        raise ValueError("P3-02 artifact budget must be a positive integer")
    root = artifact_root.resolve()
    output = path_within_root(output_dir, root)
    source = source_bundle_dir.resolve(strict=True)
    if (output == root or output.exists() or output == source
            or output in source.parents or source in output.parents):
        raise ValueError("P3-02 bundle needs a fresh output separate from the source")
    # The reviewed local candidate, contract, review, and tokenizer must never be output aliases.
    for protected in (candidate_path, review_path, contract_path):
        if output == protected.resolve() or output in protected.resolve().parents:
            raise ValueError("P3-02 bundle destination aliases a protected input")
    preflight = preflight_replacement_supervision(
        candidate_path=candidate_path, review_path=review_path,
        contract_path=contract_path, bundle_dir=source_bundle_dir,
    )
    tokenizer, identity = _completion_tokenizer(source_bundle_dir)
    if (identity["tokenizerSha256"] != EXPECTED_TOKENIZER_SHA256
            or identity["bundleManifestSha256"] != TOKENIZER_BUNDLE_SHA256):
        raise ValueError("P3-02 requires the exact frozen local P2-48 tokenizer")
    files, plan = _split_files(tokenizer)
    if (preflight["recordTokens"] != plan["recordTokens"]
            or preflight["packedTokenSha256"] != plan["packedTokenSha256"]
            or preflight["sampler"] != plan["sampler"]):
        raise ValueError("P3-02 bundle differs from read-only supervision preflight")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".p3-02-bundle-", dir=output.parent))
    try:
        for name in TOKENIZER_FILES:
            source_file = source / name
            _safe_file(source_file, 16 * 1024 * 1024 if name == "tokenizer.json" else 1024 * 1024)
            shutil.copyfile(source_file, stage / name)
        for split, parts in files.items():
            for suffix, raw in parts.items():
                (stage / (split + "." + suffix)).write_bytes(raw)
        manifest = _manifest(files, plan, sha256_file(stage / "model-config.json"))
        (stage / "manifest.json").write_bytes((json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
        enforce_storage_limit(root, limit_bytes=storage_limit_bytes)
        inspected = inspect_replacement_bundle(stage)
        if inspected["sampler"] != preflight["sampler"]:
            raise ValueError("P3-02 published sampler does not match read-only preflight")
        if output.exists():
            raise FileExistsError("P3-02 output already exists; refusing replacement")
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {
        "schemaVersion": 1, "milestone": MILESTONE,
        "status": "training-bundle-prepared-zero-update",
        "bundleManifestSha256": sha256_file(output / "manifest.json"),
        "bundleDirectory": str(output),
        **inspected,
        **SAFE_FLAGS,
    }


def inspect_replacement_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Reconstruct and verify all persisted record/token/mask bytes; no writing."""
    if bundle_dir.is_symlink() or not bundle_dir.is_dir():
        raise ValueError("P3-02 packed bundle must be a regular directory")
    root = bundle_dir.resolve(strict=True)
    manifest = json.loads(_safe_file(root / "manifest.json").decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("P3-02 manifest must be an object")
    for key, value in {
        "schemaVersion": 1, "milestone": MILESTONE,
        "pipelineVersion": PIPELINE_VERSION, "codec": CODEC,
        "actualVocabularySize": 16384, "contextLength": 512,
        "candidateSha256": EXPECTED_SHA256,
        "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
        "sourceTokenizerBundleManifestSha256": TOKENIZER_BUNDLE_SHA256,
        **SAFE_FLAGS,
    }.items():
        if type(manifest.get(key)) is not type(value) or manifest[key] != value:
            raise ValueError("P3-02 packed bundle manifest identity/authorization differs: " + key)
    tokenizer, identity = _completion_tokenizer(root)
    if (tokenizer.vocabulary_size != 16384
            or identity["tokenizerSha256"] != EXPECTED_TOKENIZER_SHA256
            or manifest.get("modelConfigSha256") != identity["modelConfigSha256"]):
        raise ValueError("P3-02 packed bundle tokenizer/config differs")
    files, plan = _split_files(tokenizer)
    if manifest != _manifest(files, plan, identity["modelConfigSha256"]):
        raise ValueError("P3-02 packed bundle manifest/sampler differs from exact frozen records")
    for split, parts in files.items():
        for suffix, expected in parts.items():
            actual = _safe_file(root / (split + "." + suffix))
            if actual != expected:
                raise ValueError("P3-02 packed bundle has changed text, tokens, index, or answer-only mask")
    return {
        "bundleVerified": True,
        "bundleManifestSha256": sha256_file(root / "manifest.json"),
        "candidateSha256": EXPECTED_SHA256,
        "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
        "records": plan["records"], "recordTokens": plan["recordTokens"],
        "answerAndEosTargetPositions": plan["answerAndEosTargetPositions"],
        "sampler": plan["sampler"],
        "trainingPerformed": False, "researchOptimizerUpdates": 0,
    }


def preflight_replacement_checkpoint_source(
    *, checkpoint_path: Path = DEFAULT_CHECKPOINT,
    bundle_dir: Path = DEFAULT_OUTPUT,
    result_path: Path = DEFAULT_P248_RESULT,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Read-only verify fixed P2-48 step-100 endpoint against exact P3-02 bundle."""
    _contract(contract_path)
    inspected = inspect_replacement_bundle(bundle_dir)
    result = json.loads(_safe_file(result_path).decode("utf-8"))
    if (not isinstance(result, dict)
            or result.get("milestone") != "P2-48"
            or result.get("trainingPerformed") is not True
            or type(result.get("researchOptimizerUpdates")) is not int
            or result["researchOptimizerUpdates"] != 100
            or result.get("finalCheckpoint") != {
                "step": 100, "path": "step-0100.pt",
                "sha256": SOURCE_CHECKPOINT_SHA256, "bytes": SOURCE_CHECKPOINT_BYTES,
            }):
        raise ValueError("P2-48 frozen final-run source record differs")
    if (checkpoint_path.is_symlink() or not checkpoint_path.is_file()
            or checkpoint_path.stat().st_size != SOURCE_CHECKPOINT_BYTES):
        raise ValueError("P3-02 requires the exact local P2-48 step-100 checkpoint file")
    if sha256_file(checkpoint_path) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("P3-02 requires the exact P2-48 step-100 checkpoint SHA-256")
    model, payload = read_checkpoint(checkpoint_path, torch.device("cpu"))
    transition = payload.get("stageTransitionRecord")
    settings = payload.get("trainingSettings")
    tokenizer = payload.get("tokenizerRecord")
    initialization = payload.get("initializationRecord")
    if (model.config != DEFAULT_CONFIG
            or parameter_count(model) != 27566080
            or type(payload.get("step")) is not int or payload["step"] != 100
            or type(payload.get("tokensProcessedTotal")) is not int
            or payload["tokensProcessedTotal"] != 429375
            or payload.get("seed") != 1337
            or payload.get("codec") != CODEC
            or not isinstance(transition, dict)
            or transition.get("kind") != "plex-request-grounded-coding-stage-transition-v1"
            or transition.get("milestone") != "P2-48"
            or not isinstance(settings, dict)
            or settings.get("kind") != "p2-48-authorized-request-grounded-training-v1"
            or not isinstance(initialization, dict)
            or initialization.get("pretrainedCheckpointLoaded") is not False
            or initialization.get("pretrainedModelWeightsLoaded") is not False
            or not isinstance(tokenizer, dict)
            or tokenizer.get("tokenizerSha256") != EXPECTED_TOKENIZER_SHA256
            or tokenizer.get("bundleManifestSha256") != TOKENIZER_BUNDLE_SHA256
            or tokenizer.get("actualVocabularySize") != 16384):
        raise ValueError("P3-02 P2-48 source provenance/model is not the approved endpoint")
    return {
        "schemaVersion": 1, "milestone": MILESTONE,
        "status": "checkpoint-source-read-only-preflight-passed",
        "baseCheckpointSha256": SOURCE_CHECKPOINT_SHA256,
        "baseCheckpointStep": 100, "baseTokensProcessed": 429375,
        "baseParameterCount": parameter_count(model),
        "bundleManifestSha256": inspected["bundleManifestSha256"],
        "bundleCompatibilityVerified": True,
        "checkpointStaged": False, "optimizerCreated": False,
        **SAFE_FLAGS,
    }
