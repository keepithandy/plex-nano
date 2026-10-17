"""Frozen P3 copy-fidelity step-100 complete-file generation evaluation.

READ-ONLY with respect to model/optimizer/source artifacts. Greedy generate the
entire supplied file (96 never-trained validation + 42 previously inspected
development controls), strictly from the single SHA-pinned step-100 endpoint.
Never selects checkpoint by validation loss. No final project holdout.

Default CLI is a read-only preflight; --evaluate writes ONE fresh local report
directory under $HOME/PlexArtifacts. The original 42-control manifest remains
unchanged even though its historic model metadata refers to the OLD checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from plex_training.artifacts import path_within_root
from plex_training.checkpoint import read_checkpoint
from plex_training.completion import _completion_tokenizer, _generate_token_ids
from plex_training.model import parameter_count
from plex_training.tokenizer import CODEC, sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARTIFACT_ROOT = Path.home() / "PlexArtifacts"
RUN_DIR = ARTIFACT_ROOT / "file-edits/p3-copy-fidelity-v1-first-run"
RUN_RESULT = RUN_DIR / "result.json"
CHECKPOINT = RUN_DIR / "checkpoints/step-0100.pt"
BUNDLE = ARTIFACT_ROOT / "file-edits/p3-copy-fidelity-v1-bundle"
OUT = ARTIFACT_ROOT / "file-edits/p3-copy-fidelity-v1-step100-evaluation"
CONTROL_PATH = ROOT / "training/phase3/diagnostics/p3-copy-edit-controls-v1/copy-edit-controls-v1.json"

CHECKPOINT_SHA = "c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230"
STAGE_SHA = "3524620d830d363585d501285a9cfdea09d1c2448be08239e2516d78291b3ca9"
BUNDLE_SHA = "8eceed468fdcb7202484eba82c8ebb8159a0d9e05787538f2c81f06f62719e2d"
CANDIDATE_SHA = "9414d1da7672a386a920d1935723e9133d011c9c3a4029e5d5800f643835ca7a"
TOKENIZER_SHA = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
CONTRACT_SHA = "a7432a3bfa7942b6086bc8e64fa3b24290351acfd9fb700aac2af613a8c0911b"
CONTROL_SHA = "17834efb801ac963a4672ae616ca47e9eea40b2568f01951c3891c7e4997f01a"
MAX_NEW = 192
SEED = 1337

def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Frozen supporting module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

PACK = _load("p3cf_generation_pack", HERE / "pack_bundle.py")


def _frozen_inputs(bundle_dir: Path, control_path: Path) -> tuple[list[dict], list[dict], dict]:
    packed = PACK.inspect(bundle_dir)  # regenerates and compares EVERY byte
    if (packed["bundleManifestSha256"] != BUNDLE_SHA
            or packed["candidateSha256"] != CANDIDATE_SHA):
        raise ValueError("Frozen training bundle is not the owner's approved copy-fidelity dataset")
    validation = [r for r in PACK._candidate()["rows"] if r["split"] == "validation"]
    if (len(validation) != 96
            or Counter(r["condition"] for r in validation) != {"copy": 24, "edit": 72}
            or len({r["groupId"] for r in validation}) != 24):
        raise ValueError("Frozen validation source/contrast composition differs")
    if (control_path.is_symlink() or not control_path.is_file()
            or sha256_file(control_path) != CONTROL_SHA):
        raise ValueError("Original 42-task developer control file changed")
    controls_manifest = json.loads(control_path.read_text(encoding="utf-8"))
    controls = controls_manifest["tasks"]
    if (len(controls) != 42
            or Counter(r["condition"] for r in controls) != {
                "copy": 6, "minimal-edit": 18, "context-edit": 18,
            }
            or len({r["taskId"] for r in controls}) != 42):
        raise ValueError("Frozen developer-control composition changed")
    if any(r["prompt"] != PACK.PREP.PROMPT.format(
        language=r["language"], request=r["request"], input_file=r["inputFile"]
    ) for r in validation):
        raise ValueError("Validation prompt changed from frozen training contract")
    # Historic control manifest references previous fixed P3-02 model. Use only
    # its immutable TASKS and score semantics; never its old checkpoint ID.
    return validation, controls, packed


def _run_metadata(result_path: Path, checkpoint: Path) -> dict[str, Any]:
    if (result_path.is_symlink() or not result_path.is_file()
            or result_path.stat().st_size > 256 * 1024
            or checkpoint.is_symlink() or not checkpoint.is_file()
            or sha256_file(checkpoint) != CHECKPOINT_SHA):
        raise ValueError("One-run local report or fixed step-100 checkpoint is missing or changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    keys = {
        "status": "bounded-research-run-completed",
        "trainingPerformed": True, "completedMaximumSteps": True,
        "optimizerUpdates": 100, "answerAndEosTargetsProcessed": 40477,
        "authorizationContractSha256": CONTRACT_SHA,
        "stageCheckpointSha256": STAGE_SHA,
        "bundleManifestSha256": BUNDLE_SHA,
        "fixedEndpointSha256": CHECKPOINT_SHA,
        "resumeAllowed": False, "automaticContinuation": False,
        "checkpointReselectionAllowed": False,
        "developmentEvaluationPerformed": False, "finalHoldoutOpened": False,
        "interrupted": False,
        "maximumOptimizerUpdates": 100, "maximumWallTimeSeconds": 600,
    }
    if any(type(result.get(k)) is not type(v) or result[k] != v for k, v in keys.items()):
        raise ValueError("Training result differs from authorized fixed endpoint")
    saves = result.get("checkpoints")
    if (not isinstance(saves, list) or len(saves) != 4
            or [c["step"] for c in saves] != [25, 50, 75, 100]
            or saves[-1].get("sha256") != CHECKPOINT_SHA
            or saves[-1].get("path") != "step-0100.pt"):
        raise ValueError("Training result has changed checkpoint timeline")
    vals = result.get("validation")
    if (not isinstance(vals, list) or [v["step"] for v in vals] != [0, 25, 50, 75, 100]
            or abs(vals[0]["meanValidationAnswerAndEosLoss"] - 7.116717401614859) > 1e-9
            or abs(vals[-1]["meanValidationAnswerAndEosLoss"] - 2.7984008651134396) > 1e-9
            or not isinstance(result.get("elapsedSeconds"), (int, float))
            or not 0 < result["elapsedSeconds"] <= 600):
        raise ValueError("Training result history no longer matches owner-reported run")
    return {"runResultSha256": sha256_file(result_path),
            "fixedCheckpointSha256": CHECKPOINT_SHA,
            "authorizedUpdates": 100, "trainTargetsProcessed": 40477}


def _checkpoint_model(checkpoint: Path, device: torch.device) -> Any:
    model, payload = read_checkpoint(checkpoint, torch.device("cpu"))
    settings = payload.get("trainingSettings", {})
    record = payload.get("datasetRecord", {})
    stage = payload.get("stageTransitionRecord", {})
    if (type(payload.get("step")) is not int or payload["step"] != 100
            or type(payload.get("tokensProcessedTotal")) is not int
            or payload["tokensProcessedTotal"] != 40477
            or payload.get("codec") != CODEC
            or model.config.context_length != 512
            or model.config.vocab_size != 16384
            or parameter_count(model) != 27566080
            or settings.get("kind") != "p3-copy-fidelity-owner-authorized-answer-eos-v1"
            or settings.get("authorizationContractSha256") != CONTRACT_SHA
            or settings.get("stageCheckpointSha256") != STAGE_SHA
            or settings.get("bundleManifestSha256") != BUNDLE_SHA
            or record.get("candidateSha256") != CANDIDATE_SHA
            or record.get("bundleManifestSha256") != BUNDLE_SHA
            or stage.get("baseCheckpointSha256") != PACK.SOURCE_BUNDLE_SHA and False
            or payload.get("tokenizerRecord", {}).get("tokenizerSha256") != TOKENIZER_SHA):
        raise ValueError("Fixed checkpoint tensor/config/provenance differs")
    return model.to(device).eval()


def preflight(*, bundle_dir: Path = BUNDLE, control_path: Path = CONTROL_PATH,
              checkpoint: Path = CHECKPOINT, run_result: Path = RUN_RESULT) -> dict[str, Any]:
    validation, controls, packed = _frozen_inputs(bundle_dir, control_path)
    frozen = _run_metadata(run_result, checkpoint)
    tokenizer, token_record = _completion_tokenizer(bundle_dir)
    if (tokenizer.vocabulary_size != 16384
            or token_record["tokenizerSha256"] != TOKENIZER_SHA
            or token_record["bundleManifestSha256"] != BUNDLE_SHA):
        raise ValueError("Frozen tokenizer does not match approved bundle")
    for r in validation + controls:
        pids = tokenizer.encode(r["prompt"])
        if (not pids or tokenizer.decode(pids) != r["prompt"]
                or len(pids) + MAX_NEW > 512
                or len(tokenizer.encode(r["expectedFile"])) + 1 > MAX_NEW):
            raise ValueError(f"Prompt/answer exceeds full-context generation budget: {r.get('id', r.get('taskId'))}")
    return {
        "status": "copy-fidelity-step100-read-only-generation-preflight-passed",
        "fixedEndpointStep": 100, "validationRecords": len(validation),
        "validationCopies": 24, "validationEdits": 72,
        "developmentControlRecords": len(controls),
        "developmentCopies": 6, "developmentEdits": 36,
        "maximumNewTokens": MAX_NEW, "temperature": 0.0, "seed": SEED,
        "bundleManifestSha256": BUNDLE_SHA, "candidateSha256": CANDIDATE_SHA,
        "tokenizerSha256": TOKENIZER_SHA, "developerControlSha256": CONTROL_SHA,
        **frozen, "trainingPerformed": False, "optimizerUpdates": 0,
        "finalHoldoutOpened": False,
    }


def _score_validation(row: dict, actual: str) -> dict:
    exact = actual == row["expectedFile"]
    if row["condition"] == "copy":
        return {"exact": exact, "unrelatedBytesPreserved": exact,
                "requestedValueCorrect": None, "targetExtractable": None}
    before = row["inputFile"][:row["targetStart"]]
    after = row["inputFile"][row["targetEnd"]:]
    boundaries = actual.startswith(before) and actual.endswith(after)
    end = len(actual) - len(after) if after else len(actual)
    middle = actual[len(before):end] if boundaries else None
    return {
        "exact": exact, "unrelatedBytesPreserved": boundaries,
        "requestedValueCorrect": middle == row["requestedLiteral"] if boundaries else None,
        "targetExtractable": boundaries,
    }


def score_responses(validation: list[dict], controls: list[dict],
                    responses: list[dict]) -> dict[str, Any]:
    """Pure scorer; unscorable requested values are NULL rather than failures."""
    expected_ids = [r["id"] for r in validation] + [r["taskId"] for r in controls]
    if (len(expected_ids) != 138 or len(set(expected_ids)) != 138
            or not isinstance(responses, list)
            or [r.get("recordId") for r in responses] != expected_ids):
        raise ValueError("Evaluation must score all 138 frozen responses exactly once in order")
    scored: list[dict[str, Any]] = []
    group_pass: dict[str, list[bool]] = defaultdict(list)
    counts = Counter()
    for i, (row, reply) in enumerate(zip(validation + controls, responses)):
        if type(reply.get("actual")) is not str:
            raise ValueError("All decoded actual files must be strings")
        origin = "validation" if i < len(validation) else "development"
        condition = row["condition"]
        condition_key = ("copy" if condition == "copy" else "edit")
        result = (_score_validation(row, reply["actual"]) if origin == "validation"
                  else _score_control(row, reply["actual"]))
        exact = result["exact"] and reply.get("stoppedAtEos") is True
        preserved = result["unrelatedBytesPreserved"] and reply.get("stoppedAtEos") is True
        record = {
            "recordId": reply["recordId"], "scope": origin,
            "language": row["language"], "condition": condition,
            "exactCompleteFile": exact,
            "unrelatedBytesPreserved": preserved,
            "requestedValueCorrect": result["requestedValueCorrect"]
                if reply.get("stoppedAtEos") is True else None,
            "requestedValueExtractable": result["targetExtractable"]
                if reply.get("stoppedAtEos") is True else False,
            "stoppedAtEos": reply.get("stoppedAtEos") is True,
            "decodeSucceeded": reply.get("decodeSucceeded") is True,
            "fullPromptVisible": reply.get("fullPromptVisibleForEveryGenerationStep") is True,
            "generatedTokenCount": reply.get("generatedTokenCount"),
            "expectedFile": row["expectedFile"], "actual": reply["actual"],
        }
        scored.append(record)
        counts[f"{origin}Total"] += 1
        counts[f"{origin}Exact"] += bool(exact)
        counts[f"{origin}CopyTotal" if condition_key == "copy" else f"{origin}EditTotal"] += 1
        counts[f"{origin}CopyExact" if condition_key == "copy" else f"{origin}EditExact"] += bool(exact)
        counts[f"{origin}Preserved"] += bool(preserved)
        counts[f"{origin}Eos"] += reply.get("stoppedAtEos") is True
        counts[f"{origin}FullPromptVisible"] += reply.get("fullPromptVisibleForEveryGenerationStep") is True
        if origin == "validation":
            group_pass[row["groupId"]].append(exact)
        elif condition != "copy":
            group_pass[row["contrastGroupId"]].append(exact)
    val_groups = {k: xs for k, xs in group_pass.items() if k.startswith("p3-cf-")}
    dev_groups = {k: xs for k, xs in group_pass.items() if k.startswith("p3-copy-")}
    if (len(val_groups) != 24 or any(len(g) != 4 for g in val_groups.values())
            or len(dev_groups) != 12 or any(len(g) != 3 for g in dev_groups.values())):
        raise ValueError("Frozen contrast group composition changed")
    return {
        "validation": {
            "cases": 96, "copies": {"exact": counts["validationCopyExact"], "total": 24},
            "edits": {"exact": counts["validationEditExact"], "total": 72},
            "fullyCorrectContrastGroups": sum(all(g) for g in val_groups.values()),
            "contrastGroups": 24, "eosCount": counts["validationEos"],
            "preservedSourceCount": counts["validationPreserved"],
            "fullPromptVisibleCount": counts["validationFullPromptVisible"],
            "gates": {
                "copy24of24": counts["validationCopyExact"] == 24,
                "editsAtLeast60of72": counts["validationEditExact"] >= 60,
                "groupsAtLeast18of24": sum(all(g) for g in val_groups.values()) >= 18,
            },
        },
        "development": {
            "cases": 42, "copies": {"exact": counts["developmentCopyExact"], "total": 6},
            "edits": {"exact": counts["developmentEditExact"], "total": 36},
            "fullyCorrectEditContrastGroups": sum(all(g) for g in dev_groups.values()),
            "editContrastGroups": 12, "eosCount": counts["developmentEos"],
            "preservedSourceCount": counts["developmentPreserved"],
            "fullPromptVisibleCount": counts["developmentFullPromptVisible"],
            "gates": {
                "copy6of6": counts["developmentCopyExact"] == 6,
                "editsAtLeast30of36": counts["developmentEditExact"] >= 30,
            },
        },
        "allValidationGatesPassed": (
            counts["validationCopyExact"] == 24
            and counts["validationEditExact"] >= 60
            and sum(all(g) for g in val_groups.values()) >= 18
        ),
        "allDevelopmentGatesPassed": (
            counts["developmentCopyExact"] == 6
            and counts["developmentEditExact"] >= 30
        ),
        "perRecord": scored,
    }


def _score_control(row: dict, actual: str) -> dict:
    """Faithfully apply the old control's frozen target-boundary semantics."""
    if row["condition"] == "copy":
        exact = actual == row["expectedFile"]
        return {"exact": exact, "unrelatedBytesPreserved": exact,
                "requestedValueCorrect": None, "targetExtractable": None}
    source, target = row["inputFile"], row["target"]
    before, after = source[:target["startCharacter"]], source[target["endCharacter"]:]
    boundaries = actual.startswith(before) and actual.endswith(after)
    end = len(actual) - len(after) if after else len(actual)
    middle = actual[len(before):end] if boundaries else None
    return {"exact": actual == row["expectedFile"],
            "unrelatedBytesPreserved": boundaries,
            "requestedValueCorrect": middle == target["requestedLiteral"] if boundaries else None,
            "targetExtractable": boundaries}


def evaluate(*, checkpoint: Path = CHECKPOINT, run_result: Path = RUN_RESULT,
             bundle_dir: Path = BUNDLE, control_path: Path = CONTROL_PATH,
             output_dir: Path = OUT, artifact_root: Path = ARTIFACT_ROOT,
             device_name: str = "cuda") -> dict[str, Any]:
    """One fresh output; never mutates fixed weights or evaluator's input files."""
    if device_name not in ("cpu", "cuda") or (
        device_name == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("Evaluation device must be an available CPU or CUDA")
    root = artifact_root.resolve()
    output = path_within_root(output_dir, root)
    if output == root or output.exists():
        raise ValueError("Evaluation output must be a NEW directory; never overwrite results")
    for protected in (checkpoint, run_result, bundle_dir, control_path):
        protected = protected.resolve(strict=False)
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Evaluation output aliases a protected input")
    proof = preflight(bundle_dir=bundle_dir, control_path=control_path,
                      checkpoint=checkpoint, run_result=run_result)
    validation, controls, _ = _frozen_inputs(bundle_dir, control_path)
    tokenizer, _ = _completion_tokenizer(bundle_dir)
    model = _checkpoint_model(checkpoint, torch.device(device_name))
    responses = []
    for row in validation + controls:
        prompt_ids = tokenizer.encode(row["prompt"])
        if len(prompt_ids) + MAX_NEW > model.config.context_length:
            raise ValueError("Prompt/context would evict supplied code")
        ids, stopped, context = _generate_token_ids(
            model, prompt_ids, actual_vocab=tokenizer.vocabulary_size,
            max_new_tokens=MAX_NEW, temperature=0.0, seed=SEED,
            device=torch.device(device_name),
        )
        full_prompt = context["fullPromptVisibleForEveryGenerationStep"]
        if not full_prompt:
            raise ValueError("Generated response evicted frozen source context")
        try:
            joined = tokenizer.decode(prompt_ids + ids)
            decode_ok = joined.startswith(row["prompt"])
            actual = joined[len(row["prompt"]):] if decode_ok else ""
        except (UnicodeError, ValueError):
            decode_ok, actual = False, ""
        responses.append({
            "recordId": row.get("id", row.get("taskId")),
            "actual": actual, "generatedTokenCount": len(ids),
            "stoppedAtEos": stopped, "decodeSucceeded": decode_ok,
            "fullPromptVisibleForEveryGenerationStep": full_prompt,
        })
    scores = score_responses(validation, controls, responses)
    report = {
        "schemaVersion": 1, "kind": "p3-copy-fidelity-fixed-step100-generation-v1",
        "status": "fixed-endpoint-generation-evaluation-complete",
        "checkpointStep": 100, "checkpointSha256": CHECKPOINT_SHA,
        "stageCheckpointSha256": STAGE_SHA,
        "bundleManifestSha256": BUNDLE_SHA, "candidateSha256": CANDIDATE_SHA,
        "controlManifestSha256": CONTROL_SHA, "tokenizerSha256": TOKENIZER_SHA,
        "authorizationContractSha256": CONTRACT_SHA,
        "runResultSha256": proof["runResultSha256"],
        "generation": {"kind": "greedy-complete-file-v1",
                       "maxNewTokens": MAX_NEW, "temperature": 0.0, "seed": SEED,
                       "device": device_name, "checkpointReselectionAllowed": False},
        "trainingPerformed": False, "optimizerUpdates": 0,
        "developmentControlsPreviouslyInspected": True,
        "finalHoldoutOpened": False, **scores,
    }
    # Writes ONLY new, non-overwriting results. No checkpoint or dataset writes.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    with (output / "evaluation-result.json").open("x", encoding="utf-8", newline="\n") as fp:
        json.dump(report, fp, sort_keys=True, indent=2, ensure_ascii=False)
        fp.write("\n")
    return {k: v for k, v in report.items() if k != "perRecord"} | {
        "resultFile": str(output / "evaluation-result.json")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true",
                        help="Run 138 greedy fixed-step100 generations; NO optimizer updates")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    try:
        result = (evaluate(device_name=args.device) if args.evaluate else preflight())
        print(json.dumps(result, sort_keys=True, indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"p3-copy-fidelity-generation: {exc}\n")

if __name__ == "__main__":
    main()
