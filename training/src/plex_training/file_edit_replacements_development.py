"""P3-02 fixed step-100 development-only greedy file-edit evaluation.

Read-only model and source files. No optimizer, gradients, checkpoint mutations,
model selection, training loop, or access to the final project holdout.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from .artifacts import path_within_root
from .checkpoint import read_checkpoint
from .completion import _completion_tokenizer, _generate_token_ids
from .file_conditioned_contract import _canonical_bytes
from .file_edit_replacements import (
    EXPECTED_SHA256 as CANDIDATE_SHA256, PROPOSED_RUN, ROOT,
    build_replacement_rows, render_prompt, score_replacement,
    validate_replacement_rows,
)
from .file_edit_replacements_bundle import inspect_replacement_bundle
from .file_edit_replacements_run import (
    DEFAULT_OUTPUT as FIRST_RUN_OUTPUT,
    EXPECTED_SAMPLER_POSITIONS, EXPECTED_STAGE_SHA256,
)
from .file_edit_replacements_stage import BUNDLE_SHA256
from .model import parameter_count
from .tokenizer import CODEC, sha256_file

DEFAULT_CHECKPOINT = FIRST_RUN_OUTPUT / "checkpoints/step-0100.pt"
DEFAULT_RUN_RESULT = FIRST_RUN_OUTPUT / "result.json"
DEFAULT_FROZEN_REPORT = ROOT / "training/pretraining/p3-02-first-run-result.json"
DEFAULT_BUNDLE = ROOT / "training/artifacts/file-edits/p3-02-training-bundle"
DEFAULT_OUTPUT = ROOT / "training/artifacts/file-edits/p3-02-development-evaluation"
EXPECTED_ENDPOINT_SHA256 = "020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d"
EXPECTED_BYTES = 330899611
EXPECTED_AUTHORIZATION_SHA256 = "b08e9e48f268d658e1a37f7b25720dcfe3d0cf68abb483f27e3f60476e101e03"
EXPECTED_TOKENIZER_SHA256 = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
EXPECTED_VALIDATION = (
    (0, 10.566839536031088), (25, 2.528637358455947),
    (50, 1.7545379514043982), (75, 1.6265562090006742),
    (100, 1.5892543557015333),
)
EXPECTED_CHECKPOINTS = (
    (25, "step-0025.pt", "9fbc09260b94ef697a24bab033f2a341bf4bc4bb1dca30712e92c9fab345fc40", 330899611),
    (50, "step-0050.pt", "cabb6555f4da7b8bc4107a37980b4df99f644adad120aa4fea25ab3e6cb3b7bb", 330899611),
    (75, "step-0075.pt", "d39d5f37a0cd491ba18b43cfa323f54026b143e914f19ab7563bca60b40cce48", 330899675),
    (100, "step-0100.pt", EXPECTED_ENDPOINT_SHA256, EXPECTED_BYTES),
)
EVALUATION_KIND = "p3-02-frozen-step100-development-exact-edit-v1"
LANGUAGES = ("html", "css", "javascript")


def _json(path: Path, maximum_bytes: int = 64 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        raise ValueError("P3-02 run result must be a bounded regular JSON file")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("P3-02 run result must be a JSON object")
    return obj


def _frozen_run_report(
    *, local_result: Path, frozen_report: Path, checkpoint: Path,
) -> dict[str, Any]:
    """Compare the complete recorded endpoint identity, not a newly chosen checkpoint."""
    local = _json(local_result)
    frozen = _json(frozen_report)
    for field, expected in {
        "schemaVersion": 1, "milestone": "P3-02",
        "kind": "p3-02-owner-reported-first-bounded-answer-eos-run-result-v1",
        "trainingPerformed": True, "researchOptimizerUpdates": 100,
        "completedMaximumSteps": True, "maximumOptimizerUpdates": 100,
        "maximumWallTimeSeconds": 600, "answerAndEosTargetsProcessed": 60445,
        "interrupted": False, "resumeAllowed": False,
        "automaticContinuation": False, "continuationAuthorized": False,
        "developmentEvaluationPerformed": False, "finalHoldoutOpened": False,
        "authorizationContractSha256": EXPECTED_AUTHORIZATION_SHA256,
        "stageCheckpointSha256": EXPECTED_STAGE_SHA256,
        "bundleManifestSha256": BUNDLE_SHA256,
    }.items():
        if type(frozen.get(field)) is not type(expected) or frozen[field] != expected:
            raise ValueError("P3-02 frozen first-run report changed: " + field)
    expected_validation = [
        {"step": step, "meanValidationAnswerAndEosLoss": loss}
        for step, loss in EXPECTED_VALIDATION
    ]
    expected_checkpoints = [
        {"step": step, "path": name, "sha256": digest, "bytes": size}
        for step, name, digest, size in EXPECTED_CHECKPOINTS
    ]
    for field, expected in {
        "validation": expected_validation, "checkpoints": expected_checkpoints,
        "fixedEndpoint": {
            "step": 100, "path": "file-edits/p3-02-first-run/checkpoints/step-0100.pt",
            "sha256": EXPECTED_ENDPOINT_SHA256, "checkpointReselectionAllowed": False,
        },
        "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
    }.items():
        if frozen.get(field) != expected:
            raise ValueError("P3-02 frozen first-run artifact record changed: " + field)

    # Local report uses the runner's schema, which differs from the evidence snapshot.
    for field, expected in {
        "schemaVersion": 1, "milestone": "P3-02",
        "kind": "p3-02-first-bounded-answer-eos-run-result-v1",
        "trainingPerformed": True, "researchOptimizerUpdates": 100,
        "completedMaximumSteps": True, "maximumOptimizerUpdates": 100,
        "maximumWallTimeSeconds": 600, "answerAndEosTargetsProcessed": 60445,
        "interrupted": False, "resumeAllowed": False,
        "automaticContinuation": False, "continuationAuthorized": False,
        "developmentEvaluationPerformed": False, "finalHoldoutOpened": False,
        "authorizationContractSha256": EXPECTED_AUTHORIZATION_SHA256,
        "stageCheckpointSha256": EXPECTED_STAGE_SHA256,
        "bundleManifestSha256": BUNDLE_SHA256,
        "fixedEndpointStep": 100, "fixedEndpointSha256": EXPECTED_ENDPOINT_SHA256,
        "validation": expected_validation, "checkpoints": expected_checkpoints,
    }.items():
        if type(local.get(field)) is not type(expected) or local[field] != expected:
            raise ValueError("P3-02 local training report differs from frozen result: " + field)
    if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size != EXPECTED_BYTES:
        raise ValueError("P3-02 fixed endpoint must be the original regular step-0100.pt")
    if checkpoint.name != "step-0100.pt" or checkpoint.parent.name != "checkpoints":
        raise ValueError("P3-02 evaluator requires the named fixed step-100 checkpoint")
    if checkpoint.resolve(strict=True) != (local_result.resolve(strict=True).parent / "checkpoints/step-0100.pt").resolve(strict=True):
        raise ValueError("P3-02 checkpoint does not belong to the frozen source run")
    if sha256_file(checkpoint) != EXPECTED_ENDPOINT_SHA256:
        raise ValueError("P3-02 fixed endpoint SHA-256 changed")
    return frozen


def _development_rows() -> list[dict[str, Any]]:
    """Only the frozen P3-02 development split; never the final project holdout."""
    rows = build_replacement_rows()
    validate_replacement_rows(rows)
    import hashlib
    if hashlib.sha256(_canonical_bytes(rows)).hexdigest() != CANDIDATE_SHA256:
        raise ValueError("P3-02 candidate identity changed before development evaluation")
    development = [r for r in rows if r["candidateSplit"] == "development"]
    if (len(development) != 36 or
            Counter(r["language"] for r in development) != dict.fromkeys(LANGUAGES, 12) or
            len({r["contrastGroupId"] for r in development}) != 12 or
            any(r["id"].split("-")[-2] != "05" for r in development)):
        raise ValueError("P3-02 development split does not contain frozen 36 tasks")
    return development


def score_development_responses(responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure scoring; never claims training quality from token loss alone."""
    frozen = {row["id"]: row for row in _development_rows()}
    if (not isinstance(responses, list) or len(responses) != 36
            or any(not isinstance(r, dict) for r in responses)):
        raise ValueError("P3-02 development evaluation requires exactly 36 responses")
    seen: set[str] = set()
    counts = Counter()
    per_language = defaultdict(Counter)
    per_family = defaultdict(Counter)
    group_scores = defaultdict(list)
    audits = []
    for response in responses:
        ident = response.get("recordId")
        if not isinstance(ident, str) or ident not in frozen or ident in seen:
            raise ValueError("P3-02 development record ID is missing, unexpected or duplicated")
        seen.add(ident)
        row = frozen[ident]
        actual = response.get("actual")
        if type(actual) is not str:
            raise ValueError("P3-02 development response must include raw decoded output")
        for key in ("stoppedAtEos", "fullPromptVisibleForEveryGenerationStep", "decodeSucceeded"):
            if type(response.get(key)) is not bool:
                raise ValueError("P3-02 development response flag is not a boolean: " + key)
        if (type(response.get("generatedTokenCount")) is not int
                or not 0 <= response["generatedTokenCount"] <= 192):
            raise ValueError("P3-02 development response has an invalid token count")
        result = score_replacement(row, actual)
        if response["decodeSucceeded"] is False:
            result = {**result, "exactMatch": False, "requestedEditCorrect": False,
                      "unrelatedCodePreserved": False, "unnecessaryEdits": True,
                      "syntaxCheckability": "not-checked"}
        exact, preserved = result["exactMatch"], result["unrelatedCodePreserved"]
        fully_visible = response["fullPromptVisibleForEveryGenerationStep"]
        for key, value in (
            ("exact", exact), ("preserved", preserved),
            ("fullPromptVisible", fully_visible),
            ("stoppedAtEos", response["stoppedAtEos"]),
            ("decodeSucceeded", response["decodeSucceeded"]),
        ):
            counts[key] += int(value)
            per_language[row["language"]][key] += int(value)
            per_family[(row["language"], row["editFamily"])][key] += int(value)
        group_scores[row["contrastGroupId"]].append(exact)
        audits.append({
            "recordId": ident, "language": row["language"],
            "editFamily": row["editFamily"], "contrastGroupId": row["contrastGroupId"],
            "request": row["request"], "inputFile": row["inputFile"],
            "expectedFile": row["expectedFile"], "actual": actual,
            "generatedTokenCount": response["generatedTokenCount"],
            "stoppedAtEos": response["stoppedAtEos"],
            "fullPromptVisibleForEveryGenerationStep": fully_visible,
            "decodeSucceeded": response["decodeSucceeded"],
            **result,
        })
    if seen != set(frozen) or len(group_scores) != 12 or any(len(v) != 3 for v in group_scores.values()):
        raise ValueError("P3-02 development groups are incomplete")
    all_correct = sum(all(results) for results in group_scores.values())
    thresholds = PROPOSED_RUN["developmentGate"]
    assert thresholds["tasks"] == 36 and thresholds["contrastGroups"] == 12
    gate = {
        "overallExact": counts["exact"] >= thresholds["minimumExact"],
        "perLanguageExact": all(
            per_language[lang]["exact"] >= thresholds["minimumExactPerLanguage"]
            for lang in LANGUAGES
        ),
        "unrelatedPreserved": counts["preserved"] >= thresholds["minimumPreserved"],
        "allThreeCorrectContrastGroups": all_correct >= thresholds["minimumAllCorrectContrastGroups"],
        "fullPromptVisibleEveryStep": counts["fullPromptVisible"] == 36,
    }
    return {
        "totalRecords": 36,
        "exactEdits": counts["exact"],
        "unrelatedCodePreserved": counts["preserved"],
        "fullPromptVisibleRecords": counts["fullPromptVisible"],
        "eosTerminatedRecords": counts["stoppedAtEos"],
        "decodeSucceededRecords": counts["decodeSucceeded"],
        "allThreeCorrectContrastGroups": all_correct,
        "perLanguage": {
            lang: {"records": 12, "exactEdits": per_language[lang]["exact"],
                   "unrelatedCodePreserved": per_language[lang]["preserved"]}
            for lang in LANGUAGES
        },
        "perFamily": {
            f"{language}/{family}": {"records": 3, "exactEdits": value["exact"],
                                     "unrelatedCodePreserved": value["preserved"]}
            for (language, family), value in sorted(per_family.items())
        },
        "gateChecks": gate,
        "developmentGatePassed": all(gate.values()),
        "perRecord": audits,
    }


def evaluate_replacement_development(
    *, checkpoint: Path = DEFAULT_CHECKPOINT,
    run_result: Path = DEFAULT_RUN_RESULT,
    frozen_report: Path = DEFAULT_FROZEN_REPORT,
    bundle_dir: Path = DEFAULT_BUNDLE,
    output_dir: Path = DEFAULT_OUTPUT,
    artifact_root: Path = ROOT / "training/artifacts",
    device: str = "cpu",
) -> dict[str, Any]:
    """One fresh report from frozen endpoint, all 36 existing development tasks."""
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("P3-02 development device must be available cpu/cuda")
    root = artifact_root.resolve()
    output = path_within_root(output_dir, root)
    if output == root or output.exists():
        raise ValueError("P3-02 development output directory must be fresh")
    protected = (checkpoint, run_result, frozen_report, bundle_dir)
    if any(output == p.resolve(strict=False) or
           output in p.resolve(strict=False).parents or
           p.resolve(strict=False) in output.parents for p in protected):
        raise ValueError("P3-02 development output must not overlap protected inputs")
    frozen = _frozen_run_report(
        local_result=run_result, frozen_report=frozen_report, checkpoint=checkpoint,
    )
    inspected = inspect_replacement_bundle(bundle_dir)
    if inspected["bundleManifestSha256"] != BUNDLE_SHA256:
        raise ValueError("P3-02 frozen tokenizer/train-validation bundle has changed")
    tokenizer, tokenizer_record = _completion_tokenizer(bundle_dir)
    if (tokenizer_record["tokenizerSha256"] != EXPECTED_TOKENIZER_SHA256
            or tokenizer.vocabulary_size != 16384):
        raise ValueError("P3-02 frozen tokenizer identity differs")
    model, payload = read_checkpoint(checkpoint, torch.device("cpu"))
    if (payload.get("codec") != CODEC
            or type(payload.get("step")) is not int or payload["step"] != 100
            or type(payload.get("tokensProcessedTotal")) is not int
            or payload["tokensProcessedTotal"] != EXPECTED_SAMPLER_POSITIONS
            or parameter_count(model) != 27566080
            or model.config.context_length != 512 or model.config.vocab_size != 16384
            or not isinstance(payload.get("tokenizerRecord"), dict)
            or payload["tokenizerRecord"].get("tokenizerSha256") != EXPECTED_TOKENIZER_SHA256
            or not isinstance(payload.get("initializationRecord"), dict)
            or payload["initializationRecord"].get("pretrainedCheckpointLoaded") is not False
            or not isinstance(payload.get("stageTransitionRecord"), dict)
            or payload["stageTransitionRecord"].get("baseCheckpointSha256") != frozen["sourceCheckpointSha256"]
            or not isinstance(payload.get("datasetRecord"), dict)
            or payload["datasetRecord"].get("bundleManifestSha256") != BUNDLE_SHA256
            or not isinstance(payload.get("trainingSettings"), dict)
            or payload["trainingSettings"].get("authorizationContractSha256") != EXPECTED_AUTHORIZATION_SHA256
            or payload["trainingSettings"].get("stageCheckpointSha256") != EXPECTED_STAGE_SHA256
            or payload["trainingSettings"].get("kind") != "p3-02-authorized-answer-eos-training-v1"):
        raise ValueError("P3-02 checkpoint model/state does not match frozen training provenance")
    model = model.to(torch.device(device))
    model.eval()
    development = _development_rows()
    rows: list[dict[str, Any]] = []
    for row in development:
        prompt = render_prompt(row)
        prompt_ids = tokenizer.encode(prompt)
        # No sliding-window eviction is allowed at any step of this experiment.
        if len(prompt_ids) + 192 > model.config.context_length:
            raise ValueError("P3-02 complete prompt + 192-token budget exceeds 512 context")
        generated, stopped, context = _generate_token_ids(
            model, prompt_ids, actual_vocab=tokenizer.vocabulary_size,
            max_new_tokens=192, temperature=0.0, seed=1337,
            device=torch.device(device),
        )
        if context["promptTokenCount"] != len(prompt_ids):
            raise ValueError("P3-02 generation prompt coverage could not be verified")
        decoded = ""
        decode_ok = True
        try:
            entire = tokenizer.decode(prompt_ids + generated)
            if not entire.startswith(prompt):
                decode_ok = False
            else:
                decoded = entire[len(prompt):]
        except (UnicodeError, ValueError):
            decode_ok = False
        rows.append({
            "recordId": row["id"], "actual": decoded if decode_ok else "",
            "generatedTokenCount": len(generated),
            "stoppedAtEos": stopped,
            "decodeSucceeded": decode_ok,
            "fullPromptVisibleForEveryGenerationStep":
                context["fullPromptVisibleForEveryGenerationStep"],
        })
    scores = score_development_responses(rows)
    report = {
        "schemaVersion": 1, "milestone": "P3-02",
        "kind": EVALUATION_KIND,
        "status": "development-evaluation-complete",
        "sourceEvidence": "owner-reported-first-run-plus-local-frozen-checkpoint-verified",
        "checkpointStep": 100, "checkpointSha256": EXPECTED_ENDPOINT_SHA256,
        "bundleManifestSha256": BUNDLE_SHA256,
        "candidateSha256": CANDIDATE_SHA256,
        "tokenizerSha256": EXPECTED_TOKENIZER_SHA256,
        "authorizationContractSha256": EXPECTED_AUTHORIZATION_SHA256,
        "generation": {"temperature": 0.0, "seed": 1337,
                       "maximumNewTokens": 192,
                       "policy": "greedy-unreselected-fixed-endpoint-v1"},
        "researchOptimizerUpdatesPerformed": 0,
        "trainingPerformedDuringEvaluation": False,
        "automaticContinuation": False,
        "finalProjectHoldoutOpened": False,
        **scores,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    with (output / "development-result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return {
        key: value for key, value in report.items()
        if key not in ("perRecord", "perFamily")
    } | {"resultFile": str(output / "development-result.json")}
