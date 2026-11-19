"""M11 bounded training WORKER (never executes on import or default CLI).

All real training requires:
  1. --execute-approved-run and complete exact paths
  2. a distinct M11, four-case, two-decoder M9 baseline
  3. a separately owner-created, source/hash/arm-bound external consent file
  4. a new output directory, sufficient storage, pinned model and tokenizer
  5. re-admission in an isolated, hard-timeout supervised child

This tests source-conditioned copying, NOT general coding proficiency.
No M11 training has been authorized or executed by the repository change.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from . import plex600_m11_admission as gate
from . import plex600_m11_source_conditioning as m11
from . import plex600_m11_training_core as core
from . import plex600_m10_worker as m10
from . import plex600_m8_eval as m8
from . import plex600_m7 as m7
from . import plex600_feasibility as hw
from . import plex600_research as research

PROTOCOL = "plex600-m11-approved-source-conditioning-worker-v1"
CONTROL = "PLEX600_M11_WORKER_NONCE"
HARD_TIMEOUT = 3600
SNAPSHOTS = core.CHECKPOINT_STEPS
DECODERS = core.DECODERS
SEED = 41
MAX_WEIGHTS = gate.CHECKPOINT_MAX_BYTES
PATH_FIELDS = (
    "bundle_dir", "m9_report", "m9_weights", "baseline_report",
    "baseline_responses", "approval_file", "output_dir",
)


class Refusal(ValueError):
    pass


def plan() -> dict[str, Any]:
    """Inert plan; does not touch checkpoints, GPU, optimizer or owner consent."""
    return {
        "protocol": PROTOCOL,
        "status": "plan-only-no-training",
        "source": "independently-verified-original-M9",
        "baseline": "distinct-M11-four-space-prompt-two-decoder-M9",
        "trainingRecordIds": [c["id"] for c in m11.cases() if c["split"] == "train"],
        "developmentOnlyRecordIds": [
            c["id"] for c in m11.cases() if c["split"] == "probe-only"],
        "maxOptimizerUpdatesPerArm": core.MAX_UPDATES,
        "checkpointSteps": list(SNAPSHOTS),
        "eosLossWeights": list(core.EOS_WEIGHTS),
        "chosenEosLossWeight": None,
        "maxNewTokens": m11.MAX_GENERATED,
        "hardTimeoutSeconds": HARD_TIMEOUT,
        "freshOptimizer": "AdamW-lr5e-5-weight-decay0-eps1e-8",
        "separateOwnerConsentPerArmRequired": True,
        "m10ApprovalReusable": False,
        "finalHoldoutAllowed": False,
        "trainingAuthorized": False,
        "optimizerUpdatesCompleted": 0,
        "modelLoaded": False,
        "checkpointWrites": False,
        "artifactWrites": False,
        "finalHoldoutUsed": False,
    }


def _paths(args: Any) -> dict[str, Path]:
    if any(getattr(args, field) is None for field in PATH_FIELDS):
        raise Refusal("M11 worker requires all seven exact source/approval/output paths")
    return {field: getattr(args, field) for field in PATH_FIELDS}


def _admit(paths: dict[str, Path], bundle_sha: str,
           eos_weight: float) -> dict[str, Any]:
    if bundle_sha != research.BUNDLE_SHA:
        raise Refusal("M11 worker requires original pinned M7 bundle manifest")
    proof = gate.admission(
        paths["bundle_dir"], bundle_sha, paths["m9_report"],
        paths["m9_weights"], paths["baseline_report"],
        paths["baseline_responses"], paths["approval_file"],
        paths["output_dir"], eos_weight)
    if (proof.get("ownerConsentGranted") is not True
            or proof.get("status") != "cpu-inputs-and-owner-consent-verified-not-executed"
            or proof.get("optimizerUpdatesCompleted") != 0
            or proof.get("chosenEosLossWeight") != float(eos_weight)
            or proof.get("checkpointWrites") is not False):
        raise Refusal("M11 exact owner consent and CPU evidence gate did not pass")
    return proof


def _source_identity() -> dict[str, Any]:
    """Verify executable and evaluation sources are clean and hash-bound."""
    files = (Path(__file__), Path(gate.__file__), Path(core.__file__),
             Path(m11.__file__), Path(m10.__file__), Path(m8.__file__))
    names = [file.relative_to(research.ROOT).as_posix() for file in files]
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *names],
        cwd=research.ROOT, capture_output=True, text=True,
        check=True, timeout=8)
    if result.stdout.strip():
        raise Refusal("M11 worker, admission, scoring, or snapshot source is modified")
    return {
        **hw.source_identity(),
        "m11WorkerSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "m11AdmissionSha256": hashlib.sha256(Path(gate.__file__).read_bytes()).hexdigest(),
        "m11TrainingCoreSha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "m11CandidatesSha256": hashlib.sha256(Path(m11.__file__).read_bytes()).hexdigest(),
        "m10SnapshotHelperSha256": hashlib.sha256(Path(m10.__file__).read_bytes()).hexdigest(),
        "m8DecoderSha256": hashlib.sha256(Path(m8.__file__).read_bytes()).hexdigest(),
    }


def _verify_observations(rows: list[dict[str, Any]],
                         checkpoint_sha: str,
                         expected_scores: dict[str, Any]) -> dict[str, Any]:
    """Rescore the eight complete, checkpoint- and prompt-bound generations."""
    if type(rows) is not list or len(rows) != 8:
        raise Refusal("M11 checkpoint must have exactly eight generated responses")
    expected = {(policy, case["id"]) for policy in DECODERS for case in m11.cases()}
    seen = set()
    simplified = []
    required = {
        "id", "split", "decodingPolicy", "output", "terminationReason",
        "generatedTokens", "promptSha256", "checkpointSha256",
    }
    for row in rows:
        if type(row) is not dict or set(row) != required:
            raise Refusal("M11 generated response contains unexpected fields")
        pair = (row["decodingPolicy"], row["id"])
        case = next((x for x in m11.cases() if x["id"] == row["id"]), None)
        if (pair not in expected or pair in seen or case is None
                or row["split"] != case["split"]
                or type(row["output"]) is not str or len(row["output"]) > 8192
                or type(row["generatedTokens"]) is not int
                or not 0 <= row["generatedTokens"] <= m11.MAX_GENERATED
                or row["terminationReason"] not in gate.RESPONSE_TERMINATIONS
                or row["checkpointSha256"] != checkpoint_sha
                or row["promptSha256"] != hashlib.sha256(
                    m11.prompt_for(case).encode("utf-8")).hexdigest()):
            raise Refusal("M11 generated response has source/decoder/checkpoint mismatch")
        seen.add(pair)
        simplified.append({key: row[key] for key in (
            "id", "decodingPolicy", "output", "terminationReason",
            "promptSha256", "checkpointSha256")})
    if seen != expected:
        raise Refusal("M11 generated cases are missing or duplicated")
    scores = core.score_unverified_generations(simplified, checkpoint_sha)["scores"]
    if expected_scores != scores:
        raise Refusal("M11 checkpoint reported scores differ from eight raw outputs")
    return scores


def _validate_preferences(source: dict[str, Any]) -> None:
    """Independent arithmetic check for BOTH measured source-contrast pairs."""
    if type(source) is not dict or set(source) != {"train", "probe-only"}:
        raise Refusal("M11 needs train and development source-contrast measurements")
    for split in ("train", "probe-only"):
        p = source[split]
        cases = [c["id"] for c in m11.cases() if c["split"] == split]
        if (type(p) is not dict or p.get("split") != split
                or p.get("caseIds") != cases
                or p.get("usesGoldAnswerPrefix") is not True
                or p.get("autonomousExactCopyingEvaluated") is not False
                or type(p.get("contrastTokenIds")) is not list
                or len(p["contrastTokenIds"]) != 2
                or type(p.get("oneMinusZeroLogitMargins")) is not list
                or len(p["oneMinusZeroLogitMargins"]) != 2
                or type(p.get("fullVocabularyGreedyIds")) is not list
                or len(p["fullVocabularyGreedyIds"]) != 2):
            raise Refusal("M11 source preference metrics are incomplete")
        if (any(type(x) is not int for x in (
                *p["contrastTokenIds"], *p["fullVocabularyGreedyIds"]))
                or any(type(x) not in (int, float) or not math.isfinite(x)
                       for x in p["oneMinusZeroLogitMargins"])):
            raise Refusal("M11 source preference contains invalid logits/tokens")
        recomputed = m11.paired_preference(
            *p["oneMinusZeroLogitMargins"],
            *p["fullVocabularyGreedyIds"], *p["contrastTokenIds"])
        if p.get("tracksSourceInBothDirections") is not recomputed:
            raise Refusal("M11 source pair claimed preference disagrees with logits")


def _verify_snapshot(folder: Path, step: int,
                     proof: dict[str, Any], arm_weight: float) -> dict[str, Any]:
    """Independently verify all three files in a completed milestone directory."""
    if step not in SNAPSHOTS or folder.is_symlink() or not folder.is_dir():
        raise Refusal("M11 snapshot folder does not match approved milestones")
    weights, responses, report = (
        folder / "weights.pt", folder / "responses.jsonl", folder / "report.json")
    for file in (weights, responses, report):
        if file.is_symlink() or not file.is_file():
            raise Refusal("M11 milestone requires regular checkpoint/report/response files")
    if (report.stat().st_size > gate.REPORT_MAX_BYTES
            or responses.stat().st_size > gate.RESPONSES_MAX_BYTES):
        raise Refusal("M11 snapshot report or responses exceed admitted sizes")
    checkpoint_sha = m7.hash_file(weights, maximum=MAX_WEIGHTS)
    response_sha = research.digest_file(
        responses, limit=gate.RESPONSES_MAX_BYTES)
    data = json.loads(report.read_text(encoding="utf-8"))
    if (type(data) is not dict
            or type(data.get("checkpoint")) is not dict
            or type(data.get("scores")) is not dict
            or type(data.get("sourcePreferences")) is not dict
            or data.get("optimizerUpdatesCompleted") != step
            or data.get("checkpointSha256") != checkpoint_sha
            or data.get("checkpoint", {}).get("sha256") != checkpoint_sha
            or data["checkpoint"].get("bytes") != weights.stat().st_size
            or data.get("responsesSha256") != response_sha
            or data.get("candidateSha256") != proof["candidateSha256"]
            or data.get("scheduleSha256") != proof["scheduleSha256"]
            or data.get("baselineResponsesSha256") != proof["baselineResponsesSha256"]
            or data.get("ownerApprovalSha256") != proof["ownerApprovalSha256"]
            or data.get("chosenEosLossWeight") != arm_weight
            or data.get("decodingPolicies") != list(DECODERS)
            or data.get("verifiedGeneration") is not True
            or data.get("finalHoldoutUsed") is not False
            or data.get("optimizerStateSaved") is not False):
        raise Refusal("M11 saved milestone does not match original approved evidence")
    rows = research.read_jsonl(responses, limit=gate.RESPONSES_MAX_BYTES)
    _verify_observations(rows, checkpoint_sha, data["scores"])
    _validate_preferences(data["sourcePreferences"])
    return data


def _publication_step(recorded: dict[str, Any], *,
                      proof: dict[str, Any], stage: Path) -> None:
    """Check a complete checkpoint snapshot before a final or partial publish."""
    step = recorded.get("optimizerUpdatesCompleted")
    folder = stage / f"step-{step}"
    actual = _verify_snapshot(
        folder, step, proof, proof["chosenEosLossWeight"])
    if actual != recorded:
        raise Refusal("M11 checkpoint snapshot report changed since worker produced it")


def _decision(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Report progress honestly; train-pair success is NOT unseen generalization."""
    observations = []
    for snap in snapshots:
        both_train = all(
            snap["scores"][policy]["trainingPairExactAndEos"]
            for policy in DECODERS)
        both_development = all(
            snap["scores"][policy]["developmentPairExactAndEos"]
            for policy in DECODERS)
        both_preferences = all(
            snap["sourcePreferences"][split]["tracksSourceInBothDirections"]
            for split in ("train", "probe-only"))
        observations.append({
            "optimizerUpdate": snap["optimizerUpdatesCompleted"],
            "trainingPairExactAndEosBothDecoders": both_train,
            "developmentPairExactAndEosBothDecoders": both_development,
            "bidirectionalSourcePreferenceBothSplits": both_preferences,
        })
    two_consecutive = (
        len(observations) >= 2
        and all(all(row[key] for key in (
            "trainingPairExactAndEosBothDecoders",
            "developmentPairExactAndEosBothDecoders",
            "bidirectionalSourcePreferenceBothSplits"))
            for row in observations[-2:]))
    return {
        "protocol": "plex600-m11-capability-review-v1",
        "measuredMilestones": observations,
        "lastTwoMilestonesMeetNarrowM11Gate": two_consecutive,
        "generalCodingCapabilityProven": False,
        "unseenBroadJavaScriptGeneralizationProven": False,
        "trainingDataMemorizationNotExcluded": True,
        "promotionAutomatic": False,
    }


def _fit_check(path: Path) -> None:
    if shutil.disk_usage(path).free < MAX_WEIGHTS + 1024**3:
        raise Refusal("M11 insufficient headroom for next model-only checkpoint")


def _measure_generation(model: Any, torch: Any, tokenizer: Any,
                        checkpoint_sha: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from . import plex600_m8_eval as decoder
    predictions = []
    for policy in DECODERS:
        for case in m11.cases():
            prompt = m11.prompt_for(case)
            tokens = tokenizer.encode(prompt)
            with torch.inference_mode():
                output, count, termination = decoder.generate_one(
                    model, torch, tokenizer, tokens,
                    device=next(model.parameters()).device,
                    max_new_tokens=m11.MAX_GENERATED,
                    decoding_policy=policy)
            predictions.append({
                "id": case["id"], "split": case["split"],
                "decodingPolicy": policy, "output": output,
                "generatedTokens": count, "terminationReason": termination,
                "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "checkpointSha256": checkpoint_sha,
            })
    compact = [{key: row[key] for key in (
        "id", "decodingPolicy", "output", "terminationReason",
        "promptSha256", "checkpointSha256")} for row in predictions]
    scored = core.score_unverified_generations(compact, checkpoint_sha)["scores"]
    _verify_observations(predictions, checkpoint_sha, scored)
    return predictions, scored


def _train_worker(paths: dict[str, Path], bundle_sha: str,
                  arm_weight: float, stage: Path) -> dict[str, Any]:
    """Child only: the sole place PyTorch and optimizer are instantiated."""
    proof = _admit(paths, bundle_sha, arm_weight)
    identity = _source_identity()
    from .tokenizer import PlexTokenizer
    tokenizer = PlexTokenizer.load(paths["bundle_dir"])
    packed = core.packed_training_samples(tokenizer)
    schedule = core.exact_schedule()
    if (len(schedule) != core.MAX_UPDATES
            or core.schedule_sha256() != proof["scheduleSha256"]
            or set(packed) != {c["id"] for c in m11.cases() if c["split"] == "train"}):
        raise Refusal("M11 source-bound training schedule or training IDs changed")

    import torch  # No GPU/model imports until child has rechecked exact consent.
    from .plex600 import Plex600Config, Plex600Model, parameter_count
    hw._check_device(torch, 0)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(hw.MEMORY_FRACTION, device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    config = Plex600Config()
    with torch.device("cpu"):
        model = Plex600Model(config)
    if parameter_count(model) != hw.PARAMETERS:
        raise Refusal("M11 must retain original exact 600M architecture")
    state = torch.load(paths["m9_weights"], map_location="cpu",
                       weights_only=True, mmap=True)
    if type(state) not in (dict, OrderedDict):
        raise Refusal("M11 requires original model-only M9 state dictionary")
    model.load_state_dict(state, strict=True)
    del state
    model.to(device=device, dtype=torch.float32)
    model.gradient_checkpointing_enable()
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=5e-5, weight_decay=0.0, eps=1e-8,
        foreach=False, fused=False)
    begun = time.monotonic()
    updates = []
    milestones = []
    probe_ids = tokenizer.encode(m11.prompt_for(m11.cases()[0]))
    for row in schedule:
        step = row["optimizerUpdate"]
        if (step > core.MAX_UPDATES or
                time.monotonic() - begun >= HARD_TIMEOUT):
            raise Refusal("M11 reached hard 200-update/elapsed-time ceiling")
        sample = packed[row["recordId"]]
        x = torch.tensor([sample["inputIds"]], dtype=torch.long, device=device)
        y = torch.tensor([sample["targetIds"]], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embedding_row = sample["inputIds"][-1]
        before = model.token_embedding.weight[embedding_row].detach().clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = core.weighted_copy_loss(torch, logits, y, arm_weight)
        if not bool(torch.isfinite(loss).item()):
            raise Refusal("M11 loss became nonfinite")
        measured_loss = float(loss.item())
        loss.backward()
        squared = 0.0
        for p in parameters:
            grad = p.grad
            if grad is None or not bool(torch.isfinite(grad).all().item()):
                raise Refusal("M11 gradient is absent or nonfinite")
            squared += float(grad.float().square().sum().item())
        gradient_norm = math.sqrt(squared)
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise Refusal("M11 gradient norm is not positive and finite")
        optimizer.step()
        torch.cuda.synchronize(device)
        embedding_delta = float((
            model.token_embedding.weight[embedding_row].detach() - before
        ).float().norm().item())
        if not math.isfinite(embedding_delta) or embedding_delta <= 0:
            raise Refusal("M11 did not update measured embedding row")
        updates.append({
            "step": step, "recordId": row["recordId"],
            "loss": measured_loss, "gradientL2Norm": gradient_norm,
            "measuredEmbeddingRowDeltaL2": embedding_delta,
            "promptPositionsIncorrectlySupervised": 0,
            "supervisedAnswerPositions": sample["supervisedAnswerPositions"],
            "eosTargetSupervised": sample["targetIds"][-1] == core.EOS_ID,
        })
        del x, y, logits, loss, before
        if step not in SNAPSHOTS:
            continue
        _fit_check(stage)
        folder = stage / f"step-{step}"
        folder.mkdir(mode=0o700)
        saved = m10._reload_parity(
            model, torch, folder / "weights.pt", probe_ids, config)
        model.eval()
        outputs, scores = _measure_generation(
            model, torch, tokenizer, saved["sha256"])
        preferences = {
            split: core.source_pair_measurement(torch, model, tokenizer, split)
            for split in ("train", "probe-only")}
        _validate_preferences(preferences)
        raw = ("".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n"
                       for item in outputs)).encode("utf-8")
        if len(raw) > gate.RESPONSES_MAX_BYTES:
            raise Refusal("M11 generated response file exceeded bounded limit")
        with (folder / "responses.jsonl").open("xb") as stream:
            stream.write(raw)
        record = {
            "optimizerUpdatesCompleted": step,
            "checkpointSha256": saved["sha256"],
            "checkpoint": saved,
            "responsesSha256": hashlib.sha256(raw).hexdigest(),
            "candidateSha256": proof["candidateSha256"],
            "scheduleSha256": proof["scheduleSha256"],
            "ownerApprovalSha256": proof["ownerApprovalSha256"],
            "baselineResponsesSha256": proof["baselineResponsesSha256"],
            "chosenEosLossWeight": arm_weight,
            "decodingPolicies": list(DECODERS),
            "scores": scores,
            "sourcePreferences": preferences,
            "verifiedGeneration": True,
            "optimizerStateSaved": False,
            "peakCudaReservedBytes": torch.cuda.max_memory_reserved(device),
            "finalHoldoutUsed": False,
        }
        with (folder / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        _publication_step(record, proof=proof, stage=stage)
        milestones.append(record)
    if len(updates) != core.MAX_UPDATES or [
            record["optimizerUpdatesCompleted"] for record in milestones
            ] != list(SNAPSHOTS):
        raise Refusal("M11 did not complete all four predeclared milestones")
    return {
        "protocol": PROTOCOL,
        "status": "bounded-200-updates-complete-unreviewed",
        "trainingAuthorized": True,
        "originalM9WeightsSha256": proof["m9WeightsSha256"],
        "baselineReportSha256": proof["baselineReportSha256"],
        "baselineResponsesSha256": proof["baselineResponsesSha256"],
        "candidateSha256": core.candidate_sha256(),
        "scheduleSha256": core.schedule_sha256(),
        "ownerApprovalSha256": proof["ownerApprovalSha256"],
        "sourceIdentity": identity,
        "chosenEosLossWeight": arm_weight,
        "optimizerUpdatesCompleted": len(updates),
        "maxOptimizerUpdates": core.MAX_UPDATES,
        "steps": updates,
        "snapshots": milestones,
        "decision": _decision(milestones),
        "elapsedSeconds": round(time.monotonic() - begun, 3),
        "peakCudaReservedBytes": torch.cuda.max_memory_reserved(device),
        "optimizerStateSaved": False,
        "historicalCheckpointWrites": False,
        "finalHoldoutUsed": False,
        "generalCodingCapabilityProven": False,
    }


def _check_final(result: dict[str, Any], proof: dict[str, Any],
                 identity: dict[str, Any], arm_weight: float,
                 stage: Path) -> None:
    """Parent revalidates the child receipts before publishing any result."""
    if (type(result) is not dict or result.get("protocol") != PROTOCOL
            or result.get("status") != "bounded-200-updates-complete-unreviewed"
            or result.get("trainingAuthorized") is not True
            or result.get("sourceIdentity") != identity
            or result.get("originalM9WeightsSha256") != proof["m9WeightsSha256"]
            or result.get("candidateSha256") != proof["candidateSha256"]
            or result.get("scheduleSha256") != proof["scheduleSha256"]
            or result.get("baselineReportSha256") != proof["baselineReportSha256"]
            or result.get("baselineResponsesSha256") != proof["baselineResponsesSha256"]
            or result.get("ownerApprovalSha256") != proof["ownerApprovalSha256"]
            or result.get("chosenEosLossWeight") != arm_weight
            or result.get("maxOptimizerUpdates") != core.MAX_UPDATES
            or result.get("optimizerUpdatesCompleted") != core.MAX_UPDATES
            or result.get("optimizerStateSaved") is not False
            or result.get("historicalCheckpointWrites") is not False
            or result.get("finalHoldoutUsed") is not False
            or result.get("generalCodingCapabilityProven") is not False):
        raise Refusal("M11 child outcome differs from exact approved source scope")
    steps = result.get("steps")
    schedule = core.exact_schedule()
    if (type(steps) is not list or len(steps) != core.MAX_UPDATES
            or any(type(done) is not dict
                   or done.get("step") != expected["optimizerUpdate"]
                   or done.get("recordId") != expected["recordId"]
                   or done.get("promptPositionsIncorrectlySupervised") != 0
                   or done.get("eosTargetSupervised") is not True
                   or type(done.get("loss")) not in (int, float)
                   or not math.isfinite(done["loss"])
                   or done["loss"] < 0
                   or type(done.get("gradientL2Norm")) not in (int, float)
                   or not math.isfinite(done["gradientL2Norm"])
                   or done["gradientL2Norm"] <= 0
                   or type(done.get("measuredEmbeddingRowDeltaL2")) not in (int, float)
                   or not math.isfinite(done["measuredEmbeddingRowDeltaL2"])
                   or done["measuredEmbeddingRowDeltaL2"] <= 0
                   for done, expected in zip(steps, schedule))):
        raise Refusal("M11 child optimizer evidence differs from frozen 200-update schedule")
    snapshots = result.get("snapshots")
    if (type(snapshots) is not list or
            [snap.get("optimizerUpdatesCompleted")
             for snap in snapshots] != list(SNAPSHOTS)):
        raise Refusal("M11 child checkpoint milestones missing or out of order")
    for recorded in snapshots:
        _publication_step(recorded, proof=proof, stage=stage)
    if result.get("decision") != _decision(snapshots):
        raise Refusal("M11 child capability decisions differ from checkpoint measurements")


def run_approved(paths: dict[str, Path], bundle_sha: str,
                 arm_weight: float, *, explicit_flag: bool) -> dict[str, Any]:
    if explicit_flag is not True:
        raise Refusal("M11 worker requires explicit --execute-approved-run")
    proof = _admit(paths, bundle_sha, arm_weight)
    identity = _source_identity()
    target = m8.require_output(
        paths["output_dir"], bundle=paths["bundle_dir"],
        checkpoint=paths["m9_weights"])
    stage: Path | None = Path(tempfile.mkdtemp(
        prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        nonce = secrets.token_hex(32)
        env = dict(os.environ, **{
            CONTROL: nonce, "PYTHONDONTWRITEBYTECODE": "1"})
        payload = {
            "nonce": nonce,
            "bundle_sha": bundle_sha,
            "eos_weight": arm_weight,
            "source_identity": identity,
            "stage": str(stage.resolve(strict=True)),
            "paths": {
                name: str(p.resolve(strict=False) if name == "output_dir"
                          else p.resolve(strict=True))
                for name, p in paths.items()
            },
        }
        return_code, stdout, stderr, timed_out = hw._supervise(
            [sys.executable, "-m", "plex_training.plex600_m11_worker",
             "--_worker"], payload, env, timeout=HARD_TIMEOUT)
        if timed_out or return_code != 0:
            raise Refusal(
                "M11 supervised child did not complete: " +
                ("hard timeout" if timed_out else "worker exited unsuccessfully") +
                "; incomplete stage not promoted; no optimizer-state resume")
        if len(stdout) > 2 * 1024**2:
            raise Refusal("M11 child report exceeds bounded output size")
        result = json.loads(stdout)
        _check_final(result, proof, identity, arm_weight, stage)
        with (stage / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        if target.exists():
            raise Refusal("M11 finished while output destination became occupied")
        os.rename(stage, target)
        stage = None
        return result
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def _worker_entry() -> None:
    raw = sys.stdin.buffer.read(64 * 1024 + 1)
    if len(raw) > 64 * 1024:
        raise Refusal("M11 private worker controls exceed size bound")
    payload = json.loads(raw)
    nonce = os.environ.pop(CONTROL, "")
    if (not nonce or type(payload) is not dict
            or set(payload) != {
                "nonce", "bundle_sha", "eos_weight",
                "source_identity", "stage", "paths"}
            or payload.get("nonce") != nonce
            or type(payload.get("paths")) is not dict
            or set(payload["paths"]) != set(PATH_FIELDS)
            or payload.get("bundle_sha") != research.BUNDLE_SHA
            or type(payload.get("eos_weight")) not in (int, float)
            or payload["eos_weight"] not in core.EOS_WEIGHTS
            or payload.get("source_identity") != _source_identity()):
        raise Refusal("M11 private worker requires its approved parent nonce and exact scope")
    paths = {name: Path(value) for name, value in payload["paths"].items()}
    expected_output = m8.require_output(
        paths["output_dir"], bundle=paths["bundle_dir"],
        checkpoint=paths["m9_weights"])
    stage = Path(payload["stage"])
    if (stage.is_symlink() or not stage.is_dir()
            or stage.resolve(strict=True).parent != expected_output.parent
            or not stage.name.startswith(f".{expected_output.name}.staging-")):
        raise Refusal("M11 private worker staging folder is outside approved output")
    report = _train_worker(
        paths, payload["bundle_sha"], float(payload["eos_weight"]), stage)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-approved-run", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    for name in PATH_FIELDS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--bundle-manifest-sha256")
    parser.add_argument("--eos-weight", type=float)
    args = parser.parse_args(argv)
    try:
        if args._worker:
            if (args.execute_approved_run or args.eos_weight is not None
                    or args.bundle_manifest_sha256 is not None
                    or any(getattr(args, name) is not None for name in PATH_FIELDS)):
                raise Refusal("M11 private worker does not accept public CLI flags")
            _worker_entry()
            return 0
        if args.execute_approved_run:
            paths = _paths(args)
            if (args.bundle_manifest_sha256 is None
                    or args.eos_weight not in core.EOS_WEIGHTS):
                raise Refusal("M11 run needs pinned M7 tokenizer and chosen 1.0/0.25 EOS arm")
            result = run_approved(
                paths, args.bundle_manifest_sha256,
                args.eos_weight, explicit_flag=True)
        elif any(getattr(args, name) is not None for name in PATH_FIELDS) or (
                args.bundle_manifest_sha256 is not None or args.eos_weight is not None):
            raise Refusal("M11 source or owner consent paths require explicit execution flag")
        else:
            result = plan()
    except (OSError, ValueError, RuntimeError, TypeError, KeyError,
            subprocess.SubprocessError) as exc:
        print("M11 worker refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
