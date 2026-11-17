"""M9: strictly bounded, separately approved Plex 600M R3 fine-tuning pilot.

Default is PLAN ONLY; --audit-curriculum reconstructs the frozen M7 tokenizer
on CPU to determine a bounded 8-example, family-disjoint source-file edit pilot.

Training REQUIRES: an externally owner-reviewed, explicit exact-scope local
approval document; SHA-verified original M7 weights/report/bundle; explicit
--execute-approved-run; a new create-only output directory. The approval is
not bundled with the repo and is NOT implied by data preparation.

Training is eight masked-answer/EOS-only optimizer updates maximum, no
pretraining from M8 development, no R2 candidate use or final holdout.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import sys
import tempfile
import time
from typing import Any
from datetime import date

from . import plex600_feasibility as hardware
from . import plex600_m7 as m7
from . import plex600_m8_eval as m8
from . import plex600_r3_curriculum as r3

PROTOCOL = "plex600-m9-r3-eight-update-pilot-v1"
APPROVAL_KIND = "plex600-m9-explicit-owner-training-approval-v1"
UPDATES = 8
MAX_POSITIONS = 224
MAX_NEW_CHECKPOINT_BYTES = 3 * 1024**3
TIMEOUT_SECONDS = 1200
SEED = 29
CONTROL = "PLEX600_M9_WORKER_NONCE"


class Refusal(ValueError):
    pass


def plan() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL, "status": "plan-only",
        "modelFamily": "plex-600m-v1", "modelParameterCount": hardware.PARAMETERS,
        "source": "quarantined-r3-proposal-owner-review-required",
        "candidateRows": 111, "candidateTrainFamilies": 28,
        "candidateValidationFamilies": 9,
        "maxInputPositionsPerExample": MAX_POSITIONS,
        "optimizerUpdatesMaximum": UPDATES, "optimizerUpdatesCompleted": 0,
        "loss": "edited-file-and-eos-only-mask-source-and-request",
        "optimizer": "new-AdamW-state-from-M7-weights-only-not-exact-resume",
        "trainingAuthorized": False,
        "ownerApprovalBundled": False,
        "modelLoaded": False, "finalHoldoutUsed": False,
        "m8DevelopmentUsedForTraining": False, "checkpointWrites": False,
        "checkpointPolicy": "new-directory-and-weights-file-only",
    }


def completed_plan() -> dict[str, Any]:
    """Mark the checkpoint created by a completed worker, without changing the plan."""
    return {**plan(), "checkpointWrites": True}


def selected_records(tokenizer: Any) -> dict[str, Any]:
    """One non-truncated original source-conditioned candidate per family."""
    rows = r3.rows()  # Source status, integrity, and M8 exact-operation checks
    by_family = r3.family_splits(rows)
    assignments = {name: set(families) for name, families in by_family.items()}
    if assignments["train"] & assignments["validation"]:
        raise Refusal("R3 grouped split leaked")
    candidates: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        "train": [], "validation": [],
    }
    for row in rows:
        if row["variant"] != "base":
            continue
        split = "train" if row["family"] in assignments["train"] else "validation"
        prompt_ids = tokenizer.encode(m8.prompt_for(row))
        answer_ids = tokenizer.encode(row["expectedFile"]) + [3]
        packed = r3.masked_record(
            prompt_ids, answer_ids, fitted_vocab_size=tokenizer.vocabulary_size)
        if len(packed["inputIds"]) <= MAX_POSITIONS:
            candidates[split].append((row, packed))
    def rank(item: tuple[dict[str, Any], dict[str, Any]]) -> str:
        return hashlib.sha256(
            (f"m9-{SEED}:" + item[0]["id"]).encode()).hexdigest()
    train = sorted(candidates["train"], key=rank)
    validation = sorted(candidates["validation"], key=rank)
    # A sequence cap may exclude some examples; refuse rather than truncate.
    train_selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    validation_selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for language in ("html", "css", "javascript"):
        subset = [item for item in train if item[0]["language"] == language]
        check = [item for item in validation if item[0]["language"] == language]
        if len(subset) < 2 or not check:
            raise Refusal("M9 lacks two bounded training families and one validation family per language")
        train_selected.extend(subset[:2])
        validation_selected.extend(check[:1])
    present = {item[0]["id"] for item in train_selected}
    train_selected.extend(
        item for item in train if item[0]["id"] not in present
    )
    train_selected = train_selected[:UPDATES]
    if (len(train_selected) != UPDATES
            or len({row["family"] for row, _ in train_selected}) != UPDATES
            or {row["family"] for row, _ in train_selected} &
               {row["family"] for row, _ in validation_selected}):
        raise Refusal("M9 bounded family selection invalid")
    selected = {
        "train": train_selected,
        "validation": validation_selected,
    }
    return {
        "protocol": PROTOCOL,
        "status": "bounded-r3-sample-prepared-not-authorized",
        "r3CandidateSha256": r3.source_sha256(),
        "bundleManifestSha256": r3.research.BUNDLE_SHA,
        "tokenizerSha256": m7.TOKENIZER_SHA,
        "trainExamplesEligible224": len(train),
        "validationExamplesEligible224": len(validation),
        "trainRecordIds": [row["id"] for row, _ in train_selected],
        "trainLanguages": [row["language"] for row, _ in train_selected],
        "validationRecordIds": [row["id"] for row, _ in validation_selected],
        "trainMaxPositions": max(len(p["inputIds"]) for _, p in train_selected),
        "validationMaxPositions": max(len(p["inputIds"]) for _, p in validation_selected),
        "trainSupervisedAnswerPositions": sum(
            p["supervisedAnswerPositions"] for _, p in train_selected),
        "validationSupervisedAnswerPositions": sum(
            p["supervisedAnswerPositions"] for _, p in validation_selected),
        "maskedSourcePromptSupervision": True,
        "trainingAuthorized": False, "optimizerUpdatesCompleted": 0,
        "modelLoaded": False, "finalHoldoutUsed": False,
        "data": selected,
    }


def public_selection(selected: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in selected.items() if k != "data"}


def approval_draft(selection: dict[str, Any], checkpoint_sha: str = "OWNER_M7_CHECKPOINT_SHA256") -> dict[str, Any]:
    """Unapproved template, never an actual permission to train."""
    return {
        "kind": APPROVAL_KIND,
        "approvedBy": None,
        "approvedAt": None,
        "approvedForTraining": False,
        "sourceReviewedByOwner": False,
        "r3CandidateSha256": selection["r3CandidateSha256"],
        "m7CheckpointSha256": checkpoint_sha,
        "bundleManifestSha256": r3.research.BUNDLE_SHA,
        "trainRecordIds": selection["trainRecordIds"],
        "validationRecordIds": selection["validationRecordIds"],
        "maxOptimizerUpdates": UPDATES,
        "maxInputPositions": MAX_POSITIONS,
        "optimizerState": "fresh-AdamW-not-M7-resume",
        "finalHoldoutAllowed": False,
        "m8DevelopmentUsedForTraining": False,
        "scope": PROTOCOL,
    }


def check_approval(path: Path, selection: dict[str, Any], *,
                   original_checkpoint_sha: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024:
        raise Refusal("Missing or oversized external owner training approval")
    data = json.loads(path.read_bytes())
    draft = approval_draft(selection, checkpoint_sha=original_checkpoint_sha)
    if (type(data) is not dict or set(data) != set(draft)
            or {k: v for k, v in data.items()
                if k not in {"approvedBy", "approvedAt", "approvedForTraining",
                             "sourceReviewedByOwner"}} !=
               {k: v for k, v in draft.items()
                if k not in {"approvedBy", "approvedAt", "approvedForTraining",
                             "sourceReviewedByOwner"}}
            or data.get("approvedForTraining") is not True
            or data.get("sourceReviewedByOwner") is not True
            or data.get("approvedBy") != "keepithandy"
            or type(data.get("approvedAt")) is not str
            or len(data["approvedAt"]) != 10
            or not data["approvedAt"].startswith("2026-")):
        raise Refusal("M9 requires separate exact source-scoped owner review and approval")
    try:
        approved_date = date.fromisoformat(data["approvedAt"])
    except (ValueError, TypeError) as exc:
        raise Refusal("M9 approval date must be a real ISO calendar date") from exc
    if approved_date < date(2026, 11, 16):
        raise Refusal("M9 approval predates the exact R3 experiment")
    return {"approved": True, "approvalSha256": m7.hash_file(path, maximum=16 * 1024)}


def code_identity() -> dict[str, Any]:
    """Pin committed model source and the exact M9 runner before CUDA."""
    return {
        **hardware.source_identity(),
        "m9RunnerSha256": m7.hash_file(Path(__file__)),
    }


def measured_loss(model: Any, torch: Any,
                  samples: list[tuple[dict[str, Any], dict[str, Any]]],
                  device: Any) -> dict[str, Any]:
    model.eval()
    total_loss, total_labels = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _, packed in samples:
            inputs = torch.tensor([packed["inputIds"]], dtype=torch.long, device=device)
            labels = torch.tensor([packed["targetIds"]], dtype=torch.long, device=device)
            _, loss = model(inputs, labels)
            if loss is None or not bool(torch.isfinite(loss)):
                raise Refusal("Non-finite gold-conditioned M9 loss")
            count = packed["supervisedAnswerPositions"]
            total_loss += float(loss.item()) * count
            total_labels += count
    if total_labels <= 0:
        raise Refusal("No validation labels")
    return {"meanAnswerNll": total_loss / total_labels,
            "supervisedAnswerPositions": total_labels,
            "validationCases": len(samples)}


def load_m7_state(torch: Any, weights: Path) -> dict[str, Any]:
    """Read the verified M7 model-only state in its native PyTorch format."""
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if type(state) not in (dict, OrderedDict):
        raise Refusal("Expected original M7 model-only checkpoint")
    return state


def train_worker(bundle: Path, m7_report: Path, weights: Path, approval: Path,
                 *, stage: Path, bundle_sha: str) -> dict[str, Any]:
    proof = m8.prerequisites(bundle, m7_report, weights, bundle_sha)
    from .tokenizer import PlexTokenizer
    selection = selected_records(PlexTokenizer.load(bundle))
    authorization = check_approval(
        approval, selection,
        original_checkpoint_sha=proof["m7Receipt"]["localCheckpointSha256"])
    identity = code_identity()

    import torch
    from .plex600 import Plex600Model, Plex600Config, parameter_count
    hardware._check_device(torch, 0)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise Refusal("M9 requires native BF16-capable GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(hardware.MEMORY_FRACTION, device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(SEED)
    with torch.device("cpu"):
        model = Plex600Model(Plex600Config())
    if parameter_count(model) != hardware.PARAMETERS:
        raise Refusal("Frozen M7 architecture mismatch")
    state = load_m7_state(torch, weights)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device=device, dtype=torch.float32)
    model.gradient_checkpointing_enable()
    # The M7 checkpoint saved NO optimizer state. This is a NEW AdamW
    # experiment initialized from original M7 weights, NOT an exact resume.
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=5e-5, weight_decay=0, eps=1e-8,
        foreach=False, fused=False)

    validation_before = measured_loss(
        model, torch, selection["data"]["validation"], device)
    updates: list[dict[str, Any]] = []
    start = time.perf_counter()
    for i, (row, packed) in enumerate(selection["data"]["train"], 1):
        if i > UPDATES or time.perf_counter() - start >= TIMEOUT_SECONDS:
            raise Refusal("M9 optimizer-update count or deadline exceeded")
        model.train()
        inputs = torch.tensor([packed["inputIds"]], dtype=torch.long, device=device)
        labels = torch.tensor([packed["targetIds"]], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(inputs, labels)
        if loss is None or not bool(torch.isfinite(loss)):
            raise Refusal("Non-finite M9 supervised loss")
        before = float(loss.item())
        loss.backward()
        for param in params:
            if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                raise Refusal("Non-finite M9 gradient")
        optimizer.step()
        torch.cuda.synchronize(device)
        updates.append({
            "step": i, "recordId": row["id"], "language": row["language"],
            "lossBeforeStep": before,
            "supervisedAnswerPositions": packed["supervisedAnswerPositions"],
        })
    validation_after = measured_loss(
        model, torch, selection["data"]["validation"], device)
    if len(updates) != UPDATES:
        raise Refusal("Incomplete M9 bounded update sequence")
    output = stage / "weights.pt"
    if output.exists():
        raise Refusal("M9 checkpoint would overwrite evidence")
    torch.save(model.state_dict(), output)
    if not 0 < output.stat().st_size <= MAX_NEW_CHECKPOINT_BYTES:
        raise Refusal("M9 weights exceed bounded new checkpoint storage")
    report = {
        **completed_plan(),
        **public_selection(selection),
        "status": "eight-updates-complete-unreviewed-improvement",
        "trainingAuthorized": True, "modelLoaded": True,
        "optimizerUpdatesCompleted": len(updates),
        "originalCheckpointSha256": proof["m7Receipt"]["localCheckpointSha256"],
        "originalReportSha256": proof["m7Receipt"]["localReportSha256"],
        "approval": authorization,
        "codeIdentity": identity,
        "originalTokenizerSha256": proof["tokenizerSha256"],
        "benchmarkSha256": proof["benchmarkSha256"],
        "validationBefore": validation_before,
        "validationAfter": validation_after,
        "steps": updates,
        "peakCudaReservedBytes": torch.cuda.max_memory_reserved(device),
        "checkpoint": {
            "file": "weights.pt",
            "bytes": output.stat().st_size,
            "sha256": m7.hash_file(output, maximum=MAX_NEW_CHECKPOINT_BYTES),
            "weightsOnly": True, "optimizerStateSaved": False,
            "resumeAuthorized": False,
        },
        "generatedCodeEvaluated": False,
        "codingImprovementProven": False,
        "finalHoldoutUsed": False,
    }
    return report


def worker_failure_message(exit_code: int, stderr: str, timed_out: bool) -> str:
    """Return a bounded local diagnostic; never reinterpret a failed worker as pass."""
    if timed_out:
        return f"M9 worker exceeded {TIMEOUT_SECONDS} second time budget"
    detail = stderr.strip()[-2048:]
    if not detail:
        detail = "no stderr captured (possible OOM kill or external termination)"
    return f"M9 worker exited with code {exit_code}: {detail}"


def execute(bundle: Path, report: Path, weights: Path, approval: Path,
            output: Path, *, bundle_sha: str, execute_flag: bool) -> dict[str, Any]:
    if execute_flag is not True:
        raise Refusal("Explicit --execute-approved-run required")
    from .tokenizer import PlexTokenizer
    target = m8.require_output(output, bundle=bundle, checkpoint=weights)
    proof = m8.prerequisites(bundle, report, weights, bundle_sha)
    selection = selected_records(PlexTokenizer.load(bundle))
    approval_proof = check_approval(approval, selection,
                   original_checkpoint_sha=proof["m7Receipt"]["localCheckpointSha256"])
    identity = code_identity()
    stage: Path | None = Path(tempfile.mkdtemp(
        prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        # A separate child process enforces the hard wall-clock stop even if
        # a single GPU update stalls; the worker revalidates ALL approvals.
        nonce = secrets.token_hex(32)
        env = dict(os.environ, **{
            CONTROL: nonce, "PYTHONDONTWRITEBYTECODE": "1"})
        payload = {
            "nonce": nonce, "bundle": str(bundle.resolve(strict=True)),
            "report": str(report.resolve(strict=True)),
            "weights": str(weights.resolve(strict=True)),
            "approval": str(approval.resolve(strict=True)),
            "stage": str(stage.resolve(strict=True)),
            "bundleSha": bundle_sha,
        }
        command = [sys.executable, "-m", "plex_training.plex600_m9", "--_worker"]
        code, stdout, stderr, timed_out = hardware._supervise(
            command, payload, env, timeout=TIMEOUT_SECONDS)
        if timed_out or code != 0:
            raise Refusal(worker_failure_message(code, stderr, timed_out))
        outcome = json.loads(stdout)
        checkpoint = outcome.get("checkpoint")
        if (outcome.get("protocol") != PROTOCOL
                or outcome.get("trainingAuthorized") is not True
                or outcome.get("optimizerUpdatesCompleted") != UPDATES
                or outcome.get("originalCheckpointSha256") !=
                    proof["m7Receipt"]["localCheckpointSha256"]
                or outcome.get("r3CandidateSha256") != r3.source_sha256()
                or outcome.get("trainRecordIds") != selection["trainRecordIds"]
                or outcome.get("validationRecordIds") != selection["validationRecordIds"]
                or outcome.get("approval") != approval_proof
                or outcome.get("codeIdentity") != identity
                or outcome.get("finalHoldoutUsed") is not False
                or outcome.get("generatedCodeEvaluated") is not False
                or outcome.get("codingImprovementProven") is not False
                or type(checkpoint) is not dict
                or checkpoint.get("file") != "weights.pt"
                or checkpoint.get("weightsOnly") is not True
                or checkpoint.get("optimizerStateSaved") is not False
                or len(outcome.get("steps", [])) != UPDATES):
            raise Refusal("M9 worker result lacks original source/optimizer evidence")
        original_checkpoint = stage / "weights.pt"
        if (checkpoint.get("bytes") != original_checkpoint.stat().st_size
                or checkpoint.get("sha256") !=
                   m7.hash_file(original_checkpoint, maximum=MAX_NEW_CHECKPOINT_BYTES)):
            raise Refusal("New M9 checkpoint failed SHA/byte-size verification")
        with (stage / "report.json").open(
                "x", encoding="utf-8", newline="\n") as stream:
            json.dump(outcome, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        if target.exists():
            raise Refusal("Destination appeared during M9 execution")
        os.rename(stage, target)
        stage = None
        return outcome
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def worker() -> None:
    # Worker is never a separately authorized CLI; approval is revalidated.
    payload = json.loads(sys.stdin.buffer.read(64 * 1024))
    nonce = os.environ.get(CONTROL)
    if not nonce or nonce != payload.get("nonce"):
        raise Refusal("M9 worker nonce missing or mismatched")
    allowed = {"nonce", "bundle", "report", "weights", "approval",
               "stage", "bundleSha"}
    if set(payload) != allowed:
        raise Refusal("Unexpected M9 worker input")
    result = train_worker(
        Path(payload["bundle"]), Path(payload["report"]),
        Path(payload["weights"]), Path(payload["approval"]),
        stage=Path(payload["stage"]), bundle_sha=payload["bundleSha"])
    print(json.dumps(result, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-curriculum", action="store_true")
    parser.add_argument("--approval-template", action="store_true")
    parser.add_argument("--execute-approved-run", action="store_true")
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--bundle-manifest-sha256")
    parser.add_argument("--m7-report", type=Path)
    parser.add_argument("--m7-weights", type=Path)
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args._worker:
            worker()
            return 0
        if args.execute_approved_run:
            if (args.audit_curriculum or args.approval_template
                    or not all((args.bundle_dir, args.bundle_manifest_sha256,
                                args.m7_report, args.m7_weights,
                                args.approval_file, args.output_dir))):
                raise Refusal("M9 training requires explicit original M7 inputs, new output and exact owner approval")
            result = execute(
                args.bundle_dir, args.m7_report, args.m7_weights,
                args.approval_file, args.output_dir,
                bundle_sha=args.bundle_manifest_sha256, execute_flag=True)
        elif args.audit_curriculum or args.approval_template:
            def prepare(tokenizer: Any) -> dict[str, Any]:
                selected = selected_records(tokenizer)
                public = public_selection(selected)
                if args.approval_template:
                    return {**public, "notAnApproval": True,
                            "ownerReviewRequired": True,
                            "approvalDraft": approval_draft(selected)}
                return public
            result = r3.with_frozen_tokenizer(prepare)
        else:
            result = plan()
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        print("Plex M9 bounded pilot refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
