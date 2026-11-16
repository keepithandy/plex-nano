# Plex 600M M9 — bounded supervised file-edit fine-tune

**November 16, 2026.** This implementation is **execution-disabled by default**, and no M9 optimizer update has occurred. Its purpose is to move from M8 research to a small real change in weights **once the original M7 checkpoint and explicit reviewed data approval are available**. The original 27M baseline, M7 four-update result, M8 benchmark and final holdout remain unchanged.

## Why a fresh fine-tune, not another full scratch run

The original M7 checkpoint was built after four short-answer updates. Its owner-authored training rows used `Request → Answer`, not `Request + source file → entire edited file`. We now have 111 R3 quarantined source-conditioned candidates. The purpose of M9 is to try **eight supervised answer-only updates** on carefully selected full-file examples using the existing M7 weights. This is the smallest useful pilot toward training the exact file-edit behavior without allocating a large unapproved run.

**The M7 checkpoint is model-only:** it contains no optimizer state. M9 therefore initializes a *new* AdamW optimizer at a lower `5e-5` learning rate. This is **not** an exact optimizer-state resume from M7, and results must not be presented as uninterrupted M7 training.

## Strict pilot contract

The new [M9 runner](../training/src/plex_training/plex600_m9.py) is disabled unless *all* conditions hold:

- The caller explicitly uses `--execute-approved-run` with a **new output directory**, the original M7 report and model-only checkpoint, the SHA-verified original M7 tokenizer bundle and a separate **owner-reviewed local authorization**.
- Authorization must exactly pin the R3 candidate SHA, the owner identity, the **eight predeclared train record IDs**, the three family-disjoint validation IDs, the precise M7 checkpoint SHA, tokenizer manifest SHA and `maxOptimizerUpdates = 8`. A draft or mismatched packet is rejected before any GPU/model initialization.
- R3 candidates remain source-provenance reviewed by code but **not owner-approved in this GitHub branch**; the approval template explicitly has `approvedForTraining: false` and `sourceReviewedByOwner: false`. No approval is bundled or inferred from prior M7 permissions.
- Use **one original base variant per distinct family**: two each from HTML/CSS/JavaScript and two further deterministic training families (eight total). Hold out three distinct validation families, one per language. Refuse rather than truncate any prompt+answer sequence above **224 model positions**.
- The original prompt and source tokens are present as input context, but **only edited-file/EOS positions** contribute to the cross-entropy loss (`-100` masks elsewhere). One AdamW optimizer step per chosen record, with gradient checkpointing and BF16 forward-pass autocast.
- The worker is separately supervised with a **1,200-second wall-clock bound**, a nonce and repeat authorization checks, and may not perform more than eight optimizer updates. No restarts or automatic retry are authorized.
- The output is a **new create-only** weights-only checkpoint and report, original checkpoint untouched; writes are staged then published atomically. The report records hashes, selected row IDs, eight step losses, answer-token-weighted internal validation loss before/after, and peak GPU reservation. No final holdout use or optimizer state is saved.

These limits are **a first GPU experiment**, not a performance assurance on a 16 GiB RTX 4080 SUPER. No sequence exceeding 224 positions is silently truncated. Actual peak memory, training throughput, semantic success and code-edit quality must be measured after an authorized run. In particular, a reduction in gold-conditioned validation loss **does not prove generated-file correctness**.

## Supported CPU-only checks, available without local checkpoint

Plain plan — no model, tokenizer fitting, source access, GPU or writes:

```powershell
uv run --project training --no-sync python -m plex_training.plex600_m9
```

Exact source-bound original tokenizer reconstruction, 224-position training selection and *unapproved* proposed record IDs:

```powershell
uv run --project training --no-sync python -m plex_training.plex600_m9 --audit-curriculum
```

Display a draft **not usable as an approval** until the owner reviews the exact candidate data and original checkpoint, fills identity/permission/date fields, and pins the local weights SHA:

```powershell
uv run --project training --no-sync python -m plex_training.plex600_m9 --approval-template
```

These CPU-only checks are run on Linux and Windows GitHub Actions against the original frozen tokenizer, with unit tests enforcing the approval gate and causal target masks.

## Separately approved local GPU run (NOT EXECUTED)

After manually reviewing every selected candidate, the M7 report and source rights and preparing the exact **local** approval JSON, a bounded eight-update experiment would use this shape:

```powershell
uv run --project training --no-sync python -m plex_training.plex600_m9 `
  --execute-approved-run `
  --bundle-dir "PATH_TO_ORIGINAL_M7_TOKENIZER_BUNDLE" `
  --bundle-manifest-sha256 "14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55" `
  --m7-report "PATH_TO_ORIGINAL_M7_REPORT.json" `
  --m7-weights "PATH_TO_ORIGINAL_M7_WEIGHTS.pt" `
  --approval-file "PATH_TO_SEPARATELY_APPROVED_M9_SCOPE.json" `
  --output-dir "training/artifacts/plex600-m9-r3-first-eight"
```

The local approval file must come from an explicit new owner decision on the **exact selected data and checkpoint**, not a historical approval or generated pretend authorization. The final holdout remains sealed. The owner may reject or revise this experiment without changing M7.

## Next scientific comparison

After the first authorized run is independently completed and the new checkpoint SHA validated, compare the original M7 weights and the new M9 weights against the **unchanged M8 development benchmark**, under the exact 96-token output budget and with the decoding mode explicitly labeled. Evaluate real generated full files, context preservation, termination mode and wall time, and track whether a training-loss decrease translates into better coding. Do not retrain on the M8 benchmark itself, and do not confuse teacher-forced probabilities with successful generated edits.
