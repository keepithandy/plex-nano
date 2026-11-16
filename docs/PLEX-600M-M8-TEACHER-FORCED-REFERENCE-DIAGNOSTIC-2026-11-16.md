# Plex 600M M8 — read-only teacher-forced reference diagnostic

**Date:** November 16, 2026. **Scope:** add a frozen-checkpoint-bound *diagnostic* to the existing M8 research tooling. This is not new training and it does not claim any model-generated edit succeeds.

## Why this is a high-value next test

The four-update M7 checkpoint is owner-reported, with local-only weights. M7 trained on short `Request → Answer` examples, while M8 asks for full-file `Request + Source file → Edited file` completion. We now know the exact M7 tokenizer can encode all 36 M8 reference outputs at 96 tokens, but **output capacity is not ability**.

An autoregressive exact-file score cannot distinguish several failure modes: the model might be unable to put probability on the right next token at all; it might predict a high-ranked token ID outside its fitted tokenizer; it might prematurely select EOS; or it might assign plausible local token probabilities but fail to sustain a full-file generation. We need a low-risk measurement that distinguishes those explanations **before choosing a new curriculum or requesting further training**.

## Implemented read-only method

`training/src/plex_training/plex600_m8_teacher_forced.py` requires the **same full original M7 receipt, SHA-verified model-only checkpoint, frozen 1,191-token BPE bundle and source-bound M8 benchmark** as the normal M8 generation runner. There is no model load, optimizer, final-holdout access, or GPU use by default.

With explicit opt-in, it runs one causal forward pass per selected case on the original model, with the *correct reference file tokens supplied as preceding context*. It collects per-case and token-weighted per-language totals for:

- **Unmasked reference negative log-likelihood (nats per target token).** This uses the full 32,768-slot original model output distribution, with no token masking.
- **Top-1 reference-token accuracy, under gold conditioning.** This is only the accuracy of the next-token prediction *given all prior correct reference tokens*. It is not generated file accuracy.
- **Probability mass within the 1,191 fitted tokenizer IDs.** A small value would indicate probability diverted to output IDs that cannot be decoded by this tokenizer.
- **Outside-fitted-vocab top-1 rate, invalid-control top-1 rate, and premature-EOS top-1 rate.** These separate model/codec mismatches from ordinary wrong-but-valid code tokens.
- **Conditional fitted-vocabulary NLL**, a clearly labeled *diagnostic ablation*: probability renormalized across the 1,191 fitted IDs. **Do not compare this masked probability directly with real unmasked generation loss or treat it as proof of a viable decoding workaround.**

All metrics are conditioned on *gold* source/edit continuations. Therefore they are **not an independent M8 model-generation benchmark**, must not be presented as exact-file success, and must never be used as training examples or to tune the frozen benchmark. The code does not apply the conditional normalization to generation.

The report contains counts, case IDs, token lengths, aggregate and by-language diagnostics, pinned input hashes, status, and peak CUDA reservation. It writes **only** a create-only `teacher-forced-report.json` in a new artifact directory, not changed weights or training data. M8 context is bounded to 1,024 positions and reference output to 96 tokens, matching the separately verified full-file capacity results.

## Independent regression tests

The CPU-only `test_plex600_m8_teacher_forced.py` exercises real PyTorch numerical scoring on synthetic logits, including reference/causal alignment, exact unmasked-vs-conditional likelihood, probability mass outside the fitted tokenizer, premature EOS, invalid control IDs, logit shift invariance, token-weighted aggregation, malformed inputs and no-load default CLI safety. It never constructs or loads the 600M model.

The test file is part of required Linux training CI and Windows portability CI. This checks **instrumentation**, not checkpoint capability.

## Invocation when the original local checkpoint exists

Plan-only; requires neither GPU nor weights:

```powershell
uv run --project training --no-sync python -m plex_training.plex600_m8_teacher_forced
```

Optional, explicit, read-only reference diagnostic against the original local M7 checkpoint:

```powershell
uv run --project training --no-sync python -m plex_training.plex600_m8_teacher_forced `
  --run-read-only-teacher-forced `
  --bundle-dir "training/artifacts/plex600-owner-authored-pilot-tokenizer-v1" `
  --bundle-manifest-sha256 "14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55" `
  --m7-report "PATH_TO_ORIGINAL_M7_REPORT.json" `
  --m7-weights "PATH_TO_ORIGINAL_M7_WEIGHTS.pt" `
  --output-dir "training/artifacts/m8-teacherforced-six-v1" `
  --max-cases 6
```

The output directory must not exist. Input report/weights paths above are placeholders, not asserted on-disk locations. The M7 checkpoint **is not available in GitHub**; the actual checkpoint diagnostic has **not been performed**.

## Scientific interpretation gates

Use a *small, explicitly selected* 6-case run first. If unmasked fitted-vocab mass is poor or greedy top-1 IDs are frequently outside the tokenizer, decoding and vocabulary capacity may dominate; changes to decoding require a separately versioned experiment and genuine generation metrics. If reference likelihood improves but exact generated files still fail, investigate exposure bias, original short-answer training format, full-file copying/identifier binding, and evaluation-time compute. If neither reference likelihood nor greedy next-token accuracy is useful, a better rights-reviewed source-conditioned file-edit curriculum may be warranted.

Do **not** infer improved coding ability from the mere availability of these metrics. No additional optimizer updates are authorized, R2 remains quarantined, and the final holdout remains sealed.
