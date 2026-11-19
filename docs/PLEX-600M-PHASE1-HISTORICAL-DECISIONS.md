# Plex 600M — Phase 1 historical experiment-decision ledger

**Ledger version:** 1 (November 19, 2026).
**Evidence class:** committed narrative reports. These are not a new local
validation of original binary checkpoints or response files.

## Decision standards

A model advances on autonomous, exact, source-conditioned output, not just
lower training loss, a correct EOS token after a supplied gold prefix, or
success on previously seen examples. M10's six evaluation-only records
are not the project's untouched final holdout.

Do not compare M10 newline-boundary input with M11 space-boundary input as
if the two were identical test protocols.

## M9 — eight-update R3 pilot

- Model weights: documented M9 independent-backup SHA-256
  e913be27855548a8f7dea766a2d367c7eaea357014ca82c80dec088958dfb123
- Reported predecessor model checkpoint SHA-256:
  69a31b4e710b284581a0b74c9e2661cc27bf134d24b8b1fe7a906f6efc004224
- Eight optimizer updates; model-only checkpoint.
- Subsequent M10 read-only copying baseline did not establish usable
  autonomous exact coding.
- Decision: **retain as frozen original checkpoint, not a capable coding release**.

Source: [M10 warmup diagnostics](PLEX-600M-M10-WARMUP-DIAGNOSTICS.md).

## M10 — 50-update exact-copy diagnostic

- Historical recorded local output name: plex600-m10-approved-20261118-182055.
- Dataset: twelve training and six evaluation-only HTML/CSS/JS snippets.
- Snapshots at optimizer updates 10, 25, 50.
- Reported gold-prefix correct content predictions:
  - M9: 1/286
  - M10-10: 27/286
  - M10-25: 24/286
  - M10-50: 35/286
- Reported correct terminal EOS after entire gold answer:
  - M9: 0/18
  - M10-10: 12/18
  - M10-25: 18/18
  - M10-50: 18/18
- Reported autonomous exact copies in focused warmup probes: 0/18
  at every recorded model checkpoint.
- Correct bidirectional source-substitution preference: 0/18 pairs
  at every checkpoint.
- Decision: **do not promote**. Terminal EOS improvements and some
  teacher-forced token gains did not demonstrate source-conditioned copying.

Source: [M10 warmup diagnostics](PLEX-600M-M10-WARMUP-DIAGNOSTICS.md).

## M10 matched EOS-loss ablation

- Compared original single-example ten-update warmup to a matched arm
  removing only the terminal EOS loss.
- Removing EOS loss prevented early immediate-stop collapse but produced
  18/18 length-cap generations at update 10 and 0/18 exact copies.
- Decision: **diagnostic only**; no qualified coding model produced.

Source: [M10 matched replay](PLEX-600M-M10-MATCHED-REPLAY.md).

## M10 space-prompt replay

- Compared ten-update experiments with newline and space final prompt
  delimiters while holding the response target and optimizer fixed.
- In the space-prompt condition at update 10, immediate EOS was avoided
  but the model emitted the same constant response on all 18 cases,
  with 0/18 exact copies.
- Decision: **prompt-delimiter sensitivity confirmed, copying not learned**.

Source: [M10 space replay](PLEX-600M-M10-SPACE-REPLAY.md).

## M11 paired source-conditioning

- Design and CPU-only training primitive/admission implemented.
- Actual M11-specific read-only M9 baseline: not implemented.
- Bounded M11 GPU worker: not implemented.
- Owner-approved M11 training: not performed.
- Planned pairs: score 0/1 for training, value 0/1 for development only.
- Planned EOS arms 1.0 and 0.25, checkpoints 0/25/50/100/200.
- Decision: **blocked pending new baseline, GPU worker and owner approval**.

Sources: [M11 proposal](PLEX-600M-M11-SOURCE-CONDITIONING-PROPOSAL.md)
and [M11 admission](PLEX-600M-M11-ADMISSION-GATE.md).

## Evidence status and next action

The above findings are source-backed historical reports, not newly audited
local checkpoint contents. Large checkpoint and generation files are
gitignored. Do not mark Phase 1 complete merely by merging this ledger.

Run the Phase 1 local historical registration command on the actual files
to produce a checkpoint registry, source-matched response comparison, and
read-only experiment-decision record; retain the original checkpoint bytes.

Then implement the M11 read-only M9 baseline before deciding whether to
run a bounded, separately approved source-conditioning experiment.
