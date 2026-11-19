# M11 — Original M9 space-prompt read-only baseline

**Implementation:** CPU-only preflight and opt-in GPU inference. No training
or owner-approval creation. This baseline is **different** from M10's
18-case newline-delimited M9 baseline.

## Research objective

Before approving any M11 training, measure how the independently backed-up
M9 checkpoint responds to a literal **space** at the end of the copying
instruction, with the source content otherwise unchanged.

The exact frozen scope is the four M11 cases, in order:

- Training candidates (NOT approved for training by this evaluator):
  M11 score=0 and score=1 source variants.
- Visible development-only probes (NEVER used as training by this code):
  M11 value=0 and value=1 source variants.

The expected answer retains its terminal newline and is followed by
the model's EOS token. The evaluator generates each case under **both**
existing decoding policies: unmasked-v1 and fitted-vocab-v1.

**Result scope: 4 cases × 2 decoders = 8 generated output records.**

This report measures exact autonomous code copying and correct termination.
It does **not** claim source-logit contrast measurement, which is a
separate M11 diagnostic, and does not claim new model improvement.

## Default plan (no files, model or GPU)

    uv run --project training --no-sync python -m plex_training.plex600_m11_baseline

It prints the planned scope without reading the model checkpoint.

## CPU-only original-source audit

Run this in the WSL Ubuntu repository containing the **actual** frozen
tokenizer files and independent M9 backup:

    /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m11_baseline --audit-inputs --bundle-dir training/artifacts/plex600-fresh-tokenizer --bundle-manifest-sha256 14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55 --m9-report /mnt/c/Users/quali/Plex600M-Backups/m9/report.json --m9-weights /mnt/c/Users/quali/Plex600M-Backups/m9/weights.pt

The CPU-only audit uses the M11 admission code directly to authenticate
the independent M9 weights, frozen M7 tokenizer, four candidate prompts,
schedule identity, and source hashes. It checks round-trip tokenization,
the exact space boundary, two paired token changes, generation budget,
and that the development-only pair is outside the training schedule.

It makes **zero** optimizer updates and writes **no** artifacts.

## Opt-in read-only GPU inference

After reviewing the CPU audit and confirming the intended new output
directory does not already exist, execute the exact same source parameters
with **--run-read-only-baseline** and an unused output:

    /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m11_baseline --run-read-only-baseline --bundle-dir training/artifacts/plex600-fresh-tokenizer --bundle-manifest-sha256 14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55 --m9-report /mnt/c/Users/quali/Plex600M-Backups/m9/report.json --m9-weights /mnt/c/Users/quali/Plex600M-Backups/m9/weights.pt --output-dir training/artifacts/plex600-m11-m9-space-baseline-20261119

If that output exists, choose another new, unused folder under
training/artifacts. Do not delete or overwrite earlier artifacts.

The execution loads the original model-only M9 state for **inference
only**, emits the eight raw model outputs, and stores two create-only
files: report.json and responses.jsonl. It does not save weights or
optimizer state or run backward/training code.

The command has a cooperative 1800-second elapsed-time budget, checked
between inference cases. It is **not** a hard GPU subprocess kill
deadline; GPU driver hangs could exceed it.

## Exact report contract

The generated report is checked against the already-merged M11 CPU-only
admission validator, which independently recomputes its two decoder
scores from the raw serialized response JSONL.

Core receipt evidence:

- protocol: plex600-m11-m9-space-prompt-read-only-baseline-v1
- status: complete-m11-m9-space-read-only-inference
- selectedCaseIds: all four M11 candidates in exact order
- decodingPolicies: unmasked-v1 and fitted-vocab-v1
- responseCount: 8
- checkpointSha256: independently validated original M9 weights
- m9ReportSha256, frozen tokenizer/bundle hash, candidateSha256,
  scheduleSha256 and sourceHashes
- responsesSha256: exact SHA-256 of the generated JSONL bytes
- scores: the same results the independent M11 CPU admission recomputes
- optimizerUpdates: 0; checkpointWrites: false;
  trainingAuthorized: false; finalHoldoutUsed: false

Each response record has exactly: id, split, decodingPolicy, output,
terminationReason, generatedTokens, promptSha256 and checkpointSha256.

This receipt records actual model inference performed by the tool, but
local saved files alone do not cryptographically prove execution origin.
Passing the admission validator is evidence of source/receipt
**consistency**, not an independently repeated GPU inference run.

## Acceptance and next decision

The code is complete when both Linux and Windows CPU regression suites
pass. The actual M9 model baseline is **not complete** until the owner
executes the opt-in read-only inference on the original local checkpoint.

M11 paired training remains separately gated. Neither this PR nor this
baseline permits training or carries forward M10 approval.

Only after inspecting the new baseline output should we consider a
bounded M11 training worker, explicit owner authorization and rigorous
source-dependent 0/1 copying measurements.
