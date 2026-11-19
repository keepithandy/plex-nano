# Plex Phase 1 — reproducible research baseline (first implementation)

**Status:** in progress. This module captures and verifies **local file evidence**.
It does **not** certify that any historical training or GPU inference really ran.
It does not load a model, train, change a tokenizer, rewrite a checkpoint, read
the final holdout, or grant approval for further model training.

## Problem and existing evidence

Plex's M7/M9/M10/M11 code includes historical reports, source-bound test
protocols, and strict model checkpoint handling. However, full weights and
generated run artifacts are intentionally kept beneath gitignored
training/artifacts/ or in other owner-controlled local directories.
GitHub source checkout alone cannot establish their existence or SHA-256.

Phase 1 needs an independent local inventory of actual evidence. Do not
create a registry from reconstructed or imagined hashes.

The first implementation supplies:

- A create-only, one-checkpoint-at-a-time v1 record containing actual
  bytes and SHA-256 for model weights, model source, frozen tokenizer
  manifest, training report, optional dataset manifest, and optional
  evaluation report and raw response JSONL.
- The owner-supplied, SHA-bound metadata spec for source commit, family,
  experiment ID, optimizer updates and parent lineage. These fields are
  declared metadata, not independently proven training history.
- A read-only audit that recomputes all bound hashes. Changed weights,
  metadata, tokenizer, report, or response bytes cause refusal.
- A read-only inventory that refuses duplicate IDs, mismatched parent
  hashes and cycles; missing parent entries are reported rather than
  silently treated as an established root lineage.
- A matched M10 M9-baseline-versus-M10-checkpoint receipt comparator,
  requiring all 18 exact M10 prompts and both frozen decoders. It checks
  receipt SHA-256 and case/decoder/prompt/checkpoint identity and reports
  changes in byte-exact-and-EOS outcomes by split. It never mixes M10
  newline-based prompts with M11 space-based prompts.

## Local registration workflow

From the repository root create a private local inventory, excluded from Git:

    mkdir -p training/artifacts/phase1/registry

Create training/artifacts/phase1/spec-m9.json by replacing the example
values below with the actual file paths in this checkout and verified
source commit. All paths must be relative to the repository root.
The source code and tokenizer files must exist locally. Do not upload
the model checkpoint or private data with the manifest.

The following is a shape example, not a real registration:

    {
      "schemaVersion": 1,
      "entryId": "m9-original",
      "experimentId": "m9-r3",
      "modelFamily": "plex-600m",
      "checkpointKind": "model-only",
      "checkpointPath": "training/artifacts/REPLACE_WITH_ACTUAL_M9/weights.pt",
      "modelSourcePath": "training/src/plex_training/plex600.py",
      "tokenizerManifestPath": "training/artifacts/REPLACE_WITH_ACTUAL_TOKENIZER/manifest.json",
      "trainingReportPath": "training/artifacts/REPLACE_WITH_ACTUAL_M9/report.json",
      "datasetManifestPath": null,
      "evaluationReportPath": null,
      "evaluationResponsesPath": null,
      "sourceCommit": "REPLACE_WITH_ACTUAL_40_HEX_COMMIT",
      "optimizerUpdates": 8,
      "parentEntryId": null,
      "parentCheckpointSha256": null
    }

The null parent fields mean unknown/unregistered parent, not proof that
this checkpoint originated from scratch. Only bind a parent entry after
confirming its actual checkpoint SHA-256. The M9 step count above is only
illustrative; obtain the actual count from the historical report.

Safe commands (on Windows use backticks for line continuation if desired):

    uv run --project training python -m plex_training.plex600_phase1_registry
    uv run --project training python -m plex_training.plex600_phase1_registry \
      --capture --root . \
      --spec training/artifacts/phase1/spec-m9.json \
      --output training/artifacts/phase1/registry/m9-original.json
    uv run --project training python -m plex_training.plex600_phase1_registry \
      --verify --root . \
      --manifest training/artifacts/phase1/registry/m9-original.json
    uv run --project training python -m plex_training.plex600_phase1_registry \
      --inventory --root . \
      --registry-dir training/artifacts/phase1/registry

Capture fails if its output already exists. Do not edit a captured record
to reflect altered source files; create a new entry with a new ID and
preserve the old evidence. The audit refuses symlinks, traversal,
unsupported schema, missing evidence and oversized files.

The registry itself belongs in local/backup storage unless the owner has
reviewed its metadata for publication. Merely storing hashes in an
untrusted JSON file does not cryptographically attest to a machine's
training history.

## Matched comparison of read-only generation receipts

Use real M10 baseline and per-checkpoint receipt directories. Each
directory must contain report.json and responses.jsonl:

    uv run --project training python -m plex_training.plex600_phase1_registry \
      --compare-m10 \
      --baseline-dir training/artifacts/ACTUAL_COMPLETE_M10_BASELINE \
      --candidate-dir training/artifacts/ACTUAL_M10_RUN/step-10

The evaluator requires the exact complete 18-case × 2-decoder M10 scope.
It validates the checkpoint hash named in each receipt, all M10 prompt
hashes, split, language, decoder, output termination and JSONL file hash,
then scores differences separately for train and heldout candidates.

Results mean receipt integrity and recomputed outcomes, not proof
that model generation occurred on the claimed machine, not authorization
for training, and not proof of general-purpose coding ability. Differences
between mismatched prompt protocols are explicitly refused.

This comparator is deliberately M10-specific. M11 needs a separate
read-only baseline generator and M11-specific protocol validation before
M9/M11 comparisons can be treated as legitimate.

## Phase 1 completion and remaining work

Implemented in this PR: CPU-only checkpoint evidence capture, read-only
reverification, lineage inventory, and M10 matched-outcome comparison,
with Linux and Windows regression coverage.

Still required before marking Phase 1 complete: capture the actual
owner-local M7/M9 and M10 checkpoint evidence; establish the real parent
lineage wherever available; link historical training receipts to immutable
entries; and produce a locally verified historical M9-vs-M10 comparison.
No claim of completing these user-local validations is made here.

We also need a standardized experiment-decision ledger and broader
cross-checkpoint comparison support as the next Phase 1 increment. A
complete M11 read-only baseline remains the next model-evaluation
implementation once Phase 1 identity evidence is reliable.
