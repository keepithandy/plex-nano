# Plex 600M — M5 preparation and measured M6 progress

**Status:** M5 adapter implemented; no real M4 rights approval, materialized corpus or 600M tokenizer bundle exists yet.

## Real measurement retained

The owner reported one successful 600M AdamW update on the RTX 4080 SUPER:
10.04 GiB peak GPU reserved, 0.1544 seconds inside the optimizer step and
436 finite optimizer moment tensors. The machine-readable
[owner-reported receipt](../training/pretraining/plex600-m6-owner-reported-optimizer-result-2026-10-31.json)
retains the measured values and distinguishes this report from an independent
rerun. The full original JSON remains in the owner's local artifacts.

## M5 adapter

The [M5 builder](../training/src/plex_training/plex600_m5.py) accepts only a
hash-verified M4 candidate and a reviewed rights decision whose exact SHA-256
matches the candidate's manifest. It refuses unknown sources and any changed
historical record or license evidence.

The builder deterministically partitions candidate records by family and
historical project group (seed 1337, 10% of groups for validation by default).
Each family must have at least two groups and appear in both splits. Identical
historical project groups cannot cross the split boundary. Source records and
license notices retain their hashes and identities. No historical Traversy or
Bulma validation corpus enters either split.

This validation is an INTERNAL held-out development split from M4 training
candidates, NOT a new independently collected coding benchmark or final
holdout. That independent benchmark still needs to be developed.

Create-only outputs are capped at 32 MiB and include train.jsonl,
validation.jsonl, manifest.json and licenses/*.txt. The dataset manifest has
the distinct pipelineVersion plex600-m5.0. The dedicated 600M tokenizer
accepts this format after verifying its family and preparation-only flags;
the frozen 27M tokenizer does not. Fitting trains the BPE only on train,
encodes both splits with source-bound hashes, and performs existing roundtrips.

## Commands after actual source-rights approval

These commands DO NOT create or invent source rights approval. See the
[M4 rights and materialization guide](PLEX-600M-M4-CORPUS-PIPELINE.md) first.

~~~powershell
uv run --project training --no-sync python -m plex_training.plex600_m4 build --historical-datasets-root training/artifacts/datasets --reviewed-decision PATH_TO_REVIEWED_DECISION.json --output-dir training/artifacts/plex600-m4-candidate-v1

uv run --project training --no-sync python -m plex_training.plex600_m5 --m4-candidate-dir training/artifacts/plex600-m4-candidate-v1 --reviewed-decision PATH_TO_REVIEWED_DECISION.json --output-dir training/artifacts/plex600-m5-dataset-v1

uv run --project training --no-sync python -m plex_training.plex600_setup tokenizer-train --dataset-dir training/artifacts/plex600-m5-dataset-v1 --output-dir training/artifacts/plex600-m5-tokenizer-v1

uv run --project training --no-sync python -m plex_training.plex600_setup tokenizer-check --bundle-dir training/artifacts/plex600-m5-tokenizer-v1
~~~

The additional source-check command takes SHA-256 identities for both the
packed bundle manifest and the M5 source manifest. The tokenizer checks
are not permission to begin optimizer training.

## Open gates

1. Actual source-use review and bounded local M4/M5 corpus preparation.
2. Independent HTML/CSS/JavaScript coding development evaluation and a
   documented separation from the internal validation and final holdout.
3. Measured token count and tokenizer roundtrip/source-binding report.
4. Explicit M7 bounded real-data training contract, separate from M6.

No new 600M loss curve, checkpoint, or coding evaluation is claimed here.
