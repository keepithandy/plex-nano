# Plex 600M — R3 source-conditioned file-edit curriculum

**Date:** 2026-11-16. **Status:** 111 new candidate examples prepared; **not approved for model training**. This is a concrete curriculum and target-only mask implementation, not a claim that the 600M checkpoint was improved yet.

## Purpose

The first M7 four-update run used 120 short `Request:\nAnswer:` examples. None contained the `Source file:\nEdited file:` contract that Plex is now expected to solve. Repeating that old training unchanged cannot directly address its file-edit objective. The new R3 path is built around the exact user-provided-file editing interface that M8 tests.

## Actual new data

[Candidate JSONL](../training/phase3/candidates/plex600-r3-source-conditioned-v1.jsonl): **111 complete source-file → edited-file examples** organized into **37 semantically distinct edit families**, with **three controlled context variants each**. These are not 111 independent tasks: the effective base task diversity is 37.

- **HTML:** 12 families / 36 rows — form handling, numeric constraints, required inputs, external links, heading semantics, image formats, page theme, dialog state, select options, downloads, mobile input methods and caption tracks.
- **CSS:** 13 families / 39 rows — layout padding, overscroll, typography, object fit, overflow ellipsis, grid shrinkage, box sizing, text wrapping, caret color, scroll offset, grid alignment, portrait ratios and tabular figures.
- **JavaScript:** 12 families / 36 rows — array quantifiers, finite numeric checks, integer parsing, mutation-safe copies, numeric sorting, inclusion checks, mutable state, debouncing, waiting for promises, URL encoding, event selection and queue order.

Each example has a natural-language request, original multi-line file, corrected full file, exactly one declared substring edit (including one compound span for listener cleanup), language/family, provenance, and explicit training denial. Original scaffold lines are preserved by exact construction; the validator refuses malformed operations or exact M8 benchmark edit/input/answer overlap. This is **assistant-authored candidate content**, not owner-reviewed production code, independent-license clearance, or evidence of semantic non-contamination.

The family-grouped deterministic proposal split is **84 training candidates from 28 families / 27 validation candidates from 9 families**; three context variants of one operation always stay together. These are internal proposed splits for a future experiment; the earlier frozen M7 internal validation and frozen 36-case M8 benchmark remain separate. Since all records are synthetic, even the R3 validation split is not a real-world coding benchmark.

## Training-ready engineering, not automatic training approval

The [R3 candidate tooling](../training/src/plex_training/plex600_r3_curriculum.py) implements:

1. Verification of all 111 rows, source status and exact before/after preservation; candidate hash and independent M8 exact-operation exclusion.
2. Grouped, fixed family validation selection rather than random variants of the same operation split across train/validation.
3. Reproduction of the **exact owner M7 1,191-token tokenizer** from only its original permitted training source, with historical tokenizer/bundle SHA checks.
4. **Answer/EOS-only causal labels:** model inputs are original request+source+previous gold target tokens; all prompt/source prediction positions use `-100`, and the final EOS label is supervised. No source/file context is scored as a training target.
5. A strict 1,024-token context bound and exact tokenizer roundtrip checks; no truncated or silently dropped examples.
6. An opt-in create-only packed **quarantined candidate** bundle with actual token IDs, loss targets, grouped split metadata, provenance hashes and explicit `trainingAuthorized: false`, `authorizingApprovalRecord: null`.

This bundle is compatible in *token ID space* with the historical M7 model, but compatibility does not approve the data, ensure 16GB VRAM feasibility for full-file sequences, or solve the inability to restore the M7 optimizer state (the owner-reported checkpoint is weights-only).

## Reproducible commands (no GPU)

Plan-only, candidate and benchmark audit:

```powershell
uv run --project training --no-sync python -m plex_training.plex600_r3_curriculum
```

Reconstruct and verify exact M7 tokenizer, then measure all 111 actual causal sequences without training or saving a new tokenizer:

```powershell
uv run --project training --no-sync python -m plex_training.plex600_r3_curriculum --measure-frozen-tokenizer
```

Create an opt-in *unapproved* local packed candidate for subsequent review, using a fresh output path below `training/artifacts`:

```powershell
uv run --project training --no-sync python -m plex_training.plex600_r3_curriculum --pack-quarantined-candidate training/artifacts/plex600-r3-proposal-v1
```

All three commands keep `optimizerUpdates = 0`; the second and third fit **the original M7 tokenizer only**, with exact historical hashes, and never fit on R3 candidate data.

The CI tests cover data validation, M8 exact-overlap exclusion, variant grouping, target-only supervised label alignment, and create-only packaging. Linux and Windows independently measure the complete candidate against the frozen tokenizer.

## What must happen before weights improve

1. Inspect this new corpus for incorrect semantics, repeated shortcuts, overly easy "only one token" edits and template generalization. Add more genuinely independent source-conditioned examples, not merely more cosmetic variants.
2. Record a **new explicit owner approval** for the exact R3 source snapshot and experimental optimizer-update scope, distinct from the old M7 approval. No historical M7 approval is reused or forged.
3. Read the original local M7 model-only checkpoint and pinned original report. Before a bounded fine-tune, measure full-sequence VRAM/runtime on the RTX 4080 SUPER, and ensure a strict execution limit and a separate predeclared success gate. No local GPU/checkpoint was accessible from this GitHub-only work.
4. Run a small, bounded supervised fine-tune with answer-only labels, preserve baseline and new checkpoint separately, and compare **actual generated edits** (not only teacher-forced loss) on the untouched M8 development cases under the explicit 96-token output cap and separate decoding policies.

No R3 candidate has entered any checkpoint. No new model capability claim is justified until a real model is trained and independently scored.
