# Plex Phase 2 — Everyday-language editing

**Status: in progress.** Training is paused.

## What works now

The built-in editor can understand a few additional ways to ask for the same safe change, including:

```text
Change the variable value to count
Call this variable count instead
Make value equal to 3
```

It checks each requested change against the supplied JavaScript before editing. It still shows changed spans, diff, undo, and safe save-as. Ambiguous or unsupported edits refuse rather than guessing.

## How to check it

```powershell
py -3.12 -I -S scripts/plex_phase2_eval.py
py -3.12 -I -S scripts/plex_phase2_smoke.py -v
```

The fixed **development** set has **33 cases: 20 expected edits/copies and 13 expected refusals**. The scorecard prints editor results and independent raw-model results. Its test cases are development checks, not an independent final holdout.

## What is not yet proven

The Plex 600M M11 model has not been shown to perform these requests reliably. The last reported local attempt returned no code at EOS. Until fresh raw results are supplied, both model comparison columns correctly display **not run**.

A separate, opt-in, inference-only comparison script exists at `scripts/plex_phase2_infer.py`. It uses the existing checkpoint, checks its digest, and writes new output files without training. Use the resulting JSONL files with `plex_phase2_eval.py --baseline-model-jsonl ... --alternative-model-jsonl ...` to measure raw model success separately.

**Phase 2 is not complete** until inference-only baseline/alternative measurements and reviewed end-to-end outcomes justify the roadmap exit gate.
