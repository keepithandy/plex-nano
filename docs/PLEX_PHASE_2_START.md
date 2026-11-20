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

## Run the real 600M comparison

From PowerShell in your current Plex checkout, run:

```powershell
.\scripts\Invoke-PlexPhase2.ps1 -Run
```

This sends **33 requests through two M11 prompt formats (66 inference-only generations)** on your existing Ubuntu/WSL CUDA installation. It does **not train**. Results are stored in a new `training/artifacts/plex600-phase2-m11-...` directory, with checkpoint and tokenizer identities and checksums protecting the scorecard.

The results distinguish exact model edits, correct explicit refusals, empty EOS, incorrect nonempty edits, and average generation time. If the run or integrity check fails, no model success is claimed. There is no need to manually edit any JSON output.

**Phase 2 is not complete** until inference-only baseline/alternative measurements and reviewed end-to-end outcomes justify the roadmap exit gate.
