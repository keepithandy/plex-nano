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

The owner ran the verified-output Phase 2 script on **November 20, 2026**. The output reported that the **built-in editor** passed 20/20 edits and 13/13 refusals, but the **raw Plex 600M** failed every request:

- Baseline prompt: **0/20 correct edits**, 0/13 explicit refusals, **33/33 empty EOS** (mean 26.0 ms).
- Alternative prompt: **0/20 correct edits**, 0/13 explicit refusals, **33/33 empty EOS** (mean 17.7 ms).

The original local evidence remains at `training/artifacts/plex600-phase2-m11-20261120-230232-36012` on WSL. These figures are transcribed from the owner's console report, not from a new GPU run by GitHub. No model improvement has been demonstrated.

## Run the real 600M comparison

From PowerShell in your current Plex checkout, run:

```powershell
.\scripts\Invoke-PlexPhase2.ps1 -Run
```

This sends **33 requests through two M11 prompt formats (66 inference-only generations)** on your existing Ubuntu/WSL CUDA installation. It does **not train**. Results are stored in a new `training/artifacts/plex600-phase2-m11-...` directory, with checkpoint and tokenizer identities and checksums protecting the scorecard.

The results distinguish exact model edits, correct explicit refusals, empty EOS, incorrect nonempty edits, and average generation time. If the run or integrity check fails, no model success is claimed. There is no need to manually edit any JSON output.

## Diagnose the immediate EOS responses

The next experiment checks **four familiar M11 lines** (two training and two development examples), each using three input templates: original M11 `Copy: `, one-delimiter-change `Copy:\n`, and today's console `Request:/Source file:/Edited file:` format. It records first-token EOS ranking and greedy results without modifying the model.

After updating your WSL checkout, run this **in Windows PowerShell** from the Plex checkout:

```powershell
.\scripts\Invoke-PlexPhase2Eos.ps1 -Run
```

The script runs **12 inference-only probes**, prints a short diagnosis, and saves complete results to a new directory under `training/artifacts`. No optimizer, training, checkpoint overwrite, or final holdout is involved. The first-token logit checks and output-report code are covered by Windows and Linux CPU-only tests; actual GPU behavior must be measured locally.

## What the completed M11 diagnostic found

The owner's follow-up GPU run measured **2/2 exact familiar training copies**, **0/2 exact unseen copies** with the original M11 prompt. A one-character final-delimiter change and the regular console prompt both caused **4/4 immediate EOS**. This confirms strong prompt sensitivity but does **not** establish general code understanding. The numeric EOS logits remain in the local raw report; they have not yet been individually audited in this conversation.

For the **final read-only M11 evidence check** (no model generation, no training), use the documented command in [M11 final investigation and M12 proposal](PLEX_M11_FINAL_INVESTIGATION_M12_PLAN.md). The proposed M12 curriculum explicitly requires owner-authorized data, unseen-identifier evaluation and separate training approval.

**Phase 2 remains open.** No new M12 training is authorized, and the 600M checkpoint remains unchanged.
