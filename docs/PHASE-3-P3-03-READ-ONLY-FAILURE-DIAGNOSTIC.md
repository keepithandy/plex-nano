# P3-03 — Diagnose Frozen Development Failures (read-only)

**Status:** diagnostic tooling implemented; full 36-response owner-local analysis not yet executed or reported. **No training authorized.**

## Evidence carried forward from P3-02

- The 27,566,080-parameter model was trained for the one authorized **100-update** P3-02 experiment (reported **60,445** answer/EOS positions).
- Teacher-forced answer/EOS validation loss improved **10.5668395360 → 1.5892543557**.
- The step-100 evaluation endpoint remains SHA-256 `020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d`.
- The frozen greedy 36-task development evaluation failed: **0/36 exact files**, **0/36 unrelated-code preservation**, and **0/12 correct three-request contrast groups** (HTML/CSS/JavaScript each 0/12).
- All 36 generations reached EOS, decoded successfully and kept the full supplied prompt visible.

The owner displayed three real failure samples. An HTML account button requested `Confirm` but Plex generated a settings button with `Save`; CSS `.account-gap: 12px` produced `.search-gap: 16px`; JavaScript `accountRetries = 4` produced `settingsRetries = 5`. These support incorrect context/value binding in the displayed examples. They are **not sufficient to establish the cause across all 36**.

The owner also reported **7 unique HTML outputs among 12 tasks**, **5 CSS among 12**, and **4 JavaScript among 12**. These represent repeated complete outputs under different requests, but do not alone prove memorization, data contamination, or that a larger model is necessary.

## New standalone read-only command

From the Plex repository root in PowerShell, first pull the merged P3-03 implementation:

```powershell
git switch master
git pull --ff-only origin master

$report = Join-Path $HOME "PlexArtifacts\file-edits\p3-02-development-evaluation\development-result.json"

uv run --project training python -m plex_training.file_edit_replacements_diagnostic --report "$report"
```

This **does not start training or model inference**. It opens only the existing `development-result.json`, the frozen P3-02 development summary committed in `training/pretraining/`, and the deterministic 216 authored P3-02 row metadata from source. It re-scores all 36 original responses with the unchanged P3-02 scorer, rejects missing/changed rows, and checks provenance and original failed-gate aggregates before reporting diagnostics.

The output is JSON printed to the terminal; it **does not overwrite or create files**. It reports:

| Diagnostic | Interpretation (not causal proof) |
|---|---|
| Unique full-file output count per language | How much the responses vary across distinct requests |
| Repeated-output clusters and SHA-256 | Exact copies of generated output across separate requests |
| One/two/three distinct outputs per contrast group | Whether different requested values produce distinct code for the same input file |
| Requested replacement literal present | A loose necessary indicator of binding, **not** proof the value was applied at the right location |
| `account` context literal present | A loose signal of development source identity retention; partial text matches aren't complete preservation |
| Foreign context labels (`cart`, `profile`, `search`, `settings`, `gallery`) | Signs of source-name substitutions, not necessarily proof of retrieving training samples |
| Exact match to known train/validation input or target | Observable text equality to a known authored fixture, **not** proof of memorization |
| Exact copy of supplied development input | Detects unchanged-file completions |
| Raw per-record flags and output SHA-256s | Makes responses traceable without dumping every full file by default |

**Do not modify the frozen model or select a new checkpoint based on these diagnostics.** P3-03 is strictly an explanation/inspection milestone. The development set is not retrained upon, and the final project holdout remains sealed.

## Decision after analysis

Only after seeing the real 36-response analysis can we responsibly determine whether failure is dominated by: repeated templates, substitution of training-context names, incorrect requested literal values, copying/omission of supplied file content, or a combination.

Any future P3-04 request-binding change must be **separately specified, reviewed, and authorized** before training. A 27M → 60M scaling experiment is also a separate future decision—not an automatic response to the P3-02 failure.
