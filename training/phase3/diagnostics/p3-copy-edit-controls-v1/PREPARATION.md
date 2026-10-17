# Phase 3 — copy-versus-edit control preparation

**Status: fixed 42-task manifest frozen, local inference/scoring gates prepared; no model run, training, or final holdout access.** This follows the October 17 **both-fail** replacement diagnostic (0/72 exact edits). It is **not** the formal P3-03 Insert and Delete milestone.

## Question and design

The previous 72-task failure lost all supplied source-name roles and failed to produce requested target literals. We need to determine whether a simpler complete-file output contract already fails at exact copying, at modifying one target literal, or only when unrelated context is added. Neither a failed copy control nor a failed edit proves an internal model mechanism.

The `prepare_controls.py` builder constructs 42 deterministic prospective tasks:

| Condition | HTML | CSS | JavaScript | Total |
|---|---:|---:|---:|---:|
| Unchanged-file copy | 2 | 2 | 2 | 6 |
| Minimal single-literal edit | 6 | 6 | 6 | 18 |
| Same edit with unrelated context | 6 | 6 | 6 | 18 |

Each language uses one replacement family and two fresh namespaces (`cobalt`, `spruce`). Each edit namespace has the *same supplied source* with three distinct request values, permitting contrast sensitivity checks. The prompt serialization matches the P3-02 request + supplied-file template. Each expected edited file changes exactly one literal; the copy controls return the file unchanged.

The builder verifies the exact P3-02 authored candidate and 72-task P3-03 manifest SHA-256 identities, checks all new input/expected complete-file texts against the prior **P3-02 training** and **72-task diagnostic** source/target sets, and refuses any overlap. Existing manifests cannot be overwritten. It uses no ML library or model resources.

## Freeze and verification

The immutable candidate is already committed. Do **not** rerun the builder's create command on the frozen path; it intentionally refuses overwrites. Verify its deterministic bytes with `python training/phase3/diagnostics/p3-copy-edit-controls-v1/prepare_controls.py --verify`, and run the regression suite. The recorded SHA-256, context-budget checks, and local execution instructions appear in the October 17 addendum below.

## Explicit result rules and limits

Score exact file bytes on every task. On edits, if the response has unchanged bytes before and after the one authorized source-literal span, extract the intervening candidate and check its literal value. If those bounding bytes are altered, report requested-value selection as **unscorable** rather than claiming an independent value-selection failure. On copy controls, require the *entire* file to match. Summarize per condition and language; report malformed or missing responses separately.

The strict candidate success thresholds are 6/6 exact copies, 18/18 minimal edits, and 18/18 context edits. If copying fails, review a source-copying representation; if copying passes but minimal edits fail, review target formation and request-value binding; if both pass but context edits fail, review preservation of unrelated code. All-pass performance would justify only fresh independent confirmation, **not** replacement of the failed P3-02 development gate.

**Remaining runtime dependency:** The repo can freeze and verify the task manifest without weights, but running the model requires the owner's local P3-02 checkpoint and frozen tokenizer bundle. The guarded runner checks the real tokenizer, checkpoint, and inference policy locally. No optimizer updates or final-holdout access are allowed.

## October 17 freeze and execution gate

The 42-task manifest is now **frozen** at [`copy-edit-controls-v1.json`](copy-edit-controls-v1.json), with SHA-256:

```text
17834efb801ac963a4672ae616ca47e9eea40b2568f01951c3891c7e4997f01a
```

The builder's exact serialized output is checked against the committed manifest in CI. Its 42 tasks, 12 three-request edit contrast groups, 144 prior P3-02 training rows, and 72 prior source-preservation diagnostic rows are verified without mutating those sources. All four cross-set complete-file overlap counts are zero. The maximum serialized ASCII prompt is **307 bytes**. Even with one token per source byte, a 192-token output allowance and one additional token for a final EOS-producing model evaluation, this fits within the fixed **512-token context** (307 + 192 + 1 = 500). The owner's *actual frozen tokenizer* must still be loaded and checked for its SHA-256 and prompt round-trip before any model evaluation.

The earlier text above documents the **pre-freeze** state; do not regenerate or overwrite the manifest. Use verification instead:

```powershell
python training/phase3/diagnostics/p3-copy-edit-controls-v1/prepare_controls.py --verify
uv run --project training python -m pytest training/tests/test_p3_copy_edit_controls.py -q
```

The local **preflight** checks the private P3-02 step-100 checkpoint file, local trainer result, P3-02 bundle, frozen tokenizer identity, exact prompt encoding, and fixed token budget, without generating model outputs:

```powershell
uv run --project training python training/phase3/diagnostics/p3-copy-edit-controls-v1/run_frozen_evaluation.py
```

The checkpoint is expected under `$HOME\PlexArtifacts\file-edits\p3-02-first-run\checkpoints\step-0100.pt`, and the matching bundle under `$HOME\PlexArtifacts\p3-02-training-bundle`. Their files are intentionally **not in GitHub**, and repository CI cannot attest to or execute inference with those private weights. The preflight must return `local-preflight-passed-no-generation` before proceeding.

A **single, separately authorized** local CUDA inference run is an explicit CLI action; it never trains or reselects weights. The owner authorized proceeding with the controlled experiment in the October 17, 2026 conversation. Record that authorization accurately when executing; do not use CI success as authorization:

```powershell
uv run --project training python training/phase3/diagnostics/p3-copy-edit-controls-v1/run_frozen_evaluation.py --run --authorization-reference "Owner approval, 2026-10-17: run the controlled 42-task copy-versus-edit evaluation"
```

The runner refuses a dirty source checkout, changed hash, unavailable CUDA, prompt eviction, invalid checkpoint payload, and any pre-existing result directory. A successful single run publishes `$HOME\PlexArtifacts\file-edits\p3-copy-edit-controls-v1-evaluation\responses.json`, including 42 frozen task IDs, decoded responses, EOS status, full-prompt-visibility records, authorization evidence, the exact code revision and runner SHA-256. No historical P3-02 or P3-03 output is overwritten.

Finally, run the **model-free scorer** on the preserved response envelope, writing a separate results directory:

```powershell
uv run --project training python training/phase3/diagnostics/p3-copy-edit-controls-v1/score_frozen_responses.py --responses "$HOME\PlexArtifacts\file-edits\p3-copy-edit-controls-v1-evaluation\responses.json" --output-dir "$HOME\PlexArtifacts\file-edits\p3-copy-edit-controls-v1-score"
```

The scorer refuses changed provenance, missing/duplicate/invalid task responses, unapproved training, prompt eviction, and any existing output directory. Its Markdown report and JSON audit display exact-file correctness by language/condition, requested values only where target boundaries survive, EOS status, complete contrast groups, and the predeclared decision branch. Synthetic regression tests establish scorer behavior only; **no empirical 42-task model result exists until the local run is verified**.

No optimizer updates, model-size changes, checkpoint reselection, P3-03 Insert and Delete research, or final-holdout access were approved by this preparation.
