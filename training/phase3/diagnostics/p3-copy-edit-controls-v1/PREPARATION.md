# Phase 3 — copy-versus-edit control preparation

**Status: executable preparation only; no model run, training, or final holdout access.** This follows the October 17 **both-fail** replacement diagnostic (0/72 exact edits). It is **not** the formal P3-03 Insert and Delete milestone.

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

## Preparing and verifying an immutable manifest

From the repository root:

```powershell
python training/phase3/diagnostics/p3-copy-edit-controls-v1/prepare_controls.py
python training/phase3/diagnostics/p3-copy-edit-controls-v1/prepare_controls.py --verify
uv run --project training pytest training/tests/test_p3_copy_edit_controls.py -q
```

The first command creates `copy-edit-controls-v1.json` **only if the path does not exist**. Record its reported SHA-256 and commit the manifest unmodified before authorizing or generating any model responses. The second command checks byte-for-byte reproducibility and never rewrites it. On Windows, commit/freeze should include an explicit LF checkout rule for the final JSON to keep its hash portable, as in the prior P3 diagnostic.

The new tests run the builder and synthetic scorers only. They do not infer model behavior.

## Explicit result rules and limits

Score exact file bytes on every task. On edits, if the response has unchanged bytes before and after the one authorized source-literal span, extract the intervening candidate and check its literal value. If those bounding bytes are altered, report requested-value selection as **unscorable** rather than claiming an independent value-selection failure. On copy controls, require the *entire* file to match. Summarize per condition and language; report malformed or missing responses separately.

The strict candidate success thresholds are 6/6 exact copies, 18/18 minimal edits, and 18/18 context edits. If copying fails, review a source-copying representation; if copying passes but minimal edits fail, review target formation and request-value binding; if both pass but context edits fail, review preservation of unrelated code. All-pass performance would justify only fresh independent confirmation, **not** replacement of the failed P3-02 development gate.

**Blocked / not authorized:** The builder does **not** freeze a manifest automatically into Git, validate tokenized prompt lengths, load the fixed step-100 checkpoint, run an evaluation, modify training weights, or open the final project holdout. Before any evaluation, separately verify the 512-token budget with the frozen tokenizer, checkpoint identity, inference configuration, and an approved one-run execution/provenance contract. No evaluation permission is implied by green CI.
