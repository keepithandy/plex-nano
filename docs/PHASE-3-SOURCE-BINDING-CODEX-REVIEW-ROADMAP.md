# Plex Nano — Codex Review Roadmap: All Open Issue Gates

**Prepared:** October 17, 2026  
**Repository:** `keepithandy/plex-nano`, branch `master`  
**Issue audit:** **One open issue, [#189 — P3-SB Source-Binding Recovery](https://github.com/keepithandy/plex-nano/issues/189).** No other open repository issues or PRs were returned by the GitHub audit.  
**Predecessor:** [#181 — FAILED capability, research completed and closed](https://github.com/keepithandy/plex-nano/issues/181).  
**This file is the Codex-facing implementation/review plan.** It is *not* a new authorization to execute model inference, train, modify frozen datasets/checkpoints, or open the final holdout.

**Technical review update (October 17):** [Recorded findings, actual SB-00 measurements, tokenizer audit and SB-02–05 specifications](PHASE-3-SOURCE-BINDING-TECHNICAL-REVIEW.md). SB-00 local evidence is now available; SB-01 bytes remain unchanged. No new model execution is authorized or claimed.

## What "green-lit" means here

**GREEN — work may proceed:** read-only inspection, source-controlled analysis/scripts, deterministic microprobe *preparation*, CI-only synthetic tests, technical design/specification, provenance checks, documentation and PR review. The implementation must never change the original experiments or make claims of observed model improvement.

**GATED — design may be reviewed, but execution is not approved:** any newly generated live model responses, new staged checkpoint or bundle, optimizer updates, alternative checkpoint selection, new train/validation split, promotion into untouched final holdout, or changes to model/tokenizer/context. Each experiment requires an explicit scope, frozen artifacts/criteria, a fresh owner decision, and independent review.

There is **one actual open GitHub issue** (#189), divided into six milestone work packages. A milestone can be *technically prepared* without its experimental execution being authorized.

## Fixed research facts (never overwrite)

| Item | Known verified value |
|---|---|
| Model | 27,566,080 parameters, 16,384 vocabulary, 512 context; single supplied HTML/CSS/JS file |
| Fixed first-run endpoint | Step 100, SHA-256 `c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230` |
| Authorized completed training | 100 updates, 40,477 answer+EOS targets, 102.9443 s; no further training permission |
| Teacher-forced validation | 7.1167174 → 2.7984009 (not proof of exact copying) |
| Complete-file validation | 0/24 copies, 0/72 edits, 0/24 complete groups |
| Original inspected development controls | 0/6 copies, 0/36 edits (0/42 exact) |
| EOS / full prompt visibility | 138/138 |
| Source-preservation failures | 138/138; six sample outputs replaced unseen `granite` with training names such as `meadow`, `prism` |
| Final project holdout | SEALED; do not open |
| Status of experiment #181 | CLOSED **with failed capability**, not success |

Older 96-case validation rows influenced loss monitoring, and original 42 developer controls were inspected. **Neither set is an untouched independent final holdout.** No same-set pretraining generation results exist for the 96-case validation set. Do not invent an improvement claim.

## Issue #189 — SB-00 through SB-05

### SB-00 · Audit the existing 138 saved outputs

**Review status: GREEN for read-only owner-local diagnosis; code and synthetic CI already implemented.**

**Files:** `training/phase3/copy-fidelity-v1/diagnose_saved_outputs.py`, `training/tests/test_p3_source_binding_diagnostic.py`.

**Codex tasks:**
- Review file read behavior, SHA verification, case ordering and reconciliation against the original full-file scorer.
- Confirm that *identifier substring appears* is labeled only as a heuristic, never exact source preservation.
- Keep literal-mentioned metrics distinct from independently extractable/correct requested values.
- Check duplicate complete outputs, EOS/decode/prompt visibility, token/character length, empty outputs and per-language/condition/context breakdowns.
- Verify there are **no writes to original report, checkpoints, tokenizer, datasets or final holdout**; no torch import, no generation or training.
- Review synthetic fail-closed tests for altered summaries, scrambled records and inconsistent scorer fields.

**Exit criterion:** diagnostic validated by CI and measured on the saved local `evaluation-result.json`; record `inputReportSha256`, full case count and observed morphology. **Local measurement is now complete**, recorded in the [technical review evidence](../training/pretraining/p3-source-binding-technical-review-evidence.json). Issue #189 remains open for the experimental decisions and unresolved capability gates.

**Read-only owner command (after pulling merged master):**

```powershell
uv run --project training python training/phase3/copy-fidelity-v1/diagnose_saved_outputs.py
```

### SB-01 · Freeze the 48-case source-binding microprobe (PREPARATION ONLY)

**Review status: GREEN for deterministic fixture design, generation, SHA and synthetic regression tests. No checkpoint inference authorized.**

**Files:** `training/phase3/copy-fidelity-v1/prepare_source_binding_microprobe.py`, `source-binding-microprobe-v1.json`, `training/tests/test_p3_source_binding_microprobe.py`.

**Actual prepared proposal:** 3 languages × 2 **new names** (`ulvar`, `nyxen`) × 2 contexts (minimal / unrelated context) × 4 requests (exact unchanged copy + 3 distinct literal edits). This yields 48 records: 12 copies, 36 edits and 12 four-case contrast groups.

**Proposal fixture SHA-256:** `0a30df999f8e1c0f12cf0f6e4a71e0eab814bdd99b3bd83d0e2184637ad31fda`. It is **not a model evaluation result**.

**Codex review tasks:**
- Confirm all four historical candidate/control files are SHA-pinned, all raw prior files exclude both new source namespaces, and no new input/expected file duplicates any previous input or target.
- Verify exact copy/edit source-to-output relation, 4-case groups, unique source literal, fully disjoint name pairs, 192-token conservative output budget and 512-token full-prompt context.
- Ensure the committed manifest is byte-identical to deterministic `serialized()` and LF-preserved on Windows/Linux. Do not adjust fixed rows after any future model output.
- Verify `evaluationAuthorized=false`, `trainingPerformed=false`, `optimizerUpdates=0`, `finalHoldoutOpened=false`.
- Freeze (not retrospectively optimize) the proposed gates: 12/12 exact copies, ≥30/36 exact edits, ≥10/12 per language, ≥9/12 full four-case contrast groups, byte-exact preservation of unrelated source.

**Exit criterion:** deterministic proposal SHA and CI tests pass. The owner still reviews the proposal before **any** live model generation on its 48 cases.

Actual frozen-tokenizer verification is recorded in the technical review: all 48 rows roundtrip within 512/192 bounds, with answer/EOS target-mask checks. Historical exclusion is established for the four pinned P3 inputs, not every earlier pretraining corpus.

**Model-free verification:**

```powershell
uv run --project training python training/phase3/copy-fidelity-v1/prepare_source_binding_microprobe.py
```

### SB-02 · Hypothesis and intervention design

**Review status: GREEN for design and evidence comparison; GATED for intervention choice or model execution.**

**Codex tasks:**
- Draft four explicit falsifiable hypotheses: (A) trained-name fallback; (B) weak copy objective; (C) source/request-value binding mismatch; (D) full-file output too demanding.
- For each, specify **one controlled independent variable**, a matched comparator, exact full-file outcome, negative-control examples and a result that would refute the hypothesis.
- Rank these **only after SB-00 real morphology data**. Do not infer mechanism from six illustrative samples or loss alone.
- Document any proposed source-span/patch alternative as an **independent contract and separate score**, never a retroactive pass on the full-file task.

**Exit criterion:** owner reviews and selects one hypothesis; **no training automatically follows**.

### SB-03 · Candidate dataset and stage design

**Review status: GREEN for a no-execution contract/checklist review; GATED for actual bundle, checkpoint staging or data transformation beyond already frozen SB-01 proposal.**

**Codex tasks:**
- Define a hypothetical reproducible candidate with manifest SHA, token-budget audit, train/validation/development namespace and full-file overlap checks, answer/EOS target masks and deterministic sampler specification.
- Require original model/tokenizer/config hashes, fresh artifact destination, no overwrites, no new final holdout accesses.
- Make clear that staging and data preparation never authorize optimizer steps; include precise owner-review boundaries.

**Exit criterion:** reviewable reproducibility specification; no stage/bundle created without separate approval.

### SB-04 · One bounded training experiment (conditional)

**Review status: GREEN for design of a future authorization gate; **NOT AUTHORIZED** for execution or enabling model-training flags.**

**Codex tasks:**
- Propose one bounded run with explicit number of optimizer updates, wall-clock cap, seed, device, data mix, learning rate, grad accumulation, checkpoint schedule, early aborts and non-overwrite/no-resume conditions.
- Show how execution refuses without **a fresh explicit owner approval** and exact independently SHA-pinned contract/stage/baseline.
- Preserve the current fixed step-100 endpoint as the historical benchmark. No retrospective promotion of the step-75 lower-loss checkpoint.
- Keep the 200 GB artifact budget, 27,566,080 parameters, 16,384 vocabulary and 512 context unchanged.

**Exit criterion:** an **unapproved** proposal ready for an owner decision. Do not train or suggest that this Codex review is permission.

### SB-05 · Complete-file generated capability gate (conditional)

**Review status: GREEN for pure scoring/benchmark specification and synthetic tests; **NOT AUTHORIZED** to generate fresh real-model answers on new probes or open final holdout.**

**Codex tasks:**
- Predeclare EOS, decode, full prompt retention, exact output bytes, source names, unrelated code, request-bound literal, contrast-group and per-language scores.
- Protect frozen microprobe manifest SHA and historical 96/42 control identities.
- Distinguish *unknown/unextractable requested value* from wrong value. Keep the raw per-case record and immutable output provenance.
- Require all SB-01 pass thresholds and track failures honestly. No checkpoint selection based on observed validation scores.
- Treat the final project holdout as sealed until a later separately named and explicitly authorized closeout.

**Exit criterion:** deterministic model-free scoring contract ready for separate authorization, *not* any model-capability pass claim.

## Review and merge gates for every green-lit code change

1. Start from the current `master` head; use a scoped branch and PR.
2. Inspect frozen evidence and SHA pins before code edits; no edits to previous raw results, checkpoints, old manifests or historically declared thresholds.
3. Linux `pytest` required workflow must pass; **Windows portability** required workflow must pass. Confirm the **latest PR-head** SHA and both green conclusions before merging.
4. Add regression tests for success and refusal: bad SHA, duplicate/scrambled rows, unsupported scope, unexpected writes, wrong split, target not extractable.
5. Use squash merge to `master` only after checks green. Confirm post-merge checks rather than extrapolating.
6. Mark only *completed tooling* green; keep owner-local evidence and future model execution gated. Maintain #189 OPEN until all required model-evidence milestones are actually satisfied.
7. Report files/commit/CI numbers and explicit **not performed** fields for training/inference/final-holdout access.

## Codex reviewer instruction (paste directly)

> Review Plex-Nano issue #189 and the current P3-SB source-binding recovery implementation against this roadmap. Validate the frozen experimental provenance and all no-training/no-inference authorization boundaries. Run the required Linux and Windows test suites, inspect all changed source-binding fixtures, hashes, scorer semantics and split contamination controls, then report P0/P1/P2/P3 findings with exact file/line references and suggested narrow fixes. For SB-00 and SB-01 preparation, fixes can be proposed and merged after both CI checks are green. For SB-02–SB-05, green-light **design and synthetic tests only**; do not execute model inference or training, select checkpoints, change frozen results or access final holdout. Treat the historical 0/138 generation performance as a FAILED capability experiment, not a successful model milestone. Keep issue #189 OPEN for future owner review.

## Handoff decision

**Green-lit now for code and research review:** #189 SB-00, #189 SB-01, and pure-documentation/synthetic-design review of #189 SB-02 through SB-05.

**Not green-lit as live experiments:** new responses from the real checkpoint on SB-01, new dataset/stage construction beyond the approved proposal, any optimizer updates, checkpoint reselection, and any final-project holdout access. These require explicit fresh owner authorization.
