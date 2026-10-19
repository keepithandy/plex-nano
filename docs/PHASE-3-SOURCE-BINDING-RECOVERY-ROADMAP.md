# Plex Nano — P3-SB Source-Binding Recovery Roadmap

**October 18 audit update:** the [merge-audit recovery roadmap](PLEX-NANO-MERGE-AUDIT-ROADMAP-2026-10-18.md) supplies the next work order. All saved capability failures replay unchanged. V1 token parity failed; V2 now matches 48 names across 384 paired token signatures against five P3 corpora in a read-only check. Full historical-source screening, reviewed V2 bytes and fresh model-execution decisions remain pending.

**Status:** REVIEWED RESEARCH PROPOSAL — SB-00 read-only diagnosis and SB-01 *design/preparation* technically green-lit. **No training, model reselection, final-project holdout, or new inference experiment is authorized by this document.** Owner roadmap review comes before any further experiment.

**Current model:** 27,566,080 scratch-trained parameters; 16,384-token vocabulary; 512-token context; supplied single HTML/CSS/JavaScript file plus a request.

**Predecessor:** [Completed P3 copy-fidelity experiment #181](https://github.com/keepithandy/plex-nano/issues/181). **Successor:** [P3-SB recovery issue #189](https://github.com/keepithandy/plex-nano/issues/189).

**October 17 technical-review update:** [SB-00 evidence and SB-01 tokenizer verification](PHASE-3-SOURCE-BINDING-TECHNICAL-REVIEW.md) are now recorded: 138/138 supplied namespaces absent, 138/138 training-name intrusions, 0/138 exact files. SB-02–04 have reviewable design proposals and SB-05 has a synthetic-only pure scorer. This adds no live evaluation or training authorization.

## 1. Decision record — what is known

The owner authorized **one** bounded P3 copy-fidelity run, which finished 100/100 optimizer updates in 102.9443 seconds, covering 40,477 answer/EOS supervision positions. Independently verified **fixed step-100** checkpoint SHA-256:

`c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230`

The frozen validation mean teacher-forced loss improved from **7.1167174 (step 0)** to **2.7984009 (step 100)**; step 75 had a lower validation loss, **2.6199503**, but the original run contract prohibits post-hoc checkpoint selection. Step 100 remains the **only** official endpoint.

The subsequent complete-file greedy test (seed 1337; temperature 0; max 192 generated tokens; no optimizer updates) **failed every capability gate**:

| Fixed set | Unchanged copies | Exact edits | Contrast groups | Source bytes preserved |
|---|---:|---:|---:|---:|
| 96 validation cases (held from gradients, used for loss monitoring) | **0/24** | **0/72** | **0/24** | **0/96** |
| 42 previously inspected developer controls | **0/6** | **0/36** | **0/12 edit groups** | **0/42** |

**All 138/138 outputs reached EOS and retained full prompt visibility.** Six owner-inspected sample outputs used the previously trained identifiers `meadow` or `prism` in place of the supplied validation identifier `granite`. One HTML edit preserved the requested `Start` literal while changing the identifier; another JavaScript copy preserved `2` while changing its identifier. These are observations **consistent with** source-binding failures and template-biased generation, not proof of a specific internal neural mechanism.

**Important interpretation:** A fall in teacher-forced loss does *not* demonstrate an improvement in exact copying/editing; neither the new validation set nor the old development set passed. The final project holdout **remains sealed**. The owner retains the full local per-record outputs; only the reported aggregates and six examples have been reviewed in GitHub.

## 2. Research goal and invariant contract

**Make the model copy *unseen* supplied identifiers and unrelated bytes exactly before requiring it to perform single-literal changes.**

Example:

```text
Supplied: const graniteRetries = 2;
Request: Change retry limit to 4; preserve the complete file.
Required: const graniteRetries = 4;
Forbidden: const meadowRetries = 13;
```

The names, values, request contrast, and unrelated context must remain **bound to the supplied input**. A plausible HTML/CSS/JavaScript template with different identifiers is an incorrect edit. Preserve the architecture, tokenizer, checkpoint provenance, language set, 512-token context, full-file output contract, and final-holdout policy.

## 3. Gate-by-gate proposed milestones

### SB-00 — Read-only morphology diagnosis — technically green-lit

**Available now:** `training/phase3/copy-fidelity-v1/diagnose_saved_outputs.py`. It processes the owner's *existing* `evaluation-result.json` and revalidates all 138 ordered responses against SHA-pinned candidate/control manifests, scorer boundaries and original aggregate totals.

Quantify per-scope, language, condition and contextual shape:

- source namespace *present in generated text* versus absent; known training-namespace intrusion; these are **lexical heuristics**, not proof of exact source copying;
- EOS, decoding, full prompt visibility, empty outputs, 192-token cap hits, code-shape proxy, length/token distributions, repeated complete output strings;
- correct requested value **only when the original source boundaries are extractable**; weaker literal occurrence is counted separately and never represented as a successful edit;
- exact-file outcome and unrelated-source preservation, cross-checked against the frozen scorer; source input-report SHA for independent owner review.

**Exit gate completed locally:** all 138 cases accounted for, original saved summary reconciled, report SHA recorded, no modified historical input files, failure families documented in the [technical review](PHASE-3-SOURCE-BINDING-TECHNICAL-REVIEW.md). The script itself produces no checkpoint, optimizer or output files, and does not import PyTorch. CI tests synthetic reports only. The existing report SHA is `971a78971387200df9523c6ca177006f60f2834be0664c2b590e6ec72f06a197`.

From the root of the local repository:

```powershell
uv run --project training python training/phase3/copy-fidelity-v1/diagnose_saved_outputs.py
```

Save the JSON output externally if needed, and supply the `inputReportSha256` and summary to the issue. **Do not delete or regenerate the original evaluation result.**

### SB-01 — Fresh source-binding microprobe design — *preparation* green-lit

**Prepared and SHA-frozen for review; do NOT run inference.** The [deterministic fixture builder](../training/phase3/copy-fidelity-v1/prepare_source_binding_microprobe.py) and [48-case proposal manifest](../training/phase3/copy-fidelity-v1/source-binding-microprobe-v1.json) are in CI with SHA-256 `0a30df999f8e1c0f12cf0f6e4a71e0eab814bdd99b3bd83d0e2184637ad31fda`. Source names `ulvar` and `nyxen` are checked against all four SHA-pinned historical candidate/control files. The model has not been run against this proposal. The fixture contains **48 model-free records**:

- 3 languages × 2 contexts (minimal / unrelated code) × 2 previously unused source-name namespaces × 4 contrast cases (unchanged copy + 3 different requested values);
- 12 unchanged-copy cases, 36 edits, 12 four-case groups;
- source spelling varied independently of requested value, with non-overlapping train, prior validation, development and diagnostic identifiers;
- source name retained exactly, adjacent HTML tags/CSS rules/JS declarations unchanged; complete-file whitespace and EOS included;
- render the exact request+file prompt; no hidden source shortening, generation truncation or sliding-window eviction.

The current exclusion check covers the four listed P3 files. A future claim of global training-history exclusion must enumerate and SHA-pin all historical training corpora, failing closed on missing history; the current check does not establish that broader claim. Do not read the final holdout to screen names. Seed and split decisions are frozen in a manifest with full-file SHA-256 and source-to-target relation proofs; never silently fold inspected developer controls back into a clean holdout.

**Proposed pass gates (must be frozen before inference):** 12/12 untouched copies, ≥30/36 exact edits, ≥10/12 exact edits for each language, ≥9/12 complete four-case groups, 100% correct source-namespace preservation on passed files. Also publish failure counts and request-conditioned contrast sensitivity. No post-hoc scoring-rule changes.

**Exit gate:** deterministic builder, manifest SHA, contamination guard and synthetic tests are implemented; owner review and any live generation authorization remain pending. Generation on the frozen probe requires a separate decision after owner review.

The actual frozen-tokenizer [read-only audit](../training/phase3/copy-fidelity-v1/review_source_binding_microprobe.py) also passes: maximum 95 prompt tokens, 50 answer/EOS tokens and 145 combined tokens; all 48 roundtrip exactly.

**Reproduce the 48-case SHA/structure (read-only):**

```powershell
uv run --project training python training/phase3/copy-fidelity-v1/prepare_source_binding_microprobe.py
```


### SB-02 — Intervention choice approved for preparation (hypothesis A)

The owner approved the next technical development step: **deterministic training-namespace diversity** (hypothesis A), not a new optimizer run or model evaluation. [SB-03 candidate and parity-readiness detail](PHASE-3-SB03-APPROVED-NAMESPACE-DIVERSITY.md) is separately documented with complete SHA identities and required owner-local checks.

### Original competing intervention proposals — preserved decision record

Prepare small, falsifiable alternatives rather than rerunning the same 100-step loop:

| Hypothesis | Controlled intervention proposal | Disconfirming evidence |
|---|---|---|
| Trained-name template dependence | source identifiers independently randomized per training record; train/eval namespace disjointness | performance remains insensitive to the supplied identifier |
| Weak explicit copy objective | emphasize unchanged file copies and source-span preservation, **with stable loss accounting** | copies remain incorrect under clean held-out names |
| Wrong request/value binding | paired same-source prompts with contrasted requested literals | output values do not follow request changes |
| Full-file generation burden | compare full-file output to a separately declared source-span/patch formulation **without changing original scorer claims** | alternative still fails exact reconstruction |

Choose **one** primary intervention based on SB-00 and SB-01 evidence. A copy/span alternative would be an explicitly separated experiment, not retroactive evidence that the existing full-file model succeeded. Avoid vague promises that more data, parameters, sampling temperature or smaller loss will solve binding.

**Exit gate:** predeclared hypothesis, fixed measurable pass/fail signal, single-variable comparison, independent owner decision.

### SB-03 — V1 real tokenizer gate failed; V2 feasibility under review

The [SHA-frozen matched 288/96 proposal](../training/phase3/copy-fidelity-v1/sb03-namespace-diversity-proposal-v1.json) and [deterministic builder](../training/phase3/copy-fidelity-v1/prepare_namespace_diversity_candidate.py) are prepared for technical review. Frozen candidate SHA-256 is `4bbdc55782d9447271323378367831656dc1f05cb5bb174138738a6a7e5e7dea`; full-coverage sampler SHA-256 is `9ddf29a3a2972cee0c8c2f79db97d7ef3f823065a2db6ee0840e89bb01760c13`. The proposed 48 new train names preserve source lengths and exact task structure; all 96 validation records are unchanged. The owner ran that preflight on the real frozen tokenizer and it **FAILED on 240/384 paired records** (40,439 versus 44,925 draw-weighted supervised positions). V1 remains preserved and **not training-ready**. [Read-only V2 tokenizer-aware feasibility procedure](PHASE-3-SB03-V2-TOKENIZER-FEASIBILITY.md) now tests 48 new names against every assigned prompt and output signature; the owner's read-only run **passed 48/48 names and 384/384 exact tokenizer signatures**; an actual local V2 proposal was materialized and SHA-verified at `5f957cc30b1c58e7e22a8e2565efbc143303431526a61ecc2b65ef0e741ebb21`. A [streaming SHA-pinned training ancestor collision screen](PHASE-3-SB03-V2-TRAINING-ANCESTRY-SCREEN.md) is the next blocked gate. Global training-history exclusion remains incomplete, and no training has occurred. **The candidate is therefore NOT training-ready**; no stage, bundle, optimizer or newly generated model response is authorized. [Detailed blockers and owner-local read-only commands](PHASE-3-SB03-APPROVED-NAMESPACE-DIVERSITY.md).

### SB-04 — One separately approved bounded intervention — not yet authorized

Require a new, explicit owner approval with approved optimizer steps **and** wall-clock cap, seed, device, learning rate, batch/sampler, unchanged checkpoint source, 200 GB artifact limit, no resume, no checkpoint reselection, exact output hashes and stop rules. Fail closed if hashes or authorizations are absent. Do not reuse the October 17, 2026 authorization.

### SB-05 — Prespecified complete-file evaluation and decision — not yet authorized

Assess new microprobes and established 96 validation / 42 developer controls without contaminating future independent evaluation. Full generated files, exact identifiers, requested values, unrelated-source preservation, per-language gates and contrast groups are the **primary** capability metrics; teacher-forced loss is secondary. Keep the final project holdout sealed until a separate named milestone authorizes it. If any core gate fails, report failure and return to hypothesis review. Do not silently continue to P3-03.

## 4. Hard stop / non-goals

- **No new training is authorized by this roadmap.** No model loading or generated completions in SB-00.
- No checkpoint reselection (including step 75), model-size/tokenizer/context changes, silent data augmentation or untracked optimizer state.
- No previously inspected development set treated as independent final holdout.
- No auto-generated benchmark that leaks names or targets from inspected results into training.
- No claim of source preservation from mere identifier substring occurrence; no claim of correct request selection when the replacement boundary is unextractable.
- No promotion to P3-03 Insert/Delete or Phase 4 until useful exact-file capability is measured.

For a complete **Codex-ready issue-by-issue greenlight matrix**, acceptance tests, refusal gates, and copy/paste reviewer instructions, see [P3-SB Codex Review Roadmap](PHASE-3-SOURCE-BINDING-CODEX-REVIEW-ROADMAP.md). Only [issue #189](https://github.com/keepithandy/plex-nano/issues/189) is currently open; SB-00 through SB-05 are its milestones, not separate authorized training tasks.

## 5. Owner review checklist

Before proceeding past SB-01 preparation, review whether:

1. These **six stages** preserve Plex Nano's intended role as a small, from-scratch coding model.
2. The exact 48-case microprobe is narrow enough to isolate source-name copy versus requested-value selection.
3. Proposed 12/12, 30/36, 10/12 per-language and 9/12 contrast gates are sufficiently strict.
4. A single intervention, explicit authorization and untouched final holdout remain mandatory.
5. The failure record from issue #181 is permanently visible, rather than revised to imply training success.

**Technical disposition:** SB-00 is locally measured and SB-01 preparation verified. SB-02–04 [design specifications](PHASE-3-SOURCE-BINDING-TECHNICAL-REVIEW.md) and SB-05 pure scoring/synthetic tests are prepared for review; later experimental execution remains conditional on fresh explicit owner approval. Closing #181 is a record-keeping action after CI, not a declaration of capability success.
