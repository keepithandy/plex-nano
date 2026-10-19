# Plex Nano · Audit & recovery roadmap

> **Current focus: R2 — establish the model's training-source history.**<br>
> R1 is complete: the audit repairs are published and both remote checks passed.<br>
> The model's recorded copy/edit capability remains **0/138 exact files**.

**Repository:** [keepithandy/plex-nano](https://github.com/keepithandy/plex-nano)<br>
**Prepared:** October 18, 2026 · **Reading mode:** review and discussion<br>
**Product:** a small coding model that edits **user-supplied HTML, CSS and JavaScript**<br>
**Controlled baseline:** 27,566,080 parameters · 512-token context · 16,384-token vocabulary

[At a glance](#at-a-glance) · [Milestones R1–R6](#milestones) · [Discussion notes](#discussion) · [Fixed boundaries](#boundaries) · [Evidence drawer](#evidence)

---

<a id="at-a-glance"></a>

## 🧭 At a glance

### Status key

| Marker | Meaning |
|---|---|
| ✅ **Complete** | The named work has supporting evidence. |
| 🟡 **Next** | The immediate work to discuss and carry out. |
| 🧱 **Blocked** | A required prerequisite is unresolved. |
| 🔒 **Owner decision** | Model execution needs its own explicit authorization. |
| ⚪ **Later** | Depends on earlier milestones. |
| 🔴 **Failed result** | A recorded capability gate failed; retain the result. |

### What is settled, and what is open?

| Area | Current position | What it means for us |
|---|---|---|
| Audit repairs and merge | ✅ **Published; Linux and Windows passed** at `caf3284` | R1's delivery gate is met. |
| Actual generated code | 🔴 **0/138 exact files** | Reliable copying/editing has not been demonstrated. |
| V1 dataset proposal | 🔴 **240/384 token-count mismatches** | Preserve V1 as the failed preparation record. |
| V2 tokenizer feasibility | ✅ **48/48 names; 384/384 signatures matched** | Token lengths can be matched against the five screened P3 corpora. |
| Saved V2 proposal | ✅ **Owner-reported file and SHA verified** in [PR #195](https://github.com/keepithandy/plex-nano/pull/195) | It remains an unapproved proposal requiring independent review. |
| Historical-source screening | 🟡 **Tool available; complete source inventory and screening pending** | This is the immediate research blocker. |
| New training or model evaluation | 🔒 **Not authorized** | Preparation and passing tests do not grant execution permission. |

### The route through the work

| Stage | Status | Plain-language objective | Completion evidence |
|---|---|---|---|
| [R1](#r1) | ✅ Complete | Deliver the reporting repairs. | Published commit and successful remote checks. |
| [R2](#r2) | 🟡 Next | Establish which source texts the model learned from. | Reviewed inventory, file hashes and collision results. |
| [R3](#r3) | 🧱 Needs R2 | Freeze a fair comparison and scoring rules. | Reviewed V2 dataset, matched conditions and fixed criteria. |
| [R4](#r4) | 🔒 Owner decision | Prepare precise execution packets. | Separate decisions for staging, each training arm and evaluation. |
| [R5](#r5) | ⚪ Later | Measure generated-file recovery. | Raw outputs, independent scores and a comparison of both arms. |
| [R6](#r6) | ⚪ Later | Confirm transfer before harder edits. | A fresh replacement test supports the next milestone. |

**How to dissect this:** use the R-number for a milestone and the Q-number for its main discussion question. Each milestone separates completed work, remaining tasks and its exit gate. The detailed audit tables are preserved in the evidence drawer.

### What is lined up after R2?

| Where to look | What we would do | What we want to learn or establish |
|---|---|---|
| [**R3 / Q3 — Design a fair comparison**](#r3) | Compare the original training-name distribution with a wider range of names, keeping the starting model and other declared conditions matched. Fix the scoring rules beforehand. | Does more varied training help Plex preserve the names in supplied code? |
| [**R4 / Q4 — Set the run limits and decisions**](#r4) | Specify the exact data, starting model, training budget, stopping rules and separate execution permissions. | Is each proposed action concrete and bounded enough to review? |
| [**R5 / Q5 — Run and measure the comparison**](#r5) | Carry out the authorized runs and test generated files for exact copying, requested edits and preservation of unrelated code. | Did coding behavior improve enough to meet the declared gates? |
| [**R6 / Q6 — Confirm on fresh supplied code**](#r6) | Test replacement edits on a new independent set, then review whether to attempt harder edits such as adding or removing requested code. | Does the improvement transfer beyond the small controlled probe? |

**Next discussion topic: R3 / Q3.** Its central idea is to test whether a wider range of training names helps source-name preservation. R2's actual inventory and screening remain the prerequisite for finalizing that experiment.

---

<a id="milestones"></a>

## 🛠 Milestones

<a id="r1"></a>

### ✅ R1 · Publish and verify the audit repairs

**Status:** complete for the implementation through `caf32842c3912bfc75b983d5e7c6bd808d8b26c1`.

**Purpose:** make the existing reports reject malformed evidence and make the relevant checks portable.

- [x] Review and fix the scorer/gate defects and Windows path assertions.
- [x] Preserve the repairs in `fe1367f` and combine them with PR #195 in merge `caf3284`.
- [x] Publish the merge. The actual delivery path used a local merge and push; no separate audit-repair PR was created.
- [x] Verify successful remote [Linux checks](https://github.com/keepithandy/plex-nano/actions/runs/39179180285) and [Windows checks](https://github.com/keepithandy/plex-nano/actions/runs/39179180184) on that exact merge SHA.
- [x] Preserve frozen data, predictions, checkpoint identities and historical thresholds.

**Validation record:** the merged tree previously passed **672** tests from the required workflow list on Windows and **314** prescribed Windows checks. Earlier **92 / 661 / 303** repair checks remain recorded in the historical audit below. This formatting pass did not rerun model or test computations.

> **Exit gate — met:** the repairs are committed, published and remotely verified. These successful runs apply to `caf3284`; later code changes need checks for their own revision.

**Q1 · Discussion:** are any reporting concerns still missing from the completed repair scope?

---

<a id="r2"></a>

### 🟡 R2 · Establish historical training-source coverage

**Status:** next. The checker exists; a complete source inventory and its real screening results are still missing.

**Why it matters:** a name can be absent from the five P3 datasets while appearing in older training text. We need evidence before describing the proposed names as unseen across the model's ancestry.

**Already in place**

- [x] [PR #195](https://github.com/keepithandy/plex-nano/pull/195) supplies a [streaming SHA and collision auditor](PHASE-3-SB03-V2-TRAINING-ANCESTRY-SCREEN.md).
- [x] The auditor checks explicit source files in bounded chunks, including name matches across chunk boundaries. Large files no longer require the earlier reader's 128 MiB aggregate in-memory approach.
- [x] Its result remains provisional: it cannot certify that a supplied inventory includes every historical training source.

**Work left**

- [ ] Trace the fixed checkpoint's actual ancestor training corpora, including older Phase 2 and private sources. Identify exact training-text paths and hashes from provenance records; do not infer ancestry from nearby folders.
- [ ] Review the inventory for completeness and identify unavailable or ambiguous sources. Keep the final project holdout outside it.
- [ ] Run the checker against the explicit inventory and record proposed train-name collisions, per-file hashes and coverage limits. The five P3 checks remain a partial exclusion claim.
- [ ] Review the frozen evaluation namespaces' exclusion claims against the relevant training history as well; do not assume a train-name collision check covers them automatically.
- [ ] If names collide, preserve the existing proposal and create a separately reviewed revision, or explicitly narrow the research claim. Do not relabel partly screened names globally unseen.

> **Exit gate:** a reproducible SHA inventory, a reviewed coverage assessment and a collision receipt support the intended unseen-name claim. An inventory file or a zero-collision result alone does not establish completeness.

**Q2 · Discussion:** can we trace every relevant training source, and what claim is defensible if some sources cannot be recovered?

---

<a id="r3"></a>

### 🧱 R3 · Freeze a defensible V2 comparison and scoring contract

**Status:** tokenizer feasibility is demonstrated; history clearance, independent review and the final comparison contract are pending.

**Why it matters:** we want to test whether a wider range of training identifiers improves supplied-code editing. A fair comparison must control the other conditions.

- [ ] Repeat V2 selection with the reviewed history inventory and freeze a distinct reviewed V2 manifest/SHA. Keep V1 and the existing unreviewed V2 proposal intact.
- [ ] Independently verify all **384** prompt/answer token signatures, the unchanged **96** validation rows, **72/24** intact train/validation groups, source-to-target relations, train/evaluation separation and the full-coverage draw indices.
- [ ] Match both arms' token exposure and answer/EOS masks. Start both from the **same fixed checkpoint**, with the same optimizer initialization, order, seed, budget and endpoint rule.
- [ ] Distinguish the fresh comparator from the completed historical run: its starting state and sampler are not interchangeable with that historical control.
- [ ] Version SB-05 scoring/provenance for the actual arm identities. The present scorer accepts only historical checkpoint `c968…0230`; it cannot certify an arbitrary future arm.
- [ ] Fix the comparison's effect measure and a result that would refute the identifier-diversity hypothesis **before** observing either arm's outputs. A small single-seed result supports a bounded claim.
- [ ] Preserve exact unrelated-byte checks, wrong/unknown requested-value distinctions, EOS/decode/context checks and raw per-case provenance.

**Absolute SB-01 gates — unchanged**

| Metric | Required outcome |
|---|---:|
| Exact unchanged copies | **12/12** |
| Exact requested edits | **≥30/36** |
| Exact edits in each language | **≥10/12** |
| Fully correct four-case contrast groups | **≥9/12** |

> **Exit gate:** a reviewed V2 manifest, full-history scope, matched-arm design, scorer version and immutable success/failure rules. The candidate remains unready while any required check is unresolved.

**Q3 · Discussion:** what measured difference would support the diversity hypothesis, and what result would make us reject it?

---

<a id="r4"></a>

### 🔒 R4 · Prepare fresh execution packets for an owner decision

**Status:** preparation can be reviewed; new model execution is not authorized.

**Why it matters:** approval must refer to concrete inputs, a bounded operation and a fixed stopping rule.

- [ ] Review new packed bundles and zero-update stages separately from optimizer permission. Pin their bytes, starting-weight equality, empty optimizer, target accounting, fresh destinations and storage budget.
- [ ] Freeze explicit per-arm contracts: maximum updates/wall time, learning rate, accumulation, seed/device, validation/checkpoint schedule, finite-loss/gradient aborts, no overwrite/resume and fixed endpoint.
- [ ] Obtain independent fresh owner authorization for each comparator/intervention training arm and for new checkpoint generation on the **48-case** probe.
- [ ] Ensure closed historical approvals cannot be reused for either new arm. Preserve the original one-run evidence even if output folders move.

> **Exit gate:** concrete reviewed execution packets and explicit owner decisions. A roadmap, successful preflight or passing CI grants no execution permission.

**Q4 · Discussion:** are the inputs, limits, stop conditions and requested permissions precise enough to review each action separately?

---

<a id="r5"></a>

### ⚪ R5 · Measure actual generated-file recovery

**Status:** conditional on R1–R4.

**Why it matters:** the product requires correct generated code changes. Lower teacher-forced loss has not established that capability.

- [ ] Execute only individually authorized arms; keep immutable run/checkpoint receipts and incomplete or failed outcomes.
- [ ] Evaluate only predeclared endpoints and response sets with the approved decoding policy. Label historical **96/42** controls as previously inspected.
- [ ] Report each arm's exact-copy/edit and group results by language/context, source retention, wrong/unknown values and EOS/decode/prompt status. Compare the predeclared effect and absolute gates.
- [ ] Record improvement below a gate as limited movement; keep capability recovery open. If both arms pass similarly, do not attribute the result to identifier diversity.
- [ ] If copying fails, review an independent objective/copying hypothesis. If copying passes but edits fail, review request/value binding. Do not extend training automatically.

> **Exit gate:** raw generated responses and an independent scorer replay support the disposition. Loss alone cannot close [issue #189](https://github.com/keepithandy/plex-nano/issues/189).

**Q5 · Discussion:** how will we distinguish recovery, limited improvement, an inconclusive comparison and failure?

---

<a id="r6"></a>

### ⚪ R6 · Confirm replacement transfer before harder edits

**Status:** future; depends on a successful R5 disposition.

**Why it matters:** passing a small controlled probe does not establish reliable edits on fresh supplied code.

- [ ] Freeze a fresh independent replacement confirmation set and criteria before model exposure, without opening the final project holdout.
- [ ] Confirm minimal correct changes on supplied HTML/CSS/JavaScript with unfamiliar identifiers and unrelated context.
- [ ] Reassess remaining weaknesses and decide whether P3-03 Insert/Delete is justified. Preserve Phase 2's closeout and its documented limitations.

> **Exit gate:** owner-reviewed evidence supports the next coding milestone. Model size, tokenizer and context remain controlled unless a distinct milestone changes them.

**Q6 · Discussion:** what independent replacement evidence is sufficient before moving to harder edit types?

---

<a id="discussion"></a>

## 💬 Discussion notes

Use this space to capture changes as we dissect the plan. These are discussion prompts, not experimental approvals.

| Reference | Topic | Notes / proposed change |
|---|---|---|
| Q1 / R1 | Any remaining reporting concerns? | — |
| **Q2 / R2** | **Training ancestry, missing sources and the unseen-name claim** | Term explained; actual inventory and screening remain pending. |
| **Q3 / R3** | **Fair comparison and falsifiable outcome** | **Next discussion: does a wider range of training names improve source-name preservation?** |
| Q4 / R4 | Exact scope and separate execution decisions | — |
| Q5 / R5 | Recovery, limited movement, inconclusive result or failure | — |
| Q6 / R6 | Evidence needed before harder edits | — |

<a id="boundaries"></a>

## 📌 Fixed boundaries

| Keep fixed | Meaning |
|---|---|
| Product contract | Coding request + user-supplied code/file context → correct code change. |
| Controlled model | 27,566,080 parameters; 512-token context; 16,384-token vocabulary. |
| Historical evidence | Keep failed results, original numeric records, datasets, raw outputs, hashes and checkpoints intact. |
| Evaluation interpretation | A literal mention is not a correct bound edit; an unextractable target is not an independently observed wrong value. |
| Research permissions | Preparation, staging, training and generation remain separate decisions. |
| Final project holdout | Keep sealed; do not use it to screen names or build the training-source inventory. |
| Scope of this edit | Document review and presentation only. No experiment was started. |

---

<a id="evidence"></a>

## 📚 Evidence drawer

The original audit's facts, hashes and all 17 merge records are retained below. Expand only the material relevant to the question being discussed.

<details>
<summary><strong>A · Completed repairs and original validation record</strong></summary>

- [x] Reconciled all **17** merged PRs against GitHub merge timestamps, descriptions, actual merge/head SHAs, changed sources, reports and test coverage. PRs #191–193 belong to October 17 locally despite October 18 UTC timestamps.
- [x] Verified **32 successful Linux/Windows PR-head runs** across the 16 code/workflow PRs, plus the successful TypeScript run for #179. #175 changed only a gitlink and triggered no workflow. Inspected job steps; checked the numerical CI claims in #186–193 directly against their job logs. PR discussion/review timelines returned no comments.
- [x] Included the day's five additional integration/diagnostic commits: `db79a79`, `497770f`, `cae6ea4`, `c33ab6a`, `f3651a6`. Replayed their frozen 72-case result. Earlier PRs incorporated through `497770f` retain their actual earlier merge dates.
- [x] Replayed the original 36-case diagnosis, original 42-case saved responses, frozen 72-case scorer summary, and full 138-case morphology evidence without generating model responses.
- [x] Verified the actual completed run report and hashed its fixed step-100 checkpoint bytes. No research checkpoint was deserialized or changed.
- [x] Reproduced V1's token-count mismatch with the real SHA-verified tokenizer; performed the existing V2 tool's default read-only feasibility search.
- [x] Applied narrow local fixes below. Historical datasets, predictions, evidence JSON, approval packets and checkpoint hashes remain unchanged.

| Priority | Reproduced defect | Local repair | Verification |
|---|---|---|---|
| P2 | The 138-case scorer accepted impossible `stoppedAtEos=true` with 192 generated text tokens, missing/invalid token counts, non-boolean flags and invalid Unicode. It could certify an impossible completion. | [Step-100 scorer](../training/phase3/copy-fidelity-v1/evaluate_step100.py) now rejects malformed completion evidence before scoring. A genuine capped, non-EOS response remains valid failure evidence. | Negative synthetic cases plus perfect-output, incomplete-output and capped-output checks. |
| P2 | The older 42-case scorer also accepted EOS at the text-token cap and invalid Unicode. | [42-case scorer](../training/phase3/diagnostics/p3-copy-edit-controls-v1/score_frozen_responses.py) now rejects both. | Added refusal tests; original saved 42-response result remains 0/42. |
| P2 | The endpoint-report gate accepted NaN losses, unchecked intermediate losses and malformed timeline entries; malformed inputs could raise incidental exceptions. | Validate every loss as a finite, nonnegative float; validate timeline object/step/hash/path types before arithmetic. Preserve the original endpoint loss tolerances and all SHA pins. | All five validation positions tested with NaN, infinity, negative, boolean and string inputs; malformed checkpoint/validation entries rejected. Genuine saved report still passes. |
| P3 | Two older preflight tests compared Windows-native paths with literal slash strings. The broader prescribed suite failed locally despite valid paths. | Compare `Path` values in [request-grounded tests](../training/tests/test_request_grounded_coding_training.py) and [evidence-composition tests](../training/tests/test_structured_plan_evidence_composition_training.py). | Broader required suite passes on this Windows machine. No production path or historical report changed. |

The generator checks EOS before appending a token to the returned text-token list. Therefore a 192-step generation ending at EOS can contain at most 191 text tokens. These repairs validate the existing generation contract; they do not change capability thresholds.

**Validation of the local repairs:** **92 focused tests passed; 661 tests passed from the current required Linux-workflow file list run on Windows; 303 prescribed Windows-portability tests passed.** Added 46 regression cases; 39 failed before the repairs and all passed afterward. Initial sandbox runs encountered temporary-directory access failures; the reported completed suites used normal temporary-folder access. Existing NumPy/CUDA initialization warnings were nonfatal. Tests use synthetic fixtures and small test models, not new Plex research training.

> **Historical delivery state:** these repairs were uncommitted at the initial audit. They were subsequently committed as `fe1367f`, included in merge `caf3284`, pushed, and verified by the Linux and Windows runs linked in R1. The 661/303 counts above belong to the original repair snapshot; the merged tree later passed 672/314 local checks.

</details>

<details>
<summary><strong>B · Recorded model results, V1/V2 findings and immutable receipts</strong></summary>

> **Historical snapshot:** the following measurements describe the original audit. Its statement that no V2 file was saved applies to that audit run; PR #195 later recorded the owner-saved proposal.

| Evaluation / observation | Verified outcome | Interpretation |
|---|---|---|
| Original P3-02 development | 0/36 exact edits; 8/12 contrast groups emitted identical outputs for different requests. | Reliable supplied-file editing was not demonstrated. |
| Frozen 72-case source-preservation diagnostic | 0/72 exact files and source-name passes; original summary and both-fail decision replay exactly. | 68/72 targets were unextractable. Do not turn 0 value passes into an independent claim that all 72 values were selected incorrectly. |
| Original 42-case controls | 0/6 copies, 0/18 minimal edits, 0/18 context edits; all 36 edit values unscorable. | Source copying fails before harder editing. |
| Completed copy-fidelity run | 100 updates; 40,477 answer/EOS positions; 102.9442637 seconds; loss 7.116717401614859 → 2.7984008651134396. | Lower teacher-forced loss did not establish generated-file capability. Step 75 is not eligible for retrospective selection. |
| Fixed endpoint's 138 responses | 0/96 validation and 0/42 historical development exact files; 0/138 supplied-namespace presence; training-name text in 138/138. EOS/decode/full prompt all 138/138. | Observed source-conditioning failure; training-name occurrence is a heuristic, not proof of memorization or a neural mechanism. |
| Requested values in that report | Literal mentioned in 17/108 edits; original target boundaries extractable in 0/108. | Mentions do not establish independent requested-value correctness. |
| SB-01 frozen 48-case microprobe | 12 copies, 36 edits, 12 groups; real tokenizer maximum 95 prompt tokens, 50 answer/EOS tokens; 1,296 answer/EOS targets. | Fixture/tokenizer preparation verified; real model has not answered this probe. |
| SB-03 V1 proposal | 240/384 token-count mismatches; 40,439 vs 44,925 draw-weighted targets (**+4,486; 11.09%**). | V1 is unsuitable for the planned token-matched comparison. Preserve its frozen bytes. |
| V2 feasibility, checked October 18 | **48/48 names selected; 384/384 paired signatures matched**, including unchanged 96-row validation; 34,237 candidates considered at the default 2,500-per-slot bound. | Feasible against the five screened P3 corpora only. No extra history files screened; training-ready remains false. |

The V2 check constructed proposal bytes in memory only. Their SHA-256 was `5f957cc30b1c58e7e22a8e2565efbc143303431526a61ecc2b65ef0e741ebb21` (330,673 bytes). **No V2 dataset file, bundle or stage was saved.** Historical screening can reject selected names and change the eventual candidate; this checksum is a feasibility receipt, not approval of a frozen training dataset.

**Important sampler distinction:** the completed run used the original epoch-shuffle SHA `6d214246…e9f4`, which covers 288/288 rows. The 287/288 coverage defect belonged to a later proposed `randrange` sampler, not the completed training run. SB-03's new SHA-permuted sampler covers 288/288 rows, 5–6 times each. Its 40,439 comparator targets differ from the historical run's 40,477 because its draw sequence differs.

Full immutable receipts verified during this audit:

| Artifact | SHA-256 |
|---|---|
| Original 72-response envelope | `07c89354fafec181d337cd3b35bc8b471d865ef25bbc29c329e83db2583323a5` |
| Original 42-response envelope, owner-local | `b817a747ce52da46008daa48bd66f23f5043ab0af64154e750dbac4a105edcc8` |
| Completed copy-fidelity run report, owner-local | `15fcc1829f4575425833b7e2f95b39e30871b2b02c03afd0efa86e62c8d6a796` |
| Fixed copy-fidelity step-100 checkpoint | `c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230` |
| 138-response evaluation report, owner-local | `971a78971387200df9523c6ca177006f60f2834be0664c2b590e6ec72f06a197` |

The complete 138-case replay equals the committed [technical-review evidence](../training/pretraining/p3-source-binding-technical-review-evidence.json). Private raw reports remain local. Historical source-code hashes describe the tools at their original review revision; the repairs do not rewrite those receipts.

</details>

<details>
<summary><strong>C · Every October 17 merge and its original CI evidence</strong></summary>

All linked CI runs below were successful at the audited original PR heads. **Merge / head** records both identities so a passed branch revision is not confused with a squash-merge commit.

| PR | Merge / head | Audit disposition | Original PR-head CI |
|---|---|---|---|
| [#174](https://github.com/keepithandy/plex-nano/pull/174) | `72bad6b` / `8f6ef03` | LF identity and commented-JavaScript regressions hold; 72-case aggregate replay matches. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39049713370) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39049713296) |
| [#175](https://github.com/keepithandy/plex-nano/pull/175) | `54b4b89` / `12550bc` | Exactly one orphaned gitlink removed; no ordinary file deleted. Path-only change triggered no PR CI. | Not triggered |
| [#176](https://github.com/keepithandy/plex-nano/pull/176) | `72df912` / `e3498c1` | 42-case construction, splits, source identities and unknown-value semantics verified. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39050625145) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39050625042) |
| [#178](https://github.com/keepithandy/plex-nano/pull/178) | `7f3f32b` / `5d925b3` | Eight gitlinks retired; all eight full commit IDs retained; current Git tree has no gitlinks. | [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39050972791) / [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39050972782) |
| [#179](https://github.com/keepithandy/plex-nano/pull/179) | `9f20d7b` / `401b474` | All four workflows updated; Python/Node versions and test commands preserved. Actual v7 checkout/setup jobs succeeded. | [TypeScript](https://github.com/keepithandy/plex-nano/actions/runs/39051152866) / [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39051152912) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39051152969) |
| [#180](https://github.com/keepithandy/plex-nano/pull/180) | `7602264` / `56cf893` | Original 42 saved responses replay to 0/42; found and repaired impossible EOS-at-cap acceptance. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39053367153) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39053367017) |
| [#182](https://github.com/keepithandy/plex-nano/pull/182) | `7f8f154` / `c003e9f` | 384-row deterministic candidate, 288/96 split, intact copy/edit groups and declared P3 exclusion verified. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39056163604) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39056163774) |
| [#183](https://github.com/keepithandy/plex-nano/pull/183) | `735c5c8` / `0454ae6` | Bundle/mask reconstruction and zero-stage refusal tests pass; original sampler covers all 288 records. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39059411737) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39059411738) |
| [#184](https://github.com/keepithandy/plex-nano/pull/184) | `43cfc94` / `649fd0d` | Separate tokenizer-origin and dataset-origin SHA checks are correct; no hash substitution needed. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39062545501) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39062545537) |
| [#185](https://github.com/keepithandy/plex-nano/pull/185) | `ae5723b` / `8009ba0` | Read-only baseline and bounded-run orchestration tested; saved run later completed. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39064386562) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39064386570) |
| [#186](https://github.com/keepithandy/plex-nano/pull/186) | `f559379` / `5430437` | Frozen baseline bytes/SHA and weighted language loss reconcile to 7.116717401614859. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39072462100) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39072461980) |
| [#187](https://github.com/keepithandy/plex-nano/pull/187) | `ceca82c` / `2431043` | Approved packet and baseline SHA match; saved run records 100 updates, 40,477 targets and 102.9443 seconds. Approval is spent. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39073015235) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39073015234) |
| [#188](https://github.com/keepithandy/plex-nano/pull/188) | `933202b` / `8f763d7` | Saved endpoint SHA and all 138 outputs verified; repaired malformed-completion and nonfinite-history acceptance. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39079181848) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39079181830) |
| [#190](https://github.com/keepithandy/plex-nano/pull/190) | `be5528f` / `075c40d` | Failed closeout is supported: 0/96 validation and 0/42 development. Morphology is descriptive, not a causal proof. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39080682093) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39080682096) |
| [#191](https://github.com/keepithandy/plex-nano/pull/191) | `18ec7a6` / `0a4d933` | 48-row/12-copy/36-edit microprobe reproduces; names excluded only from declared P3 sources. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39081684908) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39081684898) |
| [#192](https://github.com/keepithandy/plex-nano/pull/192) | `2099a56` / `311fdc8` | SB-00 replay exactly matches frozen evidence; actual tokenizer audit matches SB-01 receipt; published CI counts confirmed from logs. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39083169890) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39083170870) |
| [#193](https://github.com/keepithandy/plex-nano/pull/193) | `28a95bf` / `4c6e9b1` | V1 bytes and sampler reproduce, but real tokenizer matching fails on 240/384 rows. V1 remains blocked. | [Linux](https://github.com/keepithandy/plex-nano/actions/runs/39086948988) / [Windows](https://github.com/keepithandy/plex-nano/actions/runs/39086948978) |

[PR #194](https://github.com/keepithandy/plex-nano/pull/194) merged on **October 18** and is outside yesterday's 17-PR count. Its V2 successor tool was inspected and exercised read-only because it resolves #193's immediate tokenizer-feasibility question. This audit does not claim model improvement, fresh optimizer updates, new real-model responses or final-holdout access.

</details>

<details>
<summary><strong>D · Scope and revision history</strong></summary>

| Record | Scope / revision |
|---|---|
| Original audit day | October 17, 2026, America/New_York: 00:00–24:00 EDT; October 17 04:00 through October 18 04:00 UTC, end exclusive. |
| Original audited checkout | `5a93b65fd6abb39b10f81202c1eebf73d17d6361` |
| October 17 final merge | `28a95bf65887914a296d8bc9ed9986fab0baf88f` |
| Audit repairs | Local commit `fe1367f`, included in published merge `caf32842c3912bfc75b983d5e7c6bd808d8b26c1`. |
| October 18 successor | [PR #195](https://github.com/keepithandy/plex-nano/pull/195): streaming history checker and owner-reported saved V2 proposal. |
| Latest status verification in this review | Linux and Windows push workflows both completed successfully for `caf32842c3912bfc75b983d5e7c6bd808d8b26c1`; see R1. |

The initial audit constructed V2 proposal bytes in memory. The owner subsequently reported saving a proposal with the same SHA, as recorded in PR #195. Neither event establishes complete training ancestry, clears collisions, or authorizes model execution.

</details>

<details>
<summary><strong>E · Plain-language glossary</strong></summary>

| Term | Meaning in this roadmap |
|---|---|
| Comparator | The experiment arm using the original training-name distribution. |
| Intervention | The arm testing a wider range of training names while holding the other declared conditions fixed. |
| Training ancestry | The actual earlier source texts used to train the checkpoint's predecessors. |
| Token-count parity | Paired prompts and answers have matching token counts under the same tokenizer. |
| Checkpoint | A saved model state; its exact identity is bound by a SHA-256 hash. |
| EOS | End-of-sequence token, used to indicate that generation finished. |
| Contrast group | Several requests against the same supplied source, scored together to test request-dependent changes. |

</details>
