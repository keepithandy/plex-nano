# Plex Nano merge-audit recovery roadmap

**Merge update (October 18):** audit repairs are retained in local commit `fe1367f` while integrating [PR #195](https://github.com/keepithandy/plex-nano/pull/195), which adds the [streaming training-ancestry checker](PHASE-3-SB03-V2-TRAINING-ANCESTRY-SCREEN.md). The incoming report records a saved owner-local V2 proposal with the same SHA as the earlier in-memory receipt below. Complete ancestor-source screening, independent V2 review, fresh remote CI for the audit repairs and new model-execution approvals remain pending. The merged tree passed 672 tests from the required workflow list on Windows and 314 prescribed Windows checks. The original audit snapshot below is preserved.

**Prepared:** October 18, 2026. **Audit day:** October 17, 2026, America/New_York (00:00–24:00 EDT; October 17 04:00 through October 18 04:00 UTC, end exclusive).  
**Repository:** [keepithandy/plex-nano](https://github.com/keepithandy/plex-nano). **Audited checkout:** `5a93b65fd6abb39b10f81202c1eebf73d17d6361`. Yesterday's final merge was `28a95bf65887914a296d8bc9ed9986fab0baf88f`.  
**Purpose:** improve the 27,566,080-parameter coding model's ability to preserve and edit user-supplied HTML, CSS and JavaScript. Keep the 512-token context and 16,384-token tokenizer fixed.

**Decision:** the reports support a failed copy/edit capability result. Repair reporting reliability, complete historical-source screening, and review a newly frozen V2 comparison before any further model experiment. This roadmap is ready for owner review; subsequent milestones have not been started as training or live evaluation.

## 1. Completed audit and repairs

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

**Delivery state:** repairs are local working-tree changes, not a published PR or merged change. Historical remote CI proves the original PR heads, not these new edits. Fresh remote Linux/Windows validation is the first remaining gate.

## 2. Evidence boundary to carry forward

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

## 3. Next work, in order

### R1 — Publish and verify the audit repairs

**Status: local implementation complete; publication and remote checks pending.**

- [ ] Review the narrow scorer/gate changes and portable assertions as one scoped patch.
- [ ] Create a PR with the actual reproduced defects, 661/303 local results and immutable-result replay evidence.
- [ ] Require successful Linux and Windows checks on the final PR head, then verify the merged SHA's checks if merged.
- [ ] Confirm that no frozen JSON, checkpoint, historical threshold or saved prediction changed.

**Exit:** reviewed repair commit with fresh passing remote CI. Do not reuse yesterday's successful CI as evidence for this patch.

### R2 — Establish historical training-source coverage

**Status: blocking research prerequisite.**

- [ ] Enumerate the actual ancestor training corpora for the fixed starting checkpoint, including older Phase 2 and private sources. Identify explicit training-source paths and exact hashes; do not infer ancestry from nearby folders.
- [ ] Screen proposed train names and frozen evaluation namespaces against that inventory, recording collisions and any missing/unavailable sources. The five P3 corpus checks are only a partial exclusion claim.
- [ ] If the inventory exceeds the V2 reader's 128 MiB aggregate limit, prepare a separately reviewed streaming audit with bounded reads and boundary-safe substring matching. Retain explicit source selection.
- [ ] Keep the final project holdout outside the inventory and closed.
- [ ] Resolve any collisions through a new proposal/version or narrow the research claim explicitly; never relabel partly screened names globally unseen.

**Exit:** reproducible SHA inventory, coverage assessment and collision receipt sufficient to review the intended unseen-name claim. An inventory file alone does not prove completeness.

### R3 — Freeze a defensible V2 comparison and scoring contract

**Status: tokenizer feasibility demonstrated; final reviewed dataset and contract pending.**

- [ ] Repeat V2 selection with the reviewed history inventory and freeze a distinct reviewed V2 manifest/SHA. Keep V1 unchanged.
- [ ] Independently verify all 384 prompt/answer token signatures, unchanged 96 validation rows, 72/24 intact train/validation groups, source-to-target relations, train/evaluation separation and exact full-coverage draw indices.
- [ ] Freeze both arms' equal token exposure and answer/EOS masks. Both must start from the **same fixed checkpoint**, with the same optimizer initialization, order, seed, budget and endpoint rule.
- [ ] Distinguish the fresh comparator from the already completed historical run; neither its starting state nor its sampler is an interchangeable historical control.
- [ ] Version the pure SB-05 scoring/provenance contract for the actual arm identities. The present scorer deliberately accepts only the historical `c968…0230` checkpoint; it cannot certify an arbitrary future arm.
- [ ] Predeclare the comparison's effect measure and what outcome would refute the identifier-diversity hypothesis before seeing either arm's outputs. A small single-seed result supports only a bounded claim.
- [ ] Preserve SB-01's absolute gates: **12/12 exact copies, ≥30/36 exact edits, ≥10/12 edits per language, ≥9/12 fully correct groups**. Keep exact unrelated-byte preservation, requested-value wrong/unknown distinctions, EOS/decode/context and raw per-case provenance.

**Exit:** reviewed V2 manifest, full-history scope, matched-arm design, scorer version and immutable success/failure rules. No training-ready claim while any required check remains unresolved.

### R4 — Prepare fresh execution packets for an owner decision

**Status: not authorized as model execution.**

- [ ] Review any new packed bundle and zero-update stage separately from optimizer permission. Pin their bytes, starting-weight equality, empty optimizer, target accounting, fresh destinations and storage budget.
- [ ] Freeze explicit per-arm contracts: maximum updates/wall time, learning rate, accumulation, seed/device, validation/checkpoint schedule, finite-loss/gradient aborts, no overwrite/resume and fixed endpoint.
- [ ] Require independent fresh owner authorization for each comparator/intervention arm and for new checkpoint generation on the 48-case probe.
- [ ] Ensure consumed/closed historical approvals cannot be reused as authorization for either new arm. Preserve original one-run evidence even if local output folders are moved.

**Exit:** concrete reviewed execution packets and explicit owner decisions. This roadmap and a successful preflight grant no execution permission.

### R5 — Measure actual generated-file recovery

**Status: conditional on R1–R4.**

- [ ] Execute only the individually authorized arms; retain immutable run/checkpoint receipts and incomplete/failed outcomes.
- [ ] Evaluate only the predeclared endpoints and response sets using the approved fixed decoding policy. Keep historical 96/42 controls labeled as previously inspected.
- [ ] Report both arms' exact-copy/edit and group results by language/context, source retention, wrong/unknown values and EOS/decode/prompt status. Compare the predeclared effect and absolute gates.
- [ ] If the intervention improves a metric but still fails a gate, record limited movement and keep capability recovery open. If both arms pass similarly, do not attribute the result to identifier diversity.
- [ ] If copying fails, return to an independently reviewed objective/copying hypothesis. If copying passes but edits fail, review request/value binding. Do not extend training automatically.

**Exit:** raw generated responses and independent scorer replay support the disposition. Loss alone cannot close [issue #189](https://github.com/keepithandy/plex-nano/issues/189).

### R6 — Confirm replacement transfer before harder edits

**Status: future, dependent on a successful R5 disposition.**

- [ ] Freeze a fresh independent replacement confirmation set and criteria before model exposure, without opening the final project holdout.
- [ ] Confirm minimal correct changes on supplied HTML/CSS/JavaScript with unfamiliar identifiers and unrelated context.
- [ ] Reassess remaining weaknesses and decide whether P3-03 Insert/Delete is justified. Phase 2 remains closed with its documented limitations; do not rewrite that closeout as solved transfer.

**Exit:** owner-reviewed evidence supports the next coding milestone. Model size, tokenizer and context remain controlled unless a distinct milestone changes them.

## 4. Coverage of every October 17 merge

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

