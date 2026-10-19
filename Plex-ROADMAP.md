# Plex Nano — Development Roadmap

## Core goal

Build **Plex Nano**, a small coding model that takes a natural-language coding request plus code/file context supplied by the user and produces the correct code change.

```text
request + supplied code/file
          ↓
      Plex Nano
          ↓
   correct code change
```

Plex is not being developed as a general chatbot. The current research target is a compact model that becomes increasingly reliable at HTML, CSS, and JavaScript coding tasks while remaining small enough for practical local use.

## Current controlled model

| Item | Value |
|---|---:|
| Parameters | **27,566,080** |
| Context | **512 tokens** |
| Vocabulary | **16,384-token byte-level BPE** |
| Initialization | **Scratch / random weights** |
| Languages | **HTML / CSS / JavaScript** |
| Current phase | **Phase 3 — File-Conditioned Coding** |

The 27M model remains the controlled baseline until a roadmap milestone explicitly authorizes a model-size change.

---

# Phase 1 — Model Foundation

**Status: COMPLETE**

Phase 1 established that Plex can be trained from scratch and operated locally.

Completed capabilities include:

- deterministic model configuration
- scratch initialization
- tokenizer training
- reproducible dataset builds
- CPU/CUDA environment reporting
- bounded training runs
- checkpoint save/load
- checkpoint resume
- validation loss measurement
- deterministic generation
- training provenance and hashes
- local Windows workflow

Phase 1 proved the training system works. It did **not** prove useful coding ability by itself.

---

# Phase 2 — Coding Understanding

**Status: COMPLETE**

## Phase 2 purpose

Teach Plex to understand coding requests and map them to the intended change.

Phase 2 research covers:

- request following
- code syntax
- literal binding
- semantic roles
- edit intent
- target kinds
- constraints
- structured output
- task-format fine-tuning
- web-code pretraining
- semantic transfer
- generalization beyond exact training examples

Phase 2 is intentionally about the **model's understanding**, not application UI.

## Phase 2 completed research

The full experiment history remains in [docs/](docs/) and [training/](training/). Important findings include:

| Milestone range | Main finding |
|---|---|
| **P2-01–P2-16** | Basic request-to-code plumbing works, but early data was too small and transfer remained weak |
| **P2-17–P2-19** | Plex can fit semantic/literal/reference tasks, but unseen exact binding remains difficult |
| **P2-20–P2-23** | Broader semantic classes transfer better than exact state/literal relations |
| **P2-24–P2-29** | Plex Web corpus/tokenizer/pretraining established stronger HTML/CSS/JS priors |
| **P2-30–P2-36** | Task-format and structured-plan work improved output structure but not reliable semantics |
| **P2-37–P2-43** | Diagnostics showed training-template retrieval and weak semantic binding were major bottlenecks |
| **P2-44** | Evidence-first training improved internal validation from **3.39814 → 2.17238** over 100 updates |
| **P2-45** | Unchanged bridge evaluation still failed: **0/18 complete**, **12/18 schema-valid**, **78/162 checks** |

## P2-45 conclusion

P2-44 learned its own curriculum well, but that learning did not transfer reliably enough to independently worded coding-plan tasks.

P2-45 field results:

| Field | Correct |
|---|---:|
| response present | **18 / 18** |
| not truncated | **18 / 18** |
| schema valid | **12 / 18** |
| language | **12 / 18** |
| action | **6 / 18** |
| target kind | **12 / 18** |
| target role | **0 / 18** |
| exact constraints | **0 / 18** |
| required hint coverage | **0 / 18** |

This is a **semantic transfer failure**, not a reason to keep optimizing the unchanged P2-44 dataset.

## Remaining Phase 2 milestones

### P2-46 — Semantic Transfer Failure Diagnostic

**Status: COMPLETE**

P2-46 confirmed that P2-44 changed Plex from frequent whole-template replay toward recombining familiar semantic components, but the current request still did not reliably control which components were selected.

Key evidence:

- targetRole grounded in request: **0 / 18**
- all search hints grounded in request: **0 / 18**
- P2-44 training targetRole reuse: **17 / 18**
- P2-44 training search-hint reuse: **16 / 18**
- exact semantic-bundle reuse fell **8 → 3**
- component recombination rose **2 → 8**
- **9** tasks regressed, **2** improved

**Conclusion:** composition improved; request-conditioned composition did not.

### P2-47 — Request-Grounded Coding Representation

**Status: COMPLETE**

**Goal:** design one final Phase 2 representation specifically around transferable request-to-change understanding.

Requirements:

- grounded directly in the user's wording
- map exact request evidence to concrete coding intent
- no targetRole/searchHints as primary learning targets
- no opaque synthetic labels
- explicit train/validation combination holdouts
- validation must reuse familiar coding intents in unseen combinations
- no P2-31 development leakage
- file context remains deferred to Phase 3

**Completion:** achieved. The 108-record candidate passed real tokenizer review at **286/512 max tokens**, with **19,377 / 9,763** train/validation tokens and zero exact request/solution/non-empty binding-set overlap.

### P2-48 — Final Bounded Phase 2 Transfer Run

**Status: COMPLETE**

**Goal:** run one bounded experiment on the P2-47 representation.

Requirements:

- frozen seed/data/tokenizer/checkpoint identities
- bounded optimizer updates
- predeclared validation cadence
- no automatic continuation
- final holdout closed

**Completion:** fixed endpoint recorded with transfer metrics.

### P2-49 — Unchanged Bridge Re-evaluation

**Goal:** run the unchanged development bridge on the fixed P2-48 endpoint.

Use the same P2-31 task set and thresholds so the result is comparable with P2-39, P2-42, and P2-45.

**Completion:** recorded. The fixed P2-48 step-100 endpoint produced **0/18 complete passes, 0/18 schema-valid, and 36/162 checks** on the unchanged P2-31 gate. All 18 responses were present and non-truncated; 15 used a field set that did not match the old schema and 3 were invalid JSON.

### P2-50 — Phase 2 Closeout

**Status: COMPLETE**

**Goal:** close Phase 2 whether the final bridge fully passes or not.

Document:

- what the 27M model reliably understands
- what still fails
- which weaknesses move into Phase 3 as known limitations
- whether model size should remain 27M for Phase 3
- which evaluations become permanent regression tests

**Phase 2 ended at P2-50.** No P2-51+ continuation is planned.

Closeout decision:

- request-grounded learning improved substantially inside its own representation,
- the unchanged legacy structured-plan transfer gate still failed,
- supplied-file editing remains unproven,
- the **27,566,080-parameter model remains the controlled Phase 3 baseline**,
- model-size changes wait for direct Phase 3 editing evidence.

## Phase 2 exit condition

Phase 2 does **not** require pretending all semantic transfer is solved.

It requires:

1. completing P2-46 through P2-50,
2. running the final predeclared transfer evaluation honestly,
3. freezing the capability boundary,
4. carrying unresolved limitations forward explicitly.

The final project holdout remains closed unless separately authorized.

---

# Phase 3 — File-Conditioned Coding

**Status: ACTIVE**

## Phase 3 purpose

Make Plex useful when the user supplies a file or code block and asks for a change.

The model receives the relevant context directly.

### P3-01 — File + Request Contract

**Status: COMPLETE — contract and tokenizer/context review passed.**

Define the simplest stable input/output format for:

```text
request + supplied file/code → edited result
```

Support one supplied HTML, CSS, or JavaScript file first.

### P3-02 — Exact Small Replacements

**Status: CLOSED — bounded 100-update training finished; fixed step-100 development gate FAILED (0/36 exact edits and 0/36 unrelated-code preservation). No further training authorized.**

The reviewed candidate contains 216 authored examples (144 train / 36 validation / 36 development) in 72 intact contrast groups. Records fit within 87–138 tokens including EOS. See [P3-02 preparation](docs/PHASE-3-P3-02-EXACT-SMALL-REPLACEMENTS.md) for the frozen identities, narrow transfer boundary, proposed run, and predeclared development gate.

Examples:

- change button text
- rename a supplied variable
- change a CSS value
- replace an attribute

Measure correctness and preservation of unrelated code.

### P3-02-D1 — Read-Only Development Failure Diagnosis

**Status: full 36-response read-only diagnosis replayed during the October 18 audit.** The original development gate remains failed; 8/12 contrast groups emitted identical outputs for three different requests. [Audit recovery roadmap](docs/PLEX-NANO-MERGE-AUDIT-ROADMAP-2026-10-18.md).

After the P3-02 development gate failed, initial owner-inspected output samples showed request/source mismatches: e.g., `accountRetries = 4` was answered with `settingsRetries = 5`. Across 12 tasks per language, the owner measured 7 unique HTML outputs, 5 unique CSS outputs and 4 unique JavaScript outputs. Those observations suggest repetitive responses and weak instruction/source binding, but they do not prove model memorization or causality.

The follow-up [P3-02-D1 read-only diagnostic](docs/PHASE-3-P3-02-D1-READ-ONLY-FAILURE-DIAGNOSTIC.md) verifies and analyzes all 36 *existing* saved development completions without loading the model or creating training updates. Keep the fixed step-100 SHA, original scorer, and final project holdout untouched. P3-03 remains reserved for the original Insert and Delete research scope.

### Replacement-failure decision before harder edits

**Status: decision rules applied to the authorized 72-task fixed-endpoint evaluation; the frozen decision was both-fail, with 68/72 target boundaries unscorable.** The [saved-response review](docs/PHASE-3-P3-02-DEVELOPMENT-FAILURE-ANALYSIS.md) and replayed P3-02-D1 diagnosis cover all 36 P3-02 failures. Unextractable targets do not independently establish wrong value choices. Do not begin P3-03 Insert and Delete until replacement behavior has been addressed and independently checked.

The [prepared 72-task packet](training/phase3/diagnostics/p3-03-source-preservation-v1/PREPARATION.md) uses the proposal's historical P3-03 label; it does not rename the roadmap's P3-03 Insert and Delete milestone. Its scorer must pass controlled checks before any responses are generated. In particular, changing an unrelated source name while retaining the requested value must count as **name failure / value pass**, and changing only the requested value must count as **name pass / value failure**. If a missing or malformed target prevents an unrelated-byte check and no other preservation failure is proved, classify preservation as unscorable and review the scorer before choosing a branch.

**Predeclared thresholds for this diagnostic:** preservation passes only with **72/72** source-name-role passes **and 72/72** unrelated-byte passes. Value selection passes only with **72/72** correct target values and **24/24** contrast groups containing three distinct, correctly selected values. These are strict diagnostic thresholds, not a general coding-capability claim. Report each axis, exact-file correctness, and unscorable cases by language (**24 tasks each**) and task family (**6 tasks each**) as well as in aggregate. The manifest's separate full-pass rule remains 72/72 exact files and 24/24 fully correct groups.

| Diagnostic outcome | Next research proposal |
|---|---|
| Preservation fails; value selection passes | Propose one bounded experiment focused on retaining supplied names and every unrelated byte, while keeping request-value accuracy as a regression measure. |
| Preservation passes; value selection fails | Propose one bounded experiment with contrast requests against the same supplied file, measuring whether the requested replacement value changes correctly. |
| Both fail | Propose a simpler controlled task that separately measures copying the supplied code and applying a requested change before combining them. |
| Both pass | Investigate the disagreement with the failed 36-task gate, then confirm on independent replacement tasks before advancing to harder edits. |
| Scoring ambiguous or unreliable | Repair and validate the scorer before interpreting model results. |

Each branch selects a proposal for review, not an automatic training run. Evaluation, optimizer updates, and final-holdout access require their own authorization; the failed P3-02 gate remains the recorded outcome.

The October 17 fixed-endpoint run selected **both fail**: 0/72 source-name passes, 0/72 requested-value passes, 0/72 exact files, and 0/24 fully correct contrast groups. Target extraction failed on 68/72 outputs, so those records cannot isolate value choice inside a valid target; all 72 still visibly lost supplied names. See the [full result and selected copy-versus-edit proposal](docs/PHASE-3-REPLACEMENT-FAILURE-DECISION.md). The selected smaller copy-versus-edit control now has [executable, read-only preparation](training/phase3/diagnostics/p3-copy-edit-controls-v1/PREPARATION.md): 6 copy controls, 18 minimal edits, and 18 context edits, with frozen-source overlap checks and synthetic regression tests. Its [42-task manifest](training/phase3/diagnostics/p3-copy-edit-controls-v1/copy-edit-controls-v1.json) is now **frozen and SHA-256 pinned**, with guarded local step-100 inference and a strict read-only scorer. The owner subsequently ran the pinned evaluation locally and reported **0/42 exact files, 0/6 exact copies, 0/18 minimal edits, 0/18 contextual edits**, and all 36 target values unscorable under changed source boundaries. The decision is **copy-failure first**. The [frozen 288-train / 96-validation copy-fidelity curriculum](training/phase3/copy-fidelity-v1/README.md) was packed, staged, then used in **one explicitly authorized 100-update CUDA run**, with SHA-verified fixed step-100 endpoint `c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230`. The owner reported lower teacher-forced loss (**7.1167 → 2.7984**) but **0/96** exact validation generated files and **0/42** exact historical developer controls. EOS and full prompt visibility were present for all 138. This closed experiment therefore **failed the copy/edit capability gate**. The original model, artifacts, failed results and final holdout remain frozen. No more training is authorized. See [completed experiment #181](https://github.com/keepithandy/plex-nano/issues/181). This does not advance P3-03 Insert and Delete.

### P3-SB — Source-Binding Recovery (before P3-03)

**Status: SB-00 existing-output diagnosis measured; SB-01 fixture/tokenizer verified, not model-evaluated.** [Technical review and evidence](docs/PHASE-3-SOURCE-BINDING-TECHNICAL-REVIEW.md), [six-gate roadmap](docs/PHASE-3-SOURCE-BINDING-RECOVERY-ROADMAP.md) and [successor issue #189](https://github.com/keepithandy/plex-nano/issues/189). All 138 saved outputs omit the supplied namespace and contain training-name text; exact files remain 0/138. The 48-case fixture is unchanged and SHA-pinned. Owner review precedes further model evaluation or training.

**SB-02 hypothesis A (training-namespace diversity) was explicitly approved for preparation.** [SB-03 matched candidate and safety review](docs/PHASE-3-SB03-APPROVED-NAMESPACE-DIVERSITY.md) now freezes a **proposed** 288/96 comparator/intervention dataset with 48 new training namespaces and a complete 1,600-draw full-coverage sampler. The **owner ran the real tokenizer audit, which FAILED on 240/384 paired rows** (40,439 comparator vs 44,925 intervention draw-weighted supervised tokens). The original V1 proposal is preserved as a failed feasibility result. October 18's [audit recovery roadmap](docs/PLEX-NANO-MERGE-AUDIT-ROADMAP-2026-10-18.md) records successful **V2 tokenizer-only feasibility: 48/48 names, 384/384 token signatures matched** against five P3 corpora, with no V2 file saved. A reviewed frozen V2 and arm-specific scoring provenance remain pending. Complete historical-corpus screening and both fresh arm authorizations remain outstanding. **SB-05** remains a pure scorer tested with synthetic strings. No training bundle, checkpoint stage, live generation, optimizer updates or final-holdout access is authorized. Keep 27M parameters, original tokenizer/context, fixed step-100 provenance and sealed final project holdout. Report exact-file and source-namespace preservation as primary capability metrics, not validation loss alone.

### P3-03 — Insert and Delete

Teach:

- add requested markup
- add a CSS rule/property
- add a JavaScript condition
- remove requested code

### P3-04 — Modify Existing Logic

Handle bounded changes to:

- functions
- event handlers
- DOM code
- CSS components
- HTML structures

### P3-05 — Preserve Unrelated Code

Add explicit regression tests for accidental edits outside the requested scope.

### P3-06 — Multiple Changes in One Supplied File

Support two or more related requested changes without damaging untouched sections.

### P3-07 — Edited-File and Patch Output

Measure both:

- full-file generation/editing
- bounded patch/edit output

Choose the more reliable default from evidence.

### P3-08 — Validation-Aware Correction

Given a concrete syntax/test failure from the caller, let Plex make one corrected attempt.

No open-ended retry loop.

### P3-09 — Unseen File Benchmark

Freeze a development benchmark using code files excluded from training.

Measure:

- correct requested behavior
- syntax
- scope preservation
- unnecessary changes
- repair success

### P3-10 — Phase 3 Closeout

Record what kinds of supplied-file edits Plex can actually perform reliably.

**Phase 3 gate:** Plex can make useful, measured edits to unseen supplied HTML/CSS/JavaScript files from natural-language requests.

---

# Phase 4 — Stronger Coding Ability

**Status: PLANNED**

Move beyond simple edits.

Targets include:

- more complex JavaScript logic
- bug fixes
- refactoring
- state changes
- DOM behavior
- CSS interactions
- larger supplied files
- longer dependencies within supplied context
- better explanations of changes
- stronger error correction

Model-size changes may be considered here only if 27M limitations are clearly measured.

---

# Phase 5 — Conversational Code Editing

**Status: PLANNED**

Support an ongoing conversation about supplied code.

Examples:

```text
"Make this button smaller."
"Actually keep the width but reduce the padding."
"Also add a disabled state."
"Why did you change this selector?"
```

Goals:

- retain relevant supplied-code context across turns
- understand revisions to earlier instructions
- modify previous edits
- explain changes when asked
- avoid drifting into unrelated code

---

# Phase 6 — Local Plex App

**Status: PLANNED**

Wrap the coding model in a simple local experience.

Target interaction:

```text
drop in a file
→ chat about the code
→ request a change
→ preview the result
→ accept/save
```

The app exists to make the model easy to use. The model remains the product focus.

---

# Development Rules

1. **Plex Nano is a small coding model.**
2. The user/caller supplies relevant code or file context.
3. Keep the 27M model as the controlled baseline until a milestone explicitly changes it.
4. Prioritize coding ability over infrastructure expansion.
5. Preserve failed experiments and negative results.
6. Keep training authorization separate from preparation/evaluation.
7. Do not optimize indefinitely on a failed representation.
8. Keep development and final holdouts separate.
9. Do not claim capability from training loss alone.
10. Prefer unseen request/code tests over memorized training checks.
11. Preserve unrelated code in editing evaluations.
12. Keep HTML/CSS/JavaScript as the initial language scope.
13. Update README, roadmap, package metadata, prompts, and AGENTS files together when product direction changes.

Project-wide guardrails are defined in [AGENTS.md](AGENTS.md).

---

# Current task

## P3-02 — Replacement Failure Decision

Phase 2 is complete.

The active task is to review the [October 18 merge-audit recovery roadmap](docs/PLEX-NANO-MERGE-AUDIT-ROADMAP-2026-10-18.md), beginning with publication of the local reporting repairs, then historical-source screening and a reviewed V2 comparison. The earlier copy/edit and copy-fidelity runs are complete with failed capability gates; no new training is authorized. The frozen model contract remains:

```text
natural-language coding request
+ one supplied HTML/CSS/JavaScript file
                ↓
             Plex Nano
                ↓
        correct edited result
```

P3-01 froze the one-file request/edit contract and passed deterministic fixture, preservation, tokenizer roundtrip, and context checks. The 18-fixture contract-validation set is not a representative training corpus and no editing capability is claimed from its preparation.

**Historical P3-02 sequence (completed and failed gate):** The owner reported successful local frozen-tokenizer, train/validation bundle (SHA `0cd6df3a...b9aca354`), and P2-48 source-checkpoint (SHA `fd86d113...89b7`) verification, now recorded in the P3-02 evidence file. Guarded zero-step checkpoint staging, strict tensor-preservation verification, and 36-record answer/EOS-only validation baseline preflight were implemented and initially awaited the owner's local execution and baseline review. Stage creation requires explicit opt-in; model updates remain disabled and need separate bounded-run authorization. The owner next reported an unchanged P3-02 stage checksum `50f18e46...622246` and **36/36 validation records / 1,320 supervised targets**, mean answer/EOS loss **10.5668395360** (CSS 10.0458; HTML 10.9634; JavaScript 10.8315). The local result is preserved separately from CI. A strict draft authorization contract, bounded 100-step/600-second CUDA runner, and read-only baseline/sampler replay preflight have been implemented for review; the owner explicitly authorized the first bounded run on 2026-10-13; the checked-in packet now pins that single-run approval, still requiring a separate CLI execution flag. **The owner reported the first bounded 100-update run completed:** 60,445 supervised positions in 19.91 seconds, held-out validation loss **10.5668395 → 1.5892544**. The official fixed step-100 checkpoint SHA `020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d` and all interim checkpoint hashes are now frozen in `training/pretraining/p3-02-first-run-result.json`. The owner subsequently ran the read-only frozen endpoint evaluator: **development gate FAILED** (0/36 exact edited files, 0/36 unrelated-code preservation, 0/12 fully correct contrast groups, and 0/12 exact in each HTML, CSS, JavaScript; all 36 EOS-terminated, decoded, and maintained full prompt visibility). The summary is preserved in `training/pretraining/p3-02-development-gate-result-summary.json`; underlying generated files remain in the owner's local result. **Do not proceed as if validation-loss improvement established editing proficiency.** The October 17 read-only review of the saved completions reproduced the frozen scores: all 36 outputs lose supplied names, 8/12 request groups repeat identical code, and 25/36 outputs exactly match training targets. See [saved-response diagnosis](docs/PHASE-3-P3-02-DEVELOPMENT-FAILURE-ANALYSIS.md). The subsequent [72-task diagnostic](training/phase3/diagnostics/p3-03-source-preservation-v1/PREPARATION.md) was completed and failed as recorded above; no additional training, checkpoint reselection, or final-holdout access is authorized.

Start with **one supplied file**, one bounded requested change, and a deterministic target representation that can be scored for:

- requested behavior correctness,
- syntax validity,
- preservation of unrelated code,
- unnecessary changes.

Keep the current **27,566,080-parameter** model as the controlled baseline.

Do not add repository discovery, autonomous navigation, or application-agent behavior. The caller supplies the code context.
