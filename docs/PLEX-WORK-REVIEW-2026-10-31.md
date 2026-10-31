# Plex work review — October 31, 2026

## Scope and method

This is a navigation and preparation review, not an experiment authorization.
The local checkout is `75000c9`; GitHub `master` was verified at
`b1586d04a5ff74c85f161537706258ef8b9fc351`, 18 commits ahead. The latter adds
the separate 600M foundation and two Markdown records. The local tree contains
245 Markdown files; their paths and status/next-step markers were inventoried,
and the current roadmaps, closeout, recovery decisions and 600M records were
reviewed in detail. This is not a line-by-line independent review of all 247
Markdown files. Dataset notices and examples are retained as source records;
historical experiment plans are not automatically current tasks.

## Current routes

### 600M scratch foundation

The [600M development record at the reviewed master commit](https://github.com/keepithandy/plex-nano/blob/b1586d04a5ff74c85f161537706258ef8b9fc351/docs/PLEX-600M-DEVELOPMENT.md)
identifies a separate, randomly initialized 600,372,480-parameter model. It
preserves Nano as a separate baseline. Architecture, tokenizer preparation and
record-bounded sampling exist; the record reports Linux and Windows tests on
small configurations. These are not evidence of full-size GPU fit or coding
ability. A full tokenizer, large reviewed dataset, optimizer runner and trained
checkpoint do not yet exist according to that record.

The next proposed independent gate is a **reviewed, bounded CUDA memory and
forward/backward feasibility check**. Before execution, a concrete packet should
identify the exact code and configuration, synthetic inputs, precision, sequence
length and batch size, device and software versions, memory and time limits,
failure handling, and observations to record. It should specify that no
checkpoint or model output is retained, no existing artifacts are loaded or
overwritten, and no optimizer update occurs. Forward/backward feasibility alone
would not prove that optimizer-state allocation or a training step fits.
The specific parameters and permission to run the check remain open; this
review does not authorize model construction, inference, backward, or training.

Later prerequisites include rights and source review, disjoint hash-pinned data,
an independently reviewed tokenizer fit, and a distinct optimizer/training
contract with explicit authorization. The final project holdout stays sealed.

### Nano source-binding recovery

The [active Nano roadmap](PLEX-NANO-MERGE-AUDIT-ROADMAP-2026-10-18.md)
and [R3 comparison review](PHASE-3-R3-COMPARISON-DESIGN-REVIEW-2026-10-18.md)
propose a distinct six-versus-48 screened-name comparison. The recorded
model-free witness matches 384 paired token lengths and screens 54 spellings
against the ten listed sources. That supports feasibility within the stated
history, not model performance or global novelty. Adoption, external independent
review, separate arm manifests and arm-specific scoring provenance are pending.
The original V2 failed its source-exclusion gate with eight name collisions and
is not replaced by this witness.

The prior copy-fidelity run remains a failed capability result: 0/138 exact
files across its recorded validation and historical controls. The 96 validation
cases and previously inspected controls are not untouched final benchmarks.
Packed bundles, checkpoint staging, each training arm and model evaluation need
their own decisions and authorization. R6 fresh confirmation and P3-03
insert/delete follow only after the prerequisite recovery evidence.

## Historical and maintenance items

- [P2-50](PHASE-2-P2-50-CLOSEOUT.md) closed Phase 2 with failed structured-plan
  transfer; older Phase 2 documents and draft reviews can still say “next” or
  “pending” in their historical context. They do not reopen Phase 2.
- [Bug Atlas](../BUG-ATLAS.md) reports its 24 tracked items complete; it also
  records unverified branch protection and the manual historical experiment
  suite. That is not a claim that all current software is defect-free.
- [Documentation index](INDEX.md) still describes historical inventory and V2
  screening as pending, although the later R2 reports record the completed
  screening and failed V2 gate. It also lacks the new 600M records. Update it
  against the current checkout when the newer master files are present.
- The current 600M development record contains two literal `\n` sequences inside
  its status table, disrupting rendering. Its architecture review also describes
  the tokenizer as not built, whereas the later development record distinguishes
  implemented preparation from a full tokenizer not yet fitted. These are
  documentation maintenance candidates on the newer master, not changes to
  historical experiment results.
- Some historical training documentation still names prior objectives as next.
  Use dated closeouts and the active roadmap to interpret them; do not silently
  rewrite frozen reports or numeric results.

## Boundaries and verification

The user's supplied project instructions keep Plex Nano centered on a small coding
model receiving user-provided code or file context. The 600M code is a separate
research line and does not replace the controlled 27M baseline. This review used
repository metadata and text only; it did not load checkpoints, run model code,
fit a tokenizer, prepare training data, access the final holdout or perform GPU
work. Reported CI and historical scores are claims in the cited records, not
newly reproduced results.
