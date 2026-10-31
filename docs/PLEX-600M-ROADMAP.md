# Plex 600M · Development Roadmap

> **Current milestone:** M3 — source and evaluation-split planning<br>
> **Model state:** scratch architecture implemented; no trained 600M checkpoint<br>
> **Last reviewed:** October 31, 2026

Plex 600M is a separate scratch-model research track. Its coding goal is to
understand a natural-language request and make the correct change to HTML, CSS,
or JavaScript supplied by the caller. Progress is measured through the model's
outputs and their correctness, including preservation of unrelated code.

The existing **27,566,080-parameter Plex Nano** remains the controlled baseline
with its own [roadmap](../Plex-ROADMAP.md) and recorded results. The 600M track
does not replace that baseline or establish that a larger model solves its
failures. Both tracks depend on supplied code context; repository discovery and
autonomous workflows are not product milestones.

[Current position](#current-position) · [Milestones](#milestones) ·
[M3 review packet](#m3--source-and-evaluation-split-decision) ·
[Research boundaries](#research-boundaries) · [Evidence](#evidence)

---

## Current position

| Area | Evidence today | Still to establish |
|---|---|---|
| Architecture | A 600,372,480-parameter decoder is implemented with random initialization, a 1,024-token context and 32,768 vocabulary slots. | Whether trained weights learn useful language and coding behavior. |
| GPU feasibility | One full-size, synthetic 1,024-token forward/backward check passed on the RTX 4080 SUPER. | Optimizer-state memory, an update, sustained training and throughput. |
| Input integrity | Index hashes, bounded immutable token snapshots and an explicit source-binding check are implemented and tested. | A reviewed scalable input design if a selected packed split exceeds 256 MiB; independent source and manifest identities. |
| Data and tokenizer | Preparation tools exist. The historical P2-26 corpus is documented as a limited candidate. | A reviewed 600M corpus and split; a dedicated fitted 600M tokenizer. |
| Model capability | No trained 600M checkpoint or coding evaluation exists. | Instruction following, syntax, requested behavior and preservation on supplied code. |

The GPU result is evidence for **one fixed workload**, not a claim that a
training configuration fits. Training loss, once available, will be reported
alongside coding evaluations rather than treated as proof of coding ability.

## Milestones

| Milestone | Status | Completion evidence |
|---|---|---|
| [M1 · Architecture and foundation](#m1--architecture-and-foundation) | Implemented | Isolated architecture, preparation tools and recorded small-configuration CI. |
| [M2 · Input integrity and initial GPU check](#m2--input-integrity-and-initial-gpu-check) | Implemented within stated limits | One fixed full-size forward/backward result; tested input-integrity checks and documented bounds. |
| [M3 · Source and evaluation-split decision](#m3--source-and-evaluation-split-decision) | **Current planning gate** | A reviewed, concrete source inventory, split proposal and development-evaluation design. |
| [M4 · Reproducible corpus preparation](#m4--reproducible-corpus-preparation) | Waiting for M3 decisions | Reviewed inputs, frozen identities, filtering and overlap reports, and reproducible train/validation manifests. |
| [M5 · Dedicated tokenizer and packed inputs](#m5--dedicated-tokenizer-and-packed-inputs) | Waiting for M4 | Train-only scratch tokenizer, pinned bundle and source identities, roundtrips and source-binding verification. |
| [M6 · Optimizer feasibility](#m6--optimizer-feasibility) | Proposal and separate execution decision needed | Bounded hardware contract and recorded outcome for the proposed optimizer configuration. |
| [M7 · First bounded training pilot](#m7--first-bounded-training-pilot) | Waiting for data and hardware gates | A reviewed run contract, explicit training authorization and preserved result. |
| [M8 · Coding evaluation and next decision](#m8--coding-evaluation-and-next-decision) | Waiting for saved weights and evaluation decision | Predeclared task scores, raw outputs, failure analysis and a recorded next research decision. |

M3–M8 are a **proposed sequence**, not completed or approved experiments. No
schedule, data volume or capability threshold has been fixed for them yet.

### M1 · Architecture and foundation

**Purpose:** provide a reproducible 600M scratch-model foundation without
changing the Nano baseline.

**Delivered:** the standalone decoder, analytic parameter count, small-model
forward/backward and masking tests, isolated tokenizer preparation, and
record-bounded batch sampling. The [architecture review](PLEX-600M-ARCHITECTURE-REVIEW.md)
and [development record](PLEX-600M-DEVELOPMENT.md) preserve their scope and
earlier CI evidence.

**Boundary:** small-configuration tests verify implementation behavior; they
do not demonstrate full-size training or coding capability.

### M2 · Input integrity and initial GPU check

**Purpose:** establish a bounded first hardware observation and prevent a
changed input file from silently altering sampled token bytes.

**Delivered:** one separately authorized synthetic forward/backward pass at
batch size one and 1,024 tokens. It produced finite loss and 218 finite FP32
gradient tensors; peak allocated and reserved memory were approximately
4,947 and 6,140 MiB. The [recorded result](PLEX-600M-CUDA-FEASIBILITY-PROPOSAL.md#authorized-cuda-result--october-31-2026)
contains the exact configuration and limitations.

The [input-integrity work](PLEX-600M-INPUT-INTEGRITY-PLAN.md) verifies packed
index identities, samples from immutable verified token snapshots and offers an
explicit source-record comparison against separately supplied manifest hashes.
The sampler does not automatically certify source rights or source binding.

**Remaining limitation:** each selected packed token split is capped at
256 MiB. Larger splits need a reviewed scalable design before use; a passing
forward/backward check does not establish that optimizer allocations fit.

### M3 · Source and evaluation-split decision

**Purpose:** decide what the model may learn from and how to measure progress
on coding requests before any new data is exposed to it.

**Deliverables:**

1. A source-family inventory with source identity, immutable revision, proposed
   content and language scope, license/notice evidence, rights-decision scope,
   related repositories and known overlap risks. Mark unmeasured quantities as
   unknown instead of presenting them as accepted corpus totals.
2. A train/validation split proposal that keeps related sources together and
   records per-family counts and distribution. Explain prior exposure and any
   limits to independence.
3. A development-evaluation design for requests paired with supplied HTML, CSS
   and JavaScript. Specify expected edits and checks for syntax, requested
   behavior, exact files and unrelated-code preservation; freeze the tasks and
   outcome rules before model exposure.
4. A review of duplicate, secret, content-quality and protected-evaluation
   overlap checks, including what each check cannot establish. State the
   proposed preparation scope and storage budget.

**Starting evidence:** the [data readiness review](PLEX-600M-DATA-READINESS-REVIEW.md)
records a historical corpus of 3,140 records across three source groups. Its
validation split was already measured during Nano tokenizer selection and is
not a fresh benchmark. Its previous approvals do not by themselves approve a
600M corpus.

**Exit gate:** review the concrete packet and record its source, split and
evaluation decisions. Missing rights evidence, unclear source scope or
unresolved split independence leaves the gate open. The final project holdout
stays closed.

### M4 · Reproducible corpus preparation

**Purpose:** turn the reviewed M3 decisions into traceable training and
validation inputs.

**Deliverables:** exact source revisions and retained notices; deterministic
grouped splits; content and manifest hashes; source counts and composition;
filtering, duplicate and overlap reports; and recorded storage limits. Explain
skipped records and the coverage limits of every check.

**Exit gate:** independently review the prepared inputs against the approved
scope, verify their identities and split assignment, and freeze the manifest.
Exact deduplication alone does not prove the absence of related or
near-duplicate material. This gate does not authorize tokenizer fitting or
model training.

### M5 · Dedicated tokenizer and packed inputs

**Purpose:** give the 600M model its own reproducible scratch tokenizer and
connect packed records back to the reviewed source.

**Deliverables:** a train-only byte-level BPE fit, tokenizer settings and
identity, actual vocabulary size, exact-text roundtrips, context/encoding
checks, packed split and index hashes, and independently pinned bundle and
source-manifest identities. Run the explicit source-binding check on the
intended inputs when that work is authorized.

**Exit gate:** review the resulting bundle, source comparison, split boundaries
and storage limits. The 32,768 vocabulary size is a capacity; actual learned
vocabulary and token counts must be measured. Resolve the 256 MiB per-split
snapshot limit before using a larger split.

### M6 · Optimizer feasibility

**Purpose:** find out whether a specific training configuration fits the
target hardware rather than extrapolating from the forward/backward check.

**Deliverables:** a separately reviewed proposal identifying model source and
initialization, synthetic inputs, precision, optimizer and state dtypes,
allocation and time bounds, finite-loss/gradient/state checks, cleanup and
stop rules. Record device/software versions, peak memory, elapsed time and any
failure. Define exactly whether an optimizer update is included.

**Exit gate:** execute only the separately authorized bounded check and retain
its result, including OOM or interruption. A pass applies to its declared
configuration; it does not establish sustained training stability or coding
ability.

### M7 · First bounded training pilot

**Purpose:** test whether the scratch model can learn from the reviewed corpus
under a fixed, reproducible training contract.

**Deliverables before execution:** immutable data/tokenizer identities; random
starting-state and seed policy; objective and sampler; optimizer settings;
update, time and storage limits; validation schedule; checkpoint and resume
policy; finite-state aborts; and a predeclared endpoint. Keep the 27M baseline
and its checkpoints separate.

**Exit gate:** obtain explicit authorization for the specified run, then
preserve its configuration, hashes, measurements and outcome, including an
incomplete or failed run. Do not select a different endpoint after seeing
results unless the contract allowed it in advance.

### M8 · Coding evaluation and next decision

**Purpose:** measure what the saved model can do with a request and supplied
code, and use observed failures to choose the next bounded improvement.

**Deliverables:** an authorized evaluation of predeclared saved weights and
development tasks; raw completions and per-case provenance; syntax and
requested-behavior results; exact-file, unrelated-code-preservation and
unnecessary-change measures; and breakdowns by language and task family.
Describe unscorable cases rather than converting missing evidence into a pass.

**Exit gate:** review the actual outputs against the frozen criteria and record
both the capability boundary and a next hypothesis. A lower loss or a small
pilot result alone is not proof of general coding reliability.

## Research boundaries

- Initialize the 600M model from random weights; do not load Nano checkpoints
  as 600M weights or change the controlled Nano baseline.
- Treat preparation, diagnostics, evaluation, checkpoint staging, optimizer
  updates and training as distinct actions with their own decisions. The final
  project holdout remains sealed unless explicitly authorized by a milestone.
- Preserve source, dataset, tokenizer, checkpoint and result identities. Record
  failed and incomplete runs without rewriting historical numbers.
- Prefer one bounded hypothesis and a predeclared outcome over extending a run
  automatically. Do not claim model capability before it is measured.

## Evidence

- [600M development record](PLEX-600M-DEVELOPMENT.md) — implementation,
  hardware target and chronological status.
- [GPU feasibility result](PLEX-600M-CUDA-FEASIBILITY-PROPOSAL.md) — fixed
  workload, measured memory, source identity and result limits.
- [Data readiness review](PLEX-600M-DATA-READINESS-REVIEW.md) — historical
  source evidence and unresolved 600M data decisions.
- [Input integrity record](PLEX-600M-INPUT-INTEGRITY-PLAN.md) — verified
  snapshots, source checking, tests and the per-split size bound.
- [Nano roadmap](../Plex-ROADMAP.md) — separate 27M research status and
  source-binding recovery work.
