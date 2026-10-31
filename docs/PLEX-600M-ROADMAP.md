# Plex 600M development roadmap

**Status: proposed milestone sequence; current work is M3 source/split planning.**
This roadmap covers the separate, randomly initialized 600M research model.
The [Nano roadmap](../Plex-ROADMAP.md) and controlled 27M baseline retain their
own recorded outcomes. Coding work remains request plus supplied code/file
context to correct code changes. No milestone automatically authorizes model
execution, training, or final-holdout access.

| Milestone | State | Exit evidence |
|---|---|---|
| M1 — Architecture and foundation | Implemented | Exact architecture, isolated scratch tokenizer preparation, tiny-model tests and recorded Linux/Windows CI |
| M2 — Input integrity and initial hardware feasibility | Implemented within stated limits | One fixed full-model forward/backward pass; verified index/token snapshots and explicit source-binding checker; [evidence and limits](PLEX-600M-INPUT-INTEGRITY-PLAN.md) |
| M3 — Source and evaluation-split decision | Next: planning | Reviewed source inventory, exact scope and identities, recorded rights decisions, related-source grouping and independent evaluation design |
| M4 — Reproducible corpus preparation | Pending M3 decision | Approved source revisions, notices, frozen train/validation manifests, filtering and overlap reports, declared storage bounds |
| M5 — Dedicated tokenizer and packed inputs | Pending M4 | Train-only scratch BPE fit, exact roundtrips, externally pinned bundle/source identities, successful source-check and declared snapshot/sharding limits |
| M6 — Optimizer feasibility | Proposal and separate execution decision needed | Fixed precision/optimizer configuration, bounded memory/update check, finite-state checks, actual measurements and failure record |
| M7 — First bounded training pilot | Pending data and hardware gates | Reviewed limits, fixed starting state, seed, objective, sampling, validation and checkpoint contract; explicit training authorization |
| M8 — Coding evaluation and next decision | Pending saved weights and evaluation authorization | Frozen supplied-code tasks, raw outputs, syntax/behavior and exact-edit/preservation scores; documented failures and next hypothesis |

M2 establishes one workload fitting, not optimizer or sustained-training fit.
The snapshot input path caps each selected token split at 256 MiB; exceeding
that bound requires a separately reviewed scalable input design before use.
There is no evidence yet of 600M coding ability or an appropriate training-data
volume. Do not turn the current historical corpus into a scale recommendation.

## M3 concrete review packet

Prepare one inventory with a row per proposed source family. Record source
identity and immutable revision, content/language role, intended file scope,
notice evidence, rights-decision scope, related repositories and overlap risks.
Record quantities as measured or unknown, never estimated as accepted data.
The [historical P2-26 review](PLEX-600M-DATA-READINESS-REVIEW.md) is a candidate
starting point; it is not an approved 600M corpus. No new source is selected by
this roadmap.

Before choosing or fitting the tokenizer, specify:

- Training sources and source families reserved for validation; keep related
  repositories together and report per-family counts rather than only totals.
- Which historical material has already been inspected or loss-monitored;
  treat the old P2-26 validation as historical validation, not a fresh benchmark.
- A fresh development evaluation design for HTML/CSS/JavaScript requests with
  caller-supplied code, checking behavior, syntax, unrelated-code preservation
  and exact edited files. Freeze tasks and outcome rules before model exposure.
- How source/task overlaps and duplicates will be checked, what the checks can
  establish, and which uncertainty remains. Keep the final project holdout closed.
- The proposed preparation scope and storage budget, followed by its explicit
  decision record. Source review, building data, fitting a tokenizer, running
  model evaluation and training are distinct actions.

**M3 exit:** a concrete source/split packet is reviewed and its choices recorded.
Unknown source scope, rights, or split independence keeps the gate open; it
does not justify silently reusing a historical approval.
