# Plex Nano documentation index

[Back to the VS Code project map](../PROJECT-MAP.md) · [Project README](../README.md) · [Roadmap](../Plex-ROADMAP.md)

This folder contains the **historical research record**, not a list of current product features. Its many `PHASE-2-...` files are retained intentionally so experimental contracts, failed evaluations, and results remain traceable. The current research target is a **27,566,080-parameter scratch-trained model** for edits to user-supplied HTML, CSS, and JavaScript.

## Read these first

| Question | Document |
|---|---|
| What is Plex trying to accomplish? | [Main README](../README.md) and [roadmap](../Plex-ROADMAP.md) |
| How do I diagnose P3-02 failures without training? | [P3-02-D1 read-only diagnostic](PHASE-3-P3-02-D1-READ-ONLY-FAILURE-DIAGNOSTIC.md) |
| What happened in P3-02? | [P3-02 exact replacements](PHASE-3-P3-02-EXACT-SMALL-REPLACEMENTS.md) |
| What did the saved failed responses show? | [October 17 response review](PHASE-3-P3-02-DEVELOPMENT-FAILURE-ANALYSIS.md): source-name loss and repeated outputs |
| What did the 72-task source/value diagnostic show? | [Replacement-failure decision](PHASE-3-REPLACEMENT-FAILURE-DECISION.md): both axes failed; next controlled proposal |
| Did P3-02 pass unseen file-edit tasks? | [Frozen development-gate summary](../training/pretraining/p3-02-development-gate-result-summary.json): **No — 0/36 exact edits** |
| What was the fixed trained model checkpoint? | [Frozen first-run result](../training/pretraining/p3-02-first-run-result.json): step 100, fixed SHA-256 |
| What was the Phase 2 conclusion? | [P2-50 closeout](PHASE-2-P2-50-CLOSEOUT.md) |
| What is the package/reliability audit? | [Package audit](PACKAGE-BUG-AUDIT.md) and [Bug Atlas](../BUG-ATLAS.md) |

## Research chronology

| Section | Entry points | What you find |
|---|---|---|
| **Phase 1 — foundations** | [First experiment](FIRST-EXPERIMENT.md), [hardware profile](HARDWARE-AND-TRAINING-PROFILE.md) | Scratch model, training environment, first pilot setup |
| **Phase 2 — early binding and code tasks** | [Roadmap](../Plex-ROADMAP.md) | Request following, compositional binding, instruction format |
| **Phase 2 — model and data preparation** | [Plex Web data collection](PHASE-2-P2-24-WEB-PRETRAINING-SPEC.md), [dataset pipeline](DATASET-PIPELINE.md) | Data sourcing, tokenizer, provenance, scratch pilot |
| **Phase 2 — later transfer** | [P2-44 evidence-first work](PHASE-2-P2-44-EVIDENCE-FIRST-SEMANTIC-COMPOSITION.md), [P2-48 final run](PHASE-2-P2-48-FINAL-TRANSFER-RUN.md), [P2-49 bridge reevaluation](PHASE-2-P2-49-UNCHANGED-BRIDGE-REEVALUATION.md) | Frozen bounded runs and their limitations |
| **Phase 2 — completed** | [P2-50 closeout](PHASE-2-P2-50-CLOSEOUT.md) | Final Phase 2 capability boundary |
| **Phase 3 — current** | [P3-02 exact replacements](PHASE-3-P3-02-EXACT-SMALL-REPLACEMENTS.md), [saved-response diagnosis](PHASE-3-P3-02-DEVELOPMENT-FAILURE-ANALYSIS.md) | Failed development gate, observed failure patterns, and diagnostic proposal direction |

The complete original files remain in this folder. Use VS Code **Ctrl+P** and type the milestone number, such as `P2-48` or `P3-02`, to locate additional reports quickly.

## What is stored where?

- **Narrative reports:** this `docs/` folder.
- **Immutable contracts, approvals, manifests, experiment numbers:** `training/pretraining/`.
- **Authored and reviewed record candidates:** `training/phase3/drafts/` (current) and `training/phase2/` (historical).
- **Executable model implementation:** `training/src/plex_training/`.
- **Python tests and safety regressions:** `training/tests/`.
- **Actual local checkpoints and evaluation JSON:** ignored `training/artifacts/` or the owner's external `PlexArtifacts` folder; they are not GitHub-hosted model weights.

**Status caveat:** training-loss improvement does not establish editing ability. The first P3-02 frozen endpoint reached 1.5892544 answer/EOS validation loss, but the 36-task development gate failed all exact edits and preservation checks. No more training or use of the final project holdout is approved.
