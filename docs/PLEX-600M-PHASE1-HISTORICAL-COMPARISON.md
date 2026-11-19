# Plex 600M — Phase 1 historical comparison report

**Status:** documentary comparison completed; **local binary re-verification
pending owner-machine execution**. This is not a verified-new-run claim.
The M9/M10 original checkpoint and raw generated responses are excluded from
GitHub; the local verification tool in PR #214 is required before a new
byte-checked historical report can be released.

## Source-matched research record

The completed M10 read-only M9 baseline and the saved M10 model snapshots
were tested on the 18 M10 copy instructions, which included 12 train and
six evaluation-only snippets. Both established decoders generated output
separately. No M11-style space boundary is included in this comparison.

Published M10 warmup diagnostics report:

| Metric, 18 M10 prompts | M9 | M10 step 10 | M10 step 25 | M10 step 50 |
| --- | ---: | ---: | ---: | ---: |
| Autonomous exact copying in targeted probes | 0/18 | 0/18 | 0/18 | 0/18 |
| Gold-prefix correct content tokens | 1/286 | 27/286 | 24/286 | 35/286 |
| Correct EOS after gold answer | 0/18 | 12/18 | 18/18 | 18/18 |
| Bidirectionally source-correct substitutions | 0/18 | 0/18 | 0/18 | 0/18 |

These are the outcomes in the committed research document, not totals
freshly recomputed from the original JSONL files in this GitHub session.

Source: [M10 warmup diagnostics](PLEX-600M-M10-WARMUP-DIAGNOSTICS.md).

## Conclusions supported by published results

**Checkpoint learning occurred, but copying did not become reliable.**
The increase in gold-prefix correctness and terminal EOS decisions did
not translate into any autonomous exact copies on the targeted 18 cases.

**Prompt boundaries mattered.** An input space instead of newline
changed immediate EOS behavior, yet the ten-update space-trained
replay produced the same constant output for all 18 inputs and no
correct copies. This is an output-conditioning problem, not merely a
termination-format problem.

**M11 is the appropriate next experiment, not more blind M10 updates.**
M11 should measure whether changing the supplied variable's digit
changes the model's copied digit under the same evaluation protocol.

Sources: [prompt delimiter](PLEX-600M-M10-PROMPT-DELIMITER.md),
[space replay](PLEX-600M-M10-SPACE-REPLAY.md), and
[M11 design](PLEX-600M-M11-SOURCE-CONDITIONING-PROPOSAL.md).

## Fresh local validation: required, not yet executed here

The CPU-only historical verification tool compares the **full original
response files** (18 cases under both frozen policies, 36 responses per
checkpoint). It validates the historical checkpoint hashes and re-scores
both training/evaluation-only splits independently.

After owner-local execution succeeds, the authoritative outputs will be:

- checkpoint-registry.json — M9 and M10 step 10/25/50 model weights,
  byte sizes and SHA-256 with precise parent checkpoint links.
- historical-comparisons.json — source-matched M9 versus each M10
  checkpoint; independent exact+EOS scores per decoder and split, plus
  all 36 before-and-after raw responses for each compared checkpoint.
- experiment-decisions.json — per-checkpoint evidence and stop/promotion
  decisions. M11 is marked not evaluated.

The output status distinguishes **local file and receipt integrity**
from proof of model-generation origin. Even when those outputs are
verified, a completed history audit does not itself establish new
generalization or authorize more training.

If local evidence cannot pass the required checks, Phase 1 remains
incomplete and these documented comparisons must not be upgraded into
a verified original-checkpoint result.
