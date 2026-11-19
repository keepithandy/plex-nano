# M11 — bounded, owner-approved source-conditioning training worker

**Development status:** implementation under review; **no M11 training has started**.
This module does not itself approve any training. The exact prior M10
approval is not sufficient. A future user-approved experiment requires an
M11-specific, separately owner-created approval file and an explicit opt-in.

**Purpose:** determine whether Plex's *own* M9-trained 600M weights can
become source-conditioned on a one-token contrast, rather than collapse
to a repeated character or learn only terminal EOS.

## Starting baseline and failure mode

The owner reported the real M11 M9 space-prompt generation outputs:
all 8/8 generated responses hit their 96-token cap, none had the
correct byte-exact completion or EOS stop; almost all outputs were
repeated 'r'. The original M9 checkpoint digest in those outputs is
e913be27855548a8f7dea766a2d367c7eaea357014ca82c80dec088958dfb123.

This is transcribed user-provided output, not a new independent rehash
of the original local report/checkpoint bytes in this PR.

The committed M11 baseline report and its 8 raw JSONL responses must
be independently source/hash-validated through existing admission
before any future training.

## Exact development goal

Learn to reproduce the two approved-for-consideration train examples:

    let score = 0;
    let score = 1;

and demonstrate transfer to the two visible probe-only examples
(which must never enter training):

    let value = 0;
    let value = 1;

The evaluator measures exact autonomous output and correct EOS under
both frozen decoding policies, plus bidirectional token preference
under matched gold-answer-prefix contexts. It records the difference
between memorized training-pair copying and development-only transfer.

This microprobe would not establish general JavaScript coding skill.
That needs a broader, independently held-out copying/editing benchmark.

## Experiment contract

- Parent checkpoint: independent SHA-verified original M9 model-only weights.
- Tokenizer: frozen original M7 tokenizer; no silent replacement.
- Training exposure: exactly 200 alternating single-example updates;
  two score cases only (100 exposures each); no probe-only exposures.
- Two separately approved arms, starting from the same original M9
  weights with a fresh AdamW optimizer:
  normal terminal-EOS loss weight 1.0 and reduced weight 0.25.
- Loss is computed from answer/EOS targets only. The denominator stays
  the number of valid supervised tokens, not the sum of EOS weights.
- Fixed optimizer: AdamW lr=5e-5, weight decay=0, eps=1e-8,
  foreach=False, fused=False, seed=41, batch size=1, accumulation=1.
- 4 persistent model-only snapshots at 25, 50, 100, 200 updates.
- Every checkpoint independently reloaded against the live model
  through the validated M10 parity helper, without moving the live
  optimizer's parameter objects off its device.
- Each checkpoint writes exactly 8 generated outputs and independent
  source-token preference measurements for both pairs.
- No optimizer-state resume, no checkpoint overwrite, no final project
  holdout, no automatic capability promotion.
- Training child is nonce-bound, parent-supervised, and hard-killed if
  exceeding 3600 seconds. A timeout/error must never be reported as
  a successfully completed training run.

## Operational gates

The worker refuses execution until all of these pass:

1. An explicit --execute-approved-run flag.
2. The original bundle, M9 report and model weights.
3. The distinct, M11 space-delimited four-case/two-decoder read-only
   baseline report and raw responses.
4. Owner-created, external M11 consent, bound to the exact model, data,
   source, arm, baseline hashes and update budget. It also binds SHA-256
   of the exact M11 worker, M10 checkpoint parity helper, and M8 generation
   decoder: changing any of these invalidates a previously reviewed consent.
   M10 consent is refused.
5. An unused output folder under gitignored training/artifacts, with
   the full predeclared free-space budget.
6. Clean and SHA-verified model, gate, curriculum and executable source.
7. Re-admission independently inside the worker subprocess.

No public command supplies consent implicitly; no code creates the
owner's approval file.

## CPU-only inspection now — NO GPU TRAINING

Run the plan without opening local checkpoints:

    uv run --project training --no-sync python -m plex_training.plex600_m11_worker

Run the test suite (synthetic fixtures only):

    uv run --project training --no-sync python -m pytest training/tests/test_plex600_m11_worker.py training/tests/test_plex600_m11_admission.py training/tests/test_plex600_m11_training_core.py -q

Run the existing M11 source and baseline audits separately. They
do not train or approve anything. The local M11 baseline report was
produced using PR #216's evaluator and remains a read-only artifact.

Do not issue an M11 approval now just to test this worker; CPU tests
use synthetic representations and mocked admission/child interfaces.

## If the owner later approves an actual run

First review exact source and the two input pairs and generate the
separate, unapproved M11 template using the read-only M11 admission
command. An owner may review it and, through an explicit external
process, supply a separately approved file for one arm. Repeat
the review and approval process for the other arm if justified.

Only after that should the owner invoke the worker entrypoint with
all seven explicit paths, the frozen M7 bundle manifest SHA, the chosen
EOS arm, and the execution flag. Neither this PR nor these
instructions authorize running either arm.

The worker never writes original M9 or M10 checkpoint files. A successful
run produces an immutable output directory containing an overall report
and model-only/response/diagnostic files for all four milestones.

An interrupted execution discards its incomplete staging directory
rather than accidentally promoting incomplete optimizer work; a later
research iteration may add separately reviewed interrupted-evidence
salvage if needed, without automatic optimizer-state resume.

## Promotion / stop criteria

For a narrow M11 success report, both decoder policies must show
exact output+EOS for both training variants and both development-only
variants, and bidirectional preference on both source pairs at two
consecutive measured milestones. The report will always say
generalCodingCapabilityProven=false even if those narrow gates pass.

If only the training pair succeeds, that supports memorization rather
than transferable source-conditioning. If the model again emits a
constant string, repeat or EOS-only answers, this is evidence for
redesigning the objective/data pipeline, not blind extra updates.

Next step after this PR: obtain explicit consent for a bounded run
only if the actual M11 baseline receipt passes local hash checks,
then evaluate the two arms on original weights, and plan broader
unseen code copying and editing tasks from those results.
