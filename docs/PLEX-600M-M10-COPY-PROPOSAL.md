# Plex 600M — M10 exact-copy microprobe (REVIEW ONLY)

**Status: candidate proposal. Training is not authorized, and there is no GPU
training entry point in this PR.** No checkpoint is read or modified by the
default commands. The 18 code snippets were proposed by the assistant; they
are **not** represented as owner-authored or owner-approved material.

## Question

Can the existing M9 600M-parameter checkpoint learn to *autonomously copy a
short supplied snippet*, including whitespace and EOS, rather than generating
96 repeated periods or spaces? M7 and M9 scored zero successful generated
edits on the 36-case development benchmark. Teacher-forced validation
loss is not the same as unconditioned/causal generation quality.

## Candidate data and verification

- Review the exact 18 rows in
  [M10 copying candidates](../training/phase3/candidates/plex600-m10-copy-proposal-v1.jsonl):
  12 proposed training snippets and 6 **never trained** heldout snippets.
- Languages: HTML, CSS, JavaScript; each has 4 train and 2 heldout records.
- Every snippet is short, original assistant-proposed candidate text, with
  explicit `trainingAuthorized: false` and review-only provenance.
- Train/heldout snippets and families are unique and disjoint. The audit also
  checks for exact file overlap with the **M8 development** dataset; it does
  not access the final holdout.
- The proposed prompt includes the snippet in context, and the target is the
  **same complete snippet plus EOS**. Only copy/EOS positions receive
  supervised loss; the instruction and input snippet are masked.
- Audit the frozen, source-bound original M7 tokenizer (vocabulary size 1191),
  exact prompt/snippet round trip, 256-position input cap and 96-token
  copy+EOS generation cap. Do not silently truncate.
- In any future approved evaluation, report both the original 32,768-token
  unmasked greedy policy and the fitted-vocabulary decoding policy as
  separate experimental conditions. Do not change the model architecture,
  tokenizer mapping, weights, or logit dimensions in this proposal.

## Proposed training protocol — NOT APPROVED

| Component | Proposed bound |
| --- | --- |
| Starting checkpoint | Independently SHA-verified **backed-up M9 weights** (not M7) |
| New optimizer | Fresh AdamW state; not an optimizer-state resume |
| Optimizer hyperparameters | Proposed: lr 5e-5, weight decay 0, eps 1e-8, no fused/foreach optimizers |
| Seed | 41; deterministic round-robin over approved training IDs |
| Maximum optimizer updates | **50 total**, including the tiny-example phase — hard upper bound, not a success guarantee |
| Examples per optimizer update | **1 source-conditioned example**; one forward/backward pass, gradient accumulation **1**; exactly one optimizer step |
| Phase A: one-example copy | **Updates 1–10** repeat `m10-js-counter`, a single proposed JS snippet, 10 times |
| Phase B: full-set learning | **Updates 11–25** round-robin through all 12 training examples (15 updates) |
| Phase C: full-set consolidation | **Updates 26–50** continue the same round-robin (25 updates) |
| Evaluation snapshots | Read-only M9 **baseline at update 0**, then **10, 25, 50** optimizer updates |
| Evaluation data | All 12 train IDs plus 6 disjoint heldout IDs |
| Prompt/answer positions | At most 256 / 96 tokens including EOS |
| Selection / loss | Source-conditioned exact copy, answer-and-EOS-only mask |
| GPU | Existing RTX 4080 SUPER, native BF16; preflight only after approval |
| Stop budget | 60 minutes maximum wall clock proposed; refuse if storage cannot preserve snapshots |
| Checkpoints | New create-only directories with hashes and reports; M7/M9 immutable |
| Training authorization | **Not granted** by this plan, candidates, or previous M9 approval |

The deterministic, **CPU-only schedule** is now implemented for review:
updates 1–10 show one ID, and updates 11–50 cycle through the 12 train IDs
without resetting at update 26. Total proposed exposures are 50, and all six
heldout IDs have **zero training exposures**. The full-set segment contains
40 training examples across 40 optimizer steps: three complete cycles of
12 plus four additional exposures. The warmup snippet receives 13 total
exposures (10 warmup + 3 full-set); no individual update processes more than
one example. These counts are proposals, **not completed training**.

The executable trainer, CUDA memory budget, checkpoint save rules, exact
M9 source hash binding, and GPU execution/approval gates still require a
separate implementation and review. The proposed schedule does not grant
training authorization. All planned optimizer updates are within the 50-step
ceiling even if any evaluation milestone fails.

## Scoring and stopping

At each milestone, run an **autonomous** generation test with identical
prompts and decoding settings, separate from any teacher-forced loss report.
Score all **12 training and 6 heldout** examples even at the 10-update
warmup milestone, so the measurements have a comparable denominator.
Use the completed M9 baseline report and responses as the update-zero measurements. Record their paths and hashes and match checkpoint, case IDs, prompts, decoder policies and generation cap before comparing milestones. If a field differs, flag that comparison for review. Preparation requires no repeat inference.

The single-example CPU regression exercises the shared causal loss and model on synthetic token IDs. It checks that the prompt logits receive zero loss gradient, each answer position including EOS receives a gradient, the embedding weights change after one optimizer step, and a fresh model loaded from disk reproduces the logits and token choices. Actual M10 measurements belong to the future approved run.

Report per-example byte-for-byte exactness, EOS termination, normalized
character edit similarity (Levenshtein), correct prefix fraction, number of
whitespace-only / spaces-only / periods-only responses, longest repeated
character run, and fraction of output dominated by one character. Report
train and heldout means separately and the number of distinct generated
outputs. Keep decoder policy, reference IDs, token budget, checkpoint SHA,
and output bytes bound to every result. The current CPU-only scorer accepts
**caller-supplied responses** and marks its results unverified; it does not
claim an M9/M10 inference evaluation has occurred.

A single perfect training set is not enough. Early-stop only when **both
training (12/12) and heldout (6/6)** copies are exactly correct and end in
EOS at **two consecutive evaluation milestones** (earliest stop: update 25).
Otherwise continue only within the approved 50-update budget. A 6-case
heldout microprobe is too small to establish broad generalization, even if
perfect.

### Mechanical integrity and bounded decisions

Before interpreting a loss decrease as learning, the future authorized
training worker must independently measure and record, at each milestone:

1. **Supervision:** exactly `prompt[:-1]` has ignore targets (`-100`);
   targets from the last prompt position through code answer and terminal
   EOS are the only supervised positions. Record the non-ignored prompt
   count (must be zero), supervised answer positions, EOS target presence,
   and one input example per optimizer step. The existing CPU-only audit
   now checks this exact causal alignment.
2. **Gradient and weights:** record nonzero finite gradient L2 norm and
   nonzero model weight delta after the optimizer step. Compare the new
   checkpoint SHA against the starting M9 SHA, with no optimizer-state
   resume. A changing SHA by itself does not prove meaningful learning.
3. **Save/reload reproducibility:** hash the newly written create-only
   checkpoint, load it strictly into the identical model architecture,
   and verify that deterministic greedy token IDs match before and after
   reload. Require max absolute float-logit difference <= `1e-4` under
   the same inference environment and decoding settings. Record all checks
   and fail the milestone if save/reload is inconsistent.

The CPU-only `classify_mechanical_evidence` function provides a strict
schema and fault labels for externally supplied evidence; its
`evidenceVerifiedByThisModule: false` flag must **not** be upgraded to proof
of an actual M10 training run. The future GPU worker has to collect and bind
the measurements to weights, record IDs and optimizer updates itself.

**Decision rules:** If mask/gradient/weight/reload checks fail, stop and
diagnose the mechanism before spending further GPU time. If mechanics pass
but the update-10 exemplar remains poor, record exact, partial-copy and
repetition changes against M9 and continue only within the original cap;
do not label the model unable to copy after ten steps. If training copies
improve but the six heldout cases do not, label likely memorization rather
than generalization. If neither exactness nor partial-copy metrics improve
through update 50 despite demonstrable weight updates, diagnose training
signal, decoding and prompt conditioning before proposing more training.
A failed 50-update attempt is a valid research result, not authorization
to extend the budget.

Before approving this stage, verify the separate backup contains M9 report
and weights with matching hashes. M9's source checkpoint remains immutable;
M10 receives its own experimental output root. There is no license to run
the final project holdout or extend the cap if M10 fails.

## Current implementation: CPU-only review

From the local repository, with the existing training environment:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_copy_proposal --show-candidate
~~~

This displays every proposed candidate, split, and the immutable candidate
digest. The `--show-schedule` option lists all **50 proposed single-example
optimizer updates** (review only; none are executed):

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_copy_proposal --show-schedule
~~~

The separate `--approval-template` option prints an **unapproved**
template without writing an approval file:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_copy_proposal --approval-template
~~~

After reviewing the candidate, the frozen-tokenizer audit can be invoked
without training:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_copy_proposal --audit-tokenizer --bundle-dir training/artifacts/plex600-fresh-tokenizer --bundle-manifest-sha256 14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55
~~~

The M10 proposal now includes a **read-only M9 backup admission check**.
It reads and hashes an independently copied M9 `report.json` and
`weights.pt` (up to 3 GiB), confirms the eight-update protocol,
approved-source/checkpoint provenance, and refuses a path inside the original
M9 experiment output directory. The historical completed M9 report can
incorrectly say `checkpointWrites: false`; admission checks the actual
saved weight file byte count and SHA-256 instead. It does not unpickle or
deserialize PyTorch weights.

If an independent copy was made with the earlier `plex backup m9`
command, its **example** location is:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_copy_proposal --audit-m9-backup --m9-report /mnt/c/Users/quali/Plex600M-Backups/m9/report.json --m9-weights /mnt/c/Users/quali/Plex600M-Backups/m9/weights.pt
~~~

Adjust those two paths only after confirming where the M9 backup actually
exists. This check will **not** create a backup, authorize training, or
modify model files. The M9 checkpoint SHA reported by this command becomes
an exact input to the future separately approved M10 runner.

To avoid a false early-stop claim, the scoring contract now requires
complete train/heldout exact-copy **and EOS** evidence at two *distinct,
consecutive* checkpoints (10→25 or 25→50), with the *same* decoder
policy and candidate SHA. Caller-supplied predictions remain
**unverified external claims** until a future evaluator binds generated
outputs to a SHA-verified model and prompt run.

The review module has *no* `--execute-approved-run` mode, no CUDA imports,
no training updates and no ability to approve its own candidate.

## Approval gate and future handoff

Request separate, explicit user approval of: the **exact candidate SHA and
all 18 snippets**, source rights, train/heldout split, verified **M9 backup
weight SHA**, **one example per update**, exact **10+15+25 phase schedule**,
50-update maximum, update-zero baseline, 10/25/50 checkpoints, optimizer
parameters, time/storage envelope, mechanical acceptance criteria, and
exact-copy stopping gate. Do not borrow an M7/M9
approval.

Once authorized, design and test the *new* bounded runner in its own PR.
It must fail closed if the checkpoint or candidate SHA changes, a train row
appears in heldout, a required milestone is skipped, a destination exists,
or the time/VRAM/storage cap cannot be met. Run GPU work only on the user's
machine after the separate approval.

M10 answers a narrow research question. If generation still collapses,
investigate copy-target mask alignment, model logits, tokenizer-ID validity,
EOS rank and prompt sensitivity before proposing a longer run.
