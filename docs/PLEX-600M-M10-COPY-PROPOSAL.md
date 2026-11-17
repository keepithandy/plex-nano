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
| Maximum optimizer updates | **50** — hard upper bound, not a success guarantee |
| Evaluation snapshots | **10, 25, 50** optimizer updates |
| Evaluation data | All 12 train IDs plus 6 disjoint heldout IDs |
| Prompt/answer positions | At most 256 / 96 tokens including EOS |
| Selection / loss | Source-conditioned exact copy, answer-and-EOS-only mask |
| GPU | Existing RTX 4080 SUPER, native BF16; preflight only after approval |
| Stop budget | 60 minutes maximum wall clock proposed; refuse if storage cannot preserve snapshots |
| Checkpoints | New create-only directories with hashes and reports; M7/M9 immutable |
| Training authorization | **Not granted** by this plan, candidates, or previous M9 approval |

The experimental worker, reproducible milestone sampler, CUDA memory budget,
checkpoint save rules, exact M9 source hash binding, and GPU execution gates
must be implemented and reviewed **after** explicit owner approval of the
candidate texts and experiment limits. Until then, these are design
constraints, not claims of an existing runnable trainer.

## Scoring and stopping

At each milestone, run an **autonomous** generation test with identical
prompts and decoding settings, separate from any teacher-forced loss report.
Compare generated raw text byte-for-byte with the reference; report exact
match **and EOS** for each case, plus whether output collapses to repeated
whitespace/punctuation. Save the decoder policy and token budget.

A single perfect training set is not enough. Early-stop only when **both
training (12/12) and heldout (6/6)** copies are exactly correct and end in
EOS at **two consecutive evaluation milestones** (earliest stop: update 25).
Otherwise continue only within the approved 50-update budget. A 6-case
heldout microprobe is too small to establish broad generalization, even if
perfect.

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
digest. The separate `--approval-template` option prints an **unapproved**
template without writing an approval file:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_copy_proposal --approval-template
~~~

After reviewing the candidate, the frozen-tokenizer audit can be invoked
without training:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_copy_proposal --audit-tokenizer --bundle-dir training/artifacts/plex600-fresh-tokenizer --bundle-manifest-sha256 14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55
~~~

The review module has *no* `--execute-approved-run` mode, no CUDA imports,
no training updates and no ability to approve its own candidate.

## Approval gate and future handoff

Request separate, explicit user approval of: the **exact candidate SHA and
all 18 snippets**, source rights, train/heldout split, verified **M9 backup
weight SHA**, 50-update maximum, 10/25/50 checkpoints, optimizer parameters,
time/storage envelope, and exact-copy stopping gate. Do not borrow an M7/M9
approval.

Once authorized, design and test the *new* bounded runner in its own PR.
It must fail closed if the checkpoint or candidate SHA changes, a train row
appears in heldout, a required milestone is skipped, a destination exists,
or the time/VRAM/storage cap cannot be met. Run GPU work only on the user's
machine after the separate approval.

M10 answers a narrow research question. If generation still collapses,
investigate copy-target mask alignment, model logits, tokenizer-ID validity,
EOS rank and prompt sensitivity before proposing a longer run.
