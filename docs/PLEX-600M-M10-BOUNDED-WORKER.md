# Plex 600M M10 — bounded copying worker (DRAFT / NOT AUTHORIZED)

This branch implements the missing, explicitly opt-in GPU training worker,
stacked on PR #207 admission and PR #205's 10+15+25 design. It does not
grant owner permission. The 12 proposed train and six heldout examples
remain assistant-proposed and unapproved for training.

**No M10 GPU training, no optimizer update on the actual 600M model,
and no new 600M checkpoint have occurred in this PR.** GitHub CI is CPU-only.

## Narrow research question

Can a verified M9 600M model learn to autonomously reproduce the exact
JavaScript line *let score = 0;* including the final newline and EOS?
M9 copied 0/18 examples and emitted EOS 0/18 under both decoding policies.

## Proposed bounded execution

Fifty optimizer updates maximum, one example per step and no gradient
accumulation. Fresh AdamW state, learning rate 5e-5, epsilon 1e-8,
weight decay 0, seed 41. Steps 1–10 repeat only the JS counter example.
Steps 11–50 cycle the 12 proposed train IDs. All six heldout IDs
have zero training exposure.

At updates 10, 25 and 50, the worker writes model-only, SHA-256 verified,
create-only checkpoint snapshots. Each is loaded strictly into a fresh
CPU model for greedy-token and maximum 1e-4 logit-parity checks.
The live model and its AdamW Parameter identities remain on the original
training device: a separate CPU model receives a tensor-by-tensor snapshot,
then a new CPU model reloads the saved weights for FP32 parity. A
regression test checks that optimizer state can advance after snapshotting.
The parent independently re-scores every recorded 18-case × 2-policy
prediction from the raw checkpoint-bound responses, rather than merely
trusting the scores reported by the child.

After every milestone it generates on all 18 candidates under both
existing decoding policies. It stores raw prediction text, EOS/stop
reason, case and prompt SHAs, and source checkpoint SHA. It reports
exact/partial copying, repetition and EOS, plus warmup comparison to
M9 and teacher-forced EOS/first-answer ranks. Teacher-forced NLL
improvement is not labeled autonomous copying.

Early stop requires two **consecutive** milestones with all
12 train and six heldout examples copied exactly, including EOS,
under **both** decoding policies. No skipped evaluation is permitted.

## Safety boundary

- The default module command only prints a no-training plan.
- Actual execution requires an explicit run flag, external exact-scope
  owner approval, all independent M9 baseline and checkpoint paths,
  frozen M7 tokenizer/source manifests, approved candidate SHA,
  entirely new direct output directory and at least 13 GiB disk free.
- Parent and child recheck the admission guard **before importing torch**
  or allocating GPU memory.
- The required hardware is the existing RTX 4080 SUPER with native
  BF16 support and >=14,000 MiB available VRAM. Architecture and
  tokenizer are unchanged. Historic checkpoints are read-only.
- A supervised child is hard-killed after 3,600 seconds. This hard
  cap is distinct from cooperative checks between updates.
- New checkpoint snapshots are model-only; optimizer state is never
  written or resumed.
- On interruption, only complete, SHA-verified and independently
  re-scored partial milestones may be preserved. Any unfinished future
  snapshot is discarded before publishing the clearly labeled partial
  artifact. It is not a successful training result or authorization
  to resume.
- No final project holdout, automatic retraining, extra steps or
  self-created approval files are allowed.

## Safe read-only plan

Run in PowerShell without changing Windows execution policy:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_worker
~~~

The real execution path is **blocked until separate exact-source owner
approval** and final GPU implementation review. Its explicit required
inputs are bundle directory/manifest SHA, backed-up M9 report/weights,
completed 18-case/two-decoder baseline report/responses, owner approval
file and new output path. An M9 approval or simply merging this PR
does not count as an M10 approval.

## Remaining risk

The 18-by-two M9 baseline took approximately 13 minutes. Repeating
that at three M10 milestones plus 50 training updates and 3 checkpoint
save/reloads might exceed the immutable 60-minute cap. An interrupted
run is not success, and increasing the budget requires separate review.

Linux/Windows CI tests cannot validate 600M full-GPU memory capacity,
actual wall time or learning outcome. Do not claim that Plex learned to
copy the one-line example until approved inference evidence exists.
