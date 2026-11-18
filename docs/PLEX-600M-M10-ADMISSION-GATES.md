# M10 — CPU-only admission and exact owner approval contract

**This PR does not authorize, implement, or execute M10 GPU training.**
It is a *separate review gate* layered on [M10's unapproved copying
proposal](PLEX-600M-M10-COPY-PROPOSAL.md) (PR #205).

The completed M9 baseline, provided by the owner, found **zero exact
copies and zero EOS terminations on all 18 cases under either decoding
policy**. It provides a meaningful comparison at update zero; that
does not establish that more training will fix generation.

## Why another gate

Passing the small CPU Plex forward/backward/save/reload regression in
#205 establishes that a *synthetic tiny configuration* has correct causal
supervision and can update weights. It does not prove that the owner's
600M checkpoint is trainable under the 50-step ceiling.

Before implementing any experimental GPU runner, this PR creates a
strict read-only admission proof that binds:

- **Source and model:** independently backed-up M9 report and
  2.4 GB checkpoint hashes; frozen M7 tokenizer and source manifest;
  the exact SHA-256 of all 18 quarantined assistant-proposed snippets.
- **Read-only baseline:** complete 18-case × 2-decoder report and
  JSONL responses under a verified SHA-256. Every saved response must
  match its case ID, original M9 weight SHA, prompt SHA, split,
  decoder name and stated token cap. Split totals, exact matches,
  EOS and character similarities are recomputed from raw responses.
  The score checks allow 0.000002 rounding tolerance for aggregated
  character similarity and exact agreement for discrete counts.
  The baseline receipt is internally consistent evidence; it is
  **not cryptographic proof of where GPU operations ran**.
- **Independent consent:** one *external*, small, explicit owner
  approval file with the exact proposed source, split IDs,
  checkpoint and baseline hashes, assistant-origin acknowledgment,
  10+15+25 schedule, one sample per optimizer step, 50-step total,
  decoding policies, seed, AdamW parameters and time/storage limits.
  Existing M9 approval, PR merge, a passing CI build, and the unapproved
  JSONL do **not** count as owner consent.
- **No overwrite and storage:** a direct new output folder under
  \`training/artifacts\`, checked *before* reading source/approval, and
  at least 13 GiB of unallocated disk free (3 bounded snapshots,
  1 bounded in-flight checkpoint, 1 GiB margin). Insufficient storage
  refuses admission rather than risking a partial run.
- **No heldout training:** only the proposed 12 training IDs may
  appear in the 50-update schedule, and the single JS line
  \`let score = 0;\\n\` is the only example used during updates 1–10.
  The six heldout cases are evaluated only.

## CPU-only commands (no .ps1 execution-policy changes)

The standard command is safe, with no file or model loaded:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_admission
~~~

To display the **unapproved** template, you must supply the *actual*
completed baseline folder. It cannot be inferred from the pasted console
JSON. Use the exact local paths after checking them:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_admission --show-approval-template --bundle-dir training/artifacts/plex600-fresh-tokenizer --bundle-manifest-sha256 14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55 --m9-report /mnt/c/Users/quali/Plex600M-Backups/m9/report.json --m9-weights /mnt/c/Users/quali/Plex600M-Backups/m9/weights.pt --baseline-report /exact/path/to/completed-full-baseline/report.json --baseline-responses /exact/path/to/completed-full-baseline/responses.jsonl
~~~

The template prints with **\`approvedForTraining: false\`**, no
signature/date and no file creation. It is not authorization.

The distinct \`--audit-admission\` mode additionally requires
\`--approval-file\` and \`--output-dir\`. It remains CPU-only even
after authentic local consent, and reports
\`runEntrypointImplemented: false\`,
\`optimizerUpdatesCompleted: 0\` and \`checkpointWrites: false\`.
The admission check does not write a training approval file or
checkpoint.

## Before a real M10 trainer exists

A separate, reviewed *worker implementation* must still:

1. Revalidate every approval and hash in a child process with a
   hard wall-clock kill—not a cooperative timeout.
2. Check 4080 SUPER/native BF16, available GPU RAM and data masking,
   including nonzero gradients and actual parameter changes.
3. Run only one example/step in the exact 10+15+25 schedule and
   write create-only model-only snapshots at optimizer steps 10, 25
   and 50, with SHA-256/size verification and independent reload
   equivalence. Never overwrite M9/M7 originals.
4. Evaluate autonomous copying at milestones under **both**
   unmasked and fitted-vocabulary policies, compare against the
   already completed M9 baseline and enforce the exact early-stop
   contract without inspecting the final project holdout.
5. Refuse training when actual GPU memory, disk, or deadline
   cannot satisfy the accepted bounds; the previously measured
   18×2 read-only baseline required about 13 minutes by itself.

**No new optimizer update has occurred.** These controls are
preparation for an experiment whose exact text, provenance and
limits still require explicit owner approval.
