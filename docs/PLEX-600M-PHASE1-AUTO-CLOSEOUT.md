# Plex 600M — Phase 1 one-command local history registration

**Status:** implementation is available for review; the original files
must still pass the owner's Windows/WSL verification. No GPU, inference,
training or model modifications are performed by this command.

This procedure is the local-verification closeout of the first two
Phase 1 PRs (#213 checkpoint registry, #214 historical reconciler).
The third PR adds safe **exact baseline discovery** and **independent
re-verification of the published history files**.

## Why the command is shorter now

The M10 training report already contains the SHA-256 digests of the exact
full M9 read-only baseline report and all 36 generated responses.
The auto-registration script scans only completed M10 baseline folders,
calculates both file digests and **selects a baseline only if they match**
the recorded training-run hashes.

Missing or duplicate matches cause refusal. Short three-case test
baselines, mismatched prompt protocols, changed files, linked paths and
partial M10 run reports are not accepted as complete evidence.

The selected baseline then goes through the original full M9/ M10
source and checkpoint validators. Filename selection alone is not used
to declare an original experiment verified.

## Windows PowerShell / WSL setup

The following commands use the owner-run paths previously recorded in
the project. If the backup folder or run folder has moved, supply the
real path instead.

First, inspect the local git status before switching branches. Do not
reset or stash silently:

    wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- git status --short
    wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- git fetch origin

When local changes are preserved and a new local branch is safe,
check out the new Phase 1 branch:

    wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- git switch -c phase1-closeout --track origin/feature/plex600-phase1-auto-registration-verification

If this local branch already exists, switch to it and pull using
git pull --ff-only after checking for local modifications.

## One command to register actual local checkpoint evidence

Use this PowerShell command from the normal Windows terminal. It does
not authorize training or access the final holdout.

    wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_phase1_auto --register --m9-backup-dir /mnt/c/Users/quali/Plex600M-Backups/m9 --m10-run-dir training/artifacts/plex600-m10-approved-20261118-182055 --output-dir training/artifacts/plex600-phase1-history-20261119

The command automatically finds the *unique exact* M9 read-only
baseline referenced by the M10 run. It validates the actual M9 checkpoint,
source/tokenizer identities, all complete M10 checkpoints and their
generation receipts, and the full 18-by-2 evaluation.

On success it writes a **new** folder containing three JSON files:

- checkpoint-registry.json — rehashed local M9 and M10 checkpoints and
  their actual sequential parent chain.
- historical-comparisons.json — independently recomputed exact-copy +
  EOS outcomes under both decoders, all 36 before/after outputs for each
  available checkpoint, and train/evaluation split summaries.
- experiment-decisions.json — per-milestone promotion or rejection
  decisions and the M11 not-yet-evaluated gate.

On failure it returns a refusal rather than manufacturing the missing
evidence. If the output directory already exists, it refuses without
overwriting any existing checkpoint or report. Choose another unused
Phase 1 output directory only after inspecting the prior result.

The original checkpoint and training artifacts are never rewritten.

## Independently verify the published history again

After a successful registration, run the same source-bound audit and
compare its result against all three saved JSON files:

    wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_phase1_auto --verify --m9-backup-dir /mnt/c/Users/quali/Plex600M-Backups/m9 --m10-run-dir training/artifacts/plex600-m10-approved-20261118-182055 --output-dir training/artifacts/plex600-phase1-history-20261119

The successful report returns the number of registered checkpoints,
matched checkpoint comparisons, experiment decisions, and SHA-256 hashes
of the three published JSON files.

For debugging without writing anything, use --discover or --audit in
place of --register, with the same --m9-backup-dir and --m10-run-dir.
Do not supply --output-dir to read-only discovery or audit.

## Evidence and limitations

- The GitHub connector cannot read M9/M10 checkpoint binaries on the
  owner's Windows/WSL filesystem. PRs and CI cannot substitute for this
  local step. Do not report Phase 1 as byte-verified until the command
  succeeds against those files.
- The tool checks local hashes, frozen sources, optimizer update
  schedules, reported checkpoint lineage, and consistency of all raw
  responses with saved scores.
- Stored file evidence does not establish cryptographic provenance for
  when or where GPU training occurred, nor does it prove model quality.
- Keep the output JSON files local. They may include personal paths.
- No new M11 training approval or model checkpoint is generated.
- Phase 1 can be closed only when the original local verification succeeds
  and the recorded comparison is reviewed.

## Next development phase

Once Phase 1 passes, finish the M11 read-only M9 baseline evaluator for
the new space-delimited prompt protocol. Then build the bounded,
separately approved M11 paired source-conditioning worker, without
reusing old M10 training approval.
