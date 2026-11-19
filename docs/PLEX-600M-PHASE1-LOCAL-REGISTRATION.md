# Plex 600M — local historical checkpoint registration

This second Phase 1 increment provides a **read-only reconstruction**
of the original M9 and M10 evidence and a **create-only publication**
of results beneath the existing gitignored training/artifacts directory.
The code is on Phase 1 historical-reconciliation PR #214 (stacked on #213).

The ChatGPT GitHub connector can inspect committed code and narrative
reports, but cannot open checkpoint files residing on the owner's Windows
or WSL drive. Historical evidence registration is complete only if the
owner-local validation below succeeds. Do not fabricate a successful
registration from the committed Markdown reports.

## Validated input scope

The tool requires:

- A separate, original M9 backup with report.json and model-only weights.pt.
- The frozen M7 tokenizer bundle in training/artifacts/plex600-fresh-tokenizer.
- A completed M10 read-only M9 baseline with all eighteen prompts evaluated
  under both previously defined decoders and original newline instruction.
- The completed approved M10 run containing report.json, step-10,
  step-25 and step-50 model-only weights, report and raw responses.
  A 25-step early-stop run is accepted if it has a complete valid report.
- Optionally, the historical external M10 owner-approval JSON file to
  compare its actual SHA to the hash asserted by the run report.

The validator invokes the existing CPU-only M10 identity checker for
M9 weights/report, the original tokenizer, candidate source and the full
36-response baseline. It independently rehashes each M10 model-only
checkpoint, validates the exact optimizer update sequence, matches each
snapshot to its parent report and independently recomputes stored scores
from all thirty-six raw generated responses.

The M9 SHA pinned in the historical documentation is:

    e913be27855548a8f7dea766a2d367c7eaea357014ca82c80dec088958dfb123

A wrong M9 backup or changed training/evaluation source is refused. No new
GPU evaluation or training occurs. This is stronger than simply trusting a
report's self-reported pass/fail counts, but saved files alone cannot prove
that the historical GPU work actually occurred on the claimed device.

## One-time execution on the owner's WSL repo

Work from Windows PowerShell. Check the local working tree first:

    wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- git status --short

When local changes have been preserved and the working tree is clean,
fetch and inspect the PR #214 branch. Do not force reset or delete any
ignored training artifacts.

    wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- git fetch origin
    wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- git switch -c phase1-history --track origin/feature/plex600-phase1-historical-reconciliation

If that local branch already exists, switch to it without overwriting
local modifications and use git pull --ff-only after reviewing status.

Inspect baseline candidates without loading weights:

    wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_owner_approval --list-baselines

Choose the complete 18-case baseline folder explicitly. Do not select
an earlier three-case quick test. Multiple results require identifying
the exact baseline referenced by the M10 run report.

The intended WSL-local inputs, based on prior owner-run commands, are:

    /mnt/c/Users/quali/Plex600M-Backups/m9/report.json
    /mnt/c/Users/quali/Plex600M-Backups/m9/weights.pt
    training/artifacts/plex600-fresh-tokenizer
    training/artifacts/plex600-m10-m9-full-baseline-[actual timestamp]
    training/artifacts/plex600-m10-approved-20261118-182055

The exact baseline timestamp must be discovered, not invented.

The complete command below is deliberately read-only (replace the
baseline placeholder only after verifying that actual directory exists):

    wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_phase1_historical --audit --bundle-dir training/artifacts/plex600-fresh-tokenizer --m9-report /mnt/c/Users/quali/Plex600M-Backups/m9/report.json --m9-weights /mnt/c/Users/quali/Plex600M-Backups/m9/weights.pt --baseline-dir training/artifacts/REPLACE_WITH_ACTUAL_COMPLETE_BASELINE --m10-run-dir training/artifacts/plex600-m10-approved-20261118-182055

After a successful audit, use the same arguments with --register instead
of --audit, and append a **new** output folder directly under training/artifacts,
for example:

    --output-dir training/artifacts/plex600-phase1-history-20261119

If the output folder already exists, registration refuses rather than
overwriting any previous evidence. Historical M9/M10 artifact directories
are never altered by the command.

Optional historical approval check: append

    --approval-file /mnt/c/Users/quali/Plex600M-Approvals/m10-owner-approval.json

only when that exact owner-approval file actually exists. Otherwise the
generated report will say the M10 approval hash is historical reported
metadata, not an independently rehashed external approval file.

## Local generated artifacts

A successful registration creates three JSON files in the new folder:

1. checkpoint-registry.json — original M9 and M10 step-10/25/50
   checkpoint paths, byte counts, SHA-256, frozen candidate/optimizer
   source identities and exact parent checkpoint chain.
2. historical-comparisons.json — M9 read-only baseline versus each
   matched M10 snapshot, with independent raw-response recomputation for
   both decoders and separate 12-train/6-evaluation-only scores.
3. experiment-decisions.json — an explicit no-promotion/manual-review
   decision for each milestone, plus an M11 not-yet-evaluated entry.

The status means *local file-hash and receipt-consistency verified*,
not a cryptographic proof of historical execution origin or guaranteed
generalization to unseen code.

The final project holdout is not accessed. The file paths in the locally
generated registry may be personal; keep this directory private, and do
not upload it without reviewing what paths it reveals.

## Completion gate

Phase 1 is locally complete only when the actual source files pass the
registration, all complete checkpoints are verified, and the generated
comparisons contain all source-matched cases. If any file is absent,
inconsistent or truncated, the command refuses and the gate stays open.
Check the original source rather than synthesizing replacement data.

For documented but **not yet newly rehashed** historical findings, use:
[Historical decisions](PLEX-600M-PHASE1-HISTORICAL-DECISIONS.md).
