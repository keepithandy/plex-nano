# Plex 600M M10 — explicit local owner approval and launch

**Protocol:** 12 training examples, 6 evaluation-only heldout examples,
50 optimizer updates maximum (10 JS-only + 15 mixed + 25 mixed), one
example per update, 60-minute hard timeout, immutable independent M9
checkpoint and tokenizer. The owner approved this experiment scope in
the November 18, 2026 conversation. That chat approval is **not** an
external JSON file and the trainer will still refuse to run until a
separate file has been explicitly created *outside this repository*
using the owner-operated command below.

No training has been started or queued from GitHub. The user must run
the local commands on their Windows 11 / WSL Ubuntu workstation. None
of the commands below uses a Windows unsigned .ps1 script.

## 1. Switch safely to the reviewed training branch

In **PowerShell**, first check for local changes and stop if the result
is nonempty:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- git status --short
~~~

When clean, fetch and switch to the exact PR #208 branch:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- git fetch origin
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- git switch --track origin/feature/plex600-m10-bounded-training-worker
~~~

If the local branch already exists, use this second command instead:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- git switch feature/plex600-m10-bounded-training-worker
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- git pull --ff-only origin feature/plex600-m10-bounded-training-worker
~~~

Do not use hard reset or force switch. Existing ignored artifacts must
not be deleted. Preserve the separately backed up M9 checkpoint.

## 2. Locate the full completed 18×2 baseline automatically

The helper lists complete-size **candidate paths** but does not
certify them until the next source audit. It refuses to infer from the
short three-example run.

~~~powershell
$Repo = "/home/quali/plex-nano"
$UV = "/home/quali/.local/bin/uv"
$Located = & wsl.exe -d Ubuntu --cd $Repo -- $UV run --project training --no-sync python -m plex_training.plex600_m10_owner_approval --list-baselines
if ($LASTEXITCODE -ne 0) { throw "M10 baseline discovery failed." }
$Baselines = @(($Located | ConvertFrom-Json).baselines)
if ($Baselines.Count -ne 1) {
    $Baselines | Format-List
    throw "Expected exactly one full-baseline candidate; choose explicitly if several exist."
}
$Baseline = $Baselines[0]
$Baseline | Format-List
~~~

If no path appears, find the actual completed M9 baseline directory.
Do **not** substitute the three-example result or rerun M9 without
reviewing why the saved baseline is missing.

## 3. Inspect and bind exact sources

Copy this once; keep these PowerShell variables in the same session:

~~~powershell
$Scope = @(
  "--bundle-dir", "training/artifacts/plex600-fresh-tokenizer",
  "--bundle-manifest-sha256", "14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55",
  "--m9-report", "/mnt/c/Users/quali/Plex600M-Backups/m9/report.json",
  "--m9-weights", "/mnt/c/Users/quali/Plex600M-Backups/m9/weights.pt",
  "--baseline-report", $Baseline.baselineReport,
  "--baseline-responses", $Baseline.baselineResponses
)
& wsl.exe -d Ubuntu --cd $Repo -- $UV run --project training --no-sync python -m plex_training.plex600_m10_owner_approval --review-exact-scope @Scope
if ($LASTEXITCODE -ne 0) { throw "M10 exact source and baseline verification failed." }
~~~

The reviewer checks the M9 source, all 18 proposed records (12 training
and 6 evaluation-only), tokenizer round trips, both raw baseline
decoders, and the approved schedule SHA. It prints the 18 records and
their digests but **does not approve or write anything**.

## 4. Record explicit local consent outside GitHub

Only if the printed scope is the same one you authorized, create a
private external approval. The confirmation string is deliberately
specific to this exact 12/6/50/3600 protocol.

~~~powershell
New-Item -ItemType Directory -Force "$HOME\Plex600M-Approvals" | Out-Null
$Approval = "/mnt/c/Users/quali/Plex600M-Approvals/m10-owner-approval.json"
& wsl.exe -d Ubuntu --cd $Repo -- $UV run --project training --no-sync python -m plex_training.plex600_m10_owner_approval --record-owner-approval @Scope --approval-file $Approval --confirm-owner-review "APPROVE M10 12 TRAIN 6 HELDOUT 50 UPDATES 3600 SECONDS"
if ($LASTEXITCODE -ne 0) { throw "No new M10 consent recorded. Do not train." }
~~~

The helper uses exclusive create (no overwrite); the recorded owner
statement is **not** cryptographically signed. It is generated only
when the owner invokes that explicit command. The trainer never
creates its own consent file. If the file already exists, do not
overwrite it; instead independently validate the existing file or
investigate before proceeding.

## 5. Start the approved bounded M10 GPU experiment

This is the **only step that trains**. It requires the verified local
RTX 4080 SUPER with native BF16, at least 14,000 MiB free VRAM and
13 GiB of free disk; it still refuses if any source, authorization,
output, hardware or schedule check fails.

~~~powershell
$Out = "training/artifacts/plex600-m10-approved-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
& wsl.exe -d Ubuntu --cd $Repo -- $UV run --project training --no-sync python -m plex_training.plex600_m10_worker --execute-approved-run @Scope --approval-file $Approval --output-dir $Out
if ($LASTEXITCODE -ne 0) {
    Write-Host "M10 refused or interrupted. Inspect the emitted error. Do not auto-retry."
}
~~~

No optimization is permitted beyond 50 steps or 3,600 seconds. The
subprocess is forcibly terminated at the wall-clock deadline.
Snapshots are model-only and create-only at updates 10, 25, 50;
the original M9/M7 files are never overwritten.

A complete result includes a report and checkpoint-bound evidence for
each reached milestone. An interrupted run may preserve **partial,
SHA-verified and independently rescored** milestone results, clearly
labeled as incomplete. Its saved optimizer state is absent and resuming
from those partial snapshots is **not** authorized. Read the output
JSON before claiming any learning.

## What the experiment can prove

The single important near-term test is whether Plex can autonomously
reproduce the warmup code line, newline and EOS after update 10. We
measure exactness, EOS, token ranks and character similarity relative
to the completed M9 baseline, under **both** decoding policies.

The full 18×2 baseline took roughly 13 minutes; there are three
milestone evaluations, training, and model reload parity under the
fixed 60-minute cap. The run may time out. A failed or partial result
is evidence, not proof of competence.

**No GitHub Actions run can train on your local RTX 4080 SUPER.** The
GitHub CI checks are CPU unit tests only.
