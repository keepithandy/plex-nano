#requires -Version 5.1
<#
.SYNOPSIS
  Run the bounded Plex 600M experiments from Windows PowerShell, through WSL2.
.DESCRIPTION
  Does not run Windows PyTorch, change experiment code, or implicitly approve
  training. The existing M7/M9 Python gates remain authoritative.
  Defaults match the owner's current WSL installation; override with
  PLEX_WSL_DISTRO, PLEX_WSL_PROJECT, PLEX_WSL_UV if necessary.
.EXAMPLE
  .\scripts\Plex-Control.ps1 review m9
  .\scripts\Plex-Control.ps1 approve m9
  .\scripts\Plex-Control.ps1 train m9
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet('help', 'status', 'review', 'approve', 'train', 'backup', 'evaluate')]
  [string]$Action = 'help',
  [Parameter(Position = 1)]
  [ValidateSet('m7', 'm9')]
  [string]$Run = 'm9'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Distro = if ($env:PLEX_WSL_DISTRO) { $env:PLEX_WSL_DISTRO } else { 'Ubuntu' }
$Project = if ($env:PLEX_WSL_PROJECT) { $env:PLEX_WSL_PROJECT } else { '/home/quali/plex-nano' }
$Uv = if ($env:PLEX_WSL_UV) { $env:PLEX_WSL_UV } else { '/home/quali/.local/bin/uv' }
$Bundle = 'training/artifacts/plex600-fresh-tokenizer'
$Manifest = '14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55'
$M7 = 'training/artifacts/plex600-m7-fresh-bootstrap'
$M9 = 'training/artifacts/plex600-m9-r3-first-eight'
$Approval = 'training/artifacts/approvals/plex600-m9-r3-approved.json'

function Invoke-PlexUv {
  param([Parameter(Mandatory = $true)][string[]]$Arguments)
  & wsl.exe -d $Distro --cd $Project -- $Uv @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Plex command failed with exit code $LASTEXITCODE. No success is assumed."
  }
}

function Invoke-PlexPython {
  param(
    [Parameter(Mandatory = $true)][string]$Code,
    [string[]]$Arguments = @()
  )
  # Standard input is a Python script, not an interactive shell, so neither
  # Windows PowerShell quoting nor Linux shell expansion can alter code values.
  $Code | & wsl.exe -d $Distro --cd $Project -- $Uv run --project training --no-sync python - @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Plex Python command failed with exit code $LASTEXITCODE."
  }
}

switch ($Action) {
  'help' {
    @'
Plex 600M — Windows PowerShell launcher (WSL2 runs CUDA)

  plex status           Show local experiment status and GPU memory
  plex review m9        Display the exact 8 train / 3 validation edits
  plex approve m9       Explicitly approve those exact edits (create-only)
  plex train m9         Run the existing bounded M9 GPU runner
  plex backup m9        Verify and copy M9 weights and report to Windows
  plex backup m7        Verify and copy M7 weights and report to Windows
  plex evaluate m7      Generate M8 development outputs from the verified M7
  plex help             Show these commands

Approval is never inferred from a passing preflight. No M9 evaluation shortcut
exists yet; M8's current evaluator requires M7 checkpoint identity. Outputs
are create-only. Neither M7 nor M9 weights are overwritten.
'@
    break
  }
  'status' {
    $code = @'
from pathlib import Path
import json

root = Path("training/artifacts")
for label, name in (
    ("M7", "plex600-m7-fresh-bootstrap"),
    ("M9", "plex600-m9-r3-first-eight"),
):
    folder = root / name
    report = folder / "report.json"
    weights = folder / "weights.pt"
    if not report.is_file():
        print(f"{label}: no completed report at {folder}")
        continue
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        print(f"{label}: status={data.get('status')}, updates={data.get('optimizerUpdatesCompleted')}, weightsPresent={weights.is_file()}")
        print(f"    checkpoint SHA-256 (reported, not rehashed): {data.get('checkpoint', {}).get('sha256')}")
    except (OSError, ValueError, TypeError) as exc:
        print(f"{label}: report unreadable: {exc}")

approval = root / "approvals/plex600-m9-r3-approved.json"
if approval.is_file():
    try:
        item = json.loads(approval.read_text(encoding="utf-8"))
        print(f"M9 owner approval file: present, approvedForTraining={item.get('approvedForTraining')}")
    except (OSError, ValueError, TypeError):
        print("M9 owner approval file: invalid JSON")
else:
    print("M9 owner approval file: absent")
'@
    Invoke-PlexPython -Code $code
    & wsl.exe -d $Distro -- nvidia-smi --query-gpu=name,memory.free --format=csv
    if ($LASTEXITCODE -ne 0) { Write-Warning 'GPU memory query unavailable.' }
    break
  }
  'review' {
    if ($Run -ne 'm9') { throw 'Only M9 has this review workflow.' }
    $code = @'
from plex_training import plex600_m9 as m9
from plex_training import plex600_r3_curriculum as r3

def review(tokenizer):
    selected = m9.selected_records(tokenizer)
    for split in ("train", "validation"):
        print(f"\n=== {split.upper()} ===")
        for row, _ in selected["data"][split]:
            print(f"\nID: {row['id']}")
            print(f"Request: {row['request']}")
            print(f"Original:\n{row['inputFile']}")
            print(f"Expected:\n{row['expectedFile']}")

r3.with_frozen_tokenizer(review)
'@
    Invoke-PlexPython -Code $code
    break
  }
  'approve' {
    if ($Run -ne 'm9') { throw 'Only M9 approval is supported. M7 approval remains unchanged.' }
    Write-Host 'M9 approval scope: 8 source-conditioned R3 training updates; 3 held-out-by-family validation examples.'
    Write-Host 'The 111 proposed R3 records are assistant-authored. This approves ONLY the exact 8 selected records.'
    Write-Host 'No M8 development training, no final holdout, and a new optimizer initialized from verified M7 weights.'
    Write-Host 'First run: plex review m9; review the 11 original/expected files and their requests.'
    $answer = Read-Host 'If you personally reviewed and approve those exact examples, type APPROVE M9 8'
    if ($answer -cne 'APPROVE M9 8') { throw 'Approval cancelled; nothing written.' }
    $code = @'
from datetime import date
import json
from pathlib import Path
from plex_training import plex600_m9 as m9
from plex_training import plex600_r3_curriculum as r3
from plex_training.plex600_research import m7_local_receipt

root = Path("training/artifacts")
m7 = root / "plex600-m7-fresh-bootstrap"
receipt = m7_local_receipt(m7 / "report.json", m7 / "weights.pt")
if receipt["status"] != "report-and-checkpoint-verified":
    raise ValueError("M7 checkpoint was not independently verified")

def approve(tokenizer):
    selection = m9.selected_records(tokenizer)
    draft = m9.approval_draft(selection, checkpoint_sha=receipt["localCheckpointSha256"])
    draft.update(
        approvedBy="keepithandy",
        approvedAt=date.today().isoformat(),
        approvedForTraining=True,
        sourceReviewedByOwner=True,
    )
    path = root / "approvals/plex600-m9-r3-approved.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(draft, stream, indent=2, sort_keys=True)
        stream.write("\n")
    proof = m9.check_approval(path, selection, original_checkpoint_sha=receipt["localCheckpointSha256"])
    print(f"Owner-scoped M9 approval verified: {path}")
    print(f"Approval SHA-256: {proof['approvalSha256']}")

r3.with_frozen_tokenizer(approve)
'@
    Invoke-PlexPython -Code $code
    break
  }
  'train' {
    if ($Run -ne 'm9') { throw 'This launcher only runs the currently reviewed M9 experiment. It does not restart M7.' }
    # The Python runner independently rechecks exact local consent, hashes,
    # scope, GPU capacity, output freshness, and all other execution gates.
    Invoke-PlexUv -Arguments @(
      'run', '--project', 'training', '--no-sync',
      'python', '-m', 'plex_training.plex600_m9',
      '--execute-approved-run',
      '--bundle-dir', $Bundle,
      '--bundle-manifest-sha256', $Manifest,
      '--m7-report', "$M7/report.json",
      '--m7-weights', "$M7/weights.pt",
      '--approval-file', $Approval,
      '--output-dir', $M9
    )
    break
  }
  'backup' {
    $code = @'
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

run = sys.argv[1]
source = Path("training/artifacts") / (
    "plex600-m7-fresh-bootstrap" if run == "m7" else "plex600-m9-r3-first-eight"
)
report_path = source / "report.json"
weights_path = source / "weights.pt"
data = json.loads(report_path.read_text(encoding="utf-8"))
spec = data["checkpoint"]
if not weights_path.is_file() or spec["file"] != "weights.pt":
    raise ValueError("Missing run weights or checkpoint specification")
backup_root = Path("/mnt/c/Users/quali/Plex600M-Backups")
target = backup_root / run
backup_root.mkdir(parents=True, exist_ok=True)
if target.exists():
    raise FileExistsError(f"Backup already exists; refusing to overwrite: {target}")
if shutil.disk_usage(backup_root).free < weights_path.stat().st_size + report_path.stat().st_size + 128 * 1024**2:
    raise ValueError("Not enough Windows drive free space for backup")
def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
if sha(weights_path) != spec["sha256"] or weights_path.stat().st_size != spec["bytes"]:
    raise ValueError("Source weights fail independent checkpoint verification")
stage = Path(tempfile.mkdtemp(prefix=".plex-" + run + "-backup-", dir=backup_root))
try:
    for name in ("weights.pt", "report.json"):
        with (source / name).open("rb") as incoming, (stage / name).open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        if sha(source / name) != sha(stage / name):
            raise ValueError(f"{name} backup hash mismatch")
    if target.exists():
        raise FileExistsError(f"Backup appeared during copy: {target}")
    os.rename(stage, target)
    print(f"Backup copied and SHA-256 verified: {target}")
finally:
    if stage.exists():
        shutil.rmtree(stage)
'@
    Invoke-PlexPython -Code $code -Arguments @($Run)
    break
  }
  'evaluate' {
    if ($Run -ne 'm7') {
      throw 'The current M8 generation runner is bound to original M7 checkpoint identity. M9 evaluation needs a separate verified comparison runner; no unsupported shortcut will be used.'
    }
    Invoke-PlexUv -Arguments @(
      'run', '--project', 'training', '--no-sync',
      'python', '-m', 'plex_training.plex600_m8_eval',
      '--run-read-only-eval',
      '--bundle-dir', $Bundle,
      '--bundle-manifest-sha256', $Manifest,
      '--m7-report', "$M7/report.json",
      '--m7-weights', "$M7/weights.pt",
      '--output-dir', 'training/artifacts/plex600-m8-m7-first-eval',
      '--max-cases', '36',
      '--max-new-tokens', '96',
      '--decoding-policy', 'fitted-vocab-v1'
    )
    break
  }
}
