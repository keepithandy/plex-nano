#requires -Version 5.1
<#
.SYNOPSIS
  Verify the existing M9 backup; optionally run M10 read-only copy evaluation.
.DESCRIPTION
  Run from the cloned Plex Nano repository on Windows. This helper invokes
  ONLY the read-only baseline module in WSL2: no training, approval, optimizer,
  checkpoint overwrites or branch changes. It follows the existing
  Plex-Control.ps1 defaults and backs up NOTHING by itself.
.EXAMPLE
  .\scripts\Invoke-PlexM10Baseline.ps1
.EXAMPLE
  .\scripts\Invoke-PlexM10Baseline.ps1 -Run -Cases 3
.EXAMPLE
  .\scripts\Invoke-PlexM10Baseline.ps1 -Run -Cases 18 -Decoder both
#>
[CmdletBinding()]
param(
  [switch]$Run,
  [ValidateRange(1,18)]
  [int]$Cases = 3,
  [ValidateSet('both','unmasked-v1','fitted-vocab-v1')]
  [string]$Decoder = 'both',
  [string]$BackupDirectory = '/mnt/c/Users/quali/Plex600M-Backups/m9'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Distro = if ($env:PLEX_WSL_DISTRO) { $env:PLEX_WSL_DISTRO } else { 'Ubuntu' }
$Project = if ($env:PLEX_WSL_PROJECT) { $env:PLEX_WSL_PROJECT } else { '/home/quali/plex-nano' }
$Uv = if ($env:PLEX_WSL_UV) { $env:PLEX_WSL_UV } else { '/home/quali/.local/bin/uv' }

$Bundle = 'training/artifacts/plex600-fresh-tokenizer'
$Manifest = '14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55'
$Module = 'plex_training.plex600_m10_baseline'
$Backup = $BackupDirectory.TrimEnd('/')
if ([string]::IsNullOrWhiteSpace($Backup)) {
  throw 'Backup directory must be a nonempty WSL path.'
}
$Report = "$Backup/report.json"
$Weights = "$Backup/weights.pt"

# Paths with the actual run must always come from explicit parameters or the
# frozen launcher defaults. Never discover random checkpoint files or overwrite
# an existing output folder.
$SharedArgs = @(
  'run', '--project', 'training', '--no-sync',
  'python', '-m', $Module,
  '--bundle-dir', $Bundle,
  '--bundle-manifest-sha256', $Manifest,
  '--m9-report', $Report,
  '--m9-weights', $Weights,
  '--max-cases', "$Cases",
  '--decoding', $Decoder
)

function Invoke-ReadOnlyPlex {
  param([Parameter(Mandatory=$true)][string[]]$Arguments)
  & wsl.exe -d $Distro --cd $Project -- $Uv @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "M10 read-only command exited with $LASTEXITCODE. No success is assumed. Review the error above."
  }
}

Write-Host 'Plex M10 — READ-ONLY M9 baseline' -ForegroundColor Cyan
Write-Host "WSL project: $Project"
Write-Host "Independent backup: $Backup"
Write-Host "Cases: $Cases; decoder: $Decoder"
Write-Host 'Training: DISABLED. M7/M9 weights: NEVER MODIFIED.'

# This verifies SHA256 provenance, tokenizer, source binding, case budget,
# and the existing independent M9 backup; it does not use CUDA.
Write-Host 'Step 1: CPU-only admission and complete prompt token audit...'
Invoke-ReadOnlyPlex -Arguments (@($SharedArgs) + @('--audit-inputs'))

if (-not $Run) {
  Write-Host ''
  Write-Host 'CPU preflight completed. No GPU evaluation or training was started.' -ForegroundColor Green
  Write-Host 'To run the read-only M9 baseline, add -Run (optionally -Cases 18).'
  exit 0
}

# Explicit user request to RUN read-only inference is required.
# Timestamp uses UTC; a fresh output path is required by the Python runner.
$Time = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss-fffffff')
$Output = "training/artifacts/plex600-m10-m9-baseline-$Time"
Write-Host 'Step 2: Explicit read-only M9 inference (zero training updates)...'
Write-Host "New output directory (never overwritten): $Output"
Invoke-ReadOnlyPlex -Arguments (@($SharedArgs) + @(
  '--run-read-only-baseline', '--output-dir', $Output
))
Write-Host ''
Write-Host "Read-only M9 baseline saved: $Project/$Output" -ForegroundColor Green
Write-Host 'Report: report.json; raw generations: responses.jsonl'
Write-Host 'No M10 optimizer updates were attempted.'
