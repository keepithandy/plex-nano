#requires -Version 5.1
<#
.SYNOPSIS
  Diagnose why Plex 600M M11 immediately predicts EOS.
.DESCRIPTION
  Runs 12 read-only generations against four familiar M11 code lines under
  three prompts. Saves a new report without editing model weights or training.
.EXAMPLE
  .\scripts\Invoke-PlexPhase2Eos.ps1 -Run
#>
[CmdletBinding()]
param([switch]$Run)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $Run) {
    Write-Host 'PLEX PHASE 2 - EOS DIAGNOSTIC'
    Write-Host 'Run: .\scripts\Invoke-PlexPhase2Eos.ps1 -Run'
    Write-Host 'Compares familiar M11 prompts, changed delimiters, and the console prompt.'
    Write-Host 'Inference only; training remains paused.'
    exit 0
}

$Distro = if ($env:PLEX_WSL_DISTRO) { $env:PLEX_WSL_DISTRO } else { 'Ubuntu' }
$Project = if ($env:PLEX_WSL_PROJECT) { $env:PLEX_WSL_PROJECT } else { '/home/quali/plex-nano' }
$Uv = if ($env:PLEX_WSL_UV) { $env:PLEX_WSL_UV } else { '/home/quali/.local/bin/uv' }
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
$Output = 'training/artifacts/plex600-phase2-eos-' + $Stamp + '-' + $PID

Write-Host 'PLEX PHASE 2 - EOS DIAGNOSTIC'
Write-Host 'Training: paused. Checkpoint and tokenizer remain unchanged.'
Write-Host 'Starting 12 read-only model probes...'
& wsl.exe -d $Distro --cd $Project -- $Uv run --project training --no-sync python scripts/plex_phase2_eos_probe.py --execute-read-only --output-dir $Output
if ($LASTEXITCODE -ne 0) {
    throw "EOS diagnostic failed (exit $LASTEXITCODE). No new model result is claimed."
}
