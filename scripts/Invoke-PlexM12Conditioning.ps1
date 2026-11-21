#requires -Version 5.1
<#
.SYNOPSIS
  Measure M11 source-conditioning with teacher-forced next-token logits.
.DESCRIPTION
  Research only. Runs 8 gold-prefix scoring forwards, NOT autonomous generations,
  optimizer updates or training. Uses frozen M11 step-200 checkpoint on Ubuntu WSL.
.EXAMPLE
  .\scripts\Invoke-PlexM12Conditioning.ps1 -Run
#>
[CmdletBinding()]
param([switch]$Run)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $Run) {
    Write-Host 'PLEX M12-A - SOURCE-CONDITIONING RESEARCH'
    Write-Host 'Run: .\scripts\Invoke-PlexM12Conditioning.ps1 -Run'
    Write-Host 'Scores 8 gold-prefix cases with M11 weights, not autonomous outputs.'
    Write-Host 'No training, optimizer or checkpoint changes.'
    exit 0
}

$Distro = if ($env:PLEX_WSL_DISTRO) { $env:PLEX_WSL_DISTRO } else { 'Ubuntu' }
$Project = if ($env:PLEX_WSL_PROJECT) { $env:PLEX_WSL_PROJECT } else { '/home/quali/plex-nano' }
$Uv = if ($env:PLEX_WSL_UV) { $env:PLEX_WSL_UV } else { '/home/quali/.local/bin/uv' }
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
$Output = 'training/artifacts/plex600-m12-conditioning-' + $Stamp + '-' + $PID

Write-Host 'PLEX M12-A - SOURCE-CONDITIONING RESEARCH'
Write-Host 'Training: paused. Teacher-forced scores are not model code-generation successes.'
& wsl.exe -d $Distro --cd $Project -- $Uv run --project training --no-sync python scripts/plex_m12_source_conditioning.py --execute-read-only --output-dir $Output
if ($LASTEXITCODE -ne 0) {
    throw "M12-A read-only source-conditioning measurement failed (exit $LASTEXITCODE)."
}
