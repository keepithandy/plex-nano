#requires -Version 5.1
<#
.SYNOPSIS
  Run the frozen Plex M11 Phase 2 comparison locally, inference only.
.DESCRIPTION
  One explicit -Run invokes the existing WSL checkpoint for two prompts
  against 33 development cases, then prints independent model/edit scores.
  No training, optimizer steps, or checkpoint writes are permitted by this path.
.EXAMPLE
  .\scripts\Invoke-PlexPhase2.ps1 -Run
#>
[CmdletBinding()]
param(
  [switch]$Run
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $Run) {
  Write-Host 'Plex Phase 2: local 600M evaluation (read-only)'
  Write-Host 'Run: .\scripts\Invoke-PlexPhase2.ps1 -Run'
  Write-Host 'Uses 33 supplied-JavaScript cases with two model prompts each.'
  Write-Host 'Weights and tokenizer stay unchanged. No training.'
  exit 0
}

$Distro = if ($env:PLEX_WSL_DISTRO) { $env:PLEX_WSL_DISTRO } else { 'Ubuntu' }
$Project = if ($env:PLEX_WSL_PROJECT) { $env:PLEX_WSL_PROJECT } else { '/home/quali/plex-nano' }
$Uv = if ($env:PLEX_WSL_UV) { $env:PLEX_WSL_UV } else { '/home/quali/.local/bin/uv' }
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
$Output = 'training/artifacts/plex600-phase2-m11-' + $Stamp + '-' + $PID

Write-Host 'PLEX PHASE 2 — RAW MODEL CHECK'
Write-Host 'Training: paused. No checkpoint changes.'
Write-Host ('Output: ' + $Project + '/' + $Output)
Write-Host 'Running inference-only M11 comparison...'

& wsl.exe -d $Distro --cd $Project -- $Uv run --project training --no-sync python scripts/plex_phase2_infer.py --execute-read-only --output-dir $Output
if ($LASTEXITCODE -ne 0) {
  throw "Model evaluation failed (exit $LASTEXITCODE). No model score is claimed."
}

Write-Host ''
Write-Host 'PLEX PHASE 2 — RESULTS'
& wsl.exe -d $Distro --cd $Project -- $Uv run --project training --no-sync python scripts/plex_phase2_eval.py --baseline-model-jsonl "$Output/baseline.jsonl" --alternative-model-jsonl "$Output/alternative.jsonl"
if ($LASTEXITCODE -ne 0) {
  throw "Raw model score verification failed (exit $LASTEXITCODE). Do not use an unverified model score."
}
Write-Host ('Evidence: ' + $Project + '/' + $Output)
