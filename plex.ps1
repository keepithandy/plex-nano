# Plex Nano - U1.2: interactive terminal by default; -Prompt / -Once for single request.
# No training, checkpoint changes, or automatic model selection.
[CmdletBinding()]
param(
    [string] $Checkpoint = "",
    [string] $BundleDir = "",
    [string] $Prompt = "",
    [switch] $Once,
    [ValidateSet("auto", "cpu", "cuda")] [string] $Device = "auto",
    [ValidateRange(1, 256)] [int] $MaxNewTokens = 64,
    [ValidateRange(0, 10)] [double] $Temperature = 0
)

$ErrorActionPreference = "Stop"
$exitCode = 0

Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv was not found. Install uv, then run: uv sync --project training"
    }

    $environmentPython = Join-Path $PSScriptRoot "training\.venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $environmentPython -PathType Leaf)) {
        throw "Plex's Python environment is missing. Run once: uv sync --project training"
    }

    if (-not $Checkpoint.Trim()) {
        $Checkpoint = Read-Host "Path to a trained Plex .pt checkpoint"
    }
    $Checkpoint = $Checkpoint.Trim().Trim('"')
    if (-not $Checkpoint) {
        throw "No checkpoint selected. This launcher never chooses weights automatically."
    }
    $checkpointItem = Get-Item -LiteralPath $Checkpoint -ErrorAction Stop
    if ($checkpointItem.PSIsContainer) {
        throw "The checkpoint path must be a trained .pt file, not a folder."
    }

    if (-not $BundleDir.Trim()) {
        $defaultBundle = Join-Path $checkpointItem.DirectoryName "tokenizer"
        if (Test-Path -LiteralPath $defaultBundle -PathType Container) {
            $BundleDir = $defaultBundle
        }
        else {
            $BundleDir = Read-Host "Path to the matching tokenizer bundle folder"
        }
    }
    $BundleDir = $BundleDir.Trim().Trim('"')
    if (-not $BundleDir) {
        throw "A matching tokenizer bundle is required."
    }
    $bundleItem = Get-Item -LiteralPath $BundleDir -ErrorAction Stop
    if (-not $bundleItem.PSIsContainer) {
        throw "The tokenizer bundle must be a folder."
    }

    # Default: interactive terminal. Preserve U1.1 single-shot with -Prompt or -Once.
    $singlePrompt = $Once.IsPresent -or -not [string]::IsNullOrWhiteSpace($Prompt)
    if ($singlePrompt -and -not $Prompt.Trim()) {
        $Prompt = Read-Host "Enter one short coding request"
    }
    if ($singlePrompt -and -not $Prompt.Trim()) {
        throw "No coding request entered."
    }

    $entryPoint = if ($singlePrompt) { "plex_training.cli" } else { "plex_training.interactive" }
    $arguments = @(
        "run", "--project", "training", "--no-sync", "python", "-m",
        $entryPoint
    )
    if ($singlePrompt) {
        $arguments += "complete"
    }
    $arguments += @(
        "--checkpoint", $checkpointItem.FullName,
        "--bundle-dir", $bundleItem.FullName,
        "--device", $Device,
        "--max-new-tokens", "$MaxNewTokens",
        "--temperature", $Temperature.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    )
    if ($singlePrompt) {
        $arguments += @("--prompt", $Prompt)
    }

    Write-Host ""
    if ($singlePrompt) {
        Write-Host "Plex Nano - one completion (no training)" -ForegroundColor Cyan
    }
    else {
        Write-Host "Plex Nano - interactive mode (no training)" -ForegroundColor Cyan
    }
    Write-Host "Checkpoint: $($checkpointItem.FullName)"
    Write-Host "Tokenizer:  $($bundleItem.FullName)"
    Write-Host ""
    & uv @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Plex generation failed (exit code $LASTEXITCODE). See the error above."
    }
}
catch {
    [Console]::Error.WriteLine("Plex launcher: $($_.Exception.Message)")
    $exitCode = 2
}
finally {
    Pop-Location
}

exit $exitCode
