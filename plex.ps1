# Plex Nano - U1.2: interactive terminal by default; -Prompt / -Once for single request.
# The user-confirmed P2-48 step-100 checkpoint is the default. No training or checkpoint changes.
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

    # The owner already selected and verified this exact P2-48 endpoint on this PC.
    # Resolve relative to the launcher, never to a Windows username or search order.
    $knownCheckpoint = Join-Path $PSScriptRoot "training\artifacts\request-grounded\p2-48-first-run\checkpoints\step-0100.pt"
    $knownBundle = Join-Path $PSScriptRoot "training\artifacts\request-grounded\p2-48-training-bundle"
    $expectedCheckpointSha256 = "fd86d11375e547f05b2fab7a36188ca30a7ecd0b0f38198de62bcddbfba489b7"

    $usingKnownCheckpoint = [string]::IsNullOrWhiteSpace($Checkpoint)
    if ($usingKnownCheckpoint) {
        $Checkpoint = $knownCheckpoint
        if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
            throw "Default P2-48 checkpoint is missing: $Checkpoint. Supply -Checkpoint and -BundleDir to choose another trained model."
        }
    }
    $Checkpoint = $Checkpoint.Trim().Trim('"')
    $checkpointItem = Get-Item -LiteralPath $Checkpoint -ErrorAction Stop
    if ($checkpointItem.PSIsContainer) {
        throw "The checkpoint path must be a trained .pt file, not a folder."
    }

    # Never silently use a different endpoint at the known default path.
    if ($usingKnownCheckpoint) {
        $actualSha256 = (Get-FileHash -LiteralPath $checkpointItem.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSha256 -ne $expectedCheckpointSha256) {
            throw "Default P2-48 checkpoint hash does not match the previously verified step-100 model. Specify -Checkpoint and -BundleDir explicitly if you intend another model."
        }
    }

    if ([string]::IsNullOrWhiteSpace($BundleDir)) {
        if ($usingKnownCheckpoint) {
            $BundleDir = $knownBundle
        }
        else {
            $siblingBundle = Join-Path $checkpointItem.DirectoryName "tokenizer"
            if (Test-Path -LiteralPath $siblingBundle -PathType Container) {
                $BundleDir = $siblingBundle
            }
            else {
                throw "No tokenizer folder beside this checkpoint. Supply -BundleDir to select its matching tokenizer."
            }
        }
    }
    $BundleDir = $BundleDir.Trim().Trim('"')
    if (-not (Test-Path -LiteralPath $BundleDir -PathType Container)) {
        throw "Matching tokenizer bundle not found: $BundleDir. Supply -BundleDir to select its folder."
    }
    $bundleItem = Get-Item -LiteralPath $BundleDir -ErrorAction Stop

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
