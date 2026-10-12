$ErrorActionPreference = 'Stop'
$swarmDirectory = $PSScriptRoot
$parentDirectory = Split-Path -Parent $swarmDirectory
$archivePath = Join-Path $parentDirectory 'review-swarm.zip'
if (Test-Path -LiteralPath $archivePath) {
    throw "Archive already exists: $archivePath"
}
Compress-Archive -LiteralPath $swarmDirectory -DestinationPath $archivePath -CompressionLevel Optimal
Write-Output $archivePath
