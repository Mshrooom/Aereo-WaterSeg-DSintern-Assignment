param(
    [Parameter(Mandatory = $true)]
    [string]$ImagePath,

    [string]$OutputPath = ".\predicted_water_mask.png"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ImagePath -PathType Leaf)) {
    throw "Image not found: $ImagePath"
}

Write-Host "Health:" -ForegroundColor Cyan
curl.exe --fail --silent --show-error http://localhost:8000/health
Write-Host "`nReady:" -ForegroundColor Cyan
curl.exe --fail --silent --show-error http://localhost:8000/ready
Write-Host "`nMetadata:" -ForegroundColor Cyan
curl.exe --fail --silent --show-error http://localhost:8000/metadata
Write-Host "`nRunning segmentation..." -ForegroundColor Cyan

curl.exe --fail --silent --show-error `
    -X POST "http://localhost:8000/segment" `
    -F "image=@$ImagePath" `
    --output $OutputPath

if (-not (Test-Path $OutputPath -PathType Leaf)) {
    throw "The API did not create: $OutputPath"
}

$size = (Get-Item $OutputPath).Length
Write-Host "Saved $OutputPath ($size bytes)" -ForegroundColor Green
