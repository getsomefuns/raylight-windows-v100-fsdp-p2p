[CmdletBinding()]
param(
    [string]$EnvironmentRoot = "",
    [string]$PythonPath = "",
    [string]$ManifestPath = "",
    [string]$WheelPath = "",
    [switch]$AllowCanonical,
    [switch]$ValidateOnly,
    [switch]$PrintConfiguration
)

$ErrorActionPreference = "Stop"

$expectedVersion = "0.4.13+tcc1"
$pipArguments = @("--no-deps", "--force-reinstall")
$configuration = [ordered]@{
    canonical_requires_explicit_override = $true
    expected_version = $expectedVersion
    pip_arguments = $pipArguments
}

if ($PrintConfiguration) {
    $configuration | ConvertTo-Json -Compress
    return
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($EnvironmentRoot)) {
    $EnvironmentRoot = Split-Path -Parent $repoRoot
}
$EnvironmentRoot = [System.IO.Path]::GetFullPath($EnvironmentRoot)
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $EnvironmentRoot "runtime-candidates\Python310-torch2.8.0-cu128-aimdo-tcc1\python.exe"
}

$requestedPython = [System.IO.Path]::GetFullPath($PythonPath)
$canonicalPython = [System.IO.Path]::GetFullPath((Join-Path $EnvironmentRoot "Python310\python.exe"))
if (-not $AllowCanonical -and $requestedPython.Equals($canonicalPython, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to modify canonical Python310 without -AllowCanonical"
}
$PythonPath = (Resolve-Path -LiteralPath $requestedPython).Path

if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Join-Path $EnvironmentRoot "runtime-builds\comfy-aimdo-v0.4.13-tcc1\BUILD-MANIFEST.json"
}
$ManifestPath = (Resolve-Path -LiteralPath $ManifestPath).Path
$verifier = (Resolve-Path -LiteralPath (Join-Path $repoRoot "scripts\runtime\verify_comfy_aimdo_tcc.py")).Path

$verification = & $PythonPath $verifier manifest --manifest $ManifestPath
if ($LASTEXITCODE -ne 0) {
    throw "Build manifest verification failed: $verification"
}
$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if ($manifest.version -ne $expectedVersion) {
    throw "Manifest version mismatch. Expected $expectedVersion, got $($manifest.version)"
}

$manifestRoot = Split-Path -Parent $ManifestPath
$manifestWheel = [System.IO.Path]::GetFullPath((Join-Path $manifestRoot $manifest.artifacts.wheel.path))
if ([string]::IsNullOrWhiteSpace($WheelPath)) {
    $WheelPath = $manifestWheel
}
$WheelPath = (Resolve-Path -LiteralPath $WheelPath).Path
if (-not $WheelPath.Equals($manifestWheel, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Wheel path does not match the verified build manifest"
}

$result = [ordered]@{
    python = $PythonPath
    wheel = $WheelPath
    manifest = $ManifestPath
    expected_version = $expectedVersion
    manifest_verified = $true
    installed = $false
}
if ($ValidateOnly) {
    $result | ConvertTo-Json -Compress
    return
}

& $PythonPath -m pip install @pipArguments $WheelPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install $WheelPath"
}
$installedVersion = (& $PythonPath -c "import importlib.metadata as m; print(m.version('comfy-aimdo'))").Trim()
if ($installedVersion -ne $expectedVersion) {
    throw "Installed comfy-aimdo version mismatch. Expected $expectedVersion, got $installedVersion"
}
$result.installed = $true
$result.installed_version = $installedVersion
$result | ConvertTo-Json -Compress
