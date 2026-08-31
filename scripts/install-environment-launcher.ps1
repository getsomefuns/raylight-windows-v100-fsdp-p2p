[CmdletBinding()]
param(
    [string]$EnvironmentRoot = "",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($EnvironmentRoot)) {
    $environment = Split-Path -Parent $repoRoot
} else {
    $environment = (Resolve-Path -LiteralPath $EnvironmentRoot -ErrorAction Stop).Path
}
if (-not (Test-Path -LiteralPath $environment -PathType Container)) {
    throw "Environment root does not exist: $environment"
}

$target = Join-Path $environment "start_raylight_p2p.ps1"
$wrapper = @'
$ErrorActionPreference = "Stop"

$launcher = Join-Path $PSScriptRoot "ComfyUI\custom_nodes\raylight\scripts\start-comfyui-windows-p2p.ps1"
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Deployed Raylight launcher not found: $launcher"
}

& $launcher @args
exit $LASTEXITCODE
'@

Write-Host "Environment launcher target: $target"
Write-Host "Forwarded launcher: ComfyUI\custom_nodes\raylight\scripts\start-comfyui-windows-p2p.ps1"
if ($ValidateOnly) {
    Write-Host "Validation-only mode: launcher was not written."
    return
}

Set-Content -LiteralPath $target -Value $wrapper -Encoding UTF8
Write-Host "Environment launcher installed."
