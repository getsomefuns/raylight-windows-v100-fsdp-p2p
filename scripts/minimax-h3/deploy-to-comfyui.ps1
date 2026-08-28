[CmdletBinding()]
param(
    [string]$SourceRoot,
    [string]$DestinationRoot = "E:\ComfyUI-py310\ComfyUI\custom_nodes\raylight",
    [switch]$PlanOnly,
    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
}

$source = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $SourceRoot).Path).TrimEnd("\")
$destination = [IO.Path]::GetFullPath($DestinationRoot).TrimEnd("\")
$customNodes = [IO.Path]::GetFullPath((Split-Path -Parent $destination)).TrimEnd("\")
$comfyRoot = [IO.Path]::GetFullPath((Split-Path -Parent $customNodes)).TrimEnd("\")

if ((Split-Path -Leaf $destination) -ne "raylight" -or (Split-Path -Leaf $customNodes) -ne "custom_nodes") {
    throw "DestinationRoot must be <ComfyUI>\custom_nodes\raylight: $destination"
}
if (-not (Test-Path -LiteralPath $customNodes -PathType Container)) {
    throw "ComfyUI custom_nodes directory does not exist: $customNodes"
}

$canonicalDeployer = [IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..\deploy-to-comfyui.ps1")
)
if (-not (Test-Path -LiteralPath $canonicalDeployer -PathType Leaf)) {
    throw "Canonical deployment script not found: $canonicalDeployer"
}

$sourceCommit = @(& git -c "safe.directory=$source" -C $source rev-parse HEAD)[0]
if ($LASTEXITCODE -ne 0 -or $sourceCommit -notmatch '^[0-9a-f]{40}$') {
    throw "Unable to resolve source commit: $source"
}

$plan = [ordered]@{
    schema_version = 2
    deployment_kind = "plain-copy"
    source_root = $source
    destination_root = $destination
    source_commit = $sourceCommit
    canonical_deployer = $canonicalDeployer
}

if ($PlanOnly) {
    if ($Json) {
        $plan | ConvertTo-Json -Depth 4 -Compress
    }
    else {
        Write-Output "Raylight plain-copy deployment plan"
        Write-Output "Source commit: $sourceCommit"
        Write-Output "Destination: $destination"
        Write-Output "Canonical deployer: $canonicalDeployer"
    }
    return
}

& $canonicalDeployer -SourceRoot $source -ComfyUIRoot $comfyRoot
if ($LASTEXITCODE -ne 0) {
    throw "Canonical Raylight deployment failed with exit code $LASTEXITCODE"
}
