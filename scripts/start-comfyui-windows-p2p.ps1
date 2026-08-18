[CmdletBinding(DefaultParameterSetName = "MiB")]
param(
    [Parameter(Position = 0)]
    [string]$PythonPath = "",
    [Parameter(Position = 1)]
    [string]$ComfyRoot = "",
    [Parameter(Position = 2)]
    [string]$GlooHost = "",
    [Parameter(Position = 3)]
    [string]$GpuSelect = "0,1",
    [Parameter(Position = 4)]
    [int]$Port = 8188,
    [Parameter(Position = 5)]
    [int]$MasterPort = 29500,
    [Parameter(ParameterSetName = "MiB")]
    [ValidateSet(128, 256, 512)]
    [int]$P2PCapacityMiB = 256,
    [Parameter(ParameterSetName = "Bytes", Mandatory = $true, Position = 6)]
    [ValidateRange(1, [long]::MaxValue)]
    [long]$P2PCapacityBytes,
    [Parameter(Position = 7)]
    [double]$MinimumP2PGiBs = 50,
    [Parameter(Position = 8)]
    [int]$CudaMaxSplitSizeMiB = 128,
    [Parameter(Position = 9)]
    [double]$ReserveVramGiB = 2,
    [switch]$EnableDiagnostics,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ComfyRoot)) {
    $customNodesRoot = Split-Path -Parent $repoRoot
    $comfyRoot = Split-Path -Parent $customNodesRoot
} else {
    $comfyRoot = (Resolve-Path -LiteralPath $ComfyRoot -ErrorAction Stop).Path
}
$environmentRoot = Split-Path -Parent $comfyRoot

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $bundledPython = Join-Path $environmentRoot "Python310\python.exe"
    if (Test-Path -LiteralPath $bundledPython) {
        $PythonPath = $bundledPython
    } else {
        $PythonPath = (Get-Command python -ErrorAction Stop).Source
    }
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python executable not found: $PythonPath"
}
if ($ReserveVramGiB -lt 0) {
    throw "ReserveVramGiB must not be negative"
}
if (-not (Test-Path -LiteralPath (Join-Path $comfyRoot "main.py"))) {
    throw "ComfyUI main.py was not found below: $comfyRoot"
}

if ($PSCmdlet.ParameterSetName -eq "Bytes") {
    $selectedCapacityBytes = $P2PCapacityBytes
    $selectedCapacityMiB = [double]$selectedCapacityBytes / 1MB
} else {
    $selectedCapacityMiB = $P2PCapacityMiB
    $selectedCapacityBytes = [long]$selectedCapacityMiB * 1MB
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:USE_LIBUV = "0"
$env:MASTER_ADDR = "127.0.0.1"
$env:MASTER_PORT = [string]$MasterPort
$env:RAY_DEBUG_DISABLE_MEMORY_MONITOR = "1"
$env:RAY_memory_usage_threshold = "1"
$env:RAYLIGHT_WINDOWS_P2P = "1"
$env:RAYLIGHT_WINDOWS_P2P_CAPACITY_BYTES = [string]$selectedCapacityBytes
$env:RAYLIGHT_WINDOWS_P2P_MIN_GIB_S = [string]$MinimumP2PGiBs
$env:RAYLIGHT_RANK_DIAG = if ($EnableDiagnostics) { "1" } else { "0" }
if ([string]::IsNullOrWhiteSpace($env:PYTORCH_CUDA_ALLOC_CONF)) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:$CudaMaxSplitSizeMiB"
}
$env:CUDA_VISIBLE_DEVICES = $GpuSelect

if (-not [string]::IsNullOrWhiteSpace($GlooHost)) {
    $env:RAYLIGHT_GLOO_HOST = $GlooHost
}

Write-Host "Raylight repo: $repoRoot"
Write-Host "ComfyUI root: $comfyRoot"
Write-Host "Python: $PythonPath"
Write-Host "GPU selection: $GpuSelect"
Write-Host "P2P capacity choices: 128, 256, 512 MiB"
Write-Host "Selected P2P capacity: $selectedCapacityMiB MiB ($selectedCapacityBytes bytes per GPU)"
Write-Host "Rank/P2P diagnostics: $(if ($EnableDiagnostics) { 'enabled' } else { 'disabled' })"
Write-Host "Minimum P2P bandwidth: $MinimumP2PGiBs GiB/s"
Write-Host "Reserved VRAM: $ReserveVramGiB GiB"
Write-Host "CUDA allocator: $env:PYTORCH_CUDA_ALLOC_CONF"

if ($ValidateOnly) {
    Write-Host "Validation-only mode: ComfyUI was not started."
    return
}

Set-Location -LiteralPath $comfyRoot
$mainArgs = @("main.py", "--listen", "127.0.0.1", "--port", [string]$Port, "--disable-cuda-malloc")
if ($ReserveVramGiB -gt 0) {
    $reserveVramText = $ReserveVramGiB.ToString([Globalization.CultureInfo]::InvariantCulture)
    $mainArgs += @("--reserve-vram", $reserveVramText)
}
& $PythonPath @mainArgs
