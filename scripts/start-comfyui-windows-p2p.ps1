param(
    [string]$PythonPath = "",
    [string]$GlooHost = "",
    [string]$GpuSelect = "0,1",
    [int]$Port = 8188,
    [int]$MasterPort = 29500,
    [long]$P2PCapacityBytes = 67108864,
    [double]$MinimumP2PGiBs = 50,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$customNodesRoot = Split-Path -Parent $repoRoot
$comfyRoot = Split-Path -Parent $customNodesRoot
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
if (-not (Test-Path -LiteralPath (Join-Path $comfyRoot "main.py"))) {
    throw "ComfyUI main.py was not found below: $comfyRoot"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:USE_LIBUV = "0"
$env:MASTER_ADDR = "127.0.0.1"
$env:MASTER_PORT = [string]$MasterPort
$env:RAY_DEBUG_DISABLE_MEMORY_MONITOR = "1"
$env:RAY_memory_usage_threshold = "1"
$env:RAYLIGHT_WINDOWS_P2P = "1"
$env:RAYLIGHT_WINDOWS_P2P_CAPACITY_BYTES = [string]$P2PCapacityBytes
$env:RAYLIGHT_WINDOWS_P2P_MIN_GIB_S = [string]$MinimumP2PGiBs
$env:CUDA_VISIBLE_DEVICES = $GpuSelect

if (-not [string]::IsNullOrWhiteSpace($GlooHost)) {
    $env:RAYLIGHT_GLOO_HOST = $GlooHost
}

Write-Host "Raylight repo: $repoRoot"
Write-Host "ComfyUI root: $comfyRoot"
Write-Host "Python: $PythonPath"
Write-Host "GPU selection: $GpuSelect"
Write-Host "P2P capacity: $P2PCapacityBytes bytes; minimum: $MinimumP2PGiBs GiB/s"

if ($ValidateOnly) {
    Write-Host "Validation-only mode: ComfyUI was not started."
    return
}

Set-Location -LiteralPath $comfyRoot
& $PythonPath main.py --listen 127.0.0.1 --port $Port --disable-cuda-malloc
