param(
    [string]$PythonPath = "",
    [switch]$RunP2PProbe
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

Write-Host "=== NVIDIA driver / mode ==="
$gpuRows = @(& nvidia-smi --query-gpu=index,name,driver_version,driver_model.current,memory.total --format=csv,noheader)
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi failed"
}
$gpuRows
if ($gpuRows.Count -ne 2) {
    throw "Exactly two NVIDIA GPUs are required; found $($gpuRows.Count)"
}
if ($gpuRows | Where-Object { $_ -notmatch ',\s*TCC\s*,' }) {
    throw "Both GPUs must use the TCC driver model"
}
if ($gpuRows | Where-Object { $_ -notmatch ',\s*577\.00\s*,' }) {
    Write-Warning "Driver 577.00 is the validated release. Continue only after the full P2P probe passes."
}

Write-Host "=== NVLink links ==="
& nvidia-smi nvlink -s
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi nvlink query failed"
}

$versionCheck = @'
import importlib.metadata as metadata
import sys
import torch

expected = {
    "torch": "2.7.0+cu126",
    "torchvision": "0.22.0+cu126",
    "torchaudio": "2.7.0+cu126",
    "xformers": "0.0.30",
    "ray": "2.57.0",
    "xfuser": "0.4.5",
    "yunchang": "0.6.4",
}

failures = []
print(f"python={sys.version.split()[0]} platform={sys.platform}")
for package, wanted in expected.items():
    actual = metadata.version(package)
    print(f"{package}={actual}")
    if actual != wanted:
        failures.append(f"{package}: expected {wanted}, found {actual}")

print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_devices={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"gpu{index}={torch.cuda.get_device_name(index)}")

if sys.platform != "win32":
    failures.append("native Windows is required")
if sys.version_info[:3] != (3, 10, 11):
    failures.append(f"Python 3.10.11 was validated; found {sys.version.split()[0]}")
if torch.version.cuda != "12.6":
    failures.append(f"PyTorch CUDA 12.6 was validated; found {torch.version.cuda}")
if torch.cuda.device_count() != 2:
    failures.append(f"exactly two visible CUDA devices are required; found {torch.cuda.device_count()}")
if any("Tesla V100-SXM2-16GB" not in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())):
    failures.append("the validated release requires two Tesla V100-SXM2-16GB GPUs")

if failures:
    print("FAIL:")
    for failure in failures:
        print(f"- {failure}")
    raise SystemExit(1)
print("PASS: validated software and GPU identity match")
'@

Write-Host "=== Python / package versions ==="
$versionCheck | & $PythonPath -
if ($LASTEXITCODE -ne 0) {
    throw "Version validation failed"
}

if ($RunP2PProbe) {
    Write-Host "=== Two-Ray-actor CUDA P2P probe ==="
    & $PythonPath (Join-Path $repoRoot "tests\windows_p2p_ray_probe.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Windows CUDA P2P probe failed"
    }
}

Write-Host "Environment verification completed successfully."
