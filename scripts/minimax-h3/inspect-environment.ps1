param(
    [string]$ComfyRoot = "E:\ComfyUI-py310\ComfyUI",
    [string]$ModelRoot = "E:\ComfyUI-aki-v3\ComfyUI\models",
    [string]$PythonPath = "E:\ComfyUI-py310\Python310\python.exe",
    [switch]$SkipRuntimeProbes,
    [switch]$Json
)

$ErrorActionPreference = "Stop"

$requiredModels = @(
    [ordered]@{
        name = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
        subdirectory = "diffusion_models"
        expected_bytes = [int64]20958205608
    },
    [ordered]@{
        name = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
        subdirectory = "diffusion_models"
        expected_bytes = [int64]20958205608
    },
    [ordered]@{
        name = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        subdirectory = "text_encoders"
        expected_bytes = [int64]15687142551
    },
    [ordered]@{
        name = "minimax_h3_video_vae_fp16.safetensors"
        subdirectory = "vae"
        expected_bytes = [int64]5207808496
    },
    [ordered]@{
        name = "minimax_h3_audio_vae_fp32.safetensors"
        subdirectory = "vae"
        expected_bytes = [int64]605254808
    }
)

$modelRows = foreach ($model in $requiredModels) {
    $path = Join-Path (Join-Path $ModelRoot $model.subdirectory) $model.name
    $exists = Test-Path -LiteralPath $path -PathType Leaf
    $size = $null
    if ($exists) {
        $size = [int64](Get-Item -LiteralPath $path).Length
    }
    [ordered]@{
        name = $model.name
        subdirectory = $model.subdirectory
        expected_bytes = $model.expected_bytes
        exists = [bool]$exists
        size_bytes = $size
        complete = [bool]($exists -and $size -eq $model.expected_bytes)
    }
}

$expectedNodes = @(
    "RayInitializer",
    "RayUNETLoader",
    "XfuserSamplerCustomAdvanced",
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo"
)

$runtime = $null
$nodes = $null
if (-not $SkipRuntimeProbes) {
    $gpuRows = @()
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpuRows = @(
            nvidia-smi --query-gpu=index,name,driver_version,driver_model.current,memory.total --format=csv,noheader,nounits |
                ForEach-Object { $_.Trim() }
        )
    }

    $pythonInfo = $null
    if (Test-Path -LiteralPath $PythonPath -PathType Leaf) {
        $probe = @"
import json, sys
import torch
out = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "sm_arches": torch.cuda.get_arch_list(),
    "nccl_available": torch.distributed.is_nccl_available(),
    "gloo_available": torch.distributed.is_gloo_available(),
}
for name in ("ray", "xformers", "yunchang", "comfy_kitchen"):
    try:
        module = __import__(name)
        out[name] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        out[name] = "unavailable: " + type(exc).__name__
print(json.dumps(out))
"@
        $encodedProbe = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($probe))
        $launcher = "import base64;exec(base64.b64decode('$encodedProbe'))"
        $probeOutput = & $PythonPath -c $launcher
        if ($LASTEXITCODE -ne 0) {
            throw "Python runtime probe failed with exit code $LASTEXITCODE"
        }
        $pythonInfo = ($probeOutput | ConvertFrom-Json)
    }

    Add-Type -AssemblyName Microsoft.VisualBasic
    $computer = [Microsoft.VisualBasic.Devices.ComputerInfo]::new()
    $drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($ModelRoot))

    $runtime = [ordered]@{
        gpu = $gpuRows
        python = $pythonInfo
        ram_total_bytes = [int64]$computer.TotalPhysicalMemory
        ram_available_bytes = [int64]$computer.AvailablePhysicalMemory
        model_drive_free_bytes = [int64]$drive.AvailableFreeSpace
    }

    $sourceRoots = @(
        (Join-Path $ComfyRoot "comfy_extras"),
        (Join-Path $ComfyRoot "custom_nodes\raylight\src")
    )
    $pythonFiles = @(
        $sourceRoots |
            Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
            ForEach-Object { Get-ChildItem -LiteralPath $_ -Recurse -File -Filter "*.py" }
    )
    $nodes = foreach ($nodeName in $expectedNodes) {
        $found = $false
        foreach ($file in $pythonFiles) {
            if (Select-String -LiteralPath $file.FullName -Pattern $nodeName -SimpleMatch -Quiet) {
                $found = $true
                break
            }
        }
        [ordered]@{ name = $nodeName; source_present = [bool]$found }
    }
}

$completeCount = @($modelRows | Where-Object { $_.complete }).Count
$expectedBytes = [int64]0
foreach ($model in $requiredModels) {
    $expectedBytes += [int64]$model.expected_bytes
}
$report = [ordered]@{
    schema_version = 1
    comfy_root = $ComfyRoot
    model_root = $ModelRoot
    models = @($modelRows)
    nodes = $nodes
    runtime = $runtime
    summary = [ordered]@{
        model_count = @($modelRows).Count
        complete_count = $completeCount
        missing_or_incomplete_count = @($modelRows).Count - $completeCount
        expected_bytes = $expectedBytes
    }
}

if ($Json) {
    $report | ConvertTo-Json -Depth 8 -Compress
    exit 0
}

Write-Host "=== MiniMax H3 Windows V100 inventory ==="
Write-Host "ComfyUI: $ComfyRoot"
Write-Host "Models:  $ModelRoot"
foreach ($row in $modelRows) {
    $state = if ($row.complete) { "OK" } elseif ($row.exists) { "INCOMPLETE" } else { "MISSING" }
    Write-Host ("{0,-10} {1}\{2}" -f $state, $row.subdirectory, $row.name)
}
Write-Host ("Complete: {0}/{1}; expected total: {2:N2} GiB" -f $completeCount, @($modelRows).Count, ($report.summary.expected_bytes / 1GB))
if ($runtime) {
    Write-Host ("RAM: {0:N2} GiB total, {1:N2} GiB available" -f ($runtime.ram_total_bytes / 1GB), ($runtime.ram_available_bytes / 1GB))
    Write-Host ("Model drive free: {0:N2} GiB" -f ($runtime.model_drive_free_bytes / 1GB))
    foreach ($gpu in $runtime.gpu) {
        Write-Host "GPU: $gpu"
    }
}
if ($nodes) {
    foreach ($node in $nodes) {
        $state = if ($node.source_present) { "OK" } else { "MISSING" }
        Write-Host ("NODE {0,-8} {1}" -f $state, $node.name)
    }
}
