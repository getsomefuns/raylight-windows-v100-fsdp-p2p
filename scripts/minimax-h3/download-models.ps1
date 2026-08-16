[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("i2v-fp8", "ref2va-fp8", "i2v-int8", "ref2va-int8", "turbo")]
    [string]$Group,
    [ValidateSet("fl2va-fp8", "text-encoder-nvfp4", "video-vae-fp16", "audio-vae-fp32", "ref2va-fp8", "fl2va-int8", "ref2va-int8", "fl2v-turbo-4step", "fl2v-turbo-8step", "ref2v-turbo-4step")]
    [string]$ModelId,
    [string]$ModelRoot = "E:\ComfyUI-aki-v3\ComfyUI\models",
    [string]$ManifestPath,
    [switch]$PlanOnly,
    [switch]$Json
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Join-Path $PSScriptRoot "models.json"
}

function Test-SafetensorsHeader {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try {
        if ($stream.Length -lt 10) {
            throw "Safetensors file is too small: $Path"
        }
        $lengthBytes = New-Object byte[] 8
        if ($stream.Read($lengthBytes, 0, 8) -ne 8) {
            throw "Unable to read safetensors header length: $Path"
        }
        $headerLength = [BitConverter]::ToUInt64($lengthBytes, 0)
        $maximumHeader = [Math]::Min([uint64]134217728, [uint64]($stream.Length - 8))
        if ($headerLength -lt 2 -or $headerLength -gt $maximumHeader) {
            throw "Invalid safetensors header length $headerLength in $Path"
        }
        $headerBytes = New-Object byte[] ([int]$headerLength)
        $offset = 0
        while ($offset -lt $headerBytes.Length) {
            $read = $stream.Read($headerBytes, $offset, $headerBytes.Length - $offset)
            if ($read -le 0) {
                throw "Unexpected end of safetensors header: $Path"
            }
            $offset += $read
        }
        $headerText = [Text.Encoding]::UTF8.GetString($headerBytes)
        $null = $headerText | ConvertFrom-Json
        return $true
    }
    finally {
        $stream.Dispose()
    }
}

if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw "Model manifest not found: $ManifestPath"
}
$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$selected = @($manifest.models | Where-Object { @($_.groups) -contains $Group })
if ($selected.Count -eq 0) {
    throw "No models found for group: $Group"
}
if (-not [string]::IsNullOrWhiteSpace($ModelId)) {
    $selected = @($selected | Where-Object { [string]$_.id -eq $ModelId })
    if ($selected.Count -eq 0) {
        throw "Model $ModelId is not part of group $Group"
    }
}


$baseUrl = "$($manifest.repository)/resolve/$($manifest.revision)"
$files = foreach ($model in $selected) {
    $relativePath = [string]$model.relative_path
    $targetPath = Join-Path $ModelRoot ($relativePath -replace "/", "\")
    $partPath = "$targetPath.part"
    $exists = Test-Path -LiteralPath $targetPath -PathType Leaf
    $size = $null
    if ($exists) {
        $size = [int64](Get-Item -LiteralPath $targetPath).Length
    }
    $partBytes = $null
    if (Test-Path -LiteralPath $partPath -PathType Leaf) {
        $partBytes = [int64](Get-Item -LiteralPath $partPath).Length
    }
    $expectedBytes = [int64]$model.expected_bytes
    $status = if ($exists -and $size -eq $expectedBytes) {
        "complete"
    }
    elseif ($exists) {
        "invalid"
    }
    elseif ($partBytes -ne $null) {
        "partial"
    }
    else {
        "missing"
    }
    [ordered]@{
        id = [string]$model.id
        relative_path = $relativePath
        expected_bytes = $expectedBytes
        url = "$baseUrl/$relativePath"
        target_path = $targetPath
        part_path = $partPath
        size_bytes = $size
        part_bytes = $partBytes
        status = $status
    }
}

$totalBytes = [int64]0
foreach ($file in $files) {
    $totalBytes += [int64]$file.expected_bytes
}
$plan = [ordered]@{
    schema_version = 1
    group = $Group
    model_root = $ModelRoot
    model_id = $ModelId
    total_bytes = $totalBytes
    files = @($files)
}

if ($PlanOnly) {
    if ($Json) {
        $plan | ConvertTo-Json -Depth 6 -Compress
    }
    else {
        Write-Host "MiniMax H3 download plan: $Group"
        foreach ($file in $files) {
            Write-Host ("{0,-8} {1,8:N2} GiB  {2}" -f $file.status, ($file.expected_bytes / 1GB), $file.relative_path)
        }
        Write-Host ("Total: {0:N2} GiB" -f ($totalBytes / 1GB))
    }
    exit 0
}

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "curl.exe is required for resumable downloads"
}

foreach ($file in $files) {
    if ($file.status -eq "complete") {
        Write-Host "Complete, skipping: $($file.relative_path)"
        continue
    }
    if ($file.status -eq "invalid") {
        throw "Existing final file has the wrong size; refusing to overwrite: $($file.target_path)"
    }
    if ($file.part_bytes -ne $null -and [int64]$file.part_bytes -gt [int64]$file.expected_bytes) {
        throw "Partial file is larger than expected; remove or inspect it manually: $($file.part_path)"
    }

    $targetDirectory = Split-Path -Parent $file.target_path
    if ($PSCmdlet.ShouldProcess($file.target_path, "Download $($file.url)")) {
        New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
        Write-Host ("Downloading {0} ({1:N2} GiB)" -f $file.relative_path, ($file.expected_bytes / 1GB))
        & curl.exe -L --fail --retry 5 --retry-delay 5 -C - --output $file.part_path $file.url
        if ($LASTEXITCODE -ne 0) {
            throw "curl.exe failed with exit code $LASTEXITCODE for $($file.relative_path)"
        }

        $downloadedBytes = [int64](Get-Item -LiteralPath $file.part_path).Length
        if ($downloadedBytes -ne [int64]$file.expected_bytes) {
            throw "Downloaded size mismatch for $($file.relative_path): expected $($file.expected_bytes), got $downloadedBytes"
        }
        $null = Test-SafetensorsHeader -Path $file.part_path
        Move-Item -LiteralPath $file.part_path -Destination $file.target_path
        Write-Host "Validated: $($file.target_path)"
    }
}
