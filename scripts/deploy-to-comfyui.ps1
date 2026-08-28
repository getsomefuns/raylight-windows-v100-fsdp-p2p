[CmdletBinding()]
param(
    [string]$SourceRoot = (Split-Path -Parent $PSScriptRoot),
    [Parameter(Mandatory = $true)]
    [string]$ComfyUIRoot,
    [switch]$VerifyOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-NormalizedFullPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return [IO.Path]::GetFullPath($Path).TrimEnd([IO.Path]::DirectorySeparatorChar)
}

function Get-TrackedFiles {
    param([Parameter(Mandatory = $true)][string]$RepositoryRoot)

    $output = @(& git -c "safe.directory=$RepositoryRoot" -C $RepositoryRoot ls-files)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to read the Git tracked-file manifest from: $RepositoryRoot"
    }

    $files = @($output | Where-Object { $_ -and -not $_.StartsWith(".git/") } | Sort-Object -Unique)
    if ($files.Count -eq 0) {
        throw "The source repository has no tracked files: $RepositoryRoot"
    }

    foreach ($relative in $files) {
        if ([IO.Path]::IsPathRooted($relative) -or $relative -match '(^|/)\.\.(/|$)') {
            throw "Unsafe tracked path: $relative"
        }
    }
    return $files
}

function Get-RelativeFilePath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$FullName
    )

    $prefix = $Root.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $FullName.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "File escaped deployment root: $FullName"
    }
    return $FullName.Substring($prefix.Length).Replace("\", "/")
}

function Test-IsRuntimeCachePath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $normalized = $RelativePath.Replace("\", "/")
    $segments = @($normalized.Split("/"))
    if (
        $segments -contains "__pycache__" -or
        $segments -contains ".pytest_cache" -or
        $segments -contains ".ruff_cache" -or
        $segments -contains "_ray_runtime_env"
    ) {
        return $true
    }
    return $normalized.EndsWith(".pyc", [StringComparison]::OrdinalIgnoreCase) -or
        $normalized.EndsWith(".pyo", [StringComparison]::OrdinalIgnoreCase)
}

function Test-DeploymentTree {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$DeploymentRoot,
        [Parameter(Mandatory = $true)][string[]]$TrackedFiles
    )

    $issues = [Collections.Generic.List[string]]::new()
    $expected = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)

    if (-not (Test-Path -LiteralPath $DeploymentRoot -PathType Container)) {
        $issues.Add("MISSING_DEPLOYMENT_ROOT: $DeploymentRoot")
        return $issues.ToArray()
    }

    foreach ($relative in $TrackedFiles) {
        $null = $expected.Add($relative.Replace("\", "/"))
        $sourceFile = Join-Path $RepositoryRoot ($relative.Replace("/", "\"))
        $deployedFile = Join-Path $DeploymentRoot ($relative.Replace("/", "\"))
        if (-not (Test-Path -LiteralPath $deployedFile -PathType Leaf)) {
            $issues.Add("MISSING: $relative")
            continue
        }
        $sourceHash = (Get-FileHash -LiteralPath $sourceFile -Algorithm SHA256).Hash
        $deployedHash = (Get-FileHash -LiteralPath $deployedFile -Algorithm SHA256).Hash
        if ($sourceHash -ne $deployedHash) {
            $issues.Add("CHANGED: $relative")
        }
    }

    foreach ($file in Get-ChildItem -LiteralPath $DeploymentRoot -Force -File -Recurse) {
        $relative = Get-RelativeFilePath -Root $DeploymentRoot -FullName $file.FullName
        if (Test-IsRuntimeCachePath -RelativePath $relative) {
            continue
        }
        if (-not $expected.Contains($relative)) {
            $issues.Add("EXTRA: $relative")
        }
    }

    return $issues.ToArray()
}

function Assert-SafeTemporaryPath {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$ExpectedParent,
        [Parameter(Mandatory = $true)][string]$ExpectedPrefix
    )

    $candidateFull = Get-NormalizedFullPath $Candidate
    $parentFull = Get-NormalizedFullPath (Split-Path -Parent $candidateFull)
    $leaf = Split-Path -Leaf $candidateFull
    if (-not $parentFull.Equals($ExpectedParent, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Temporary path escaped custom_nodes: $candidateFull"
    }
    if (-not $leaf.StartsWith($ExpectedPrefix, [StringComparison]::Ordinal)) {
        throw "Unexpected temporary path: $candidateFull"
    }
}

$source = Get-NormalizedFullPath (Resolve-Path -LiteralPath $SourceRoot).Path
$comfy = Get-NormalizedFullPath (Resolve-Path -LiteralPath $ComfyUIRoot).Path
$customNodes = Get-NormalizedFullPath (Join-Path $comfy "custom_nodes")
$target = Get-NormalizedFullPath (Join-Path $customNodes "raylight")

if ($source.Equals($target, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Source repository and deployment target must be different directories."
}
if (-not (Test-Path -LiteralPath $customNodes -PathType Container)) {
    throw "ComfyUI custom_nodes directory does not exist: $customNodes"
}

$tracked = @(Get-TrackedFiles -RepositoryRoot $source)

if ($VerifyOnly) {
    $issues = @(Test-DeploymentTree -RepositoryRoot $source -DeploymentRoot $target -TrackedFiles $tracked)
    if ($issues.Count -gt 0) {
        $issues | ForEach-Object { Write-Output $_ }
        throw "Deployment drift detected ($($issues.Count) issue(s))."
    }
    Write-Output "DEPLOYMENT_VERIFIED files=$($tracked.Count) target=$target"
    return
}

$token = [Guid]::NewGuid().ToString("N")
$stage = Join-Path $customNodes ".raylight.deploy.$token"
$backup = Join-Path $customNodes ".raylight.backup.$token"
Assert-SafeTemporaryPath -Candidate $stage -ExpectedParent $customNodes -ExpectedPrefix ".raylight.deploy."
Assert-SafeTemporaryPath -Candidate $backup -ExpectedParent $customNodes -ExpectedPrefix ".raylight.backup."

$oldTreeMoved = $false
try {
    New-Item -ItemType Directory -Path $stage | Out-Null
    foreach ($relative in $tracked) {
        $sourceFile = Join-Path $source ($relative.Replace("/", "\"))
        if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
            throw "Tracked source file is missing: $relative"
        }
        $stageFile = Join-Path $stage ($relative.Replace("/", "\"))
        $stageParent = Split-Path -Parent $stageFile
        if (-not (Test-Path -LiteralPath $stageParent -PathType Container)) {
            New-Item -ItemType Directory -Force -Path $stageParent | Out-Null
        }
        Copy-Item -LiteralPath $sourceFile -Destination $stageFile
    }

    $stageIssues = @(Test-DeploymentTree -RepositoryRoot $source -DeploymentRoot $stage -TrackedFiles $tracked)
    if ($stageIssues.Count -gt 0) {
        throw "Staged deployment verification failed: $($stageIssues -join '; ')"
    }

    if (Test-Path -LiteralPath $target) {
        [IO.Directory]::Move($target, $backup)
        $oldTreeMoved = $true
    }
    [IO.Directory]::Move($stage, $target)

    $deployedIssues = @(Test-DeploymentTree -RepositoryRoot $source -DeploymentRoot $target -TrackedFiles $tracked)
    if ($deployedIssues.Count -gt 0) {
        throw "Final deployment verification failed: $($deployedIssues -join '; ')"
    }

    if ($oldTreeMoved -and (Test-Path -LiteralPath $backup)) {
        Remove-Item -LiteralPath $backup -Recurse -Force
        $oldTreeMoved = $false
    }
    Write-Output "DEPLOYMENT_OK files=$($tracked.Count) target=$target"
}
catch {
    $failure = $_
    if ($oldTreeMoved -and (Test-Path -LiteralPath $backup)) {
        if (Test-Path -LiteralPath $target) {
            Remove-Item -LiteralPath $target -Recurse -Force
        }
        [IO.Directory]::Move($backup, $target)
        $oldTreeMoved = $false
    }
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
    throw $failure
}
