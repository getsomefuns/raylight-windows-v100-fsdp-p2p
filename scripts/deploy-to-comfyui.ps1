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
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][string]$Commit
    )

    $output = @(& git -c "safe.directory=$RepositoryRoot" -C $RepositoryRoot ls-tree -r --name-only $Commit)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to read the immutable Git HEAD manifest from: $RepositoryRoot"
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
    if ($normalized.Equals(".raylight-deployed-commit", [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
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

function Get-ExpectedHashes {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string[]]$TrackedFiles
    )

    $hashes = @{}
    foreach ($relative in $TrackedFiles) {
        $file = Join-Path $Root ($relative.Replace("/", "\"))
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
            throw "Immutable archive is missing a tracked file: $relative"
        }
        $hashes[$relative.Replace("\", "/")] = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash
    }
    return $hashes
}

function Test-DeploymentTree {
    param(
        [Parameter(Mandatory = $true)][string]$DeploymentRoot,
        [Parameter(Mandatory = $true)][string[]]$TrackedFiles,
        [Parameter(Mandatory = $true)][System.Collections.IDictionary]$ExpectedHashes,
        [Parameter(Mandatory = $true)][string]$SourceCommit
    )

    $issues = [Collections.Generic.List[string]]::new()
    $expected = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)

    if (-not (Test-Path -LiteralPath $DeploymentRoot -PathType Container)) {
        $issues.Add("MISSING_DEPLOYMENT_ROOT: $DeploymentRoot")
        return $issues.ToArray()
    }

    $metadataPath = Join-Path $DeploymentRoot ".raylight-deployed-commit"
    if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
        $issues.Add("MISSING_DEPLOYMENT_METADATA: .raylight-deployed-commit")
    }
    elseif ((Get-Content -LiteralPath $metadataPath -Raw).Trim() -ne $SourceCommit) {
        $issues.Add("CHANGED_DEPLOYMENT_METADATA: .raylight-deployed-commit")
    }

    foreach ($relative in $TrackedFiles) {
        $normalized = $relative.Replace("\", "/")
        $null = $expected.Add($normalized)
        $deployedFile = Join-Path $DeploymentRoot ($relative.Replace("/", "\"))
        if (-not (Test-Path -LiteralPath $deployedFile -PathType Leaf)) {
            $issues.Add("MISSING: $relative")
            continue
        }
        $deployedHash = (Get-FileHash -LiteralPath $deployedFile -Algorithm SHA256).Hash
        if ($ExpectedHashes[$normalized] -ne $deployedHash) {
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

$sourceCommit = @(& git -c "safe.directory=$source" -C $source rev-parse HEAD)[0]
if ($LASTEXITCODE -ne 0 -or $sourceCommit -notmatch '^[0-9a-f]{40}$') {
    throw "Unable to resolve the source repository commit: $source"
}
$sourceChanges = @(& git -c "safe.directory=$source" -C $source status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect tracked source changes: $source"
}
if ($sourceChanges.Count -gt 0) {
    throw "Source repository has tracked modifications; commit them before deployment."
}
$tracked = @(Get-TrackedFiles -RepositoryRoot $source -Commit $sourceCommit)

$token = [Guid]::NewGuid().ToString("N")
$stage = Join-Path $customNodes ".raylight.deploy.$token"
$backup = Join-Path $customNodes ".raylight.backup.$token"
$archive = Join-Path $customNodes ".raylight.archive.$token.tar"
Assert-SafeTemporaryPath -Candidate $stage -ExpectedParent $customNodes -ExpectedPrefix ".raylight.deploy."
Assert-SafeTemporaryPath -Candidate $backup -ExpectedParent $customNodes -ExpectedPrefix ".raylight.backup."
Assert-SafeTemporaryPath -Candidate $archive -ExpectedParent $customNodes -ExpectedPrefix ".raylight.archive."

$oldTreeMoved = $false
$targetInstalled = $false
$deploymentCommitted = $false
try {
    New-Item -ItemType Directory -Path $stage | Out-Null

    # Materialize one immutable Git object tree: git archive HEAD, never mutable worktree files.
    & git -c "safe.directory=$source" -C $source archive --format=tar "--output=$archive" $sourceCommit
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $archive -PathType Leaf)) {
        throw "Unable to create immutable source archive for commit $sourceCommit"
    }
    & tar.exe -xf $archive -C $stage
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to extract immutable source archive for commit $sourceCommit"
    }

    [IO.File]::WriteAllText(
        (Join-Path $stage ".raylight-deployed-commit"),
        "$sourceCommit$([Environment]::NewLine)",
        [Text.UTF8Encoding]::new($false)
    )

    $expectedHashes = Get-ExpectedHashes -Root $stage -TrackedFiles $tracked
    $stageIssues = @(Test-DeploymentTree -DeploymentRoot $stage -TrackedFiles $tracked -ExpectedHashes $expectedHashes -SourceCommit $sourceCommit)
    if ($stageIssues.Count -gt 0) {
        throw "Staged deployment verification failed: $($stageIssues -join '; ')"
    }

    if ($VerifyOnly) {
        $issues = @(Test-DeploymentTree -DeploymentRoot $target -TrackedFiles $tracked -ExpectedHashes $expectedHashes -SourceCommit $sourceCommit)
        if ($issues.Count -gt 0) {
            $issues | ForEach-Object { Write-Output $_ }
            throw "Deployment drift detected ($($issues.Count) issue(s))."
        }
        Write-Output "DEPLOYMENT_VERIFIED files=$($tracked.Count) target=$target"
        return
    }

    if (Test-Path -LiteralPath $target) {
        [IO.Directory]::Move($target, $backup)
        $oldTreeMoved = $true
    }
    [IO.Directory]::Move($stage, $target)
    $targetInstalled = $true

    $deployedIssues = @(Test-DeploymentTree -DeploymentRoot $target -TrackedFiles $tracked -ExpectedHashes $expectedHashes -SourceCommit $sourceCommit)
    if ($deployedIssues.Count -gt 0) {
        throw "Final deployment verification failed: $($deployedIssues -join '; ')"
    }
    $deploymentCommitted = $true

    if ($oldTreeMoved -and (Test-Path -LiteralPath $backup)) {
        try {
            Remove-Item -LiteralPath $backup -Recurse -Force
        }
        catch {
            Write-Warning "Deployment succeeded but backup cleanup failed: $backup ($($_.Exception.Message))"
        }
        $oldTreeMoved = $false
    }
    Write-Output "DEPLOYMENT_OK files=$($tracked.Count) target=$target"
}
catch {
    $failure = $_
    if (-not $deploymentCommitted) {
        if ($targetInstalled -and (Test-Path -LiteralPath $target)) {
            Remove-Item -LiteralPath $target -Recurse -Force
            $targetInstalled = $false
        }
        if ($oldTreeMoved -and (Test-Path -LiteralPath $backup)) {
            [IO.Directory]::Move($backup, $target)
            $oldTreeMoved = $false
        }
    }
    throw $failure
}
finally {
    if (Test-Path -LiteralPath $stage) {
        try {
            Remove-Item -LiteralPath $stage -Recurse -Force
        }
        catch {
            Write-Warning "Temporary stage cleanup failed: $stage ($($_.Exception.Message))"
        }
    }
    if (Test-Path -LiteralPath $archive) {
        try {
            Remove-Item -LiteralPath $archive -Force
        }
        catch {
            Write-Warning "Temporary archive cleanup failed: $archive ($($_.Exception.Message))"
        }
    }
}
