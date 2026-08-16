[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$SourceRoot,
    [string]$DestinationRoot = "E:\ComfyUI-py310\ComfyUI\custom_nodes\raylight",
    [switch]$PlanOnly,
    [switch]$Json
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
}

function Invoke-GitLines {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $lines = @(& git.exe -C $Root @Arguments 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed in ${Root}: $($lines -join [Environment]::NewLine)"
    }
    return @($lines | ForEach-Object { [string]$_ })
}

function Resolve-GitRoot {
    param([Parameter(Mandatory = $true)][string]$Root)

    if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
        throw "Repository directory not found: $Root"
    }
    $resolved = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Root).Path)
    $gitRoot = @(Invoke-GitLines -Root $resolved -Arguments @("rev-parse", "--show-toplevel"))[0]
    $gitRoot = [IO.Path]::GetFullPath($gitRoot)
    if (-not [string]::Equals($resolved.TrimEnd("\"), $gitRoot.TrimEnd("\"), [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path must be a Git repository root: $resolved (Git root: $gitRoot)"
    }
    return $gitRoot
}

function Get-RuntimeFiles {
    param([Parameter(Mandatory = $true)][string]$Root)

    $files = Invoke-GitLines -Root $Root -Arguments @(
        "ls-files", "--", "__init__.py", "icon.png", "src"
    )
    return @(
        $files |
            Where-Object {
                $_ -eq "__init__.py" -or
                $_ -eq "icon.png" -or
                ($_.StartsWith("src/") -and $_.EndsWith(".py"))
            } |
            Sort-Object -Unique
    )
}

$source = Resolve-GitRoot -Root $SourceRoot
$destination = Resolve-GitRoot -Root $DestinationRoot
if ([string]::Equals($source, $destination, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Source and destination repositories must be different"
}

$sourceChanges = Invoke-GitLines -Root $source -Arguments @(
    "status", "--porcelain", "--untracked-files=no", "--", "__init__.py", "icon.png", "src"
)
if ($sourceChanges.Count -gt 0) {
    throw "Source runtime files have tracked modifications; commit or restore them before deployment"
}

$destinationChanges = Invoke-GitLines -Root $destination -Arguments @(
    "status", "--porcelain", "--untracked-files=no"
)
if ($destinationChanges.Count -gt 0) {
    throw "Destination repository has tracked modifications; refusing deployment"
}

$sourceFiles = @(Get-RuntimeFiles -Root $source)
if ($sourceFiles.Count -eq 0 -or $sourceFiles -notcontains "__init__.py") {
    throw "Source repository does not look like a Raylight custom node"
}
$destinationFiles = @(Get-RuntimeFiles -Root $destination)
$sourceSet = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$destinationSet = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($file in $sourceFiles) { $null = $sourceSet.Add($file) }
foreach ($file in $destinationFiles) { $null = $destinationSet.Add($file) }

foreach ($file in $sourceFiles) {
    $target = Join-Path $destination ($file -replace "/", "\")
    if ((Test-Path -LiteralPath $target) -and -not $destinationSet.Contains($file)) {
        throw "Untracked destination path would be overwritten: $target"
    }
}

$removeFiles = @($destinationFiles | Where-Object { -not $sourceSet.Contains($_) })
$sourceCommit = @(Invoke-GitLines -Root $source -Arguments @("rev-parse", "HEAD"))[0]
$plan = [ordered]@{
    schema_version = 1
    source_root = $source
    destination_root = $destination
    source_commit = $sourceCommit
    copy_files = @($sourceFiles)
    remove_files = @($removeFiles)
}

if ($PlanOnly) {
    if ($Json) {
        $plan | ConvertTo-Json -Depth 5 -Compress
    }
    else {
        Write-Host "Raylight deployment plan"
        Write-Host "Source commit: $sourceCommit"
        Write-Host "Copy: $($sourceFiles.Count) runtime files"
        Write-Host "Remove: $($removeFiles.Count) stale runtime files"
    }
    exit 0
}

foreach ($file in $sourceFiles) {
    $sourcePath = Join-Path $source ($file -replace "/", "\")
    $targetPath = Join-Path $destination ($file -replace "/", "\")
    if ($PSCmdlet.ShouldProcess($targetPath, "Copy tested Raylight runtime file")) {
        $targetDirectory = Split-Path -Parent $targetPath
        New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
    }
}

foreach ($file in $removeFiles) {
    $targetPath = Join-Path $destination ($file -replace "/", "\")
    if ($PSCmdlet.ShouldProcess($targetPath, "Remove stale tracked Raylight runtime file")) {
        Remove-Item -LiteralPath $targetPath -Force
    }
}

$gitDirectory = @(Invoke-GitLines -Root $destination -Arguments @("rev-parse", "--git-dir"))[0]
if (-not [IO.Path]::IsPathRooted($gitDirectory)) {
    $gitDirectory = Join-Path $destination $gitDirectory
}
$markerPath = Join-Path ([IO.Path]::GetFullPath($gitDirectory)) "raylight-deployed-commit"
if ($PSCmdlet.ShouldProcess($markerPath, "Record deployed source commit")) {
    [IO.File]::WriteAllText($markerPath, "$sourceCommit$([Environment]::NewLine)", [Text.Encoding]::UTF8)
}

Write-Host "Deployed Raylight runtime from $sourceCommit"
$pathspec = @("__init__.py", "icon.png", "src")
$gitOutput = @(& git.exe -C $destination add -A -- @pathspec 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to stage deployed runtime files: $($gitOutput -join [Environment]::NewLine)"
}

& git.exe -C $destination diff --cached --quiet -- @pathspec
$stagedDiffExit = $LASTEXITCODE
if ($stagedDiffExit -eq 1) {
    $gitOutput = @(
        & git.exe -C $destination `
            -c "user.name=Raylight Runtime Deployer" `
            -c "user.email=raylight-runtime@localhost" `
            commit --no-verify `
            -m "deploy: runtime from $sourceCommit" `
            -- @pathspec 2>&1
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to commit deployed runtime files: $($gitOutput -join [Environment]::NewLine)"
    }
}
elseif ($stagedDiffExit -ne 0) {
    throw "Unable to inspect staged runtime deployment (git exit $stagedDiffExit)"
}

$remainingChanges = Invoke-GitLines -Root $destination -Arguments @(
    "status", "--porcelain", "--untracked-files=no"
)
if ($remainingChanges.Count -gt 0) {
    throw "Deployment completed but destination is not clean: $($remainingChanges -join ', ')"
}
