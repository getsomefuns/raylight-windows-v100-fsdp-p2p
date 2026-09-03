[CmdletBinding()]
param(
    [string]$SourceRoot = "",
    [string]$DetoursRoot = "",
    [string]$WindowsSdkRoot = "",
    [string]$OutputRoot = "",
    [string]$PythonPath = "",
    [switch]$PrintConfiguration
)

$ErrorActionPreference = "Stop"

$aimdoTag = "v0.4.13"
$aimdoCommit = "3f4259a502e7df7508dca7c30bd07eb936edf5a6"
$detoursTag = "v4.0.1"
$detoursCommit = "e4bfd6b03e50de46b47abfbd1e46b384f0c5f833"
$detoursBuildScope = "library-only"
$detoursTargetProcessor = "X64"
$windowsSdkNugetVersion = "10.0.26100.1"
$localVersion = "0.4.13+tcc1"
$nvmlPressureHeadroomBytes = 512MB
$pythonBuildDependencies = [ordered]@{
    build = "1.6.0"
    packaging = "25.0"
    setuptools = "80.9.0"
    "setuptools-scm" = "8.3.1"
    tomli = "2.3.0"
    wheel = "0.45.1"
}

$configuration = [ordered]@{
    aimdo_commit = $aimdoCommit
    aimdo_tag = $aimdoTag
    detours_commit = $detoursCommit
    detours_build_scope = $detoursBuildScope
    detours_tag = $detoursTag
    detours_target_processor = $detoursTargetProcessor
    local_version = $localVersion
    nvml_pressure_headroom_bytes = [long]$nvmlPressureHeadroomBytes
    python_build_isolation = $false
    python_build_dependencies = $pythonBuildDependencies
    windows_sdk_nuget_version = $windowsSdkNugetVersion
}

if ($PrintConfiguration) {
    $configuration | ConvertTo-Json -Compress
    return
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$environmentRoot = Split-Path -Parent $repoRoot
if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Join-Path $environmentRoot "runtime-sources\comfy-aimdo-v0.4.13-tcc1"
}
if ([string]::IsNullOrWhiteSpace($DetoursRoot)) {
    $DetoursRoot = Join-Path $environmentRoot "runtime-sources\Detours"
}
if ([string]::IsNullOrWhiteSpace($WindowsSdkRoot)) {
    $WindowsSdkRoot = Join-Path $environmentRoot "runtime-sources\windows-sdk-nuget-10.0.26100.1"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $environmentRoot "runtime-builds\comfy-aimdo-v0.4.13-tcc1"
}
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $environmentRoot "runtime-candidates\Python310-torch2.8.0-cu128-aimdo-tcc1\python.exe"
}

$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
$DetoursRoot = (Resolve-Path -LiteralPath $DetoursRoot).Path
$WindowsSdkRoot = (Resolve-Path -LiteralPath $WindowsSdkRoot).Path
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$patchPath = (Resolve-Path -LiteralPath (Join-Path $repoRoot "third_party\comfy-aimdo-tcc\patches\v0.4.13-tcc-nvml.patch")).Path
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path

$buildVersions = (& $PythonPath -c "import importlib.metadata as m; print('|'.join(m.version(n) for n in ('build', 'packaging', 'setuptools', 'setuptools-scm', 'tomli', 'wheel')))").Trim()
$expectedBuildVersions = "$($pythonBuildDependencies.build)|$($pythonBuildDependencies.packaging)|$($pythonBuildDependencies.setuptools)|$($pythonBuildDependencies.'setuptools-scm')|$($pythonBuildDependencies.tomli)|$($pythonBuildDependencies.wheel)"
if ($buildVersions -ne $expectedBuildVersions) {
    throw "Python build dependency mismatch. Expected $expectedBuildVersions, got $buildVersions"
}

function Assert-GitCommit {
    param([string]$Path, [string]$Expected)
    $actual = (& git -C $Path rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actual -ne $Expected) {
        throw "Source pin mismatch for $Path. Expected $Expected, got $actual"
    }
}

Assert-GitCommit -Path $SourceRoot -Expected $aimdoCommit
Assert-GitCommit -Path $DetoursRoot -Expected $detoursCommit

$sdkIncludeRoot = Join-Path $WindowsSdkRoot "cpp\c\Include\10.0.26100.0"
$sdkUmInclude = Join-Path $sdkIncludeRoot "um"
$sdkSharedInclude = Join-Path $sdkIncludeRoot "shared"
$sdkUcrtInclude = Join-Path $sdkIncludeRoot "ucrt"
$sdkWinrtInclude = Join-Path $sdkIncludeRoot "winrt"
$sdkUmLib = Join-Path $WindowsSdkRoot "x64\c\um\x64"
$sdkUcrtLib = Join-Path $WindowsSdkRoot "x64\c\ucrt\x64"
if (-not (Test-Path -LiteralPath (Join-Path $sdkUmInclude "Windows.h") -PathType Leaf)) {
    throw "Local Windows SDK is incomplete: Windows.h is missing"
}
if (-not (Test-Path -LiteralPath (Join-Path $sdkUmLib "OneCore.Lib") -PathType Leaf)) {
    throw "Local Windows SDK is incomplete: OneCore.Lib is missing"
}

& git -C $SourceRoot apply --check $patchPath 2>$null
if ($LASTEXITCODE -eq 0) {
    & git -C $SourceRoot apply $patchPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to apply TCC patch"
    }
} else {
    & git -C $SourceRoot apply --reverse --check $patchPath 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "TCC patch is neither cleanly applicable nor already applied"
    }
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw "Visual Studio vswhere.exe was not found"
}
$vsPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if (-not $vsPath) {
    throw "Visual Studio C++ Build Tools were not found"
}
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path -LiteralPath $vcvars -PathType Leaf)) {
    throw "vcvars64.bat was not found: $vcvars"
}

$cmdPath = Join-Path $OutputRoot "build-native.cmd"
$wheelRoot = Join-Path $OutputRoot "wheelhouse"
$cmdLines = @(
    "@echo off",
    "setlocal",
    "call ""$vcvars"" >nul",
    "if errorlevel 1 exit /b 10",
    "set INCLUDE=$sdkUmInclude;$sdkSharedInclude;$sdkUcrtInclude;$sdkWinrtInclude;%INCLUDE%",
    "set LIB=$sdkUmLib;$sdkUcrtLib;%LIB%",
    "set DETOURS_TARGET_PROCESSOR=$detoursTargetProcessor",
    "pushd ""$DetoursRoot\src""",
    "nmake",
    "if errorlevel 1 exit /b 20",
    "popd",
    "pushd ""$SourceRoot""",
    "cl.exe /LD /O2 /DAIMDO_CUDA src\*.c src-cuda\dispatch.c src-win\*.c /Isrc /Isrc-win /I""$DetoursRoot\include"" /FIcompiler.h /Fe:comfy_aimdo\aimdo.dll /link /LIBPATH:""$DetoursRoot\lib.X64"" dxgi.lib dxguid.lib detours.lib onecore.lib",
    "if errorlevel 1 exit /b 30",
    "set SETUPTOOLS_SCM_PRETEND_VERSION=$localVersion",
    """$PythonPath"" -m build --wheel --no-isolation --outdir ""$wheelRoot""",
    "if errorlevel 1 exit /b 40",
    "popd"
)
[System.IO.File]::WriteAllLines($cmdPath, $cmdLines, [System.Text.Encoding]::ASCII)

& $env:ComSpec /d /c $cmdPath
if ($LASTEXITCODE -ne 0) {
    throw "Native build failed with exit code $LASTEXITCODE"
}

$dllSource = Join-Path $SourceRoot "comfy_aimdo\aimdo.dll"
$wheel = Get-ChildItem -LiteralPath $wheelRoot -File -Filter "*.whl" |
    Where-Object { $_.Name -match "0\.4\.13\+tcc1" } |
    Select-Object -First 1
if (-not (Test-Path -LiteralPath $dllSource -PathType Leaf)) {
    throw "Built AIMDO DLL was not found"
}
if (-not $wheel) {
    throw "Built 0.4.13+tcc1 wheel was not found"
}

$dllArtifact = Join-Path $OutputRoot "aimdo.dll"
$patchArtifact = Join-Path $OutputRoot "v0.4.13-tcc-nvml.patch"
Copy-Item -LiteralPath $dllSource -Destination $dllArtifact -Force
Copy-Item -LiteralPath $patchPath -Destination $patchArtifact -Force

function New-ArtifactEntry {
    param([string]$Path)
    return [ordered]@{
        path = [System.IO.Path]::GetRelativePath($OutputRoot, $Path)
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
    }
}

$manifest = [ordered]@{
    schema = 1
    version = $localVersion
    pressure_headroom_bytes = [long]$nvmlPressureHeadroomBytes
    upstream = [ordered]@{
        repository = "https://github.com/Comfy-Org/comfy-aimdo.git"
        tag = $aimdoTag
        commit = $aimdoCommit
    }
    detours = [ordered]@{
        repository = "https://github.com/microsoft/Detours.git"
        tag = $detoursTag
        commit = $detoursCommit
    }
    windows_sdk = [ordered]@{
        package = "Microsoft.Windows.SDK.CPP"
        architecture_package = "Microsoft.Windows.SDK.CPP.x64"
        version = $windowsSdkNugetVersion
    }
    tools = [ordered]@{
        python = (& $PythonPath --version 2>&1 | Out-String).Trim()
        python_build_dependencies = $pythonBuildDependencies
        cl = (& $env:ComSpec /d /c """$vcvars"" >nul && cl 2>&1" | Select-Object -First 1 | Out-String).Trim()
    }
    artifacts = [ordered]@{
        patch = New-ArtifactEntry -Path $patchArtifact
        dll = New-ArtifactEntry -Path $dllArtifact
        wheel = New-ArtifactEntry -Path $wheel.FullName
    }
}
$manifestPath = Join-Path $OutputRoot "BUILD-MANIFEST.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "Built wheel: $($wheel.FullName)"
Write-Host "Build manifest: $manifestPath"
