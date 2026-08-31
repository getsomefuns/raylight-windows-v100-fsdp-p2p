param(
    [string]$Workflow = "E:\ComfyUI-py310\ComfyUI\user\default\workflows\MMH3Director.json",
    [ValidateRange(1, 20)][int]$Sessions = 3,
    [ValidateRange(1024, 65535)][int]$Port = 8193,
    [ValidateRange(60, 7200)][int]$TimeoutSeconds = 1800,
    [string]$PythonPath = "E:\ComfyUI-py310\Python310\python.exe",
    [string]$ResultRoot = ""
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "minimax-h3\benchmark_text_encoder_tp2.py"
$arguments = @(
    $scriptPath,
    "--workflow", $Workflow,
    "--sessions", $Sessions,
    "--port", $Port,
    "--timeout", $TimeoutSeconds
)
if ($ResultRoot) {
    $arguments += @("--result-root", $ResultRoot)
}

& $PythonPath @arguments
exit $LASTEXITCODE
