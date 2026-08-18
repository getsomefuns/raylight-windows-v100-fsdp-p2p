# 2026-08-19 Launcher Capacity and Diagnostic Controls

## Goal

Normal use and performance tests now share an explicit, reproducible launch profile, while verbose
`[RAYLIGHT_RANK_DIAG]` / `[RAYLIGHT_P2P_DIAG]` messages no longer dominate normal terminal output.

## User-visible changes

- `scripts/start-comfyui-windows-p2p.ps1` defaults to a 256 MiB P2P staging buffer per GPU.
- `-P2PCapacityMiB` accepts only `128`, `256`, or `512`; the launcher prints all choices and the selected value.
- Detailed Rank/P2P diagnostics are off by default and enabled only with `-EnableDiagnostics`.
- `-ValidateOnly` prints the resolved capacity and diagnostic state without starting ComfyUI.
- The legacy `-P2PCapacityBytes` parameter remains available for existing automation; new commands should use MiB.

## Examples

```powershell
# Default: 256 MiB, diagnostics off
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath $PY

# Explicit choices
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath $PY -P2PCapacityMiB 128
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath $PY -P2PCapacityMiB 256
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath $PY -P2PCapacityMiB 512

# Performance or synchronization diagnostics
.\scripts\start-comfyui-windows-p2p.ps1 `
  -PythonPath $PY `
  -P2PCapacityMiB 256 `
  -EnableDiagnostics
```

Stop ComfyUI and stale Ray workers before changing capacity or diagnostic state, then restart them.

## Capacity semantics

Capacity is the persistent CUDA staging-buffer limit per GPU, not a direct mapping from video duration
or resolution. A larger choice does not inherently improve speed or reserve additional host RAM/pagefile,
but it consumes more resident VRAM per GPU: 256 uses about 128 MiB/GPU more than 128, and 512 uses
about 256 MiB/GPU more than 256. The accepted 1120x768 MiniMax H3 O6 workflow contains a
239,826,944-byte Ulysses remote payload and therefore requires the 256 MiB or 512 MiB choice.

The low-level worker still falls back to 128 MiB when the repository launcher is not used. This retains
library-level compatibility but is not the launcher's current default. Insufficient capacity fails explicitly
instead of silently routing data through host memory.

## Verification scope

Release-profile tests cover the default, all three valid choices, rejection of `384 MiB`, explicit diagnostic
enablement, default diagnostic disablement, and legacy byte-parameter compatibility. This update does not
change the CUDA P2P/FSDP data path, model precision, or workflows.
