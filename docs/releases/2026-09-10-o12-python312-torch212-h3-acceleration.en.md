# 1.14.0 / O12: Python 3.12, Torch 2.12, and the H3 accelerator bridge

Version 1.14.0 moves the preferred runtime to Python 3.12.10 and PyTorch 2.12.0+cu126. Raylight can now replay optional TE-Speed 3.3 and Sol-Attn V100 MODEL patches inside both real Director/FSDP workers without modifying ComfyUI, MiniMaxH3Director, or either external node.

## Main changes

- Launch and verification scripts prefer `Python312` and retain `Python310` as a fallback.
- Added `requirements-windows-v100-torch212.txt`; the Torch 2.7 lock file remains available for rollback.
- Main-process MODEL patches are normalized to plain data and reconstructed inside both Ray workers. Closed-source objects and callables are never serialized through Ray.
- A weightless exact-type view bridges TE-Speed 3.3's strict MiniMax class check to the dynamic composable-FSDP subclass without duplicating weights.
- Sol-Attn executes after Ulysses all-to-all on each rank's local sequence. Raylight retains final control of its validated FP16 safety path.
- Fixed a Torch 2.12 multi-chunk CUDA IPC event-reuse race and Windows xFuser subgroup Gloo routing.
- Added eight matched REF2VA workflows covering 4/8-step Base, TE, Sol, and TE+Sol variants.

## Measured performance

Matched warm runs at 864x480, 124 frames, 24 fps, and about 5.17 seconds:

| Steps | Base | TE | Sol | TE+Sol |
|---:|---:|---:|---:|---:|
| 4-step end to end | 142.04 s | 149.27 s | 136.49 s | **130.67 s** |
| 4-step worker sampling | 85.23 s | 88.88 s | **69.28 s** | 73.32 s |
| 8-step end to end | 228.86 s | 228.49 s | 210.54 s | **200.05 s** |
| 8-step worker sampling | 170.90 s | 162.52 s | **138.06 s** | 138.13 s |

Sol provides the repeatable sampling gain, about 19% in this matrix. TE does not show a stable standalone benefit on these short distilled trajectories. The combined runs have the lowest observed end-to-end times, but their worker sampling is not faster than Sol alone. Detailed cold/warm, resource, media, and three-run lifecycle evidence is in the [Chinese O12 test report](../testing/minimax-h3/O12_PYTHON312_H3_ACCELERATION_2026-09.zh-CN.md).

## Compatibility and boundaries

- Acceleration is opt-in. Existing JSON files and the Base path are unchanged, and missing external packages do not affect workflows that do not use them.
- External nodes are not bundled. Install compatible builds of [TE-Speed-MiniMaxH3-OSS](https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS) and [ComfyUI-MiniMaxH3-SolAttn-V100](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100), including the V100/Python 3.12 kernel required by Sol-Attn.
- Sol is approximate sparse attention, so it is not pixel-identical to dense Base. TE-only, Sol-only, and combined behavior is activated only by explicit nodes.
- PyTorch 2.8/2.9 Windows Gloo cannot provide this project's control plane. O12 therefore uses the real-machine validated Torch 2.12 runtime instead of forcing the originally proposed 2.8 build.
