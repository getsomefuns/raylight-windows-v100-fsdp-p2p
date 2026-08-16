# MiniMax H3 Validation Log

This directory contains the maintained validation summary for MiniMax H3 on native Windows with two Tesla V100-SXM2-16GB GPUs.

## Scope

- Raylight Windows CUDA P2P/FSDP branch
- MiniMax H3 FL2VA/I2V and REF2VA
- FP8-scaled compatibility baseline
- INT8 ConvRot comparison after baseline correctness
- cold/warm performance, GPU participation, RAM/commit and output checks

## Fixed baseline

- Windows 23H2
- NVIDIA driver 577.00
- 2x Tesla V100-SXM2-16GB in TCC mode
- Python 3.10.11
- PyTorch 2.7.0+cu126
- xformers 0.0.30
- Ray 2.57.0
- yunchang 0.6.4
- comfy-kitchen 0.2.30
- ComfyUI launched with `--disable-cuda-malloc`

## Current status

| Stage | Status | Result |
|---|---|---|
| Environment and workflow audit | Complete | All 19 executable I2V node types import in ComfyUI 0.31.0 |
| Original sample mapping | Complete | Spear/portal image maps to I2V; green robots map to both REF2VA reference inputs; SHA-256 matches upstream assets |
| FP8 model inventory | Complete | Five-file baseline manifest totals 63,416,617,071 bytes |
| I2V asset acquisition | Complete | All required files pass exact-size and safetensors-header validation |
| Static FSDP/quant preflight | Complete | 5 MiniMax contracts and 31 related FP8/FSDP/P2P regressions pass |
| V100 attention preflight | Complete | TORCH_EFFICIENT produces finite FP16 output on both sm_70 GPUs; sync Ulysses is enabled |
| V100 compute policy | Complete | FP8 is storage-only; MiniMax diffusion compute manual-casts to FP32, not FP16/BF16/FP8 |
| Runtime I2V FSDP preflight | Complete | Both ranks register 684 FSDP wrappers; CUDA P2P health is about 58.96 GiB/s per direction |
| I2V smoke validation | Complete | 448x448, 39-frame, 12-step cold run passes in 214.93 s; tensors are finite and rank outputs match exactly |
| REF2VA smoke validation | Complete | 608x352, 56-frame, 12-step cold run passes in 268.85 s; preprocessing cleanup barrier prevents rank-0 startup OOM |
| I2V full validation | Pending | First reduce the roughly 16.1 GiB smoke sampling peak, then run the upstream 0.4 MP, 2-second, 20-step profile |
| REF2VA full validation | Pending | First reduce the roughly 16.2 GiB smoke sampling peak, then scale toward 0.4 MP and 5 seconds |
| INT8 comparison | Pending | Run only after FP8 correctness |
| Final optimization | Pending | Compare base and official Turbo LoRA variants |

Accepted runs: [I2V smoke validation](I2V_SMOKE_2026-08-17.md) and [REF2VA smoke validation](REF2VA_SMOKE_2026-08-17.md). MiniMax H3 requires `--reserve-vram 2` in addition to `--disable-cuda-malloc` on this stack. REF2VA also requires the preprocessing-to-worker cleanup barrier recorded in the REF2VA report.

## Evidence policy

The Markdown summary records settings, stage timings, peak resource values and conclusions. Long logs, telemetry series and generated media remain local artifacts and are referenced rather than embedded in the repository.

A run is not accepted solely because ComfyUI reaches 100%. Video, audio, finite tensors, both-rank progress, actual FSDP sharding and CUDA P2P transport must all pass.
