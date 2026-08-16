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
- 2? Tesla V100-SXM2-16GB in TCC mode
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
| Environment and workflow audit | Complete | Required core/Raylight nodes are present in source |
| FP8 model inventory | Complete | Five-file baseline manifest totals 63,416,617,071 bytes |
| I2V asset acquisition | Pending | Download only after manifest/dry-run validation |
| I2V FSDP preflight | Pending | Requires downloaded FL2VA FP8 model |
| I2V smoke/full validation | Pending | Reduced workload precedes upstream 0.4 MP run |
| REF2VA validation | Pending | First run duplicates the supplied green-robot image |
| INT8 comparison | Pending | Run only after FP8 correctness |
| Final optimization | Pending | Compare base and official Turbo LoRA variants |

## Evidence policy

The Markdown summary records settings, stage timings, peak resource values and conclusions. Long logs, telemetry series and generated media remain local artifacts and are referenced rather than embedded in the repository.

A run is not accepted solely because ComfyUI reaches 100%. Video, audio, finite tensors, both-rank progress, actual FSDP sharding and CUDA P2P transport must all pass.
