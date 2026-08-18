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
| V100 compute policy | Complete | Default mode keeps FP8 storage with FP32 compute; opt-in `fp16_h3_safe` keeps FP32 numerical islands and runs the dominant attention/MLP branches in FP16 |
| Runtime I2V FSDP preflight | Complete | Both ranks register 684 FSDP wrappers; CUDA P2P health is about 58.96 GiB/s per direction |
| I2V smoke validation | Complete | 448x448, 39-frame, 12-step cold run passes in 214.93 s; tensors are finite and rank outputs match exactly |
| REF2VA smoke validation | Complete | 608x352, 56-frame, 12-step cold run passes in 268.85 s; preprocessing cleanup barrier prevents rank-0 startup OOM |
| I2V full validation | Complete | 640x640, 56-frame, 20-step run passes in 642.01 s with FSDP CPU offload |
| REF2VA full validation | Complete | 864x480, 124-frame, 20-step cold run passes in 1324.98 s with FSDP CPU offload |
| Checkpoint lifecycle validation | Complete | Different checkpoints recycle both workers and reclaim OS commit; unchanged checkpoints reuse the same PIDs and FSDP shards |
| INT8 comparison | Pending | Run only after FP8 correctness |
| Repeatable cold/warm benchmark | Complete | I2V and REF2VA each pass one cold plus two warm runs with stable workers, resources and media |
| Speed/quality variants | Complete | Official FL2V 8-step and REF2V 4-step Turbo LoRAs pass; warm end-to-end time improves by 54.9% and 58.9% against 20-step baselines |
| Full-specification Turbo release validation | Complete | Full I2V and REF2VA Turbo presets pass; cold end-to-end time improves by 39.6% and 68.0% against full 20-step baselines |
| User-loadable Turbo release candidate | Complete | Dedicated Turbo8/Turbo4 GUI workflows pass live node/API conversion and exact generated-artifact checks |
| O6 matched 1120x768 baseline | Complete | I2V Turbo8 and REF2VA Turbo4 cold runs pass at 124 frames/24 FPS; stage, resource, rank, P2P and media gates are locked |
| O6 safe-FP16 implementation | Functional/quality complete; performance gate pending | I2V reaches 3.299x and REF2VA 3.416x; best optional REF2VA result is 3.714x. Exact two-rank finite output and media gates pass; neither workflow has met the initial 4x gate |
| O7 LTX safe-FP16 research | Planned | Separate model-specific precision-island investigation after O6; global LTX FP16 remains rejected |

Accepted runs: [I2V smoke validation](I2V_SMOKE_2026-08-17.md), [REF2VA smoke validation](REF2VA_SMOKE_2026-08-17.md), [full I2V/REF2VA validation](FULL_WORKFLOWS_2026-08-17.md), [checkpoint recycling validation](CHECKPOINT_RECYCLING_2026-08-17.md), [repeatable cold/warm benchmark](COLD_WARM_BENCHMARK_2026-08-17.md), [speed/quality variants](SPEED_QUALITY_VARIANTS_2026-08-17.md), [full Turbo release validation](FULL_TURBO_RELEASE_2026-08-17.md), [Turbo workflow usage](TURBO_WORKFLOW_USAGE.md), and the [O6 matched FP32/safe-FP16 report](SAFE_FP16_FSDP_2026-08.md). The [Chinese](../../releases/2026-08-18-o6-minimax-h3-safe-fp16.zh-CN.md) and [English](../../releases/2026-08-18-o6-minimax-h3-safe-fp16.en.md) upgrade records contain the complete implemented/rejected experiment matrix. The O6 design and baseline protocol are in [the safe-FP16 FSDP plan](../../superpowers/plans/2026-08-18-minimax-h3-safe-fp16-fsdp.md). MiniMax H3 requires `--reserve-vram 2` in addition to `--disable-cuda-malloc` on this stack. Full workflows default to FSDP CPU offload. The original smoke validations kept CPU offload disabled; the O2 and O3 repeatability benchmarks intentionally enabled it to match the reusable full-workflow operating mode. REF2VA also requires the preprocessing-to-worker cleanup barrier recorded in the REF2VA report.

## Evidence policy

The Markdown summary records settings, stage timings, peak resource values and conclusions. Long logs, telemetry series and generated media remain local artifacts and are referenced rather than embedded in the repository.

A run is not accepted solely because ComfyUI reaches 100%. Video, audio, finite tensors, both-rank progress, actual FSDP sharding and CUDA P2P transport must all pass.
