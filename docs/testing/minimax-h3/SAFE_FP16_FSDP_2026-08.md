# MiniMax H3 O6 Matched FP32 and Safe-FP16 Validation — 2026-08-18

[简体中文](SAFE_FP16_FSDP_2026-08.zh-CN.md) | [English](SAFE_FP16_FSDP_2026-08.md)

## Validation target

Lock the local comparison denominator for O6 safe-FP16 work. No external or projected timing is used. Both runs use the current accepted FP32-compute implementation, identical 1120x768 geometry, 124 frames at 24 FPS, the accepted O5 prompts/inputs, FSDP CPU offload, two V100 ranks and Windows CUDA P2P.

## Fixed conditions

| Condition | I2V | REF2VA |
|---|---|---|
| Variant | FL2V Turbo8 | REF2V Turbo4 |
| Startup | independent cold start from idle GPUs | independent cold start from idle GPUs |
| Geometry | 1120x768, 124 frames, 24 FPS | 1120x768, 124 frames, 24 FPS |
| Precision policy | FP8-scaled storage, V100 FP32 diffusion compute | FP8-scaled storage, V100 FP32 diffusion compute |
| Parallel mode | 2 ranks, FSDP=true, CPU offload=true | 2 ranks, FSDP=true, CPU offload=true |
| Ulysses / Ring / CFG / DP | 2 / 1 / 1 / 1 | 2 / 1 / 1 / 1 |
| P2P capacity | explicitly 256 MiB per rank | explicitly 256 MiB per rank |
| Source/deployment commit | `394d1cffb668fd62d87ae632ef86e28e4d9c04b4` | same |

The release default remains 128 MiB. A preliminary I2V attempt proved that this exact geometry needs a larger buffer: its Ulysses remote payload was 239,826,944 bytes and was correctly rejected by the 134,217,728-byte default. The benchmark-only 256 MiB setting covers that measured payload without changing the release default.

## Stage results

All values are seconds unless stated otherwise.

| Metric | I2V Turbo8 | REF2VA Turbo4 |
|---|---:|---:|
| End-to-end wall time | 1463.67 | 932.03 |
| Ray initialization | 36.20 | 37.99 |
| Model-loader nodes | 20.98 | 20.98 |
| ComfyUI preprocessing | 33.53 | 42.84 |
| Max worker model-to-GPU | 4.66 | 1.29 |
| Sampler node | 1318.63 | 774.78 |
| Max-rank sampling interval | 1285.76 | 740.81 |
| Aggregate sampling / step | 160.72 s/it | 185.20 s/it |
| First observed tqdm value | 132.0 s/it | 98.1 s/it |
| Final tqdm rolling value | 153.0 s/it | 149.0 s/it |
| Rank sampling spread | 4.52 | 1.23 |
| VAE decode | 42.81 | 43.88 |
| Video creation | 0.03 | 0.02 |
| Video save | 9.18 | 9.17 |

The O6 performance denominator is the max-rank sampling interval divided by the exact step count. This includes all measured sampling work and is more stable than selecting one transient tqdm line.

## Resource and communication results

| Metric | I2V Turbo8 | REF2VA Turbo4 |
|---|---:|---:|
| GPU0 peak VRAM | 16,162 MiB | 16,237 MiB |
| GPU1 peak VRAM | 16,214 MiB | 16,137 MiB |
| GPU0/GPU1 peak utilization | 100% / 100% | 100% / 100% |
| GPU0/GPU1 peak power | 366.31 / 369.20 W | 366.52 / 365.98 W |
| Peak physical memory used | 65,380.48 MiB | 65,435.59 MiB |
| Peak committed memory | 128,225.80 MiB | 129,546.86 MiB |
| Peak pagefile used | 14,495.65 MiB | 15,549.33 MiB |
| Rank count | 2 | 2 |
| Transport | Windows CUDA P2P/NVLink | Windows CUDA P2P/NVLink |

Both ranks entered and completed sampling. Continuous hardware samples reached 100% utilization on both GPUs; VAE decode then returned to the ordinary single-GPU ComfyUI path.

## Media acceptance

| Check | I2V Turbo8 | REF2VA Turbo4 |
|---|---|---|
| Video | H.264, 1120x768, 124 frames, 24 FPS, 5.167 s | same |
| Audio | AAC, 32 kHz, stereo | same |
| Unique decoded frame hashes | 124/124 | 124/124 |
| Black intervals | 0 | 0 |
| Result | PASS | PASS |

## O6 numeric gates

Safe FP16 must satisfy the initial sampling acceleration gate for both workflows. The 11x column is retained as the later optimization target and is not an initial release blocker:

| Workflow | Baseline `B_i` | Initial 4x gate `B_i / 4` | 11x optimization target `B_i / 11` | Required initial result |
|---|---:|---:|---:|---|
| I2V Turbo8 | 160.7195 s/it | 40.1799 s/it | 14.6109 s/it | no higher than 40.1799 s/it |
| REF2VA Turbo4 | 185.2034 s/it | 46.3008 s/it | 16.8367 s/it | no higher than 46.3008 s/it |

Model loading, preprocessing, VAE decode and video save must each be no slower than the matched value above; end-to-end wall time must improve. Numerical correctness, finite tensors, two-rank completion, CUDA P2P transport and media acceptance remain mandatory regardless of speed.

## Implemented safe-FP16 result

The opt-in worker-side implementation keeps FP32 numerical islands around FP16 attention/MLP computation while preserving FP8 checkpoint/FSDP storage. Both full workflows complete with exact finite output across two ranks, 124/124 unique decoded frames, no detected black interval, H.264 1120x768 video and 32 kHz stereo AAC.

| Workflow | FP32 baseline | Initial safe FP16 | Initial speedup | Best accepted experiment | Best speedup | 4x result |
|---|---:|---:|---:|---:|---:|---|
| I2V Turbo8 | 160.7195 s/it | 48.7121 s/it | 3.299x | 48.7121 s/it | 3.299x | FAIL |
| REF2VA Turbo4 | 185.2034 s/it | 54.2174 s/it | 3.416x | 49.8680 s/it with optional 5 GiB host registration | 3.714x | FAIL |

The functional and quality goals pass, but O6 does not pass its initial 4x performance gate. `fp16_h3_safe` therefore remains explicit and experimental. V100 GEMM extent alignment is retained as an automatic matching-path improvement. Bounded host registration is implemented but remains disabled by default because it improves sampling while the matched cold end-to-end result regresses. Rejected kernel, chunk-size, attention, topology, registration-capacity and prefetch experiments are documented with resource data in the [English](../../releases/2026-08-18-o6-minimax-h3-safe-fp16.en.md) and [Chinese](../../releases/2026-08-18-o6-minimax-h3-safe-fp16.zh-CN.md) upgrade records.

## Evidence references

Raw logs, telemetry CSVs, API prompts and benchmark JSON remain local:

- `logs/minimax-h3/o2/20260818-155526-i2v-full-o6-baseline-fp32-p2p256/`
- `logs/minimax-h3/o2/20260818-162159-ref2va-full-o6-baseline-fp32-p2p256/`
- `ComfyUI/output/video/raylight_o3/minimax_h3_i2v_o6-baseline-fp32-p2p256_run0_00001_.mp4`
- `ComfyUI/output/video/raylight_o3/minimax_h3_ref2va_o6-baseline-fp32-p2p256_run0_00001_.mp4`
- `logs/minimax-h3/o2/20260818-184842-i2v-full-o6-safe-fp16-full-reviewed/`
- `logs/minimax-h3/o2/20260818-185943-ref2va-full-o6-safe-fp16-full-reviewed/`
- `logs/minimax-h3/o2/20260818-201232-ref2va-full-o6-safe-fp16-align8-full-reviewed/`
- `logs/minimax-h3/o2/20260818-212923-ref2va-full-o6-safe-fp16-hostreg5g-scoped-full/`

These are one cold run per workflow under a fixed machine state. If geometry, frame count, step count, P2P capacity, precision policy, model assets or core runtime versions change, a new matched baseline is required.
