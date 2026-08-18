# O6 MiniMax H3 Safe-FP16 Upgrade and Experiment Record

Date: 2026-08-18

Status: functional and quality gates pass; the initial 4x performance gate does not pass; the feature remains explicit and experimental

Frozen commit: `30cbb69` (its runtime tree is identical to stable point `bcc7198`)

Comparison base: `d49adc`

## 1. Release conclusion

This upgrade implements a model-specific safe-FP16 path for MiniMax H3 on native Windows with two Tesla V100-SXM2-16GB GPUs, TCC, CUDA P2P/NVLink, and FSDP CPU offload. It does not force global FP16. Residual accumulation, condition projection, and numerically sensitive outputs remain FP32; the dominant attention/MLP branches and V100 FP8 fallback matrix multiplications use FP16.

Both full workflows finish without errors. The two ranks produce exact, finite results. Each output is a 1120x768, 124-frame, 24 FPS, 5.167-second video with 32 kHz stereo AAC. Every validated video has 124 unique decoded-frame hashes and no detected black interval.

| Workflow | FP32 baseline | Initial safe FP16 | Speedup | Best current full result | Best speedup | 4x gate |
|---|---:|---:|---:|---:|---:|---:|
| I2V Turbo8 | 160.7195 s/it | 48.7121 s/it | 3.299x | 48.7121 s/it | 3.299x | <=40.1799 s/it |
| REF2VA Turbo4 | 185.2034 s/it | 54.2174 s/it | 3.416x | 49.8680 s/it | 3.714x | <=46.3008 s/it |

The functional and quality objectives are complete, but this version must not claim 4x. I2V is still about 21.2% above its limit, and the best REF2VA result is about 7.7% above its limit.

## 2. Validation environment and timing definition

| Item | Fixed condition |
|---|---|
| GPUs | 2x Tesla V100-SXM2-16GB, TCC, NVLink/P2P |
| Driver | 577.00 |
| Python | 3.10.11 |
| PyTorch / CUDA runtime | 2.7.0+cu126 / 12.6 |
| Ray | 2.57.0 |
| ComfyUI | v0.31.0-15-g62b3c94b |
| Distributed topology | 2 ranks, Ulysses=2, Ring=1, FSDP CPU offload, Windows CUDA P2P ProcessGroup |
| Full geometry | 1120x768, 124 frames, 24 FPS, nominal 5-second setting |
| I2V / REF2VA | Turbo8 / Turbo4 with pinned input, prompt, model, LoRA, and seed |
| Sampling metric | complete sampling interval of the slower rank divided by the exact step count; not a transient tqdm value |

Windows physical, committed, and page-file peaks depend on the system state before a run. The raw measurements are retained below, but matched sampling time, sampler-node time, rank consistency, and media output have priority when deciding whether an optimization is real.

## 3. Implemented and retained changes

### 3.1 MiniMax H3 model-specific safe FP16 (`051aff1`)

Implemented:

- Added the explicit `fp16_h3_safe` RayUNETLoader mode without modifying ComfyUI's global dtype allowlist.
- Installed an idempotent compatibility adapter in every Ray worker before model construction and FSDP wrapping.
- Kept condition projection, residual flow, post-modulation/gating accumulation, and sensitive outputs in FP32.
- Moved dominant attention/MLP branch inputs, FP8 fallback dequantization, and LoRA sidecar matrix multiplications to FP16.
- Preserved FP8 checkpoint/FSDP-shard storage instead of expanding the whole checkpoint during load.
- Added two separate experimental workflows plus dtype, rank, finite-value, and regression checks.

Why retained: this is the main performance gain. I2V and REF2VA sampling time falls by 69.69% and 70.73%; sampler-node time falls by 68.03% and 68.14%. Media and two-rank numerical validation pass. This model-specific design avoids the black video and NaN/Inf behavior previously observed with global FP16.

### 3.2 V100 FP16 GEMM chunk alignment (`434743c`)

Implemented: the V100 FP8 fallback aligns FP16 output-chunk extents to multiples of eight, giving Volta Tensor Cores friendlier matrix shapes. The default temporary-buffer budget remains 32 MiB.

Why retained: the full REF2VA sampling result improves from 54.2174 to 51.9470 s/it, another 4.19%, reaching 3.565x over FP32. The change is limited to the matching V100 FP16 fallback path. Cold end-to-end time moved from 394.11 to 397.14 seconds because Ray initialization and preprocessing varied, so the claimed benefit is sampling-only.

### 3.3 Bounded Windows FSDP host registration (`fb7e4b0`, `6258ca0`)

Implemented:

- Optionally registers a bounded amount of FSDP CPU-offload shard storage as CUDA host memory.
- Deduplicates underlying storage and reports registered, capacity-skipped, and failed amounts.
- Limits the registration lifetime to sampling and releases it on normal or exceptional return from all three samplers.
- Remains disabled by default and is not enabled in the ordinary launcher.

Opt-in settings:

```powershell
$env:RAYLIGHT_FSDP_CPU_OFFLOAD_HOST_REGISTER = "1"
$env:RAYLIGHT_FSDP_CPU_OFFLOAD_HOST_REGISTER_MIB = "5120"
```

Why retained but not enabled by default: 5 GiB registration lowers full REF2VA sampling from the aligned 51.9470 to 49.8680 s/it, a further 4.00%, for the current best 3.714x result. P2P-profile collective control wait falls from 2.732 to 1.318 seconds, and submit time falls from 3.669 to 2.244 seconds. However, model load, Ray initialization, and cold end-to-end time are worse in that run, and larger locked capacity increases host-memory/page-file pressure. This is an implemented opt-in experiment, not a default recommendation or an end-to-end speed claim.

### 3.4 Optional CUDA sampling profiler (`7a35842`)

Implemented: rank 0 can enable one CPU/CUDA profile with `RAYLIGHT_TORCH_PROFILE=1`. It is disabled by default, and setup/report failures do not alter an otherwise valid sample.

Why retained: this is diagnostics, not an optimization. It attributes about 62.9% of CUDA self time to attention, 16.9% to matrix multiplication, and 14.3% to copies; pageable H2D accounts for about 7.7%, while P2P accounts for about 0.7%. The evidence shows that NVLink/P2P bandwidth is not the primary remaining bottleneck.

## 4. Matched full-run data

### 4.1 Time and throughput

| Run | s/it | Speedup vs FP32 | End to end | Sampler node | Model load | Preprocess | Ray init | VAE decode | Video save |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I2V FP32 baseline | 160.7195 | 1.000x | 1463.670 s | 1318.626 s | 20.985 s | 33.532 s | 36.197 s | 42.806 s | 9.183 s |
| I2V safe FP16 | 48.7121 | 3.299x | 563.283 s | 421.513 s | 19.371 s | 31.865 s | 40.927 s | 38.321 s | 8.627 s |
| REF2VA FP32 baseline | 185.2034 | 1.000x | 932.031 s | 774.778 s | 20.982 s | 42.839 s | 37.985 s | 43.885 s | 9.174 s |
| REF2VA safe FP16 | 54.2174 | 3.416x | 394.110 s | 246.844 s | 19.400 s | 36.134 s | 37.614 s | 42.984 s | 8.965 s |
| REF2VA + GEMM alignment | 51.9470 | 3.565x | 397.136 s | 245.483 s | 21.136 s | 37.769 s | 37.859 s | 43.385 s | 9.631 s |
| REF2VA + alignment + 5 GiB registration | 49.8680 | 3.714x | 424.241 s | 249.977 s | 30.766 s | 40.122 s | 50.819 s | 42.055 s | 8.893 s |

### 4.2 Resource peaks

| Run | GPU0 / GPU1 VRAM | GPU0 / GPU1 utilization | GPU0 / GPU1 power | Physical memory | Committed memory | Page file |
|---|---:|---:|---:|---:|---:|---:|
| I2V FP32 baseline | 16162 / 16214 MiB | 100 / 100% | 366.31 / 369.20 W | 65380.5 MiB | 128225.8 MiB | 14495.7 MiB |
| I2V safe FP16 | 16163 / 14862 MiB | 100 / 100% | 355.42 / 367.87 W | 65384.3 MiB | 129062.9 MiB | 1562.8 MiB |
| REF2VA FP32 baseline | 16237 / 16137 MiB | 100 / 100% | 366.52 / 365.98 W | 65435.6 MiB | 129546.9 MiB | 15549.3 MiB |
| REF2VA safe FP16 | 16237 / 15826 MiB | 100 / 100% | 360.28 / 362.35 W | 65432.3 MiB | 128221.3 MiB | 16403.5 MiB |
| REF2VA + GEMM alignment | 16237 / 15826 MiB | 100 / 100% | 370.02 / 370.17 W | 65438.8 MiB | 130381.7 MiB | 1455.1 MiB |
| REF2VA + alignment + 5 GiB registration | 16237 / 15836 MiB | 100 / 100% | 368.37 / 371.95 W | 65439.6 MiB | 129369.8 MiB | 1454.2 MiB |

VRAM does not fall in proportion to FP16 use. FSDP shards, output/residual tensors, P2P buffers, VAE work, and the ComfyUI host process still occupy memory. The main safe-FP16 benefit is compute throughput and smaller temporary tensors, not a permanent half-size checkpoint copy.

### 4.3 P2P and correctness

| Run | Collective calls | Total payload | Remote bytes | Control wait | Submit | Exact across ranks | Media |
|---|---:|---:|---:|---:|---:|---|---|
| I2V safe FP16 | 5282 | 545.95 GB | 272.97 GB | 1.274 s | 3.064 s | PASS | PASS |
| REF2VA safe FP16 | 2650 | 285.73 GB | 142.86 GB | 4.008 s | 4.993 s | PASS | PASS |
| REF2VA + GEMM alignment | 2650 | 285.73 GB | 142.86 GB | 2.732 s | 3.669 s | PASS | PASS |
| REF2VA + 5 GiB registration | 2650 | 285.73 GB | 142.86 GB | 1.318 s | 2.244 s | PASS | PASS |

The table uses decimal GB as recorded by the logs. All full safe-FP16 results are finite. Rank 0 and rank 1 output and denoised-output comparisons are exact with maximum absolute difference zero.

## 5. Experimental attempts and decisions

### 5.1 Full 1120x768, 124-frame experiments

| Attempt | s/it | Speedup vs FP32 | End to end | GPU0 / GPU1 VRAM | Physical / committed / page-file peaks | Decision |
|---|---:|---:|---:|---:|---:|---|
| Initial safe FP16 | 54.2174 | 3.416x | 394.110 s | 16237 / 15826 MiB | 65432 / 128221 / 16404 MiB | Shipped |
| 128 MiB FP8 temporary chunk | 88.9425 | 2.082x | 537.555 s | 16239 / 16238 MiB | 65435 / 129138 / 14300 MiB | Rejected; larger peaks disrupt the pipeline |
| 64 MiB FP8 temporary chunk | 64.9555 | 2.851x | 458.682 s | 16237 / 15826 MiB | 65436 / 130419 / 19479 MiB | Rejected; 19.8% slower than the 32 MiB default |
| 64 MiB LoRA chunk | 58.6683 | 3.157x | 429.454 s | 16237 / 15826 MiB | 65438 / 129742 / 14383 MiB | Rejected; 8.2% slower than initial safe FP16 |
| GEMM extent aligned to 8 | 51.9470 | 3.565x | 397.136 s | 16237 / 15826 MiB | 65439 / 130382 / 1455 MiB | Shipped; sampling improves 4.19% |
| Alignment + 5 GiB host registration | 49.8680 | 3.714x | 424.241 s | 16237 / 15836 MiB | 65440 / 129370 / 1454 MiB | Mechanism shipped, default off; sampling improves but cold total worsens |
| Full pin-memory policy | No completed run | — | no run after about 478 s | — | — | Rejected; did not complete reliably on Windows/Ray |

### 5.2 608x352, 39-frame smokes

| Attempt | s/it | End to end | GPU0 / GPU1 VRAM | Physical / committed / page-file peaks | Relative to 5 GiB smoke | Decision |
|---|---:|---:|---:|---:|---:|---|
| GEMM alignment | 10.8802 | 172.100 s | 15949 / 4331 MiB | 65344 / 129487 / 959 MiB | 25.3% slower | Unregistered reference |
| 4 GiB host registration | 9.9379 | 198.367 s | 15949 / 4340 MiB | 65439 / 131716 / 1767 MiB | 14.4% slower | Helpful, but below 5 GiB |
| 5 GiB scoped registration | 8.6862 | 169.355 s | 15949 / 4342 MiB | 65421 / 131247 / 1066 MiB | Reference | Best smoke |
| 6 GiB scoped registration | 11.2885 | 202.043 s | 15949 / 4344 MiB | 65439 / 128986 / 8808 MiB | 30.0% slower | Rejected; paging pressure reverses the gain |
| Single-ring direct fast path | 11.1691 | 192.076 s | 15949 / 4416 MiB | 65440 / 130725 / 4652 MiB | 28.6% slower | Reverted; correct synchronization is slower |
| 128 MiB FSDP forward prefetch | 10.1368 | 182.809 s | 15949 / 4342 MiB | 65437 / 129338 / 7024 MiB | Not attributable | Log says `configured=0`; fully removed |

The forward-prefetch run configured no modules, so 10.1368 s/it is ordinary no-prefetch variance and is not evidence for the design. Commit `30cbb69` fully reverts the implementation introduced by `15eadfa`; neither source nor the deployed node retains it.

### 5.3 Kernel, attention, and topology microbenchmarks

| Attempt | Observation | Why not shipped |
|---|---|---|
| 32→128 MiB FP8 fallback microbench | 49.92→48.81 ms, only about 2.3% for one projection; about 169 MiB more peak memory | The full workflow regresses from 54.22 to 88.94 s/it; the local microbenchmark does not represent the FSDP pipeline |
| Direct write into output slices | About 2% for one projection | Too little total gain for the added complexity; no full-run evidence |
| Ulysses vs KV-allgather attention | 541.34 vs 546.11 ms | KV gather is about 0.9% slower, so Ulysses stays |
| xFormers auto / torch SDPA / no-LSE variants | About 0.5% slower or incompatible semantics | No measurable gain; some paths lack the LSE/synchronization behavior required by Raylight/xFuser |
| SageAttention 1.0.6 | sm_70 INT8 `tl.dot` lowering fails | Triton/kernel path does not support the current Windows V100 stack |
| flash-attn-v100 Windows wheel | Default path about 48.6% slower; MMA_NATIVE is slower with larger error | Fails speed and numerical criteria; isolated test environment is not a project dependency |
| Single-ring bypass microbenchmark | About 0.5% local gain | Real synchronized dual-GPU smoke is 28.6% slower; `use_sync=True` cannot be skipped for cosmetic speed |

## 6. Why 4x is not reached yet

- Safe FP16 removes most FP32 GEMM cost, but attention still accounts for about 62.9% of CUDA self time across 50 blocks.
- FSDP CPU offload still requires H2D/all-gather every step. Host registration reduces control and submit overhead but cannot remove transfer and synchronization.
- VAE decode, preprocessing, Ray actor initialization, and media writing are not accelerated by dual-GPU sampling parallelism.
- P2P/NVLink is active and consumes little profiled self time; larger P2P buffers or a different communication topology cannot by themselves solve the attention compute bottleneck.
- Both 16 GiB GPUs already operate near their VRAM limit. Aggressive prefetching, large dequantization chunks, and extra caches tend to become OOM, fragmentation, or host paging.

## 7. Current delivery boundary

| Item | Current status |
|---|---|
| `fp16_h3_safe` loader and two experimental workflows | Implemented; enabled only when explicitly selected |
| FP32 numerical islands + FP16 attention/MLP | Implemented and full-quality validated |
| V100 GEMM extent alignment | Implemented; automatic on the matching path |
| 5 GiB FSDP host registration | Mechanism implemented; default off; controlled testing only |
| CUDA profiler | Implemented; default off; diagnostics only |
| Global MiniMax/LTX FP16 allowlist | Not modified and must remain unchanged |
| Initial 4x performance gate | Not met |
| Later 11x optimization target | Not met; a research target, not a current capability |

User-loadable safe-FP16 workflows:

- `example_workflows/Minimax_H3_I2V_Windows_V100_FSDP_Turbo8_FP16_Experimental.json`
- `example_workflows/Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4_FP16_Experimental.json`

RayUNETLoader must retain `weight_dtype=fp16_h3_safe` in these workflows. Ordinary `default` workflows remain the FP32-compute comparison path. Global `fp16` or dtype-allowlist edits are not equivalent to this feature.

## 8. Freeze verification

- Relevant safe-FP16, FP8 fallback, host-registration, sampling-profiler, and sampler-lifecycle tests: 64 passed plus 2 passed subtests.
- Six core runtime modules pass `py_compile`.
- The Chinese and English records each contain 210 lines, 20 headings, and 9 tables; the structures match. A relative-link check across eight related Markdown files reports zero missing targets.
- Independent code review: High 0, Medium 0, Low 0.
- The runtime-code tree is frozen at `30cbb6965d8f956fd9abb462a8103862097e7056`. Deployment is repeated after the documentation commit, and source HEAD must match the ComfyUI deployment marker.

## 9. Evidence index

Raw benchmark JSON, logs, and monitor CSV files remain local under `logs/minimax-h3/o2/<run>/`. Principal runs:

- `20260818-155526-i2v-full-o6-baseline-fp32-p2p256`
- `20260818-162159-ref2va-full-o6-baseline-fp32-p2p256`
- `20260818-184842-i2v-full-o6-safe-fp16-full-reviewed`
- `20260818-185943-ref2va-full-o6-safe-fp16-full-reviewed`
- `20260818-201232-ref2va-full-o6-safe-fp16-align8-full-reviewed`
- `20260818-212923-ref2va-full-o6-safe-fp16-hostreg5g-scoped-full`
- `20260818-214113-ref2va-full-o6-safe-fp16-hostreg5g-profile-full-4step`

This document preserves maintainable conclusions without copying large raw logs into the repository. The Chinese version is [2026-08-18-o6-minimax-h3-safe-fp16.zh-CN.md](2026-08-18-o6-minimax-h3-safe-fp16.zh-CN.md).
