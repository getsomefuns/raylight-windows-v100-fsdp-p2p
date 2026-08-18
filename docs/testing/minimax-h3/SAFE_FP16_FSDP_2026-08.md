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

At the time of these O6 measurements, the launcher default was 128 MiB. A preliminary I2V attempt proved that this exact geometry needs a larger buffer: its Ulysses remote payload was 239,826,944 bytes and was correctly rejected by that capacity. Formal benchmarks therefore used 256 MiB. Since the 2026-08-19 launcher-control update, the public launcher defaults to 256 MiB and exposes 128/256/512 MiB choices; the measurements below remain unchanged historical evidence.

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

The functional and quality goals pass, but O6 does not pass its initial 4x performance gate. `fp16_h3_safe` therefore remains explicit and experimental. V100 GEMM extent alignment is retained as an automatic matching-path improvement. Bounded host registration is implemented but remains disabled by default because it improves sampling while the matched cold end-to-end result regresses.

## Complete experiment ledger

This section is the O6 test ledger, not a success-only summary. Status terms mean:

- **Implemented**: retained in the current runtime tree with workflow or focused-test evidence.
- **Optional implementation**: retained but disabled by default and activated only by an explicit switch.
- **Rejected after experiment**: a comparable full or smoke workflow demonstrated that it should not enter the default path.
- **No valid performance conclusion**: the attempt did not produce a complete, matched, or probe-free result; the reached stage and exclusion reason are recorded.
- **Microbenchmark/backend probe only**: did not enter the formal ComfyUI workflow and cannot support a workflow-speed claim.

### A. Matched full workflows

Unless noted, these are dual-V100, FSDP CPU offload, Ulysses=2, Ring=1, Windows CUDA P2P, 1120x768 and 124 frames. `s/it` is the slowest rank's complete sampling interval divided by the exact step count.

| Time/experiment | Workflow | Sampling s/it | vs FP32 | End-to-end / Sampler (s) | GPU0/GPU1 peak VRAM (MiB) | Peak physical / committed / pagefile (MiB) | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| 15:55 FP32 baseline | I2V, 8 steps | 160.7195 | 1.000x | 1463.670 / 1318.626 | 16162 / 16214 | 65380.5 / 128225.8 / 14495.7 | accepted control |
| 16:21 FP32 baseline | REF2VA, 4 steps | 185.2034 | 1.000x | 932.031 / 774.778 | 16237 / 16137 | 65435.6 / 129546.9 / 15549.3 | accepted control |
| 18:48 initial safe FP16 | I2V, 8 steps | 48.7121 | 3.299x | 563.283 / 421.513 | 16163 / 14862 | 65384.3 / 129062.9 / 1562.8 | implemented; quality passed, 4x failed |
| 18:59 initial safe FP16 | REF2VA, 4 steps | 54.2174 | 3.416x | 394.110 / 246.844 | 16237 / 15826 | 65432.3 / 128221.3 / 16403.5 | implemented; quality passed, 4x failed |
| 19:09 FP8 fallback chunk=128 MiB | REF2VA, 4 steps | 88.9425 | 2.082x | 537.555 / 387.751 | 16239 / 16238 | 65434.5 / 129137.9 / 14299.6 | 64.0% slower than initial FP16; rejected |
| 19:36 FP8 fallback chunk=64 MiB | REF2VA, 4 steps | 64.9555 | 2.851x | 458.682 / 298.009 | 16237 / 15826 | 65436.3 / 130418.7 / 19479.4 | 19.8% slower and higher paging; rejected |
| 19:45 LoRA fallback chunk=64 MiB | REF2VA, 4 steps | 58.6683 | 3.157x | 429.454 / 282.343 | 16237 / 15826 | 65438.4 / 129741.8 / 14383.3 | 8.2% slower; rejected |
| 20:12 GEMM extent alignment=8 | REF2VA, 4 steps | 51.9470 | 3.565x | 397.136 / 245.483 | 16237 / 15826 | 65438.8 / 130381.7 / 1455.1 | 4.19% sampling gain; implemented |
| 21:29 alignment + 5 GiB host registration | REF2VA, 4 steps | 49.8680 | 3.714x | 424.241 / 249.977 | 16237 / 15836 | 65439.6 / 129369.8 / 1454.2 | another 4.00% sampling gain, but 6.83% worse cold E2E; optional and default-off |

Full-run node-stage details distinguish sampling improvement from model-load, preprocessing, Ray initialization, VAE, and save regressions:

| Experiment | Model load | Preprocess | Ray init | Sampler | VAE decode | Video save |
|---|---:|---:|---:|---:|---:|---:|
| I2V initial safe FP16 | 19.371 | 31.865 | 40.927 | 421.513 | 38.321 | 8.627 |
| REF2VA initial safe FP16 | 19.400 | 36.134 | 37.614 | 246.844 | 42.984 | 8.965 |
| FP8 chunk=128 MiB | 19.302 | 38.784 | 36.195 | 387.751 | 45.882 | 9.032 |
| FP8 chunk=64 MiB | 19.825 | 44.617 | 39.761 | 298.009 | 45.095 | 9.469 |
| LoRA chunk=64 MiB | 19.558 | 40.520 | 36.452 | 282.343 | 38.854 | 9.955 |
| GEMM extent alignment=8 | 21.136 | 37.769 | 37.859 | 245.483 | 43.385 | 9.631 |
| 5 GiB host registration | 30.766 | 40.122 | 50.819 | 249.977 | 42.055 | 8.893 |

| Experiment | GPU0/GPU1 peak utilization | GPU0/GPU1 peak power (W) | Participation conclusion |
|---|---:|---:|---|
| I2V initial safe FP16 | 100% / 100% | 355.42 / 367.87 | both GPUs participate |
| REF2VA initial safe FP16 | 100% / 100% | 360.28 / 362.35 | both GPUs participate |
| FP8 chunk=128 MiB | 100% / 100% | 368.42 / 370.78 | both participate, throughput regressed |
| FP8 chunk=64 MiB | 100% / 100% | 362.31 / 366.19 | both participate, throughput regressed |
| LoRA chunk=64 MiB | 100% / 100% | 349.41 / 361.98 | both participate, throughput regressed |
| GEMM extent alignment=8 | 100% / 100% | 370.02 / 370.17 | both participate, sampling improved |
| 5 GiB host registration | 100% / 100% | 368.37 / 371.95 | both participate; sampling improved but cold E2E regressed |

Every completed full safe-FP16 output passed exact two-rank equality, finite-tensor, 124/124 unique-frame and no-black-interval checks. None reached 4x; smoke tests and microbenchmarks do not override that result.

### B. Matched smoke-workflow screening

Except for the first I2V row, these are REF2VA, 608x352, 39 frames and 4 steps. They screen candidates quickly and are not mixed into the 1120x768 acceptance gate.

| Time/experiment | Sampling s/it | End-to-end (s) | GPU0/GPU1 peak VRAM (MiB) | Peak physical / committed / pagefile (MiB) | Decision |
|---|---:|---:|---:|---:|---|
| 18:38 I2V initial safe-FP16 smoke | 14.0002 | 244.787 | 15679 / 4179 | 65305.0 / 128316.5 / 1318.9 | completed; 97% / 97% peak GPU utilization; functional screening |
| 18:44 REF2VA initial safe-FP16 smoke | 14.1099 | 192.713 | 15949 / 4331 | 65439.1 / 130084.7 / 8388.3 | completed; 100% / 97% peak GPU utilization; functional screening |
| 20:07 GEMM alignment smoke | 10.8802 | 172.100 | 15949 / 4331 | 65344.3 / 129486.9 / 959.3 | smoke reference |
| 20:33 CUDA profiler smoke | 40.0292 | 301.169 | 15949 / 4331 | 65418.4 / 130346.9 / 1050.8 | instrumentation heavily distorted timing; collection proof only |
| 21:04 4 GiB host registration | 9.9379 | 198.367 | 15949 / 4340 | 65438.5 / 131715.8 / 1767.4 | sampling improved but E2E regressed; continue capacity screening, not a default |
| 21:24 scoped 5 GiB host registration | 8.6862 | 169.355 | 15949 / 4342 | 65420.5 / 131247.3 / 1066.0 | best smoke; promoted to full validation |
| 22:29 scoped 6 GiB host registration | 11.2885 | 202.043 | 15949 / 4344 | 65439.2 / 128986.2 / 8808.3 | 30.0% slower than 5 GiB with much more paging; rejected |
| 22:49 single-ring direct return | 11.1691 | 192.076 | 15949 / 4416 | 65439.5 / 130725.1 / 4652.0 | 28.6% slower and bypassed `use_sync=True`; fully reverted |
| 23:04 forward prefetch=128 MiB attempt | 10.1368 | 182.809 | 15949 / 4342 | 65437.4 / 129338.2 / 7024.0 | runtime logged `configured=0`; prefetch never activated, so timing is not evidence; feature and fix attempt reverted |

The two initial safe-FP16 smoke runs establish that both graph families complete at reduced geometry, but are not part of the matched 4x gate.

### C. Workflow attempts without a valid completed result

| Time/directory | What was tested and the reached stage | Observation | Exclusion/non-implementation basis |
|---|---|---|---|
| 15:44 I2V FP32 with the then-default 128 MiB P2P | Workflow, two workers and 684 FSDP wrappers/rank entered; stopped at the first Ulysses collective | Remote payload was 239,826,944 bytes, greater than the 134,217,728-byte capacity; explicit `ValueError`; `runs=0` | Benchmark capacity was too small for this geometry, not a model or P2P failure. Formal runs used 256 MiB and passed; the launcher default was later changed to 256 MiB on 2026-08-19 |
| 19:20 pinned-memory attempt | Two workers, safe FP16, Sampler and FSDP model preparation entered; logs end during preparation; `runs=0` | No exception was saved, but there is no `FSDP registered successfully`, sampling progress, video or benchmark result; interrupted after about 478 s | No timing/resource conclusion and no invented root cause; unreliable, not implemented |
| 19:33 first FP8 chunk=64 invocation | Worker/Sampler/FSDP preparation entered; only one rank logged registration; no completed sampling | `geometry=null`, `runs=0`; invocation was not the fixed 1120x768 formal case | Unmatched and incomplete, excluded; rerun correctly at 19:36 for the decision |
| 22:29 first 6 GiB host-registration invocation | ComfyUI server started, but benchmark submitted/recorded no API prompt | `runs=0`; the `_ray_runtime_env/__init__.py` warning also exists in successful runs | Warning is not causal evidence; no workflow ran, so excluded and corrected in the later valid 6 GiB smoke run |

The shared missing `_ray_runtime_env` temporary-directory warning is classified as **non-causal startup noise** because it also appears in multiple successful workflows.

### D. Profiler, microbenchmark and backend probes

| Experiment | Test level and result | Why it did not enter the default implementation |
|---|---|---|
| CUDA profiler smoke | 40.0292 s/it, 301.169 seconds E2E; instrumentation materially changed timing | Validates collection only; not performance evidence |
| CUDA profiler full REF2VA | 98.6803 s/it, 601.111 seconds E2E; peak VRAM 16237/15836 MiB, physical/committed/pagefile peaks 65439.5/128575.9/7291.2 MiB, and 100%/100% peak GPU utilization; 193.728 s self-CUDA total. Efficient attention 121.852 s (62.90%), `aten::mm` 32.803 s (16.93%), `aten::copy_` 27.682 s (14.29%) | Timing is profiler-distorted; used only to identify attention as the primary bottleneck and GEMM/copy as secondary. Profiler remains explicit and default-off |
| FP8 fallback chunk 32→128 MiB microbenchmark | Local projection about 49.92→48.81 ms, roughly 2.3% faster, with about 169 MiB additional peak VRAM | Full workflow regressed from 54.2174 to 88.9425 s/it; local gain was overwhelmed by graph scheduling/memory behavior; rejected |
| Direct output-slice projection write | About 2% faster in isolated projection | No full-workflow evidence and too little gain for added complexity; not implemented |
| KV-allgather attention topology | Ulysses H28/S33792: 541.34 ms; KV-allgather H56/Q16896/K33792: 546.11 ms | Candidate was about 0.9% slower at matched shape; rejected |
| No-LSE Torch SDPA / xFormers auto | About 0.5% slower locally; some candidates did not satisfy current interface/LSE semantics | No local advantage plus semantic mismatch; never entered workflow |
| SageAttention 1.0.6 + triton-windows | Standalone backend probe failed during sm_70 INT8 `tl.dot` lowering; workflow never started | V100 backend incompatible; not added as a dependency |
| ai-bond flash-attn-v100 Windows wheel | Isolated environment only; default path about 48.6% slower; MMA_NATIVE slower again with larger numerical error; never integrated into ComfyUI | Neither speed nor numerical behavior justified replacement; isolated environment removed and not a dependency |
| Single-ring merge bypass microbenchmark | About 0.5% local gain | Full smoke workflow was 28.6% slower and synchronization semantics were bypassed; `92b56bb` fully reverted by `bcc7198` |
| Forward-prefetch helper | Helper tests passed, but FSDP2 dynamic wrapping changed the class name; runtime matched zero blocks and logged `configured=0` | It never activated and therefore has no performance evidence; `15eadfa` and subsequent matching fix attempt fully reverted by `30cbb69` |
| Host registration 4/5/6 GiB | 4 GiB gave a local gain; 5 GiB was best smoke and still improved full sampling; 6 GiB regressed with more paging | Only bounded opt-in implementation retained; default-off because 5 GiB still regressed matched cold E2E |

Detailed profiler copies were pageable H2D 14.975 s (4,080 calls), pinned H2D 3.260 s (2,305 calls), and P2P 1.345 s (2,654 calls). P2P was therefore not the dominant sampling bottleneck, so NVLink microbenchmark bandwidth alone cannot justify ever-larger P2P chunks as a workflow optimization.

P2P profiles also prove real dual-GPU exchange: initial safe-FP16 I2V made 5,282 calls with 545.945 GB total payload (272.973 GB remote); REF2VA made 2,650 calls with 285.726 GB total payload (142.863 GB remote). REF2VA control-wait / submit fell from 4.008 / 4.993 s initially, through 2.732 / 3.669 s after GEMM alignment, to 1.318 / 2.244 s with 5 GiB host registration. This supports reduced submission waiting, but does not overturn the default-off decision caused by cold-start E2E regression.

## Intermediate bug fixes and small optimizations

### 1. MiniMax H3 safe-FP16 numerical islands (implemented, `051aff1`)

- Added explicit `fp16_h3_safe` loader mode without changing global ComfyUI precision policy; disabled mode preserves the original path.
- ComfyUI otherwise sends unsupported checkpoint dtypes to FP32 on V100. A model-local compute override applies only during `MiniMaxH3` construction and does not widen global `supported_inference_dtypes`.
- Attention/MLP compute uses FP16 while residual, condition projection and critical output projection retain FP32 islands. Power-of-two scaling uses 64 for K output projection and 256 for FC2 to avoid the FP16 limit of 65,504.
- Safe attention output scaling runs after LoRA injection, protecting BF16 LoRA sidecars. Focused tests cover 120,000 and LoRA 180,000 extremes and require finite output.
- All 208 BF16 LoRA sidecars follow the FP16 branch dtype with zero unsupported entries.
- Context-local activation and idempotent markers prevent double wrapping when Ray workers are reused. An API-signature guard fails explicitly on future incompatible ComfyUI changes instead of silently producing wrong results.

### 2. GEMM extent alignment and bias-tail correction (implemented, `434743c`)

- V100 FP16 fallback linear/addmm chunk extents align automatically to multiples of eight. Full REF2VA sampling improved from 54.2174 to 51.9470 s/it, or 4.19%.
- The same change fixes `fp8_addmm_fallback_chunked` bias broadcasting/tail handling by broadcasting bias to full output shape before slicing. Tests cover an `[8,1]` tail and multiple bias shapes. This is retained for correctness, not merely speed.

### 3. Optional CUDA sampling profiler (diagnostic, `7a35842`)

- Rank0-only and explicit; disabled by default.
- Setup/report failure still returns a valid workflow result, and sampling is never retried after it starts merely for profiler collection.
- It established that efficient attention consumed 62.90% of measured CUDA time; its distorted timings are not release-performance claims.

### 4. Bounded Windows FSDP host registration (optional, `fb7e4b0`)

- Deduplicates shared storage, registers largest-first under a GiB capacity bound, captures quantized `qdata+scale`, and excludes dynamic `all_gather_inputs`.
- Storage references remain alive until unregister completes.
- Worker cleanup changed from `self.model=None` to `_free_current_model()` so unregistration and model-resource cleanup occur.
- 5 GiB reduces P2P control-wait/submit and improves sampling, but regresses full cold E2E by 6.83%; retained, default-off.

### 5. Host-registration lifecycle bug (fixed, `6258ca0`)

- Initial registration spanned the whole model lifetime, retaining pinned/registered host memory and increasing pressure outside sampling.
- Registration is now scoped to sampling in all three sampler methods. Normal and exceptional exits synchronize then unregister; cached FSDP models re-register on the next sample.
- Cleanup failure cannot mask the original sampling exception. This is a reliability/memory-lifecycle fix, not just tuning.

### 6. Single-ring fast-path semantic regression (found and reverted, `92b56bb` → `bcc7198`)

- Intended to skip redundant world-size-one merge and showed about 0.5% local gain.
- It also bypassed Raylight/Yunchang's required `use_sync=True` path and made smoke workflow 28.6% slower.
- Revert restored the exact pre-experiment tree; this optimization is absent from the deliverable.

### 7. FSDP forward-prefetch matching failure (found and reverted, `15eadfa` → `30cbb69`)

- Helper tests passed, but FSDP2 dynamic wrapping changed model class names; runtime matched no blocks and configured zero prefetch entries.
- The workflow proves only that the configuration did not activate; 10.1368 s/it cannot be used to judge prefetch performance.
- Feature commit and dynamic-class matching fix attempt were removed to avoid shipping an unverified dead switch.

## Current implementation and commit mapping

| Capability/attempt | Current tree | Commit(s) | Evidence basis |
|---|---|---|---|
| MiniMax H3 `fp16_h3_safe` numerical protection | retained, explicit | `051aff1` | both full workflows passed quality |
| V100 FP16 GEMM extent alignment and bias fix | retained, automatic on matching path | `434743c` | full REF2VA gain plus focused tests |
| CUDA sampling profiler | retained, default-off | `7a35842` | diagnostic only; default timing remains clean |
| Bounded host registration | retained, default-off | `fb7e4b0`, `6258ca0` | sampling gain but cold E2E regression |
| chunk=64/128, pinned, 6 GiB | not default implementations | no retained configuration | full regression, incomplete run, or smoke regression |
| Single-ring fast path | fully reverted | `92b56bb`, `bcc7198` | workflow regression and wrong sync semantics |
| Forward prefetch | fully reverted | `15eadfa`, `30cbb69` | configured zero at runtime, no valid evidence |
| SageAttention / flash-attn-v100 | not project dependencies | none | backend probe failure or substantial regression |

## Evidence references

Raw logs, telemetry CSVs, API prompts and benchmark JSON remain under local directory `<environment-root>\logs\minimax-h3\o2`. This index includes successful, failed, incomplete and profiler-distorted O6 attempts:

| Category | Evidence directory |
|---|---|
| Default-P2P-capacity failure | `20260818-154455-i2v-full-o6-baseline-fp32/` |
| Formal FP32 baselines | `20260818-155526-i2v-full-o6-baseline-fp32-p2p256/`; `20260818-162159-ref2va-full-o6-baseline-fp32-p2p256/` |
| Safe-FP16 smoke workflows | `20260818-183849-i2v-full-o6-safe-fp16-smoke-reviewed/`; `20260818-184408-ref2va-full-o6-safe-fp16-smoke-reviewed/` |
| Safe-FP16 formal full workflows | `20260818-184842-i2v-full-o6-safe-fp16-full-reviewed/`; `20260818-185943-ref2va-full-o6-safe-fp16-full-reviewed/` |
| Chunk / pinned experiments | `20260818-190934-ref2va-full-o6-safe-fp16-full-chunk128/`; `20260818-192040-ref2va-full-o6-safe-fp16-full-pinned/`; `20260818-193319-ref2va-full-o6-safe-fp16-full-fp8chunk64/`; `20260818-193659-ref2va-full-o6-safe-fp16-full-fp8chunk64-1120x768/`; `20260818-194555-ref2va-full-o6-safe-fp16-full-lorachunk64-1120x768/` |
| GEMM alignment | `20260818-200751-ref2va-full-o6-safe-fp16-align8-smoke/`; `20260818-201232-ref2va-full-o6-safe-fp16-align8-full-reviewed/` |
| Profiler | `20260818-203313-ref2va-full-o6-safe-fp16-profile-ref-smoke/`; `20260818-214113-ref2va-full-o6-safe-fp16-hostreg5g-profile-full-4step/` |
| Host registration 4/5 GiB | `20260818-210458-ref2va-full-o6-safe-fp16-hostreg4g-smoke/`; `20260818-212455-ref2va-full-o6-safe-fp16-hostreg5g-scoped-smoke/`; `20260818-212923-ref2va-full-o6-safe-fp16-hostreg5g-scoped-full/` |
| First no-run and corrected 6 GiB attempts | `20260818-222908-ref2va-smoke-o6-safe-fp16-hostreg6g-scoped-smoke/`; `20260818-222956-ref2va-full-o6-safe-fp16-hostreg6g-scoped-smoke/` |
| Single-ring regression | `20260818-224904-ref2va-full-o6-safe-fp16-hostreg5g-single-ring-fastpath-smoke/` |
| Prefetch did not activate | `20260818-230432-ref2va-full-o6-safe-fp16-hostreg5g-prefetch128-smoke/` |

All O6 benchmark videos are now consolidated under `<ComfyUI>\output\video\raylight_o6`. Historical `run0-prompt.json` files retain the original incorrect `raylight_o3` prefix as immutable run evidence; media organization does not rewrite those prompts. Future `o6-*` benchmark tags now route directly to `raylight_o6`.

These are one cold run per workflow under a fixed machine state. If geometry, frame count, step count, P2P capacity, precision policy, model assets or core runtime versions change, a new matched baseline is required.
