# MiniMax H3 Director MODEL Adapter: O8 Validation Record

[简体中文](MMH3DIRECTOR_ADAPTER_2026-08.zh-CN.md) | [English](MMH3DIRECTOR_ADAPTER_2026-08.md)

## 1. Validation goal

Add an explicit opt-in Ray `MODEL` proxy for Director without modifying MiniMaxH3Director, existing Raylight node schemas, or accepted workflow JSON. Each Director diffusion pass must execute exactly once on the two Ray workers through the accepted Windows FSDP2 and CUDA P2P/NVLink data plane.

This report is maintained from the pre-development baseline onward. See the [upstream baseline report](MMH3DIRECTOR_UPSTREAM_BASELINE_2026-08-22.md) for the original Director single-GPU boundary and single-/multi-segment results.

## 2. Pre-O8 baseline (2026-08-23)

### 2.1 Software and hardware gates

| Item | Result |
|---|---|
| Baseline commit | `f5c455d` (adapter runtime code not started) |
| GPUs | 2× Tesla V100-SXM2-16GB, TCC, driver 577.00 |
| Python / PyTorch | 3.10.11 / 2.7.0+cu126 |
| Key dependencies | torchvision 0.22.0+cu126; torchaudio 2.7.0+cu126; xformers 0.0.30; Ray 2.57.0; xFuser 0.4.5; yunchang 0.6.4 |
| Full test suite | `342 passed, 1 warning, 30 subtests passed` in 68.19 seconds |
| P2P correctness | Synchronous/asynchronous, chunk tails, FP32/FP16/BF16/UINT8 all pass |
| P2P bandwidth | Main results about 59.2–61.6 GiB/s/direction, above the 50 GiB/s gate |

The first baseline run exposed an environment dependency on `Get-FileHash` in the model download script and ended with `341 passed, 1 failed`. Replacing it with direct .NET SHA-256 made both the focused regression and full suite pass. The fix does not alter Raylight runtime behavior, node interfaces, or workflows.

### 2.2 Existing public node-schema snapshot

| Node ID / display name | Required inputs | Optional inputs | Outputs | Function |
|---|---|---|---|---|
| `RayInitializer` / Ray Init Actor | ray_cluster_address, ray_cluster_namespace, GPU, ulysses_degree, ring_degree, cfg_degree, dp_degree, sync_ulysses, clear_vram_after_sampling, FSDP, FSDP_CPU_OFFLOAD, XFuser_attention, skip_comm_test, use_mmap | wait_for | `RAY_ACTORS_INIT` (`ray_actors_init`) | `spawn_actor` |
| `RayInitializerAdvanced` / Ray Init Actor (Advanced) | ray_cluster_address, ray_cluster_namespace, GPU, GPU_SELECT, ulysses_degree, ring_degree, clear_vram_after_sampling, cfg_degree, dp_degree, sync_ulysses, FSDP, FSDP_CPU_OFFLOAD, XFuser_attention, skip_comm_test, use_mmap | wait_for, ray_object_store_gb, ray_dashboard_address, torch_dist_address | `RAY_ACTORS_INIT` (`ray_actors_init`) | `spawn_actor` |
| `RayUNETLoader` / Load Diffusion Model (Ray) | unet_name, weight_dtype, ray_actors_init | lora | `RAY_ACTORS` (`ray_actors`) | `load_ray_unet` |
| `RayKill` / Kill Ray | ray_actors, kill_mode | — | none | `kill_ray` |
| `XFuserSamplerCustomAdvanced` / XFuser SamplerCustom Advanced | add_noise, noise_seed, guider, sampler, sigmas, latent_image | — | `LATENT`, `LATENT`, `RAY_ACTORS` | `ray_sample` |
| `XFuserSamplerCustom` / XFuser SamplerCustom | ray_actors, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image | — | `LATENT`, `RAY_ACTORS` | `ray_sample` |

The first four nodes retain category `Raylight`; the samplers retain `Raylight/extra/custom_sampling/samplers`. Task 1 converts this snapshot into automated compatibility contracts.

### 2.3 Accepted workflow JSON hashes

| Workflow | SHA256 |
|---|---|
| `LTX2_3_i2v_Raylight_Windows_FSDP_5s.json` | `2b0b6e3bab1a7adb7f5e3a8b627b0bfe5535b3d8b3e949ecb7644225f71eef84` |
| `Minimax_H3_I2V_Windows_V100_FSDP_Turbo8.json` | `2f261d43e0477e21165793edfade592e4d4eef948b0434c81db8180a8fc926a5` |
| `Minimax_H3_I2V_Windows_V100_FSDP_Turbo8_FP16_Experimental.json` | `faa485b80b87126bb45223f06c935f8372611f0d98efcd4106ee7111d3e41b02` |
| `Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4.json` | `af038101e582efd163d93f1fd22be7abe5d0c3a93b00b90c332f52d9ff2377ee` |
| `Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4_FP16_Experimental.json` | `fccd1b7ed6519cc81fa66f26a956061587f8cebce567f3ddef07ba0752a32f35` |

## 3. Implemented architecture

- Added the explicit `Ray MODEL Adapter (MMH3Director)` node. It accepts one loaded `RAY_ACTORS` value and returns a weightless `MODEL` proxy.
- Only that exact proxy type is intercepted at `comfy.sample.sample()`; ordinary ComfyUI models continue through the original sampler unchanged.
- Director-prepared conditioning, latent, noise, sigmas, seed, steps, sampler, scheduler, and shifts are forwarded unchanged to both ranks. The main process does not run a second diffusion sample.
- Operation ID, rank, shape, dtype, finite values, and exact final tensors must agree across both ranks before the result returns to Director. Failure never falls back to local single-GPU diffusion.
- Director remains the sole generation-control surface. The example has one model path, `RayInitializer → RayLoraLoader → RayUNETLoader → Ray MODEL Adapter → MiniMaxH3Director`, without a duplicate XFuser sampler/guider/scheduler stack.
- Compatibility is pinned to MiniMaxH3Director `1.0.0` at commit `d655912`; an incompatible sampling seam fails with an actionable error.

Example: `example_workflows/MMH3Director_Windows_V100_FSDP.json`. Local deployed copy: `ComfyUI/user/default/workflows/MMH3Director.json`.

## 4. First real-machine t2v results (2026-08-23)

All runs use two V100s in TCC, FSDP CPU offload, Ulysses 2, CUDA P2P, a 256 MiB buffer, `fp16_h3_safe`, the FL2VA FP8 checkpoint, and the official Turbo8 LoRA. Rank diagnostics remain disabled; startup P2P health is about 58.88 GiB/s/direction.

| Run | Geometry / duration | Status | Sampling | End to end | Peak GPU MiB | Peak GPU utilization | Peak physical / virtual / pagefile MiB |
|---|---|---|---:|---:|---:|---:|---:|
| Reduced cold after handoff fix | 864×480 / 52 frames / 24 FPS / 2.167 s | Pass | about 1:16; 8.85 s/it | 259.77 s | whole run 13738 / 5027; sampling about 5186 / 5027 | 99% / 96% | 90575.9 / 132370.9 / 1467 |
| Reduced warm, seed 43 | same | Pass; reused PIDs 20012 / 22160 | about 1:13; 8.83 s/it | 131.67 s | whole run 15302 / 5027; sampling about 5190 / 5027 | 99% / 94% | 78081.3 / 96592.7 / 1467 |
| 720p-class cold | 1280×704 / 124 frames / 24 FPS / 5.167 s | Pass | about 5:51; 42.2 s/it | 595.10 s | whole run 15422 / 14664; sampling about 14822 / 14664 | 100% / 100% | 97763.8 / 138466.8 / 1467 |
| 720p-class warm | same | Pass | about 5:49; 42.0 s/it | 470.82 s | 14826 / 14664 | 100% / 100% | 82795.5 / 101251.1 / 1467 |

The original single-GPU Director cold baseline at the same 720p-class geometry was 41:21 and 271.81 s/it. The current dual-GPU cold run is about 4.17x faster end to end and 6.44x faster per step; the warm run is about 5.27x and 6.47x faster. Cold and warm conditions differ, so the warm ratio is explicitly an interactive-reuse result, not presented as a strict cold comparison.

## 5. Preprocessing-VRAM handoff defect and fix

The first 864×480 cold run succeeded, but the text encoder overlapped both Ray ranks on GPU 0, producing a 16198 / 4721 MiB whole-run sampling peak. The cause was not unbalanced FSDP sharding: Director must first encode text in the main process, and the bridge did not unload those preprocessing models before remote sampling.

Commit `40a71db` calls ComfyUI's standard `unload_all_models()` and `soft_empty_cache()` before proxy sampling and synchronizes CUDA when available. Patched 864×480 sampling is about 5186 / 5027 MiB; 1280×704 sampling is about 14822 / 14664 MiB. The handoff keeps reloadable CLIP/VAE weights in host memory, so cold physical/virtual peaks increase; this machine still passes without a pagefile spike or OOM. Later larger/longer mode gates must retain that trade-off.

## 6. Output and reuse checks

- Cold and warm 720p outputs are H.264 1280×704, 124 frames at 24 FPS for 5.167 seconds, with 32 kHz stereo AAC.
- Contact sheets show continuous motion with no black frames, colored noise, corruption, or frozen interval. The prompt asks for three robots while the result mainly retains two, matching the upstream baseline's subject-count adherence limitation rather than distributed numerical damage.
- Same-seed cold/warm MP4 container SHA256 values differ, but video bitstream MD5 `3c38daa0228f118cb5dbd951114d7e27`, audio bitstream MD5 `b4cb8900c83a5d70b69c07fcb004d02b`, and final decoded-frame MD5 `84d8b99e1cf05a6b9abfdfda93f55f34` are identical.
- A changed-seed warm run reused worker PIDs 20012 / 22160, issued exactly one new remote operation, and did not repeat Gloo initialization or FSDP registration.
- Raw prompts, history, telemetry CSVs, logs, and contact sheets stay under local `<environment-root>/o8_artifacts/mmh3director_adapter`; videos stay under `ComfyUI/output/video/O8_MMH3Director` and are not committed.

## 7. Director mode, Refine, and multi-segment acceptance

| Mode | Geometry | End to end | Sampling | Peak GPU MiB | Result |
|---|---|---:|---:|---:|---|
| i2v | 864x480 / 39 frames / 24 FPS | 135.77 s | about 8.50 s/it | 15692 / 4861 | Pass |
| fl2v | 864x480 / 39 frames / 24 FPS | 140.35 s | about 8.50 s/it | 15833 / 4923 | Pass |
| refine | same, main 8 + refine 2 steps | 170.28 s | main 8.13, refine 4.26 s/it | 15541 / 4861 | Pass; same-model reuse |
| r2v | 864x480 / 39 frames / 24 FPS | 176.70 s | 7.40 s/it (4 steps) | 15538 / 4781 | Pass; REF2VA checkpoint switch |
| v2v | 864x480 / 39 frames / 24 FPS | 137.76 s | one remote sample | 15821 / 5681 | Pass; mismatched mode fails before GPU work |
| rv2v | 864x480 / 39 frames / 24 FPS | 133.39 s | one remote sample | 16039 / 6867 | Pass |
| two-segment r2v | 864x480 / 78 frames / 24 FPS / 3.25 s | 196.90 s | two remote operations | 15543 / 4787 | Pass; worker/FSDP/Gloo reuse |

Each Director pass produces exactly one unique remote operation and zero local diffusion samples. The real two-segment R2V run produces exactly two operations without another model load, Gloo initialization, or FSDP registration. A negative test connecting an R2V group while selecting `v2v` fails before GPU execution, proving that mode contracts are not silently ignored.

## 8. Legacy-workflow performance regression

Each legacy path ran once cold and three times warm. The gate uses the median of the three matched warm runs. No unexplained sampling regression exceeded 5%.

| Workflow | Current warm median | Historical reference | Change | Result |
|---|---:|---:|---:|---|
| MiniMax I2V Turbo8, 448x448 / 39 frames | E2E 109.44 s; sample 94.22 s | 107.35 / 96.15 s | E2E +1.95%; sample about -2.0% | Pass |
| MiniMax REF2VA Turbo4, 608x352 / 56 frames | E2E 82.96 s; sample 64.58 s | 82.07 / 67.14 s | E2E +1.08%; sample about -3.8% | Pass |
| LTX 2.3 two-stage, 1280x704 / 121 frames | E2E 392.73 s; stages 15.0 / 47.8 s/it | 15.0 / 45.6 s/it | stage two about +4.82% | Pass, close to the gate |
| O6 REF2VA safe-FP16, 1120x768 / 124 frames | 49.48 s/it | 54.22 s/it | about -8.7% | Pass |
| O6 REF2VA FP32, same geometry | 177.89 s/it | 185.20 s/it | about -3.95% | Pass |

An initial REF2VA sequence contained roughly 99-second E2E outliers, while remote submission to worker entry was only 0.2-0.4 seconds. New phase timing identified main-process VAE/preprocessing-model offload to host memory (normally 7.3-8.7 seconds, longer in the outlier) rather than Ray, Gloo, CUDA P2P, or rank skew. A matched repeat passed at an 82.96-second warm median. These timings are diagnostic-only and remain silent by default.

## 9. Resources, media, and reclamation

- Both GPUs reach high utilization during sampling; all received tensors are finite and the two ranks agree exactly.
- Representative Director, MiniMax, O6, and LTX media pass ffprobe decode; black intervals are zero, audio NaN/Inf is zero, and no decode errors occur.
- Same-checkpoint warm runs reuse workers and FSDP shards. FL2VA/REF2VA checkpoint changes recycle both workers as a pair.
- Final `RayKill` removes both workers, raylet, and GCS. GPU processes disappear and physical/virtual commit falls from about 76835/95561 MiB to 34196/44655 MiB. The roughly 700 MiB CUDA context per worker is released too.
- Raw logs, telemetry, and videos remain in the local evidence tree; Git stores maintainable conclusions and paths only.

## 10. Final conclusion

Task 8 is complete. O8 validates t2v, i2v, fl2v, r2v, v2v, rv2v, same-model Refine, and real multi-segment generation. Each pass executes exactly one FSDP + CUDA P2P diffusion operation on the two Ray workers with no duplicate local sample. Media, reuse, checkpoint switching, final reclamation, and all five legacy performance gates pass. This does not claim that CLIP, Video/Audio VAE, or mux/save are FSDP-sharded; those remain in the Director/ComfyUI main process.

## 11. Task 10 final release QA

| Check | Result |
|---|---|
| Full pytest | `418 passed, 1 warning, 30 subtests passed` in 85.21 seconds |
| Focused tests after O8 formatting | `41 passed` in 14.24 seconds |
| Ruff | All O8 Python files since the rollback tag pass lint; six new O8 files pass the formatter after scoped formatting |
| Director-installed startup | Port 8188 starts; both adapter and Director nodes are discoverable with the correct display name |
| Director-blocked startup | With only Raylight whitelisted, port 8188 starts; the adapter exists, Director does not, and Raylight has no import dependency |
| One ComfyUI session | Legacy I2V Turbo8 smoke succeeds in 260.51 s, followed without restart/manual cache clearing by Director T2V smoke in 310.52 s |
| Same-session remote sample | Director has one operation; eight remote steps run at 8.87 s/it; no second local diffusion |
| Checkpoint isolation | Legacy FP32-compute workers and Director safe-FP16 workers switch as a pair without stale model state |
| Final reclamation | RayKill removes workers/CUDA contexts; after stopping test services, GPU, Ray, and ComfyUI processes are all absent |
| Git/repository | `git diff --check` passes; only final format/docs remain before commit; Git pack is about 8.06 MiB and no runtime video/log enters Git |

A full-repository Ruff scan still reports 32 unused-import/variable findings that predate O8 in upstream or historical files. O8 deliberately avoids unrelated mass cleanup. The pre-commit Python runner is not installed, but its two configured hooks were run directly: O8 lint passes and new O8 files pass the formatter. Git `fsck` reports only ordinary dangling objects from historical commits/staging and no corruption.
