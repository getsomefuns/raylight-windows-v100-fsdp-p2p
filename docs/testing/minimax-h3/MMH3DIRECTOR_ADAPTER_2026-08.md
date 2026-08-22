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

## 7. Current conclusion and open gates

O8 now proves that Director t2v uses the MODEL proxy to execute exactly one FSDP + CUDA P2P diffusion pass across both V100s. Full 720p-class cold and warm outputs are correct and materially faster than the upstream single-GPU baseline. Task 8 remains open: i2v, fl2v, r2v, v2v, rv2v, real multi-segment/Refine, old LTX/MiniMax regressions, and final full reclamation still require validation, so this report does not yet mark all of O8 complete.
