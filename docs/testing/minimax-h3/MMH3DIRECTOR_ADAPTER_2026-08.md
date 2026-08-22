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

## 3. Current conclusion

The O8 starting point passes the test, dual-GPU identity, TCC, dependency-version, P2P correctness, and bandwidth gates. Adapter runtime development has not started, so this stage does not claim that Director already has Raylight FSDP support.
