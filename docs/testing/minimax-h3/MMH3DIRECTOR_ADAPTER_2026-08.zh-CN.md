# MiniMax H3 Director MODEL 适配器：O8 测试与验收记录

[简体中文](MMH3DIRECTOR_ADAPTER_2026-08.zh-CN.md) | [English](MMH3DIRECTOR_ADAPTER_2026-08.md)

## 1. 验证目标

在不修改 MiniMaxH3Director、既有 Raylight 节点 schema 和旧工作流 JSON 的前提下，为 Director 提供显式可选的 Ray `MODEL` 代理，使 Director 发起的扩散采样只在两个 Ray Worker 上执行一次，并继续使用已验收的 Windows FSDP2 + CUDA P2P/NVLink 数据面。

本文从开发前基线开始持续维护。原版 Director 的单卡边界、单段与多段结果见 [原版基线报告](MMH3DIRECTOR_UPSTREAM_BASELINE_2026-08-22.zh-CN.md)。

## 2. O8 开发前基线（2026-08-23）

### 2.1 软件与硬件门槛

| 项目 | 结果 |
|---|---|
| 基线提交 | `f5c455d`（适配器代码尚未开始） |
| GPU | 2× Tesla V100-SXM2-16GB，TCC，驱动 577.00 |
| Python / PyTorch | 3.10.11 / 2.7.0+cu126 |
| 关键依赖 | torchvision 0.22.0+cu126；torchaudio 2.7.0+cu126；xformers 0.0.30；Ray 2.57.0；xFuser 0.4.5；yunchang 0.6.4 |
| 全量测试 | `342 passed, 1 warning, 30 subtests passed`，68.19 秒 |
| P2P 正确性 | 同步/异步、分块尾部、FP32/FP16/BF16/UINT8 全部通过 |
| P2P 带宽 | 主要结果约 59.2–61.6 GiB/s/方向，高于 50 GiB/s 门槛 |

基线首次运行暴露了模型下载脚本对 `Get-FileHash` 的环境依赖。测试结果为 `341 passed, 1 failed`；将哈希实现改为直接使用 .NET SHA-256 后，聚焦测试和全量测试均通过。该修复不改变 Raylight 运行时、节点接口或工作流。

### 2.2 旧公共节点 schema 快照

| 节点 ID / 显示名 | Required 输入 | Optional 输入 | 输出 | 函数 |
|---|---|---|---|---|
| `RayInitializer` / Ray Init Actor | ray_cluster_address, ray_cluster_namespace, GPU, ulysses_degree, ring_degree, cfg_degree, dp_degree, sync_ulysses, clear_vram_after_sampling, FSDP, FSDP_CPU_OFFLOAD, XFuser_attention, skip_comm_test, use_mmap | wait_for | `RAY_ACTORS_INIT` (`ray_actors_init`) | `spawn_actor` |
| `RayInitializerAdvanced` / Ray Init Actor (Advanced) | ray_cluster_address, ray_cluster_namespace, GPU, GPU_SELECT, ulysses_degree, ring_degree, clear_vram_after_sampling, cfg_degree, dp_degree, sync_ulysses, FSDP, FSDP_CPU_OFFLOAD, XFuser_attention, skip_comm_test, use_mmap | wait_for, ray_object_store_gb, ray_dashboard_address, torch_dist_address | `RAY_ACTORS_INIT` (`ray_actors_init`) | `spawn_actor` |
| `RayUNETLoader` / Load Diffusion Model (Ray) | unet_name, weight_dtype, ray_actors_init | lora | `RAY_ACTORS` (`ray_actors`) | `load_ray_unet` |
| `RayKill` / Kill Ray | ray_actors, kill_mode | — | 无 | `kill_ray` |
| `XFuserSamplerCustomAdvanced` / XFuser SamplerCustom Advanced | add_noise, noise_seed, guider, sampler, sigmas, latent_image | — | `LATENT`, `LATENT`, `RAY_ACTORS` | `ray_sample` |
| `XFuserSamplerCustom` / XFuser SamplerCustom | ray_actors, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image | — | `LATENT`, `RAY_ACTORS` | `ray_sample` |

所有节点类别保持原值：前四项为 `Raylight`，两种采样器为 `Raylight/extra/custom_sampling/samplers`。任务 1 将这些内容固化为自动化契约测试。

### 2.3 已验收工作流 JSON 哈希

| 工作流 | SHA256 |
|---|---|
| `LTX2_3_i2v_Raylight_Windows_FSDP_5s.json` | `2b0b6e3bab1a7adb7f5e3a8b627b0bfe5535b3d8b3e949ecb7644225f71eef84` |
| `Minimax_H3_I2V_Windows_V100_FSDP_Turbo8.json` | `2f261d43e0477e21165793edfade592e4d4eef948b0434c81db8180a8fc926a5` |
| `Minimax_H3_I2V_Windows_V100_FSDP_Turbo8_FP16_Experimental.json` | `faa485b80b87126bb45223f06c935f8372611f0d98efcd4106ee7111d3e41b02` |
| `Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4.json` | `af038101e582efd163d93f1fd22be7abe5d0c3a93b00b90c332f52d9ff2377ee` |
| `Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4_FP16_Experimental.json` | `fccd1b7ed6519cc81fa66f26a956061587f8cebce567f3ddef07ba0752a32f35` |

## 3. 当前结论

O8 的开发起点满足测试、双卡身份、TCC、依赖版本、P2P 正确性和带宽门槛。尚未开始适配器运行时代码，不能在本阶段宣称 Director 已经获得 Raylight FSDP 支持。
