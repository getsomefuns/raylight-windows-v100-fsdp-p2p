# MiniMax H3 Turbo 工作流使用说明

## 选择哪个工作流

| 用途 | 工作流 | 采样步数 | 定位 |
|---|---|---:|---|
| I2V 质量基线 | `Minimax_H3_I2V_Windows_V100_FSDP.json` | 20 | 保留更多采样步骤，适合最终质量对照 |
| I2V 速度预设 | `Minimax_H3_I2V_Windows_V100_FSDP_Turbo8.json` | 8 | 官方 FL2V Turbo LoRA，推荐日常试生成 |
| REF2VA 质量基线 | `Minimax_H3_REF2VA_Windows_V100_FSDP.json` | 20 | 完整质量基线 |
| REF2VA 速度预设 | `Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4.json` | 4 | 官方 REF2V Turbo LoRA，推荐日常试生成 |

Turbo 已通过技术正确性检查，但更少的采样步数不等于与 20 步视觉质量完全相同。最终使用哪个预设，应以生成结果的画面、动作和提示词遵循程度为准。

## 所需文件

输入图片位于 ComfyUI 的 `input` 目录：

- I2V：`minimax_h3_i2v_spear_portals.jpg`
- REF2VA：`minimax_h3_ref2va_green_robots.jpg`，工作流中的两个参考输入都使用这张图片

Turbo LoRA 位于 ComfyUI 的 `models/loras` 目录：

- `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`
- `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`

Diffusion Model、Text Encoder、Video VAE 和 Audio VAE 的完整清单及下载位置见 `scripts/minimax-h3/models.json`。

## 启动和运行

1. 在 PowerShell 中进入 `<ComfyUI>\custom_nodes\raylight`。
2. 执行 `.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath $PY -P2PCapacityMiB 256`。
3. 浏览器打开 `http://127.0.0.1:8188`。
4. 从 Raylight 的 `example_workflows` 目录载入所需 JSON。
5. 检查图片文件名和提示词，然后加入队列。

启动脚本已经设置原生 Windows 双 V100 所需的环境变量，并使用 `--disable-cuda-malloc`、`--reserve-vram 2`、Windows CUDA P2P transport 及默认关闭的详细诊断日志。两个 Ray worker、Ulysses 2、Ring 1、FSDP CPU offload 和 `TORCH_EFFICIENT` attention 由工作流中的 `RayInitializer` 配置。

启动脚本显示 `128/256/512 MiB` 三档，默认 256 MiB。只有排查性能或同步问题时才追加 `-EnableDiagnostics`；修改容量或诊断状态后必须重启 ComfyUI 和 Ray worker。

不要删除 `--disable-cuda-malloc`，否则 V100 的 VAE 路径可能出现 `operation not supported`。完整 REF2VA 运行时两卡采样显存约为 12.7/12.5 GiB，CPU offload 是 16GB V100 的必要容量模式。

## 已验收性能

参考机器冷启动完整工作流：I2V 从 20 步 642.01 秒降至 Turbo 8 步 387.98 秒，缩短 39.6%；REF2VA 从 20 步 1324.98 秒降至 Turbo 4 步 424.59 秒，缩短 68.0%。两份 Turbo 工作流均达到双卡 100% 峰值利用率，并通过黑屏、冻结帧、音频和 NaN 检查。详细数据见 `docs/testing/minimax-h3/FULL_TURBO_RELEASE_2026-08-17.md`。
