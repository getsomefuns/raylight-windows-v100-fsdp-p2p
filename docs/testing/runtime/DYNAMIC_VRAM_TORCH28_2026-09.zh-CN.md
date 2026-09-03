# Windows 双 V100 DynamicVRAM / PyTorch 2.8 测试汇总

## 文档状态

- 阶段：运行时升级执行中
- 设计：`docs/superpowers/specs/2026-09-03-dynamic-vram-runtime-upgrade-design.md`
- 计划：`docs/superpowers/plans/2026-09-03-dynamic-vram-runtime-upgrade.md`
- 当前结论：PyTorch 2.7 基线已固化；尚未升级正式环境

## 测试验证目标

验证 PyTorch 2.8/CUDA 12.8 DynamicVRAM 能否在 Windows、双 Tesla V100 TCC 环境中真实启用，并且不破坏 Raylight 的 xFormers、CUDA IPC/P2P、FSDP、safe-FP16、量化权重回退、真实视频质量和资源回收能力。

## 关注信息

- 运行时版本是否严格配套，Torch wheel 是否包含 `sm_70`；
- 两张 V100 是否都能执行 FP16 和 xFormers attention；
- AIMDO 是否注册两张设备，主进程和两个 worker 是否使用 DynamicVRAM；
- CUDA IPC、P2P all-to-all/all-gather 和大 payload 自动分块是否正确；
- FSDP/量化/LoRA/采样生命周期是否保持；
- 相同工作流的加载、编码、采样、解码、保存耗时和显存、RAM、commit/pagefile 峰值；
- 输出是否可解码、逐帧有限、无整段黑屏/噪声和音频 NaN/Inf。

## PyTorch 2.7 基线

采集前已停止属于本项目环境的 ComfyUI、Raylet、GCS 和 worker，GPU 无残留进程。基线探针由独立进程执行，原始 JSON 保存在本地 `runtime-results/dynamic-vram-torch27-baseline.json`，不作为仓库正文提交。

| 项目 | 结果 |
|---|---|
| Python | 3.10.11 |
| PyTorch / CUDA runtime | 2.7.0+cu126 / 12.6 |
| torchvision / torchaudio | 0.22.0+cu126 / 2.7.0+cu126 |
| xFormers | 0.0.30 |
| comfy-aimdo | 0.4.13 |
| GPU | 2× Tesla V100-SXM2-16GB，capability 7.0 |
| Torch arch | 含 `sm_70` |
| xFormers FP16 forward | 两张 V100 通过，输出有限 |
| CUDA IPC round-trip | 真实父/子进程跨进程通过 |
| AIMDO device init | 未注册设备；`init_devices` 返回未启用状态 |
| DynamicVRAM | 未启用，legacy ModelPatcher |

### 基线总结

1. 当前 2.7/cu126 栈的 V100、xFormers 和 CUDA IPC 基础能力正常。
2. DynamicVRAM 未启用是版本门禁和 AIMDO 初始化条件未满足，不是 GPU 不可见。
3. 初版探针曾在同一进程重建 CUDA IPC handle，得到 `invalid device context`；该测试方法不符合 CUDA IPC 的跨进程语义，已改成父/子进程 round-trip，修正后基线通过。这是探针修正，不是 Raylight 数据面修复。
4. 导入当前 ComfyUI 时 `comfy-kitchen` 提示 cu130 才能使用其优化 CUDA 操作；当前 V100 路径仍使用已验收回退。本阶段会单独验证 torch 2.8 下的量化权重路径，不能把该提示等同于工作流失败。

## 后续结果

### 候选 wheel

从 PyTorch 官方 cu128 索引只下载、不安装，并完成离线 `pip --dry-run --force-reinstall`。解析退出码为 0，选择了精确的四件套和正常依赖，没有选择 CPU Torch。

| wheel | 大小 | SHA256 |
|---|---:|---|
| torch 2.8.0+cu128 / cp310 / Windows | 3,461,427,674 B | `43938E9A...14D34F9E` |
| torchvision 0.23.0+cu128 | 7,540,139 B | `8EC6F228...FE5A6C3F` |
| torchaudio 2.8.0+cu128 | 4,672,929 B | `5D7A9D91...174C2366` |
| xformers 0.0.32.post2 / cp39-abi3 / Windows | 100,221,079 B | `87FA13F4...079E6F09` |

本地 wheel 缓存中的 `SHA256SUMS.json` 保存全部 14 个文件的完整字节数和哈希。二进制不提交仓库。

环境迁移、AIMDO、P2P/FSDP、真实工作流和回退结论将在执行相应 Gate 后逐项追加。
