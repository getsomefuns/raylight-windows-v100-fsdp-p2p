# Windows 双 V100 DynamicVRAM / PyTorch 2.8 测试汇总

## 文档状态

- 阶段：TCC 设备发现补丁通过，但 V100 缺少 CUDA VMM，候选未推广
- 设计：`docs/superpowers/specs/2026-09-03-dynamic-vram-runtime-upgrade-design.md`
- 计划：`docs/superpowers/plans/2026-09-03-dynamic-vram-runtime-upgrade.md`
- 当前结论：`0.4.13+tcc1` 解决了 Windows TCC 设备发现和 NVML 压力监控，但两张 V100 均报告不支持 CUDA Virtual Memory Management。ComfyUI 虽打印 DynamicVRAM 已启用，首个真实 Text Encoder 在创建最小 VBAR 时即失败。正式 `Python310` 保持 Torch 2.7/cu126，候选不得推广。

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

### 候选环境迁移与静态门禁

原 `Python310` 已完整移入 `runtime-backups/Python310-torch2.7.0-cu126-20260903`，再复制回规范路径作为候选。安装前两份目录均为 104,584 个文件、8,592,682,973 字节，统计完全一致。回滚清单保存了 285 条 `pip freeze` 记录、升级前 `pip check`、Torch 环境信息和 xFormers 信息。

| 项目 | Torch 2.8 候选结果 |
|---|---|
| PyTorch / CUDA runtime | 2.8.0+cu128 / 12.8 |
| torchvision / torchaudio | 0.23.0+cu128 / 2.8.0+cu128 |
| xFormers | 0.0.32.post2 |
| Torch arch | 含 `sm_70` |
| GPU | 两张 V100 均为 capability 7.0 |
| xFormers FP16 forward | 两张 V100 真实前向通过，输出形状、dtype 与有限值检查通过 |
| CUDA IPC round-trip | 父/子进程跨进程通过 |
| flash-attn | 候选中未安装，无 Torch ABI 包需要移除 |
| 新增依赖冲突 | Pillow 12.3.0 与现有 MoviePy 2.1.1、Streamlit 1.50.0 不兼容；其余五项是升级前已记录冲突 |

虽然 xFormers wheel 的构建信息未列出 `sm_70`，其 CUTLASS 路径在两张 V100 上均完成了真实 FP16 attention forward，因此没有把构建列表当成失败结论。

### 官方 0.4.13 的失败门禁：AIMDO / TCC

独立探针和最小 CUDA 上下文探针均确认 `comfy_aimdo.control.init_devices()` 返回 `False`。启用 AIMDO 详细日志后得到直接根因：

- AIMDO 的 CUDA hooks 能安装；
- 在 TCC V100 上调用 `cuDeviceGetLuid` 返回 CUDA 801 `operation not supported`；
- 随后 Windows WDDM adapter 初始化失败，AIMDO 清理 hooks 并拒绝注册设备；
- 单卡、双卡以及 NVML pressure 开/关均不能改变结果。

这不是 Torch 2.8、xFormers、GPU 可见性或探针缺少 CUDA context 导致。当前官方 Windows 实现把 WDDM/DXGI 预算接口作为初始化硬条件，而本项目为了 CUDA P2P/NVLink 保持两张 V100 为 TCC，两者直接冲突。

官方仓库的 PR #68 曾验证把 WDDM 初始化改为非致命、退回 `cuMemGetInfo` 可以运行，但未被合并；维护者指出缺少 WDDM 预算信息可能导致共享显存溢出和性能退化。因此本项目没有采用无条件绕过，而是把“NVML 压力设备已成功映射”作为 TCC 注册的硬条件。

### `0.4.13+tcc1` 构建与候选验证

早期失败候选已原地迁移为 `runtime-candidates/Python310-torch2.8.0-cu128-aimdo-tcc1`，不再使用语义错误的 `runtime-failed` 目录。正式 Torch 2.7 环境未改变。

| 验证项 | TCC 兼容候选结果 |
|---|---|
| 本地包 | `comfy-aimdo 0.4.13+tcc1`；满足当前 ComfyUI 的 `==0.4.13` 版本语义 |
| 固定来源 | comfy-aimdo `3f4259a...`；Detours `e4bfd6b...`；Windows SDK NuGet 10.0.26100.1 |
| 构建产物 | wheel、DLL、补丁均有 SHA-256，并由 `BUILD-MANIFEST.json` 复验 |
| 后端策略 | WDDM 优先；WDDM 不可用时仅允许已成功映射的 NVML；无 NVML 必须失败关闭 |
| Torch / GPU | 2.8.0+cu128，含 `sm_70`；两张 V100 capability 7.0 |
| xFormers / CUDA IPC | 两卡 FP16 forward 与父子进程 CUDA IPC 均通过 |
| AIMDO | GPU0/GPU1 均注册；禁用 NVML 的反向探针按设计拒绝注册 |
| CUDA VMM | GPU0/GPU1 的 `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` 均为 0 |
| 最小 VBAR | 32 MiB `cuMemAddressReserve` 返回 CUDA 801 `operation not supported` |
| DynamicVRAM | 仅初始化标志可置为 `true`；加入 VBAR 实际分配后为 `false`，不能用于模型 |
| 节点启动 | Raylight 与 MiniMax H3 Director 导入成功；KJNodes 仅有不相关的可选 Triton VAE 警告 |
| 聚焦自动测试 | 14 passed；另有 Windows pytest 临时目录清理警告，无测试失败 |

真实非 Director I2V 使用完整 MiniMax H3 配置、448×448、39 帧、Turbo8、safe-FP16。工作流在 Text Encoder 创建 VBAR 时于 3.02 秒内失败，未进入 Ray worker、FSDP/P2P 或采样。因此没有生成视频，也没有性能或质量数据。原始候选 JSON 与该次失败日志保存在本机 `runtime-results`；正文不复制长日志。

### 阶段决定

`0.4.13+tcc1` 设备发现补丁本身有复现价值，但不足以让 V100 使用官方 DynamicVRAM 架构。Gate D 在真实模型负载前失败，后续 P2P/FSDP、I2V/REF2VA、性能与质量测试按计划停止。若继续，需要另立“V100 Unified Memory/非 VMM 后端”设计；它不是当前补丁的安全小改。正式版本锁、README 支持结论和 `Python310` 不变，当前没有推送 GitHub。
