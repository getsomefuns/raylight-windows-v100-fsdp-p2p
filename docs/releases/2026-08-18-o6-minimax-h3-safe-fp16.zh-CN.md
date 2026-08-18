# O6 MiniMax H3 安全 FP16 升级与实验记录

日期：2026-08-18

状态：功能与质量验收通过；初始 4× 性能门槛未通过；继续保持实验性、显式启用

冻结提交：`30cbb69`（运行时代码树与稳定点 `bcc7198` 完全一致）

对比起点：`d49adc`

## 1. 版本结论

本次升级已经在原生 Windows、双 Tesla V100-SXM2-16GB、TCC、CUDA P2P/NVLink 和 FSDP CPU offload 环境中完成 MiniMax H3 的模型专用安全 FP16 路径。它不是全局强制 FP16：残差累积、条件投影和数值敏感输出继续使用 FP32，主要 Attention/MLP 分支及 V100 FP8 权重回退矩阵乘法使用 FP16。

两个正式工作流均不报错，双 rank 输出逐项一致且全部有限，生成视频为 1120×768、124 帧、24 FPS、5.167 秒，含 32 kHz 双声道 AAC；每个视频均有 124 个唯一帧哈希且未检测到黑帧区间。

| 工作流 | FP32 基线 | 初始安全 FP16 | 加速比 | 当前最佳正式结果 | 最佳加速比 | 4× 门槛 |
|---|---:|---:|---:|---:|---:|---:|
| I2V Turbo8 | 160.7195 s/it | 48.7121 s/it | 3.299× | 48.7121 s/it | 3.299× | ≤40.1799 s/it |
| REF2VA Turbo4 | 185.2034 s/it | 54.2174 s/it | 3.416× | 49.8680 s/it | 3.714× | ≤46.3008 s/it |

因此，本次升级的功能和质量目标已经完成，但不能宣称达到 4×。I2V 仍差约 21.2%，REF2VA 当前最佳值仍差约 7.7%。

## 2. 验证环境和计算口径

| 项目 | 固定条件 |
|---|---|
| GPU | 2× Tesla V100-SXM2-16GB，TCC，NVLink/P2P |
| 驱动 | 577.00 |
| Python | 3.10.11 |
| PyTorch / CUDA runtime | 2.7.0+cu126 / 12.6 |
| Ray | 2.57.0 |
| ComfyUI | v0.31.0-15-g62b3c94b |
| 分布式拓扑 | 2 ranks，Ulysses=2，Ring=1，FSDP CPU offload，Windows CUDA P2P ProcessGroup |
| 正式几何 | 1120×768，124 帧，24 FPS，5 秒配置 |
| I2V / REF2VA | Turbo8 / Turbo4，固定输入、提示词、模型、LoRA 和 seed |
| 采样口径 | 两个 rank 中较慢者的完整 sampling interval ÷ 精确步数；不使用单条 tqdm 滚动值 |

Windows 的物理内存、提交内存和分页文件峰值会受运行前系统状态影响。表中保留原始监控值，但判断优化时优先使用同规格采样时间、sampler node 时间、rank 一致性和媒体结果。

## 3. 已实装并保留的改动

### 3.1 MiniMax H3 模型专用安全 FP16（`051aff1`）

实装内容：

- 新增 RayUNETLoader 的 `fp16_h3_safe` 显式模式，不修改 ComfyUI 全局 dtype 白名单。
- 在每个 Ray worker 构造模型、应用 FSDP 之前安装幂等兼容层。
- 保持条件投影、残差流、调制/门控后的累积及敏感输出为 FP32。
- 将主 Attention/MLP 分支输入、FP8 回退反量化结果和 LoRA sidecar 矩阵乘法置于 FP16。
- FP8 仍为 checkpoint/FSDP shard 存储格式，不在加载时整体膨胀成 FP16/FP32。
- 增加两份独立实验工作流、dtype/rank/有限值诊断和回归测试。

保留理由：这是本轮主要收益来源。I2V 和 REF2VA 采样分别减少 69.69% 和 70.73%，sampler node 分别减少 68.03% 和 68.14%，同时媒体和双 rank 数值验收通过。它避免了此前“全局 FP16”产生黑视频或 NaN/Inf 的数值风险。

### 3.2 V100 FP16 GEMM 分块对齐（`434743c`）

实装内容：V100 FP8 fallback 在 FP16 路径上把输出分块宽度对齐到 8 的倍数，使主矩阵乘法更符合 Volta Tensor Core 的形状要求；默认临时块预算仍为 32 MiB。

保留理由：REF2VA 正式采样从 54.2174 降到 51.9470 s/it，额外改善 4.19%，相对 FP32 达到 3.565×。该改动只作用于 V100 FP16 fallback，不改变其他 GPU/dtype 路径。单次冷启动的端到端时间受 Ray 初始化和预处理波动影响，从 394.11 增至 397.14 秒，因此收益声明限定为采样阶段。

### 3.3 有界 Windows FSDP host registration（`fb7e4b0`、`6258ca0`）

实装内容：

- 可选择性地把限定容量的 FSDP CPU-offload shard 注册为 CUDA host memory。
- 按底层 storage 去重，记录成功、容量跳过和失败量。
- 注册生命周期仅覆盖 sampling；三个 sampler 返回或异常时均执行释放，避免长期锁页和显存/内存泄漏。
- 默认关闭；普通启动脚本没有开启它。

启用参数：

```powershell
$env:RAYLIGHT_FSDP_CPU_OFFLOAD_HOST_REGISTER = "1"
$env:RAYLIGHT_FSDP_CPU_OFFLOAD_HOST_REGISTER_MIB = "5120"
```

保留但不默认启用的理由：5 GiB 配置使 REF2VA 正式采样从对齐后的 51.9470 降到 49.8680 s/it，改善 4.00%，达到当前最佳 3.714×；P2P profile 的 collective control wait 从 2.732 秒降至 1.318 秒，submit 时间从 3.669 秒降至 2.244 秒。但该次冷启动的模型加载、Ray 初始化和端到端时间反而变慢，且锁页容量过大时会加重系统内存/分页压力。因此它是已实装的可选实验优化，不是默认推荐项，也未用于宣称完整工作流提速。

### 3.4 可选 CUDA sampling profiler（`7a35842`）

实装内容：rank 0 可用 `RAYLIGHT_TORCH_PROFILE=1` 启用一次 CUDA/CPU profile；默认关闭，profile 建立或报告失败不会改变有效采样结果。

保留理由：这是诊断能力，不是性能优化。它定位出本工作负载 CUDA self time 约 62.9% 在 Attention、16.9% 在矩阵乘法、14.3% 在 copy；其中 pageable H2D 约 7.7%，P2P 约 0.7%。这证明当前瓶颈主要不是 NVLink/P2P 带宽，也解释了为何继续只优化通信难以达到 4×。

## 4. 正式同规格数据

### 4.1 时间和速度

| 运行 | s/it | 对 FP32 加速 | 端到端 | Sampler node | 模型加载 | 预处理 | Ray 初始化 | VAE 解码 | 视频保存 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I2V FP32 基线 | 160.7195 | 1.000× | 1463.670 s | 1318.626 s | 20.985 s | 33.532 s | 36.197 s | 42.806 s | 9.183 s |
| I2V 安全 FP16 | 48.7121 | 3.299× | 563.283 s | 421.513 s | 19.371 s | 31.865 s | 40.927 s | 38.321 s | 8.627 s |
| REF2VA FP32 基线 | 185.2034 | 1.000× | 932.031 s | 774.778 s | 20.982 s | 42.839 s | 37.985 s | 43.885 s | 9.174 s |
| REF2VA 安全 FP16 | 54.2174 | 3.416× | 394.110 s | 246.844 s | 19.400 s | 36.134 s | 37.614 s | 42.984 s | 8.965 s |
| REF2VA + GEMM 对齐 | 51.9470 | 3.565× | 397.136 s | 245.483 s | 21.136 s | 37.769 s | 37.859 s | 43.385 s | 9.631 s |
| REF2VA + 对齐 + 5 GiB 注册 | 49.8680 | 3.714× | 424.241 s | 249.977 s | 30.766 s | 40.122 s | 50.819 s | 42.055 s | 8.893 s |

### 4.2 资源峰值

| 运行 | GPU0 / GPU1 显存峰值 | GPU0 / GPU1 利用率峰值 | GPU0 / GPU1 功率峰值 | 物理内存峰值 | 提交内存峰值 | 分页文件峰值 |
|---|---:|---:|---:|---:|---:|---:|
| I2V FP32 基线 | 16162 / 16214 MiB | 100 / 100% | 366.31 / 369.20 W | 65380.5 MiB | 128225.8 MiB | 14495.7 MiB |
| I2V 安全 FP16 | 16163 / 14862 MiB | 100 / 100% | 355.42 / 367.87 W | 65384.3 MiB | 129062.9 MiB | 1562.8 MiB |
| REF2VA FP32 基线 | 16237 / 16137 MiB | 100 / 100% | 366.52 / 365.98 W | 65435.6 MiB | 129546.9 MiB | 15549.3 MiB |
| REF2VA 安全 FP16 | 16237 / 15826 MiB | 100 / 100% | 360.28 / 362.35 W | 65432.3 MiB | 128221.3 MiB | 16403.5 MiB |
| REF2VA + GEMM 对齐 | 16237 / 15826 MiB | 100 / 100% | 370.02 / 370.17 W | 65438.8 MiB | 130381.7 MiB | 1455.1 MiB |
| REF2VA + 对齐 + 5 GiB 注册 | 16237 / 15836 MiB | 100 / 100% | 368.37 / 371.95 W | 65439.6 MiB | 129369.8 MiB | 1454.2 MiB |

显存峰值没有按 FP16 比例减半：FSDP shard、输出/残差、P2P buffer、VAE 和 ComfyUI 主进程仍占用显存；安全 FP16 的主要收益是计算吞吐和临时张量，而不是把整个 checkpoint 永久变成半精度副本。

### 4.3 P2P 和正确性

| 运行 | collective calls | 总 payload | 远端字节 | control wait | submit | 两 rank 精确一致 | 媒体验收 |
|---|---:|---:|---:|---:|---:|---|---|
| I2V 安全 FP16 | 5282 | 545.95 GB | 272.97 GB | 1.274 s | 3.064 s | PASS | PASS |
| REF2VA 安全 FP16 | 2650 | 285.73 GB | 142.86 GB | 4.008 s | 4.993 s | PASS | PASS |
| REF2VA + GEMM 对齐 | 2650 | 285.73 GB | 142.86 GB | 2.732 s | 3.669 s | PASS | PASS |
| REF2VA + 5 GiB 注册 | 2650 | 285.73 GB | 142.86 GB | 1.318 s | 2.244 s | PASS | PASS |

以上为日志中的十进制 GB。所有正式安全 FP16 输出均为有限张量，rank 0/1 的 output 和 denoised output 比较均为 exact、最大绝对差 0。

## 5. 实验性尝试和取舍

### 5.1 正式 1120×768、124 帧实验

| 尝试 | s/it | 对 FP32 加速 | 端到端 | GPU0 / GPU1 显存 | 物理 / 提交 / 分页峰值 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 初始安全 FP16 | 54.2174 | 3.416× | 394.110 s | 16237 / 15826 MiB | 65432 / 128221 / 16404 MiB | 实装 |
| FP8 临时块 128 MiB | 88.9425 | 2.082× | 537.555 s | 16239 / 16238 MiB | 65435 / 129138 / 14300 MiB | 拒绝；大块增加峰值并破坏流水 |
| FP8 临时块 64 MiB | 64.9555 | 2.851× | 458.682 s | 16237 / 15826 MiB | 65436 / 130419 / 19479 MiB | 拒绝；比 32 MiB 默认慢 19.8% |
| LoRA chunk 64 MiB | 58.6683 | 3.157× | 429.454 s | 16237 / 15826 MiB | 65438 / 129742 / 14383 MiB | 拒绝；比初始慢 8.2% |
| GEMM 宽度对齐 8 | 51.9470 | 3.565× | 397.136 s | 16237 / 15826 MiB | 65439 / 130382 / 1455 MiB | 实装；采样改善 4.19% |
| 对齐 + 5 GiB host register | 49.8680 | 3.714× | 424.241 s | 16237 / 15836 MiB | 65440 / 129370 / 1454 MiB | 机制实装、默认关闭；采样改善但冷启动总耗时变差 |
| 全量 pin-memory | 无完成结果 | — | 约 478 s 后无 run | — | — | 拒绝；Windows/Ray 路径未稳定完成 |

### 5.2 608×352、39 帧短测

| 尝试 | s/it | 端到端 | GPU0 / GPU1 显存 | 物理 / 提交 / 分页峰值 | 相对 5 GiB 短测 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| GEMM 对齐 | 10.8802 | 172.100 s | 15949 / 4331 MiB | 65344 / 129487 / 959 MiB | 慢 25.3% | 作为无注册参考 |
| 4 GiB host register | 9.9379 | 198.367 s | 15949 / 4340 MiB | 65439 / 131716 / 1767 MiB | 慢 14.4% | 有收益但不如 5 GiB |
| 5 GiB scoped register | 8.6862 | 169.355 s | 15949 / 4342 MiB | 65421 / 131247 / 1066 MiB | 基准 | 最佳短测 |
| 6 GiB scoped register | 11.2885 | 202.043 s | 15949 / 4344 MiB | 65439 / 128986 / 8808 MiB | 慢 30.0% | 拒绝；分页压力和收益反转 |
| single-ring direct fast path | 11.1691 | 192.076 s | 15949 / 4416 MiB | 65440 / 130725 / 4652 MiB | 慢 28.6% | 已 revert；正确同步语义下实际更慢 |
| 128 MiB FSDP forward prefetch | 10.1368 | 182.809 s | 15949 / 4342 MiB | 65437 / 129338 / 7024 MiB | 不可归因 | 日志 `configured=0`，候选完整撤回 |

forward-prefetch 运行没有真正配置任何模块，因此 10.1368 s/it 只是一次无预取波动，不能作为该方案的效果数据。`15eadfa` 的新增实现已由 `30cbb69` 完整撤回，当前源码和已部署节点均无残留。

### 5.3 内核、Attention 和拓扑微基准

| 尝试 | 观测 | 未实装原因 |
|---|---|---|
| FP8 fallback 32→128 MiB microbench | 49.92→48.81 ms，单投影仅约 2.3%；峰值增加约 169 MiB | 正式工作流反而从 54.22 恶化到 88.94 s/it；局部 microbench 不能代表 FSDP 流水 |
| 直接写入输出 slice | 单投影约 2% 收益 | 总体收益不足，且增加实现复杂度，未形成正式收益证据 |
| Ulysses vs KV-allgather Attention | 541.34 vs 546.11 ms | KV gather 慢约 0.9%，不替换现有 Ulysses |
| xFormers auto / torch SDPA / 无 LSE 变体 | 约慢 0.5% 或不满足接口语义 | 无可测收益，部分路径缺少 Raylight/xFuser 所需 LSE/同步行为 |
| SageAttention 1.0.6 | sm_70 INT8 `tl.dot` lowering 失败 | Triton/内核不支持当前 V100 Windows 路径 |
| flash-attn-v100 Windows wheel | 默认路径慢约 48.6%；MMA_NATIVE 更慢且误差更大 | 不满足速度和数值标准，测试环境已隔离，不进入项目依赖 |
| single-ring bypass microbench | 约 0.5% 局部收益 | 真实双卡同步短测慢 28.6%；不能以跳过 `use_sync=True` 换取表面速度 |

## 6. 为什么还没有达到 4×

- 安全 FP16 已经消除了主要 FP32 GEMM 成本，但 50 个 block 的 Attention 仍占 CUDA self time 约 62.9%。
- FSDP CPU offload 每步仍需要权重 all-gather/H2D；即使 host registration 降低控制和提交开销，也不能消除传输与同步。
- VAE 解码、预处理、Ray actor 初始化和视频保存没有由双卡采样并行加速。
- P2P/NVLink 数据通路已经工作且只占 profile 中很小的 self time；继续扩大 P2P buffer 或替换通信拓扑不会自动解决 Attention 计算瓶颈。
- 两张 16 GiB V100 的显存已经接近上限，激进预取、大块反量化和额外缓存容易转化成 OOM、碎片或系统分页。

## 7. 当前交付边界

| 项目 | 当前状态 |
|---|---|
| `fp16_h3_safe` loader 和两份实验工作流 | 已实装，显式选择时启用 |
| FP32 数值岛 + FP16 Attention/MLP | 已实装并通过正式质量验证 |
| V100 GEMM 宽度对齐 | 已实装，自动作用于匹配路径 |
| 5 GiB FSDP host registration | 机制已实装，默认关闭，仅建议受控测试 |
| CUDA profiler | 已实装，默认关闭，仅诊断 |
| 全局 MiniMax/LTX FP16 allowlist | 未修改，也不应修改 |
| 4× 初始性能门槛 | 未通过 |
| 11× 后续优化目标 | 未通过，仍是研究目标而非当前能力 |

可直接载入的安全 FP16 工作流：

- `example_workflows/Minimax_H3_I2V_Windows_V100_FSDP_Turbo8_FP16_Experimental.json`
- `example_workflows/Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4_FP16_Experimental.json`

工作流中的 RayUNETLoader 必须保持 `weight_dtype=fp16_h3_safe`。普通 `default` 工作流仍是 FP32-compute 对照组；不要把全局 `fp16` 或 dtype allowlist 修改当作本功能的替代。

## 8. 冻结验证

- 相关 safe-FP16、FP8 fallback、host registration、sampling profiler 和 sampler lifecycle 测试：64 passed，另有 2 个 subtests passed。
- 六个核心运行时模块通过 `py_compile`。
- 中英文升级记录各 210 行、20 个标题、9 个表格，结构一致；10 份相关 Markdown 的本地相对链接检查为 0 缺失。
- 独立代码审查：High 0、Medium 0、Low 0。
- 运行时代码冻结树为 `30cbb6965d8f956fd9abb462a8103862097e7056`；文档提交完成后重新部署，并要求源码 HEAD 与 ComfyUI 部署标记一致。

## 9. 证据索引

原始 benchmark JSON、日志和 monitor CSV 保留在本机 `logs/minimax-h3/o2/<run>/`。核心运行：

- `20260818-155526-i2v-full-o6-baseline-fp32-p2p256`
- `20260818-162159-ref2va-full-o6-baseline-fp32-p2p256`
- `20260818-184842-i2v-full-o6-safe-fp16-full-reviewed`
- `20260818-185943-ref2va-full-o6-safe-fp16-full-reviewed`
- `20260818-201232-ref2va-full-o6-safe-fp16-align8-full-reviewed`
- `20260818-212923-ref2va-full-o6-safe-fp16-hostreg5g-scoped-full`
- `20260818-214113-ref2va-full-o6-safe-fp16-hostreg5g-profile-full-4step`

本文件只汇总可维护结论，不把大量原始日志复制进仓库。英文版本见 [2026-08-18-o6-minimax-h3-safe-fp16.en.md](2026-08-18-o6-minimax-h3-safe-fp16.en.md)。
