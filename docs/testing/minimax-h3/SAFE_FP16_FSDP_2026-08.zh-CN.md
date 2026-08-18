# MiniMax H3 O6 同规格 FP32 与安全 FP16 验证——2026-08-18

[简体中文](SAFE_FP16_FSDP_2026-08.zh-CN.md) | [English](SAFE_FP16_FSDP_2026-08.md)

## 验证目标

锁定 O6 安全 FP16 工作的本机对比分母，不使用任何外部或预测计时。两个基线都使用当前已验收 FP32-compute 实现、相同的 1120×768 几何、124 帧/24 FPS、已验收 O5 提示词和输入、FSDP CPU offload、两个 V100 rank 以及 Windows CUDA P2P。

## 固定条件

| 条件 | I2V | REF2VA |
|---|---|---|
| 变体 | FL2V Turbo8 | REF2V Turbo4 |
| 启动方式 | GPU 空闲后的独立冷启动 | GPU 空闲后的独立冷启动 |
| 几何 | 1120×768、124 帧、24 FPS | 1120×768、124 帧、24 FPS |
| 精度策略 | FP8-scaled 存储、V100 FP32 扩散计算 | FP8-scaled 存储、V100 FP32 扩散计算 |
| 并行模式 | 2 ranks、FSDP=true、CPU offload=true | 2 ranks、FSDP=true、CPU offload=true |
| Ulysses / Ring / CFG / DP | 2 / 1 / 1 / 1 | 2 / 1 / 1 / 1 |
| P2P 容量 | 每个 rank 显式设置 256 MiB | 每个 rank 显式设置 256 MiB |
| 源码/部署提交 | `394d1cffb668fd62d87ae632ef86e28e4d9c04b4` | 相同 |

发布默认值仍为 128 MiB。I2V 预跑证明该精确几何需要更大的 buffer：Ulysses 远端 payload 为 239,826,944 字节，134,217,728 字节的默认容量正确拒绝了它。仅 benchmark 使用的 256 MiB 设置可以覆盖实测 payload，不改变发布默认值。

## 阶段结果

除特别标注外，所有数值单位均为秒。

| 指标 | I2V Turbo8 | REF2VA Turbo4 |
|---|---:|---:|
| 端到端总耗时 | 1463.67 | 932.03 |
| Ray 初始化 | 36.20 | 37.99 |
| 模型加载节点 | 20.98 | 20.98 |
| ComfyUI 预处理 | 33.53 | 42.84 |
| worker 模型转入 GPU 最大耗时 | 4.66 | 1.29 |
| Sampler 节点 | 1318.63 | 774.78 |
| 最慢 rank 采样区间 | 1285.76 | 740.81 |
| 汇总采样 / 步 | 160.72 s/it | 185.20 s/it |
| 首个观测 tqdm 值 | 132.0 s/it | 98.1 s/it |
| 最终 tqdm 滚动值 | 153.0 s/it | 149.0 s/it |
| rank 采样时间差 | 4.52 | 1.23 |
| VAE 解码 | 42.81 | 43.88 |
| 视频创建 | 0.03 | 0.02 |
| 视频保存 | 9.18 | 9.17 |

O6 性能分母采用最慢 rank 的完整采样区间除以精确步数。该值包含所有被测采样工作，比选择某一条瞬时 tqdm 行更稳定。

## 资源和通信结果

| 指标 | I2V Turbo8 | REF2VA Turbo4 |
|---|---:|---:|
| GPU0 显存峰值 | 16,162 MiB | 16,237 MiB |
| GPU1 显存峰值 | 16,214 MiB | 16,137 MiB |
| GPU0/GPU1 利用率峰值 | 100% / 100% | 100% / 100% |
| GPU0/GPU1 功率峰值 | 366.31 / 369.20 W | 366.52 / 365.98 W |
| 物理内存使用峰值 | 65,380.48 MiB | 65,435.59 MiB |
| 提交内存峰值 | 128,225.80 MiB | 129,546.86 MiB |
| 分页文件使用峰值 | 14,495.65 MiB | 15,549.33 MiB |
| rank 数量 | 2 | 2 |
| 传输方式 | Windows CUDA P2P/NVLink | Windows CUDA P2P/NVLink |

两个 rank 都进入并完成采样。连续硬件监控显示两张 GPU 均达到 100% 利用率；进入 VAE 解码后，工作流回到普通单 GPU ComfyUI 路径。

## 媒体验收

| 检查项 | I2V Turbo8 | REF2VA Turbo4 |
|---|---|---|
| 视频 | H.264、1120×768、124 帧、24 FPS、5.167 秒 | 相同 |
| 音频 | AAC、32 kHz、双声道 | 相同 |
| 唯一解码帧哈希 | 124/124 | 124/124 |
| 黑屏区间 | 0 | 0 |
| 结果 | 通过 | 通过 |

## O6 数值门槛

安全 FP16 必须同时满足两个工作流的初始采样加速门槛。11× 一栏只作为后续优化目标，不阻塞第一个 release candidate：

| 工作流 | 基线 `B_i` | 初始 4× 门槛 `B_i / 4` | 11× 优化目标 `B_i / 11` | 初始要求 |
|---|---:|---:|---:|---|
| I2V Turbo8 | 160.7195 s/it | 40.1799 s/it | 14.6109 s/it | 不高于 40.1799 s/it |
| REF2VA Turbo4 | 185.2034 s/it | 46.3008 s/it | 16.8367 s/it | 不高于 46.3008 s/it |

模型加载、预处理、VAE 解码和视频保存均不得慢于对应基线，端到端总耗时必须改善。无论速度如何，数值正确性、有限张量、双 rank 完成、CUDA P2P 传输和媒体检查都必须通过。

## 已实装安全 FP16 结果

可选 worker 端实现通过 FP32 数值岛保护 FP16 Attention/MLP 计算，同时保留 FP8 checkpoint/FSDP 存储。两个完整工作流均由两个 rank 精确输出有限结果，具有 124/124 个唯一解码帧、没有检测到黑屏区间，输出 H.264 1120×768 视频和 32 kHz 双声道 AAC。

| 工作流 | FP32 基线 | 初始安全 FP16 | 初始加速 | 最佳已验收实验 | 最佳加速 | 4× 结果 |
|---|---:|---:|---:|---:|---:|---|
| I2V Turbo8 | 160.7195 s/it | 48.7121 s/it | 3.299× | 48.7121 s/it | 3.299× | 未通过 |
| REF2VA Turbo4 | 185.2034 s/it | 54.2174 s/it | 3.416× | 可选 5 GiB host registration：49.8680 s/it | 3.714× | 未通过 |

功能和质量目标通过，但 O6 尚未通过初始 4× 性能门槛，因此 `fp16_h3_safe` 继续保持显式实验模式。V100 GEMM extent 对齐作为匹配路径的自动改进保留。有界 host registration 已实现但默认关闭，因为它改善采样速度的同时，使同规格冷启动端到端结果变慢。

## 完整实验台账

本节是 O6 的测试总账，而不是只保留成功结果的摘要。状态含义如下：

- **已实装**：代码保留在当前运行树中，并有工作流或专项测试证据。
- **可选实装**：代码保留，但默认关闭；只有明确设置开关才生效。
- **实验后拒绝**：完成了可比较的工作流或短流程，结果证明不值得进入默认路径。
- **无有效性能结论**：尝试未形成完整、同规格或未被探针扰动的结果；只记录达到的阶段和排除原因。
- **仅微基准/后端探针**：没有进入 ComfyUI 正式工作流，不能用局部结果宣称工作流提速。

### A. 同规格完整工作流

除表内特别说明外，均为双 V100、FSDP CPU offload、Ulysses=2、Ring=1、Windows CUDA P2P、1120×768、124 帧。`s/it` 采用最慢 rank 的完整采样区间除以步数。

| 时间/实验 | 工作流 | 采样 s/it | 相对 FP32 | 端到端 / Sampler（秒） | GPU0/GPU1 显存峰值（MiB） | 物理 / 提交 / 分页峰值（MiB） | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| 15:55 FP32 基线 | I2V 8 步 | 160.7195 | 1.000× | 1463.670 / 1318.626 | 16162 / 16214 | 65380.5 / 128225.8 / 14495.7 | 对照组，已验收 |
| 16:21 FP32 基线 | REF2VA 4 步 | 185.2034 | 1.000× | 932.031 / 774.778 | 16237 / 16137 | 65435.6 / 129546.9 / 15549.3 | 对照组，已验收 |
| 18:48 初始安全 FP16 | I2V 8 步 | 48.7121 | 3.299× | 563.283 / 421.513 | 16163 / 14862 | 65384.3 / 129062.9 / 1562.8 | 已实装；质量通过，未达 4× |
| 18:59 初始安全 FP16 | REF2VA 4 步 | 54.2174 | 3.416× | 394.110 / 246.844 | 16237 / 15826 | 65432.3 / 128221.3 / 16403.5 | 已实装；质量通过，未达 4× |
| 19:09 FP8 fallback chunk=128 MiB | REF2VA 4 步 | 88.9425 | 2.082× | 537.555 / 387.751 | 16239 / 16238 | 65434.5 / 129137.9 / 14299.6 | 比初始 FP16 慢 64.0%，拒绝 |
| 19:36 FP8 fallback chunk=64 MiB | REF2VA 4 步 | 64.9555 | 2.851× | 458.682 / 298.009 | 16237 / 15826 | 65436.3 / 130418.7 / 19479.4 | 比初始 FP16 慢 19.8%，且分页更高，拒绝 |
| 19:45 LoRA fallback chunk=64 MiB | REF2VA 4 步 | 58.6683 | 3.157× | 429.454 / 282.343 | 16237 / 15826 | 65438.4 / 129741.8 / 14383.3 | 比初始 FP16 慢 8.2%，拒绝 |
| 20:12 GEMM extent 对齐 8 | REF2VA 4 步 | 51.9470 | 3.565× | 397.136 / 245.483 | 16237 / 15826 | 65438.8 / 130381.7 / 1455.1 | 比初始 FP16 采样快 4.19%，已实装 |
| 21:29 GEMM 对齐 + 5 GiB host registration | REF2VA 4 步 | 49.8680 | 3.714× | 424.241 / 249.977 | 16237 / 15836 | 65439.6 / 129369.8 / 1454.2 | 采样再快 4.00%，但冷启动端到端比对齐版慢 6.83%；可选实装、默认关闭 |

完整实验的节点阶段明细如下，便于把采样改善与模型加载、预处理、Ray 初始化、VAE 和保存退化区分开：

| 实验 | 模型加载 | 预处理 | Ray 初始化 | Sampler | VAE 解码 | 视频保存 |
|---|---:|---:|---:|---:|---:|---:|
| I2V 初始安全 FP16 | 19.371 | 31.865 | 40.927 | 421.513 | 38.321 | 8.627 |
| REF2VA 初始安全 FP16 | 19.400 | 36.134 | 37.614 | 246.844 | 42.984 | 8.965 |
| FP8 chunk=128 MiB | 19.302 | 38.784 | 36.195 | 387.751 | 45.882 | 9.032 |
| FP8 chunk=64 MiB | 19.825 | 44.617 | 39.761 | 298.009 | 45.095 | 9.469 |
| LoRA chunk=64 MiB | 19.558 | 40.520 | 36.452 | 282.343 | 38.854 | 9.955 |
| GEMM extent 对齐 8 | 21.136 | 37.769 | 37.859 | 245.483 | 43.385 | 9.631 |
| 5 GiB host registration | 30.766 | 40.122 | 50.819 | 249.977 | 42.055 | 8.893 |

| 实验 | GPU0/GPU1 利用率峰值 | GPU0/GPU1 功率峰值（W） | 参与结论 |
|---|---:|---:|---|
| I2V 初始安全 FP16 | 100% / 100% | 355.42 / 367.87 | 双卡参与 |
| REF2VA 初始安全 FP16 | 100% / 100% | 360.28 / 362.35 | 双卡参与 |
| FP8 chunk=128 MiB | 100% / 100% | 368.42 / 370.78 | 双卡参与，但吞吐退化 |
| FP8 chunk=64 MiB | 100% / 100% | 362.31 / 366.19 | 双卡参与，但吞吐退化 |
| LoRA chunk=64 MiB | 100% / 100% | 349.41 / 361.98 | 双卡参与，但吞吐退化 |
| GEMM extent 对齐 8 | 100% / 100% | 370.02 / 370.17 | 双卡参与，采样改善 |
| 5 GiB host registration | 100% / 100% | 368.37 / 371.95 | 双卡参与，采样改善但冷启动退化 |

所有完成的完整安全 FP16 输出均检查了双 rank 精确一致、张量有限、124/124 唯一帧和无黑屏区间。表中实验没有达到 4×；不能用短流程或微基准替代这一结论。

### B. 同一短流程筛选实验

除首行 I2V 外，以下均为 REF2VA、608×352、39 帧、4 步。它们用于快速筛选，不与 1120×768 正式性能门槛混算。

| 时间/实验 | 采样 s/it | 端到端（秒） | GPU0/GPU1 显存峰值（MiB） | 物理 / 提交 / 分页峰值（MiB） | 判断 |
|---|---:|---:|---:|---:|---|
| 18:38 I2V 初始安全 FP16 短流程 | 14.0002 | 244.787 | 15679 / 4179 | 65305.0 / 128316.5 / 1318.9 | 完整运行；97% / 97% GPU 利用率峰值，用于功能筛选 |
| 18:44 REF2VA 初始安全 FP16 短流程 | 14.1099 | 192.713 | 15949 / 4331 | 65439.1 / 130084.7 / 8388.3 | 完整运行；100% / 97% GPU 利用率峰值，用于功能筛选 |
| 20:07 GEMM 对齐短流程 | 10.8802 | 172.100 | 15949 / 4331 | 65344.3 / 129486.9 / 959.3 | 短流程参照 |
| 20:33 CUDA profiler 短流程 | 40.0292 | 301.169 | 15949 / 4331 | 65418.4 / 130346.9 / 1050.8 | 探针显著扰动时序，只证明可采集，不作性能判断 |
| 21:04 4 GiB host registration | 9.9379 | 198.367 | 15949 / 4340 | 65438.5 / 131715.8 / 1767.4 | 采样改善但端到端变慢；继续探索容量，不作为默认值 |
| 21:24 5 GiB scoped host registration | 8.6862 | 169.355 | 15949 / 4342 | 65420.5 / 131247.3 / 1066.0 | 短流程最佳，进入完整验证 |
| 22:29 6 GiB scoped host registration | 11.2885 | 202.043 | 15949 / 4344 | 65439.2 / 128986.2 / 8808.3 | 比 5 GiB 慢 30.0%，分页显著增加，拒绝 |
| 22:49 单 Ring 直接返回 | 11.1691 | 192.076 | 15949 / 4416 | 65439.5 / 130725.1 / 4652.0 | 比 5 GiB 慢 28.6%；语义还绕过 `use_sync=True`，代码已完整回退 |
| 23:04 forward prefetch=128 MiB 尝试 | 10.1368 | 182.809 | 15949 / 4342 | 65437.4 / 129338.2 / 7024.0 | 日志显示 `configured=0`，实际未启用 prefetch；该时序不能作为效果证据，功能提交及修复尝试均已回退 |

两类初始安全 FP16 短流程证明图在降低几何后可完整运行，但不参与同规格 4× 验收。

### C. 未形成有效完整结果的工作流尝试

| 时间/目录 | 实际做了什么、到达哪一阶段 | 观察结果 | 排除/不实装依据 |
|---|---|---|---|
| 15:44 I2V FP32、默认 128 MiB P2P | 工作流、双 worker、684 个 FSDP wrapper/rank 均已进入；在第一个 Ulysses collective 前停止 | 远端 payload 为 239,826,944 字节，大于 134,217,728 字节容量，抛出明确 `ValueError`；`runs=0` | 这是该几何的 benchmark 容量配置不够，不是模型或 P2P 失效。仅将正式测试改为 256 MiB 后通过；发布默认值仍为 128 MiB |
| 19:20 pinned-memory 尝试 | 双 worker、安全 FP16、Sampler 和 FSDP 模型准备均已进入，日志停在模型准备阶段；`runs=0` | 保存日志没有异常，也没有 `FSDP registered successfully`、采样进度、视频或 benchmark 结果；约 478 秒后被中止 | 无法证明 pinning 的速度或资源效果，也不能臆测具体根因；可靠性不足，不实装 |
| 19:33 FP8 chunk=64 首次调用 | 已进入 worker/Sampler/FSDP 准备，只有一个 rank 记录注册完成；没有完整采样 | `geometry=null`、`runs=0`，调用并非固定 1120×768 正式规格 | 无同规格、无完整运行，排除；随后用 19:36 的正确规格完整重跑作判断 |
| 22:29 6 GiB host registration 首次调用 | ComfyUI 服务启动完成，但 benchmark 没有提交/记录 API prompt | `runs=0`；日志中的 `_ray_runtime_env/__init__.py` 缺失警告在成功运行中也存在 | 警告不是因果证据；该次没有执行工作流，排除，随后在 22:29 的纠正运行中得到有效结果 |

共同出现的 `_ray_runtime_env` 临时目录缺失警告被标记为**非致因启动噪声**：同一警告存在于多个成功工作流中，因此不能把未完成尝试归因给它。

### D. Profiler、微基准和后端探针

| 实验 | 测试层级和结果 | 为什么没有进入默认实现 |
|---|---|---|
| CUDA profiler 短流程 | 40.0292 s/it、端到端 301.169 秒；探针本身显著改变时序 | 只用于验证 profiler 能采集，不能作为性能数据 |
| CUDA profiler 完整 REF2VA | 采样 98.6803 s/it、端到端 601.111 秒；显存峰值 16237/15836 MiB，物理/提交/分页峰值 65439.5/128575.9/7291.2 MiB，双 GPU 利用率峰值 100%/100%；self CUDA 总计 193.728 秒。efficient attention 121.852 秒（62.90%）、`aten::mm` 32.803 秒（16.93%）、`aten::copy_` 27.682 秒（14.29%） | 性能数字受 profiler 扰动；仅用于确定 Attention 是首要瓶颈、矩阵乘和复制是次要瓶颈。Profiler 保留为显式诊断开关、默认关闭 |
| FP8 fallback chunk 32→128 MiB 微基准 | 局部 projection 从约 49.92 ms 降至 48.81 ms，约快 2.3%，显存峰值增加约 169 MiB | 正式工作流反而从 54.2174 恶化到 88.9425 s/it；局部收益被全图调度/内存行为吞没，拒绝 |
| projection 直接写 output slice | 隔离 projection 约快 2% | 没有完整工作流证据，收益太小且增加实现复杂度，不实装 |
| KV-allgather Attention 拓扑 | Ulysses：H28/S33792 为 541.34 ms；KV-allgather：H56/Q16896/K33792 为 546.11 ms | 候选拓扑在同形状微基准慢约 0.9%，拒绝 |
| no-LSE Torch SDPA / xFormers auto | 局部约慢 0.5%，部分候选还不满足现有接口/LSE 语义 | 没有局部优势且存在接口语义差异，未进入工作流 |
| SageAttention 1.0.6 + triton-windows | 独立后端探针在 sm_70 INT8 `tl.dot` lowering 阶段失败，工作流未启动 | V100 后端不兼容；未加入项目依赖 |
| ai-bond flash-attn-v100 Windows wheel | 在隔离环境测试；默认实现慢约 48.6%，MMA_NATIVE 更慢且数值误差更大；未接入 ComfyUI | 速度和数值均不构成替换依据；隔离环境已移除，不是项目依赖 |
| 单 Ring merge bypass 微基准 | 局部约快 0.5% | 完整短流程慢 28.6%，且绕开同步语义；提交 `92b56bb` 后由 `bcc7198` 原样回退 |
| forward prefetch helper | helper 单元测试通过，但 FSDP2 动态包装改变了模型类名，运行时匹配为 0，日志为 `configured=0` | 没有实际启用，也就没有性能证据；`15eadfa` 及后续匹配修复尝试全部由 `30cbb69` 回退 |
| host registration 4/5/6 GiB | 4 GiB 有局部改善；5 GiB 短流程最佳且完整采样继续改善；6 GiB 退化并增加分页 | 只保留有界、显式可选实现；默认关闭，因为 5 GiB 的完整冷启动端到端仍退化 |

Profiler 中更细的复制数据为：pageable H2D 14.975 秒（4080 次）、pinned H2D 3.260 秒（2305 次）、P2P 1.345 秒（2654 次）。这说明当时采样瓶颈主要不在 P2P 链路；因此不能仅凭 NVLink 微基准带宽继续扩大 P2P chunk 来推导工作流提速。

P2P profile 也证明双卡确实在交换数据：初始安全 FP16 I2V 共 5282 次、总 payload 545.945 GB（远端 272.973 GB）；REF2VA 共 2650 次、总 payload 285.726 GB（远端 142.863 GB）。REF2VA 的 control wait / submit 从初始 4.008 / 4.993 秒，经 GEMM 对齐的 2.732 / 3.669 秒，降至 5 GiB host registration 的 1.318 / 2.244 秒；这支持“注册减少提交等待”，但不改变其冷启动默认关闭的决策。

## 中途修复与小型优化

### 1. MiniMax H3 安全 FP16 数值岛（已实装，`051aff1`）

- 新增显式 `fp16_h3_safe` loader 模式，避免全局修改 ComfyUI 精度策略；不启用时保持原路径。
- ComfyUI 对 V100 不支持的 checkpoint dtype 原本会回退到 FP32；实现只对 `MiniMaxH3` 构建过程设置模型专用 FP16 compute override，没有扩大全局 `supported_inference_dtypes`。
- Attention/MLP 主计算走 FP16，但 residual、condition projection 和关键输出投影保留 FP32 数值岛；K 输出投影使用 64、FC2 使用 256 的 2 的幂缩放，避免超过 FP16 的 65504 上限。
- 安全 Attention 输出缩放放在 LoRA 注入之后，因此 BF16 LoRA sidecar 也被保护；专项测试覆盖 120,000 和 LoRA 180,000 的极值并要求有限输出。
- 208 个 BF16 LoRA sidecar 均跟随 FP16 分支 dtype，没有不支持项。
- 使用 context-local 激活和幂等标记，避免 Ray worker 复用时重复包装；API signature guard 会在未来 ComfyUI 接口不兼容时明确失败，而不是静默算错。

### 2. GEMM extent 对齐与 bias 尾块修复（已实装，`434743c`）

- V100 FP16 fallback linear/addmm 的 chunk extent 自动对齐到 8 的倍数；REF2VA 完整采样从 54.2174 降至 51.9470 s/it，改善 4.19%。
- 同一改动修复 `fp8_addmm_fallback_chunked` 的 bias 广播/尾块问题：先把 bias 广播到完整输出形状，再按块切片；测试覆盖 `[8,1]` 尾块和多种 bias 形状。该项是正确性修复，不以速度作为保留理由。

### 3. 可选 CUDA 采样 profiler（诊断功能，`7a35842`）

- 只在 rank0、显式开启时采样；默认关闭。
- profiler 初始化或报告失败时仍返回有效工作流结果，采样一旦开始绝不为了 profiler 自动重跑，避免改变生成语义。
- 它用于定位 62.90% CUDA 时间在 efficient attention，不用于宣称工作流性能。

### 4. 有界 Windows FSDP host registration（可选实装，`fb7e4b0`）

- 按共享 storage 去重、从大到小注册并受 GiB 容量上限约束；捕获量化 `qdata+scale`，但不注册动态 `all_gather_inputs`。
- storage 引用保持到 unregister 完成，防止生命周期悬空。
- worker 清理从简单的 `self.model=None` 改为 `_free_current_model()`，保证反注册和模型资源清理发生。
- 5 GiB 可降低 P2P control wait/submit 并改善采样，但完整冷启动端到端退化 6.83%，所以代码保留、默认关闭。

### 5. Host registration 生命周期 bug（已修复，`6258ca0`）

- 初版注册跨越整个模型生命周期，持续占用 pinned/registered host memory，扩大了非采样阶段的资源压力。
- 修复后只在三个 sampler 方法的采样区间注册；正常返回和异常退出都会先同步再反注册，缓存 FSDP 模型下次采样时重新注册。
- cleanup 失败不能覆盖原始采样异常。该修复直接关系到可靠性和内存生命周期，并非单纯调参。

### 6. 单 Ring 快路语义回归（已发现并回退，`92b56bb` → `bcc7198`）

- 原意是省掉 world-size=1 的冗余 merge，微基准约快 0.5%。
- 实际实现同时绕开 Raylight/Yunchang 所需的 `use_sync=True` 路径，短流程慢 28.6%。
- 回退后代码树恢复到实验前状态；该优化不在成品中。

### 7. FSDP forward prefetch 匹配失效（已发现并回退，`15eadfa` → `30cbb69`）

- helper 级测试通过，但 FSDP2 动态包装后类名变化，运行时未匹配任何 block，实际配置数为 0。
- 这次工作流只证明“配置没有生效”，不能拿 10.1368 s/it 判断 prefetch 快慢。
- 为避免留下未被真实验证的死开关，功能提交和针对动态类的修复尝试全部撤销。

## 当前实现状态与提交对应

| 能力/尝试 | 当前树 | 提交 | 依据 |
|---|---|---|---|
| MiniMax H3 `fp16_h3_safe` 数值保护 | 保留，显式开启 | `051aff1` | 两个完整工作流质量通过 |
| V100 FP16 GEMM extent 对齐和 bias 修复 | 保留，匹配路径自动启用 | `434743c` | 完整 REF2VA 改善且专项测试通过 |
| CUDA sampling profiler | 保留，默认关闭 | `7a35842` | 诊断用途，不污染默认时序 |
| 有界 host registration | 保留，默认关闭 | `fb7e4b0`, `6258ca0` | 采样改善、冷启动端到端退化 |
| chunk=64/128、pinned、6 GiB | 不实装为默认策略 | 无保留配置 | 完整退化、未完成或短流程退化 |
| 单 Ring fast path | 已完整回退 | `92b56bb`, `bcc7198` | 工作流退化且同步语义错误 |
| forward prefetch | 已完整回退 | `15eadfa`, `30cbb69` | 工作流中实际配置为 0，无有效证据 |
| SageAttention / flash-attn-v100 | 不属于项目依赖 | 无 | 后端探针失败或明显退化 |

## 证据索引

原始日志、遥测 CSV、API prompt 和 benchmark JSON 保留在本机 `E:\ComfyUI-py310\logs\minimax-h3\o2`。以下索引包含成功、失败、未完成和被探针扰动的 O6 尝试：

| 类别 | 证据目录 |
|---|---|
| 默认 P2P 容量失败 | `20260818-154455-i2v-full-o6-baseline-fp32/` |
| FP32 正式基线 | `20260818-155526-i2v-full-o6-baseline-fp32-p2p256/`；`20260818-162159-ref2va-full-o6-baseline-fp32-p2p256/` |
| 安全 FP16 短流程 | `20260818-183849-i2v-full-o6-safe-fp16-smoke-reviewed/`；`20260818-184408-ref2va-full-o6-safe-fp16-smoke-reviewed/` |
| 安全 FP16 正式完整运行 | `20260818-184842-i2v-full-o6-safe-fp16-full-reviewed/`；`20260818-185943-ref2va-full-o6-safe-fp16-full-reviewed/` |
| chunk / pinned 实验 | `20260818-190934-ref2va-full-o6-safe-fp16-full-chunk128/`；`20260818-192040-ref2va-full-o6-safe-fp16-full-pinned/`；`20260818-193319-ref2va-full-o6-safe-fp16-full-fp8chunk64/`；`20260818-193659-ref2va-full-o6-safe-fp16-full-fp8chunk64-1120x768/`；`20260818-194555-ref2va-full-o6-safe-fp16-full-lorachunk64-1120x768/` |
| GEMM 对齐 | `20260818-200751-ref2va-full-o6-safe-fp16-align8-smoke/`；`20260818-201232-ref2va-full-o6-safe-fp16-align8-full-reviewed/` |
| Profiler | `20260818-203313-ref2va-full-o6-safe-fp16-profile-ref-smoke/`；`20260818-214113-ref2va-full-o6-safe-fp16-hostreg5g-profile-full-4step/` |
| Host registration 4/5 GiB | `20260818-210458-ref2va-full-o6-safe-fp16-hostreg4g-smoke/`；`20260818-212455-ref2va-full-o6-safe-fp16-hostreg5g-scoped-smoke/`；`20260818-212923-ref2va-full-o6-safe-fp16-hostreg5g-scoped-full/` |
| 6 GiB 首次无运行及纠正运行 | `20260818-222908-ref2va-smoke-o6-safe-fp16-hostreg6g-scoped-smoke/`；`20260818-222956-ref2va-full-o6-safe-fp16-hostreg6g-scoped-smoke/` |
| 单 Ring 回归 | `20260818-224904-ref2va-full-o6-safe-fp16-hostreg5g-single-ring-fastpath-smoke/` |
| Prefetch 未生效 | `20260818-230432-ref2va-full-o6-safe-fp16-hostreg5g-prefetch128-smoke/` |

正式 FP32 视频位于 `E:\ComfyUI-py310\ComfyUI\output\video\raylight_o3`，文件名分别为 `minimax_h3_i2v_o6-baseline-fp32-p2p256_run0_00001_.mp4` 和 `minimax_h3_ref2va_o6-baseline-fp32-p2p256_run0_00001_.mp4`。安全 FP16 的完整运行输出可由对应 benchmark JSON 中记录的 output path 逐项定位。

这些数据来自固定机器状态下每个工作流各一次冷启动。若几何、帧数、步数、P2P 容量、精度策略、模型资源或核心运行时版本发生变化，必须重新建立同规格基线。
