# Windows Dual-V100 FSDP + Ulysses Hybrid Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在原生 Windows 双 Tesla V100 TCC/NVLink 环境中，测清 FSDP 权重 `all_gather` 与 Ulysses 序列 `all_to_all` 的成本，并只在相对匹配单卡热启动中位数快至少 20% 且输出有效时发布混合模式。

**Architecture:** 两张 GPU 上的同一对 rank 同时组成 FSDP2 权重分片组和 Ulysses2 序列并行组；Gloo 仅保留控制面，CUDA IPC/P2P 承担两类 CUDA payload。先增加默认关闭的分类型通信统计，再建立严格匹配基线，最后依证据逐项评估缓冲复用、流水和预取；任何优化都保留关闭开关和旧模式。

**Tech Stack:** Windows 11、2× Tesla V100-SXM2-16GB TCC、NVLink/CUDA P2P、Python 3.10.11、PyTorch 2.7.0+cu126、Ray、ComfyUI、Raylight、xFuser、yunchang、FFmpeg/ffprobe。

**Spec:** [2026-08-15-windows-v100-fsdp-next-stage-plan.md](2026-08-15-windows-v100-fsdp-next-stage-plan.md) Phase F4；F3 已验收记录见 [2026-08-16-windows-v100-fsdp-phase-f3-plan.md](2026-08-16-windows-v100-fsdp-phase-f3-plan.md)。

## Global Constraints

- 只修改独立的 `<environment-root>` 内内容，不影响其他 Python、ComfyUI、CUDA 或驱动环境。
- 显卡保持 TCC；驱动保持 577.00；PyTorch 保持 2.7.0+cu126，除非独立证据证明必须调整。
- ComfyUI 启动继续使用 `--disable-cuda-malloc`；V100 不安装 FlashAttention 2/3。
- 正式 CUDA payload 不得退回 Gloo/CPU/系统页面文件；失败必须明确中止。
- LTX/LTXAV 保持模型支持的 BF16/FP32 数值路径；不得使用已证实会产生黑屏和 NaN/Inf 的全局 FP16。
- 所有性能结论使用相同输入图片、提示词、模型、LoRA、种子、121 帧、25 fps、VAE tile 384 和相同缓存状态。
- `20%` 的最终门槛是“混合模式热启动中位数 ≤ 匹配单卡热启动中位数 × 0.80”，不是双卡自身冷/热差。
- 本阶段只做本地修改和测试；未经用户明确要求不提交、不推送 GitHub。

---

## File Map

- `src/raylight/distributed_worker/parallel_group_manager.py`：描述并校验 FSDP/Ulysses 共享双 rank 的拓扑。
- `src/raylight/distributed_worker/collective_profile.py`：默认关闭的通信分类、聚合和快照，不保存逐调用大日志。
- `src/raylight/distributed_worker/windows_p2p.py`：在 P2P endpoint 中上报 all-gather/all-to-all 的调用、字节、分块和等待时间；保留原数据路径。
- `src/raylight/distributed_worker/ray_worker.py`：安装 profiler、在采样边界取快照并输出每 rank 精简 JSON。
- `src/raylight/nodes.py`：启动前调用拓扑校验，必要时输出模式建议。
- `tests/test_hybrid_topology.py`：两 rank 合法组合和错误组合。
- `tests/test_collective_profile.py`：分类计数、聚合、重置、关闭时零开销路径。
- `tests/windows_p2p_hybrid_probe.py`：FSDP all-gather 与 Ulysses all-to-all 交错的双 V100 合成探针。
- `example_workflows/LTX2_3_i2v_Raylight_Windows_Hybrid_5s.json`：仅在合成探针与显存门禁通过后创建的混合工作流。
- `docs/TESTING.md`、`docs/test-results-2026-08.csv`：精简结果和证据入口。

## Task 1: 固化双 rank 混合拓扑契约

**Interfaces:**
- Produces: `validate_hybrid_topology(world_size: int, shard_size: int, parallel_dict: dict) -> str`，返回 `single`、`ulysses`、`fsdp` 或 `hybrid`；不改变输入。
- Consumes: `is_fsdp`、`is_xdit`、`ulysses_degree`、`ring_degree`、`cfg_degree`、`dp_degree`、`shard_size`。

- [x] 在 `tests/test_hybrid_topology.py` 写失败测试：`world=2, shard=2, FSDP=true, Ulysses=2, ring=1, cfg=1, dp=1` 返回 `hybrid`。
- [x] 写失败测试：混合模式的 `dp!=1`、`shard!=2`、`world!=2`、`ring!=1` 或 `cfg!=1` 都在 actor 创建前给出可读错误。
- [x] 运行 `<environment-root>\Python310\python.exe -m unittest tests.test_hybrid_topology -v`，确认新测试先失败。
- [x] 在 `parallel_group_manager.py` 实现纯函数校验，并在 `nodes.py` 与 Windows P2P worker 启动门禁复用。
- [x] 重跑聚焦测试与现有 session/topology 测试，确认旧 Ulysses、纯 FSDP 和单卡配置不变。

**Acceptance:** 两张卡可复用同一 rank 对完成两种集合通信；不需要错误地要求四张卡；所有不受支持的组合在占用模型显存前失败。

## Task 2: 增加低开销的分类型通信统计

**Interfaces:**
- Produces: `CollectiveProfiler.record(kind, payload_bytes, remote_bytes, chunks, control_wait_ns, submit_ns)`、`snapshot(reset=False) -> dict`。
- Produces: 环境开关 `RAYLIGHT_P2P_PROFILE=1`；关闭时不创建 CUDA event、不写日志文件。
- Consumes: P2P endpoint 已验证后的 all-gather shard bytes 与 all-to-all total bytes。

- [x] 在 `tests/test_collective_profile.py` 写失败测试，验证两类 collective 分开累计调用数、payload/remote bytes、分块数、控制等待和最大 payload。
- [x] 写失败测试，验证 `snapshot(reset=True)` 原子清零，禁用对象的 `record()` 不改变状态。
- [x] 实现只保存聚合值的 `collective_profile.py`，不保留逐调用张量或事件对象。
- [x] 在 `windows_p2p.py` 的成功路径记录 all-gather/all-to-all；异常调用不得计为成功。
- [x] 在 `ray_worker.py` 的采样开始/结束边界输出每 rank 一行 `[RAYLIGHT_P2P_PROFILE]` JSON。
- [x] 用硬件微探针比较统计开/关各 5 轮中位数；统计开销必须小于 2%，否则停止进入真实工作流并先降低统计成本。

**Acceptance:** 能回答两类通信各调用多少次、移动多少远端字节、切多少块、控制面等待多少时间；默认关闭且没有大日志。

## Task 3: 建立完全匹配的性能基线

**Interfaces:**
- Produces: `F4-SINGLE-WARM`、`F4-ULYSSES-WARM`、`F4-FSDP-WARM` 三组可复查中位数与阶段耗时。
- Consumes: 相同的 LTX 5 秒输入、提示词、模型、LoRA、种子、121 帧、25 fps、VAE tile 384。

- [ ] 清理遗留 ComfyUI/Ray worker 和显存，仅保留测试需要的进程。
- [ ] 生成匹配单卡工作流/API payload；运行一次冷启动和至少三次真正完整的热启动，不能把下游缓存短路当作完整热启动。
- [ ] 对 Ulysses2 速度模式和 FSDP2 显存模式各运行一次冷启动、至少三次完整热启动。
- [ ] 记录模型加载、两段 sampler、VAE、音频、保存、每步 it/s、GPU 显存/利用率、物理/提交内存和页面文件变化。
- [ ] 用 ffprobe/blackdetect 和日志验证每组输出的帧数、音频、非黑屏、无 NaN/Inf。
- [ ] 将中位数和原始附件路径写入 `docs/TESTING.md`，明确旧的 316.60 s/284.06 s 仅是历史参考，除非与当前固定条件完全一致。

**Acceptance:** 得到可直接计算 `0.80 × matched_single_warm_median` 的硬阈值；每种模式至少三次有效热运行，变异系数目标不超过 5%。

## Task 4: 双 collective 交错合成探针

**Interfaces:**
- Produces: `tests/windows_p2p_hybrid_probe.py`，在两个 Ray actor 中以可配置尺寸交错执行 all-gather 与 all-to-all。
- Consumes: Task 1 的拓扑校验与 Task 2 的 profiler。

- [ ] 先写 CPU 级测试定义确定性输入、交错顺序和期望逐元素输出。
- [ ] 实现 64/128/288 MiB all-gather 与 LTX 实测尺寸 all-to-all 的同步交错探针。
- [ ] 验证 FP32/FP16/BF16、非整块尾部、连续 100 轮和严格 10 秒故障退出。
- [ ] 记录每类通信的有效带宽、控制等待、rank 偏差、峰值 VRAM 和页面文件前后值。
- [ ] 连续运行三次，要求零误差、零 Gloo CUDA payload fallback、无句柄/worker 泄漏。

**Acceptance:** 两类 collective 使用同一 endpoint 时操作号、事件和共享缓冲不互相污染；所有结果精确或在模型 dtype 允许误差内一致。

## Task 5: 评估并优化缓冲复用、流水和预取

**Interfaces:**
- Produces: 每个候选优化独立环境开关及 A/B 数据；默认值只在通过门槛后改变。
- Consumes: Task 2 分类数据和 Task 4 合成探针。

- [ ] 按总耗时占比排序候选项，只处理占端到端时间至少 3% 的瓶颈。
- [ ] 为第一个候选写失败测试，覆盖关闭时旧行为和开启时的新调度顺序。
- [ ] 依次 A/B 测试：现有串行单缓冲、双缓冲通信流水、固定缓冲复用，以及 FSDP 可控预取；每次只改变一个变量。
- [ ] 每个候选先跑合成探针，再跑一次 5 秒冷启动和至少三次热启动；记录 profiler 开/关的性能差。
- [ ] 若候选增加峰值显存导致任一 rank 低于 256 MiB 安全余量、引入页面文件增长、黑屏、NaN/Inf 或 >5% 不稳定，立即回退该候选。

**Acceptance:** 只保留在真实工作流中可重复缩短中位数且不损害显存、稳定性或输出的变化；不因微基准峰值单独发布优化。

## Task 6: 混合工作流、最终性能门和模式推荐

**Interfaces:**
- Produces: 通过门槛才保留 `LTX2_3_i2v_Raylight_Windows_Hybrid_5s.json`。
- Produces: 基于模型是否能放入单卡与匹配实测数据的 `speed` / `memory` / `hybrid` 建议说明。

- [ ] 创建 FSDP2 + Ulysses2 + dp1 + ring1 + cfg1 的独立 5 秒混合工作流，不修改已验收的 Ulysses/FSDP 工作流。
- [ ] 运行一次冷启动和至少三次完整热启动，保存阶段耗时、通信分类、资源监控和媒体质检。
- [ ] 计算混合热启动中位数相对匹配单卡、Ulysses 与 FSDP 的变化，并给出 95% 可解释范围或至少 min/median/max。
- [ ] 运行全量单元测试、发布版 P2P 探针、FSDP 探针和混合探针。
- [ ] 仅当混合中位数 `≤ 单卡中位数 × 0.80`、三次都有效、变异系数 `≤5%`、无页面文件 payload 依赖且峰值 VRAM 安全时标记 `GO`。
- [ ] 若未过门，标记 `NO-GO` 并保留纯 Ulysses 速度模式和纯 FSDP 显存模式；文档明确瓶颈及下一步，不对用户暴露未验证的自动选择。

## F4 Quality Gates

- **正确性：** all-gather/all-to-all 合成与真实输出均通过；视频非黑、音频有效、无 NaN/Inf。
- **性能：** 最终目标是相对匹配单卡完整热启动中位数快至少 20%；冷/热缓存收益不能代替该指标。
- **稳定性：** 每个发布候选至少三次热运行，变异系数不超过 5%，rank 偏差定位到阶段且不靠放宽 10 秒超时。
- **显存：** 任一 GPU 不 OOM，峰值至少保留 256 MiB 安全余量；FSDP 持久权重仍保持分片。
- **主存：** 页面文件不承担 CUDA payload；mmap 文件读取与页面文件增长分开记录。
- **通信：** Gloo 只处理元数据；两类大型 CUDA collective 都由 CUDA IPC/P2P 路由，失败时严格中止。
- **可维护性：** profiler 默认关闭、日志为聚合摘要、旧模式与工作流不被覆盖、全量回归通过。

## Execution Status

- [x] F3 代码、双卡工作流、媒体输出、硬件探针和 104 项回归已验收并保存。
- [x] F3 精简结果已补入维护文档和 CSV。
- [x] F4 详细计划已建立；最终指标已纠正为相对匹配单卡热启动快 20%。
- [x] Task 1：固化双 rank 混合拓扑契约。`19` 项聚焦测试与语法检查通过。
- [x] Task 2：分类型通信统计。`116` 项全量测试通过；硬件开销 `1.47%`。
- [ ] Task 3：匹配性能基线。
- [ ] Task 4：双 collective 交错探针。
- [ ] Task 5：证据驱动的通信优化。
- [ ] Task 6：混合工作流与最终 GO/NO-GO。
