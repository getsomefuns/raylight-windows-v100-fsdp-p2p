# O9：Windows CUDA P2P 自适应分块计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.
>
> **状态：待用户评审，尚未批准开发。**
>
> **执行要求：**批准后按任务顺序执行；每项代码改动先写失败测试，再做最小实现，再独立审查。不得直接修改部署目录；唯一源码树为 `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`，提交后使用仓库部署脚本安装到 ComfyUI。

**目标：**消除普通用户为不同视频长度、分辨率和工作流手动在 256/512 MiB 之间调整 P2P staging buffer 的要求。无论单次 Ulysses `all_to_all_single` 的远端 payload 是否超过物理缓冲区，通信层都自动按有界小块连续搬运，并保持 CUDA P2P/NVLink 数据面、正确结果、可预测显存和现有 FSDP 行为。

**推荐架构：**保留每个 rank 一个固定、可复用的 CUDA staging buffer；默认物理预算回到已经充分验证的 128 MiB。`all_to_all_single` 根据实际远端 payload 调用现有 `chunk_ranges()` 生成确定性分块，每个子块沿用 operation id、ready/consumed CUDA event 和共享内存控制面完成交换。大于 128 MiB 的逻辑 collective 不再失败，也不扩容 buffer，而是自动执行多次。现有已分块的 `all_gather_into_tensor` 复用同一规划语义并做回归验证。

**固定技术栈：**原生 Windows、2× Tesla V100-SXM2-16GB、TCC、驱动 577.00、Python 3.10.11、PyTorch 2.7.0+cu126、xformers 0.0.30、Ray 2.57.0、ComfyUI v0.31.0-15-g62b3c94b、Raylight `windows-v100-fsdp-p2p`。

**阶段顺序：**本计划是新的 O9，优先于 MiniMax H3 双卡 Text Encoder。原 Text Encoder TP2 计划顺延为 O10，只有 O9 的功能、显存、性能和真实工作流门全部通过后才允许开始。

---

## 1. 已确认的根因

当前实现并不是所有 P2P collective 都缺少分块：

- FSDP `all_gather_into_tensor` 已经通过 `chunk_ranges(shard_bytes, capacity_bytes)` 自动分块。历史 F2 验收证明 128 MiB buffer 可以正确搬运 256 MiB shard，同步/异步、尾块和多 dtype 均通过。
- Ulysses/USP 使用的 `all_to_all_single` 仍由 `validate_collective()` 强制检查 `remote_bytes <= capacity_bytes`。例如 MiniMax H3 某 I2V 规格的远端 payload 为 239,826,944 字节；128 MiB buffer 会在采样前明确失败，只能人工改为 256 MiB。
- 启动脚本因此暴露 128/256/512 MiB 三档，并把 256 MiB 作为默认值。更大的档位会在每个 rank 上永久分配同等大小的 CUDA staging buffer，即使绝大多数 collective 用不到。

所以本阶段的主缺口是 **all-to-all 的逻辑 payload 上限错误地等于物理 buffer 大小**，不是 NVLink 带宽不足，也不是 FSDP all-gather 需要重写。

## 2. “动态分块”的准确含义

本计划中的“动态”只表示：运行时读取本次 collective 的实际字节数，自动计算子块数量和尾块范围。

它不表示：

- 不根据视频长度自动把 128 MiB buffer 扩容到 256/512 MiB；
- 不在 OOM 后捕获异常再盲目重试；
- 不把 CUDA payload 回退到 Gloo、CPU 内存或页面文件；
- 不把两个 16GB V100 伪装成透明 32GB 显卡；
- 不改变 `ReserveVramGiB`，该参数仍是 ComfyUI 主进程的独立显存策略。

完成后，`P2PCapacityMiB` 表示“每个 rank 愿意长期占用多少显存作为 P2P 搬运窗口”，而不再表示“工作流能处理的最大逻辑块”。普通用户无需指定；高级测试仍可显式覆盖以研究吞吐与显存取舍。

## 3. 候选方案与决定

### 3.1 采用：单缓冲区、串行子块

每个 rank 继续保留一个 staging buffer。对于远端 payload `R` 和 buffer 容量 `C`，自动生成 `ceil(R/C)` 个半开区间。每块完成“本地写入 → 对端 ready → P2P 读取 → consumed”后再复用同一 buffer。

优点：显存恒定、沿用现有正确性协议、改动集中、故障定位清楚。其控制面调用次数随分块数增加，但 128 MiB 已足够大，预计常见 MiniMax H3 仅增加到 2 至 4 块。

### 3.2 暂不采用：自动扩容后缩容

这种方式代码短，但大 payload 仍会瞬时申请 256/512 MiB，增加碎片和 OOM 风险；缩容也无法保证 CUDA allocator 立即把显存交还给系统。它没有解决用户希望降低固定显存成本的核心问题。

### 3.3 后续优化候选：双缓冲流水

两个 staging buffer 可让第 N 块 P2P 读取与第 N+1 块本地拷贝重叠，但会翻倍持久显存并显著增加事件协议复杂度。O9 首版采用 pipeline depth=1；只有单缓冲正确性通过且真实工作流 profiler 证明分块控制开销成为主要瓶颈，才允许在同阶段增加可选 depth=2 实验，且不得作为默认值。

## 4. 数据布局与分块契约

两 rank 等分 `all_to_all_single` 的输入总字节数为 `T`，远端半块为 `R=T/2`。对子块 `[a,b)`：

- rank 0 发送 `input[R+a:R+b]`，本地保留 `input[a:b]`；
- rank 1 发送 `input[a:b]`，本地保留 `input[R+a:R+b]`；
- rank 0 输出本地块到 `[a:b]`，对端块到 `[R+a:R+b]`；
- rank 1 输出对端块到 `[a:b]`，本地块到 `[R+a:R+b]`；
- staging buffer 内每次只使用 `[0:b-a)`；尾块不要求等于容量；
- 所有范围按原 tensor 的 byte view 处理，传输本身不做 dtype 转换。

`all_to_all_single` 对外仍是一个同步 collective：只有所有子块都已经提交并由当前 CUDA stream 等待最终 consumed event 后才返回。现有路由函数和 yunchang 调用方不感知子块数量。

## 5. 控制协议与失败语义

- 每个子块获得一个单调递增的内部 operation id，与现有 all-gather 一致。
- 两个 rank 必须由相同 `remote_bytes` 和 `capacity_bytes` 生成完全相同的区间顺序。
- 每个子块发布其 `chunk_bytes`；对端尺寸不同立即报错并 poison 当前 P2P group。
- 首块前额外校验完整逻辑 `total_bytes`，不能等到尾块不同才超时。实现时复用现有 64 位 control value：先做一次只含总尺寸的几何握手，再进入子块，不改变共享内存结构。
- 任意 timeout、operation id、总尺寸、子块尺寸或 CUDA copy 失败都保持 fail-fast；不得跳过坏块或转 CPU。
- buffer 只在对端 consumed event 后复用，不能因循环化而覆盖仍在读取的数据。
- profiler 对外记录一次 logical call，同时记录真实 `chunks`；诊断日志必须同时包含 logical payload、remote payload、capacity 和 chunk count。

## 6. 启动和兼容策略

- 仓库启动脚本默认 `P2PCapacityMiB=128`，普通命令不再要求用户选择容量。
- 保留 `-P2PCapacityMiB` 和旧 `-P2PCapacityBytes` 作为高级覆盖，避免破坏已有自动化；它们控制性能/显存预算，不控制正确性上限。
- 仍拒绝非正数和奇数字节容量；MiB 参数保留 128/256/512 兼容档。首版不承诺任意小值都具有可接受性能。
- 启动输出改为 `P2P staging budget`，并明确显示 `larger collectives are auto-chunked`，不再提示“payload 超过容量需调大”。
- 环境根目录 `E:\ComfyUI-py310\start_raylight_p2p.ps1` 不作为源码直接维护。O9 发布时将它变成只转发到已部署仓库启动脚本的薄包装，避免今后出现两份容量默认值；生成/安装规则必须保存在仓库脚本中。
- 已有显式 256/512 MiB 命令继续可运行，但 README 推荐命令移除该参数。

## 7. 范围与非目标

O9 包含：

- 两 rank、equal-split、连续 CUDA tensor 的 `all_to_all_single` 自动分块；
- 现有 `all_gather_into_tensor` 分块回归和公共规划语义整理；
- 同步接口、CUDA stream/event 生命周期、profiler 和故障注入；
- 启动器默认值、文档、真实 MiniMax H3/LTX 工作流回归；
- 固定 staging VRAM 与系统 RAM/commit/pagefile 监控。

O9 不包含：

- 三卡及更多 GPU、跨机器、非等分 all-to-all、训练反向传播；
- NCCL、Gloo CUDA payload、Text Encoder 双卡、VAE 双卡；
- 通过降低分辨率/帧数掩盖容量问题；
- 以放宽 timeout 代替同步根因修复；
- 在首版默认启用双缓冲流水。

## 8. 质检标准

### Gate A：纯逻辑

- 远端 payload 为 0、1、容量−1、容量、容量+1、2×容量、2×容量+4 KiB 时覆盖每个字节且无重叠。
- rank 0/rank 1 的 send、local output、peer output 偏移有字面期望值测试。
- 非法 rank、奇数总字节、dtype/shape/contiguous 不匹配继续明确失败。
- `validate_collective()` 不再因为 `remote_bytes > capacity_bytes` 拒绝合法输入。

### Gate B：双卡 live correctness

- 使用 64 MiB buffer 完成 64、128、256、512、768 MiB 远端 payload。
- FP32、FP16、BF16 原始字节和 uint8 全部通过；BF16 仅代表搬运正确，不宣称 V100 支持 BF16 计算。
- 每种 dtype 验证首字节、容量边界−1、容量边界、每个子块尾字节和最终字节。
- 每个规格连续 20 次，另对主发布规格连续 100 次；最大误差为 0，不能出现 rank drift、timeout 或 orphan worker。
- `all_gather_into_tensor` 原有同步/异步、大 shard 和尾块测试全部继续通过。

### Gate C：显存和资源回收

- 128 MiB 默认模式的每 rank staging allocation 不随 128/256/512/768 MiB 逻辑 payload 增长。
- 相对旧 512 MiB 配置，默认模式理论固定节省 384 MiB/rank；实测 `torch.cuda.memory_allocated()` 与 `nvidia-smi` 必须证明每卡至少减少 350 MiB 持久占用。
- 20 次大块循环后 allocated/reserved 不呈单调增长；进程结束两卡恢复到环境基线，不留 Ray/Python worker。
- 物理内存、commit 和 pagefile 不因 P2P 分块出现与 payload 等量的 host staging 增长。

### Gate D：性能

- 以“旧实现、buffer 与 payload 匹配”的同机结果为对照，不引用理论 PCIe/NVLink 数值。
- 128 MiB 及以上远端 payload的自动分块有效带宽中位数不得低于匹配大 buffer 基线的 85%。
- 256/512 MiB payload 的 P95 不得出现超过中位数 25% 的长尾。
- profiler 必须把 logical calls 与 chunks 分开，确认性能差异来自子块数量，而不是 Gloo、分页或 rank 等待。
- 若 128 MiB 默认未过 85% 门槛，可在不增加持久显存的前提下优化控制循环；不得直接把默认值恢复到 512 MiB 规避验收。

### Gate E：真实工作流

- 复现历史上 128 MiB 因 239,826,944 字节远端 payload 失败的 MiniMax H3 I2V 规格；保持模型、尺寸、帧数、steps、seed、LoRA 和精度不变，以默认启动命令完成。
- 复现当前曾要求 512 MiB 的 MMH3Director 工作流；不得通过降低分辨率、时长或关闭 Refine 通过。
- MiniMax H3 REF2VA、现有 MMH3Director 多段/Refine 和 LTX 5 秒工作流各完成一次冷启动及一次同进程热启动。
- 视频不得黑屏、噪声、首帧后乱码、NaN/Inf 或音视频 mux 失败；固定 seed 的帧摘要与已接受基线保持在既定容差内。
- 两卡采样均有计算负载，正式 CUDA payload 仍走 P2P/NVLink；日志不得出现 Gloo host fallback。
- 记录模型加载、预处理、Ray 初始化、采样 s/it、VAE、音频、保存、端到端、双卡显存峰值、GPU 利用率、RAM、commit、pagefile 和回收状态。

### Gate F：发布与回归

- 完整单元/集成测试通过，`git diff --check` 通过。
- README 中英文部分、测试汇总、启动说明、故障排查和 release note 与代码一致。
- 源仓库提交后使用 `scripts/deploy-to-comfyui.ps1` 部署；部署 SHA 与源码 HEAD 一致且无 drift。
- 部署副本节点导入、P2P probe、真实工作流通过后才可标记 O9 完成。
- O9 完成前禁止启动 O10 Text Encoder TP2 代码开发。

## 9. 实施任务

### Task 1：冻结现状和可复现失败基线

**Files:**

- Modify: `tests/windows_p2p_profile_probe.py`
- Modify: `scripts/minimax-h3/benchmark_cold_warm.py`
- Create: `docs/testing/windows-p2p/ADAPTIVE_CHUNKING_2026-08.zh-CN.md`

- [ ] 保存 128/256/512 MiB staging 的 idle 显存、匹配 payload 带宽和 profiler 数据。
- [ ] 用 128 MiB buffer 复现 239,826,944 字节远端 payload 的当前 `ValueError`，保存精确异常和 `runs=0`。
- [ ] 固定真实工作流的模型、工作流 JSON、输入素材、尺寸、帧数、steps、seed、LoRA、精度与部署 SHA。
- [ ] 运行命令：`E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_windows_p2p.py tests/test_collective_profile.py tests/test_windows_release_profile.py -q`。
- [ ] 提交仅含基线与测试工具的变更，不能先修改 transport。

### Task 2：先定义 all-to-all 分块布局

**Files:**

- Modify: `src/raylight/distributed_worker/windows_p2p.py`
- Modify: `tests/test_windows_p2p.py`

- [ ] 新增失败测试，要求 `all_to_all_chunk_copy_plan(rank, total_bytes, chunk_start, chunk_end)` 返回第 4 节规定的精确 byte ranges。
- [ ] 覆盖 0、容量边界、尾块、两个 rank、非法范围和奇数总字节。
- [ ] 先运行目标测试并确认因函数不存在/旧容量拒绝而失败。
- [ ] 实现最小纯函数并让表驱动测试通过；复用 `chunk_ranges()`，不复制第二套分块算法。
- [ ] 将 `validate_collective()` 改为只验证 tensor 契约并返回总字节，不再把 capacity 当逻辑上限。
- [ ] 运行命令：`E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_windows_p2p.py -q`。

### Task 3：实现单缓冲区 chunked all-to-all

**Files:**

- Modify: `src/raylight/distributed_worker/windows_p2p.py`
- Modify: `tests/test_windows_p2p.py`
- Modify: `tests/test_collective_profile.py`

- [ ] 用模拟 stream/event/control 写失败测试：大于 capacity 时产生多个内部 operation id，但对外只完成一个 call。
- [ ] 添加完整 total_bytes 几何握手，两个 rank 在复制前确认逻辑 collective 一致。
- [ ] 抽取 `_all_to_all_chunk()`，每块严格等待 peer consumed 后再复用 staging buffer。
- [ ] 对本地半块和远端半块按相同 chunk offsets 写入 output，尾块不得读取 buffer 的旧数据。
- [ ] profiler 的 calls 增加 1、chunks 增加实际块数，payload/remote/control wait/submit 统计保持含义不变。
- [ ] 为 size mismatch、operation mismatch、timeout 和中途 CUDA copy 失败验证 poison 语义。
- [ ] 运行目标测试并完成 Python 代码审查；不在本任务修改启动默认值。

### Task 4：扩展真实双 V100 correctness probe

**Files:**

- Modify: `tests/windows_p2p_ray_probe.py`
- Modify: `tests/windows_p2p_full_correctness_probe.py`
- Modify: `tests/windows_p2p_failure_probe.py`
- Modify: `tests/test_windows_p2p_fsdp_probe_source.py`

- [ ] 新增 64 MiB buffer 对 64/128/256/512/768 MiB 远端 payload 的 live cases。
- [ ] 覆盖 FP32/FP16/BF16/uint8、边界字节、尾块、20 次循环和 100 次发布压力规格。
- [ ] 故障 probe 在第二个子块注入不匹配，证明失败不会返回部分成功输出，也不会继续使用 poisoned group。
- [ ] 同一 session 运行 all-to-all 与 all-gather，证明 operation id 和 reusable events 不互相破坏。
- [ ] 采集每次逻辑调用的 chunk count、带宽、P50/P95、allocated/reserved 和进程回收。
- [ ] 使用 TCC/P2P 探针先验检查；任一卡非 TCC 或 peer access 不可用时停止，不回退 Gloo。

### Task 5：把容量从正确性门槛改成显存预算

**Files:**

- Modify: `scripts/start-comfyui-windows-p2p.ps1`
- Modify: `tests/test_windows_release_profile.py`
- Modify: `tests/test_windows_launcher_vram_reserve.py`
- Modify: `src/raylight/distributed_worker/ray_worker.py`
- Modify: `tests/test_ray_worker_windows_p2p.py`

- [ ] 先把启动器测试改为：无参数时 128 MiB、输出 `staging budget` 和 `larger collectives are auto-chunked`。
- [ ] 保留 128/256/512 MiB 和 legacy bytes 参数测试，确认它们只是覆盖物理预算。
- [ ] worker 健康检查必须生成大于默认 capacity 的逻辑 all-to-all，防止只验证单块后误报健康。
- [ ] `-ValidateOnly` 不启动 ComfyUI，准确显示默认预算、自动分块状态、显存预留与 diagnostics。
- [ ] 为环境根启动入口新增仓库内生成/安装规则，使根脚本仅转发参数，不能保存第二份 P2P 默认值。
- [ ] 运行 launcher、worker 和 release-profile 测试。

### Task 6：微基准与显存门

**Files:**

- Modify: `tests/windows_p2p_profile_probe.py`
- Update: `docs/testing/windows-p2p/ADAPTIVE_CHUNKING_2026-08.zh-CN.md`

- [ ] 在相同 GPU 温度/功率状态下交错运行旧匹配大 buffer 与新 128 MiB 分块，避免顺序偏差。
- [ ] 每个规格至少 5 次独立 session，报告中位数和 P95，不以单次峰值作为结论。
- [ ] 证明逻辑 payload 增长时 staging allocation 恒定，并测量相对 512 MiB 的实际显存回收。
- [ ] 记录 P2P、控制等待、submit、GPU 利用率、功率、RAM、commit 和 pagefile。
- [ ] 未通过 Gate B/C/D 时记录根因和拒绝项；不得进入真实工作流或 O10。

### Task 7：真实工作流验收

**Files:**

- Modify: `scripts/minimax-h3/benchmark_cold_warm.py`
- Update: `docs/testing/windows-p2p/ADAPTIVE_CHUNKING_2026-08.zh-CN.md`
- Update: `docs/TESTING.md`
- Update: `docs/WINDOWS_V100_FSDP_TESTING.md`

- [ ] 停止 ComfyUI/Ray，确认两卡、RAM、commit 和 pagefile 回到基线后开始冷启动。
- [ ] 先跑历史 239,826,944 字节失败规格，确认 128 MiB 默认自动产生 2 块并进入/完成采样。
- [ ] 跑曾要求 512 MiB 的当前 MMH3Director 工作流，记录真实最大 remote payload 和 chunk count。
- [ ] 跑 MiniMax H3 REF2VA、MMH3Director 多段/Refine、LTX 5 秒的冷/热组合。
- [ ] 做视频有限值、帧摘要、首/中/尾帧和音频检查；保存输出索引，不把大视频提交仓库。
- [ ] 对比旧 256/512 MiB 同规格结果，分别报告速度变化和每卡显存变化，不能只写“通过”。
- [ ] 每个工作流结束检查 worker 生命周期和资源回收，确认没有逐轮叠加。

### Task 8：文档、发布和部署

**Files:**

- Modify: `README.md`
- Modify: `docs/windows-v100-p2p.md`
- Modify: `docs/testing/minimax-h3/TURBO_WORKFLOW_USAGE.md`
- Create: `docs/releases/2026-08-o9-adaptive-p2p-chunking.zh-CN.md`
- Create: `docs/releases/2026-08-o9-adaptive-p2p-chunking.en.md`
- Modify: `docs/superpowers/plans/2026-08-29-o10-minimax-h3-dual-v100-text-encoder-tp2.zh-CN.md`

- [ ] README 中英文用户指南改为无容量参数的默认启动；高级覆盖与 `ReserveVramGiB` 分开解释。
- [ ] 将旧“payload 超容量就调大”的排障说明改为自动分块，并保留真实故障的诊断方法。
- [ ] release note 同时写明实装项、实测数据、未实装的双缓冲路线及理由。
- [ ] 更新 O10 状态：只有 Gate A-F 全部通过才从“被 O9 阻塞”改为“可开始评审/开发”。
- [ ] 运行完整测试、`git diff --check`、代码审查和安全审查。
- [ ] 提交源码；执行 `scripts/deploy-to-comfyui.ps1`；验证部署 SHA、节点导入、P2P probe 和最终 smoke。

## 10. 最终交付物

O9 完成时必须同时具备：

1. 不因 logical remote payload 超过 128/256/512 MiB staging budget 而失败的 chunked `all_to_all_single`；
2. 保持现有 FSDP chunked all-gather 的无回归证明；
3. 固定 staging VRAM、无 host payload fallback、可重复的带宽/长尾数据；
4. 历史失败工作流和当前 512 MiB 工作流在默认启动下的真实完成证据；
5. 中英文用户说明、release note、测试汇总与部署一致性；
6. 明确的 O10 Text Encoder TP2 开始条件。

## 11. 停止条件

出现以下任一情况必须停止发布并回到对应任务排查：

- 任意 CUDA payload 经过 CPU/Gloo 才能完成；
- 大 payload 导致 staging allocation 随之增长；
- 两 rank 分块数、operation id 或完成顺序不一致；
- 通过放宽 timeout、降低视频规格或增大默认 buffer 才能通过；
- 视频数值/质量回归或 worker/显存/commit 逐轮增长；
- 自动分块的 128 MiB+ 带宽未达到 Gate D 且没有可验证根因；
- 源仓库与部署副本 SHA/内容不一致。

这些停止条件只阻止 O9 发布和 O10 开始，不允许用“实验性默认值”绕过质检。
