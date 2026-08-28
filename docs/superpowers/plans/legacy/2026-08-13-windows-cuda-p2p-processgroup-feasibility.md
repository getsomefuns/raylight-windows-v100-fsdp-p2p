# Raylight 原生 Windows CUDA P2P 通信后端：可行性评估与实施方案

> 状态：路线 A 已实施并通过端到端验收；路线 B/C 暂未实施<br>
> 日期：2026-08-13<br>
> 目标环境：原生 Windows、2× Tesla V100-SXM2-16GB、TCC、NVLink、PyTorch 2.7.0+cu126、Raylight + LTX 2.3<br>
> 目标：替代 Windows Gloo 的大张量数据面，使 Raylight Ulysses 通信真正使用 CUDA IPC/P2P/NVLink。

> 2026-08-13 实施结果：正式复用会话端到端中位数 284.06 秒；后续严格单卡热启动复测为 316.60 秒，公平热对热提升为 10.28%，尚未达到最终 20% 门槛。110 MiB Ray actor 探针约 59.28 GiB/s/方向。最终实现使用常驻 64 MiB/卡 CUDA IPC staging buffer、跨进程 CUDA event 和 Windows named shared-memory 控制面；Gloo 保留用于 rendezvous 与回退。详细证据见同目录验收报告和单卡冷/热报告。

## 1. 执行摘要

该路线具备技术可行性，且已经越过最关键的硬件、驱动和 PyTorch 运行时门槛。

在不修改 ComfyUI 的临时探针中，两个独立 Python 进程分别仅能看到一张 V100；进程间通过 PyTorch CUDA IPC 共享 CUDA storage，再从对端 GPU 直接复制到本卡。4–128 MiB 的双向同时传输达到约 99–108 GiB/s/方向，合计约 198–217 GiB/s。该结果与机器已有的 240+ GB/s CUDA P2P 双向测试处于同一量级，证明以下条件均成立：

- Windows TCC 下 CUDA IPC 可用；
- Ray 风格“一 actor 一张可见 GPU”不妨碍 IPC；
- PyTorch 缓存分配器产生的 CUDA tensor storage 可以导出和导入 IPC 句柄；
- 跨进程数据可以走 GPU P2P/NVLink，而非必须经过 Gloo TCP 大张量路径。

当前 Gloo 大块通信约为 0.35 GiB/s/方向，与 CUDA IPC 实测相差约 300 倍。因此，自定义 P2P 数据面有充分的性能空间。

推荐先实现“双卡、单机、TCC、Ulysses equal-split、FSDP 关闭”的专用 P2P all-to-all，再决定是否封装成完整第三方 ProcessGroup。不建议从一开始重造通用 NCCL。

## 2. 已验证事实

### 2.1 当前软件与硬件条件

- GPU：2× Tesla V100-SXM2-16GB；
- 驱动模式：TCC；
- NVLink：6 条链路/GPU，每条由 `nvidia-smi nvlink --status` 报告 25.781 GB/s；
- PyTorch：2.7.0+cu126；
- CUDA runtime：12.6；系统 CUDA Toolkit：12.9；
- Gloo 可用，NCCL 不可用；
- Raylight：Ulysses=2、Ring=1、FSDP 关闭；
- yunchang 当前每个注意力层对 Q、K、V 和输出分别调用 all-to-all。

### 2.2 CUDA IPC/P2P 临时探针

探针文件：`C:\tmp\cuda_ipc_torch_probe.py`

探针模型：

1. 使用 Windows `spawn` 创建两个独立 Python 进程；
2. rank 0 设置 `CUDA_VISIBLE_DEVICES=0`，rank 1 设置 `CUDA_VISIBLE_DEVICES=1`；
3. 两个进程内部都只使用逻辑设备 `cuda:0`；
4. 通过 PyTorch multiprocessing CUDA reduction 导出/导入 CUDA storage；
5. 两个进程同时从对端 storage 向本地 GPU tensor 执行异步 D2D copy；
6. 验证结果数据内容和进程退出码。

实测结果：

| 数据量 | rank 0 有效带宽 | rank 1 有效带宽 | 单次延迟范围 |
|---:|---:|---:|---:|
| 1 MiB | 82.96 GiB/s | 48.23 GiB/s | 11.8–20.2 μs |
| 4 MiB | 99.58 GiB/s | 99.08 GiB/s | 39.2–39.4 μs |
| 16 MiB | 105.99 GiB/s | 105.44 GiB/s | 147.4–148.2 μs |
| 64 MiB | 107.95 GiB/s | 107.92 GiB/s | 579 μs |
| 128 MiB | 108.48 GiB/s | 108.50 GiB/s | 1.152 ms |

这只是对端读/本地写的数据面上限，不包含完整 collective 的控制面、切片、stream/event 协调和 Python 调度成本。

### 2.3 当前 Gloo 基线

Windows Gloo CUDA `all_to_all_single` 大块通信实测：

| Gloo 绑定地址 | 64 MiB 每卡远端有效带宽 | 128 MiB 每卡远端有效带宽 |
|---|---:|---:|
| `<LAN Gloo address>` | 0.276 GiB/s | 0.285 GiB/s |
| `127.0.0.1` | 0.355 GiB/s | 0.349 GiB/s |

回环地址只能带来有限改善，无法让 Gloo 使用 NVLink 数据面。

## 3. 范围与非目标

### 3.1 首版范围

- 原生 Windows；
- 单机；
- 两张 NVIDIA GPU；
- 两卡之间具备 CUDA P2P；
- TCC；
- 一进程/actor 一张 GPU；
- Ulysses=2；
- Ring=1；
- CFG=1；
- DP=1；
- equal-split `all_to_all_single`；
- BF16、FP16、FP32 以及原始字节视图；
- 推理前向；
- LTX 2.3 当前工作流；
- FSDP 关闭。

### 3.2 首版非目标

- 不实现跨机器通信；
- 不支持超过两张 GPU；
- 不支持无 P2P 的 PCIe 拓扑；
- 不支持 WDDM；
- 不支持 FSDP、Ring、PipeFusion 或通用 DP/CFG 组合；
- 不替代 NCCL 的全套 collective；
- 不承诺训练和 autograd；
- 不改变 USP“每卡仍保存完整模型权重”的语义；
- 不在首版支持任意不等长 split。

## 4. 候选路线

### 4.1 路线 A：只替换 yunchang 的 all-to-all 数据面（推荐起点）

保留 Gloo process group，用它完成 rendezvous、group 构建、少量控制消息、barrier 和不在首版范围内的 collective。对 yunchang 实际调用的 CUDA `all_to_all_single` 注入专用 CUDA IPC/P2P 实现。

优点：

- 修改面最小；
- 能最快验证 LTX 端到端收益；
- 不需要立即满足完整 ProcessGroup ABI；
- 失败时可以逐次或逐工作流回退 Gloo；
- 可以直接记录真实 Q/K/V 通信形状和频率。

缺点：

- 与 yunchang/Raylight 集成点存在版本耦合；
- xFuser 的其他 collective 仍然走 Gloo；
- 不是通用后端。

### 4.2 路线 B：Python 第三方 ProcessGroup

继承 `torch.distributed.ProcessGroup`，实现 `alltoall_base` 和必要的 `barrier`，其他操作委托 Gloo。通过 `Backend.register_backend` 注册 CUDA 后端。

优点：

- yunchang/xFuser 可以继续使用标准 `torch.distributed` API；
- 后续扩展其他 collective 更清晰；
- 可减少上层 monkey patch。

缺点：

- PyTorch 第三方 ProcessGroup API 仍属于实验接口；
- Python 调度、Store 操作和 GIL 可能影响小消息延迟；
- Work、异步完成、异常传播和 DeviceMesh 兼容需要额外实现；
- 调试难度高于路线 A。

### 4.3 路线 C：C++/CUDA ProcessGroup 扩展

使用 LibTorch/c10d C++ API实现专用 backend 和 Work 对象，CUDA IPC、事件和复制均由 C++ 管理。

优点：

- 最低 Python 调度开销；
- 可以精确管理 CUDA stream、event 和 tensor 生命周期；
- 最接近长期可维护的高性能通信后端。

缺点：

- 与 PyTorch 2.7 C++ ABI 强绑定；
- 需要维护 MSVC、CUDA、LibTorch 编译链；
- Torch 升级很可能需要重新编译甚至适配接口；
- Work/错误处理/缓存分配器交互复杂；
- 在证明 LTX 能获益前投入过大。

### 4.4 推荐顺序

1. 路线 A 验证真实工作流；
2. 达到性能与稳定性门槛后，转为路线 B；
3. 只有 Python ProcessGroup 被证明确为瓶颈时，才实施路线 C。

## 5. 推荐架构

大张量和控制消息分离：

- 数据面：CUDA IPC memory handle + CUDA interprocess event + P2P/NVLink copy；
- 控制面：TCPStore，仅发送操作序号、句柄、shape、stride、dtype、offset、错误状态和完成确认；
- 回退面：Gloo；
- 能力检测：启动阶段验证 TCC、两卡、UVA、IPC、P2P、拓扑和基准带宽。

### 5.1 一次 all-to-all 的建议流程

1. 为 collective 生成单调递增的 `operation_id`；
2. 检查 input/output 为连续 CUDA tensor，dtype、shape 和 split 合法；
3. 在生产 CUDA stream 上记录 `ready_event`；
4. 导出 input storage 的 CUDA IPC memory handle；
5. 导出可跨进程等待的 `ready_event` handle；
6. 通过 Store 发布 metadata；
7. 对端导入 storage 和 event；
8. 对端通信 stream 等待 `ready_event`；
9. 两个 rank 同时将对端分片 P2P copy 到本地 output 的远端区域；
10. 本地 input 对应的本地分片执行 D2D copy；
11. 在通信 stream 上记录 `done_event`；
12. 消费计算 stream 等待本地 `done_event`；
13. 双方确认完成后释放本次 IPC 映射引用；
14. 生产者收到完成确认前不得允许缓存分配器复用 input storage。

### 5.2 生命周期原则

- IPC memory handle 只能在底层 allocation 存活期间使用；
- 生产者必须保留原 tensor/storage 强引用，直到所有消费者完成；
- consumer 必须在 actor 退出前关闭 IPC mapping；
- operation id 必须同时用于 Store key、超时、错误报告和资源回收；
- 禁止仅依赖 Python GC 决定通信资源生命周期；
- 任何 rank 超时或退出时，整个通信组进入失败状态，不能继续下一个 collective。

### 5.3 缓冲策略

首版优先共享调用方现有 input storage，以避免常驻大缓冲池占用显存。当前工作流每卡峰值接近16GB，因此：

- 常驻控制结构和小型 staging buffer 目标低于96 MiB/卡；
- 硬上限128 MiB/卡；
- 不预分配数百MiB双缓冲；
- 只有真实通信形状证明高度固定且显存允许时，才考虑固定通信缓冲池。

## 6. 分阶段实施方案

### Phase 0：真实负载观测

目标：确定 Raylight LTX 中所有 all-to-all 的真实 tensor shape、dtype、字节数、调用频率、stream和显存余量。

交付物：

- 不改变结果的通信 trace；
- 按阶段/注意力层/消息大小汇总；
- P50/P95 消息大小；
- 每步 collective 次数；
- 每卡最小剩余显存；
- 用于后续微基准的代表性 shape 集合。

预计：0.5–1个开发日。

### Phase 1：独立双卡 P2P all-to-all

目标：实现脱离 ComfyUI 的、具备正确 stream/event/lifetime 语义的两卡 equal-split all-to-all。

交付物：

- IPC handle exchange；
- interprocess event；
- 本地和远端分片复制；
- operation id；
- timeout；
- worker crash 检测；
- dtype/shape/split 校验；
- 10,000次循环和错误注入测试。

预计：1–2个开发日。

### Phase 2：接入 Ray actor 和 Gloo 回退

目标：在 Ray actor 模型中保持50 GiB/s以上的大块有效带宽，并能安全回退。

交付物：

- Ray actor 能力握手；
- Store key 命名和 group 隔离；
- backend 选择开关；
- 自动能力检测；
- Gloo fallback；
- actor退出时的资源释放；
- 通信健康检查。

预计：2–4个开发日。

### Phase 3：接入 yunchang/Raylight/LTX

目标：替换 Ulysses 的大张量 Gloo all-to-all，完成现有两段 LTX 2.3 采样。

交付物：

- Raylight 内部注入点；
- 不修改全局 site-packages 的集成方式；
- 当前工作流兼容；
- 与 Gloo逐操作结果对照；
- 端到端冷/热基准。

预计：2–3个开发日。

### Phase 4：稳定性与性能收敛

目标：验证长时间运行、异常恢复、显存和端到端收益。

交付物：

- 30分钟稳定性测试；
- 连续5次完整工作流；
- 显存泄漏报告；
- 3次冷启动和3次热运行中位数；
- Nsight/PyTorch profiler 通信时间分解；
- 是否继续 ProcessGroup 封装的决策报告。

预计：2–4个开发日。

### Phase 5：封装第三方 ProcessGroup

前置条件：Phase 4 证明采样阶段至少比 Gloo 快25%，且接近或超过原生单卡。

目标：移除 yunchang 专用数据面补丁，通过标准 `torch.distributed` 接口提供 P2P all-to-all。

交付物：

- backend 注册；
- `alltoall_base`；
- `barrier`/必要委托；
- Work完成语义；
- xFuser group和DeviceMesh验证；
- 版本兼容检测；
- 打包、加载和卸载测试。

预计：4–7个开发日。

## 7. 总工作量评估

| 交付深度 | 预计工作量 |
|---|---:|
| 证明 Ray actor 内 P2P all-to-all 可用 | 2–4个开发日 |
| 具备 LTX 端到端价值的实验版 | 4–7个开发日 |
| 可长期使用的双卡专用版本 | 7–12个开发日 |
| 完整 Python/C++ 自定义 ProcessGroup | 12–20个开发日 |
| 通用多卡/NCCL替代品 | 至少4–8周，不建议纳入当前目标 |

工作量按当前环境已经配置完成、无需更换驱动、仅支持双V100专用路径估算。工具链、PyTorch私有接口或Ray生命周期出现额外兼容问题时，时间可能上浮。

## 8. 预期效果

### 8.1 通信层

- 16–128 MiB P2P all-to-all 有效带宽目标：50–100 GiB/s/方向；
- 相比当前 Gloo：至少50倍，理想超过100倍；
- 小消息 P95 延迟目标：1 MiB不超过100 μs；
- 不允许大张量 D2H/H2D 中转。

### 8.2 工作流层

合理目标而非保证值：

- Raylight 采样阶段相对当前 Gloo 缩短30%–55%；
- 双卡采样耗时达到原生单卡的70%–95%；
- 当前同会话热运行由11:36降至约6:00–8:00；
- 若同时解决两段采样间模型重载，端到端目标约5:30–7:00；
- 达到“单卡27秒、双卡20秒”约26%耗时下降，属于进取但合理的目标。

### 8.3 不会改变的限制

- USP仍然在每卡保存完整模型权重；
- 不会自动获得FSDP模型权重分片；
- VAE编解码仍可能仅在GPU0；
- 模型装载、CPU调度和ComfyUI缓存开销仍然存在；
- 通信快300倍不等于端到端快300倍。

## 9. 风险登记

### R1：PyTorch 私有 CUDA IPC ABI

当前 `_share_cuda_/_new_shared_cuda` 已实测可用，但属于内部接口。Torch升级后必须重新验证；长期可以迁移到CUDA Driver API或C++扩展降低对Python私有接口的依赖。

### R2：缓存分配器与张量生命周期

生产者过早释放input，缓存分配器可能复用allocation，造成静默数据破坏。必须用强引用、done确认和group失败状态管理。

### R3：CUDA stream/event顺序

必须保证生产计算、P2P copy、消费计算的stream依赖。错误实现可能偶发读到未完成或已覆盖的数据。

### R4：actor异常退出和死锁

某个actor死亡时，另一端可能永久等待IPC event或Store key。所有等待必须有超时；任何rank失败均终止整个通信组。

### R5：显存余量

当前每卡峰值接近16GB。大常驻通信缓冲可能导致原本可运行的LTX工作流OOM。

### R6：ProcessGroup实验接口

PyTorch第三方backend接口可能随版本改变；完整ProcessGroup还必须正确实现Work、异步语义和DeviceMesh group行为。

### R7：Ray逻辑设备编号

每个actor内部可能都把自己的物理GPU称为`cuda:0`。当前IPC探针已模拟通过，但metadata必须明确区分物理rank、逻辑device和IPC来源device。

### R8：范围膨胀

如果首版同时支持FSDP、Ring、多卡、跨机器和任意split，工程会迅速变成NCCL重实现。必须坚持首版范围。

## 10. 质检与验收标准

### 10.1 一级：正确性

必须全部满足：

- FP16、BF16、FP32、整数和原始字节传输结果正确；
- 覆盖1、4、16、64、128 MiB；
- 覆盖yunchang实际4D/5D tensor和切分维度；
- equal-split和空边界输入均有明确行为；
- 连续10,000次all-to-all无错位、无崩溃、无死锁；
- 两个rank调用顺序或operation id不一致时，10秒内明确失败；
- 任一worker异常退出后，另一端10秒内退出通信组；
- 相同输入下与Gloo输出逐元素一致；
- 禁止静默回退或静默输出错误。

### 10.2 二级：通信性能

硬门槛：

- 16–128 MiB：每方向不低于50 GiB/s；
- 4 MiB：每方向不低于30 GiB/s；
- 1 MiB：P95延迟不高于100 μs；
- 相比当前Gloo至少快50倍；
- profiler确认没有大张量D2H/H2D；
- 两方向同时工作时性能不应低于单方向的70%。

若16–128 MiB低于20 GiB/s，停止上层集成，先修复数据面。

### 10.3 三级：资源与稳定性

- 常驻额外显存不超过128 MiB/卡，目标低于96 MiB；
- 30分钟循环无显存持续增长；
- 连续完成至少5次完整LTX工作流；
- 每轮结束稳定态显存差异不超过64 MiB；
- 结束后GPU回到预期空闲显存；
- 不遗留Ray/Python/编译辅助进程；
- ComfyUI取消队列或worker异常时能够清理IPC资源。

### 10.4 四级：端到端效果

固定使用：

- 工作流：`<ComfyUI>/user/default/LTX2_3_i2v_Raylight.json`；
- 输入图：`<ComfyUI>/input/LTX2_3_i2v_Raylight.jpg`；
- 相同模型、LoRA、提示词、seed、分辨率、帧数、步数和VAE；
- 每组3次冷启动、3次热运行，比较中位数；
- 分开记录模型加载、两段采样、VAE和总耗时。

验收门槛：

- 采样阶段至少比当前Gloo快25%；
- 第一阶段目标：双卡采样不比原生单卡慢超过5%；
- 正式通过目标：双卡采样比原生单卡快至少15%；
- GPU0/GPU1采样期间平均利用率均高于55%；
- 端到端热运行目标不超过7分钟；
- 若未达标，必须用profile区分通信、模型重载、VAE和其他计算瓶颈。

## 11. Go/No-Go 决策门槛

| 检查点 | Go | No-Go / 回退动作 |
|---|---|---|
| 独立P2P all-to-all | ≥50 GiB/s，10,000次稳定 | <20 GiB/s则停止上层集成 |
| Ray actor集成 | ≥50 GiB/s，退出可清理 | 先解决Store/event/生命周期 |
| LTX通信替换 | 输出与Gloo一致 | 任一静默错误即停止 |
| LTX性能 | 采样比Gloo快≥25% | 不做完整ProcessGroup，先profile |
| 相对单卡 | 接近或超过单卡 | 若明显更慢，评估USP计算/同步结构 |
| 长期稳定 | 5次完整工作流、无泄漏 | 不能设为默认后端 |
| ProcessGroup封装 | 上述全部通过 | 保留专用注入层，不扩大范围 |

## 12. 建议决策

建议批准 Phase 0–2，暂不承诺 Phase 5：

1. 先获得LTX真实消息形状；
2. 完成带超时和错误注入的双卡P2P all-to-all；
3. 在Ray actor内保持50 GiB/s以上；
4. 达标后再接入yunchang和真实工作流；
5. 真实采样比Gloo至少快25%并接近单卡后，才封装第三方ProcessGroup。

该路径用最小工程投入验证最大的不确定性，同时保留Gloo回退，不会把当前可运行环境一次性押在未成熟后端上。

## 13. 参考资料

- NVIDIA CUDA Runtime IPC：<https://nvidia.github.io/cuda-python/cuda-bindings/12.9.0/module/runtime.html>
- NVIDIA CUDA VMM/Win32共享句柄：<https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__VA.html>
- NVIDIA CUDA VMM Programming Guide：<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html>
- NVIDIA CUDA Samples：<https://github.com/NVIDIA/cuda-samples>
- PyTorch Distributed：<https://docs.pytorch.org/docs/stable/distributed.html>
- PyTorch第三方backend注册实现：<https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py>
- Raylight：<https://github.com/komikndr/raylight>
- xDiT：<https://github.com/xdit-project/xDiT>

## 14. 讨论中待定事项

以下问题留待方案评审，不在本稿中自动决定：

1. 首版采用“共享每次输入storage”还是“固定通信缓冲池”；
2. 是否先做yunchang专用注入，还是直接Python ProcessGroup；
3. 是否要求首版支持BF16以外的全部dtype；
4. `use_sync=true`在P2P event语义正确后是否可以关闭；
5. 是否在首版合并Q/K/V为packed collective；
6. P2P后端失败时采用逐次回退、整次工作流回退还是直接中止；
7. 是否把后端作为Raylight本地分支维护，或最终向上游提交；
8. 端到端目标优先级：首先追求速度，还是首先扩大可生成分辨率/帧数。
