# Raylight Windows 双 V100 测试与验证记录

本文件是本分支的持续维护测试总表。正文只保留能够支持发布判断的条件、关键结果和结论；逐行日志、完整监控采样与生成视频不进入仓库。

- 最近更新：2026-08-15
- 适用分支：`windows-v100-p2p`
- 上游基线：Raylight 1.9.0 / `9a7c33d52b3d35e29f75ecff3c227de987f0d4cf`
- 精简数据附件：[test-results-2026-08.csv](test-results-2026-08.csv)

## 维护规则

每次影响通信、模型加载、内存策略或端到端性能的修改，都应增加一条带唯一 ID 的记录。必须写明冷/热启动定义、模型精度、时长、mmap、P2P 缓冲区和超时；失败测试不得从历史中删除，应标记为 `FAIL` 或 `REJECTED` 并说明用途。

状态含义：

- `PASS`：满足该测试预先定义的正确性或稳定性标准。
- `FAIL`：未满足标准，且问题仍需修复。
- `REJECTED`：技术路径能够运行，但质量或架构不符合项目目标。
- `DIAGNOSTIC`：只用于定位问题，数据不能直接作为发布性能结论。

## 基准环境

| 类别 | 已验证条件 |
|---|---|
| 系统 | 原生 Windows 23H2，单机 |
| GPU | 2× Tesla V100-SXM2-16GB，TCC，NVLink/P2P 可用 |
| 驱动 | NVIDIA 577.00 |
| Python / PyTorch | Python 3.10.11；PyTorch 2.7.0+cu126，保留 sm_70 |
| 主要依赖 | Ray 2.57.0；xFuser 0.4.5；yunchang 0.6.4；xformers 0.0.30 |
| ComfyUI | 0.31.0，启动时使用 `--disable-cuda-malloc` |
| 并行设置 | GPU=2，Ulysses=2，Ring/CFG/DP=1，FSDP 关闭 |
| Attention | `TORCH_EFFICIENT`，`sync_ulysses=true`，不安装 FlashAttention |
| Windows 通信 | TCPStore/Gloo 控制面；受支持的 CUDA all-to-all 使用 CUDA IPC/P2P 数据面 |

详细版本见 [../environment-windows-v100.json](../environment-windows-v100.json)。

## 测试条件变量

端到端结果只有在以下变量一致时才允许直接比较：

| 变量组 | 必须记录的内容 |
|---|---|
| 启动状态 | 冷启动、worker 复用、ComfyUI 节点缓存命中情况 |
| 工作流 | 输入图片、提示词、种子、分辨率、帧率、视频时长、采样阶段与步数 |
| 模型 | UNet/LoRA/VAE/文本编码器版本，`weight_dtype` |
| 内存 | `use_mmap`、P2P capacity、ComfyUI 显存策略、物理内存与提交内存峰值 |
| 通信 | GPU/rank 数、Ulysses 配置、TCC/WDDM、严格超时、张量大小 |
| 质量 | 输出帧数、音频流、全黑帧、NaN/Inf、可播放性 |

“热启动”必须进一步区分：worker/模型复用与 ComfyUI 上游节点缓存不是同一件事。只重跑工作流下游节点的结果不能作为完整热启动基准。

## 测试关注信息

1. **正确性**：两个 rank 输出是否一致，是否存在 mismatch、NaN/Inf、黑屏或静音。
2. **数据路径**：实际 CUDA 张量是否走 P2P/NVLink，而不是 Gloo 主存中转。
3. **稳定性**：严格 10 秒超时下是否有 rank 偏差、actor death、OOM 或分页停顿。
4. **资源**：两卡显存、GPU 利用率/功率、物理内存、提交内存和页面文件。
5. **性能**：端到端时间与各 sampler 的 s/it；冷对冷、热对热分别比较。
6. **可复现性**：公开脚本、示例工作流和环境矩阵是否与实测条件一致。

## 测试数据结果

### 1. 硬件与通信基线

| ID | 目标 | 条件 | 关键结果 | 结论 |
|---|---|---|---|---|
| HW-001 | 验证原生 CUDA P2P/NVLink | 双 V100、TCC、NVIDIA CUDA sample | peer access/正确性通过；工具显示双向聚合带宽 240+ GB/s | `PASS`，硬件链路可用 |
| TR-001 | 验证 Windows Gloo 功能 | 双进程 Gloo all-to-all | 最小正确性测试通过，但 CUDA 张量需经主存 | `REJECTED`，只保留控制面/兼容路径 |
| TR-002 | 验证自定义 CUDA P2P 正确性 | 两个 Ray actor，真实 LTX 尺寸 | 所有尺寸 0 mismatch / 0 maximum error | `PASS` |
| TR-003 | 评估传输上限与项目开销 | 约 100 MiB 分块 | 独立探针峰值约 108 GiB/s；项目启动探针约 59.2 GiB/s 每方向 | `PASS`，但仍有优化空间 |
| TR-004 | 验证 xFuser 接入 | 双 rank subgroup | 集成探针通过 | `PASS` |

不同工具对单向、双向聚合和有效载荷的定义不同，240+ GB/s、108 GiB/s 与 59.2 GiB/s 不应直接互相替代。

### 2. Windows P2P 实现与故障保护

| ID | 目标 | 条件 | 关键结果 | 结论 |
|---|---|---|---|---|
| REL-001 | 单元与故障测试 | P2P、trace、session、runtime、mmap、模型加载同步 | 正确性、operation ID 不一致、peer 缺失和 poisoned endpoint 均有覆盖 | `PASS`；以当前测试命令输出为准 |
| REL-002 | 验证严格超时 | 正常默认 10 秒；故障探针 2 秒 | 故障能在限定时间退出，不静默降级为主存路径 | `PASS` |
| REL-003 | 验证 10 秒张量容量 | 128 MiB/rank 持久缓冲区 | 已覆盖最大 230,686,720 字节 collective input | `PASS`；64 MiB 仅适合较短序列 |
| REL-004 | 验证发布配置一致性 | 脚本、worker 默认值、环境清单、Ray 环境转发 | 128 MiB 与严格 10 秒一致；诊断/超时变化会重建 worker 会话 | `PASS` |

### 3. LTX 2.3 端到端性能

同一模型、输入、提示词和工作流规模下，已完成的公平基准为：

| ID | 场景 | 端到端时间 | 结论 |
|---|---|---:|---|
| PERF-001 | 单 V100 冷启动 | 519.94 s | 冷启动参考，不用于计算加速比 |
| PERF-002 | 单 V100 热启动 | 316.60 s | 公平热启动基线 |
| PERF-003 | 双 V100 P2P，复用会话中位数 | 284.06 s | 相对单卡热启动快 10.28% |

当前真实提升尚未达到项目的 20% 目标。双卡运行与 P2P 数据路径已经验证，但模型加载、非分布式 VAE/文本编码以及小 collective 开销仍占明显比例。

### 4. rank 偏差、mmap 与 10 秒稳定性

| ID | 目标 | 条件变化 | 关键结果 | 结论 |
|---|---|---|---|---|
| STAB-001 | 复现 10 秒不稳定 | FP8 模型未走 mmap，rank 独立决定加载预算 | 可出现长时间 rank 偏差、严格超时或 actor death | `FAIL`，用于定位根因 |
| STAB-002 | 验证量化 mmap | FP8 safetensors 开启 mmap，并传递 metadata | 模型类型识别正确，减少重复实体化与分页压力 | `PASS` |
| STAB-003 | 验证模型加载同步 | 两 rank 取最小显存预算，加载后 barrier | 模型加载完成时间对齐到毫秒以内 | `PASS` |
| STAB-004 | 验证修复后的 10 秒工作流 | mmap=true、128 MiB、严格 10 秒、默认 dtype | 9.96 秒视频与 AAC 正常保存；端到端 581.35 s | `PASS` |
| STAB-005 | 验证修复后的 rank 偏差 | 同上，诊断开启 | sampler 入口/开始偏差约 0.06 s 以内；未通过放宽超时掩盖问题 | `PASS` |

原始逐行日志保存在开发机的独立 `benchmark_results` 目录，不作为仓库内容。可公开复查的精简数字见 CSV 附件。

### 5. dtype 数值验证

| ID | 目标 | 条件 | 速度现象 | 质量结果 | 结论 |
|---|---|---|---|---|---|
| NUM-001 | 测试 5 秒 LTXAV 全局 FP16 | `RayUNETLoader weight_dtype=fp16` | sampler 明显加快，P2P 数据量减半 | 视频全黑；音频含 NaN/Inf | `REJECTED` |
| NUM-002 | 测试把 LTX dtype 白名单加入 FP16 | 10 秒、双 V100、FP16 | 第一阶段 8.31 s/it；第二阶段 18.1 s/it | 249 帧全部为黑色；AAC 拒绝 NaN/Inf | `REJECTED`，白名单已恢复 |

LTX/LTXAV 在当前 ComfyUI 配置中继续只声明 BF16/FP32。V100 擅长 FP16 不代表该模型的所有运算范围都能安全使用 FP16；不得用失败运行的速度数字宣传性能。

### 6. 内存与缓存诊断

| ID | 目标 | 关键结果 | 结论 |
|---|---|---|---|
| MEM-001 | 观察 5/10 秒物理与提交内存 | 大模型双 worker 会逼近 64 GiB 物理内存上限，提交内存明显更高 | 必须同时监控物理内存、提交内存和页面文件 |
| MEM-002 | 验证 ComfyUI 热缓存影响 | 5 秒默认运行出现 385.37 s 冷启动与 145.15 s 下游缓存重跑 | `DIAGNOSTIC`，145.15 s 不是完整热启动基准 |
| MEM-003 | 验证 mmap 与显存容量 | 量化 safetensors mmap + 128 MiB P2P 缓冲区可完成 10 秒默认精度工作流 | `PASS`，仍需为更大模型保留内存余量 |

> Windows FSDP 分支的计划、失败样本、根因修复与视觉验收已独立维护在 [WINDOWS_V100_FSDP_TESTING.md](WINDOWS_V100_FSDP_TESTING.md)，不再混入本 P2P/Ulysses 历史汇总。

## 测试总结

- 原生 Windows 无 NCCL，但 Gloo 控制面配合定向 CUDA IPC/P2P 数据面，可以让已匹配的双 rank Ulysses all-to-all 实际使用 NVLink。
- 当前实现已通过硬件 P2P、独立传输、双 Ray actor、xFuser subgroup 和 LTX 端到端验证。
- 10 秒不稳定的根因不是“超时太短”，而是量化模型未 mmap、metadata 丢失以及两 rank 模型加载预算不同步；修复后仍保持严格 10 秒标准。
- 128 MiB/rank 是已验证 10 秒工作流的发布默认值；更长序列必须重新验证最大 collective，而不是盲目增大缓冲区。
- 全局 FP16 对 LTXAV 数值不安全。速度提高伴随黑屏和音频 NaN/Inf，因此不属于可用优化。
- 当前公平热对热加速为 10.28%，低于 20% 目标。后续优化应聚焦通信复制、控制面和非并行阶段，而不是牺牲输出质量。

## 证据与复现入口

- 环境自检：[../scripts/verify-windows-v100.ps1](../scripts/verify-windows-v100.ps1)
- 启动脚本：[../scripts/start-comfyui-windows-p2p.ps1](../scripts/start-comfyui-windows-p2p.ps1)
- Windows P2P 单元测试：[../tests/test_windows_p2p.py](../tests/test_windows_p2p.py)
- 故障探针：[../tests/windows_p2p_failure_probe.py](../tests/windows_p2p_failure_probe.py)
- 双 Ray actor 探针：[../tests/windows_p2p_ray_probe.py](../tests/windows_p2p_ray_probe.py)
- mmap/metadata 测试：[../tests/test_windows_mmap_policy.py](../tests/test_windows_mmap_policy.py)、[../tests/test_lazy_loader_metadata.py](../tests/test_lazy_loader_metadata.py)
- rank 模型加载同步测试：[../tests/test_model_load_sync.py](../tests/test_model_load_sync.py)
- 发布配置测试：[../tests/test_windows_release_profile.py](../tests/test_windows_release_profile.py)

## 新测试记录模板

```markdown
### TEST-ID：简短名称

- 日期：YYYY-MM-DD
- 状态：PASS / FAIL / REJECTED / DIAGNOSTIC
- 验证目标：
- 条件与变量：
- 关注信息：
- 精简结果：
- 质量检查：
- 总结：
- 数据附件或测试入口：
```