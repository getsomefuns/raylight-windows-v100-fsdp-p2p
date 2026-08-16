# Windows 双 V100 FSDP 测试与验收记录

本文只维护 `windows-v100-fsdp-p2p` 分支的 FSDP 计划、诊断、失败记录与验收结果。旧版 Windows P2P/Ulysses 项目的数据继续保留在 [TESTING.md](TESTING.md)，两者不再混写。

- 最近更新：2026-08-16
- 精简数据附件：[windows-v100-fsdp-test-results-2026-08.csv](windows-v100-fsdp-test-results-2026-08.csv)
- 当前状态：FSDP 显存分片、CUDA P2P 数据面和 LTX 5 秒视觉输出已通过；性能优化目标尚未通过
- 历史阶段：F0 方案评估、F2 合成模型门禁、F3/F4 LTX 集成与输出修复

## 测试验证目标

1. 在原生 Windows、双 Tesla V100 TCC 环境中运行 FSDP2 推理。
2. 权重分片通信使用 CUDA IPC/P2P/NVLink；Gloo/TCPStore 只承担控制面。
3. 22B FP8 scaled LTX 模型的持久权重分摊到两张 16GB 卡，而不是每卡完整复制。
4. 两个 rank 在每个采样阶段返回一致、有限且可解码的 latent。
5. 输出视频不仅“能播放”，还必须逐帧呈现连贯内容；黑屏、彩色噪声和稳定颗粒噪声均判定失败。
6. 最终性能目标为快于公平单卡基线 20%；该目标与“显存模式可用”分开验收。

## 测试条件与变量

| 类别 | 最终验收条件 |
|---|---|
| 系统与 GPU | 原生 Windows 23H2；2× Tesla V100-SXM2-16GB；TCC；NVLink/P2P 可用 |
| 驱动 | NVIDIA 577.00 |
| Python / PyTorch | Python 3.10.11；PyTorch 2.7.0+cu126；保留 sm_70 |
| 分布式依赖 | Ray 2.57.0；xFuser 0.4.5；yunchang 0.6.4；xformers 0.0.30 |
| ComfyUI | 0.31.0；`--disable-cuda-malloc` |
| 拓扑 | GPU=2；FSDP=true；Ulysses/Ring/CFG=0；DP=1；CPU offload=false |
| 通信 | Gloo/TCPStore 控制面；自定义 CUDA IPC/P2P 数据面；严格 10 秒超时 |
| 内存 | `use_mmap=true`；P2P capacity=128 MiB；VAE tile=384 |
| 模型 | `ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors` |
| LoRA | `ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors`，strength=1.0 |
| 工作流 | 相同输入图、提示词、种子；两阶段 8+3 步；1280×704；121 帧；25 fps；约 4.84 秒 |
| 数值路径 | FP8 scaled 主权重；BF16/FP32 辅助权重；worker 返回 FP32 latent；VAE 解码为 FP32 |

## 测试关注信息与质检标准

- 通信：P2P 健康探针不低于 50 GiB/s，不能静默回退到主存传输。
- 分片：记录 wrapper、DTensor、每 rank 持久模型负载和最大聚合张量。
- 同步：两个 rank 的视频/音频 latent 形状和 dtype 相同，最终逐元素一致。
- 数值：采样返回值必须全部有限；但“无 NaN/Inf”不能代替视觉检查。
- 视觉：至少抽取首、中、尾多个时间点；内容需连贯，不能只检查编码、帧数或非黑像素。
- 资源：记录双卡峰值显存、利用率、功耗，以及物理/提交内存和页面文件。
- 性能：冷启动、worker 复用和 ComfyUI 节点缓存必须分别标注，不能交叉计算加速比。

## 分阶段测试结果

### A. 通信与分片基础门禁

| ID | 验证目标 | 关键结果 | 状态 |
|---|---|---|---|
| F2-P2P-001 | FSDP all-gather 分块与尾块正确性 | 64–384 MiB、FP32/FP16/BF16/uint8、同步/异步均通过；最大误差 0 | `PASS` |
| F2-P2P-002 | 大于固定 staging buffer 的分片 | 256 MiB 本地分片可通过 128 MiB buffer 分两次传输，边界字节正确 | `PASS` |
| F2-FSDP-001 | 512 MiB 合成权重嵌套 FSDP | 每 rank 只持有 256 MiB 分片；forward 后重新分片；五个新会话结果有限且误差 0 | `PASS` |
| F3-COMM-001 | LTX 权重通信实际走 P2P | 多次启动约 51.6 GiB/s；后续工作流健康探针约 59 GiB/s | `PASS` |
| F3-MEM-001 | LTX 权重实际分片 | 2,999 个 FSDP wrapper；每 rank 持久模型 payload 约 11,203 MiB | `PASS` |

F2 的完整合成模型数据见 [windows-v100-fsdp-phase-f2-results.md](windows-v100-fsdp-phase-f2-results.md)。TCPStore 的已知 Windows IPv6 候选警告分类见 [windows-v100-fsdp-phase-f2-diagnostics.md](windows-v100-fsdp-phase-f2-diagnostics.md)。

### B. 输出失败、隔离与修复

| ID | 条件 | 结果 | 状态 |
|---|---|---|---|
| F3-OUT-001 | 早期 FSDP，带 LoRA | H.264/AAC、帧数、非黑、无 NaN/Inf 均通过，但实际视频除首帧外为噪声 | `FAIL`；历史 `PASS` 作废 |
| F4-DIAG-001 | Ray 单 worker，无任何 collective | 修复前仍为彩色颗粒噪声，定位到 Ray mmap 量化加载公共路径 | `DIAGNOSTIC` |
| F4-FIX-001 | 保留 FP8 `comfy_quant` 与 `weight_scale` | 实际 4096×4096 权重反量化误差约 3e-8；Ray 单卡和 Ulysses 成片恢复正常 | `PASS` |
| F4-FIX-002 | FSDP FP8 all-gather 恢复全局逻辑 shape | 单层硬件探针误差 0；但完整 FSDP 视频仍为噪声 | `PASS`，但非最终根因 |
| F4-DIAG-002 | FSDP 两 rank 返回值指纹 | 两阶段视频/音频 latent 均逐元素完全一致，`sample_max_abs=0` | `PASS`；排除 rank 漂移/P2P 损坏 |
| F4-DIAG-003 | FSDP 去掉 LoRA，其余变量不变 | 仍为明亮彩色噪声；说明 LoRA 不是主根因 | `FAIL`，用于隔离 |
| F4-FIX-003 | RMSNorm 分块计算不再覆盖输入 | 单元测试验证输出等价且原 residual 输入保持不变 | `PASS` |
| F4-OUT-001 | 修复后 FSDP，无 LoRA | 121 帧森林/机器人场景连贯；不再是噪声 | `PASS` |
| F4-OUT-002 | 修复后 FSDP，恢复原 BF16 LoRA | 121 帧、H.264/AAC、五个时间点画面连贯；双 rank 完全一致 | `PASS`，当前视觉验收基线 |

## 根因说明

最终 FSDP 噪声不是 BF16 指数范围不足，也不是 FP32 溢出或 VAE 缺少 BF16 算子。

真正的 FSDP 专用根因是：为了降低峰值显存，V100 路径把 `torch.nn.functional.rms_norm` 改成了分块原地写入。LTX Transformer 会先计算归一化分支，随后继续使用原输入作为 residual。原地版本提前改写了 residual，导致每一层的主干状态都被破坏。该错误保持有限数值，且两个 rank 会计算出完全相同的错误结果，因此旧的“可播放、非黑、无 NaN/Inf”检查无法发现它。

修复后的实现仍按 16 MiB 临时预算分块，但写入独立输出张量，不修改输入 residual。新增输出缓冲没有使双 16GB V100 溢出。

另有两个独立的真实缺陷已经一并修复：

1. Ray mmap 懒加载曾把 FP8 原始数值直接转换为 FP32，却忽略 `weight_scale`，会让权重放大数百倍。
2. FSDP FP8 all-gather 曾保留本地半分片的逻辑 shape，而不是完整聚合后的 shape。

前者会同时影响 Ray 单卡和 Ulysses；后者会影响 FSDP 张量元数据。两项都需要保留，但完整 FSDP 的剩余噪声最终由 RMSNorm residual 破坏造成。

## 最终实测数据

| 场景 | 冷端到端 | Sampler 1 | Sampler 2 | GPU0/1 峰值显存 | 视觉结果 |
|---|---:|---:|---:|---:|---|
| FSDP 无 LoRA，修复前 | 476.72 s | 约 10.8 s/it | 约 41.1 s/it | 约 16.0/16.2 GiB | 彩色噪声，`FAIL` |
| FSDP 无 LoRA，修复后 | 479.83 s | 约 11.0 s/it | 约 41.7 s/it | 16,218/16,208 MiB | 连贯画面，`PASS` |
| FSDP + 原 LoRA，修复后 | 551.82 s | 约 15.0 s/it | 约 45.6 s/it | 16,224/16,156 MiB | 连贯画面，`PASS` |

最终带 LoRA 运行的两卡利用率均达到 100%，峰值功耗约 355W/348W；物理内存峰值约 62.6 GiB，提交内存峰值约 110.0 GiB，页面文件未承担模型权重传输。

当前 FSDP 解决的是“每卡 16GB 无法完整容纳 22B 模型”的显存问题。带 LoRA 冷端到端仍慢于正确的 Ray/Ulysses 速度路径，尚未达到“快于公平单卡 20%”的最终性能目标，不得把本次正确性通过描述为性能验收通过。

## 证据与测试入口

仓库内可复现入口：

- FSDP 方案基线：[windows-v100-fsdp-phase-f0.md](windows-v100-fsdp-phase-f0.md)
- F2 合成模型验收：[windows-v100-fsdp-phase-f2-results.md](windows-v100-fsdp-phase-f2-results.md)
- FP8 mmap scale 回归：[../tests/test_lazy_quantized_safetensor_ops.py](../tests/test_lazy_quantized_safetensor_ops.py)
- FP8 gather shape 回归：[../tests/test_fp8_fsdp_gather_shape.py](../tests/test_fp8_fsdp_gather_shape.py)
- RMSNorm residual 回归：[../tests/test_chunked_rms_norm.py](../tests/test_chunked_rms_norm.py)
- rank latent 指纹回归：[../tests/test_rank_sampling_diagnostics.py](../tests/test_rank_sampling_diagnostics.py)
- FSDP 硬件探针：[../tests/windows_p2p_fsdp_probe.py](../tests/windows_p2p_fsdp_probe.py)
- LTX 模式基准工具：[../tests/windows_ltx_mode_benchmark.py](../tests/windows_ltx_mode_benchmark.py)

开发机原始附件不进入仓库正文；复查时按环境根目录定位：

- `{ENV_ROOT}/logs/f4/f4-fsdp-no-lora-benchmark.json`
- `{ENV_ROOT}/logs/f4/f4-fsdp-no-lora-rmssafe-benchmark.json`
- `{ENV_ROOT}/logs/f4/f4-fsdp-rmssafe-benchmark.json`
- `{ENV_ROOT}/logs/f4/visual-review/f4_fsdp-no-lora_run0_00001_.jpg`（失败样本）
- `{ENV_ROOT}/logs/f4/visual-review/f4_fsdp-no-lora-rmssafe_run0_00001_.jpg`（无 LoRA 通过）
- `{ENV_ROOT}/logs/f4/visual-review/f4_fsdp-rmssafe_run0_00001_.jpg`（带 LoRA 通过）

## 维护规则

- 媒体“可解码、非黑、无 NaN/Inf”只能作为基础门禁，不能替代视觉验收。
- 失败记录不得删除；修复后新增独立 `PASS`，并明确旧结论作废。
- P2P/Ulysses 历史性能继续维护在 `TESTING.md`；FSDP 只维护在本文。
- 新模型、10 秒及更长视频必须重新验证最大 collective、峰值显存、rank 指纹和视觉质量。
- 性能数据必须同条件比较，且只能使用视觉通过的运行。
