# O15.2 双 V100 MiniMax H3 VideoVAE 解码验收

## 验证目标

验证 Raylight 1.18.0 在不修改 ComfyUI 或 MiniMaxH3Director 的前提下，能把 Director 标准 `VAEDecode` 的时间块分配给 GPU0 主进程与 GPU1 Ray worker，并同时满足：

- 输出与官方 ComfyUI `vae.decode()` 一致；
- 107 帧 1920×1088 解码至少降低 35% 耗时，热态不高于 42 秒；
- 两张 V100 都实际参与，单卡峰值不高于 7168 MiB；
- 五次重复运行不持续增长 worker、物理内存或 pagefile；
- Director 多段工作流在 VAE 与下一段采样之间正确切换；
- 不支持、单卡、非代理或异常路径回到原生解码。

## 测试条件与变量

| 项目 | 条件 |
|---|---|
| 系统 | 原生 Windows 11 23H2 |
| GPU | 2× Tesla V100-SXM2-16GB，TCC，NVLink/CUDA P2P |
| 驱动 | 577.00 |
| Python / Torch / Ray | 3.12.10 / 2.12.0+cu126 / 2.57.0 |
| VAE | MiniMax H3 VideoVAE FP16 |
| 大规格 latent | `[1,24,32,68,120]`，官方输出 `[1,3,107,1088,1920]`，6 个时间块 |
| Director 回归 | 两段 R2V，每段 73 帧；864×480、24 fps；诊断采样 1 step |
| 采样间隔 | 资源监控 0.25 秒；独立双卡解码连续 5 次 |

可复现入口为 `scripts/minimax-h3/benchmark_dual_vae_decode.py`。长遥测、模型路径、缓存 latent 与生成媒体保留为本机 `.local` 证据，不提交仓库。

## 测试关注信息

1. 官方单卡与双卡使用同一个 VAE、同一个 latent、同一个输出比较。
2. 检查 MSE、PSNR、最大绝对误差和 NaN/Inf，而不是只看“能出图”。
3. 同时记录显存、GPU 利用率/功率、物理内存、系统 commit 和 pagefile。
4. Director 回归必须出现两次 `remote sampling completed ... ranks=2` 和两次 `dual-V100 VideoVAE decode completed`，且不能出现 native fallback、Gloo timeout 或 actor death。
5. 输出媒体必须用 ffprobe 验证实际帧数、尺寸、帧率与时长；结束后检查 GPU/ComfyUI/Ray 无残留进程。

## 独立解码结果

| 路径 | 用时 | GPU0 峰值 | GPU1 峰值 | GPU 利用率峰值 |
|---|---:|---:|---:|---:|
| 官方 ComfyUI 单卡 | 66.734 s | 6993 MiB | 316 MiB | 99% / 0% |
| 双卡 run 1 | 42.066 s | 6993 MiB | 6513 MiB | 99% / 100% |
| 双卡 run 2 | 40.117 s | 6993 MiB | 6513 MiB | 99% / 100% |
| 双卡 run 3 | 40.660 s | 6993 MiB | 6513 MiB | 99% / 100% |
| 双卡 run 4 | 39.885 s | 6993 MiB | 6513 MiB | 99% / 100% |
| 双卡 run 5（热态） | 40.508 s | 6993 MiB | 6513 MiB | 99% / 100% |

- 热态加速比：`1.647×`；耗时降低 `39.30%`。
- 双卡五次中位数：`40.508 s`。
- 质量：MSE `0`、最大绝对误差 `0`、PSNR `∞`，所有值有限；输出逐元素等同官方结果。
- rank1 CPU 后台预载：`15.527 s`，与采样重叠，不占用预载期 GPU 显存。
- 双卡 run 1 到 run 5 的进程结束物理内存增长为 `2621.74 MiB`，低于 4096 MiB 门槛；后四次保持同一平台，pagefile 使用峰值为 `0`。
- 所有有限值、质量、42 秒、35%、7168 MiB 和生命周期门禁均通过。

## Director 完整工作流结果

| 指标 | 结果 |
|---|---:|
| 基准器端到端 | 211.675 s |
| ComfyUI Prompt | 209.66 s |
| 两段 worker 采样 | 41.764 / 17.722 s |
| Director 报告 conditioning | 32.2 / 30.5 s |
| Director 报告 sample | 42.5 / 18.3 s |
| 双卡 VideoVAE | 9.723 / 9.282 s |
| Director 报告 cache | 1.5 / 1.5 s |
| 保存 | 5.804 s |
| 整卡峰值 | 9431 / 9409 MiB |
| GPU 利用率峰值 | 99% / 100% |
| 物理内存峰值 | 68640.65 MiB |
| 系统 commit 峰值 | 114212.43 MiB |

最终 MP4 为 864×480、24 fps、146 帧、6.083 秒。两个采样 operation 都由相同两个 rank 完成；每段解码后下一段扩散模型可重新同步加载；运行结束后无 GPU、Ray 或 ComfyUI 残留进程。

自动化全量结果为 `797 passed, 235 warnings, 35 subtests passed`。从 O15.1 基线到 O15.2 的全部 Python 变更通过 Ruff。全仓 Ruff 仍报告 27 个历史未使用导入/变量，均位于本阶段未修改的 Wan、GGUF、debug 和旧 SeedVR2 文件；本版没有借功能发布之机改写这些无关路径。

## 实验性尝试与修复记录

| 尝试/问题 | 结果 | 是否实装 |
|---|---|---|
| 每次解码后立即卸载 rank1 VAE | 正确但第二次约 42.97 秒，未通过 42 秒绝对门槛；反复重载也增加内存抖动 | 否 |
| 在同一 Director `execute` 内复用 resident VAE，下一次采样前释放 | 热态 40.508 秒；连续五次内存受控；不把峰值延迟到扩散阶段 | 是 |
| 独立 harness 默认启用 autograd | 产生额外图与内存，不能代表推理路径 | 否；基准和生产解码均强制 `torch.inference_mode()` |
| rank1 辅助 VAE 误入双 rank 模型加载预算同步 | 完整工作流在 Gloo 等待 60 秒后 actor 退出 | 已修复；仅辅助 VAE 作用域绕过 rank 同步，扩散/FSDP 同步保持原样 |
| 在 `except` 内立即调用原生回退 | traceback 会暂时持有失败解码的显存张量，OOM 后回退可能再次 OOM | 已修复；退出异常作用域并释放 traceback 后再调用原生解码 |
| 五次基准只比较最后一次输出 | 不能证明前四次也通过质量门禁 | 已修复；每次都记录 MSE/PSNR/max-error/finite 并共同门禁 |
| 修改 ComfyUI/Director 或实现通用 VAE FSDP | 成本、兼容面和回归风险明显高于时间块并行 | 否 |

## 能力边界与总结

O15.2 通过全部验收。它优化的是 Ray MODEL 代理 Director 执行作用域内的**标准 VideoVAE `VAEDecode`**，不是把 VAE 权重做 FSDP 分片。主解码以及通过标准节点进入的 Refine/FaceRefine 解码可使用本路径。AudioVAE、SelfLift 内部直接调用 `vae.decode()`、非 Director、单卡和未知 VAE 保持原生路径；并行失败会记录警告并调用原始解码，不改变工作流结果语义。

