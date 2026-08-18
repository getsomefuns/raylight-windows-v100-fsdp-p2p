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

功能和质量目标通过，但 O6 尚未通过初始 4× 性能门槛，因此 `fp16_h3_safe` 继续保持显式实验模式。V100 GEMM extent 对齐作为匹配路径的自动改进保留。有界 host registration 已实现但默认关闭，因为它改善采样速度的同时，使同规格冷启动端到端结果变慢。被拒绝的内核、chunk 大小、Attention、拓扑、注册容量和 prefetch 实验，以及对应资源数据，见[中文升级记录](../../releases/2026-08-18-o6-minimax-h3-safe-fp16.zh-CN.md)和[英文升级记录](../../releases/2026-08-18-o6-minimax-h3-safe-fp16.en.md)。

## 证据索引

原始日志、遥测 CSV、API prompt 和 benchmark JSON 保留在本机：

- `logs/minimax-h3/o2/20260818-155526-i2v-full-o6-baseline-fp32-p2p256/`
- `logs/minimax-h3/o2/20260818-162159-ref2va-full-o6-baseline-fp32-p2p256/`
- `ComfyUI/output/video/raylight_o3/minimax_h3_i2v_o6-baseline-fp32-p2p256_run0_00001_.mp4`
- `ComfyUI/output/video/raylight_o3/minimax_h3_ref2va_o6-baseline-fp32-p2p256_run0_00001_.mp4`
- `logs/minimax-h3/o2/20260818-184842-i2v-full-o6-safe-fp16-full-reviewed/`
- `logs/minimax-h3/o2/20260818-185943-ref2va-full-o6-safe-fp16-full-reviewed/`
- `logs/minimax-h3/o2/20260818-201232-ref2va-full-o6-safe-fp16-align8-full-reviewed/`
- `logs/minimax-h3/o2/20260818-212923-ref2va-full-o6-safe-fp16-hostreg5g-scoped-full/`

这些数据来自固定机器状态下每个工作流各一次冷启动。若几何、帧数、步数、P2P 容量、精度策略、模型资源或核心运行时版本发生变化，必须重新建立同规格基线。
