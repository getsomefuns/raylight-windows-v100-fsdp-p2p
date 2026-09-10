# O12 Python 3.12 / Torch 2.12 与 MiniMax H3 加速节点验收

- 日期：2026-09-10
- 版本：Raylight Windows V100 FSDP/P2P 1.14.0
- 状态：`PASS`（兼容、稳定性和媒体质量）；性能收益按节点分别定级

## 验证目标

1. 将独立运行时升级为 Python 3.12，并保留 V100 `sm_70`、Windows Gloo、FSDP2 与 CUDA P2P/NVLink 能力。
2. 只修改 Raylight，让 MiniMaxH3Director 的 MODEL 链可选串联 TE-Speed 3.3 与 Sol-Attn V100；不修改 ComfyUI、Director 或两个外部节点源码。
3. 保持原有 Base 工作流、未安装外部节点时的旧路径和既有节点接口不变。
4. 在同一输入、提示词、分辨率、帧数和运行参数下，对 4/8 步的 Base、TE、Sol、TE+Sol 做冷/热矩阵测试。

## 条件与变量

| 项目 | 固定条件 |
|---|---|
| 硬件 | 2× Tesla V100-SXM2-16GB，TCC，NVLink/CUDA P2P |
| 驱动 | 577.00 |
| 运行时 | Python 3.12.10；Torch 2.12.0+cu126；xFormers 0.0.35；Ray 2.57.0 |
| 并行 | 2 rank；Ulysses=2；FSDP=true；CPU offload=true；128 MiB 自适应 P2P staging |
| 工作流 | MiniMaxH3Director REF2VA，864×480，124 帧，24 fps，约 5.17 秒 |
| 输入 | `minimax_h3_ref2va_green_robots.jpg`；各组保持同一提示词 |
| 采样 | 官方 REF2VA Turbo4 或 Turbo8 LoRA；Raylight `fp16_h3_safe` |
| 资源 | `-ReserveVramGiB 4`；DynamicVRAM 关闭；详细 P2P/rank 诊断开启用于验收 |
| 冷/热 | 每组在一个新 ComfyUI 进程内连续跑两次；第二次复用同一 worker 对与 FSDP shard |

本矩阵是 REF2VA，不是社区 132/200 秒样例所用的 FL2V，因此社区数字只作量级参考；发布判断优先使用本机 Base 对照。

## 关注信息

- 外部 MODEL patch 是否在两个真实 Ray worker 中重放，而非只停留在主进程代理。
- FSDP、Ulysses、LoRA、FP8 checkpoint 的安全 FP16 解量化和 P2P collective 是否保持正常。
- 采样、端到端、conditioning、解码和保存时间；两卡显存/利用率/功率、物理内存和提交内存峰值。
- 黑屏、噪声、NaN/Inf、冻结帧、音频/视频容器损坏、worker 泄漏和连续运行增长。

## 矩阵结果

### 热启动（主要性能口径）

| 步数 | MODEL 链 | 端到端 | worker 采样 | 进度条 | GPU 峰值 MiB（0/1） | 物理内存峰值 MiB | 提交峰值 MiB | 结论 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4 | Base | 142.04 s | 85.23 s | 17.3 s/it | 9311 / 9287 | 71072 | 95482 | 基线通过 |
| 4 | TE | 149.27 s | 88.88 s | 18.0 s/it | 9311 / 9287 | 70741 | 95552 | 无稳定收益 |
| 4 | Sol | 136.49 s | 69.28 s | 14.0 s/it | 9311 / 9287 | 71169 | 96101 | 采样较 Base -18.7% |
| 4 | TE+Sol | **130.67 s** | 73.32 s | 14.8 s/it | 9311 / 9287 | 71542 | 95678 | 本轮端到端最快；采样不快于 Sol |
| 8 | Base | 228.86 s | 170.90 s | 20.5 s/it | 9311 / 9287 | 70717 | 95579 | 基线通过 |
| 8 | TE | 228.49 s | 162.52 s | 19.5 s/it | 9311 / 9287 | 70842 | 95503 | 小幅波动，不能确认缓存收益 |
| 8 | Sol | 210.54 s | **138.06 s** | 16.5 s/it | 9237 / 9213 | 71145 | 96202 | 采样较 Base -19.2% |
| 8 | TE+Sol | **200.05 s** | 138.13 s | 16.6 s/it | 9311 / 9287 | 68010 | 92435 | 端到端最快；采样与 Sol 相同 |

### 冷启动

| 步数 | Base | TE | Sol | TE+Sol |
|---:|---:|---:|---:|---:|
| 4 步端到端 | 209.13 s | 221.24 s | 207.91 s | **204.12 s** |
| 4 步采样 | 103.25 s | 107.77 s | 99.01 s | **97.62 s** |
| 8 步端到端 | 305.32 s | 302.60 s | **273.75 s** | 280.40 s |
| 8 步采样 | 192.82 s | 189.03 s | **163.76 s** | 167.07 s |

### 结果解释

- Sol 是本矩阵中可重复确认的采样加速来源：热启动 4/8 步采样分别缩短 18.7%/19.2%。
- TE 单独没有给 4 步带来收益，8 步的单次小幅变化不足以证明稳定加速。同 seed 下 Base 与 TE 的视频码流 MD5 完全一致，Sol 与 TE+Sol 也完全一致；collective 次数也相同。这说明当前短轨迹验收中 TE 没有改变可见结果，也没有减少 Raylight 采样通信工作量。
- TE+Sol 的热启动端到端恰好是两组最快值，但其采样与 Sol 相同或略慢。conditioning 在独立运行间波动约 25–42 秒，不能把组合组的端到端优势全部归因于 TE。
- 因此日常追求确定的采样收益优先使用 Sol 工作流；需要复现社区串联拓扑或进一步验证多段缓存时使用 TE+Sol。TE-only 保留为可选实验对照，不设为默认。

## 连续三次与回收

4 步 TE+Sol 在同一 ComfyUI 进程连续执行三次，worker PID 始终为同一对。冷/热/热端到端为 206.65/129.32/128.10 秒，采样为 93.87/71.80/71.17 秒。后两轮物理内存峰值由 69270 降到 68111 MiB，提交峰值由 94066 降到 92470 MiB；GPU 峰值从 9247/9223 到 9391/9367 MiB 的 144 MiB 变化没有继续累加或触发 OOM。进程退出后两张卡均回到 10 MiB，目标 worker 不存在。

## 媒体与质量

- 16 个矩阵输出全部为 H.264 + AAC、864×480、24 fps、124 帧、5.1667 秒；FFmpeg 全流解码返回 0。
- 每个视频均为 124/124 个唯一解码帧，blackdetect 事件为 0。
- Base、Sol、8 步组合的首/中/尾联系表人工检查未见黑屏、块状乱码、首帧后噪声或主体崩坏。
- Sol 使用稀疏注意力，输出与 dense Base 按设计不同；本轮只确认可播放性和明显故障，不声称与 Base 像素等价。最终主观质量仍应由用户按生产素材复核。

本机原始 benchmark JSON、遥测、日志和联系表保存在 `.local/artifacts/o12`，不进入 Git；生成视频保存在 `ComfyUI/output/video/raylight_o12`。

## 运行时迁移结论

- PyTorch 2.8.0+cu126 Windows wheel 的最小 Gloo 探针失败，报 `makeDeviceForHostname(): unsupported gloo device`，不能作为 Raylight Windows FSDP 控制面。
- Python 3.12.10 + PyTorch 2.12.0+cu126 通过 Gloo、双 Ray actor、xFuser subgroup、FSDP2量化权重和 CUDA P2P 探针；P2P 健康检查约 59.87 GiB/s。
- Python310/Torch2.7 环境仍保留为回滚 profile；1.14.0 启动脚本在环境根目录存在 Python312 时优先选择它。

## 总结

O12 达成运行时升级、旧路径兼容、两个外部节点的 Ray worker 桥接、4/8 步全矩阵、媒体质量和三连稳定性验收。推荐生产默认仍是 Base 或 Sol 二选一：Base 保持精确原路径，Sol 提供约 19% 的采样阶段收益。TE 与 TE+Sol 是显式可选能力，不能把本次短轨迹的端到端波动宣传成稳定 TE 加速。
