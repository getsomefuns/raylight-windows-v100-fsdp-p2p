# 1.11.0 / O9：Windows CUDA P2P 自适应分块

日期：2026-08-31

## 用户可见变化

- 默认每 rank 的 CUDA staging budget 从 256 MiB 改为 128 MiB。
- 合法的双 rank、equal-split、连续 CUDA `all_to_all_single` 与 FSDP `all_gather_into_tensor` 即使大于 staging budget，也会在同一个逻辑调用中自动拆成多个 CUDA P2P/NVLink 子块。
- 普通用户直接运行启动脚本，不再按视频时长、分辨率或模型手工选择 256/512 MiB。
- `-P2PCapacityMiB 256/512` 保留为高级性能对照；它不再是修复大 payload 的方法。
- `-ValidateOnly` 会显示 staging budget 和 `Larger collectives are auto-chunked: enabled`。
- 详细 rank/P2P 日志仍默认关闭；LTX 基准固定 `RAY_DEDUP_LOGS=0`，避免 Ray 合并两个 rank 的相同诊断文本。

## 实装内容

- 新增严格的 all-to-all byte-range 规划，分别处理 rank、本地半块、对端半块和尾块。
- 单一持久 buffer 依次完成子块；每块在复用前等待 peer consumed，保持跨进程 CUDA event/stream 生命周期正确。
- 完整逻辑 geometry 在传输前握手；任一 size/operation/copy/timeout 错误 poison endpoint，不返回部分成功，也不静默回退 CPU/Gloo。
- profiler 分开记录 logical `calls` 与物理 `chunks`。
- worker 启动健康检查主动跨越默认 128 MiB 边界，避免只验证单块路径。
- 环境根启动入口改为转发到部署仓库脚本的薄包装，唯一默认值只存在于源码启动器。
- LTX 基准工具同时识别唯一工作树旁的 `ComfyUI` 与 `custom_nodes` 部署布局。

## 通信与显存结果

| 项目 | 结果 |
|---|---|
| 历史失败 payload | 239,826,944 字节/rank，128 MiB endpoint 自动 2 chunks，20 次正确 |
| 64 MiB endpoint live coverage | 64/128/256/512/768 MiB，FP32/FP16/BF16 bytes/uint8，最大误差 0 |
| 发布压力 | 512 MiB/rank、8 chunks/call、100 次；无 drift/timeout/orphan worker |
| 256 MiB payload | 128 MiB 自动分块中位 61.696 GiB/s，为匹配 buffer 的 98.91% |
| 512 MiB payload | 128 MiB 自动分块中位 61.774 GiB/s，为匹配 buffer 的 98.50% |
| 固定显存 | 相对 256 MiB 少 128 MiB/rank；相对 512 MiB 少 384 MiB/rank |

没有实现双 buffer/pipeline depth=2。单 buffer 已达到性能门槛，双 buffer 会翻倍持久 staging 显存并增加 event/control 协议复杂度，收益不足以进入默认路径。

## 真实工作流验收

所有任务均使用默认 128 MiB、严格 10 秒 timeout，未降低规格或关闭 Refine。

| 工作流 | 冷启动 | 热启动 | 关键结果 |
|---|---:|---:|---|
| MiniMax H3 I2V safe-FP16 Turbo8 | 520.55 s | 422.54 s | 1120×768/124 帧；双卡 PASS |
| MiniMax H3 REF2VA safe-FP16 Turbo4 | 362.21 s | 256.92 s | 1120×768/124 帧；双卡 PASS |
| MMH3Director 五段 + Refine | 1445.67 s | 1362.48 s | 10 次采样；1280×736/365 帧/24 FPS |
| LTX 2.3 五秒 FSDP | 517.13 s | 386.36 s | 1280×704/121 帧/25 FPS；14.9/45.9 与 14.9/47.8 s/it |

Director 四个输出和 LTX 两个输出均全部解码、全部帧唯一、0 黑屏区间、音频 NaN/Inf=0。每个最终 benchmark 退出后两卡回到 10 MiB，目标 worker 不存在。物理内存、commit 与 pagefile 没有出现与 collective payload 等量的 host staging 增长。

## 升级与兼容

旧工作流 JSON、`RayInitializer`、O6 safe-FP16、MMH3Director adapter、FSDP CPU offload、RayKill 和 128/256/512 参数保持兼容。O9 没有增加 NCCL、GGUF FSDP、多机、多于两卡、Text Encoder/VAE 多卡或训练能力。

从旧版本升级后，推荐删除启动命令中的 `-P2PCapacityMiB 256/512`，先使用默认值。若默认 128 MiB 失败，应先检查 TCC/P2P、tensor contract、rank geometry、endpoint poison 和显存峰值；不要把增大 buffer 当作容量修复。

完整数据与复现入口见 [O9 验收记录](../testing/windows-p2p/ADAPTIVE_CHUNKING_2026-08.zh-CN.md)。
