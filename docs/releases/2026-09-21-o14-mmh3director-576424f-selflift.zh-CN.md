# Raylight 1.16.0：MiniMaxH3Director 576424f 与 SelfLift

## 升级范围

本版把经过验证的 MiniMaxH3Director 坐标从 `19b6a73` 更新到官方 `576424f`。Director 和 ComfyUI 源码均未修改；Raylight 只扩展自己的 MODEL 代理、远程采样协议、测试和示例工作流。

上游这 13 个提交包含 SelfLift、Semantic Bridge、FaceRefine、Refine 前显存清理、外部组缓存修复、真实参考视频时长和独立 Refine seed/画幅修复。Raylight 新增对 `on_step_state`、`zero_noise`、`noise_override` 的阶段桥接，使 SelfLift 的低清前缀、高清尾段和后续 Refine 都能继续由同一对双 V100 FSDP worker 计算。

## 关键修复：SelfLift 音频尺度

首版 O14 兼容层能够生成正确画面，但音频是宽带杂音。根因不是 SOL、工作流参数、Safe-FP16、Audio VAE 或保存节点，而是权重无关的 Ray MODEL 代理没有复刻 `comfy.model_base.MiniMaxH3.process_latent_in/process_latent_out`。

MiniMax H3 在 AV 采样期间把第二条流（音频 latent）按 `shift_video / shift_audio` 搬到视频 flow schedule；默认值是 `12 / 3 = 4`。真实模型会在内部空间乘 4、回到存储空间时除 4。SelfLift 转场发生在主进程代理上，遗漏这组变换后，画面仍正常，音频却以错误尺度进入高清阶段，并在 Refine 中进一步放大。

1.16.0 在代理 facade 中补齐同样的 AV 语义：视频流保持不变，音频流精确乘/除 `audio_scale`，往返无损。该修复不增加模型权重、通信轮次或 staging 显存。

## 实机结果

固定工作流为 REF2VA、864×480、124 帧、24 fps、Turbo4、SOL、SelfLift `0.5`、Director Refine 2 步、双 V100 FSDP/P2P。

| 项目 | 修复前 | 修复后 |
|---|---:|---:|
| 端到端 | 277.71 s | 288.55 s |
| 双卡采样合计 | 133.13 s | 143.83 s |
| 两卡显存峰值 | 9223 / 9201 MiB | 9351 / 9329 MiB |
| 物理内存峰值 | 59883.91 MiB | 59355.73 MiB |
| Commit 峰值 | 104328.43 MiB | 105183.29 MiB |
| 音频 DC offset（L/R） | +0.016429 / +0.020130 | -0.000056 / -0.000043 |
| 音频 zero-crossing（L/R） | 0.122354 / 0.132005 | 0.047671 / 0.063260 |

修复后媒体为 864×480、24 fps、124 帧、5.1667 秒视频和 5.1670 秒音频；124 个视频帧哈希全部不同，音频 NaN/Inf 为 0。采样时间差包含独立冷启动波动，修复代码本身不增加 collective 或模型 forward。

严格升级前后基线（不启用 SelfLift）中，Director `52f8fb7 → 576424f` 的端到端从 314.05 秒降到 288.51 秒，采样从 165.75 秒降到 150.11 秒；这是同一 O13 工作流的匹配冷启动结果，不把差值全部归因于某一个上游提交。

## 控制实验

- 关闭 SOL 后杂音仍存在，排除 SOL。
- 无 Refine 时杂音减轻但 DC 偏置仍异常，证明首采已经受影响，Refine 只是放大问题。
- `fp16_h3_safe` 与默认精度均产生同类杂音；默认精度采样反而由 77.70 秒增至 156.66 秒，排除 Safe-FP16。
- SelfLift `scale=1.0` 退化为单阶段 Ray 采样后，DC offset 回到约 `-0.0003`、zero-crossing 回到 `0.027–0.030`，把缺陷严格锁定在低清到高清状态接力。
- 官方未修改 Director 的单 GPU SelfLift 控制音频正常，但端到端约 648 秒；它只用于归因，不作为推荐配置。

## 能力边界

- SelfLift、Director Refine、SOL、实时预览、连续段 remask 和旧工作流继续使用同一个显式 Ray MODEL 代理；普通 ComfyUI MODEL 不受影响。
- Semantic Bridge 与 FaceRefine 的接口契约已覆盖；本机没有对应可选权重，因此本版不声明它们完成质量/性能实机验收。
- 示例工作流位于 `example_workflows/o14/`。其中带 `Control`、`NoRefine`、`DefaultDType` 的文件是归因工具，不是日常推荐配置。
- Director 保持官方干净提交 `576424f`；回退 archive 与 bundle 只保存在本机 `.local/backups/director/`，不提交仓库。

