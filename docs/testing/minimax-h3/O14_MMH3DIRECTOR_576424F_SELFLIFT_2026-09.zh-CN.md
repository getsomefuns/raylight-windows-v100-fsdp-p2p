# O14 MiniMaxH3Director 576424f / SelfLift 验收记录

## 1. 验证目标与环境

目标是在不修改 ComfyUI 和 Director 源码的条件下，升级到 Director `576424f`，让 SelfLift 的两阶段采样、后续 Refine、实时预览与既有 Raylight FSDP/P2P 路径共存，并保留可回退、可复现的证据。

- Windows，2× Tesla V100-SXM2-16GB，TCC，NVLink/CUDA P2P。
- Python 3.12.10，PyTorch 2.12.0+cu126，Ray 2.57。
- Raylight 1.16.0；Director 官方干净提交 `576424f`。
- 固定 864×480、124 帧、24 fps、REF2VA、Turbo4、SOL、SelfLift 0.5、Refine 2。
- `P2PCapacityMiB=256`，`ReserveVramGiB=4`；同一模型、seed、输入和输出规格。

原始 JSON、CSV、日志和提示 API 保存在本地 `.local/artifacts/o14/`，正文只保留结论数据，不提交输入素材、视频或长日志。

## 2. 上游升级前后基线

| Director | 端到端 | 采样 | Ray 初始化 | 模型节点 | Director 节点 | 保存 | GPU 峰值 | RAM 峰值 | Commit 峰值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `52f8fb7` | 314.05 s | 165.75 s | 35.47 s | 15.37 s | 255.45 s | 5.76 s | 9223 / 9201 MiB | 62555 MiB | 106073 MiB |
| `576424f` | 288.51 s | 150.11 s | 32.17 s | 12.99 s | 236.42 s | 5.56 s | 9223 / 9201 MiB | 59703 MiB | 104734 MiB |

相同 O13 工作流升级后端到端缩短 8.13%，采样缩短 9.44%，没有新增显存峰值或媒体异常。该表只说明匹配冷启动结果，不把收益归因于单一上游功能。

## 3. SelfLift 功能结果与音频缺陷

最初的 Ray SelfLift + SOL + Refine 工作流完成三次双 rank 采样，端到端 277.71 秒、采样 133.13 秒、GPU 峰值 9223/9201 MiB。画面可播放，但音频 DC offset 达到 `+0.016429/+0.020130`，zero-crossing 达到 `0.122354/0.132005`，听感为连续杂音。

控制矩阵：

| 控制 | 采样 | GPU 峰值 | 音频结果 | 判定 |
|---|---:|---:|---|---|
| SelfLift + Refine + SOL，修复前 | 133.13 s | 9223 / 9201 MiB | 严重杂音 | 缺陷复现 |
| SelfLift + Refine，无 SOL | 144.44 s | 同量级 | 仍为杂音 | 排除 SOL |
| SelfLift，无 Refine，Safe-FP16 | 77.70 s | 9351 / 9329 MiB | 偏置异常 | 首采已损坏 |
| SelfLift，无 Refine，默认精度 | 156.66 s | 10112 / 10070 MiB | 同类异常 | 排除 Safe-FP16 |
| SelfLift scale=1，无 Refine | 90.05 s | 9351 / 9329 MiB | 正常 | 锁定两阶段状态接力 |
| 官方单 GPU SelfLift + Refine | 约 514 s 采样 / 648 s 端到端 | GPU0 约 15.8–16.18 GiB | 正常 | 排除 Director 算法本身 |

根因是 `RayMMH3DirectorModelProxy` 的权重无关 facade 缺少真实 MiniMaxH3 的 AV latent 音频尺度变换。视频/音频的 flow shift 默认为 12/3，音频流必须在内部空间乘 4、存储空间除 4。遗漏该操作只破坏音频，符合“画面正常、声音全是杂音”的症状。

## 4. 修复后完整验收

修复提交 `0405b76` 部署后，重跑完整 SelfLift + SOL + Refine：

- 端到端 288.55 秒；三次远程采样 41.21 + 38.34 + 64.28 = 143.83 秒。
- Ray 初始化 31.52 秒；模型节点 12.91 秒；Director 节点 237.22 秒；保存 5.64 秒。
- GPU0/GPU1 峰值 9351/9329 MiB，利用率峰值 100%/99%。
- 物理内存峰值 59355.73 MiB，Commit 峰值 105183.29 MiB，pagefile used 未增长。
- 音频 DC offset `-0.000056/-0.000043`，RMS `-23.22/-29.78 dB`，zero-crossing `0.047671/0.063260`，NaN/Inf 为 0。
- 输出 864×480、24 fps、124 帧；视频/音频时长 5.1667/5.1670 秒；124/124 帧哈希唯一。

验收结论：通过。SelfLift 低清、高清和 Refine 都由同一对双 V100 worker 完成，修复恢复音频质量且没有引入额外权重、collective 或 staging buffer。

## 5. 自动化与边界

- O14 定向代理/桥/真实 Director 契约/工作流/基准测试：59 passed（修复开发阶段）。
- 最终全量回归：`753 passed, 217 warnings, 35 subtests passed`；警告为既有 Torch 弃用、NestedTensor 原型提示和 Windows pytest 临时目录清理提示，没有测试失败。
- O14 改动文件定向 Ruff 与 `git diff --check` 通过。全仓库 Ruff 仍报告 29 个 O14 之前已存在、位于 WAN/LTX/SeedVR 等未改文件的 lint 债务；本阶段没有为追求零告警而修改无关模型路径。
- Semantic Bridge 和 FaceRefine 完成参数/采样契约覆盖；缺少本机可选模型资产，未声明画质与速度实机验收。
- Director 源码保持官方干净状态；Raylight 部署必须来自唯一源码工作树的已提交 HEAD。
