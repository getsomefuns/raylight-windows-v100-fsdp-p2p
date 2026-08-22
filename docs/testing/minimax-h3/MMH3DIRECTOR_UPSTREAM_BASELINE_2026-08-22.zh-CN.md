# MiniMax H3 Director 原版基线（2026-08-22）

## 目的

在实现 `Ray MODEL Adapter (MMH3Director)` 前，先确定未修改的 MiniMax H3 Director 在 Windows、单张 Tesla V100 16GB 上的性能、显存和媒体质量边界。所有生成均通过 ComfyUI 页面中的工作流排队；没有用 Python 脚本提交任务。

## 固定条件

- Windows，2× Tesla V100-SXM2-16GB，TCC；严格单卡轮次使用 `CUDA_VISIBLE_DEVICES=0`。
- ComfyUI `v0.31.0-15-g62b3c94b`，Python 3.10.11，PyTorch 2.7.0+cu126，xFormers 0.0.30。
- 启动参数：`--disable-cuda-malloc --lowvram`。普通显存模式的首次尝试在文本编码阶段 OOM，故不作为可运行基线。
- 16:9，864×480（约 0.4MP），24 FPS，5 秒输入对应 124 帧，seed 固定为 42。
- FL2VA：FP8 scaled 模型 + Turbo8 LoRA，8 步。
- REF2VA：FP8 scaled 模型 + Turbo4 LoRA，4 步，官方绿色机甲参考图。
- 原始日志、遥测、视频和五帧接触表仅保存在本机 `E:\ComfyUI-py310\logs\minimax-h3\director-baseline` 与 ComfyUI output；正文只保留精简数据。

## 已完成结果

| 工作流 | 输出规格（分辨率 / 实际时长 / 帧数 @ 帧率） | 状态 | 端到端 | 采样 | 平均步速 | GPU0 峰值 | GPU1 | RAM 峰值 | Commit 峰值 | 分页变化 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FL2VA Turbo8，普通显存 | 目标 864×480 / 5.167 秒 / 124 帧 @ 24 FPS；OOM，无输出 | 文本编码 OOM | 未进入采样 | — | — | 约 16064 MiB | 基础上下文 | 约 56053 MiB | 约 85720 MiB | 无新增 |
| FL2VA Turbo8，lowvram | 864×480 / 5.167 秒 / 124 帧 @ 24 FPS | 通过 | 15:45 | 11:11 / 8 步 | 83.92 s/it | 15534 MiB | 347 MiB 基础上下文 | 约 64065 MiB | 约 74405 MiB | 无新增 |
| REF2VA Turbo4，lowvram，严格单卡 | 864×480 / 5.167 秒 / 124 帧 @ 24 FPS | 通过 | 11:56 | 06:02 / 4 步 | 90.73 s/it | 16120 MiB | 10 MiB、无进程 | 约 66318 MiB | 约 75387 MiB | 无新增 |
| REF2VA Turbo4，2 秒，热启动 | 864×480 / 2.333 秒 / 56 帧 @ 24 FPS | 通过 | 06:49.71 | 02:16 / 4 步 | 34.16 s/it | 15640 MiB | 10 MiB、无进程 | 约 65286 MiB | 约 74426 MiB | 无新增 |
| REF2VA Turbo4，10 秒，热启动 | 864×480 / 10.125 秒 / 243 帧 @ 24 FPS | 通过 | 21:55 | 16:03 / 4 步 | 240.80 s/it | 15283 MiB | 10 MiB、无进程 | 约 71986 MiB | 约 81905 MiB | 约 +658 MiB |
| REF2VA Turbo4，12 秒，冷启动 | 864×480 / 12.250 秒 / 294 帧 @ 24 FPS | 通过 | 28:44 | 21:57 / 4 步 | 329.46 s/it | 13088 MiB | 10 MiB、无进程 | 约 68609 MiB | 约 79354 MiB | 无新增 |
| REF2VA Turbo4，720p 档 5 秒，冷启动 | 1280×704 / 5.167 秒 / 124 帧 @ 24 FPS | 通过 | 27:05 | 19:41 / 4 步 | 295.41 s/it | 12504 MiB | 10 MiB、无进程 | 约 69386 MiB | 约 79295 MiB | 无新增 |
| FL2VA Turbo8，720p 档 5 秒，冷启动 | 1280×704 / 5.167 秒 / 124 帧 @ 24 FPS | 通过（主体数量偏差） | 41:21 | 36:14 / 8 步 | 271.81 s/it | 15160 MiB | 10 MiB、无进程 | 约 68577 MiB | 约 79545 MiB | 约 +670 MiB |
| REF2VA Turbo4，480p，3 秒×5 段，冷启动 | 864×480 / 14.584 秒 / 350 帧 @ 24 FPS | 通过 | 54:48 | 29:27 / 20 步 | 88.35 s/it | 16230 MiB | 10 MiB、无进程 | 约 73752 MiB | 不可用（计数器无效） | 1409–1464 MiB，未增长 |
| FL2VA Turbo8，480p，3 秒×5 段，冷启动 | 864×480 / 14.584 秒 / 350 帧 @ 24 FPS | 通过（提示词服从度偏差） | 01:12:19 | 53:30 / 40 步 | 80.29 s/it | 16117 MiB | 10 MiB、无进程 | 约 73055 MiB | 约 84548 MiB（OS virtual used） | 峰值 1685 MiB，净增约 290 MiB |

### 阶段观察

- FL2VA 冷加载约 3 分 38 秒；稳定采样每步约 83.76–84.97 秒；采样后 Video/Audio VAE 与保存约 56 秒。
- REF2VA 冷加载和参考图预处理约 5 分 05 秒；稳定采样每步约 90.60–91.73 秒；采样后 Video/Audio VAE 与保存约 49 秒。
- REF2VA 的参考条件使采样显存比 FL2VA 高约 586 MiB，步速慢约 8.1%。
- 2 秒热启动使用 56 帧，四步稳定在 33.97–34.20 s/it；固定的模型切换、VAE 和保存开销约 4 分 34 秒。
- 10 秒热启动使用 243 帧；四步为 239.52、240.13、240.77、241.09 秒。扩大的 latent 迫使扩散模型 `0 MiB loaded / 21592.91 MiB offloaded`，采样明显受 CPU 卸载限制，并产生约 658 MiB 新分页使用。
- 12 秒冷启动使用 294 帧；四步总计 21:57，单步约 328.4–330.6 秒，未出现逐步变慢。扩散模型仍为 `0 MiB loaded / 21592.91 MiB offloaded`；端到端除采样外的加载、条件编码、VAE 和保存共约 6:47。
- 720p 档 REF2VA 使用 H3 对齐的 1280×704、124 帧；四步总计 19:41，单步约 294.0–296.0 秒。扩散模型同样全量卸载；相对 864×480 的 90.73 s/it 慢约 3.26 倍，超过 2.17 倍像素面积增幅。
- 720p 档 FL2VA 使用 1280×704、124 帧；八步总计 36:14，单步稳定在约 270.7–274.3 秒。采样显存峰值达到 15160 MiB，已经非常接近单卡容量；相对 480p FL2VA 的 83.92 s/it 慢约 3.24 倍。
- REF2VA 3 秒×5 段的各段采样为 47.15、98.71、98.86、98.83、98.92 s/it。加入 22 帧视频潜空间和 40 个音频步的连续性上下文后，第 2 段相对首段慢约 2.09 倍；第 2–5 段没有继续逐段恶化。全程 GPU0 平均利用率约 57.65%，GPU1 始终 0%。
- FL2VA 3 秒×5 段的各段采样为 40.54、90.14、90.26、90.27、90.23 s/it。连续段相对首段慢约 2.22 倍，但第 2–5 段同样稳定。全程 GPU0 平均利用率约 76.05%，GPU1 始终 0%。
- 两个五段工作流的逻辑输入均为 365 帧；Director 为相位对齐在中间三次各裁 5 帧，最终合并为 350 帧。多段模式没有 OOM，但端到端分别达到 54:48 和 01:12:19，已经明确暴露原版单卡连续生成的性能成本。

## 媒体验收

- 两个输出均为 864×480、24 FPS、124 帧、5.167 秒，H.264 视频和 32kHz 双声道 AAC 音频。
- FL2VA：机器人、绿色装甲、雨夜霓虹街道与连续向前运动正常，无黑屏、噪声或冻结。
- REF2VA：保持参考图中的绿色机甲、紫色传送门和武器外观，镜头连续运动，无黑屏、噪声或冻结。
- 两个音轨均非静音；FL2VA 平均/峰值约 -29.3/-10.9 dB，REF2VA 约 -19.4/-2.2 dB。
- REF2VA 10 秒输出为 864×480、24 FPS、243 帧、10.125 秒；五点抽帧显示主体和场景连续，无黑屏、噪声或结构崩坏，并带 32kHz 双声道 AAC 音轨。
- REF2VA 12 秒输出为 864×480、24 FPS、294 帧、12.25 秒；六点抽帧连续，无黑屏、噪声或结构崩坏；32kHz 双声道 AAC 平均/峰值约 -14.0/-1.2 dB。
- REF2VA 720p 档输出为 1280×704、24 FPS、124 帧、5.167 秒；五点抽帧保持机甲、传送门和城市运动连续，无黑屏、噪声或结构崩坏；AAC 平均/峰值约 -16.4/-2.3 dB。
- FL2VA 720p 档输出为 1280×704、24 FPS、124 帧、5.167 秒；五点抽帧无黑屏、噪声或结构崩坏，机器人和城市运动连续，AAC 平均/峰值约 -18.2/-4.8 dB。但提示词要求三台机器人，实际稳定生成两台，故媒体通过但提示词主体数量服从度不完全通过。
- REF2VA 3 秒×5 段输出为 864×480、24 FPS、350 帧、14.584 秒。按实际接缝帧 73、141、209、277 的前后帧检查，主体、机位、背景和运动方向连续，无黑屏、噪声或结构崩坏；AAC 平均/峰值约 -14.9/-0.6 dB。
- FL2VA 3 秒×5 段输出为 864×480、24 FPS、350 帧、14.584 秒。四处接缝检查连续，无黑屏、噪声或结构崩坏；AAC 平均/峰值约 -14.0/-4.0 dB。作为 t2v 路径，机器人数量和动作存在轻微漂移，提示词/主体一致性弱于带参考图的 REF2VA。

## 本机证据索引

| 运行 | 工作流 | 输出视频 | 遥测 | 接缝表 | 主日志 |
|---|---|---|---|---|---|
| REF2VA Turbo4，3 秒×5 | `ComfyUI/user/default/workflows/MMH3DirctorTest/MMH3Director_REF2VA_Turbo4_480p_3s_x5_Continuity.json` | `ComfyUI/output/video/MMH3DirctorTest/multisegment/REF2VA_Turbo4_480p_3s_x5_00001_.mp4` | `logs/minimax-h3/director-baseline/REF2VA_Turbo4_480p_3s_x5_telemetry.csv` | `logs/minimax-h3/director-baseline/REF2VA_Turbo4_480p_3s_x5_seams.jpg` | `logs/minimax-h3/director-baseline/comfyui-single-gpu-ref4-3sx5.stderr.log` |
| FL2VA Turbo8，3 秒×5 | `ComfyUI/user/default/workflows/MMH3DirctorTest/MMH3Director_FL2VA_Turbo8_480p_3s_x5_Continuity.json` | `ComfyUI/output/video/MMH3DirctorTest/multisegment/FL2VA_Turbo8_480p_3s_x5_00001_.mp4` | `logs/minimax-h3/director-baseline/FL2VA_Turbo8_480p_3s_x5_telemetry.csv` | `logs/minimax-h3/director-baseline/FL2VA_Turbo8_480p_3s_x5_seams.jpg` | `logs/minimax-h3/director-baseline/comfyui-single-gpu-fl8-3sx5.stderr.log` |

两份 5 秒×5 段工作流 JSON 继续保留，方便未来按需复测，但没有对应运行结果、视频或遥测，不能作为已验证数据引用。

## 测试阶段结论

- 0.4MP、24 FPS 的 2、10、12 秒已通过；12 秒已达到 329.46 s/it，15 秒即使不 OOM 也没有实际使用意义，因此按用户决定停止。
- 方向 1 的 5 秒、24 FPS、720p 档 REF2VA Turbo4 与 FL2VA Turbo8 已完成；H3 对齐后的实际尺寸为 1280×704。
- 方向 2/3 已完成 REF2VA Turbo4 与 FL2VA Turbo8 的 864×480、3 秒×5 段压力与接缝测试。两组都能完成且接缝连续，但连续段单步成本约为首段的 2.09–2.22 倍。
- 原计划的两组 5 秒×5 段未执行。用户基于 3 秒×5 段的 54:48/01:12:19 实测确认已获得所需结论，继续测试耗时过高且缺乏实际意义，因此结束原版基线测试阶段；这两组状态是“取消”，不是“通过”。
- 后续 Raylight 适配版必须与同 seed、同参数、同模型的这些原版结果比较。
