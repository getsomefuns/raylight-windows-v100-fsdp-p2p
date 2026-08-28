# MiniMax H3 Director MODEL 适配器：O8 测试与验收记录

[简体中文](MMH3DIRECTOR_ADAPTER_2026-08.zh-CN.md) | [English](MMH3DIRECTOR_ADAPTER_2026-08.md)

## 1. 验证目标

在不修改 MiniMaxH3Director、既有 Raylight 节点 schema 和旧工作流 JSON 的前提下，为 Director 提供显式可选的 Ray `MODEL` 代理，使 Director 发起的扩散采样只在两个 Ray Worker 上执行一次，并继续使用已验收的 Windows FSDP2 + CUDA P2P/NVLink 数据面。

本文从开发前基线开始持续维护。原版 Director 的单卡边界、单段与多段结果见 [原版基线报告](MMH3DIRECTOR_UPSTREAM_BASELINE_2026-08-22.zh-CN.md)。

## 2. O8 开发前基线（2026-08-23）

### 2.1 软件与硬件门槛

| 项目 | 结果 |
|---|---|
| 基线提交 | `f5c455d`（适配器代码尚未开始） |
| GPU | 2× Tesla V100-SXM2-16GB，TCC，驱动 577.00 |
| Python / PyTorch | 3.10.11 / 2.7.0+cu126 |
| 关键依赖 | torchvision 0.22.0+cu126；torchaudio 2.7.0+cu126；xformers 0.0.30；Ray 2.57.0；xFuser 0.4.5；yunchang 0.6.4 |
| 全量测试 | `342 passed, 1 warning, 30 subtests passed`，68.19 秒 |
| P2P 正确性 | 同步/异步、分块尾部、FP32/FP16/BF16/UINT8 全部通过 |
| P2P 带宽 | 主要结果约 59.2–61.6 GiB/s/方向，高于 50 GiB/s 门槛 |

基线首次运行暴露了模型下载脚本对 `Get-FileHash` 的环境依赖。测试结果为 `341 passed, 1 failed`；将哈希实现改为直接使用 .NET SHA-256 后，聚焦测试和全量测试均通过。该修复不改变 Raylight 运行时、节点接口或工作流。

### 2.2 旧公共节点 schema 快照

| 节点 ID / 显示名 | Required 输入 | Optional 输入 | 输出 | 函数 |
|---|---|---|---|---|
| `RayInitializer` / Ray Init Actor | ray_cluster_address, ray_cluster_namespace, GPU, ulysses_degree, ring_degree, cfg_degree, dp_degree, sync_ulysses, clear_vram_after_sampling, FSDP, FSDP_CPU_OFFLOAD, XFuser_attention, skip_comm_test, use_mmap | wait_for | `RAY_ACTORS_INIT` (`ray_actors_init`) | `spawn_actor` |
| `RayInitializerAdvanced` / Ray Init Actor (Advanced) | ray_cluster_address, ray_cluster_namespace, GPU, GPU_SELECT, ulysses_degree, ring_degree, clear_vram_after_sampling, cfg_degree, dp_degree, sync_ulysses, FSDP, FSDP_CPU_OFFLOAD, XFuser_attention, skip_comm_test, use_mmap | wait_for, ray_object_store_gb, ray_dashboard_address, torch_dist_address | `RAY_ACTORS_INIT` (`ray_actors_init`) | `spawn_actor` |
| `RayUNETLoader` / Load Diffusion Model (Ray) | unet_name, weight_dtype, ray_actors_init | lora | `RAY_ACTORS` (`ray_actors`) | `load_ray_unet` |
| `RayKill` / Kill Ray | ray_actors, kill_mode | — | 无 | `kill_ray` |
| `XFuserSamplerCustomAdvanced` / XFuser SamplerCustom Advanced | add_noise, noise_seed, guider, sampler, sigmas, latent_image | — | `LATENT`, `LATENT`, `RAY_ACTORS` | `ray_sample` |
| `XFuserSamplerCustom` / XFuser SamplerCustom | ray_actors, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image | — | `LATENT`, `RAY_ACTORS` | `ray_sample` |

所有节点类别保持原值：前四项为 `Raylight`，两种采样器为 `Raylight/extra/custom_sampling/samplers`。任务 1 将这些内容固化为自动化契约测试。

### 2.3 已验收工作流 JSON 哈希

| 工作流 | SHA256 |
|---|---|
| `LTX2_3_i2v_Raylight_Windows_FSDP_5s.json` | `2b0b6e3bab1a7adb7f5e3a8b627b0bfe5535b3d8b3e949ecb7644225f71eef84` |
| `Minimax_H3_I2V_Windows_V100_FSDP_Turbo8.json` | `2f261d43e0477e21165793edfade592e4d4eef948b0434c81db8180a8fc926a5` |
| `Minimax_H3_I2V_Windows_V100_FSDP_Turbo8_FP16_Experimental.json` | `faa485b80b87126bb45223f06c935f8372611f0d98efcd4106ee7111d3e41b02` |
| `Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4.json` | `af038101e582efd163d93f1fd22be7abe5d0c3a93b00b90c332f52d9ff2377ee` |
| `Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4_FP16_Experimental.json` | `fccd1b7ed6519cc81fa66f26a956061587f8cebce567f3ddef07ba0752a32f35` |

## 3. 已实现结构

- 新增显式节点 `Ray MODEL Adapter (MMH3Director)`：输入一个已加载的 `RAY_ACTORS`，输出不持有权重的 `MODEL` 代理。
- 只有该代理进入 `comfy.sample.sample()` 时才转交两个 Ray Worker；普通 ComfyUI `MODEL` 继续原样调用原采样函数。
- Director 准备的 conditioning、latent、noise、sigmas、seed、steps、sampler、scheduler 和 shift 原样送到两个 rank；本地主进程不进行第二次扩散采样。
- 两个 rank 的 operation ID、rank、shape、dtype、finite 值及最终张量完全相等才返回 Director；失败时禁止回退到单卡本地采样。
- 保留原 Director 节点作为唯一生成参数面板；示例图只有 `RayInitializer → RayLoraLoader → RayUNETLoader → Ray MODEL Adapter → MiniMaxH3Director` 一条模型路径，没有 XFuser sampler/guider/scheduler 重复栈。
- 当前适配目标锁定 MiniMaxH3Director `1.0.0`、提交 `d655912`；不匹配的采样接口会给出明确错误。

示例工作流：`example_workflows/MMH3Director_Windows_V100_FSDP.json`。本机部署副本：`ComfyUI/user/default/workflows/MMH3Director.json`。

## 4. 首轮实机 t2v 结果（2026-08-23）

所有结果均使用两张 V100、TCC、FSDP CPU offload、Ulysses 2、CUDA P2P、256 MiB buffer、`fp16_h3_safe`、FL2VA FP8 checkpoint 与官方 Turbo8 LoRA。日志诊断开关保持关闭；P2P 启动健康检查约 58.88 GiB/s/方向。

| 运行 | 几何 / 时长 | 状态 | 采样 | 端到端 | GPU 显存峰值 MiB | GPU 利用率峰值 | 物理 / 虚拟 / pagefile 峰值 MiB |
|---|---|---|---:|---:|---:|---:|---:|
| 缩小冷启动，交接修复后 | 864×480 / 52 帧 / 24 FPS / 2.167 秒 | 通过 | 约 1:16；8.85 s/it | 259.77 秒 | 全流程 13738 / 5027；采样约 5186 / 5027 | 99% / 96% | 90575.9 / 132370.9 / 1467 |
| 缩小热运行，seed 43 | 同上 | 通过；复用 PID 20012 / 22160 | 约 1:13；8.83 s/it | 131.67 秒 | 全流程 15302 / 5027；采样约 5190 / 5027 | 99% / 94% | 78081.3 / 96592.7 / 1467 |
| 720p 档冷启动 | 1280×704 / 124 帧 / 24 FPS / 5.167 秒 | 通过 | 约 5:51；42.2 s/it | 595.10 秒 | 全流程 15422 / 14664；采样约 14822 / 14664 | 100% / 100% | 97763.8 / 138466.8 / 1467 |
| 720p 档热运行 | 同上 | 通过 | 约 5:49；42.0 s/it | 470.82 秒 | 14826 / 14664 | 100% / 100% | 82795.5 / 101251.1 / 1467 |

原版 Director 单卡同规格 720p 档冷启动为 41:21、271.81 s/it。当前双卡冷启动端到端约快 4.17 倍、单步约快 6.44 倍；热运行端到端约快 5.27 倍、单步约快 6.47 倍。冷/热条件不同，因此热运行加速比只代表交互复用收益，不伪装成严格冷启动对照。

## 5. 预处理显存交接缺陷与修复

首个 864×480 冷启动虽然成功，但文本编码器与两个 Ray rank 在 GPU0 上重叠，采样全流程峰值达到 16198 / 4721 MiB。根因不是 FSDP 分片失衡，而是 Director 必须先在主进程完成文本编码，原桥接入口没有在远程采样前卸载这些预处理模型。

`40a71db` 在代理采样开始前调用 ComfyUI 标准的 `unload_all_models()` 与 `soft_empty_cache()`，并在 CUDA 可用时同步设备。修复后的 864×480 采样显存约为 5186 / 5027 MiB；1280×704 采样约为 14822 / 14664 MiB。显存交接换来主存中保留可重载的 CLIP/VAE 权重，因此冷启动物理/虚拟峰值会上升；本机仍通过，没有发生 pagefile 激增或 OOM。该权衡必须保留在后续更长/更大模式的资源门槛中。

## 6. 输出与复用检查

- 冷、热 720p 输出均为 H.264 1280×704、124 帧、24 FPS、5.167 秒，并含 32 kHz 双声道 AAC。
- 接触表检查显示连续运动、无黑屏、彩色噪声、乱码或冻结段；提示词要求三名机器人，但画面主要维持两名，属于与原版单卡基线相同类别的主体数量服从度偏差，不是分布式数值损坏。
- 相同 seed 的冷/热 MP4 容器 SHA256 不同，但视频码流 MD5 `3c38daa0228f118cb5dbd951114d7e27`、音频码流 MD5 `b4cb8900c83a5d70b69c07fcb004d02b` 和最后一帧解码 MD5 `84d8b99e1cf05a6b9abfdfda93f55f34` 完全一致。
- 同一会话改 seed 热运行复用 worker PID 20012 / 22160，只产生一个新的远程 operation，没有第二次 Gloo 初始化或 FSDP 注册。
- 原始 prompt、history、遥测 CSV、日志和接触表保留在本机 `<仓库根目录>/.local/artifacts/o8/mmh3director_adapter`；视频保留在 `ComfyUI/output/video/O8_MMH3Director`，不提交 Git。

## 7. Director 模式、Refine 与多段验收

| 模式 | 规格 | 端到端 | 采样 | GPU 显存峰值 MiB | 结论 |
|---|---|---:|---:|---:|---|
| i2v | 864×480 / 39 帧 / 24 FPS | 135.77 秒 | 约 8.50 s/it | 15692 / 4861 | 通过 |
| fl2v | 864×480 / 39 帧 / 24 FPS | 140.35 秒 | 约 8.50 s/it | 15833 / 4923 | 通过 |
| refine | 同规格，主 8 步 + refine 2 步 | 170.28 秒 | 主 8.13、refine 4.26 s/it | 15541 / 4861 | 通过；同模型复用 |
| r2v | 864×480 / 39 帧 / 24 FPS | 176.70 秒 | 7.40 s/it（4 步） | 15538 / 4781 | 通过；按需切换 REF2VA checkpoint |
| v2v | 864×480 / 39 帧 / 24 FPS | 137.76 秒 | 远程单次采样 | 15821 / 5681 | 通过；模式不匹配会在 GPU 前失败 |
| rv2v | 864×480 / 39 帧 / 24 FPS | 133.39 秒 | 远程单次采样 | 16039 / 6867 | 通过 |
| r2v 两段 | 864×480 / 78 帧 / 24 FPS / 3.25 秒 | 196.90 秒 | 两次远程 operation | 15543 / 4787 | 通过；Worker/FSDP/Gloo 均复用 |

每个 Director pass 只出现一个唯一远程 operation，本地主进程扩散次数为零。真实两段 R2V 只出现两个远程 operation，没有额外模型加载、Gloo 初始化或 FSDP 注册。外接 R2V group 却选择 `v2v` 的负向测试在进入 GPU 前明确拒绝，证明模式契约不会被静默忽略。

## 8. 旧工作流性能回归

旧路径均在同条件下执行一冷三热；发布门槛以三次热运行中位数为准。没有发现无法解释的采样退化超过 5%。

| 工作流 | 当前热中位数 | 历史对照 | 变化 | 判定 |
|---|---:|---:|---:|---|
| MiniMax I2V Turbo8，448×448 / 39 帧 | 端到端 109.44 秒；采样 94.22 秒 | 107.35 / 96.15 秒 | 端到端 +1.95%；采样约 -2.0% | 通过 |
| MiniMax REF2VA Turbo4，608×352 / 56 帧 | 端到端 82.96 秒；采样 64.58 秒 | 82.07 / 67.14 秒 | 端到端 +1.08%；采样约 -3.8% | 通过 |
| LTX 2.3 两阶段，1280×704 / 121 帧 | 端到端 392.73 秒；两段 15.0 / 47.8 s/it | 15.0 / 45.6 s/it | 第二段约 +4.82% | 通过，接近但未越过门槛 |
| O6 REF2VA safe-FP16，1120×768 / 124 帧 | 49.48 s/it | 54.22 s/it | 约 -8.7% | 通过 |
| O6 REF2VA FP32，同规格 | 177.89 s/it | 185.20 s/it | 约 -3.95% | 通过 |

REF2VA 首轮热运行出现过约 99 秒的端到端离群值，但远程提交到 Worker 入口只需 0.2–0.4 秒。新增分阶段诊断后确认，波动来自主进程 VAE/预处理模型卸载回主存（约 7.3–8.7 秒，异常轮更长），不是 Ray、Gloo、CUDA P2P 或两个 rank 失步。匹配条件复测热中位数 82.96 秒并通过门槛。诊断字段默认不打印，仅在性能诊断开启时使用。

## 9. 资源、媒体与回收

- 两张 GPU 在采样阶段均达到高利用率，所有接收张量 finite，两个 rank 输出精确一致。
- 代表性 Director、MiniMax、O6 与 LTX 视频均通过 ffprobe 解码；黑帧区间为 0，音频 NaN/Inf 为 0，无解码错误。
- 同 checkpoint 热运行复用原 Worker 和 FSDP shard；切换 FL2VA/REF2VA checkpoint 时两个 Worker 成对重建，避免跨模型污染。
- 最终 `RayKill` 后两个 Worker、raylet 与 GCS 均退出；GPU 任务清空，物理/虚拟提交量从约 76835/95561 MiB 回落到约 34196/44655 MiB。Ray 初始化时每卡约 700 MiB 的 CUDA context 也随 Worker 退出释放。
- 原始日志、遥测和视频保留在本机证据目录，不进入仓库。仓库只保留可维护结论、参数和证据路径。

## 10. 最终结论

Task 8 完成。O8 已验证 t2v、i2v、fl2v、r2v、v2v、rv2v、同模型 Refine 和真实多段工作流；每个 pass 只在两个 Ray Worker 上执行一次 FSDP + CUDA P2P 扩散，本地无重复采样。媒体、复用、checkpoint 切换、完整回收和五条旧工作流性能门槛均通过。该结论不表示 CLIP、Video/Audio VAE 或封装保存被 FSDP 分片；这些仍由 Director/ComfyUI 主进程执行。

## 11. Task 10 最终发布质检

| 检查 | 结果 |
|---|---|
| 完整 pytest | `418 passed, 1 warning, 30 subtests passed`，85.21 秒 |
| O8 格式化后聚焦测试 | `41 passed`，14.24 秒 |
| Ruff | 从回退标签以来的 O8 Python 文件 lint 全部通过；6 个 O8 新文件经 formatter 整理后通过 |
| Director 已安装启动 | 8188 正常启动；适配节点与 Director 节点均可发现，显示名正确 |
| Director 被屏蔽启动 | 仅白名单加载 Raylight 时 8188 正常启动；适配节点存在，Director 节点不存在，Raylight 无导入依赖 |
| 同一 ComfyUI 会话 | 先旧 I2V Turbo8 smoke 260.51 秒成功，再 Director T2V smoke 310.52 秒成功；未重启或手工清缓存 |
| 同会话远程采样 | Director 只有一个 operation；8 步远程采样 8.87 s/it；无第二次本地扩散 |
| checkpoint 隔离 | 旧路径 FP32 计算 Worker 与 Director safe-FP16 Worker 成对切换，无旧模型状态污染 |
| 最终回收 | RayKill 后 Worker/CUDA context 清空；关闭测试服务后 GPU、Ray 与 ComfyUI 进程均为零 |
| Git 与仓库 | `git diff --check` 通过；工作树只包含待提交的最终格式/文档；Git pack 约 8.06 MiB，无运行视频或日志进入仓库 |

Ruff 全仓扫描仍报告 32 个 O8 之前就存在的上游/历史未使用导入或变量问题；本阶段没有借机大范围格式化或修改这些无关文件。仓库的 pre-commit Python runner 未安装，但其两个 hook 已直接执行：O8 lint 通过，O8 新文件 formatter 通过。Git `fsck` 只发现正常的 dangling 对象（历史提交/暂存产生），没有对象损坏。

## 12. 双参考图 0.9MP 显存回归与修复（2026-08-24）

复现工作流为本机 `MMH3Director.json` 的第 5 段：R2V、1280×736、72 输出帧、24 FPS、
两张参考图、Turbo4，以及同模型 2 步 Refine。完整五段任务在第 5 段开始时 OOM；为排除
多段累积，又在全新 ComfyUI/Ray 进程中只运行第 5 段。

| 条件 | 结果 | 关键证据 |
|---|---|---|
| 冷进程，仅第 5 段，`--reserve-vram 2` | 失败 | 123.39 秒后在 Qwen3-VL MLP 的 NVFP4 解量化阶段申请 500 MiB 失败；GPU0 只余约 463 MiB；尚未进入 Ray 采样或 Refine |
| 冷进程，仅第 5 段，`--reserve-vram 3` | 通过 | 435.27 秒端到端；双卡采样约 13.26/13.00 GiB、100%/100%；任务状态 success |
| 同一会话复用 Worker，完整 5 段，`--reserve-vram 3` | 通过 | 1475.655 秒端到端；5×73=365 帧、1280×736、24 FPS；第 5 段双参考图正常越过 Qwen 编码并完成主采样与 Refine |

失败栈位于主 ComfyUI 进程的 `clip.encode_from_tokens_scheduled()`，不是两个 rank 失步，
也不是上一段解码、封装、Refine 或下一段并行执行。第二张参考图增加 Qwen 多模态 token 与
激活峰值；旧 2 GiB 预留让过多 Text Encoder 权重同时驻留 GPU0，挤掉了量化权重临时
解量化所需的连续工作区。修复只把公共启动脚本默认值提高到 3 GiB，并保留参数覆盖；
没有删除参考图、降低参考分辨率、改变 seed/步骤或修改生成数值路径。

成功输出为 H.264/AAC、1280×736、72 帧、24 FPS、3.042 秒。首/中/末帧人工检查为连贯
复古片头画面，最终 `LATE CHECKOUT` 标题正常，无黑屏、彩色噪声或冻结。该聚焦测试没有
同步采集物理内存、提交内存和 pagefile 峰值，因此不对这三项补造数据。

随后未重启 ComfyUI、未重建 Worker，直接执行用户修改后的完整五段工作流。五段按“主采样 →
Refine → GPU0 VAE/音频解码与缓存 → 下一段”串行推进，没有观察到上一段解码与下一段采样
重叠。第 5 段 Qwen 双参考图预处理时 GPU0 约 15.50 GiB、余量约 739 MiB；进入远程采样后
两卡约 13.26/13.00 GiB，利用率 100%/100%。任务成功结束后两卡回落到约 1.27/1.01 GiB。
完成时系统可用物理内存约 45.2 GiB、已提交内存约 100.6 GiB；最终合并阶段 pagefile 使用率
从中途观察到的约 2.03% 上升到约 2.84%，但没有形成逐段单调增长或触发 OOM。完整任务在
Director 历史结果中输出 365 帧合并图像与音频；本次最小 API 验证图未连接最终视频保存节点，
因此不把它记作完整 MP4 文件产物。
