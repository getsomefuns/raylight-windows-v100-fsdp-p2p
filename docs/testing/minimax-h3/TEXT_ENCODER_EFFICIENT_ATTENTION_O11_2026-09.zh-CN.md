# O11 MiniMax H3 Text Encoder 高效注意力测试汇总

日期：2026-09-01  
版本目标：1.13.0  
唯一工作树：`E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`

## 1. 测试验证目标

O11 验证原生 Windows 双 V100 TP2 Text Encoder 在长多模态输入下不再因 PyTorch SDPA 数学回退和平方注意力矩阵而 OOM，并验证：

- `AUTO`、`XFORMERS_GQA`、`TORCH_EFFICIENT` 和受限 `MATH_DEBUG` 路由正确；
- 15 秒视频、7 张不同图片、7000 个有效字符的冻结输入可完整返回有限 conditioning；
- 用户真实 Director 输入（15 秒、7 图、1 视频、1 音频、5802 字符）有额外余量；
- Text Encoder 权重在完成后释放，连续执行不持续增长；
- 旧工作流、短视频采样和既有 O6/O8/O9/O10 功能不回归。

O11 只验收 Text Encoder 最大上下文。最大输入后的 Diffusion 采样属于独立显存边界。

## 2. 测试条件与变量

| 项目 | 固定条件 |
|---|---|
| 系统/GPU | 原生 Windows，2× Tesla V100-SXM2-16GB，TCC，NVLink/P2P |
| 驱动 | 577.00 |
| Python / Torch | 3.10.11 / 2.7.0+cu126 |
| xFormers / ComfyUI | 0.0.30 / 0.31 |
| Text Encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| TP | 两卡 Linear TP2，CUDA P2P SUM |
| Text Encoder Attention | AUTO、强制 XFORMERS_GQA、强制 TORCH_EFFICIENT |
| XFuser Attention | 独立保持 TORCH_EFFICIENT，只影响 Diffusion |

冻结容量工作流为 `example_workflows/MMH3_TextEncoder_O11_MaxContext.json`：7 个不同图片散列、15.0 秒/360 帧视频、7000 字符提示词，1056×608，conditioning-only。公开工作流不包含本机绝对路径或素材本体。

用户真实 Director 工作流部署为 `ComfyUI/user/default/workflows/MMH3Director.json`：Picture 1 为红模参考，Picture 2–7 为白模参考 1–6，另含 15 秒动作视频和独立 WAV；REF2VA Turbo4、365 帧、24 fps、1056×608、单次采样、无 Refine。O11 视频前缀统一为 `video/raylight_o11`。

## 3. 测试关注信息

- 实际注意力后端及是否回退 Math；
- expanded sequence、文字/视觉位置数量；
- 视觉、语言权重加载、语言计算、清理和 finalize 耗时；
- PyTorch Text Encoder 分配峰值、整卡 NVIDIA 峰值、RAM、Commit 和页面文件；
- 双 rank 层数、P2P 调用和计算重叠；
- conditioning 有限性、异常清理和第二次运行后的 Commit 增量；
- 完整 Director 工作流若失败，错误属于 Text Encoder 还是 Diffusion。

## 4. 实验性尝试与取舍

### 4.1 已实装

1. 删除实体 FP32 `[S,S]` 因果遮罩。xFormers 使用结构化 `LowerTriangularMask`；Torch 路径使用 `is_causal=True`。
2. xFormers 使用五维 GQA，不把 4 个 K/V 头实体复制成 32 头；Torch Efficient 明确扩展 K/V，内核不可用时直接失败，不回退 Math。
3. 最大输入第一次在 Attention 通过后于 MLP `down_proj` OOM。实装 4096-token MLP 分块、就地 SiLU/乘法，并在 MLP 前释放 Q/K/V/Attention 临时张量；相同输入随后通过。
4. 长序列 xFormers 按两个独立 KV 组执行，并复用已失效 Q 的存储保存输出。28k 峰值由 1753.38 MiB 降至 1314.13 MiB，语言阶段没有可测退化。
5. Benchmark 模式强制重新编码，普通使用仍保留 ComfyUI 缓存；否则第二次“热运行”会被整图缓存跳过，不能作为 Text Encoder 测试。
6. UI→API 转换器对旧工作流缺失的新可选 widget 使用 schema 默认值，避免旧 Director JSON 因新增字段错位。
7. O11 benchmark 结果只写入唯一工作树 `.local/artifacts`，O11 视频只写入 `output/video/raylight_o11`。

### 4.2 已测试但未实装

| 实验 | 结果 | 未实装原因 |
|---|---|---|
| 按 8192 query token 切 xFormers | 28k 峰值约 1752.75 MiB | 峰值几乎不变，增加 kernel 调用 |
| 按 4096 query token 切 xFormers | 28k 峰值约 1753.38 MiB | 同样无收益，可能拖慢 50 层 |
| 仅切 KV 组并新建完整输出 | 28k 峰值约 1752.13 MiB | 完整输出仍占峰值；改为复用 Q 后才有效 |
| 切换其他 xFormers kernel | V100 仅 CUTLASS 可用 | Flash2 要求 SM80，Flash3 要求 SM90，Triton Split-K 要求 SM80 |

上述失败实验均保留在本地 O11 artifact 和提交历史中，没有通过修改门槛或减少输入掩盖。

## 5. 测试数据结果

### 5.1 Gate B 单卡本地 Attention 容量

所有输出有限，均未调用 Math。

| 后端 | 序列 | 峰值 MiB | 耗时秒 | 门槛 |
|---|---:|---:|---:|---|
| XFORMERS_GQA | 6992 | 442.00 | 0.047 | 通过 |
| XFORMERS_GQA | 18000 | 846.94 | 0.283 | ≤1.10 GiB，通过 |
| XFORMERS_GQA | 28000 | 1314.13 | 0.652 | ≤1.65 GiB，通过 |
| TORCH_EFFICIENT | 6992 | 468.00 | 0.042 | 通过 |
| TORCH_EFFICIENT | 18000 | 1200.00 | 0.258 | ≤1.40 GiB，通过 |
| TORCH_EFFICIENT | 28000 | 1861.38 | 0.627 | ≤2.10 GiB，通过 |

本地原始 JSON：`.local/artifacts/o11/attention-capacity-accepted/`。

### 5.2 Gate C/D 冻结最大 conditioning

AUTO 两次均选择 `XFORMERS_GQA`。expanded sequence 为 24,236，其中文字位置 5,034、视觉位置 19,202。

| 指标 | 冷运行 | 热文件缓存 |
|---|---:|---:|
| Text Encoder 总耗时 | 112.94 s | 112.55 s |
| 视觉预处理 | 14.95 s | 14.95 s |
| 语言分片加载 | 17.29 s | 16.73 s |
| 50 层语言主干 | 76.08 s | 76.29 s |
| Text Encoder PyTorch 峰值 GPU0/GPU1 | 10950/11317 MiB | 10966/11334 MiB |
| 整卡 NVIDIA 峰值 GPU0/GPU1 | 16219/16195 MiB | 16241/16217 MiB |
| RAM 使用峰值 | 52719 MiB | 52820 MiB |
| Commit 峰值 | 64299.7 MiB | 64682.2 MiB |
| 页面文件使用峰值 | 0 MiB | 0 MiB |
| 结束后 GPU0/GPU1 | 1388/1330 MiB | 1408/1350 MiB |

两次完成后的 Commit 为 42036.3 和 41978.3 MiB，第二次比第一次低 58.0 MiB；未出现持续增长。每次 50 层均由两卡参与，350 个 Linear pair 全部有重叠，冷/热总 P2P 发送量均为 99,270,656,000 bytes。

强制 `TORCH_EFFICIENT` 的同输入也已通过：Text Encoder 117.50 s，语言主干 78.09 s，整卡峰值 15895/16163 MiB，页面文件 0。

最终冷/热报告：`.local/artifacts/o11/max-context-runs-post-group-chunk/20260901-152653-auto/report.json`。

### 5.3 用户真实 Director 强度测试

输入为 5802 字符（4898 非空白）、7 图、1 个 15.0 秒/854×480/30 fps 视频、1 个 WAV。Text Encoder 实际 expanded sequence 为 12,302，其中文字 1,387、视觉 10,915。

| 指标 | 结果 |
|---|---:|
| 实际后端 | XFORMERS_GQA |
| Text Encoder 总耗时 | 62.88 s |
| 视觉预处理 | 8.30 s |
| 分片加载 | 16.73 s |
| 语言主干 | 33.41 s |
| GPU0/GPU1 PyTorch 峰值 | 9244/9451 MiB |
| 清理后 GPU0/GPU1 分配 | 24.4/16.3 MiB |

因此该输入完整通过 Text Encoder；相对冻结 24,236 序列门槛还保留约 1.97 倍序列容量验证余量。

同一工作流随后进入 1056×608、365 帧 Diffusion 第一轮时 OOM：每卡约 13.19 GiB 已分配，RMSNorm 还需 1.77 GiB，而 GPU0 仅余 1.59 GiB。该失败属于 Diffusion 最大采样边界，不属于 Text Encoder，未生成视频。完整日志保存在 `.local/artifacts/o11/director-max-input/20260901-150832-director/`。

按用户要求，仅把输出从 0.6 MP 降为 0.4 MP（864×480），其余 15 秒、365 帧、7 图、1 视频、1 音频、5802 字符、REF2VA Turbo4、一采直出条件全部保持不变。该运行同样在 Diffusion 第一轮 OOM，但已从 RMSNorm 推进到 MLP `fc2` 投影：完成 1/4 步前后，`fc2` 还需 1.40 GiB，每卡约 13.99 GiB 已分配，GPU0 剩余约 0.79–0.92 GiB。端到端到报错为 240.93 秒，P2P profile 记录 32 次 collective、42 个子块和约 2.29 GiB 单向远端数据；未生成视频。证据位于 `.local/artifacts/o11/director-04mp/20260901-155246-director/`。这证明降分辨率有效减少激活，但 365 帧仍超过当前 Diffusion 安全边界。

### 5.4 Gate E 集成采样

缩小的 REF2VA Turbo4 集成工作流使用 864×480、73 帧、约 3.04 秒：

- 完整执行 265.79 s；预采样 79.91 s；采样 143.63 s（35.91 s/it）；VAE 解码 16.04 s；保存 3.67 s；
- GPU0/GPU1 峰值 10164/10122 MiB，双卡利用率均达到 100%；页面文件 0；
- 输出 H.264/AAC，73 帧、24 fps，无黑帧，接触表检查内容连贯。

视频位于 `ComfyUI/output/video/raylight_o11/minimax_h3_ref2va_o11_gate_e_run0_00001_.mp4`；本地证据位于 `.local/artifacts/o11/integration/`。

### 5.5 自动化回归

- O11 最终全仓：`664 passed, 25 warnings, 35 subtests passed`，100.59 秒；
- Ruff：修改范围使用 `--no-cache`，全部通过；
- 发布提交包含 463 个跟踪文件；部署脚本完成后必须返回 `DEPLOYMENT_OK files=463`，随后用 `-VerifyOnly` 再次核对；
- 发布前导入自检和双 V100/TCC/P2P 探针通过；128–384 MiB all-gather 约 61.54–61.62 GiB/s；
- 旧工作流缺少 `text_encoder_attention` 时按 AUTO 解析；普通 `CLIPLoader` 未修改。

## 6. 测试总结

Gate A–F 的 Text Encoder 范围全部通过。O11 解决了原始 6992-token 数学注意力 OOM，并把可验证上限扩展到真实 15 秒视频、7 图和 7000 字符；用户的 7 图+视频+音频 Director 输入也通过且有明显余量。AUTO 在目标机器确定选择 xFormers GQA，Torch Efficient 是明确可用的正式回退，Math 只允许不超过 2048 token 的调试对照。

能力边界必须保持准确：O11 不会把两张 16GB V100 变成透明 32GB，也没有让 15 秒/0.6 MP 或 0.4 MP 的 Diffusion Model 自动可采样。两种最大 Director 采样均已实测 OOM；当前可确认的是 Text Encoder 容量和短视频集成采样均正常。
