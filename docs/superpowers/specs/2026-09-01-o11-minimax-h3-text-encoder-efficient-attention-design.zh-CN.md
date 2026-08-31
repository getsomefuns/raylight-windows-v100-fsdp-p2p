# O11：MiniMax H3 双 V100 Text Encoder 高效注意力设计

> 状态：架构方向已获用户批准；本文书面规格待用户复核；实现尚未开始。
>
> 日期：2026-09-01
>
> 唯一工作树：`E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`

## 1. 目标

O11 修复 O10 双卡 Qwen3-VL Text Encoder 在长多模态上下文下从 PyTorch SDPA 静默回退到数学注意力、创建平方级注意力矩阵并在进入采样前 OOM 的问题。

本阶段的最终容量门不是合成探针，而是以下真实组合必须完整通过 Text Encoder：

- 1 个实际时长为 15.0 秒（误差不超过 1 个源视频帧）的参考视频；
- 7 张不同的参考图片；
- 1 段规范化后恰好 7000 个 Unicode 字符的有效提示词；
- MiniMax H3 Qwen3-VL-32B NVFP4/AWQ Text Encoder；
- 原生 Windows、2× Tesla V100-SXM2-16GB、TCC、PyTorch 2.7.0+cu126、xFormers 0.0.30。

完成的定义是：ComfyUI 真实工作流成功返回有限、结构正确的 conditioning，Text Encoder 阶段无 OOM、无 NaN/Inf、无数学注意力回退，并在结束后释放两卡语言模型分片。仅通过 6992/18000 token 微基准不算完成。

## 2. 非目标

- O11 不承诺上述最大上下文随后一定能完成 Diffusion 采样；采样显存是下一层独立边界。
- 不把视觉塔拆到两张 GPU；视觉预处理继续在 GPU0 串行执行。
- 不修改 ComfyUI 核心、MiniMaxH3Director 或 NVIDIA 驱动。
- 不升级 Python、PyTorch、CUDA、xFormers 或当前模型格式。
- 不实现训练、反向传播、三卡以上或跨机器 Text Encoder。
- 不把 NCCL 作为本阶段依赖。

## 3. 已确认根因与基线

O10 在提交 `eb7e04b` 中新增 TP2 Qwen3-VL 语言块，并直接调用：

```python
F.scaled_dot_product_attention(..., enable_gqa=True)
```

每卡有 32 个 Query 头和 4 个 Key/Value 头。当前 PyTorch Efficient 内核要求 Q/K/V 头数一致；V100 环境没有可用 Flash 后端，cuDNN Attention 未启用，因此自动调度回退到 Math。

对本次 6992 token 输入，每卡单个 FP32 注意力矩阵为：

```text
32 × 6992 × 6992 × 4 bytes ≈ 5.83 GiB
```

这与实际 OOM 的 5.83 GiB 申请完全一致。O10 的 221 token 验收没有暴露问题，因为同一平方矩阵当时只有约 6 MiB。

当前 TP2 还显式构造 `(sequence, sequence)` FP32 因果遮罩。它在 18000 token 时约为 1.21 GiB，在 28000 token 时约为 2.92 GiB，也必须删除。

已完成的只读本机探针：

| 序列 | 后端 | 注意力探针峰值 | 单次耗时 |
|---:|---|---:|---:|
| 6992 | PyTorch Efficient（扩展 K/V） | 468 MiB | 0.056 秒 |
| 6992 | xFormers 原生 GQA | 360 MiB | 0.047 秒 |
| 18000 | PyTorch Efficient（扩展 K/V） | 1200 MiB | 0.264 秒 |
| 18000 | xFormers 原生 GQA | 917 MiB | 0.256 秒 |

xFormers GQA 对数学 SDPA 的小规模数值对照为 `max_abs=1.67e-6`、`mean_abs=6.44e-8`，在 `atol=rtol=1e-4` 下通过。

## 4. 保留的两阶段生命周期

视觉阶段和语言阶段保持严格串行：

```text
GPU0：token embedding + Qwen3-VL 视觉塔
GPU1：空闲
        ↓
视觉 embedding、DeepStack、token tags 全部转回 CPU
        ↓
销毁视觉塔和 token embedding，清理 GPU0
        ↓
GPU0 + GPU1：加载各自约一半语言权重并执行 50 层 TP2
        ↓
conditioning 转 CPU，销毁语言分片，归还 Ray worker GPU 租约
```

高效注意力只替换语言主干的 Attention，不改变视觉塔。视觉阶段必须单独记录 GPU0 峰值和耗时，以防 15 秒视频与 7 张图片使视觉输出或 DeepStack 累积成为新瓶颈。

## 5. 后端架构

新增 Text Encoder 专用后端枚举：

```text
AUTO
XFORMERS_GQA
TORCH_EFFICIENT
MATH_DEBUG
```

后端在一次 encode 开始时选择一次，50 层必须使用同一后端。禁止逐层静默切换。

### 5.1 XFORMERS_GQA

这是目标双 V100 环境的首选后端。

- Q 使用 `[B, S, 4, 8, 128]` 五维 GQA 布局。
- K/V 使用 `[B, S, 4, 1, 128]`，通过 view/expand 广播，不实体复制成 32 头。
- 调用 `xformers.ops.memory_efficient_attention`。
- 因果关系使用 `xformers.ops.LowerTriangularMask()`。
- 仅使用 forward；不依赖 xFormers 的 GQA backward。

### 5.2 TORCH_EFFICIENT

这是无可用 xFormers GQA 时的正式回退。

- 将每卡 4 个 K/V 头显式扩展到 32 头。
- 使用 `torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)` 强制 Efficient 后端。
- 使用 `is_causal=True`，不传实体平方遮罩。
- 如果 Efficient 内核不可用，明确失败；不得继续回退 Math。

### 5.3 MATH_DEBUG

只用于小规模数值对照。

- 序列长度上限固定为 2048。
- 超过上限立即报错，并提示选择 `AUTO`、`XFORMERS_GQA` 或 `TORCH_EFFICIENT`。
- 不能作为真实工作流 fallback。

### 5.4 AUTO

AUTO 是新旧工作流的推荐默认值：

1. 在加载 50 层语言权重前运行一次小型、可缓存的设备能力探针。
2. xFormers GQA + 因果遮罩通过时选择 `XFORMERS_GQA`。
3. 否则探测 PyTorch Efficient + 扩展 K/V。
4. 两者都不可用时终止并报告两条拒绝原因。
5. 绝不自动选择 Math。

目标机器上 AUTO 必须选择 `XFORMERS_GQA`。

## 6. 节点接口与兼容性

只修改专用节点 `Load MiniMax H3 Text Encoder (Dual V100 TP2)`：

- 新增可选输入 `text_encoder_attention`；
- 默认值为 `AUTO`；
- 已保存、没有该字段的 O10 工作流按 `AUTO` 加载；
- 保留现有 `execution_mode`、`clip_name` 和 `ray_actors_init` 语义；
- 不修改 `RayInitializer` 的公开 schema。

`RayInitializer.XFuser_attention=TORCH_EFFICIENT` 继续只控制 Diffusion/xFuser 采样注意力。UI tooltip 和文档必须明确它不控制 Text Encoder。

## 7. 因果遮罩与有效 token

删除 TP2 Backbone 当前 `_causal_mask(sequence, float32)` 的实体矩阵。

- xFormers 路径使用符号化 `LowerTriangularMask`。
- Torch 路径使用 `is_causal=True`。
- 当前单 batch、无有效 padding 的 MiniMax conditioning 不能改变语义。
- 如果输入存在 padding，必须在进入后端前裁掉无效尾部/头部，或使用不实体化的结构化 bias；禁止重新创建 `[S,S]` 浮点 mask。
- conditioning 返回前恢复与 tokenizer 契约一致的序列位置和 `minimax_token_tags`。

## 8. 可观测性

每次 Text Encoder 正常日志只打印一条聚合摘要：

- 请求后端和实际后端；
- 总 sequence token；
- 文本、图片、视频视觉 token 分项；
- 参考视频实际读取帧数、进入 Qwen 的 2 FPS 帧数和视觉块数；
- 参考图片数量及每张视觉 token；
- `preprocess_and_vision`、`backbone_load`、`language_backbone`、cleanup 耗时；
- GPU0 视觉峰值；
- GPU0/GPU1 语言阶段峰值；
- 是否检测到 Math fallback；
- 清理后的两卡显存占用。

逐层日志继续默认关闭。诊断开关只能增加聚合性能数据，不得打印大型 tensor 或完整提示词。

## 9. 错误处理与清理

- 后端探针失败必须在加载语言权重前失败。
- 任意 rank OOM、NaN/Inf、shape 不一致或输出分歧必须保留原始错误。
- 异常路径必须释放 backbone、临时 Q/K/V、视觉结果 GPU 副本和外部 GPU 租约。
- 失败后允许用户不重启 ComfyUI，再次以另一个显式后端执行。
- 不允许捕获 xFormers 错误后在已高占用的 GPU 上静默重试 Math。

## 10. 冻结验收输入

实现开始后、第一次正式工作流测试前创建本地验收 manifest。manifest 记录相对 ComfyUI input 的文件名、SHA-256、媒体元数据和工作流 SHA-256，不保存绝对用户路径。

### 10.1 视频

- 恰好 1 个真实参考视频；
- 容器可被当前 ComfyUI loader 完整解码；
- 实际时长与 15.0 秒的差值不超过 1 个源视频帧；
- 分辨率不得低于 854×480；
- 不以黑帧、静帧或无意义重复帧伪造容量测试；
- 记录源 FPS、源帧数、24 FPS 后帧数、MiniMax 对齐后实际使用帧数和最终视频视觉 token。

MiniMax 对齐可能把 15 秒源视频裁到最近的 `17k+5` 帧边界。报告必须同时写出“输入是 15 秒”和“模型实际消费多少帧”，不得把裁剪后的时长冒充完整 15 秒。

### 10.2 图片

- 恰好 7 张不同内容、不同 SHA-256 的真实参考图；
- 每张短边不低于 480 像素；
- 使用 `ref_image_size=match`；
- 不重复同一文件凑数量；
- 记录每张预处理尺寸和视觉 token。

### 10.3 提示词

- 换行统一为 LF 后，Python `len(prompt)` 必须恰好为 7000；
- 使用有实际镜头、主体、动作、场景、风格、音频或连续性含义的提示词；
- 禁止用单字重复、空格、随机字符或复制无意义段落凑长度；
- 记录字符数、tokenizer 文本 token 数和提示词 SHA-256；
- 正文不写入公开测试汇总，只保存受控本地 fixture 或散列。

### 10.4 工作流

- 使用 ComfyUI 工作流执行，不以独立 Python attention probe 替代最终验收；
- 使用 `MiniMax H3 Reference to Video` 和 O11 双卡 CLIP Loader；
- 生成画布固定 1056×608，`length=362`，24 FPS；
- 连接 1 个参考视频和 7 张参考图片；
- 工作流在 conditioning 输出处终止，不连接 Diffusion sampler；
- 保存到 `example_workflows/`，同时部署一份到用户 workflows 目录用于手工复现；
- 保存 API prompt 和 UI workflow 两种 JSON 的 SHA-256。

## 11. 验收标准

### Gate A：后端单元正确性

- `XFORMERS_GQA`、`TORCH_EFFICIENT`、`AUTO`、`MATH_DEBUG` 路由均有测试。
- 64、221、256、2048 token 下，两种高效后端与 Math 对照 `max_abs <= 1e-4`、`mean_abs <= 1e-5`。
- 所有输出有限，无 NaN/Inf。
- `MATH_DEBUG` 在 2049 token 明确拒绝。
- 模拟 xFormers 不可用时 AUTO 选择 Torch Efficient；两者都不可用时明确失败。

### Gate B：V100 注意力容量

在真实 GPU0 上分别测试 6992、18000、28000 token：

- 两种高效后端全部完成，不创建实体 `[S,S]` mask。
- 运行时不得调用 Math backend。
- 18000 token 时 xFormers GQA 注意力探针峰值不高于 1.10 GiB，Torch Efficient 不高于 1.40 GiB。
- 28000 token 时 xFormers GQA 注意力探针峰值不高于 1.65 GiB，Torch Efficient 不高于 2.10 GiB。
- xFormers GQA 峰值必须低于同序列 Torch Efficient。

这些阈值只验收 Attention 内核，不代替完整 50 层验收。

### Gate C：GPU0 视觉阶段

使用第 10 节冻结的 15 秒视频、7 张图片和 7000 字符提示词：

- GPU0 完成 token embedding、视觉塔和 DeepStack；GPU1 在视觉阶段不错误加载完整语言权重。
- 视觉阶段无 OOM、无 NaN/Inf。
- 所有 prepared embedding、DeepStack 和 tags 在加载语言 backbone 前已转到 CPU。
- 视觉塔和 token embedding 的 GPU 引用被释放；随后语言加载前 GPU0 不保留视觉权重。
- 报告实际视频/图片/text token 分项和 GPU0 峰值。

### Gate D：真实最大 Text Encoder 硬门槛

这是 O11 的最终功能门，必须使用第 10 节同一个 ComfyUI 工作流和同一组散列输入。

AUTO/XFORMERS_GQA 至少完成 1 次冷运行和 1 次热文件缓存运行；强制 TORCH_EFFICIENT 至少完成 1 次：

- 输入确实为 15 秒视频、7 张不同图片、7000 字符有效提示词。
- `encode_from_tokens_scheduled` 完整返回 conditioning，不进入采样也不 OOM。
- conditioning 形状为 `[1, S, 5120]`，`S` 等于日志记录的最终 expanded sequence。
- conditioning 和 `minimax_token_tags` 全部有限且长度一致。
- 两卡 50/50 层均参与，TP2 汇总后两个 rank 输出一致。
- 任意单卡 Text Encoder 峰值不得超过 15.0 GiB。
- 日志明确显示没有 Math fallback，也没有实体平方 mask。
- Text Encoder 完成后两卡语言权重和临时张量被释放；只允许 CUDA context/Ray worker 已知基础占用。
- 不增加持续增长的主进程私有内存；同一工作流连续执行两次时第二次结束后的 commit 与第一次结束后相比增长不超过 512 MiB。

只要以上任意一项未通过，O11 就不能宣称支持该最大输入组合。

### Gate E：集成与旧问题回归

- 原 11.33 秒、854×480、6992 token OOM 样本必须通过 Text Encoder。
- O10 221 token TP2 smoke 必须继续通过。
- 使用最大 conditioning 之后，至少用一个缩小的 3 秒/480p/4 步工作流完成采样集成 smoke，证明 conditioning 契约未损坏；它不代替最大 Text Encoder Gate D。
- 普通 `CLIPLoader`、O6、O8、O9、LTX、MiniMax I2V/REF2VA 和 MMH3Director 既有回归测试不得失败。
- 旧 O10 工作流未保存新字段时必须按 AUTO 正常加载。

### Gate F：发布与记录

- 默认后端为 AUTO，目标机器实际选择 xFormers GQA。
- 中英文 README、用户指南、O11 release note 和 MiniMax H3 测试汇总与代码、节点 schema、实测结果一致。
- 测试汇总分别记录视觉、语言、清理阶段的时间，双卡显存峰值，RAM/commit/pagefile 峰值和实际 token 分解。
- 实验失败项必须记录未采用原因；不得只保留成功数据。
- 唯一工作树通过测试、审查和提交后，才部署到 `ComfyUI/custom_nodes/raylight`。
- 部署版本与唯一工作树跟踪文件 hash 一致后才允许推送 GitHub。

## 12. 停止条件

- 若高效 Attention 通过但 Gate D 在 MLP 阶段 OOM，必须记录峰值与张量生命周期并转入 MLP 内存优化；不得降低 15 秒/7 图/7000 字符标准。
- 若视觉阶段先 OOM，必须优化视觉输出/DeepStack 的 CPU 流式落盘或生命周期；不得用减少图片、缩短视频或截断提示词伪造通过。
- 若最大 conditioning 通过但采样 OOM，O11 仍可按“Text Encoder 容量通过、采样边界未通过”收口，但不能宣称完整最大工作流可生成视频。
- 不允许通过提高页面文件、把正式 CUDA Attention 转到 CPU、关闭输入 reference 或截断 token 达成验收。

## 13. 设计完成后的下一步

用户审核本设计后，另行编写 O11 可执行实施计划。实施必须从失败回归测试开始，先实现后端抽象和无平方 mask，再依次通过 Gate A–F。本文在用户批准前不授权代码修改。
