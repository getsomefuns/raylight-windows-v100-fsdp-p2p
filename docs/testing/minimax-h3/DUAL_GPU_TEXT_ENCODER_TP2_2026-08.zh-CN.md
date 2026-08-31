# MiniMax H3 双 V100 Text Encoder TP2 测试汇总

## 文档状态

- 阶段：O10。
- 当前任务：Task 8，节点、工作流与 Director 接入。
- 状态：完成；Gate A–D、Worker 生命周期和图结构接入通过，准备进入真实显存、数值、视频与多段稳定性验收。

## 测试验证目标

在开发双 GPU Text Encoder 之前，固定同一台机器、同一 checkpoint、同一提示词与同一参考图的单 GPU 基线。后续 TP2 的速度、显存、内存、分页文件与数值质量均以本基线为对照，不引用外部作者数据替代本机结果。

## 测试条件变量

- 工作流来源：当前 `MMH3Director.json`。
- Text Encoder：工作流现用 `CLIPLoader`，类型为 `minimax`、设备策略为 `default`。
- 输入：读取 Director 时间轴第一段的提示词与第一张参考图；正文不复制开发者提示词和素材。
- 执行路径：ComfyUI API 工作流，仅保留 CLIP、VAE、参考图 conditioning 和只读输出节点，不进入扩散模型采样。
- 冷/热定义：每个独立 ComfyUI 进程先执行一次冷编码，再在不卸载 Text Encoder 的情况下执行一次热编码；共 3 个独立进程，得到冷 3 次、热 3 次。
- 防缓存：热编码只改变 latent 帧数，不改变提示词或参考图，强制 conditioning 节点重新执行且保持 token 数一致。
- 计时开关：`RAYLIGHT_TEXT_ENCODER_BENCHMARK=1` 时才安装只读钩子；正常启动不修改 `CLIP` 类，也不打印逐层诊断。

## 测试关注信息

- checkpoint 文件选择器与原工作流 SHA-256。
- token 数。
- Text Encoder 从模型换入开始到编码完成的总耗时。
- Text Encoder 有效时间窗口内的 GPU 0 显存峰值、系统物理内存峰值和分页文件峰值。
- 冷、热各 3 次的中位数，以及三次测试中的资源峰值。

## 数据记录契约

每次有效记录必须包含：`checkpoint`、`workflow_sha256`、`mode`、`run_index`、`token_count`、`te_total_s`、`gpu0_peak_mib`、`ram_peak_mib` 和 `pagefile_peak_mib`。缺字段、负值、非有限数、空 checkpoint 或非法工作流 hash 都会被拒绝。

原始 JSON、CSV、监控采样和进程日志保存在被 Git 忽略的 `.local/artifacts/o10/text-encoder-baseline/`，仓库正文只维护汇总数字与必要的相对附件位置。

## 测试数据结果

有效基线目录：`.local/artifacts/o10/text-encoder-baseline/20260831-142005-single-v100-cold-warm/`。

固定条件：

- checkpoint：`MMH3\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`。
- 工作流 SHA-256：`97defbaff3f4046d53573d2bb36dbfe01e9b4ecb451ed9481a1afe7690a08cd4`。
- token 数：六次均为 221。
- GPU 1 计算利用率峰值：六次均为 0%；因此本表是 GPU 0 单卡计算基线。GPU 1 的 326–336 MiB 仅为进程启动产生的上下文占用，不计为并行计算。

| 温度 | 第 1 次 | 第 2 次 | 第 3 次 | 中位数 | GPU 0 显存峰值 | 系统物理内存峰值 | 分页文件峰值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 冷 | 18.91 s | 20.97 s | 20.76 s | 20.76 s | 13,763 MiB | 35,882 MiB | 760 MiB |
| 热 | 10.37 s | 10.41 s | 10.35 s | 10.37 s | 13,906 MiB | 40,358 MiB | 760 MiB |

两个温度组的 GPU 0 计算利用率峰值均达到 100%。表中的系统物理内存与分页文件是 Text Encoder 有效窗口内的全机已用量，不是单进程独占量；后续 TP2 必须用相同采样定义比较。

第一次实机试跑曾把包含视觉占位项的 token 容器误计为 3。该结果已作废且未进入本表；修复后的单元测试使用真实 MiniMax `(token_or_vision_entry, weight)` 结构，正式六次记录稳定为 221。

## 当前总结

基线工具不会改变 conditioning 或工作流输出；其 begin/complete 事件把资源采样严格裁剪在 Text Encoder 有效区间，避免把参考图 VAE 编码或后续采样误算入 Text Encoder。

Task 1 已关闭。O10 后续验收的本机对照值为：冷中位数 20.76 秒、热中位数 10.37 秒、GPU 0 显存峰值 13,906 MiB。此处只建立单卡基线，尚未声明 TP2 已有性能收益。

## Task 2：双 GPU P2P SUM / Gate A

测试附件：`.local/artifacts/o10/text-encoder-tp2/p2p-sum-probe-20260831.json`。

测试对象为单进程、固定 GPU 0/1 的 `CudaPeerGroup2`。每次操作并发提交两个方向的 CUDA peer copy，两卡分别把本地 partial 与接收缓冲相加；不建立 Gloo/NCCL ProcessGroup，也不构造 world-size 倍 all-gather 缓冲。operation ID 必须严格递增，shape、dtype、连续性、设备或拓扑不匹配均立即失败。

| dtype | 每卡 payload | 正式迭代 | 平均通信与本地求和 | 双向聚合带宽 | 迭代显存增长 | 数值 |
|---|---:|---:|---:|---:|---:|---|
| FP16 | 1 MiB | 100 | 0.299 ms | 6.54 GiB/s | 0 / 0 B | 逐元素一致、finite |
| FP16 | 16 MiB | 100 | 0.535 ms | 58.45 GiB/s | 0 / 回落 | 逐元素一致、finite |
| FP16 | 64 MiB | 100 | 1.517 ms | 82.39 GiB/s | 0 / 回落 | 逐元素一致、finite |
| FP32 | 1 MiB | 100 | 0.302 ms | 6.47 GiB/s | 0 / 回落 | 逐元素一致、finite |
| FP32 | 16 MiB | 100 | 0.527 ms | 59.32 GiB/s | 0 / 回落 | 逐元素一致、finite |
| FP32 | 64 MiB | 100 | 1.523 ms | 82.09 GiB/s | 0 / 回落 | 逐元素一致、finite |

每种组合先执行 1 次不计时预热，表中均为后续 100 次结果。1 MiB 由固定的 Python/CUDA event 提交成本主导；16/64 MiB 更接近 Text Encoder 层内 partial 的有效数据面性能。负的“增长”表示前一个更大测试缓冲在后续采样期间释放，不是测量错误。

Task 2 的 19 项定向测试与当时全仓库 522 项测试均通过。Gate A 通过，可以进入 NVFP4 row/column 分片；该结果只证明 SUM 原语可用，尚不代表完整 Text Encoder TP2 已通过 Gate B–F。

## Task 3：真实 NVFP4 row/column 分片 / Gate B

测试附件：`.local/artifacts/o10/text-encoder-tp2/nvfp4-shard-probe-20260831.json`。

测试 checkpoint 为 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`。测试只读取 layer 0 四种代表性矩阵；先从 packed qdata、FP8 block-scale 与全局 scale 构造量化权重，再分别创建 rank 0/1 分片。分片构造完成后才执行完整反量化作为验收基线，因此基线本身不属于正式分片路径。

| 线性层 | 分片方式 | 源逻辑 shape | 每 rank shape | 两分片 qdata 总量 | 两分片 scale 总量 | 分片构造时间 | 数值结果 |
|---|---|---:|---:|---:|---:|---:|---|
| q_proj | 输出行 row | 8192×5120 | 4096×5120 | 20,971,520 B | 2,621,440 B | 22.3 ms | 逐元素一致、finite |
| o_proj | 输入列 column | 5120×8192 | 5120×4096 | 20,971,520 B | 2,621,440 B | 14.2 ms | 逐元素一致、finite |
| up_proj | 输出行 row | 25600×5120 | 12800×5120 | 65,536,000 B | 8,192,000 B | 15.3 ms | 逐元素一致、finite |
| down_proj | 输入列 column | 5120×25600 | 5120×12800 | 65,536,000 B | 8,192,000 B | 35.5 ms | 逐元素一致、finite |

四种矩阵中，两分片 packed qdata 总字节数均严格等于源 qdata；两分片 block-scale 总字节数也严格等于源 block-scale。因此 row/column 分片没有丢失数据，也没有为了分片保留第二份完整量化权重。

NVFP4 block-scale 采用 cuBLAS tiled/swizzled 排列，不能按表面二维 shape 直接切。实现仅把这张体积约为 qdata 八分之一的 FP8 缩放表还原布局、切片、重新排列；packed 4-bit qdata 直接切片，全局 `weight_scale_2` 只读复用。单元测试通过 monkeypatch 禁止调用 `QuantizedTensor.dequantize()`，证明分片构造过程不会生成完整 FP16/FP32 权重临时副本。

Gate B 结论：通过。该结论证明真实 checkpoint 的量化权重可以无损拆为两个 TP rank，但尚未证明 Linear 数学、完整 Qwen3-VL block、GPU 显存或端到端速度；这些分别属于 Gate C–F。

## Task 4：双 V100 并行 Linear

测试附件：`.local/artifacts/o10/text-encoder-tp2/linear-probe-20260831.json`。

Column Parallel Linear 按输出通道拆权重，两卡接收相同 hidden，各自产生一半输出通道；Row Parallel Linear 按输入通道拆权重，两卡计算同一完整输出的 partial，再用 Task 2 的 CUDA P2P SUM 聚合。Row bias 只在 SUM 之后各加一次，避免被重复累加。

| 测试 | 条件 | 数值 | rank 0 / rank 1 | 墙钟 | 重叠率 |
|---|---|---|---:|---:|---:|
| dense FP32 Column | 2048 tokens，每卡 4096×4096 | 单元测试 max abs ≤1e-4 | 5.47 / 5.89 ms | 6.34 ms | 91.8% |
| 真实 NVFP4 q_proj Column | 221 tokens，源 8192×5120，每卡 4096×5120 | 两卡 finite | 2.86 / 2.22 ms | 3.21 ms | 84.2% |
| 合成 NVFP4 Column | 256×128 | cosine 0.99999964，finite | 2.51 / 2.89 ms | 3.41 ms | 79.4% |
| 合成 NVFP4 Row + P2P SUM | 128×256 | cosine 0.99999988，finite，两卡逐元素一致 | 1.68 / 1.69 ms | 2.17 ms | 71.0% |

Row 小矩阵正式 P2P SUM payload 为 9,728 B，耗时 0.469 ms。小矩阵重叠率受双线程调度与 CUDA event 固定开销影响，因此并行能力判断以真实 221-token q_proj 和 FP32 大矩阵为主；数值正确性仍覆盖两个小型 NVFP4 方向。

### 已发现并修复的性能问题

初版虽使用两张独立 CUDA stream，但由一个 Python 主线程先调用 rank 0 的 `F.linear`，再调用 rank 1。V100 无原生 NVFP4，comfy_kitchen eager 路径要先在 Python 调度多步反量化；真实 q_proj 的局部 GPU 时间只有约 1.2–1.4 ms，而两次顺序 Python 提交已经足以让两张卡近似串行，实测重叠率约 7%。

实装方案使用两个持久主机线程同时向 cuda:0/cuda:1 的独立 stream 提交。修复后同一真实 q_proj 重叠率为 84.2%，不是通过放宽标准获得。线程池会在后续完整 50 层实现中由整个 TP2 主干共享，不为每层创建一组线程。

Task 4 定向测试、双 V100 探针以及当时全仓库 543 项测试、35 个子测试均通过。该阶段证明并行 Linear 原语可用；完整 Attention、MLP、残差与误差累积仍由 Task 5 / Gate C 验收。

## Task 5：真实 Qwen3-VL block / Gate C

测试附件：`.local/artifacts/o10/text-encoder-tp2/qwen3vl-block-probe-20260831.json`。

当前 ComfyUI MiniMax H3 language 配置已冻结为：hidden 5120、intermediate 25600、50 层、64 个 Q head、8 个 KV head、head_dim 128、RMSNorm epsilon 1e-6、Qwen3-VL interleaved MRoPE。TP2 后每卡 32 个 Q head、4 个 KV head，GQA 的 8:1 比例保持不变。

真实探针读取 checkpoint layer 0 的 q/k/v/o、gate/up/down 七个 NVFP4 Linear、input/post-attention/q/k 四组 norm，以及 o_proj/down_proj 两条 BF16 `pre_quant_scale`。V100 执行前将 norm/scale 明确转换为 FP16；`pre_quant_scale` 按 o/down 的本地输入通道切半后在局部 Linear 前应用，不会遗漏 ModelOpt AWQ smoothing。

| 项目 | 结果 |
|---|---:|
| token 数 | 32 |
| TP2 对单 GPU同语义基线 cosine | 0.99999595 |
| max abs error | 0.015625 |
| finite | 是 |
| rank 0 / rank 1 hidden | 逐元素一致 |
| 每个测量 block P2P SUM | 2 次 |
| o_proj / down_proj operation ID | 3 / 4 |
| GPU0 / GPU1 显存峰值 | 663.7 / 655.6 MiB |

| Linear | rank 0 / rank 1 | 墙钟 | 重叠率 |
|---|---:|---:|---:|
| q_proj | 2.01 / 2.18 ms | 2.67 ms | 75.8% |
| k_proj | 2.18 / 2.24 ms | 2.78 ms | 75.1% |
| v_proj | 2.29 / 2.16 ms | 2.76 ms | 78.5% |
| o_proj | 2.39 / 2.27 ms | 2.90 ms | 77.7% |
| gate_proj | 3.14 / 3.18 ms | 3.49 ms | 90.3% |
| up_proj | 3.11 / 3.15 ms | 3.48 ms | 89.4% |
| down_proj | 2.89 / 2.84 ms | 3.15 ms | 90.9% |

单元测试另覆盖 dense FP32 单 block、合成 NVFP4 单 block、2/8/50 层 FP32 误差累积。50 层测试要求两卡每层输出一致、operation ID 严格覆盖 1–100、finite 且最终 max abs error 不超过 1e-3。Task 5 定向测试、真实双 V100 探针以及当时全仓库 550 项测试、35 个子测试均通过。

Gate C 结论：通过。该阶段证明真实单个 language block 的完整 Attention、MLP、RoPE、RMSNorm、残差、AWQ smoothing 和两次 P2P SUM 正确；尚未证明真实 50 层全 checkpoint、vision/DeepStack、conditioning 返回、生命周期或 Text Encoder 总速度，这些属于 Gate D–F。

## Task 6：完整 CLIP 代理 / Gate D

测试附件位于 `.local/artifacts/o10/text-encoder-tp2/`：

- `clip-proxy-single-20260831.json` / `.pt`：普通单 GPU ComfyUI 纯文本参照。
- `clip-proxy-tp2-20260831.json` / `.pt`：纯文本 TP2 及逐元素 conditioning 附件。
- `clip-proxy-i2v-single-20260831.json` / `.pt`：普通单 GPU 单图 I2V 参照。
- `clip-proxy-i2v-tp2-20260831.json` / `.pt`：单图视觉塔、DeepStack 与 50 层 TP2。
- `clip-proxy-ref2va-tp2-20260831.json` / `.pt`：双图 REF2VA TP2。
- `clip-proxy-video-tp2-20260831.json` / `.pt`：两帧视频 reference TP2。

代理不实例化普通 ComfyUI `CLIP` 的完整 Qwen3-VL-32B 模型。真实 checkpoint 中语言层量化权重约 12.78 GiB，按 row/column TP2 后每卡约 6.39 GiB；0.725 GiB INT8 token embedding 和视觉塔只在 GPU0 的短生命周期预处理器中运行。预处理完成后，embedding、DeepStack、MRoPE 元数据先回到 conditioning 中间设备，GPU0 预处理权重释放，再加载两卡语言分片，避免 GPU0 同时持有完整单卡语言模型和 TP2 分片。

| 路径 | token / tags（文本、视觉） | 单卡/TP2 总时长 | 单卡 GPU 峰值 | TP2 GPU0 / GPU1 峰值 | 数值与清理 |
|---|---:|---:|---:|---:|---|
| 纯文本 | 29 / 29、0 | 17.83 / 17.41 s | 11,778.5 MiB | 7,350.0 / 7,350.0 MiB | cosine 0.99999999997；tags 相同；关闭后 16.25 / 16.25 MiB |
| 单图 I2V，320×320 输入 | 137 / 35、102 | 19.03 / 16.63 s | 11,813.3 MiB | 7,396.2 / 7,390.0 MiB | cosine 0.99999930685；相对 RMSE 0.164%；tags 相同；关闭后 24.38 / 16.25 MiB |
| 双图 REF2VA，128×128×2 | 77 / 41、36 | 未做单卡重复参照 / 16.66 s | — | 7,376.5 / 7,368.3 MiB | finite；关闭后 25.27 / 17.15 MiB |
| 视频 reference，128×128×2 帧 | 59 / 41、18 | 未做单卡重复参照 / 19.43 s | — | 7,369.8 / 7,360.9 MiB | finite；关闭后 24.38 / 16.25 MiB |

表中总时长均为一次独立进程的冷探针，包含 checkpoint 读取、选择性预处理权重加载、语言权重加载、编码和清理。它用于 Gate D 数值/容量判定，不代替 Task 9 的冷 3 次、热 3 次中位数，也不能据此提前认定达到 20% 性能门槛。

单图对照的最大单点绝对误差为 37.98，但该位置的参照 conditioning 绝对峰值为 18,130.09；整体平均绝对误差为 0.0070、相对 RMSE 为 0.164%、cosine 为 0.99999931。结合 shape、token tags、finite 与后续视频质量门槛，Gate D 判定通过；正文保留最大值，避免只报告有利指标。

单元测试覆盖纯文本、单图 I2V、双图 REF2VA、视频 reference、当前 ComfyUI MRoPE 精确一致、DeepStack 注入层序、CLIP clone/metadata、Text Encoder LoRA 明确拒绝和编码异常清理。Task 6 完成时全仓库 `559 passed`、`35 subtests passed`；18 条 warning 为已有 Windows pytest 临时目录清理和 PyTorch nested tensor 原型提示，不是功能失败。

Gate D conditioning 结论：通过。真实 50 层 checkpoint、GPU0 视觉塔、DeepStack、`minimax_token_tags` 和两卡语言主干均已给出实机 conditioning 证据，TP2 显存峰值从约 11.8 GiB 单卡降为每卡约 7.4 GiB，异常/正常关闭后未遗留 6.7 GiB 语言分片。节点尚未公开；Worker 显存租约由下面的 Task 7 补充验收关闭。

## Task 7：Worker 双 GPU 显存租约

实机附件：`.local/artifacts/o10/text-encoder-tp2/worker-lease-probe-20260831.json`。

本阶段把 Ray Worker 的 GPU 使用从隐式约定改成显式状态：`idle`、`loading_diffusion`、`loading_lora`、`sampling` 和 `leased_for_text_encoder`。Text Encoder 只有在两个 rank 都空闲时才能取得同一 `operation_id`/`owner` 的租约；任何一个 rank 失败都会归还另一 rank，不允许单卡租出后继续伪装成 TP2。

| 验证项 | 结果 |
|---|---|
| 真实 Worker 初始/结束状态 | rank 0/1 均为 `idle` |
| 双卡租约 | 两 rank 同时为 `leased_for_text_encoder`，operation/owner 一致 |
| free-memory 回执 | GPU0 15.25 GiB；GPU1 15.55 GiB |
| 冲突保护 | 租约期间两 rank 的 `load_unet` 均在入口处明确拒绝 |
| 单 rank 失败 | 已验证另一 rank 自动回滚 |
| 编码异常/用户取消 | 普通异常与 `KeyboardInterrupt` 均由 `finally` 归还 |
| 重复释放 | 同一 operation/owner 幂等；错误 owner 被拒绝 |
| Diffusion 缓存保持 | ModelPatcher 对象、CPU/mmap cache key、active request key 不替换；归还后可重新进入 sampling 状态 |

真实探针不加载 Diffusion 大模型，只验证两个实际 RayWorker 的远程协议、CUDA 清理、free-memory 回执、冲突拒绝和归还。缓存保持使用带真实 `clear_sampling_vram` 语义的生命周期回归验证；完整 Diffusion 模型的“编码后继续采样”由 Task 9 端到端工作流验收，不在本阶段冒充已经完成。

Task 7 定向生命周期回归为 `29 passed`；完成时全仓库为 `573 passed`、`35 subtests passed`。全量测试的 warning 仍来自 Windows pytest 临时目录清理和 PyTorch nested tensor 原型提示，不是租约失败。

Task 7 结论：通过。显存租约解决的是 RayWorker 与主进程 TP2 Text Encoder 争用同两张卡的问题，不把两卡显存合并成统一 32 GiB 地址空间，也不改变现有 Diffusion FSDP/Ulysses 的模型缓存格式。下一步是 Task 8 的显式节点和工作流接入。

## Task 8：节点、工作流与 Director 接入

测试附件：`.local/artifacts/o10/text-encoder-tp2/task8/`，包含由真实 ComfyUI `/object_info` 转换得到的 `i2v-api.json`、`ref2va-api.json` 和 `director-api.json`。

新增节点 `RayMiniMaxH3DualGPUCLIPLoader` 的真实服务端注册结果：

- 显示名：`Load MiniMax H3 Text Encoder (Dual V100 TP2)`。
- 分类：`Raylight/text_encoder`。
- 必需输入：`clip_name`、`execution_mode`、`ray_actors_init`。
- 输出：`CLIP`、`RAY_ACTORS_INIT`。
- 模式：`tp2` 或 `single_gpu_fallback`；不存在 `auto`，TP2 错误不会被单卡执行掩盖。
- 首个支持 checkpoint：`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`；其他文件名在节点入口明确拒绝。

三个新工作流均由冻结的已验收工作流机械生成，构建脚本为 `scripts/minimax-h3/build_tp2_workflows.py`，测试要求提交后的 JSON 与重新构建逐项相等：

| 路径 | 来源 | API 节点数 | Text Encoder / 采样旁路检查 |
|---|---|---:|---|
| `Minimax_H3_I2V_DualGPU_Text_Encoder_TP2_Raylight.json` | I2V FSDP Turbo8 | 21 | 1 个 TP2 Loader；0 个普通 CLIPLoader |
| `Minimax_H3_REF2VA_DualGPU_Text_Encoder_TP2_Raylight.json` | REF2VA FSDP Turbo4 | 21 | 1 个 TP2 Loader；0 个普通 CLIPLoader |
| `Minimax_H3_DualGPU_Text_Encoder_TP2_Raylight.json` | MMH3Director FSDP | 13 | 1 个 TP2 Loader；1 个 Ray MODEL Adapter；0 个普通 CLIPLoader；0 个本地采样器 |

普通 I2V/REF2VA 中原 conditioning → `RayInitializer.wait_for` 连接会与新 Loader 的 RayWorker 依赖形成环，因此新工作流明确移除该 wait link；运行期安全由双卡租约保证。Initializer 先输出到 TP2 Loader，Loader 的 `RAY_ACTORS_INIT` 再传给 RayUNETLoader。Director 原有 seed、steps、sampler、scheduler、shift、FPS、duration、resolution、timeline、prompt groups、VAE 和保存链均保持原值，只替换 CLIP 来源并串行化 actor 初始化。

41 项 Task 8 定向测试通过；全仓库为 `595 passed`、`35 subtests passed`。部署到 `E:\ComfyUI-py310\ComfyUI\custom_nodes\raylight` 后真实启动无节点导入错误，验证结束已关闭 ComfyUI 和 Ray，双卡无残留进程。

Task 8 结论：通过。这里证明的是节点注册、显式模式、工作流无环、Director 控制面和无重复本地采样旁路；尚未宣称三个工作流已完成真实视频质量/显存/多段验收，该证据必须由 Task 9 提供。
