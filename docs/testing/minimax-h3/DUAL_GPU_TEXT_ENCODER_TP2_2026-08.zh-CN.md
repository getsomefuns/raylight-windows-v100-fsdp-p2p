# MiniMax H3 双 V100 Text Encoder TP2 测试汇总

## 文档状态

- 阶段：O10。
- 当前任务：Task 3，MiniMax H3 NVFP4 row/column 分片。
- 状态：完成；Gate A、Gate B 均通过，准备进入并行 Linear。

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
