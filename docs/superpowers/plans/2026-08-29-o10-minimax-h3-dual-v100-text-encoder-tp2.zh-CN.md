# O10：MiniMax H3 双 V100 Text Encoder TP2 计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.
>
> **状态：用户已批准；O9 前置依赖已满足，开始 Task 1。**
>
> **执行要求：**按任务顺序执行；每个任务先写失败测试，再做最小实现，再完成审查和提交。不得直接修改部署目录，所有代码只在 E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p 修改，通过现有部署脚本安装到 ComfyUI。
>
> **前置依赖状态（2026-08-31）：已满足。**O9 Windows CUDA P2P 自适应分块的 Gate A–F 已全部通过，1.11.0 已提交、部署并完成真实工作流与最终 smoke。本计划解除阻塞，可以从 Task 1 基线审计开始；O10 自身的 Gate A–F 仍必须逐级通过。

**目标：**让 MiniMax H3 的 Qwen3-VL-32B Text Encoder 在两张 Tesla V100-SXM2-16GB 上分别保存约一半语言模型权重，并在每个语言层中并发计算 Attention 和 MLP，降低 GPU0 单卡权重峰值、减少 CPU offload，并缩短进入采样前的 Text Encoder 阶段。

**推荐架构：**在 ComfyUI 主进程中实现固定两卡的 Tensor Parallel 2（TP2）。视觉塔在 GPU0 运行；Qwen3-VL 语言层的 Q/K/V、Attention 输出投影和 MLP 权重按头或通道拆到 GPU0/GPU1。两卡并发计算局部结果，通过 TCC + CUDA P2P/NVLink 做两卡求和，不经 Gloo、CPU 内存或页面文件传递正式 CUDA payload。

**固定技术栈：**原生 Windows、2× Tesla V100-SXM2-16GB、TCC、驱动 577.00、Python 3.10.11、PyTorch 2.7.0+cu126、xformers 0.0.30、Ray 2.57.0、ComfyUI v0.31.0-15-g62b3c94b、Raylight Windows CUDA P2P ProcessGroup。

**Spec：**本文件第 1 至 12 节同时构成本阶段设计规格；第 13 节以后是可执行实施任务。

## 全局约束

- 首个正式支持对象固定为 qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors。
- 首个版本只支持恰好两张 V100、TCC、同机 CUDA P2P；不研究三卡及更多 GPU。
- 不修改 NVIDIA 驱动，不升级当前 PyTorch/CUDA/Python 主环境。
- 不修改 ComfyUI 核心源码，不修改 MiniMaxH3Director 源码。
- 不改变现有 RayInitializer、RayUNETLoader、Ray MODEL Adapter、采样器和工作流节点的公开接口。
- 新路径必须显式选择；旧 CLIPLoader 和旧工作流保持原样并作为可靠回退。
- Text Encoder 的正式 CUDA 张量不得经 Gloo、Ray object store、CPU RAM 或页面文件在两卡之间中转。
- 任何 P2P、数值、内存租约或权重布局校验失败都必须明确报错；不得静默回退到“看似双卡、实际单卡”。
- GPU0/GPU1 在 Text Encoder 完成后必须释放 TP2 GPU 权重和临时缓存，之后才能启动或恢复 Raylight Diffusion 采样。
- 现有 LTX、MiniMax H3 I2V/REF2VA、O6 safe-FP16、MMH3Director 与 RayKill 行为不得回归。
- GGUF、Text Encoder FSDP、VAE 多卡、训练反向传播和跨机器通信不在 O10 范围。

---

## 1. 当前问题与已确认事实

当前 MiniMax H3 Text Encoder 在 ComfyUI 主进程的 GPU0 执行。NORMAL_VRAM 路径不是严格逐层换入，而是先尽可能把权重常驻 GPU0，再只对剩余权重做 offload。

现有日志样本：

    loaded partially;
    9071.21 MB loaded,
    5888.99 MB offloaded,
    710.94 MB buffer reserved

这说明约 9.1 GB 权重在整个编码阶段常驻 GPU0，约 5.9 GB 通过 CPU offload 换入。Qwen3-VL 没有按输入图片数量、视觉 Token 和 DeepStack 大小提供显存估算，因此固定 ReserveVram 可能低估真实峰值。

O10 不把“调大 ReserveVram”视为最终修复，也不把“前半层放 GPU0、后半层放 GPU1”误称为并行计算。目标是让语言主干的大部分矩阵计算在两张卡上同时发生。

## 2. 三种候选路线

### 2.1 推荐：主进程 Tensor Parallel 2

每个语言层都拆分到两张卡：

- Q/K/V 按 Attention 头拆分，两卡分别计算一半头。
- Attention 在各自本地头上执行。
- 输出投影按输入通道拆分，两卡得到局部输出，再通过 P2P 求和。
- MLP gate/up 按中间通道拆分。
- MLP down 按输入通道拆分，两卡局部输出再 P2P 求和。
- RMSNorm、残差、小型标量和位置编码在两卡复制。

优点：

- 两卡在主要语言层同时计算，满足本阶段并行目标。
- 每张卡只保存约一半语言层大矩阵权重。
- 同一进程内可直接使用 CUDA P2P copy、stream 和 event，不需要为 Text Encoder 新建 Gloo ProcessGroup。
- 输出仍可包装成 ComfyUI CLIP 对象，兼容普通工作流和 MMH3Director。

代价：

- 每个语言层需要两次左右的跨卡求和，50 层约 100 次集合通信。
- NVFP4 权重必须按量化块边界正确切分，不能先完整解量化再切。
- 需要独立的双 GPU 生命周期租约，避免主进程 TP2 与 Ray worker 同时占用同一显存。

### 2.2 降级候选：连续层分段 Pipeline

GPU0 保存前半层，GPU1 保存后半层，隐藏状态只跨卡一次。

优点是通信少、实现短、每卡权重约减半；缺点是单条提示词的前后层存在严格依赖，GPU0 和 GPU1 大部分时间不能同时做语言层计算。只有多提示词或多段微批次时才能形成有限流水。

该路线只在 TP2 因 NVFP4 布局或通信开销无法达到验收线时启用，发布名称必须标为“层分段容量模式”，不得宣称双卡并行加速。

### 2.3 不采用：Text Encoder FSDP

FSDP 会在每层前把完整权重聚合出来，两个 rank 通常还会重复计算相同 batch=1 的 Text Encoder 前向。它能降低静态分片权重，却增加频繁 all-gather，不能自然带来本任务要求的并行算力收益。

## 3. 对外节点与兼容契约

新增独立节点，不修改现有节点参数：

- 内部节点 ID：RayMiniMaxH3DualGPUCLIPLoader
- 显示名：Ray Dual-GPU Text Encoder Loader (MiniMax H3, TP2)
- 输入：
  - ray_actors_init: RAY_ACTORS_INIT
  - clip_name: models/text_encoders 中的模型文件
  - device_0: 默认 0
  - device_1: 默认 1
  - execution_mode: tp2 或 single_gpu_fallback
  - unload_gpu_after_encode: 固定默认 true
- 输出：
  - clip: CLIP
  - ray_actors_init: 原类型透传，用于强制工作流拓扑顺序

执行模式说明：

- tp2：严格要求两张 TCC V100、CUDA peer access、受支持的 MiniMax H3 NVFP4 checkpoint。
- single_gpu_fallback：显式使用现有 ComfyUI CLIP 加载路径，只用于用户主动选择和对照测试。
- tp2 发生错误时不自动切换 single_gpu_fallback。

CLIP 代理保持以下 ComfyUI 调用契约：

- tokenize(text: str, **kwargs) -> dict
- encode_from_tokens_scheduled(tokens: dict, add_dict: dict | None = None) -> list
- clone() -> MiniMaxH3TP2ClipProxy
- add_patches(patches: dict, strength_patch: float = 1.0, strength_model: float = 1.0) -> None

首版只承诺当前 MiniMax H3 Text Encoder 的 tokenizer、图像/视频 reference、minimax_token_tags 和标准 CLIP LoRA 契约。无法正确拆分的 Text Encoder LoRA 必须在加载前拒绝，不允许只在一个 rank 生效。

## 4. 运行时数据流

推荐工作流连接：

    RayInitializer
        -> Ray Dual-GPU Text Encoder Loader
             -> CLIP -> MiniMax H3 / MMH3Director
             -> RAY_ACTORS_INIT -> RayUNETLoader
        -> Ray MODEL Adapter / XFuser sampler

一次 Text Encoder 调用：

1. CLIP 代理根据 token、图片和 reference 元数据生成唯一 operation_id。
2. 向现有 Ray workers 申请“主进程双 GPU 租约”。
3. 两个 worker 同步当前 CUDA stream，释放采样临时张量和可释放 GPU shard，返回各自可用显存。
4. 主进程验证 GPU0/GPU1 peer access、空闲显存和租约 operation_id。
5. 从 safetensors/mmap CPU 权重直接构造两张卡的 NVFP4 分片；不创建完整 FP16/FP32 权重副本。
6. GPU0 完成 tokenizer 后的 embedding 与 Qwen3-VL 视觉塔；视觉 embedding、DeepStack 和 token tags 复制到 GPU1。
7. 50 个语言层执行 TP2。每层两卡并发执行局部 QKV、Attention 和 MLP；输出投影与 down 投影使用 P2P 求和恢复两卡一致的隐藏状态。
8. 最终 conditioning 按 ComfyUI 当前 intermediate_device 契约返回，形状、dtype、minimax_token_tags 与单卡路径一致。
9. 主进程同步两卡，卸载 TP2 GPU 权重，清空临时 CUDA cache。
10. 释放租约；Ray workers 只有在收到同一 operation_id 的成功释放后才能进入采样。

异常路径必须执行第 9、10 步，并把原始错误返回给 ComfyUI。

## 5. TP2 数学划分

设 hidden_size=5120、tp_degree=2。

### 5.1 Attention

- RMSNorm：两卡复制，输入 hidden state 在两卡一致。
- q_proj/k_proj/v_proj：按输出行和 Attention 头拆分。
- 每卡持有一半 Q 头和一半 KV 头；GQA 的 head 分组必须保持完整，不允许切开单个 head。
- RoPE：每卡只处理本地头。
- SDPA：每卡独立处理本地头，使用 V100 已验证的 efficient attention。
- o_proj：按输入列拆分；每卡产生形状为 batch × sequence × 5120 的 partial。
- residual 前执行两卡 SUM，求和结果在两卡复制。

### 5.2 MLP

- gate_proj/up_proj：按输出行和中间通道拆分。
- 激活函数在各卡本地执行。
- down_proj：按输入列拆分，各卡产生完整 hidden_size 的 partial。
- residual 前执行两卡 SUM。

### 5.3 Embedding、视觉塔与小参数

- token embedding 首版放 GPU0，输出一次性复制到 GPU1。
- 视觉塔首版完整放 GPU0，避免在 O10 同时开发第二套视觉并行。
- DeepStack merger 放 GPU0；其输出复制到 GPU1。
- RMSNorm、位置编码参数和小型 bias 在两卡复制。
- 根据实测权重平衡，允许把 token embedding 迁到 GPU1，但不得改变公开节点接口。

## 6. CUDA P2P 求和

新增固定两卡、单进程的 CudaPeerGroup2，不复用 Gloo。

接口：

- PeerCollectiveStats(bytes_sent: int, elapsed_ms: float, operation_id: int)
- CudaPeerGroup2.validate() -> None
- CudaPeerGroup2.all_reduce_sum(rank0_partial: torch.Tensor, rank1_partial: torch.Tensor, operation_id: int) -> tuple[torch.Tensor, torch.Tensor, PeerCollectiveStats]

实现约束：

- 两张卡各保留一个输出缓冲和一个 peer 接收缓冲。
- 使用独立 CUDA stream 与 event；两个方向复制必须并发提交。
- 两卡只接收对方 partial 并本地相加，不构造 world_size 倍 all-gather 大缓冲。
- 支持 FP16 和 FP32；BF16 在 V100 上只作为存储兼容，不作为正式计算 dtype。
- 输入 shape、dtype、contiguous、device、operation_id 不一致时立即失败。
- 默认不打印逐层日志；诊断开关只输出聚合次数、总字节、累计通信时间和最慢层。

## 7. NVFP4 分片

新增只读分片器，直接处理 TensorCoreNVFP4Layout 的 qdata、weight_scale 和 weight_scale_2。

接口：

- QuantizedLinearShard(axis: Literal["row", "column"], logical_shape: tuple[int, int], rank: int, world_size: int, qweight: QuantizedTensor)
- shard_nvfp4_linear(weight: QuantizedTensor, axis: Literal["row", "column"], rank: int, world_size: int = 2) -> QuantizedLinearShard

验收要求：

- 分片边界满足 NVFP4 量化块、scale 和 block_scale 对齐。
- qdata 与所有 scale 采用同一逻辑切分。
- 分片构造过程不得创建完整 FP16/FP32 权重。
- 两个分片独立解量化再拼接，必须与完整权重解量化结果在数值上等价。
- 不支持的 shape 在模型加载阶段报出具体参数名、shape 和违反的对齐条件。

## 8. Worker 显存租约

新增协议，不复用 RayKill：

    @dataclass(frozen=True)
    class ExternalGpuLease:
        operation_id: str
        owner: str
        free_bytes: int
        worker_state: str

RayWorker 新增：

- enter_external_gpu_lease(operation_id: str, owner: str) -> dict
- exit_external_gpu_lease(operation_id: str, owner: str) -> dict
- get_external_gpu_lease() -> dict | None

状态规则：

- idle -> leased_for_text_encoder -> idle。
- sampling、loading_diffusion、loading_lora 状态不得授予租约。
- 两个 worker 必须全部授予同一 operation_id 才算成功。
- 只成功一个 worker 时，协调器必须撤销已授予租约。
- 租约期间任何采样请求必须明确失败并指出占用方。
- 异常、ComfyUI 中断和节点取消都必须在 finally 中释放租约。
- 租约释放不销毁 worker，不丢失已验证可复用的 CPU/mmap Diffusion state。

## 9. 缓存与多段 Director

首版缓存策略：

- tokenizer/config 常驻主进程。
- NVFP4 CPU/mmap 分片描述和文件映射可复用。
- TP2 GPU 权重每次 encode 后强制释放，不允许与 Ray Diffusion 采样重叠。
- 相同 checkpoint 的后续 segment 复用 CPU/mmap 描述，重新把两个分片搬到两卡。
- conditioning 缓存键必须包含 checkpoint hash、prompt、reference 文件内容 hash、尺寸、裁剪方式和 LoRA 状态。
- 默认不缓存包含可变 reference 视频帧的 conditioning，除非所有输入 hash 完全一致。

不得通过保留两卡 GPU 权重来伪造热启动速度，因为这会重新引入采样 OOM。

## 10. 回退与失败边界

TP2 禁用条件：

- 任意 GPU 不是 CUDA、不是 V100、不是 TCC 或 peer access 失败。
- world size 不是 2。
- checkpoint 不是 MiniMax H3 Qwen3-VL-32B NVFP4/AWQ。
- NVFP4 参数 shape 不满足分片对齐。
- Text Encoder LoRA 不能在两卡以相同语义拆分。
- 两个 worker 未能同时授予显存租约。
- 任意 rank 输出 NaN/Inf、shape 不一致或 P2P operation_id 不一致。

禁用时给出可操作提示，用户可以手动把节点切换到 single_gpu_fallback。不得自动改变工作流语义。

## 11. 验收口径

所有性能比较使用本机现有单卡 Qwen3-VL 路径作为基线，不引用第三方机器数据。

### 11.1 基线冻结

在开发前记录同一 checkpoint、输入、prompt、reference、缓存状态：

- Text Encoder 模型准备时间。
- 视觉预处理时间。
- 视觉塔时间。
- 50 层语言主干时间。
- conditioning 输出和清理时间。
- 进入采样前总时间。
- GPU0/GPU1 显存峰值、GPU 利用率、功率。
- 物理内存、进程私有提交和系统页面文件峰值。
- 冷启动 3 次、热启动 3 次，使用中位数。

### 11.2 数值正确性

- 两卡输出 shape、dtype、token count 和 minimax_token_tags 与单卡完全一致。
- 全部 conditioning 有限，无 NaN/Inf。
- FP16 TP2 对单卡同 checkpoint 输出：cosine similarity 不低于 0.9995。
- 合成单层 FP32 对照：max_abs_error 不高于 1e-4。
- NVFP4 分片解量化拼接与完整解量化逐元素一致。
- 两卡每层 all-reduce 后隐藏状态彼此一致。

### 11.3 显存与内存

- TP2 编码期间任意单卡显存峰值不超过 14.0 GiB。
- GPU0 峰值相对单卡基线至少下降 25%。
- 每张卡的语言大矩阵权重偏差不超过总语言大矩阵权重的 5%。
- Text Encoder 完成并释放租约后，两卡不得残留 TP2 模型权重；只允许 CUDA context 和已知 Ray worker 基础占用。
- 相对单卡基线，主进程 commit 峰值增长不超过 20%。
- 测试期间页面文件 used delta 不超过 1.0 GiB；超过时判定缓存/复制策略失败。

### 11.4 并行与性能

- 语言主干期间两张卡必须存在重叠计算，不能是 GPU0 完成后才启动 GPU1。
- 使用 CUDA event 证明至少 80% 的语言层中，GPU0/GPU1 局部 Linear 时间区间发生重叠。
- nvidia-smi 采样中，两张卡同时处于非空闲计算状态的样本占语言主干窗口至少 60%。
- TP2 Text Encoder 热启动中位数不高于单卡热启动中位数的 80%。
- TP2 进入采样前总时间中位数不高于单卡对应阶段的 85%。
- 如果显存门槛通过但速度门槛失败，只能发布为实验性“容量模式”，不得设为推荐默认值。

### 11.5 端到端质量

- 普通 MiniMax H3 I2V Turbo8 完成一次。
- 普通 MiniMax H3 REF2VA Turbo4 完成一次。
- MMH3Director 完成 3 秒、480p、24 FPS、4 步单段视频。
- MMH3Director 完成 5 秒、960×544、24 FPS、4 步单段视频。
- 至少一次 5 段连续生成，用于验证重复租约、CPU/mmap 复用和无逐段显存增长。
- 视频无全黑、首帧后噪声、固定乱码、NaN/Inf；音频可解码。
- 固定 seed 的 TP2 成品与单卡基线进行盲看，人物身份、动作、镜头指令和 reference 一致性不得出现明显退化。

### 11.6 回归

- 当前完整单元测试通过。
- Windows P2P、FSDP2、O6 safe-FP16、MMH3Director lifecycle 和 deployment 测试通过。
- 旧工作流不替换 CLIPLoader 时，节点图、输出和性能保持在既有回归阈值内。
- RayKill 后 Ray workers、CUDA context、VRAM 和大部分进程 commit 正确回收。

## 12. 阶段决策门

- Gate A：单进程双 V100 P2P SUM 合成测试通过，才进入 NVFP4。
- Gate B：NVFP4 row/column 分片无需完整解量化且数值正确，才进入 Qwen3-VL。
- Gate C：单个 Qwen3-VL block 的 TP2 输出通过数值门槛，才加载完整 50 层。
- Gate D：真实 conditioning 通过数值、显存和租约门槛，才运行视频。
- Gate E：端到端视频通过质量和回归门槛，才做性能定级。
- Gate F：性能达到 20% Text Encoder 加速，才标为推荐模式；否则保留为实验性容量模式或转向层分段 Pipeline。

任一 Gate 失败都保留证据并停止扩大改动，不通过放宽误差、显存或性能标准伪造成功。

---

## 13. 文件结构

计划新增：

- src/raylight/text_encoder_parallel/__init__.py：导出 TP2 节点与公共类型。
- src/raylight/text_encoder_parallel/topology.py：双 V100、TCC、peer access 和设备校验。
- src/raylight/text_encoder_parallel/collectives.py：单进程 CudaPeerGroup2 和统计。
- src/raylight/text_encoder_parallel/nvfp4_sharding.py：NVFP4 row/column 分片。
- src/raylight/text_encoder_parallel/linear.py：ColumnParallelLinear 与 RowParallelLinear。
- src/raylight/text_encoder_parallel/qwen3vl_tp.py：TP2 Attention、MLP、语言 block 和完整 50 层主干。
- src/raylight/text_encoder_parallel/clip_proxy.py：ComfyUI CLIP 代理、tokenizer、视觉塔和 conditioning 返回。
- src/raylight/text_encoder_parallel/nodes.py：新节点 schema 与显式 fallback。
- src/raylight/text_encoder_parallel/lifecycle.py：两 worker 租约协调、GPU 权重装卸和异常回收。
- tests/test_text_encoder_tp_topology.py
- tests/test_text_encoder_tp_collectives.py
- tests/test_text_encoder_tp_nvfp4.py
- tests/test_text_encoder_tp_linear.py
- tests/test_text_encoder_tp_qwen3vl.py
- tests/test_text_encoder_tp_clip_proxy.py
- tests/test_text_encoder_tp_lifecycle.py
- tests/test_text_encoder_tp_nodes.py
- tests/windows_minimax_h3_text_encoder_tp2_probe.py
- scripts/benchmark-minimax-h3-text-encoder-tp2.ps1
- example_workflows/Minimax_H3_DualGPU_Text_Encoder_TP2_Raylight.json
- docs/testing/minimax-h3/DUAL_GPU_TEXT_ENCODER_TP2_2026-08.zh-CN.md

计划修改：

- __init__.py：注册新的 TP2 节点映射，更新版本号只在发布任务执行。
- src/raylight/distributed_worker/ray_worker.py：增加显存租约状态和三个远程方法。
- src/raylight/distributed_worker/sampling_lifecycle.py：采样前拒绝未释放租约，统一清理入口。
- src/raylight/nodes.py：Ray worker session 暴露租约协调所需的稳定引用，不改变旧节点输入输出。
- tests/test_mmh3director_lifecycle.py：增加多段 Director 与 TP2 租约回归。
- tests/test_ray_worker_lifecycle.py：增加租约、取消和异常状态转换。
- README.md 与 README.zh-CN.md：验收完成后加入能力、限制、连接方式和故障回退。
- docs/releases 下的中英文发布说明：只在 Gate F 定级后创建。

## 14. Task 1：冻结单卡基线与阶段计时

**Files:**

- Create: scripts/benchmark-minimax-h3-text-encoder-tp2.ps1
- Create: tests/test_text_encoder_tp_benchmark_contract.py
- Create: docs/testing/minimax-h3/DUAL_GPU_TEXT_ENCODER_TP2_2026-08.zh-CN.md

**Produces:**

- BaselineRecord 数据结构和 JSON/CSV 字段契约。
- 单卡冷 3 次、热 3 次基线。

- [x] 写失败测试，要求解析器拒绝缺少 checkpoint、token_count、te_total_s、gpu0_peak_mib、ram_peak_mib 或 pagefile_peak_mib 的记录。
- [x] 运行 python -m pytest tests/test_text_encoder_tp_benchmark_contract.py -q，确认测试因实现不存在而失败。
- [x] 实现只读基线记录器，阶段计时默认关闭，基准脚本显式开启。
- [x] 使用当前标准 CLIPLoader 和现有 MMH3Director 输入完成冷 3 次、热 3 次。
- [x] 把中位数、峰值和工作流 hash 写入测试汇总，不复制长篇控制台日志。
- [x] 运行测试并提交：test: freeze MiniMax H3 text encoder baseline（`526a9f6`）。

Task 1 实机结果：冷中位数 20.76 秒，热中位数 10.37 秒，token 数固定 221，GPU 0 峰值 13,906 MiB；详见 `docs/testing/minimax-h3/DUAL_GPU_TEXT_ENCODER_TP2_2026-08.zh-CN.md`。Task 2 可以开始。

## 15. Task 2：双 GPU P2P SUM 原语

**Files:**

- Create: src/raylight/text_encoder_parallel/topology.py
- Create: src/raylight/text_encoder_parallel/collectives.py
- Create: tests/test_text_encoder_tp_topology.py
- Create: tests/test_text_encoder_tp_collectives.py
- Create: tests/windows_minimax_h3_text_encoder_tp2_probe.py

**Interfaces:**

- Produces: CudaPeerGroup2.validate() 和 CudaPeerGroup2.all_reduce_sum(rank0_partial, rank1_partial, operation_id)。

- [x] 写 CPU 级失败测试，覆盖设备错误、shape/dtype 不一致、operation_id 重复、缓冲复用和统计聚合。
- [x] 运行对应 pytest，确认失败原因是类和接口尚不存在。
- [x] 实现设备无关的 copy plan 和状态机，再实现 CUDA stream/event 后端。
- [x] 运行单元测试，确认关闭诊断时不生成逐层日志。
- [x] 运行双 V100 探针，覆盖 1 MiB、16 MiB、64 MiB，FP16/FP32，各 100 次。
- [x] 验证输出与 torch.add 基线一致、正式 payload 不经过 Gloo/CPU、无逐次显存增长。
- [x] 提交：feat: add two-GPU CUDA peer sum collective（`d90a4ae`）。

Task 2 / Gate A 结果：六种组合全部逐元素一致且 finite；100 次正式迭代无正向显存增长。16 MiB 聚合带宽为 58.45–59.32 GiB/s，64 MiB 为 82.09–82.39 GiB/s。正式数据面只执行跨设备 CUDA `copy_` 与各卡本地 `torch.add`，未建立 ProcessGroup、Gloo 或 CPU payload 缓冲。Gate A 通过，Task 3 可以开始。

## 16. Task 3：NVFP4 row/column 分片

**Files:**

- Create: src/raylight/text_encoder_parallel/nvfp4_sharding.py
- Create: tests/test_text_encoder_tp_nvfp4.py

**Interfaces:**

- Consumes: comfy_kitchen TensorCoreNVFP4Layout。
- Produces: QuantizedLinearShard 和 shard_nvfp4_linear(weight, axis, rank, world_size=2)。

- [x] 用真实 checkpoint 中四种代表 shape 写失败测试：QKV row shard、o_proj column shard、MLP up/down shard。
- [x] 运行测试，确认因分片模块不存在而失败。
- [x] 实现 qdata、weight_scale、weight_scale_2 的一致切分和对齐校验。
- [x] 验证构造分片期间没有完整 FP16/FP32 临时张量。
- [x] 解量化两个分片并重建逻辑权重，逐元素比较完整解量化基线。
- [x] 加入不对齐 shape 的明确错误消息测试。
- [x] 提交：feat: shard MiniMax H3 NVFP4 text encoder weights（`9aeb65c`）。

Task 3 / Gate B 结果：真实 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` 的 layer 0 `q_proj`、`o_proj`、`up_proj`、`down_proj` 均通过。两个 rank 的反量化结果分别与完整权重对应切片逐元素一致且 finite；两分片 qdata 与 block-scale 的总字节数严格等于源权重，不存在数据复制膨胀。分片器只解开和重排小型 FP8 block-scale 表，构造期间不会调用完整权重反量化。Gate B 通过，Task 4 可以开始。

## 17. Task 4：并行 Linear 与单层数学正确性

**Files:**

- Create: src/raylight/text_encoder_parallel/linear.py
- Create: tests/test_text_encoder_tp_linear.py

**Interfaces:**

- Consumes: QuantizedLinearShard、CudaPeerGroup2。
- Produces: ColumnParallelLinear.forward(x) 和 RowParallelLinear.forward(x, operation_id)。

- [ ] 写 dense FP32 小矩阵失败测试，要求 TP2 输出匹配完整 Linear。
- [ ] 写 NVFP4 小矩阵失败测试，要求输出满足正式 cosine 和 finite 门槛。
- [ ] 实现两设备并发 kernel 提交；RowParallelLinear 使用 P2P SUM。
- [ ] 用 CUDA event 证明两张卡局部 Linear 时间区间重叠。
- [ ] 运行单元测试和双 V100 探针。
- [ ] 提交：feat: add TP2 linear primitives。

## 18. Task 5：Qwen3-VL 语言 block TP2

**Files:**

- Create: src/raylight/text_encoder_parallel/qwen3vl_tp.py
- Create: tests/test_text_encoder_tp_qwen3vl.py

**Interfaces:**

- Produces: TP2Qwen3VLBlock.forward(hidden_rank0: torch.Tensor, hidden_rank1: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor | None, operation_id: int) -> tuple[torch.Tensor, torch.Tensor]

- [ ] 从 ComfyUI 当前 Qwen3-VL block 冻结 forward 参数和 GQA 头配置测试。
- [ ] 写单 block FP32/NVFP4 对照失败测试。
- [ ] 实现 replicated RMSNorm/residual、head-sharded Attention 和 channel-sharded MLP。
- [ ] 验证每层正好两次 P2P SUM，operation_id 单调且两卡 hidden 一致。
- [ ] 扩展到 2 层、8 层、50 层合成模型，检查误差累积、显存和通信统计。
- [ ] 提交：feat: add Qwen3-VL TP2 language blocks。

## 19. Task 6：完整 MiniMax H3 CLIP 代理

**Files:**

- Create: src/raylight/text_encoder_parallel/clip_proxy.py
- Create: tests/test_text_encoder_tp_clip_proxy.py

**Interfaces:**

- Consumes: TP2Qwen3VLBlock、现有 MiniMaxH3Tokenizer 和视觉处理逻辑。
- Produces: MiniMaxH3TP2ClipProxy。

- [ ] 写纯文本、单图 I2V、双图 REF2VA、视频 reference 的 token/shape/tag 契约失败测试。
- [ ] 实现 tokenizer/config 复用、GPU0 视觉塔和两卡语言主干。
- [ ] 保持 minimax_token_tags、DeepStack 注入和 conditioning metadata。
- [ ] 对真实 checkpoint 运行 encoder-only，比较单卡输出 cosine、finite 和 shape。
- [ ] 验证 encode 异常后两卡 GPU 权重被释放。
- [ ] 提交：feat: add MiniMax H3 TP2 CLIP proxy。

## 20. Task 7：Worker 双 GPU 显存租约

**Files:**

- Create: src/raylight/text_encoder_parallel/lifecycle.py
- Modify: src/raylight/distributed_worker/ray_worker.py
- Modify: src/raylight/distributed_worker/sampling_lifecycle.py
- Create: tests/test_text_encoder_tp_lifecycle.py
- Modify: tests/test_ray_worker_lifecycle.py

**Interfaces:**

- Produces: ExternalGpuLease、DualGpuLeaseCoordinator.acquire()/release() 和 RayWorker 三个远程方法。

- [ ] 写状态机失败测试：idle、loading、sampling、leased、取消、单 rank 失败、重复释放。
- [ ] 实现 worker 锁、stream synchronize、clear_sampling_vram 和 free-memory 回执。
- [ ] 实现协调器的全有或全无租约。
- [ ] 注入编码异常和用户取消，验证 finally 释放。
- [ ] 验证租约释放后原 Diffusion CPU/mmap state 可继续采样，不重建无关 worker。
- [ ] 提交：feat: coordinate dual-GPU text encoder leases。

## 21. Task 8：节点、工作流与 Director 接入

**Files:**

- Create: src/raylight/text_encoder_parallel/nodes.py
- Modify: src/raylight/text_encoder_parallel/__init__.py
- Modify: __init__.py
- Modify: src/raylight/nodes.py
- Create: tests/test_text_encoder_tp_nodes.py
- Modify: tests/test_mmh3director_lifecycle.py
- Create: example_workflows/Minimax_H3_DualGPU_Text_Encoder_TP2_Raylight.json

**Interfaces:**

- Produces: RayMiniMaxH3DualGPUCLIPLoader 节点和 CLIP/RAY_ACTORS_INIT 输出。

- [ ] 写节点 schema、默认值、显示名和旧节点冻结测试。
- [ ] 实现显式 tp2/single_gpu_fallback，不允许自动回退。
- [ ] 建立普通 I2V/REF2VA 和 MMH3Director 的最少改线工作流。
- [ ] 验证 Director 的 seed、steps、FPS、duration、resolution、prompt groups 等仍由 Director 控制。
- [ ] 验证 Text Encoder 只运行一次，Director 不会再次本地编码或采样。
- [ ] 提交：feat: expose MiniMax H3 dual-GPU text encoder node。

## 22. Task 9：真实显存、数值和多段稳定性

**Files:**

- Modify: scripts/benchmark-minimax-h3-text-encoder-tp2.ps1
- Modify: docs/testing/minimax-h3/DUAL_GPU_TEXT_ENCODER_TP2_2026-08.zh-CN.md

- [ ] encoder-only 冷 3 次、热 3 次，记录全部 11.1 指标。
- [ ] 运行 I2V Turbo8 和 REF2VA Turbo4。
- [ ] 运行 Director 3 秒 480p、5 秒 960×544。
- [ ] 运行 5 段压力测试，绘制每段 GPU0/GPU1、RAM、commit、pagefile 峰值。
- [ ] 对所有 conditioning 做 shape、tag、finite、cosine 检查。
- [ ] 对视频执行 ffprobe、唯一帧 hash、黑帧/噪声检查和固定 seed 盲看。
- [ ] 任何门槛失败时停止，不进入性能定级。
- [ ] 提交：test: validate MiniMax H3 TP2 text encoder end to end。

## 23. Task 10：性能定级与失败路线决策

**Files:**

- Modify: docs/testing/minimax-h3/DUAL_GPU_TEXT_ENCODER_TP2_2026-08.zh-CN.md

- [ ] 汇总两卡 kernel overlap、P2P 次数/字节/时间、视觉塔时间、语言层时间和清理时间。
- [ ] 计算 TP2 相对单卡冷/热中位数，不使用最佳单次。
- [ ] 若达到 Text Encoder 至少 20% 加速且所有质量门槛通过，定级为推荐模式。
- [ ] 若只通过显存门槛，定级为实验性容量模式，默认关闭。
- [ ] 若 TP2 慢于单卡或通信占语言层时间超过 35%，停止 TP2 发布，按 2.2 设计层分段 Pipeline 的独立后续计划。
- [ ] 提交：docs: classify dual-GPU text encoder performance。

## 24. Task 11：回归、文档、部署与发布

**Files:**

- Modify: README.md
- Modify: README.zh-CN.md
- Create: docs/releases/2026-08-29-dual-v100-text-encoder-tp2.en.md
- Create: docs/releases/2026-08-29-dual-v100-text-encoder-tp2.zh-CN.md
- Modify: __init__.py

- [ ] 运行完整 pytest 与 Windows P2P/FSDP/Director 关键回归。
- [ ] 用 scripts/deploy-to-comfyui.ps1 -VerifyOnly 验证部署前 source of truth。
- [ ] 更新中英文 README：支持范围、节点连接、显存租约、限制、fallback 和实际验收数据必须一致。
- [ ] 发布说明明确区分“语言层 TP2”“GPU0 视觉塔”和“Diffusion FSDP/Ulysses”，不得写成统一 32GB 显存池。
- [ ] 更新版本号并提交。
- [ ] 部署到 E:\ComfyUI-py310\ComfyUI\custom_nodes\raylight，再运行部署一致性和一次端到端烟雾测试。
- [ ] 用户明确要求后才推送 GitHub。

## 25. 计划自审

- 无待定接口：节点名、输入输出、核心类和验收门槛均已明确。
- 无范围混淆：只做 MiniMax H3 Qwen3-VL-32B NVFP4 Text Encoder TP2。
- 无“假双卡”：必须提供 CUDA event 重叠证据和双卡同时计算采样。
- 无“格式即调度”假设：NVFP4 负责权重存储，显存租约和 TP2 负责执行与生命周期。
- 无兼容性偷换：旧 CLIPLoader、现有 Raylight 和 Director 路径保持可用。
- 无提前发布：Gate F 未通过时只能标为实验性容量模式。
