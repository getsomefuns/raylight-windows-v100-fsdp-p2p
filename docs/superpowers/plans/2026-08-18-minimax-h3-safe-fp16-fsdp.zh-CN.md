# Raylight FSDP 上的 MiniMax H3 安全 FP16 实施计划

[简体中文](2026-08-18-minimax-h3-safe-fp16-fsdp.zh-CN.md) | [English](2026-08-18-minimax-h3-safe-fp16-fsdp.md)

> **面向执行代理：**按任务逐项实施本计划时，建议使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`。任务状态使用复选框（`- [ ]`）跟踪。

**目标：**在保留 FP8 checkpoint 存储、CUDA P2P 和已验收 FP32 工作流的前提下，为原生 Windows 双 V100 Raylight FSDP 增加可选、数值安全的 MiniMax H3 FP16 计算路径。

**架构：**把外部 MIT 补丁移植成独立的 Raylight 模块，在每个 Ray actor 构造 MiniMax H3 模型及进行 FSDP 包装之前幂等安装。条件投影、残差流和输出头保持 FP32，只把 Attention/MLP 分支输入转换为 FP16；FSDP 仍保存 FP8 权重，并在 V100 上分块反量化。通过专用 RayUNETLoader 模式公开该功能，保证通用 FP16 和所有已验收基线工作流不变。

**技术栈：**Windows 23H2、2× Tesla V100-SXM2-16GB TCC、Python 3.10.11、PyTorch 2.7.0+cu126、ComfyUI 0.31.0、Ray 2.57.0、Raylight FSDP2、yunchang/USP、comfy-kitchen FP8、Windows CUDA P2P。

**规格来源：**`docs/superpowers/plans/2026-08-17-minimax-h3-memory-performance.md#phase-o6-safe-fp16-compute-for-minimax-h3-on-raylight-fsdp`

## 全局约束

- 只在 `E:\ComfyUI-py310` 内工作；不修改 NVIDIA 577.00 驱动，也不修改该目录之外的环境。
- 保留 `--disable-cuda-malloc --reserve-vram 2`、两个 FSDP worker、CPU offload、Ulysses 2、Ring 1 和 Windows CUDA P2P。
- 不在 ComfyUI `MiniMaxH3.supported_inference_dtypes` 中全局加入 FP16，也不覆盖已验收的 O5 工作流。
- 不把 FP8 模型、BF16 Turbo LoRA、模型权重、原始日志和生成媒体提交到 Git。
- 适配代码时注明外部 MIT 项目来源，并保留其许可证声明。
- 在参考双 V100 机器复现之前，所有速度数字都只能视为假设。
- 创建 O6 代码之前，先把已验收的 O1–O5 实现和文档同步到 GitHub。

## 证据基线

### 已在本机确认

- 外部来源：`Amduraznak/minimax-h3-fp16-fix`，本地/远端提交 `b09897c`，MIT 许可证；作者在 ComfyUI 0.30 和单张 32GB V100 上测试。
- 外部模块可在当前 ComfyUI 0.31 进程中导入，`MiniMaxH3Model`、`DiTBlock` 和 `MLP` 的名称与签名仍匹配。
- O5 I2V/REF2VA 使用 `minimax_h3_*_pruned_fp8_scaled.safetensors`，`weight_dtype=default`。
- 已验收日志包含 `[Raylight][comfy_kitchen][fp8] fallback dequantize dtype=torch.float32`；现有测试明确描述 V100 FP32 计算路径。
- Ray actor 通过 `py_modules` 获取 Raylight、通过 `PYTHONPATH` 获取 ComfyUI，但不会扫描任意 `ComfyUI/custom_nodes`；因此外部项目的普通 custom-node 安装方式无法进入 worker。
- MiniMax USP 会替换各 Attention forward 和模型 `_forward`，但仍调用 `block(...)`、`block.attn.out_proj(...)` 和现有 final layer。只要在模型构造前安装，安全 FP16 数学可以与之共存。

### 需要验证的预测

- FP8 存储和 P2P 参数传输应保持不变，因为 V100 fallback 根据分支输入张量的 dtype 选择反量化矩阵 dtype。
- FP32 残差可以避免 50 个 block 累积时溢出；FP16 分支输入则让 V100 Tensor Core 执行主要 QKV/out-projection 和 MLP 矩阵乘法。
- Raylight `LoRAAdapter.h()` 会把 sidecar 张量转换为 `x.dtype`，理论上可保留 LoRA 正确性；但重复转换开销和内存驻留仍需测量。
- 采样速度应显著改善，但 CPU offload、FSDP unshard/reshard、预处理和 VAE 会限制端到端收益。

## 本机性能基线和发布门槛

任何外部计时都不是 O6 验收门槛。在任务 1 开始前，必须用当前精确实现建立新的同规格本机基线：

| 变量 | I2V 基线 | REF2VA 基线 |
|---|---|---|
| 工作流 | Turbo8 I2V | Turbo4 REF2VA |
| 计算策略 | 现有 `default` FP32 计算 | 现有 `default` FP32 计算 |
| 分辨率 | 1120×768 | 1120×768 |
| 帧数 / FPS | 124 / 24 | 124 / 24 |
| 启动状态 | 干净冷启动 | 干净冷启动 |
| 输入和提示词 | 现有 O5 输入/提示词 | 现有 O5 参考图/提示词 |

每次运行在 Git 中只保留精简数据，原始日志、遥测和媒体留在本机：

- 完整视频端到端耗时；
- ComfyUI 预处理时间；
- Ray worker 创建和 checkpoint/模型加载时间；
- 每个 sampler 的首步、稳定步、汇总 `s/it` 和 sampler 总耗时；
- VAE 解码、媒体保存和容器写入时间；
- GPU0/GPU1 利用率、功率和显存峰值；
- 主机物理内存、提交内存和分页文件峰值；
- P2P 健康度/流量以及两个 rank 的完成情况。

设上述工作流 `i` 的稳定采样速度为 `B_i`，匹配的 O6 安全 FP16 结果为 `F_i`。O6 初始功能/性能门槛为：

```text
B_i / F_i >= 4.0
```

两个工作流都必须满足。初始门槛通过后，`11×` 继续作为 O6 后续优化目标，但不阻塞第一个正确的安全 FP16 release candidate。基线完成后才把准确数值门槛写入验证报告。模型加载、预处理、VAE 解码和媒体保存均不得慢于对应基线，端到端时间必须改善。无论速度如何，数值、媒体、rank、内存和 P2P 门槛都必须满足。

## 计划文件结构

- 新建 `src/raylight/comfy_dist/minimax_h3_fp16.py`：幂等补丁安装器、模式识别、FP32 数值岛/FP16 分支 forward 和诊断。
- 修改 `src/raylight/nodes.py`：新增明确的 `fp16_h3_safe` RayUNETLoader 选项，并传递可序列化的模型选项标记。
- 修改 `src/raylight/distributed_worker/ray_worker.py`：消费该标记，在每个 rank 构造模型前安装补丁，并明确拒绝不受支持的组合。
- 测试 `tests/test_minimax_h3_fp16_patch.py`：数值范围、dtype 数值岛、幂等性、worker 安装和 ComfyUI 签名契约。
- 扩展 `tests/test_minimax_h3_fsdp_quant.py`：验证 FP8 存储、FP16 V100 fallback 和 FSDP wrapper 契约。
- 扩展 `tests/test_minimax_h3_turbo.py`：验证 BF16 Turbo LoRA sidecar 跟随 FP16 分支输入，同时不改变已验收 FP32 测试。
- 修改 `scripts/minimax-h3/build_workflows.py`：生成独立安全 FP16 实验工作流，不改变基线产物。
- 新建 `example_workflows/Minimax_H3_I2V_Windows_V100_FSDP_Turbo8_FP16_Experimental.json`。
- 新建 `example_workflows/Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4_FP16_Experimental.json`。
- 新建 `docs/testing/minimax-h3/SAFE_FP16_FSDP_2026-08.md`：包含精简结果和本机证据链接的持续维护验证报告。
- 新建 `docs/third-party/minimax-h3-fp16-fix.md`：记录上游 URL、固定提交、MIT 归属、借鉴思路和本地差异。

### 任务 0：同步 O1–O5 并锁定同规格本机基线

- [x] 让 `README.md`、`README_EN.md`、实施计划和 MiniMax 验证汇总与 O5 前已验收的代码及工作流一致。
- [x] 运行完整测试套件，提交整理后的仓库状态，并在 O6 实现前推送公开 `main` 分支。
- [x] 在不修改 O5 GUI 工作流的情况下，生成 1120×768、124 帧、24 FPS 的 Turbo8 I2V 和 Turbo4 REF2VA 隔离 benchmark 副本。
- [x] 从 GPU/Ray/ComfyUI 空闲状态分别冷启动并完整运行两个工作流。
- [x] 记录总耗时、预处理、worker/模型加载、各 sampler 首步/稳定/汇总 `s/it`、VAE 解码、媒体保存、显存/利用率/功率、主机内存/提交/分页以及 P2P/rank 证据。
- [x] 验证尺寸、帧数、视频/音频流、非黑时间变化和有限输出。
- [x] 把精简基线表、各工作流初始 `B_i / 4` 门槛和 `B_i / 11` 优化目标写入 `docs/testing/minimax-h3/SAFE_FP16_FSDP_2026-08.md`，原始证据保留在本机。
- [x] 两个基线完成且数据内部一致之前不得开始任务 1。

### 任务 1：用预期失败测试锁定数值和激活契约

**接口：**

```python
def install_minimax_h3_safe_fp16_patch() -> bool: ...
def safe_fp16_requested(model_options: dict) -> bool: ...
```

- [x] 增加测试，证明安装是幂等的，并且对 FP32/BF16 模型不生效。
- [x] 增加微型层测试，证明残差保持 FP32、Attention/MLP 输入为 FP16、条件投影接收 FP32，且缩放前等效幅值超过 65,504 时线性层输出仍为有限值。
- [x] 为当前 ComfyUI `DiTBlock.forward` 增加签名保护；上游重构后给出可操作的兼容性错误。
- [x] 运行 `E:\ComfyUI-py310\Python310\python.exe -m pytest -q tests/test_minimax_h3_fp16_patch.py`，并在实现前记录符合预期的 RED 失败。
- [x] 把 RED 输出作为本机证据保存，只实现使测试通过的最小改动，focused tests 变为 GREEN 后才提交。

### 任务 2：移植安全 FP16 数学并安装到两个 Ray worker

**接口：**

```python
SAFE_FP16_OPTION = "minimax_h3_safe_fp16"
SAFE_FP16_LOADER_VALUE = "fp16_h3_safe"
```

- [x] 把外部项目的 condition projection、residual、`out_proj` 和 MLP 保护移植到 `minimax_h3_fp16.py`，保留归属信息和 64/256 两个 2 的幂缩放常数。
- [x] 标记已补丁的类/实例，确保 worker 复用或 checkpoint 切换时不会重复包装 forward。
- [x] 将 `RayUNETLoader.weight_dtype=fp16_h3_safe` 映射为 `dtype=torch.float16` 与 `minimax_h3_safe_fp16=True`；通用 `fp16` 行为保持不变。
- [x] 在 `RayWorker.load_unet` 中消费私有标记，在调用 `fsdp_load_diffusion_model` 前安装补丁；除非显式测试覆盖，否则拒绝非 V100/无 CUDA 用法。
- [x] 每个 rank 输出一行诊断，包含模型 dtype、manual-cast dtype、安全 FP16 状态和 compute capability。
- [x] 运行任务 1 测试、`tests/test_ray_worker_lifecycle.py` 和 `tests/test_minimax_h3_workflows.py`，要求全部 GREEN。
- [x] 提交经过审查的 worker-safe 补丁和 loader 模式。

### 任务 3：证明 FP8 FSDP 和 Turbo LoRA 跟随 FP16 分支输入

- [x] 增加 V100 FP8 fallback 测试：量化存储保持 FP8，`fp8_linear_fallback_chunked` 接收 FP16 输入并返回有限 FP16 输出。
- [x] 保留现有 FP32-compute 测试不变，并新增独立安全 FP16 策略测试，不用后者替换前者。
- [x] 使用 BF16 up/down 张量和 FP16 分支输入增加 LoRA sidecar 测试；确认两个 sidecar 矩阵乘法都以 FP16 执行且结果有限。
- [x] 增加 FSDP 策略测试，证明安全 FP16 不会意外启用仅适用于 BF16 的 `MixedPrecisionPolicy`，也不会在 all-gather 前把 FP8 存储稠密化。
- [x] 运行 `tests/test_minimax_h3_fsdp_quant.py`、`tests/test_minimax_h3_turbo.py`、`tests/test_fp8_fsdp_gather_shape.py` 和 `tests/test_fsdp_lora_streaming.py`，要求全部 GREEN。
- [x] 提交经过审查的 FP8/FSDP/LoRA 兼容性契约。

### 任务 4：生成隔离的实验 GUI 工作流

- [x] 扩展工作流生成器，增加接受 `default` 和 `fp16_h3_safe` 的 `compute_dtype` 参数；在完整 MiniMax H3 profile 之外拒绝安全 FP16。
- [x] 生成两个 `_FP16_Experimental.json`，提示词、seed、输入、尺寸、帧数、Turbo LoRA 和 8/4 步均与 O5 相同；只有 RayUNETLoader 模式可以不同。
- [x] 增加逐字节再生成测试和 SHA-256 保护，证明上游、20 步和 O5 Turbo 工作流均未改变。
- [x] 启动 ComfyUI 获取 `/object_info`，把两个实验 GUI 工作流转换成 API prompt，验证 21 个可执行节点，并在缩减 smoke 后让两个 GPU 回到空闲。
- [x] 提交经过审查的生成器和实验工作流。

### 任务 5：分阶段运行 CUDA 验证

- [x] 在已审查的 LoRA projection 修复后重新运行双 rank 模型加载探针，确认每个 rank 684 个 FSDP wrapper、FP8 checkpoint 存储、CUDA P2P collective，且不存在主机中转张量传输。
- [x] 重新运行单步缩减 I2V smoke，记录各 rank dtype/存储、最大绝对值和有限值诊断。
- [x] 使用 Turbo LoRA 和相同诊断重新运行匹配的缩减 REF2VA smoke。
- [x] 遇到 NaN/Inf、黑输出、rank 不一致、LoRA dtype 不一致、collective fallback 或 FP32 Attention/MLP 分支时立即拒绝。
- [x] smoke 通过后，使用已验收 O5 输入和设置分别运行一次冷启动完整 I2V Turbo8 和 REF2VA Turbo4。
- [x] 记录预处理、worker/模型加载、主采样、sampler 总耗时、解码/写入、显存、GPU 利用率、物理/提交内存、分页文件和 P2P 流量。

### 任务 6：比较、审查并在本地发布

- [x] 分别以端到端和阶段耗时把安全 FP16 与任务 0 的同规格 FP32 基线比较；只发布实测值，不用外部或预测数字替代。
- [x] 验证视频尺寸/帧数、唯一帧哈希、黑屏检测、有限非静音音频和同 seed 视觉行为。
- [ ] 要求两个工作流都达到 `baseline stable s/it / safe-FP16 stable s/it >= 4.0` 的初始门槛，并单独报告距离 `11×` 优化目标的进度。
- [ ] 要求模型加载、预处理、VAE 解码和媒体保存均不慢于对应基线，同时端到端耗时改善。
- [ ] 公开发布前运行 `py_compile`、完整 pytest、工作流 hash guards 和代码审查，要求没有未解决的 High/Medium 问题。
- [x] 更新 O6 状态、中英文 README、双语升级记录和 `docs/testing/minimax-h3/README.md`，准确反映功能/质量通过但性能门槛未关闭。
- [ ] 在所有正确性和性能门槛通过之前，安全 FP16 保持显式启用。

## 当前门槛状态

任务 0–5 已实现。缩减 smoke 和完整分辨率 I2V/REF2VA 均通过数值、rank、P2P 和媒体检查。初始完整安全 FP16 加速分别为 3.299× 和 3.416×；REF2VA 最佳可选实验达到 3.714×。因此 4× 门槛仍未关闭，模式继续保持显式启用。拒绝的 prefetch 和 single-ring 实验已完整撤回，`30cbb69` 是记录的运行时代码冻结点。完整时间和资源矩阵见 `docs/releases/` 中的双语升级记录。
