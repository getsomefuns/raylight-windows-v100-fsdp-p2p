# O12 Python 3.12 / PyTorch 2.8 与 MiniMax H3 双加速适配实施计划

> **执行要求：** 使用 `superpowers:executing-plans` 在当前独立工作树逐项执行；按用户要求禁止使用子代理。所有行为变更先写失败测试，再写实现。

**目标：** 在不修改 ComfyUI、MiniMaxH3Director、TE-Speed 和 Sol-Attn 源码的前提下，将运行时迁移到 Python 3.12.10 + PyTorch 2.8.0+cu126，并让现有 Raylight 工作流保持兼容，同时支持 `Ray MODEL Adapter -> TE-Speed -> Sol-Attn -> MiniMaxH3Director`。

**架构：** 保留 Python310 作为回滚环境，在 `E:\ComfyUI-py310\Python312` 建立候选运行时。Raylight 代理负责记录外部 MODEL patch，桥接层将其归一化为可序列化配置，两个 Ray worker 在真实 MiniMax H3 模型和 Ulysses/FSDP 路径中执行缓存与稀疏注意力。

**技术栈：** Windows、2× Tesla V100-SXM2-16GB TCC、驱动 577.00、CUDA 12.6、Python 3.12.10、PyTorch 2.8.0+cu126、Ray 2.57、xFuser 0.4.5、yunchang 0.6.4、TE-Speed 3.3、Sol-Attn V100 1.2.x。

## 全局约束

- 唯一源码工作树为 `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`。
- 不修改 `E:\ComfyUI-py310\ComfyUI` 的核心源码，不修改其他自定义节点源码。
- Python310 在 O12 全部验收前不得删除或覆盖。
- 保持现有节点名称、必填输入、输出类型和旧工作流 JSON 兼容。
- V100 运行时固定 CUDA 12.6/SM70，不使用 cu128/cu130。
- 启动时继续使用 `--disable-cuda-malloc`，并默认禁用不适合当前 V100 的 DynamicVRAM。
- 所有新加速默认关闭；缺少外部节点时旧工作流仍可运行。

---

### Task 1：建立可回滚的 Python312 候选环境

**文件：**
- 新建：`E:\ComfyUI-py310\Python312`
- 新建：`requirements-windows-v100-torch28.txt`
- 修改：`requirements-windows-v100.txt`

- [ ] 记录 Python310 的 `pip freeze`、Torch CUDA 架构和当前启动基线。
- [ ] 安装 Python 3.12.10 到独立目录，不覆盖 Python310。
- [ ] 安装 Torch 2.8.0+cu126、torchvision 0.23.0+cu126、torchaudio 2.8.0+cu126、xformers 0.0.32.post2及当前 Raylight/ComfyUI 依赖。
- [ ] 执行 `pip check`，确认 `torch.cuda.get_arch_list()` 包含 `sm_70`。
- [ ] 分别隔离导入 TE-Speed 和通过 `torch.ops.load_library()` 注册 Sol-Attn，保存结果。

### Task 2：让启动与验证脚本支持双运行时

**文件：**
- 修改：`scripts/start-comfyui-windows-p2p.ps1`
- 修改：`scripts/verify-windows-v100.ps1`
- 修改：`tests/test_path_helpers.py`
- 新建：`tests/test_windows_runtime_profile.py`

- [ ] 先写失败测试：默认优先 Python312、显式路径优先、缺失时回退 Python310、Torch 2.8配置不被旧2.7固定检查拒绝。
- [ ] 运行测试并确认因尚未支持运行时 profile 而失败。
- [ ] 实现 `torch27` 与 `torch28` 两个验证 profile，启动脚本默认选择 Python312。
- [ ] Torch 2.8/V100 启动参数加入 `--disable-dynamic-vram`，同时保留 `--disable-cuda-malloc`。
- [ ] 运行脚本测试和 PowerShell语法验证。

### Task 3：保持旧 XFORMERS_MEMORY_EFFICIENT 工作流兼容

**文件：**
- 修改：`src/raylight/distributed_modules/attention.py`
- 新建：`tests/test_v100_attention_backend_resolution.py`
- 修改：`tests/test_minimax_parallel.py`

- [ ] 先写失败测试：旧枚举值保持不变；V100无可用 xFormers kernel时选择 Torch Efficient；可用时仍选择 xFormers；两者不可用时明确报错。
- [ ] 运行测试并确认失败原因是缺少能力路由。
- [ ] 在现有 Ulysses拥有式注意力函数内加入窄范围能力选择，不改变调用方和工作流字符串。
- [ ] 对真实 V100执行 FP16小/中/长序列正确性、显存和速度探针。

### Task 4：验证 Raylight 在 Torch 2.8上的核心能力

**文件：**
- 仅在发现真实不兼容时修改对应 Raylight拥有层，并先补回归测试。

- [ ] 运行完整单元测试。
- [ ] 运行 CUDA P2P、Ray双 actor、Gloo控制面、FSDP2、量化权重、Text Encoder TP2探针。
- [ ] 运行旧 MiniMax H3 Director 4步工作流，确认旧路径正常出片。
- [ ] 连续运行三次，确认 worker生命周期、内存和显存不持续增长。

### Task 5：识别两个外部节点的 patch 契约

**文件：**
- 新建：`tests/test_mmh3_acceleration_patch_contract.py`
- 新建：`src/raylight/integrations/mmh3director/acceleration.py`

- [ ] 在 Python312中记录 TE-Speed 的输入字段、clone调用、模型访问和写入的 patch状态。
- [ ] 记录 Sol-Attn写入的 `model_options`、层索引需求和 CUDA库路径。
- [ ] 先写失败测试，定义只含字符串、数字、布尔值和列表的 `MMH3AccelerationConfig` 归一化结果。
- [ ] 实现严格的 MiniMax H3边界归一化；不把外部函数对象传入 Ray。

### Task 6：让 Ray MODEL代理接受外部节点串联

**文件：**
- 修改：`src/raylight/integrations/mmh3director/proxy.py`
- 修改：`src/raylight/integrations/mmh3director/bridge.py`
- 修改：`tests/test_mmh3director_proxy.py`
- 修改：`tests/test_mmh3director_bridge.py`

- [ ] 先写失败测试：原 Adapter输出可以经过 TE-Speed/Sol节点所需的 clone和模型外观接口；旧代理行为不变。
- [ ] 实现无权重的 MiniMax H3结构外观，只记录 patch配置，不在主进程加载第二份模型。
- [ ] 桥接层提取归一化配置并随 operation发送；缺少配置时生成 `off`。
- [ ] 确认外部节点插入与不插入两种图都能验证通过。

### Task 7：在每个 Ray worker执行 Sol-Attn

**文件：**
- 新建：`src/raylight/diffusion_models/minimax/sol_attn_v100.py`
- 修改：`src/raylight/diffusion_models/minimax/xdit_context_parallel.py`
- 修改：`src/raylight/distributed_worker/ray_worker.py`
- 新建：`tests/test_sol_attn_v100_bridge.py`

- [ ] 先写失败测试：每个 worker只注册一次外部 CUDA op；配置窗口/dense block/prefix/top-k正确；dense回退走 Raylight精确后端。
- [ ] 实现外部节点路径解析和进程内 `torch.ops.load_library()`。
- [ ] 在 Ulysses all-to-all之后调用 Sol稀疏 kernel，保持输出布局和 dtype。
- [ ] Raylight已有 FP16安全逻辑拥有最终控制权，阻止外部内嵌 FP16Safe重复缩放。
- [ ] 真机对比 dense/Sol的数值有限性、显存、每步速度和视频质量。

### Task 8：在 Ray worker执行 TE-Speed兼容缓存

**文件：**
- 新建：`src/raylight/diffusion_models/minimax/te_speed_cache.py`
- 修改：`src/raylight/diffusion_models/minimax/xdit_context_parallel.py`
- 修改：`src/raylight/distributed_worker/ray_worker.py`
- 新建：`tests/test_te_speed_cache.py`

- [ ] 根据 Task 5实测契约先写缓存模式、失效条件和双 rank一致性的失败测试。
- [ ] 实现 operation内的 rank本地残差缓存，rank 0广播命中决定。
- [ ] 跳过的 blocks不得继续触发 FSDP预取或权重调度。
- [ ] seed、conditioning、latent shape、LoRA、shift、scheduler、Director片段变化时失效。
- [ ] worker回收和异常路径必须清空缓存。

### Task 9：组合加速与端到端验收

**文件：**
- 更新：`docs/TESTING.md`
- 更新：`README.md`
- 更新：`README_EN.md`
- 新建：`docs/releases/2026-09-10-o12-python312-torch28-h3-acceleration.zh-CN.md`
- 新建：`docs/releases/2026-09-10-o12-python312-torch28-h3-acceleration.en.md`

- [ ] 运行相同素材/提示词/seed的5秒、0.4MP、16:9、FL2V工作流。
- [ ] 分别测试无加速、TE、Sol、TE+Sol的4步与8步热启动。
- [ ] 记录加载、预处理、Text Encoder、worker启动、采样、VAE/音频解码、保存、端到端时间及GPU/内存/分页峰值。
- [ ] 组合路径目标达到或优于社区参考：4步132秒、8步200秒；若未达到，必须给出实测瓶颈而非降低标准。
- [ ] 检查黑屏、噪声、NaN、音频异常、动作断裂和重复运行泄漏。
- [ ] 部署唯一工作树到 `ComfyUI/custom_nodes/raylight`，复验部署一致性并更新中英文文档。

## 自审结论

- 覆盖 Python/Torch迁移、回滚、V100/SM70、xFormers退路、DynamicVRAM隔离、Raylight旧工作流、两个外部节点串联、双 rank一致性和端到端质量性能验收。
- 外部节点的闭源 TE patch契约必须先在 Python312实测，计划没有假设其内部字段。
- 所有 Raylight行为改动均安排了 RED/GREEN测试；环境安装与基准采集属于配置/验证步骤，不写生产代码。
