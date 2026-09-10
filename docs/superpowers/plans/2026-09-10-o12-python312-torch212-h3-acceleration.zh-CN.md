# O12 Python 3.12 / PyTorch 2.12 与 MiniMax H3 双加速适配实施计划

> **执行要求：** 使用 `superpowers:executing-plans` 在当前独立工作树逐项执行；按用户要求禁止使用子代理。所有行为变更先写失败测试，再写实现。

**目标：** 在不修改 ComfyUI、MiniMaxH3Director、TE-Speed 和 Sol-Attn 源码的前提下，将运行时迁移到 Python 3.12.10 + PyTorch 2.12.0+cu126，并让现有 Raylight 工作流保持兼容，同时支持 `Ray MODEL Adapter -> TE-Speed -> Sol-Attn -> MiniMaxH3Director`。

> **2026-09-10 实机修订：** 原候选 PyTorch 2.8.0+cu126 在 Windows wheel 中无法创建任何 Gloo transport device，最小单进程复现也会报 `makeDeviceForHostname(): unsupported gloo device`。PyTorch 官方 issue #150381 确认 2.8/2.9 Windows 构建关闭了 Gloo libuv transport，修复进入后续版本。由于 Raylight Windows FSDP 需要 Gloo 控制面，运行时基线改为补丁作者实测使用且本机已通过 Gloo 探针的 PyTorch 2.12.0+cu126；Python 3.12 与 CUDA 12.6/SM70 约束不变。

**架构：** 保留 Python310 作为回滚环境，在 `E:\ComfyUI-py310\Python312` 建立候选运行时。Raylight 代理负责记录外部 MODEL patch，桥接层将其归一化为可序列化配置，两个 Ray worker 在真实 MiniMax H3 模型和 Ulysses/FSDP 路径中执行缓存与稀疏注意力。

**技术栈：** Windows、2× Tesla V100-SXM2-16GB TCC、驱动 577.00、CUDA 12.6、Python 3.12.10、PyTorch 2.12.0+cu126、torchvision 0.27.0+cu126、torchaudio 2.11.0+cu126、xFormers 0.0.35、Ray 2.57、xFuser 0.4.5、yunchang 0.6.4、TE-Speed 3.3、Sol-Attn V100 1.1.1。

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
- 新建：`requirements-windows-v100-torch212.txt`
- 修改：`requirements-windows-v100.txt`

- [x] 记录 Python310 的 Torch CUDA 架构和当前启动基线。
- [x] 安装 Python 3.12.10 到独立目录，不覆盖 Python310。
- [x] 安装 Torch 2.12.0+cu126、torchvision 0.27.0+cu126、torchaudio 2.11.0+cu126、xFormers 0.0.35及当前 Raylight/ComfyUI 依赖。
- [x] 执行 `pip check`，确认 `torch.cuda.get_arch_list()` 包含 `sm_70`。
- [x] 分别隔离导入 TE-Speed 和注册/执行 Sol-Attn CUDA op，保存结果。

### Task 2：让启动与验证脚本支持双运行时

**文件：**
- 修改：`scripts/start-comfyui-windows-p2p.ps1`
- 修改：`scripts/verify-windows-v100.ps1`
- 修改：`tests/test_path_helpers.py`
- 新建：`tests/test_windows_runtime_profile.py`

- [x] 先写失败测试：默认优先 Python312、显式路径优先、缺失时回退 Python310、新运行时不被旧2.7固定检查拒绝。
- [x] 运行测试并确认因尚未支持运行时 profile 而失败。
- [x] 实现旧2.7与新2.12验证 profile，启动脚本默认选择 Python312。
- [x] V100启动参数加入 `--disable-dynamic-vram`，同时保留 `--disable-cuda-malloc`。
- [x] 运行脚本测试和 PowerShell语法验证。

### Task 3：保持旧 XFORMERS_MEMORY_EFFICIENT 工作流兼容

**文件：**
- 修改：`src/raylight/distributed_modules/attention.py`
- 修改：`tests/test_diffusion_xformers_attention.py`
- 修改：`tests/test_minimax_parallel.py`

- [x] 先写失败测试：旧枚举值保持不变；V100无可用 xFormers kernel时选择 Torch Efficient；可用时仍选择 xFormers；两者不可用时明确报错。
- [x] 运行测试并确认失败原因是缺少能力路由。
- [x] 在现有 Ulysses拥有式注意力函数内加入窄范围能力选择，不改变调用方和工作流字符串。
- [x] 对真实 V100执行 FP16小/中/长序列正确性、显存和速度探针。

### Task 4：验证 Raylight 在 Torch 2.12上的核心能力

**文件：**
- 仅在发现真实不兼容时修改对应 Raylight拥有层，并先补回归测试。

- [x] 运行完整单元测试（最终 Python312/Torch2.12 回归725项与35项子测试通过）。
- [x] 运行 CUDA P2P、Ray双 actor、Gloo控制面、xFuser子组、FSDP2量化权重探针。
- [x] 修复 xFuser硬编码 `backend="gloo"` 在新运行时另建隐式设备而卡死的问题，统一路由到 Raylight显式 Windows Gloo backend。
- [x] 修复多分块复用 CUDA IPC event 时快 rank 可能提前覆盖共享缓冲区的问题；UINT8/FP16/BF16同步与异步尾块压力测试通过，实测带宽仍约59–61 GiB/s。
- [x] 运行旧 MiniMax H3 Director 4步 Base 工作流，确认旧路径正常出片。
- [x] 4步 TE+Sol 连续运行三次；同一 worker 对复用，退出后两卡回到10 MiB，未出现持续内存/显存增长。

### Task 5：识别两个外部节点的 patch 契约

**文件：**
- 新建：`tests/test_mmh3director_external_acceleration.py`
- 新建：`src/raylight/integrations/mmh3director/external_acceleration.py`

- [x] 在 Python312中记录 TE-Speed 的输入字段、clone调用、模型访问和写入的 patch状态。
- [x] 记录 Sol-Attn写入的 `model_options`、层索引需求和 CUDA库路径。
- [x] 先写失败测试，定义只含字符串、数字、布尔值、字典和列表的归一化结果。
- [x] 实现严格的 MiniMax H3边界归一化；不把外部函数对象传入 Ray。

### Task 6：让 Ray MODEL代理接受外部节点串联

**文件：**
- 修改：`src/raylight/integrations/mmh3director/proxy.py`
- 修改：`src/raylight/integrations/mmh3director/bridge.py`
- 修改：`tests/test_mmh3director_proxy.py`
- 修改：`tests/test_mmh3director_bridge.py`

- [x] 先写失败测试：原 Adapter输出可以经过 TE-Speed/Sol节点所需的 clone和模型外观接口；旧代理行为不变。
- [x] 实现无权重的 MiniMax H3结构外观，只记录 patch配置，不在主进程加载第二份模型。
- [x] 桥接层提取归一化配置并随 operation发送；缺少配置时保持旧路径。
- [x] 确认外部节点插入与不插入两种图都能验证通过。

### Task 7：在每个 Ray worker执行 Sol-Attn

**文件：**
- 修改：`src/raylight/integrations/mmh3director/external_acceleration.py`
- 修改：`src/raylight/distributed_modules/attention.py`
- 修改：`src/raylight/distributed_worker/ray_worker.py`
- 测试：`tests/test_mmh3director_external_acceleration.py`

- [x] 先写失败测试：配置窗口/dense block/prefix/top-k正确；dense回退走 Raylight精确后端。
- [x] worker内导入并调用真实外部 Sol节点，不复制其算法或维护第二份实现。
- [x] 在 Ulysses all-to-all之后调用 Sol稀疏 override，保持输出布局和 dtype。
- [x] Raylight已有 FP16安全逻辑拥有最终控制权，强制关闭外部内嵌 FP16Safe重复缩放。
- [x] 真机合成张量验证28/56头布局、数值有限性和CUDA kernel实际消费Q/K/V；用户原始 `.pyd` 在Torch2.12下默认路径加载并执行通过。
- [x] 完整4/8步视频验证速度、显存与质量；Sol实际作用于两个 rank，16个矩阵视频全部通过媒体检查。

### Task 8：在 Ray worker执行 TE-Speed兼容缓存

**文件：**
- 修改：`src/raylight/integrations/mmh3director/external_acceleration.py`
- 修改：`src/raylight/distributed_worker/ray_worker.py`
- 测试：`tests/test_mmh3director_external_acceleration.py`

- [x] 根据 Task 5实测契约先写序列化和worker重放失败测试。
- [x] worker内导入并调用真实 TE-Speed节点，使其闭源 Cython缓存算法直接作用于真实 worker ModelPatcher。
- [x] 每个 rank使用一致的纯数据配置重建patch，不跨 Ray传递 Cython对象或函数。
- [x] 缺失外部节点时给出明确错误；未串联时旧路径不受影响。
- [x] 完整视频验证双 rank一致性、速度与异常回收；短轨迹中TE未减少collective或产生稳定独立收益，保留为显式实验选项。

### Task 9：组合加速与端到端验收

**文件：**
- 更新：`docs/TESTING.md`
- 更新：`README.md`
- 更新：`README_EN.md`
- 新建：`docs/releases/2026-09-10-o12-python312-torch212-h3-acceleration.zh-CN.md`
- 新建：`docs/releases/2026-09-10-o12-python312-torch212-h3-acceleration.en.md`
- 新建：`scripts/minimax-h3/build_o12_director_workflow.py`

- [x] 运行相同素材/提示词/几何的5.17秒、0.4MP、16:9 REF2VA工作流；社区FL2V只作量级参考，不混作同工作流对照。
- [x] 分别测试无加速、TE、Sol、TE+Sol的4步与8步冷/热启动。
- [x] 记录加载、conditioning、worker启动、采样、VAE/音频解码、保存、端到端时间及GPU/内存/分页峰值。
- [x] 对照社区量级：组合4步130.67秒，优于132秒；组合8步200.05秒，比200秒慢0.05秒，按严格数值未超越。相对本机Base，Sol采样4/8步均约快19%。
- [x] 检查黑屏、噪声、NaN、音频异常和重复运行泄漏；16个矩阵视频均124/124唯一帧、0黑段、完整解码，联系表无明显故障。
- [x] 从已提交的唯一工作树 HEAD 部署到 `ComfyUI/custom_nodes/raylight`，487个受管理文件哈希一致；中英文README与发布说明已同步更新。

## 自审结论

- 覆盖 Python/Torch迁移、回滚、V100/SM70、xFormers退路、DynamicVRAM隔离、Raylight旧工作流、两个外部节点串联、双 rank一致性和端到端质量性能验收。
- 外部节点的闭源 TE patch契约必须先在 Python312实测，计划没有假设其内部字段。
- 所有 Raylight行为改动均安排了 RED/GREEN测试；环境安装与基准采集属于配置/验证步骤，不写生产代码。
