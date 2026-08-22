# Raylight 1.10.0 / O8 MMH3Director MODEL 适配器实施计划

[简体中文](2026-08-22-mmh3director-model-adapter.zh-CN.md) | [English](2026-08-22-mmh3director-model-adapter.md)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 新增一个可选的 `Ray MODEL Adapter (MMH3Director)`，让未经修改的 MiniMax H3 Director 使用现有 Windows 双 V100 Raylight FSDP + CUDA P2P Worker，同时冻结所有现有节点和工作流接口。

**架构：** 保持已验收的 `RAY_ACTORS` 路径不动，新增轻量 `MODEL` 代理、严格类型门控的 `comfy.sample.sample` 桥接器，以及独立的 Director Worker 采样入口。代理只携带元数据和 actor 句柄，不携带权重；普通 MODEL 完整透传到原采样器，代理不兼容时直接失败，绝不静默回退到主进程单卡采样。

**技术栈：** 原生 Windows 23H2、双 Tesla V100-SXM2-16GB TCC、NVIDIA 577.00、Python 3.10.11、PyTorch 2.7.0+cu126、ComfyUI 0.31.0、Ray 2.57.0、Raylight FSDP2、xFuser/yunchang、自定义 CUDA P2P/NVLink 数据面、Gloo 控制面。

**设计规格：** `docs/superpowers/specs/2026-08-22-mmh3director-model-adapter-design.zh-CN.md`

## 全局约束

- 只操作 `E:\ComfyUI-py310`，不修改 NVIDIA 577.00，也不影响该目录外的环境。
- 沿用 `raylight-windows-v100-fsdp-p2p` 仓库和 `windows-v100-fsdp-p2p` 分支，不新建仓库或长期分支。
- 目标版本 `1.10.0`，阶段 `O8`；写代码前给已验收起点打标签 `v1.9.0-pre-mmh3director`。
- 首次兼容验收固定到 Director `1.0.0`、提交 `d655912262bcd70ad43c3a431879be9539d8e054`；没有 Git 元数据的压缩包安装必须通过等价的运行时签名检查。
- 不修改或 fork `ComfyUI_MiniMaxH3_Director`。
- 新节点显示名固定为 `Ray MODEL Adapter (MMH3Director)`，内部 ID 为 `RayMMH3DirectorModelAdapter`。
- 不修改 `RayInitializer`、`RayInitializerAdvanced`、`RayUNETLoader`、现有采样器和现有工作流的公开 schema。
- 通用节点不添加 Director 后缀；只有失败的集成测试证明确实需要专属初始化器时，才允许新增 `Ray Initializer (MMH3Director)`。
- 保留已验收的 Windows CUDA P2P、FSDP、O6 安全 FP16、LTX 和 MiniMax H3 路径。
- Ray 代理请求失败时不得静默转为主进程扩散采样。
- Director 是可选依赖；未安装时 Raylight 必须正常导入和运行。
- CLIP、Video VAE 和 Audio VAE 仍是主进程阶段，不宣称被 FSDP 分片。
- 测试先行、小提交、使用同条件实机回归数据。

## 计划文件结构

- 新建 `src/raylight/integrations/__init__.py`：集成命名空间。
- 新建 `src/raylight/integrations/mmh3director/__init__.py`：节点映射与幂等桥接安装。
- 新建 `src/raylight/integrations/mmh3director/metadata.py`：紧凑元数据 schema 与 rank 校验。
- 新建 `src/raylight/integrations/mmh3director/proxy.py`：无权重 MODEL facade。
- 新建 `src/raylight/integrations/mmh3director/bridge.py`：类型门控、请求构造、结果校验和最终回调策略。
- 新建 `src/raylight/integrations/mmh3director/nodes.py`：适配节点。
- 修改 `src/raylight/distributed_worker/ray_worker.py`：新增元数据导出、Director 请求采样和状态字段。
- 修改根 `__init__.py`：不导入 Director 的可选集成注册。
- 新建八组 O8 单元、契约、生命周期和工作流测试。
- 新建 `example_workflows/MMH3Director_Windows_V100_FSDP.json`，验收后部署到 `E:\ComfyUI-py310\ComfyUI\user\default\workflows\MMH3Director.json`。
- 新建中英文 O8 release notes 和维护型测试报告。
- 只有实现和实机验收完成后才修改 `README.md`、`README_EN.md` 和 `pyproject.toml`。

---

### 任务 0：冻结已验收的 1.9.0 起点

**文件：**
- 核对：`README.md`、`README_EN.md`
- 核对：`docs/testing/minimax-h3/SAFE_FP16_FSDP_2026-08.md`
- 核对：`example_workflows/` 下已验收工作流

**接口：**
- 输入：当前已验收 HEAD 和 O6/LTX 基线。
- 输出：回退标签 `v1.9.0-pre-mmh3director` 和干净的基线清单。

- [ ] 执行 `git status --short`；若无关用户改动与本计划文件重叠则停止处理。
- [ ] 执行 `E:\ComfyUI-py310\Python310\python.exe -m pytest -q tests`，保存精简通过/失败摘要。
- [ ] 运行 README 已记录的 P2P/FSDP 验证脚本，只保留摘要，不提交原始机器日志。
- [ ] 在 O8 测试报告记录现有节点 schema 和已验收工作流 JSON 哈希。
- [ ] 执行 `git tag -a v1.9.0-pre-mmh3director -m "Accepted state before MMH3Director adapter"`。
- [ ] 若基线报告发生变化，仅提交报告；本任务不修改运行时代码。

### 任务 1：用契约测试冻结向后兼容性

**文件：**
- 新建：`tests/test_mmh3director_regression_contracts.py`

**接口：**
- 输入：现有节点类和工作流。
- 输出：发布必须通过的旧接口快照测试。

- [ ] 精确断言 `RayInitializer`、`RayInitializerAdvanced`、`RayUNETLoader`、`RayKill` 和现有采样器的 `RETURN_TYPES`、`RETURN_NAMES`、`FUNCTION` 与显示名。
- [ ] 加载所有已验收 LTX/MiniMax JSON，检查节点类型和连线输出索引仍可解析。
- [ ] 断言 `RayUNETLoader.RETURN_TYPES == ("RAY_ACTORS",)`，防止未来追加代理输出。
- [ ] 运行本文件测试，要求在未修改的 1.9.0 代码上全部 GREEN。
- [ ] 提交绿色兼容围栏；新节点的 RED/GREEN 循环放在任务 3，分支中不保留已知失败的提交。

### 任务 2：实现紧凑元数据和无权重 MODEL 代理

**文件：**
- 新建：`src/raylight/integrations/__init__.py`
- 新建：`src/raylight/integrations/mmh3director/metadata.py`
- 新建：`src/raylight/integrations/mmh3director/proxy.py`
- 新建：`tests/test_mmh3director_proxy.py`

**接口：**

```python
@dataclass(frozen=True)
class MMH3DirectorModelMetadata:
    model_family: str
    sampling_settings: dict[str, object]
    latent_channels: int
    latent_dimensions: int
    spacial_downscale_ratio: int
    temporal_downscale_ratio: int
    noise_scale: float | None

def validate_rank_metadata(rows: list[dict]) -> MMH3DirectorModelMetadata: ...

class RayMMH3DirectorModelProxy:
    def clone(self) -> "RayMMH3DirectorModelProxy": ...
    def get_model_object(self, name: str): ...
    def add_object_patch(self, name: str, obj) -> None: ...
```

- [ ] 测试 rank 元数据字段不一致时明确指出 rank/字段，一致时生成不可变 dataclass。
- [ ] 测试代理不包含 `torch.nn.Parameter`、state dict 或 diffusion model。
- [ ] 测试 clone 的 `model_options` 和 object patches 与父对象相互隔离。
- [ ] 测试 `model_sampling`、`latent_format`、未知对象和 `add_object_patch`。
- [ ] 使用真实 `MiniMaxH3SigmaShift.execute(proxy, 12.0, 3.0)` 做兼容测试，确认 clone 捕获两个 shift。
- [ ] 先运行得到 RED，再用只包含 ComfyUI 实际访问字段的轻量 facade 实现。
- [ ] 聚焦测试 GREEN 后提交。

### 任务 3：增加可选适配节点和安全注册

**文件：**
- 新建：`src/raylight/integrations/mmh3director/nodes.py`
- 新建：`src/raylight/integrations/mmh3director/__init__.py`
- 修改：根 `__init__.py`
- 新建：`tests/test_mmh3director_nodes.py`
- 新建：`tests/test_mmh3director_optional_dependency.py`

**接口：**

```python
class RayMMH3DirectorModelAdapter:
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "adapt_model"

    def adapt_model(self, ray_actors: dict):
        return (RayMMH3DirectorModelProxy(...),)
```

- [ ] 测试只有一个 `RAY_ACTORS` 输入和一个 `MODEL` 输出，以及分类和准确显示名。
- [ ] 测试注册新集成后 initializer/loader schema 没有变化。
- [ ] 屏蔽所有 Director import，确认 Raylight 根包仍能导入。
- [ ] 用 fake actor 测试从所有 rank 读取元数据，拒绝非 MiniMax 模型和未加载 Worker。
- [ ] 先运行 RED，再实现不 import Director 的节点注册。
- [ ] 保持通用 `RayInitializer` 不动；没有测试证据就不新增专属 initializer。
- [ ] 运行聚焦测试和 `tests/test_root_package_import.py`，GREEN 后提交。

### 任务 4：安装幂等、严格类型门控的采样桥

**文件：**
- 新建：`src/raylight/integrations/mmh3director/bridge.py`
- 新建：`tests/test_mmh3director_bridge.py`

**接口：**

```python
def install_mmh3director_sample_bridge() -> bool: ...
def uninstall_mmh3director_sample_bridge_for_test() -> None: ...
def dispatch_mmh3director_sample(model, noise, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent_image,
                                 **kwargs) -> torch.Tensor: ...
```

- [ ] 普通 sentinel MODEL 必须一次且逐参数调用原函数。
- [ ] 两次安装只能产生一层 wrapper，测试卸载能恢复同一个原函数对象。
- [ ] 代理路径即使报错也绝不能调用本地原采样器。
- [ ] 覆盖 noise、denoise、noise_mask、seed、steps、CFG、sampler、scheduler、conditioning、latent 的请求形状。
- [ ] 覆盖模型族错误、actor 失效、Director 签名不支持和 rank 结果缺失的关闭式失败。
- [ ] 固定提交 `d655912262bcd70ad43c3a431879be9539d8e054` 的 `sample_single_stage` 签名，并分别模拟删除和改名参数，要求给出可操作的兼容错误。
- [ ] 覆盖 rank 不一致、NaN/Inf 和输出 dtype/device 归一化。
- [ ] RED 后做最小实现，GREEN 后从集成模块幂等安装并提交。

### 任务 5：增加独立的 Director Worker 采样入口

**文件：**
- 修改：`src/raylight/distributed_worker/ray_worker.py`
- 新建：`tests/test_mmh3director_worker.py`

**接口：**

```python
def get_mmh3director_proxy_metadata(self) -> dict: ...
def sample_mmh3director_request(self, request: dict) -> dict: ...
def get_mmh3director_sample_status(self, operation_id: str) -> dict: ...
```

- [ ] 测试元数据只含模型族、sampling settings、latent format 和 noise scale，不含权重 tensor/state dict。
- [ ] 测试请求在 Worker 模型上应用 `shift_video`/`shift_audio` 后再采样。
- [ ] 测试严格使用 Director 已准备的 noise，而不是用不同 seed 重新生成。
- [ ] 测试 noise mask、denoise、禁用 callback、有限值摘要和 operation ID。
- [ ] 测试成功/失败后状态清理，并保证现有 `common_ksampler` 签名与行为不变。
- [ ] RED 后新增独立方法；只有现有采样器等价测试通过时才抽取共用私有 helper。
- [ ] 不调用主进程原采样器，不向 Worker 序列化 Director callback 闭包。
- [ ] 运行新测试、现有 sampler、lifecycle 和 shutdown 测试，GREEN 后提交。

### 任务 6：完善多分镜、Refine、进度和清理

**文件：**
- 修改：`bridge.py`、`proxy.py`
- 扩展：`tests/test_mmh3director_bridge.py`
- 新建：`tests/test_mmh3director_lifecycle.py`

**接口：**
- 输入：任务 5 的 operation status 与现有 Raylight 清理函数。
- 输出：每个 pass 一次远程调用、确定性的复用/最终清理。

- [ ] 两分镜必须对应两个远程 operation、一次模型加载和相同 Worker PID。
- [ ] 同模型 Refine 的初采和每次 refine 都恰好远程执行一次。
- [ ] 用户显式连接普通自定义 refine MODEL 时透传到 Director 原生路径，并明确不属于 Raylight FSDP。
- [ ] 覆盖 `clear_vram_after_sampling=False/True`，不修改现有默认值。
- [ ] 最终回调最多在完成时使用有限 latent 调用一次，不伪造不存在的中间 x0。
- [ ] 分镜失败不能留下 active operation，也不能污染下一次热运行。
- [ ] 状态轮询有界，普通模式每个 pass 只打印开始和完成，不刷诊断日志。
- [ ] 优先复用现有 Worker 清理和 `RayKill`；只有工作流测试证明现有顺序不能表达时才新增 Finalizer。
- [ ] lifecycle、session cache、shutdown 测试 GREEN 后提交。

### 任务 7：构建 Director 工作流与 API/系统测试

**文件：**
- 新建：`example_workflows/MMH3Director_Windows_V100_FSDP.json`
- 新建：`tests/test_mmh3director_workflow.py`
- 评审后部署：`E:\ComfyUI-py310\ComfyUI\user\default\workflows\MMH3Director.json`

- [ ] 工作流包含现有 initializer、UNET loader、一个新适配节点、原版 Director、CLIP/Video VAE/Audio VAE 和保存节点。
- [ ] 不在 Director 外重复 seed、steps、sampler、scheduler、shift、帧数、FPS、尺寸、百万像素或提示词控件。
- [ ] JSON 测试确认 adapter 到 Director MODEL 连线、没有 XFuser sampler 节点、Director 参数保留。
- [ ] 转为 API 格式，确认依赖和唯一可执行模型路径。
- [ ] 启动 ComfyUI，确认无缺失节点和类型错误。
- [ ] 先跑缩小的单分镜 smoke，确认两个 Worker 采样且本地扩散次数为零。
- [ ] 将评审后的 JSON 复制到用户指定路径，Git 只提交仓库示例。

### 任务 8：实机功能与旧功能回归验收

**文件：**
- 新建：`docs/testing/minimax-h3/MMH3DIRECTOR_ADAPTER_2026-08.md`
- 新建：`docs/testing/minimax-h3/MMH3DIRECTOR_ADAPTER_2026-08.zh-CN.md`
- 精简数据追加：`docs/windows-v100-fsdp-test-results-2026-08.csv`

- [ ] 先用缩小样例验证 Director 的 t2v、i2v、fl2v、r2v、v2v、rv2v；上游本身不支持的模式如实记录。
- [x] 原版单卡方向 1 不再执行 15 秒单段：12 秒已达到 329.46 s/it，虽未 OOM但失去实际使用意义。5 秒、24 FPS、1280×704 的 REF2VA Turbo4 与 FL2VA Turbo8 两组边界均已完成。
- [x] 方向 2 多段压力和方向 3 接合质量完成 864×480、24 FPS、3 秒×5 段的 REF2VA Turbo4 与 FL2VA Turbo8 两组。原计划的两组 5 秒×5 段因 3 秒×5 段已分别耗时 54:48 和 01:12:19、继续测试缺乏实际意义而由用户取消；状态明确记为取消而非通过。原版基线测试阶段于 2026-08-23 结束。
- [ ] 验证上述多分镜矩阵和同模型 Refine。
- [ ] 每次记录模型加载、预处理、首步/稳定/平均 s/it、采样总时长、Video VAE、Audio VAE、拼接保存和端到端耗时。
- [ ] 记录双 GPU 利用率、功率、显存峰值，RAM、commit/pagefile 峰值，P2P、rank 完成差和 Worker PID。
- [ ] 检查 finite latent、尺寸、帧数、FPS、时长、画面时序变化、音频长度，无黑屏/噪声/乱码。
- [ ] 重跑 LTX 2.3、MiniMax I2V Turbo8、REF2VA Turbo4、O6 安全 FP16 和 FP32 对照。
- [ ] 旧工作流各跑三次同条件热运行取中位数；无法解释的采样下降超过 5% 或内存增长即阻止发布。
- [ ] 代理每个 Director pass 只有一次远程 operation、本地扩散为零、没有第二次模型加载。
- [ ] 分别验证热复用和完整最终回收。
- [ ] 原始日志/遥测/视频留在本机；Git 只提交精简表格、结论和证据路径。

### 任务 9：更新版本、README 和发布记录

**文件：**
- 修改：`pyproject.toml`、`README.md`、`README_EN.md`
- 新建：`docs/releases/2026-08-22-o8-mmh3director-adapter.zh-CN.md`
- 新建：`docs/releases/2026-08-22-o8-mmh3director-adapter.en.md`

- [ ] 任务 8 通过后才把版本改为 `1.10.0`。
- [ ] 中英文 README 增加能力矩阵，区分原显式 Ray sampler 和新 Director MODEL 代理路径。
- [ ] 增加从启动、加载 `MMH3Director.json`、Director 调参、复用/清理到双 GPU 验证的用户指南。
- [ ] 明确 Director 管生成参数、Raylight 管分布式参数。
- [ ] 说明主进程 CLIP/VAE/audio 边界、多分镜 RAM/pagefile、同模型 Refine、自定义 refine 限制和预览行为。
- [ ] 完整保留此前 TCC、FSDP、CUDA P2P/NVLink、Gloo、O6 安全 FP16、LTX/MiniMax、启动档位和限制说明。
- [ ] 中英文 release notes 记录实装改动、兼容保证、实测数据、限制和回退标签。
- [ ] 校验中英文链接、路径、节点名和事实逐项一致后单独提交文档与版本。

### 任务 10：最终质检和发布门槛

**文件：**
- 评审：从 `v1.9.0-pre-mmh3director` 起的全部改动。

- [ ] 运行完整 pytest，保留最终精简结果。
- [ ] 运行仓库配置的 Ruff/pre-commit，只修复 O8 相关问题。
- [ ] Director 安装状态和屏蔽发现状态各启动一次 ComfyUI。
- [ ] 同一 ComfyUI 会话先运行一个旧工作流，再运行新 Director 工作流。
- [ ] 检查 `git diff --check`、`git status --short`、仓库大小和忽略内容；未经授权不删除用户文件。
- [ ] 人工检查所有运行时代码，排除通用 schema 变化、静默回退、权重序列化和无界全局副作用。
- [ ] 代码审查、测试、实机验收、中英文文档和 README 对齐全部完成前不得推送。

## 验收总结

只有满足以下条件才算 O8 完成：适配器为显式可选；Director 未修改；旧 JSON 无需迁移；普通 MODEL 精确透传；每个代理 pass 只远程执行一次；双 V100 通过现有 FSDP/P2P 路径参与；媒体正确；复用和清理可预测；旧工作流中位性能不越过退化门槛；中英文 README 同时准确描述新旧功能。
