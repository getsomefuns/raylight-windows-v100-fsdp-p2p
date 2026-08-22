# Raylight 1.10.0 / O8：MMH3Director MODEL 代理兼容层设计

[English](2026-08-22-mmh3director-model-adapter-design.md) | [简体中文](2026-08-22-mmh3director-model-adapter-design.zh-CN.md)

## 1. 目标

在不修改 `ComfyUI_MiniMaxH3_Director`、不破坏 Raylight 已有 LTX 与 MiniMax H3 工作流、也不新建仓库或长期维护分支的前提下，让 MiniMax H3 Director 把它内部的一次 `comfy.sample.sample()` 调用交给当前 Raylight Windows 双 V100 FSDP + CUDA P2P Worker 执行。

Director 继续作为唯一的生成参数面板。用户仍然在 Director 中设置 seed、steps、CFG、sampler、scheduler、shift_video、shift_audio、任务类型、分镜组、帧数、FPS、分辨率、百万像素、连续性和导出选项。Raylight 新节点只负责把 `RAY_ACTORS` 适配成 Director 能接收的 `MODEL`。

## 2. 版本定位

- 目标版本：`1.10.0`。
- 阶段代号：`O8`。
- 仓库和分支保持不变：`raylight-windows-v100-fsdp-p2p` / `windows-v100-fsdp-p2p`。
- 这是向后兼容的可选功能，不是 Director 专用重写，也不是对现有 XFuser/FSDP 采样路径的替换。
- 当前 `1.9.0` 已验收状态在开发前打 Git 标签 `v1.9.0-pre-mmh3director`，作为可回退坐标；标签不是新分支。
- O8 的首个兼容坐标是 `AIMixer/ComfyUI_MiniMaxH3_Director` `1.0.0`、提交 `d655912262bcd70ad43c3a431879be9539d8e054`。运行时仍以接口能力检查为准，因为压缩包安装可能没有 Git 元数据。

## 3. 节点命名和接口契约

### 3.1 新节点

- 内部节点 ID：`RayMMH3DirectorModelAdapter`。
- 前端显示名：`Ray MODEL Adapter (MMH3Director)`。
- 输入：`ray_actors: RAY_ACTORS`。
- 输出：`model: MODEL`。
- 分类：`Raylight/integrations/MMH3Director`。

用户给出的 `MMH3Dirctor` 少了字母 `e`。公开节点名和文档统一采用项目原名的正确拼写 `MMH3Director`，避免形成永久 API 拼写错误。

### 3.2 现有节点冻结

以下节点的 ID、显示名、输入输出类型、输出顺序、默认值和工作流连线必须保持不变：

- `RayInitializer`；
- `RayInitializerAdvanced`；
- `RayUNETLoader`；
- `XFuserSamplerCustomAdvanced` 及现有 Ray 采样节点；
- `RayKill`、现有诊断节点及 O6 节点。

特别禁止给 `RayUNETLoader` 追加 `MODEL` 输出。新增适配节点多占一个工作流节点，但它把旧加载器接口完全隔离，长期回归风险更低。

### 3.3 Initializer 命名规则

本设计不需要修改 `RayInitializer`：MMH3Director 的兼容检查和代理创建全部由适配节点完成。

如果实现阶段用测试证明必须存在 Director 专属的初始化参数，只允许新增：

- 内部 ID：`RayInitializerMMH3Director`；
- 显示名：`Ray Initializer (MMH3Director)`。

不得给通用 `RayInitializer` 添加只服务于 Director 的参数，也不得给通用节点添加 Director 后缀。

## 4. 执行架构

### 4.1 旧路径保持原样

```text
RayInitializer
  -> RayUNETLoader
  -> RayBasicGuider / RayBasicScheduler / XFuserSamplerCustomAdvanced
  -> 现有 Ray Worker common_ksampler/custom_sampler_advanced
```

没有接入新适配节点的工作流不会创建代理，也不会进入新分发逻辑。

### 4.2 Director 新路径

```text
RayInitializer
  -> RayUNETLoader
  -> Ray MODEL Adapter (MMH3Director)
  -> MiniMaxH3Director.model
  -> Director 准备 conditioning / latent / noise / 参数
  -> comfy.sample.sample(RayMMH3DirectorModelProxy, ...)
  -> Raylight 类型门控桥接器
  -> 两个 Ray Worker 只采样一次
  -> 主进程收到 sampled latent
  -> Director 继续 VAE、音频、拼接和保存
```

每个 Director 分镜对应一次远程采样。启用同模型 Refine 时，每个 refine pass 对应额外一次明确的远程采样。不得同时在主进程再运行一次扩散模型。

### 4.3 通用扩展层

新代码分为三层：

1. `RayModelProxy` 基础协议：描述轻量 MODEL 代理必须支持的 ComfyUI 元数据和 patch 行为。
2. `comfy.sample` 类型门控桥：只识别 Raylight 自己的代理，其他 MODEL 原样传给原函数。
3. `MMH3Director` 适配器：把 MiniMax H3 Sigma Shift、latent/noise/conditioning 和 Director 参数转换成现有 Worker 请求。

未来适配其他一体式节点时新增适配器，不向 FSDP、P2P 或采样核心散布模型名称判断。

## 5. MODEL 代理能力

代理不包含扩散模型权重，不在主进程占用一份 UNET 显存。它只保存：

- 当前 `ray_actors` 句柄；
- 会话/模型身份和 generation ID；
- MiniMax H3 `sampling_settings`；
- `latent_format` 所需的通道数、空间/时间降采样信息；
- `model_options` 与 Director 写入的 `transformer_options`；
- Sigma Shift 产生的 `model_sampling` 对象补丁；
- 诊断计数和生命周期策略。

代理实现 Director 和 ComfyUI 在进入采样前实际调用的最小接口：

```python
class RayMMH3DirectorModelProxy:
    def clone(self) -> "RayMMH3DirectorModelProxy": ...
    def get_model_object(self, name: str): ...
    def add_object_patch(self, name: str, obj) -> None: ...
```

并提供 `model.model_config`、`model.latent_format`、`model_options`、`load_device`。所有元数据由 Worker 从已经加载的模型导出为小型字典，主进程重建轻量 facade；禁止通过 Ray 复制模型权重或完整 state dict。

## 6. 采样桥与失败策略

桥接器保存原始 `comfy.sample.sample`，只安装一次：

```python
def raylight_sample_dispatch(model, *args, **kwargs):
    if isinstance(model, RayMMH3DirectorModelProxy):
        return dispatch_mmh3director_sample(model, *args, **kwargs)
    return original_comfy_sample(model, *args, **kwargs)
```

要求：

- 普通 ComfyUI `MODEL` 必须逐参数透传给原函数；
- 重复导入或节点重载不能多重包装；
- 代理请求不兼容时立即报错，禁止回退到本地单卡；
- Director 未安装、版本不支持或未使用适配节点时，Raylight 仍能正常导入和运行；
- 可选依赖按需探测，不在模块导入阶段强制 import Director；
- 错误信息必须包含当前 Director 版本、支持范围、失败字段和恢复建议。
- 对无法取得 Git commit 的安装，检查 `MiniMaxH3Director` 的 `MODEL` 输入、`sample_single_stage` 参数签名，以及 `comfy.sample.sample` 调用契约；任一不一致都关闭式失败。

桥接器接收 Director 已经准备好的 noise、latent tensor、conditioning、steps、CFG、sampler、scheduler、denoise、noise_mask、seed 和 callback，然后构造与现有 `common_ksampler` 等价的 Worker 请求。Worker 返回的 rank 结果必须先通过有限值和 rank 一致性检查，再把 rank 0 latent 返回 Director。

## 7. 参数所有权

Director 是以下参数的唯一权威来源：

- seed 与生成后控制；
- steps、CFG、sampler、scheduler；
- shift_video、shift_audio；
- task type；
- Prompt Group、全局提示词和 conditioning；
- total_frames/duration、FPS；
- width、height、aspect、megapixels；
- i2v/r2v/ref/video 输入和连续性；
- export_source_images、export mode；
- Refine 的次数、denoise、步数和相同模型模式。

Raylight 只拥有分布式参数：GPU 数量、FSDP、CPU offload、Ulysses/ring/CFG 拓扑、P2P、attention、mmap、诊断和 Worker 生命周期。不得为了适配 Director 再创建一组 seed、steps、FPS 或分辨率节点。

## 8. 多分镜、Refine、预览和生命周期

- Worker 默认跨分镜复用，避免每段重新加载模型。
- Director 的 `Clear VRAM between segments` 只影响 ComfyUI 主进程；Raylight 的 `clear_vram_after_sampling` 负责 Worker 采样后显存。
- 当前分支默认行为保持 `Reuse Workers`。
- `RayFinalize`/最终销毁能力作为可选终端路径，不改变旧工作流。整条 Director 工作流结束后才允许销毁，不得在相邻分镜之间销毁。
- 同一个代理模型的 Refine 必须继续走双 Worker。自定义第二扩散模型在没有第二个 Ray 代理时属于 Director 原生本地路径，首版文档必须明确；禁止把它宣称为 FSDP。
- 首版必须保证最终视频正确。实时 TAE 预览若无法安全跨 Ray 回传，必须显示明确的“远程采样仅阶段进度”提示，而不是假装提供逐步预览。
- 后续进度桥使用 Worker 状态轮询，不能序列化 Director 的 Python callback 闭包到 Ray Worker。

## 9. 兼容性与发布门槛

### 9.1 必须回归的旧功能

- LTX 2.3 已验收工作流；
- MiniMax H3 I2V Turbo8；
- MiniMax H3 REF2VA Turbo4；
- O6 `fp16_h3_safe` 与 FP32 对照；
- 冷启动、热启动、相同模型复用和切换 checkpoint；
- P2P/FSDP preflight；
- `clear_vram_after_sampling`、RayKill 和 Worker 回收；
- 不安装 Director 时的根包导入；
- 安装 Director 但不连接适配节点时的旧路径。

旧节点 schema 和现有示例工作流 JSON 必须通过快照测试。旧工作流无需迁移或重新保存。

### 9.2 新功能验收

- 单分镜 t2v、i2v、fl2v、r2v、v2v、rv2v 按 Director 当前支持范围逐项验证；
- 多 Prompt Group/多分镜按选择运行；
- 同模型 Refine 一次和多次；
- 每个采样 pass 恰好一次远程调用，本地扩散采样为零；
- GPU0/GPU1 均有持续计算，FSDP/P2P 诊断成立；
- sampled latent、视频和音频均无 NaN/Inf；
- 无黑屏、全噪声、首帧后乱码；
- 尺寸、帧数、FPS、时长、音频时长和源图导出符合 Director 参数；
- 热运行能复用 Worker，最终销毁模式能等待 PID 退出并恢复显存、工作集和私有提交量。

### 9.3 性能回归

旧工作流采用相同输入和参数进行三次热运行，取中位数。若采样中位数相对已记录基线下降超过 5%，或者显存/RAM/commit/pagefile 峰值出现无法解释的增长，禁止发布。

Director 适配的初始速度目标不另造数字：与相同 MiniMax H3 模型、分辨率、帧数、步数、提示词和输入素材的现有 Raylight 工作流对比。适配器本身除 Director 原有预处理外，不应增加第二次模型加载或第二次采样。

## 10. 文档和发布内容

实现和实机验收完成后同步修改：

- `README.md`：中文项目定位、能力矩阵、Director 用户指南、生命周期、限制和测试结果；
- `README_EN.md`：与中文逐项一致；
- 中英文 release notes；
- 中英文维护型测试汇总；
- Director 示例工作流，使用仓库可再分发的提示词和输入引用，不提交模型或运行日志；
- `pyproject.toml` 版本更新为 `1.10.0`。

README 必须同时保留此前版本的核心说明：Windows 双 V100 TCC、FSDP 权重分片、CUDA P2P/NVLink、Gloo 控制面、O6 安全 FP16、LTX 和 MiniMax H3 原工作流、RAM/pagefile 关系、启动参数和已知限制。新增 Director 章节不能覆盖或弱化这些内容。

## 11. 非目标

- 不修改或 fork `ComfyUI_MiniMaxH3_Director`；
- 不让 Raylight 和 Director 各采样一次；
- 不让 Ray MODEL 代理携带完整模型权重；
- 不改变 Video VAE、Audio VAE、CLIP 默认仍在主进程运行的边界；
- 不声称 Windows Gloo 等价于 NCCL；
- 不在 O8 中顺带实现 O7 LTX 安全 FP16；
- 不为本次功能建立新仓库或长期开发分支。
