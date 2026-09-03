# MiniMaxH3Director b8f721c 兼容更新

## 更新原因

MiniMaxH3Director 从提交 `d655912` 更新到 `b8f721c` 后，不再由旧路径直接调用 `comfy.sample.sample()`，而是改为：

`sample_single_stage → BasicGuider/CFGGuider → SamplerCustomAdvanced`

Raylight 原代理不是完整的 ComfyUI `ModelPatcher`。新版 Director 因此在进入远程采样前访问 `is_dynamic()` 并报错。仅补一个同名方法还会继续触发其他 ModelPatcher 接口，不能解决根因。

## 实装内容

- 保留 `d655912` 使用的旧 `comfy.sample.sample()` 桥。
- 为 `b8f721c` 增加 `sample_single_stage()` 阶段桥。
- 只拦截 `RayMMH3DirectorModelProxy`；普通 MODEL 完全走 Director 原函数。
- 在适配器节点执行时重新安装阶段桥，消除自定义节点加载顺序影响。
- 透传 seed、steps、CFG、sampler、scheduler、denoise、video/audio shift、noise mask 和可选外接 SIGMAS。
- Worker 支持 Director 的 `apply_shift=False`，并保持旧请求默认启用 shift。
- 远程失败明确报错，不允许静默退回主进程 GPU0。

Director、ComfyUI 核心和模型文件均未修改。

## 运行环境

- Windows，双 Tesla V100-SXM2-16GB，TCC
- Python 3.10.11
- PyTorch 2.7.0+cu126
- xFormers 0.0.30
- comfy-aimdo 0.4.13
- MiniMaxH3Director `b8f721c`

PyTorch 2.8/AIMDO TCC DynamicVRAM 试验没有进入正式环境，其候选 Python、编译树、运行日志和专用仓库文件已清理。

## 验证

- 新旧 Director 合同分别通过能力检测。
- 当前 Director 的 `core_sampling`、一采执行器和 Refine 执行器别名均可安装同一个阶段桥并完整恢复。
- 普通 MODEL 透传、Ray 代理不回退、双 rank 结果一致、NaN/Inf 拦截、外接 SIGMAS、shift 开关、多段复用和清理均有回归覆盖。

最终双卡实机工作流结果在完成部署后补记。

## 边界

- 远程采样仍不提供独立的逐步 TAE 预览流；完成 latent 会回交 Director。
- 未列出的未来 Director 接口需要重新运行合同和实机验收。
- 本次更新只改变 Director 到 Ray worker 的调度入口，不改变 FSDP、P2P、Text Encoder、VAE 或模型精度策略。
