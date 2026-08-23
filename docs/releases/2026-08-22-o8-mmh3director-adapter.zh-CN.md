# 1.10.0：MiniMaxH3Director MODEL 适配器

[简体中文](2026-08-22-o8-mmh3director-adapter.zh-CN.md) | [English](2026-08-22-o8-mmh3director-adapter.en.md)

## 发布内容

1.10.0 新增显式可选节点 `Ray MODEL Adapter (MMH3Director)`。它把 Director 已准备好的 conditioning、latent、noise、sigmas 与采样参数交给现有双 V100 FSDP2 Worker，并通过 CUDA P2P/NVLink 数据面返回 latent。Director 仍是唯一的 seed、steps、sampler、scheduler、shift、分辨率、FPS、时长、分镜和输出参数面板。

桥只拦截 Raylight 自己的精确代理类型。普通 `MODEL`、旧 `XFuserSamplerCustom`/`Advanced`、旧 JSON 和公共节点 schema 不变。代理无法处理请求时会明确报错，绝不静默回退成本地单卡，也不会让 Director 和 Raylight 各采样一次。MiniMaxH3Director 源码没有修改。

## 已验证能力

- t2v、i2v、fl2v、r2v、v2v、rv2v。
- 同 checkpoint Refine、多段 R2V、热 Worker/FSDP shard 复用。
- FL2VA/REF2VA checkpoint 切换时成对回收并重建 Worker。
- `RayKill` 最终完整释放 Worker、Ray 控制进程、CUDA context、显存和大部分进程提交内存。
- 同 seed 冷/热运行的视频码流、音频码流和末帧解码 MD5 完全一致。

1280×704、124 帧、24 FPS 的 5.167 秒 T2V：双卡冷启动 595.10 秒、42.2 s/it；热运行 470.82 秒、42.0 s/it。上游单 V100 同规格冷启动为 41:21、271.81 s/it；双卡冷启动端到端约 4.17×、采样单步约 6.44×，热交互路径约 5.27× / 6.47×。

旧工作流一冷三热回归覆盖 MiniMax I2V Turbo8、REF2VA Turbo4、LTX 2.3、O6 safe-FP16 和 O6 FP32。所有热采样中位数均未出现超过 5% 的无法解释退化；LTX 第二阶段约 +4.82%，接近门槛但仍通过。

完整数据与方法见 [O8 中文测试汇总](../testing/minimax-h3/MMH3DIRECTOR_ADAPTER_2026-08.zh-CN.md)。

## 关键修复

- Director 文本编码后、Ray 远程采样前，使用 ComfyUI 标准接口卸载主进程预处理模型，避免 GPU0 上 CLIP/VAE 与 rank 0 重叠导致 OOM。
- 对预处理模型卸载、Python GC 与 CUDA cache 清理增加可选分阶段耗时，定位到 REF2VA 端到端离群值来自主进程模型回主存，而非 rank 失步或 P2P 退化。
- 严格验证 operation ID、rank、shape、dtype、finite 和最终 rank 一致性；多段每个 pass 只有一个远程 operation。

## 能力边界

- FSDP 分片对象仍是 Diffusion Model；CLIP/Text Encoder、Video VAE、Audio VAE、预处理、拼接和保存留在主进程。
- 多段会复用 Worker，但 Director/ComfyUI 主进程仍可能保留可重载的 CLIP/VAE 权重；更长视频需关注物理内存与 commit/pagefile，而不能只看两张 GPU 显存。
- 已验证同模型 Refine。需要第二 checkpoint、不同权重或无法表达为标准 `comfy.sample.sample()` 的自定义 refine 路径不在本版本保证内。
- 预览由 Director/ComfyUI 主进程消费返回 latent；远程 Worker 不提供独立逐帧预览。
- 兼容目标为 MiniMaxH3Director 1.0.0、提交 `d655912`。接口不匹配会给出明确错误。

## 兼容与回退

升级不要求迁移旧工作流。若需要完整回到适配前状态，可使用标签 `v1.9.0-pre-mmh3director`。仓库示例为 `example_workflows/MMH3Director_Windows_V100_FSDP.json`，本机用户工作流建议保存为 `ComfyUI/user/default/workflows/MMH3Director.json`。
