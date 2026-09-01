# 1.13.0 / O11：MiniMax H3 Text Encoder 高效注意力

O11 将双 V100 MiniMax H3 TP2 Text Encoder 从可能回退数学 SDPA 的实现升级为显式高效注意力路由。

## 新增与优化

- 新增 Text Encoder 独立选项：`AUTO`、`XFORMERS_GQA`、`TORCH_EFFICIENT`、`MATH_DEBUG`。
- AUTO 在目标 V100 环境优先选择 xFormers GQA；两种正式后端不可用时明确失败，不回退 Math。
- 删除平方级浮点因果遮罩。
- 长序列 xFormers 分批处理独立 KV 组并复用 Q 存储；28k 峰值从 1753.38 MiB 降至 1314.13 MiB。
- MLP 使用 4096-token 分块并提前释放 Attention 临时量，修复最大 conditioning 在 `down_proj` 的 OOM。
- 新 widget 对旧工作流保持默认 AUTO；UI→API 转换也能处理缺失可选字段。
- O11 测试结果和视频统一写入唯一工作树 artifact 与 `output/video/raylight_o11`。
- MiniMax H3 safe-FP16 Diffusion 追加释放打包输入、独立持有 rank 本地序列、缩短 Attention/MLP 临时张量生命周期，并加入 xFormers Ulysses 头/Query 有界分块。
- `fp16_h3_safe` worker 默认开启已验收的块边界 cache trim；显式设置 `RAYLIGHT_H3_BLOCK_CACHE_TRIM=0` 仍可关闭，不依赖启动脚本。

## 实测边界

- 15 秒参考视频、7 张不同图片、7000 字符提示词完整通过，expanded sequence 24,236。
- 冷/热 Text Encoder 为 112.94/112.55 秒，语言阶段 76.08/76.29 秒。
- 用户真实 7图+1视频+1音频 Director 输入展开 12,302 序列，Text Encoder 62.88 秒，双卡分配峰值 9244/9451 MiB。
- 3 秒/480p REF2VA Turbo4 完整采样和媒体质量通过。
- 历史基线中，15 秒 Director 的 0.6 MP 在 RMSNorm OOM，0.4 MP 在 MLP `fc2` OOM。
- 最终0.4 MP固定工作流（365帧、24fps、7图+1视频+1音频、REF2VA Turbo4）完整通过：采样1456.12秒、端到端1823.77秒；对照为1999.41/2332.86秒，采样快27.17%、端到端快21.82%。
- 采样窗口整卡峰值由16215/16230 MiB降至13005/12873 MiB；PyTorch active峰值7.42 GiB，较13.99 GiB失败基线下降约6.57 GiB。active达到下降6 GiB验收，但整卡物理峰值仅下降约3.14/3.28 GiB，未冒充物理口径下降6 GiB。
- 正式输出为864×480、365帧、15.209秒，H.264/AAC全量解码通过且无黑段。与此前正常4步输出逐帧SSIM为0.9416；一步诊断输出不属于质量验收。
- 0.6 MP仍未完成相同端到端验收，不在当前能力声明内。

详细数据见[中文测试汇总](../testing/minimax-h3/TEXT_ENCODER_EFFICIENT_ATTENTION_O11_2026-09.zh-CN.md)。
