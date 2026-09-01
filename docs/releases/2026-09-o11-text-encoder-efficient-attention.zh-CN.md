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

## 实测边界

- 15 秒参考视频、7 张不同图片、7000 字符提示词完整通过，expanded sequence 24,236。
- 冷/热 Text Encoder 为 112.94/112.55 秒，语言阶段 76.08/76.29 秒。
- 用户真实 7图+1视频+1音频 Director 输入展开 12,302 序列，Text Encoder 62.88 秒，双卡分配峰值 9244/9451 MiB。
- 3 秒/480p REF2VA Turbo4 完整采样和媒体质量通过。
- 15 秒 Director 在后续 Diffusion 第一轮 OOM；0.6 MP 失败于 RMSNorm，降到 0.4 MP 后推进到 MLP `fc2` 但仍 OOM。本版本不宣称该最大采样可生成视频。

详细数据见[中文测试汇总](../testing/minimax-h3/TEXT_ENCODER_EFFICIENT_ATTENTION_O11_2026-09.zh-CN.md)。
