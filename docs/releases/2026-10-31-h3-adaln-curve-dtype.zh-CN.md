# Raylight 1.18.5：MiniMax H3 压缩时间曲线精度兼容

## 原因和修复

部分曲线压缩模型将 `adaln_t_table` 保存为 BF16。加载后该 buffer 可能保留 BF16，而时间坐标和插值系数为 FP32，导致 USP forward 的 `torch.lerp` 报 `expected dtype ... BFloat16 for weight but got dtype float`。这里的 weight 是插值系数，不是整个模型的权重。FP16 时间表也有相同问题。

只修改 Raylight 的 `use_adaln_curves` 分支：通过现有 `cast_to` 在使用点取得 FP32 时间表。保留 FP32 时间坐标、插值与调制输入；不修改 checkpoint buffer，不把系数降为 BF16，不改变量化主权重、安全 FP16、Attention、分片或缓存策略。旧 `time_embedder` 分支不变。不修改 ComfyUI、Director、模型文件或其他节点。

## 三个辅助张量的核查

对触发错误的 `10Eros_Max_h3_TURBO-hybrid_beta3_int8_convrot_skip_edges.safetensors` 仅读取所需小张量，得到：

| 张量 | 形状 | 用途 |
|---|---|---|
| `adaln_t_table` | 1025×8 | 实际推理使用的低秩时间坐标 |
| `adaln_basis` | 8×2688 | 压缩基底，当前模型结构没有此运行时参数 |
| `adaln_mean` | 2688 | 压缩均值，当前模型结构没有此运行时参数 |
| `silu_t_emb_grid` | 1025×2688 | 原始高维时间嵌入网格 |

第一层及最终层的 AdaLN 线性权重均为 8 维输入，与低秩时间表一致。`table @ basis + mean` 与原始网格的相对 L2 差约 0.6414%，符合这些张量的压缩对应关系；这是 checkpoint 的现有差异，不是本修复引入的误差，也不是生成质量评测。

三项辅助数据不是当前推理结构需要加载的参数，不能重复展开到 2688 维后再送入 8 维线性层。保留现有 unexpected-key 警告，不伪造加载成功、不屏蔽未知权重。三项合计约 5.30 MiB；实际时间表的 FP32 临时张量仅 32,800 字节（约 32.0 KiB），不会额外复制整个模型。

## 验证结果和边界

- 环境：Windows、Python 3.12.10、PyTorch 2.12.0+cu126、双 V100-SXM2-16GB。
- 修复前新增测试：6 failed / 4 passed，准确复现 BF16 和 FP16 插值错误。
- 修复后新增 13 项测试通过：BF16/FP16/FP32 曲线、端点/分数坐标、buffer 不变、旧时间嵌入分支、辅助数据不影响输出，以及每张 V100 的小模型 forward。
- 小模型 fixture 不含 Transformer blocks，隔离了本次修改的嵌入和输出路径；不代表完整双卡 FSDP 生成测试。
- 定向兼容回归：57 passed，2 subtests passed；覆盖 USP、安全 FP16 与 FSDP 量化路径。
- 全量回归：830 passed，35 subtests passed；6 条上游弃用/原型 API 警告，无失败。首次沙箱运行受管理员临时目录权限影响，改用独立测试目录重跑后全部通过，未修改程序来规避失败。
- 真实 checkpoint 的时间表与第一层调制权重在两张 V100 上测试，均无 NaN/Inf，输出与 CPU FP32 参考最大绝对差为 0。
- 本次没有重跑完整用户视频，没有声称视频质量、端到端速度或峰值显存获得提升。完整原始记录保存在本地忽略目录 `.local/artifacts/adaln-curves-20261031/`，不打包进部署或上传。

## 使用

工作流与启动参数不变。部署后需在安全时机完整重启 ComfyUI，使 Python 模块及 Ray worker 使用新代码；仅刷新网页不足以更新后端。无需清除 Director 分段缓存。本次不自动重启正在运行的 ComfyUI，不主动推送 GitHub。
