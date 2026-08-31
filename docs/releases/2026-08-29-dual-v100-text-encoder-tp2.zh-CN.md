# 1.12.0 / O10：MiniMax H3 双 V100 Text Encoder TP2

日期：2026-08-31

状态：容量、数值、双卡计算与真实工作流质量通过；性能门槛失败；功能保持显式、默认关闭

## 用户可见变化

- 新增 `Load MiniMax H3 Text Encoder (Dual V100 TP2)` 节点，支持 `tp2` 与 `single_gpu_fallback` 两种显式模式。
- 仅支持 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`；旧 `CLIPLoader`、旧工作流和默认路径不变。
- TP2 中 GPU0 负责视觉塔，50 层语言主干的 q/k/v/o/gate/up/down Linear 在两张 V100 上同时执行，row/column partial 通过 CUDA P2P/NVLink SUM 合并。
- 编码窗口使用两张卡同步租约：先卸载主进程 GPU 模型和 worker 驻留 Diffusion 权重，结束或异常时完整归还；不会让采样和 Text Encoder 同时争抢显存。

## 实装内容

- NVFP4 packed 权重直接 row/column 分片，不把完整 32B 权重反量化为 FP16/FP32。
- 双 CUDA device executor、严格 shape/dtype/device 检查、递增 operation ID 和故障后显式失败。
- 两卡 P2P SUM 原语、CUDA event 重叠 profile、双 rank 租约状态机与采样加载互斥。
- MiniMax H3 conditioning CLIP 代理、普通 I2V/REF2VA 和 MMH3Director 工作流接入。
- 冷/热三次中位数基准、GPU/RAM/commit/pagefile 监控、逐阶段计时、真实视频与五段压力验收。

## 正式数据

| 项目 | 单 V100 | TP2 | 判定 |
|---|---:|---:|---|
| 冷中位数 | 20.76 s | 27.17 s | TP2 慢 30.9%，失败 |
| 热中位数 | 10.37 s | 26.91 s | TP2 慢 159.6%，失败 |
| GPU0 峰值 | 13,906 MiB | 9,467 MiB | 下降 31.9%，通过 |
| GPU1 峰值 | 非计算路径 | 9,449 MiB | 低于 14 GiB，通过 |
| commit 峰值增长 | 基线 | +6.9% | 低于 20%，通过 |
| 页面文件 used delta | 基线 | 0 | 低于 1 GiB，通过 |

完整 50 层中有 350 次双卡 Linear；冷/热加权 kernel 重叠为 96.49%/96.81%，50/50 层达到层级 80% 重叠。热编码固定 100 次 P2P SUM，双向累计 4,453.12 MiB，耗时 70.70 ms，约占语言计算 1.96%。因此性能失败的根因不是 GPU1 未计算或 NVLink 太慢，而是每次约 16.4 秒的 NVFP4 主干分片加载。

## 真实工作流验收

普通 I2V Turbo8、普通 REF2VA Turbo4、Director 864×480 3 秒、Director 960×544 5 秒和 Director 864×480 3 秒×5 段均完成。输出分别为 56、124、73、124、350 帧；全部逐帧可解码、唯一帧数等于总帧数、blackdetect 为 0，音视频无 NaN/Inf。五段复用同一组 worker，GPU 峰值只从首段后约 9,367/9,323 MiB 增到末段 9,447/9,403 MiB；最终进程退出后两卡回到约 10 MiB。

## 能力边界与升级建议

- TP2 是独立 Text Encoder 张量并行，不是 FSDP，也不是两张 16GB 透明合并。
- Diffusion Model 继续由 Raylight FSDP/Ulysses 处理；Video/Audio VAE、Upscaler、拼接和保存不变。
- 不支持其他 checkpoint、Text Encoder LoRA、batch>1、多于两卡、多机或训练。
- 需要降低单卡 Text Encoder 显存峰值时可显式选择 `tp2`；需要速度时使用普通 `CLIPLoader` 或 `single_gpu_fallback`。
- 后续优化必须先解决 NVFP4 loader；不能继续优化只占热语言计算约 2% 的 P2P SUM，也不能用 Diffusion FSDP 的端到端收益冒充 Text Encoder TP2 加速。

完整原始附件索引和 Gate A–F 证据见[测试汇总](../testing/minimax-h3/DUAL_GPU_TEXT_ENCODER_TP2_2026-08.zh-CN.md)。
