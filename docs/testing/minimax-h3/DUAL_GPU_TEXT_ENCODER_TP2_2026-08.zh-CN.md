# MiniMax H3 双 V100 Text Encoder TP2 测试汇总

## 文档状态

- 阶段：O10。
- 当前任务：Task 1，冻结单卡 Text Encoder 冷/热基线。
- 状态：完成；实机冷 3 次、热 3 次均通过。

## 测试验证目标

在开发双 GPU Text Encoder 之前，固定同一台机器、同一 checkpoint、同一提示词与同一参考图的单 GPU 基线。后续 TP2 的速度、显存、内存、分页文件与数值质量均以本基线为对照，不引用外部作者数据替代本机结果。

## 测试条件变量

- 工作流来源：当前 `MMH3Director.json`。
- Text Encoder：工作流现用 `CLIPLoader`，类型为 `minimax`、设备策略为 `default`。
- 输入：读取 Director 时间轴第一段的提示词与第一张参考图；正文不复制开发者提示词和素材。
- 执行路径：ComfyUI API 工作流，仅保留 CLIP、VAE、参考图 conditioning 和只读输出节点，不进入扩散模型采样。
- 冷/热定义：每个独立 ComfyUI 进程先执行一次冷编码，再在不卸载 Text Encoder 的情况下执行一次热编码；共 3 个独立进程，得到冷 3 次、热 3 次。
- 防缓存：热编码只改变 latent 帧数，不改变提示词或参考图，强制 conditioning 节点重新执行且保持 token 数一致。
- 计时开关：`RAYLIGHT_TEXT_ENCODER_BENCHMARK=1` 时才安装只读钩子；正常启动不修改 `CLIP` 类，也不打印逐层诊断。

## 测试关注信息

- checkpoint 文件选择器与原工作流 SHA-256。
- token 数。
- Text Encoder 从模型换入开始到编码完成的总耗时。
- Text Encoder 有效时间窗口内的 GPU 0 显存峰值、系统物理内存峰值和分页文件峰值。
- 冷、热各 3 次的中位数，以及三次测试中的资源峰值。

## 数据记录契约

每次有效记录必须包含：`checkpoint`、`workflow_sha256`、`mode`、`run_index`、`token_count`、`te_total_s`、`gpu0_peak_mib`、`ram_peak_mib` 和 `pagefile_peak_mib`。缺字段、负值、非有限数、空 checkpoint 或非法工作流 hash 都会被拒绝。

原始 JSON、CSV、监控采样和进程日志保存在被 Git 忽略的 `.local/artifacts/o10/text-encoder-baseline/`，仓库正文只维护汇总数字与必要的相对附件位置。

## 测试数据结果

有效基线目录：`.local/artifacts/o10/text-encoder-baseline/20260831-142005-single-v100-cold-warm/`。

固定条件：

- checkpoint：`MMH3\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`。
- 工作流 SHA-256：`97defbaff3f4046d53573d2bb36dbfe01e9b4ecb451ed9481a1afe7690a08cd4`。
- token 数：六次均为 221。
- GPU 1 计算利用率峰值：六次均为 0%；因此本表是 GPU 0 单卡计算基线。GPU 1 的 326–336 MiB 仅为进程启动产生的上下文占用，不计为并行计算。

| 温度 | 第 1 次 | 第 2 次 | 第 3 次 | 中位数 | GPU 0 显存峰值 | 系统物理内存峰值 | 分页文件峰值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 冷 | 18.91 s | 20.97 s | 20.76 s | 20.76 s | 13,763 MiB | 35,882 MiB | 760 MiB |
| 热 | 10.37 s | 10.41 s | 10.35 s | 10.37 s | 13,906 MiB | 40,358 MiB | 760 MiB |

两个温度组的 GPU 0 计算利用率峰值均达到 100%。表中的系统物理内存与分页文件是 Text Encoder 有效窗口内的全机已用量，不是单进程独占量；后续 TP2 必须用相同采样定义比较。

第一次实机试跑曾把包含视觉占位项的 token 容器误计为 3。该结果已作废且未进入本表；修复后的单元测试使用真实 MiniMax `(token_or_vision_entry, weight)` 结构，正式六次记录稳定为 221。

## 当前总结

基线工具不会改变 conditioning 或工作流输出；其 begin/complete 事件把资源采样严格裁剪在 Text Encoder 有效区间，避免把参考图 VAE 编码或后续采样误算入 Text Encoder。

Task 1 已关闭。O10 后续验收的本机对照值为：冷中位数 20.76 秒、热中位数 10.37 秒、GPU 0 显存峰值 13,906 MiB。此处只建立单卡基线，尚未声明 TP2 已有性能收益。
