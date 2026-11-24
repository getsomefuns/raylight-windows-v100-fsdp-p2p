# RelayVSR 正式环境迁移评估与设计草案

日期：2026-11-24。状态：研究完成、建议方案待确认；未升级正式核心、未部署、未进行本方案验收。

## 目标和边界

把已隔离验证的双 V100 RelayVSR 能力用于正式 ComfyUI / MiniMax H3 Director，且旧工作流不接 VSR 时不改变行为。接入时按段超分，保留音轨、帧数、缓存、单段重跑和下一段生成能力。仅在本仓库开发 Raylight；不修改 ComfyUI 核心源码，不新增 Director fork，不更改驱动、CUDA/Python/Torch 主版本或外部秋叶环境。

“不修改核心源码”与“升级官方核心版本”是不同操作。推荐方案需要用户确认后才能升级正式核心；不能用本次研究请求冒充升级授权。源码仍只在唯一工作树修改，再部署到 custom_nodes/raylight。

## 已核实的版本和依赖

| 内容 | 正式环境 | 已验证的隔离环境 |
|---|---|---|
| ComfyUI | `62b3c94bd45154f6486c7abf1b9efcacee96ea69` | `2fcb6cef9ab1b6dc556b7cff1765a0724ad5e884` |
| Raylight | 部署标识 `c403c91fad6daf5b80a7ac6954da95c23209ca6c` | 源码 HEAD `720a258a4351381e095c7b86c871b368a1576942` |
| Python / Torch | 3.12.10 / 2.12.0+cu126 | system-site-packages venv，复用相同 Torch |
| comfy-kitchen / comfy-aimdo | 0.2.30 / 0.4.13 | 0.2.42 / 0.6.0 |
| 前端 | 1.48.7 | 1.59.11 |
| Director | `13eba63`，官方 `aff2579` + 本地补丁 | 现有 RelayVSR 基准并未验收 Director + H3 连续生成 |

版本表中正式 Python 依赖来自既有环境记录；实施前必须再次导出实际包清单，不能只依赖此表。正式 Core 和 Director 的 tracked worktree 在本轮只读核查时均干净。Core 所需附属包应遵循锁定版本的 requirements；venv 使用 system-site-packages 并不等于所有包都与正式环境一致。

官方 RelayVSR 于 `2f8943324e1eae89a5ee6c7db93602299246b5f1` 引入，涉及模型、检测、VAE、内存估算和节点等 10 个文件，不是单纯缺一个节点注册。禁止只复制 nodes_wan.py 来伪装支持。

官方来源：

- [RelayVSR 原生支持提交](https://github.com/Comfy-Org/ComfyUI/commit/2f8943324e1eae89a5ee6c7db93602299246b5f1)
- [后续 H3 Image / 模型接口变更](https://github.com/Comfy-Org/ComfyUI/commit/ede4e5fe8d15445c6fcd6828d70b202c22f9502c)
- [既有双卡验收报告](../../reports/2026-11-24-relayvsr-dual-flash.zh-CN.md)
- [既有完整测试计数和失败 ID](../../reports/data/2026-11-24-relayvsr-validation.json)

## 三个实际阻碍

1. **旧核心没有原生 RelayVSR。** 复制 Raylight 节点不能让旧核心加载其 UNET 和 Flash VAE。现有双卡节点只加速 Flash，RelayVSR 采样仍走官方单卡实现，不是双卡 H3/FSDP 模型。
2. **新版 H3 接口尚未适配。** 隔离核心测试为 865 passed / 14 failed / 1 skipped / 35 subtests；旧核心 869 passed / 11 skipped / 35 subtests。13 个失败经过同一个旧 PackedLayout(frame_count=...) 构造点，另一个 forward 契约用例缺 video/audio denoise_mask。不能将这些概括成 14 个互不相关的产品故障，也不能仅改测试或删除参数来消除红灯。必须核查布局、分段调制、掩码对时间步和输出的完整语义，尤其段间引导和 Refine。
3. **VSR 与活跃 Ray 会话互斥。** relayvsr/local.py 在 ray.is_initialized() 时直接拒绝；Director H3 完成后通常保留 Ray 会话。Director 自己的 VSR 阶段目前调用原生 VSRDecode，尚未自动接入我们的双卡 Flash。因此独立 VSR 通过不等于 Director 内已双卡加速。

现有 TP2 GPU 租约是可复用基础，但目前名称、状态和清理语义以 TE 为中心。接入 VSR 前必须验证 actor 的 driver/session/代次、两张物理 GPU 的映射、任务互斥和异常归还；不能直接去掉 is_initialized 检查。clear_sampling_vram 会记录并吞掉部分清理异常，新准入还必须验证实际释放结果，不能只信其 True 返回值。

## 可选路线和推荐

### A：锁定官方核心升级 + Raylight 向前/向后兼容（推荐）

在新的生产候选隔离目录重建正式 13 个节点和原依赖基线；使用已经测过 VSR 的 `2fcb6ce` 官方核心，不追随浮动最新版。首先只验证旧 H3 图，修复我们代码与新核心的差异；再加入 VSR 和 Director 桥接。旧核心也继续测试，防止回滚后 Raylight 不可用。若发现更早官方版本确能显著缩小兼容面，需要单独记录版本选择证据，不能临时无记录换基线。

优点：一个正式界面、原生模型加载、内存帧传递，不增加长期维护的外部服务。代价：需要官方核心及其附属依赖升级，存在模型数学语义变化，必须完成旧功能回归才能迁移。DynamicVRAM 仍保持禁用，不承诺 V100 因升级获得该能力。

### B：冻结正式核心，隔离 VSR 后处理服务

正式 Raylight 只增加隔离调用接口，新核心/依赖保留独立进程。避免升级正式核心，但引入服务存活、任务取消、帧传输、临时空间、GPU 共享和跨进程结果校验等工作。若按段调用，还要处理 Director 的缓存与错误恢复；不能以有损 MP4 中转冒充既有逐帧完全一致。仅在不接受核心升级或 A 失败时另行设计，不在 A 中悄悄混入。

不推荐把官方 RelayVSR 核心复制进 Raylight：这会长期维护模型检测、加载和底层 API 的第二套实现，背离最小改动目标。

## 推荐集成方式

- 普通 H3 图 / Director 未连接 vsr：不获取 VSR GPU 租约，不加载 VSR 权重，不改采样、解码、缓存指纹或输出目录。
- Director 连接其官方 RelayVSR 配置节点：沿用 model、vae、seed、steps、cfg、sampler、scheduler、denoise、detail；不偷偷覆盖用户参数。只对 Ray MODEL 代理所属 Director 调用启用双卡 Flash 桥接，原生单卡 Director 保持原路由。
- H3 采样 → H3 解码 / Refine / FaceRefine → 获取同一会话的两卡独占租约 → 释放 GPU 上暂不使用的 H3/TE/VAE → 单卡 RelayVSR 采样 → 双卡 Flash → 释放 VSR GPU 权重与临时缓冲 → 归还租约 → 导出/下一段。
- 不通过 ray.shutdown、销毁 H3 actor 或清空 Director 分段缓存来腾空间。保留 H3 的 CPU 模型对象供下一段复用；它再次上 GPU 的耗时必须计入全流程，不能只报告 Flash 加速。
- 复用已有窗口规划、FP32 Flash 和精确合并；默认 keep_workers=false。不把之前撤回的 Ray actor Flash 原型重新列为正式能力。
- 新桥接建议独立放在 integrations/mmh3director/relay_vsr.py；不全局替换所有 VAE.decode。绑定的 Director 函数别名需一并识别，未知上游契约明确报错，不静默绕过。
- 核验模型/倍率和 GPU 所有权后再卸载模型。取消、OOM、worker 失效、第二卡失败和写文件失败都应先清理本次 VSR 资源再归还租约；清理失败则禁止下一段抢先使用，保留原错误。

## 验收与发布门禁

以下均是待执行，不是本次已通过结果：

1. 旧/新核心跑完整 Raylight 测试，消除上述 14 个真实兼容失败；新增掩码/引导/曲线与原生参考的小模型数值对照，不能新增 skip/xfail 来冒充通过。
2. 候选环境执行真实节点图：非 Director H3、Director FL2VA/Ref2VA 4/8 步、TP2 TE、LoRA/段 LoRA、SelfLift、Refine、FaceRefine、双卡 H3 VAE；各可组合特性需代表用例，不要求全部组合穷举。预览、静音/有声保存、缓存复用与只重跑指定段均覆盖。
3. 不接 VSR 时原缓存指纹相同；接入/改变倍率/detail/seed 后成品正确失效，一采缓存仍可复用。缓存状态检查不能解码 FFV1 或阻塞事件循环，且不能假报匹配。
4. 同素材/seed/实际画布/帧数，候选环境先跑 Director 不带 VSR，再跑带 x2；至少连续 3 段并做缓存重跑。验证 H3→VSR→H3 无死锁、无损坏、无旧结果串入和无持续增长的空闲 GPU 模型池。
5. 独立 VSR 回归 480p/10秒/x2、1MP/5秒/x2 和 x4；不把未重测的 480p/10秒/x4 列为通过。对相同 RelayVSR latent 的双卡 Flash 要保持所有 raw 帧与单卡参考相同，所有成片完整可解码，无丢帧、错序、音轨截断、音频重采样或变速。
6. 对照正式旧环境的旧图冷/热数据；TE、采样、H3 VAE、VSR 采样/Flash、保存、端到端分别计时。500ms 记录整卡 MiB/利用率/功耗及进程树工作集/私有提交、系统 RAM/commit/分页；注明短瞬峰可能漏采。局部 Torch active 不能替代整卡峰值。未测试项目如实列为未验收。
7. VSR 480p/10秒/x2 的既有孤立基准为默认释放池 Flash 71.363s、节点全流程 110.428s，物理观察峰值 GPU0/1 11249/5991MiB。生产候选独立图按相同条件复测，三次中位数不得比该基准退后超过 10%；Director 内必须另列 H3 offload/reload、残留 CUDA 上下文和端到端开销，不套用独立图预算/加速比。旧图性能若退后超过 5%，应分析和修复或提交明确取舍，不自动视为无影响。
8. 正式切换前做完整可回滚备份：核心 tracked/untracked 修改、包清单和安装包/可恢复 Python 环境、Director bundle+补丁、Raylight 部署清单、启动参数/环境变量；不复制大模型/用户缓存冒充环境备份。已确认的恢复命令和文件校验落盘，回滚隔离演练通过后才能切正式。
9. 正式环境先停队列并完成备份，部署同一验收提交、校验全部文件和锁定依赖；正式烟测旧 Director 与 VSR 后保留恢复点。修改 README、中英文升级记录、测试报告和复现图；不上传素材、权重、日志和环境备份，最后按授权同步 GitHub。

## 状态与下一步

本轮仅核对现状和形成方案，没有新增产品代码、修改依赖、启动 GPU 测试或部署正式环境。推荐先确认路线 A 的“官方核心升级”边界；确认后再将上述门禁拆为可执行任务计划，并在新隔离候选中开发。旧 RelayVSR 隔离环境继续保留，可供手测和回退对照；它不是第二个 Raylight 源码工作树。
