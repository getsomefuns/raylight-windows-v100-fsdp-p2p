# O14 MiniMaxH3 Director 576424f 兼容设计

## 目标

将已部署的 MiniMaxH3 Director 从 `52f8fb7` 更新到官方 `576424f`，并让 Raylight Windows 双 V100 FSDP/P2P 路径兼容 SelfLift、Semantic Bridge、FaceRefine、独立二采种子、二采前显存清理及最新版缓存/参考视频修复。升级不得修改 ComfyUI、Director、SOL Attention 或 TE-Speed 源码。

## 上游差异基线

`52f8fb7..576424f` 共 13 个提交、47 个文件，主要分为：

- 缓存与输入修复：外接组逐段见证、参考图尺寸、包导入提示词、参考视频真实时长。
- 采样功能：SelfLift 低清前缀、3D latent 提升、高清尾段；高清空间分块；独立二采 seed。
- conditioning 与质量：Semantic Bridge、FaceRefine。
- 生命周期：二采前显存清理、放大画布跟随一采比例。

## 兼容架构

### 单一采样入口

继续拦截 Director 的 `sample_single_stage`，不复制 Director 调度器，也不维护 Director 分支。现有一采、Refine、FaceRefine 与 SelfLift 的低清/高清阶段统一进入同一 Raylight bridge。

### SelfLift 状态回传

最新版新增 `on_step_state`、`zero_noise` 与 `noise_override`。bridge 将零噪声或指定噪声转换为 worker 请求；rank 0 在采样回调中保留最终完整 `x0` 与 `x`，随采样结果返回主进程。主进程在返回 Director 前调用 `on_step_state`。当前 SelfLift 只读取最后一次状态，因此无需逐步跨进程传输全部 latent，避免显著通信和内存开销。

### Semantic Bridge 与 FaceRefine

Semantic Bridge 在主进程修改 conditioning 后再进入 Raylight，不复制权重到 worker。FaceRefine 的跟踪、裁切、贴回和 VAE 留在 Director；其局部扩散采样通过同一 Ray MODEL 代理进入双 worker。缺少可选权重时必须明确报告未实测，而不能宣称已支持质量效果。

### 回退与部署

更新前把 Director `52f8fb7` 保存为 Raylight `.local/backups/director/` 下的 git archive 和元数据。Raylight 只在唯一源码工作树修改，测试通过并提交后再用部署脚本实装到 `ComfyUI/custom_nodes/raylight`。

## 性能与质量口径

使用同一个 O13 REF2VA Turbo4 + SOL + Refine2 工作流、相同输入、seed、分辨率、帧数、P2P 分块和显存预留，分别在升级前后各跑一次冷启动。记录端到端、Ray 初始化、模型节点、首采、Refine、VAE/保存、两卡显存与利用率峰值、物理内存与 Commit 峰值。兼容升级不得让同口径端到端或采样稳定退化超过 5%，输出必须保持有限值、可播放且无新伪影。

SelfLift 另用固定输入建立功能/收益样本：低清和高清两个阶段都必须由同一对 worker 完成，输出可播放且与关闭 SelfLift 的主体/运动语义一致。只有实测更快或更省显存时才在版本说明中写收益；没有数据时不沿用社区数字。

## 验收标准

- Director 工作树严格位于 `576424f` 且干净；可用快照恢复 `52f8fb7`。
- 新旧 `sample_single_stage` 契约均通过；未知参数和不支持回调失败关闭。
- SelfLift 的低清、高清和最终状态回传均进入双 worker，噪声语义正确。
- 既有一采、Director Refine、连续段 remask、空间分块、实时预览、SOL/TE 路径不退化。
- Semantic Bridge conditioning 可穿过 bridge；FaceRefine 采样契约通过自动化测试，可选资产齐全时完成实机验证。
- 基准报告包含升级前后分阶段时间、显存、内存、Commit、GPU 利用率和媒体质量检查。
- 源码、部署副本、README、双语发布说明、测试汇总、git 提交和 GitHub 远端一致。
