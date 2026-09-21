# O14 MiniMaxH3 Director 576424f 兼容实施计划

**目标：** 在不修改 Director 或 ComfyUI 源码的前提下，以最小 bridge 扩展完整适配官方 `576424f`，并给出可追溯的升级前后实机数据。

**技术栈：** Windows、Python 3.12、PyTorch 2.12+cu126、Ray 2.57、2× Tesla V100 TCC、pytest、PowerShell。

**设计：** `docs/superpowers/specs/2026-09-21-o14-mmh3director-576424f-compatibility-design.zh-CN.md`

## Task 1：版本审计、回退快照与升级前基线

- [x] 获取官方 main，确认本地 `52f8fb7` 到 `576424f` 的 13 个提交和真实接口差异。
- [x] 保存 Director git archive、commit/status/依赖元数据和 Raylight 当前部署标记。
- [x] 用固定 O13 工作流跑一次冷启动升级前基准，保存 JSON、CSV、日志和媒体摘要。

## Task 2：先写失败测试锁定新契约

- [x] 为 `on_step_state`、`zero_noise`、`noise_override`、最终状态回传和 rank 一致性添加 bridge/worker 测试。
- [x] 为 SelfLift 两阶段、Semantic Bridge conditioning、FaceRefine 代理采样和旧契约添加回归测试。
- [x] 在当前代码确认新测试按预期失败，而不是测试夹具错误。

## Task 3：更新 Director 并实现最小兼容层

- [x] 快进 Director 到 `576424f`，安装新增的必要 Python 依赖，不修改其源码。
- [x] 扩展 bridge 请求协议和 worker callback，仅回传 SelfLift 所需的最终完整状态。
- [x] 保持预览通道、continuity remask、tiling、SOL/TE model options 与 worker 生命周期不变。
- [x] 运行定向测试，逐项修复真实 API 差异。

## Task 4：工作流与双 V100 实机验收

- [x] 复跑同一 O13 工作流，完成严格升级前后比较。
- [x] 创建最小 SelfLift 验收工作流并跑一次，确认低清/高清均为同一对双卡 worker。
- [x] 对 Semantic Bridge/FaceRefine 按本机资产决定实机或契约验收，明确未具备资产的边界。
- [x] 检查输出帧数、尺寸、帧率、有限值、首中末帧和可播放性；测试后确认进程与显存按策略回收。

## Task 5：文档、版本、部署与同步

- [x] 更新版本号、README 中英文、CHANGELOG/双语发布说明和 O14 测试汇总。
- [x] 运行 Director 定向测试、相关 O11/O12/O13 回归、全量 pytest、Ruff、`git diff --check`。
- [x] 提交唯一源码工作树，部署到 `custom_nodes/raylight`，验证部署提交标记与源码一致。
- [ ] 推送 GitHub，核对本地分支与远端提交一致；Director 保持官方干净状态。
