# 2026-11-24 Director aff2579 更新与兼容核查（非新功能完整验收）

## 已完成

- Director 官方基线从 `4c6245e` 更新到 `aff2579`：RelayVSR、素材组批量删除、大导演包导出线程池/素材去重/下载优化等；没有更换为第三方 Director Pro 分支。
- 本地 PR231 缓存轻量检查、请求取消/去重及 FL2V 指纹修正重放为 `131b714`。
- 更新前发现的 5 个修改文件先备份再保存，重放为 `0b0daef`：导出尾帧引导、左右选脸、tracking_reach 与相应缓存身份。没有用 stash 丢弃、reset 或整目录覆盖。
- 可重放源码补丁见 [patches/director](../../patches/director/README.md)。完整原目录副本、未提交差异、前后 Git bundle 保存在唯一工作树 `.local/backups/director/20261124-aff2579/`，不上传。
- Raylight 运行时代码仍为 1.18.5；本次没有给 H3 代理增加不必要的全局拦截或宣称支持新的 FSDP 模型。

## 测试

| 检查 | 结果 | 边界 |
|---|---|---|
| Raylight 全量回归 | 830 passed，35 subtests passed，6 条上游警告 | 不等于完整视频生成 |
| PR231 缓存回归 | 7/7 | 含 R2V/I2V/FL2V 指纹、禁止状态检查解码素材、正常生成解码、缺文件、空选择 |
| Refine 前端逻辑 | PASS | 去重、取消旧请求、只显示最新结果、清理、空选择；不是浏览器端到端计时 |
| Director 官方音频回归 | 2/2 | 真实采样率与回退重采样 |
| JavaScript 语法 | PASS | minimax_refine.js |
| 补丁恢复重放 | PASS | 从干净 aff2579 重放两份补丁，10 个运行时文件与当前安装哈希一致 |

补丁恢复测试首次使用 Windows tar 时遇到上游中文样例文件名编码错误；改用 Python 标准归档解压后完整重跑通过，不修改上游样例文件。

详细本地记录：`.local/artifacts/director-aff2579-20261124/`。未启动完整 GPU 工作流、未生成新视频、没有速度或画质提升结论。当前 ComfyUI 服务未运行。

## VSR 阻碍和正确接法

新增的是 **RelayVSR**，不是 Refine 中原先的 NVIDIA RTX VSR。它在 Refine/FaceRefine 后，对解码画面使用独立 RelayVSR MODEL + 同倍数 Flash VAE，走 conditioning → KSampler → VSRDecode，音频不变。H3 的 Ray MODEL Adapter 不能接到该 MODEL 输入口来冒充 RelayVSR；本轮不承诺 VSR 双卡 FSDP。

当前 ComfyUI 核心 `62b3c94b` 缺少官方调用的 `RelayVSRConditioning`、`VSRDecode` 和模型后端，模型库也缺 RelayVSR 权重。因此只有 Director 节点更新完成，不能称 VSR 已可用。下一步需用户明确允许更新 ComfyUI 核心后，选择兼容 cu126/V100 的固定版本、准备权重并隔离验证，再进行端到端验收。未经授权不修改核心。

## Pika 核查

此次 AIMixer 官方 main 的新增提交中没有 Pika 专用代码。Pika H3 Flex 官方提供按 FL2VA/Ref2VA 匹配的 LoRA 和精确六步采样配置，明确指出普通六步 Euler 不等价；并非通用 Attention 加速器。

来源：https://huggingface.co/Pika-Labs/Pika-H3-Flex 。本机未找到 Pika 权重，不能宣称已实测兼容或提速。应按其官方 ComfyUI 模板核查 sampler/sigmas，不能只选六步或套用 Turbo4 的配置。若用户指的是第三方 Director Pro 中的 Pika 功能，需要给出确切项目，不能悄悄把官方节点替换为另一个分支。

## 如何保留修复

独立审查另发现两项待修：状态检查未包含 VSR/FaceRefine 的完整指纹，可能误报成片缓存匹配（执行器实际缓存判断仍使用完整计划）；元数据素材路径需补 input 目录边界校验。这两项不在本轮 7 项缓存回归覆盖内，尚未修复。导出的补丁是当前安装的可恢复快照，不是完整新功能验收版；后续修复后需重新导出。没有声称 VSR/Pika 已交付。

保留 Director 的 `local/cache-status-pr231` 本地分支，通过 fetch + rebase 更新；每次先做完整备份，并运行缓存/音频/Raylight 回归。不要对此节点使用 Manager 的 Update All、强制更新或重新安装。补丁文件提供其他机器的恢复途径，但未来代码冲突仍需人工审核。旧分段缓存和工作流本次未删除或改写。

需求和后续验收见 [中文计划](../superpowers/plans/2026-11-24-director-pika-relayvsr.zh-CN.md)。
