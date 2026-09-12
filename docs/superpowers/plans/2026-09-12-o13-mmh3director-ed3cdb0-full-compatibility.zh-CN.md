# O13 MiniMaxH3 Director ed3cdb0 完整兼容实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 更新 Director 到 `ed3cdb0`，让 Raylight 完整支持新版连续段、Director Refine、空间分块和逐步预览。

**Architecture:** bridge 将新版参数转换为稳定的 worker 请求协议；连续段回调在实际 worker MODEL 上重建；空间分块在 worker sampler 上执行；rank 0 通过非阻塞有界队列发送轻量预览事件。

**Tech Stack:** Python 3.12、PyTorch 2.12+cu126、Ray 2.57、ComfyUI、pytest、PowerShell、双 Tesla V100 TCC。

**Spec:** `docs/superpowers/specs/2026-09-12-o13-mmh3director-ed3cdb0-full-compatibility-design.zh-CN.md`

## Global Constraints

- 唯一源码工作树为 `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`。
- 部署目录只能通过 `scripts/deploy-to-comfyui.ps1` 更新。
- 不修改 ComfyUI、Director、SOL Attention 或 TE-Speed 的功能源码。
- 所有生产代码遵循测试先失败、实现后通过的 TDD 顺序。
- 实机验收只运行一次4步工作流，除非失败需要定位。

---

### Task 1: 锁定新版 Director 契约与连续段协议

**Files:** `tests/test_mmh3director_bridge.py`、`src/raylight/integrations/mmh3director/bridge.py`

**Interfaces:** 输入新版 `sample_single_stage` 关键字；输出 `serialize_director_after_shift(callback) -> str | None` 和扩展的 stage 请求。

- [ ] 添加测试：完整签名、remask 能力标识、未知回调失败、新参数进入所有 worker 请求。
- [ ] 运行 bridge 测试并确认因功能缺失而失败。
- [ ] 扩展 bridge 契约和请求字段，只复制 remask 所需 latent 元数据。
- [ ] 重跑 bridge 测试并确认通过。

### Task 2: 在 worker 恢复 remask 并执行空间分块

**Files:** `tests/test_mmh3director_worker.py`、`src/raylight/distributed_worker/ray_worker.py`

**Interfaces:** 消费 `after_shift_capability`、`director_latent` 和 tile 配置；输出实际 MODEL remask 与可恢复的 sampler wrapper。

- [ ] 添加测试：shift 后应用 remask、tile 参数、异常恢复、关闭分块保持旧路径。
- [ ] 运行 worker 测试并确认新测试失败。
- [ ] 重建受支持回调；仅分块时构造 sampler、安装 Director wrapper并调用 `sample_custom`。
- [ ] 重跑 worker 测试并确认通过。

### Task 3: 增加非阻塞逐步预览通道

**Files:** 新建 `src/raylight/integrations/mmh3director/preview_transport.py` 和 `tests/test_mmh3director_preview_transport.py`；修改 bridge、worker。

**Interfaces:** 输出 `extract_preview_event(...)`、`publish_preview_nonblocking(...)` 和 driver 队列生命周期。

- [ ] 添加测试：NestedTensor 仅取视频、最多16帧、CPU结果、满队列不阻塞、关闭预览零传输。
- [ ] 运行新测试及受影响测试并确认失败。
- [ ] 实现有界队列、rank 0 callback、等待期间排空和最终回调。
- [ ] 重跑三个测试文件并确认通过。

### Task 4: 更新 Director 并验证真实接口

**Files:** Director 仓库、`tests/test_mmh3director_optional_dependency.py`

- [ ] 停止 ComfyUI/Ray 并确认 worker 退出。
- [ ] `git pull --ff-only` 到 `ed3cdb0`，确认工作树干净。
- [ ] 添加实际导入/签名测试并确认旧兼容常量使其失败。
- [ ] 更新兼容常量并运行 Director 测试组。

### Task 5: 版本、双语文档、提交与部署

**Files:** `pyproject.toml`、`__init__.py`、两个 README、双语 release note、O13 测试报告。

- [ ] 更新至 Raylight `1.15.0`，说明分块用途、预览开销和回退路径。
- [ ] 运行 Director/O11/O12/部署/节点契约测试和 Ruff。
- [ ] 审查 `git diff --check`、绝对路径、敏感信息和无关文件后提交源码。
- [ ] 运行 `scripts/deploy-to-comfyui.ps1`，确认部署标记等于源码提交。

### Task 6: 双 V100 四步实机验收

**Files:** `docs/testing/minimax-h3/O13_MMH3DIRECTOR_ED3CDB0_2026-09.zh-CN.md`

- [ ] 用正式脚本启动 ComfyUI，确认节点导入正常。
- [ ] 运行一次4步 Director 工作流，开启逐步预览和2块 Refine。
- [ ] 验证双卡、有限输出、动态预览、无明显分块接缝和仅一对 worker。
- [ ] 保存精简证据并停止进程；失败则回到对应 TDD 任务。

