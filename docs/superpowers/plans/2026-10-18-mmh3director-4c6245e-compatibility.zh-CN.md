# Raylight 1.17.1 / O15.1 MiniMaxH3Director 4c6245e 兼容计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将本机官方 MiniMaxH3Director 从 `8151b29` 快进到 `4c6245e`，在不修改 ComfyUI 或 Director 源码的前提下，确认 Raylight 双 V100 FSDP/P2P 采样、Refine、SelfLift、FaceRefine、分段 LoRA、实时预览和缓存路径不退化。

**Architecture:** 上游唯一新增内容是导演包导入/导出进度 UI；Python 采样契约没有变化。Raylight 不复制该前端实现，只把已验证兼容坐标更新到 `4c6245e`，通过真实上游签名、节点启动、自动化与双卡工作流回归完成兼容认证。

**Tech Stack:** Windows、Python 3.12.10、PyTorch 2.12.0+cu126、ComfyUI、Ray 2.57.0、CUDA P2P、FSDP2、pytest、MiniMaxH3Director `4c6245e`。

## 边界与结论

- PR #182 的 GPU 批量优化只加速连续片段解码后的 opening-grade 匹配，不执行 VAE 神经网络解码，不能解释或消除本机约 66 秒的 107 帧 1920×1088 VideoVAE 解码。
- `8151b29..4c6245e` 只有 1 个提交、2 个前端 JavaScript 文件：新增导演包上传、解压、打包、下载、保存和时间轴写入进度；没有 Python、节点 schema、采样、VAE、Refine 或缓存格式变化。
- 不为纯前端变化增加新的采样桥或 worker 协议；这会制造没有消费者的兼容代码。
- 唯一源码工作树仍为 `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`；部署目录只由部署脚本生成。

## 实施任务

### Task 1：锁定最新版兼容坐标

**Files:**
- Modify: `tests/test_mmh3director_bridge.py`
- Modify: `src/raylight/integrations/mmh3director/bridge.py`

- [ ] 先把契约测试期望提交改为 `4c6245e518dd8a494f3afc4f3cccd53f22d82d10`，运行单测并确认 RED。
- [ ] 最小更新 `SUPPORTED_DIRECTOR_COMMIT` 和错误信息测试，运行 bridge 测试并确认 GREEN。
- [ ] 不改采样参数、worker 请求、FSDP、P2P 或精度路径。

### Task 2：升级官方 Director 并验证上游边界

- [ ] 确认 Director 工作树干净，记录旧提交 `8151b29`，执行 `git pull --ff-only` 到 `4c6245e`。
- [ ] 验证差异仍只有 `web/js/minimax_i18n.js` 与 `web/js/minimax_pack.js`。
- [ ] 运行 Director 官方测试、Python 编译检查和真实 `sample_single_stage` 签名检查。
- [ ] 启动 ComfyUI，确认 Director、Refine、SelfLift、FaceRefine、Semantic Bridge 和 Ray MODEL Adapter 节点均可加载。

### Task 3：回归 Raylight 并实机验收

- [ ] 运行 MMH3Director bridge、worker、分段 LoRA、生命周期、SOL/TE 和部署测试。
- [ ] 运行 Raylight 全量测试；任何失败必须分类为回归、环境或既有问题。
- [ ] 从已提交源码部署到 `ComfyUI/custom_nodes/raylight`，确认部署副本与工作树一致。
- [ ] 运行一次短 Director 双卡工作流；要求两个 rank 完成同一个远程 operation、无本地重复采样、输出媒体可解码、无 NaN/Inf。

### Task 4：版本、文档与发布

- [ ] 版本升至 `1.17.1`，更新中英文 README、发布说明和测试记录。
- [ ] 明确本版新增的是 Director `4c6245e` 兼容认证和导演包进度 UI，不是 VAE 解码提速。
- [ ] 审查 Git diff、运行最终验证、提交并推送 GitHub `fsdp/main`。

## 验收标准

1. Director HEAD 与 `origin/main` 均为干净的 `4c6245e`。
2. Raylight 契约测试读取真实上游 `sample_single_stage` 并通过；旧版桥和普通 MODEL 透传不回归。
3. ComfyUI 启动无 Director/Raylight 导入错误，最新版导演包进度前端资源存在。
4. 双 V100 实机工作流完成，只有一次远程采样操作，两张卡参与，输出逐帧可解码且音频有效。
5. 中英文 README、发布说明、测试记录与代码版本一致；Git 工作树干净且 GitHub 已同步。

