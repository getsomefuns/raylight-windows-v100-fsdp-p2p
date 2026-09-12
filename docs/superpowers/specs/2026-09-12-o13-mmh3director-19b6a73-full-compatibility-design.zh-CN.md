# O13 MiniMaxH3 Director 19b6a73 完整兼容设计

## 目标

将 MiniMaxH3 Director 从 `b8f721c` 更新到 `19b6a73`，同时保持 Raylight Windows 双 V100 FSDP/P2P、SOL Attention、TE 加速、热 worker 复用和旧工作流兼容。适配范围必须覆盖 Director 一采、Director 自带二采 Refine、连续段重锁定、Refine 空间分块、逐步循环 WebP 预览，以及新版 `shift_cache` 调用契约。

## 边界

- 只修改唯一源码工作树 `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`；部署目录不直接编辑。
- Director 作为独立上游节点仓库快进更新，不在其源码中维护本项目补丁。
- 不修改 ComfyUI 核心、不修改 SOL Attention、TE-Speed 或其他节点。
- 旧 Director `d655912`、`b8f721c` 接口继续被 Raylight 接受；新版 `19b6a73` 作为当前验证目标。
- 空间分块用于降低显存，不承诺比不分块更快；显存足够时默认关闭。

## 架构

### 新版采样契约

Raylight 的 Director bridge 接受新增关键字：`after_shift`、`enable_tiling`、`tile_count`、`tile_overlap`、`shift_cache`。`after_shift` 不直接序列化闭包，而转换为受控能力标识；当前只接受 Director 的 `install_continue_prefix_remask`，未知回调必须明确报错。`shift_cache` 只缓存主进程的 ComfyUI MODEL 克隆；Raylight 的真实权重位于远端 worker，因此 bridge 接收该参数但不把主进程缓存跨进程序列化，worker 继续由 Raylight 自己管理生命周期。

### 连续段重锁定

主进程把 latent 中的 `_director_continue_prefix_steps` 和 `_director_continue_seam_min` 连同完整 latent 样本、noise mask 和 sigma 调度发送到 worker。每个 worker 在实际 MODEL 上执行 `install_continue_prefix_remask`，确保双 rank 应用相同的逐步 mask 和 APPLY_MODEL wrapper。

### Refine 空间分块

分块在每个 worker 的实际 sampler 对象上安装。每个空间块仍由两张 V100 通过既有 FSDP/P2P/USP 路径共同推理；不能在主进程的无权重 MODEL 代理上切块。关闭分块时继续走已有 `comfy.sample.sample` 路径，避免影响最快路径。

### 逐步预览

仅 rank 0 在 sampler callback 中抽取最多 16 个视频 latent 帧并复制到 CPU，通过容量为 1 的非阻塞 Ray 队列发送到主进程。主进程在等待 worker 结果时持续排空队列并调用 Director 原有 `on_step_preview`；队列满时丢弃中间更新，最终步骤由主进程保证回调。rank 1 不生成预览，预览关闭时不创建队列。

## 验收标准

- 新版 Director 能被严格契约校验识别，普通模型仍走原 Director 路径。
- 一采与 Director 自带 Refine 都由双 worker 运行；未知 `after_shift` 失败关闭。
- 连续段重锁定元数据在两张卡上生效，不静默退化为静态 mask。
- Refine 空间分块 2 块可完成一次4步采样，输出有限值且无空间接缝异常。
- 开启实时预览时，4步采样能在结束前收到逐步事件并显示循环 WebP；关闭时无队列和额外传输。
- 原有 O11/O12 工作流和节点接口测试不退化，热运行只保留一对 worker。
- 部署副本与源码提交完全一致，Director 固定在 `19b6a73`。
