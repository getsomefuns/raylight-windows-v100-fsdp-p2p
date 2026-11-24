# Director 最新版、Pika 与 RelayVSR 适配需求和执行计划

## 目标与约束

用户要求保留本地修复，更新官方 Director，使用 Pika 加速与 VSR，并保留旧双 V100 工作流。唯一 Raylight 源码目录为本仓库；修改只能先在工作树进行，测试后部署和同步 GitHub。不改 NVIDIA 驱动，不覆盖用户工作流或分段缓存，不擅自修改 ComfyUI 核心。开发与测试证据不能冒充完整生成验收。

## 核查基线（2026-11-24）

- Raylight：`af4500d` / 1.18.5。
- 本地 Director：`6e14f23`，官方基线 `4c6245e` + PR231 缓存检查补丁；另有 5 个未提交修改文件，涉及导出尾帧引导、左右选脸与 tracking_reach。不得用 reset/reinstall 丢弃。
- 获取的官方 main：`aff2579`，新增 RelayVSR、批量删除与大导演包导出优化等 5 个提交。相对此前基线没有 Pika 专用节点、采样分支或调度代码；唯一 pika 字符串来自 Pikachu 示例，不能当成适配证据。
- 当前 ComfyUI：`62b3c94b`；缺少 `RelayVSRConditioning`、`VSRDecode` 和 RelayVSR 模型实现。当前模型库也未找到 RelayVSR UNET/Flash VAE 或 Pika LoRA。

## 方案

采用官方 Director 更新 + 可重放本地补丁，而不是整目录覆盖或维护第二个上游 fork。补丁以 Director 本地独立提交保存，并在 Raylight 仓库提供补丁文件与说明，完整源码和 Git bundle 保留在忽略的 `.local/backups/director/` 中。

H3 首采、SelfLift、Refine、FaceRefine 继续通过 Ray MODEL Adapter 送往双卡；RelayVSR 是独立的 Wan 系超分模型，应使用普通 MODEL 和对应 Flash VAE，沿官方 Director 的后处理路径执行，不得冒充 MiniMax H3 MODEL 或声称已获得双卡 FSDP 支持。其帧数与音频必须保持对应。

Pika 首先核实官方模型/LoRA、任务类型和 sigma 配置。若仅为兼容 LoRA + sigma 表，复用现有 RayLoraLoader 和自定义 SIGMAS 传递，不新增空壳加速节点；若需专用 sampler/patch，再针对实际调用路径立项。不能将 6 步相对 20 步的结果套成相对现有 4 步工作流的加速。

## 执行与门禁

- [x] 核对两个 Git 基线、上游差异及本地修改。
- [x] 完整备份 Director，保存未提交差异；创建本地提交，再 rebase 到 `aff2579`，不改 ComfyUI 核心。
- [x] 验证 PR231、FL2V/R2V 缓存身份、正常素材加载、前端取消/去重、官方音频测试。
- [x] 执行 Raylight 全量测试：830 passed / 35 subtests；新 VSR 普通 MODEL 路径代码已核查，尚无 VSR 实机数据。
- [ ] 保存可再应用的补丁、中英文操作说明和检查结果，确认无隐私/模型/缓存入库后同步 GitHub。
- [ ] 实机启动新版 Director，验证旧图首次/缓存复用运行。
- [ ] 获得允许更新 ComfyUI 的明确授权后，选定含 RelayVSR 的核心版本，先隔离验证 V100、cu126、安全精度与旧工作流，再替换环境。或者用户选择保持核心冻结，暂停 VSR。不能在本次范围内静默更新核心。
- [ ] 下载核实来源的 RelayVSR x2 UNET/Flash VAE 与 Pika 权重到共享模型库；体积、版本、校验和另记录。
- [ ] 同素材/同种子/同画布的短流程：旧 Turbo 基线、Pika、基线+VSR、Pika+VSR；分开记录冷/热加载、TE、采样、H3解码、VSR采样/解码、保存及端到端、双卡显存/主存/提交峰值。
- [ ] 检查全帧可解码、音画时长、x2尺寸、首中尾视觉、二采和缓存；只有完整实测才标记可用或发布速度结论。

## 当前边界

独立代码审查发现必须补充的门禁：缓存状态接口未重建 FaceRefine/RelayVSR 指纹，不能准确声称“成片匹配”，应补可序列化 witness 或显示“成片未验证”，并测试未接→已接/有效缓存/参数变化；metadata-only 素材路径还应统一验证 input 目录边界。两项本轮尚未修复，不能把基础回归通过当作这些行为已验收。

缺少核心节点和模型是可确认的前置依赖，不是 Raylight 代理取参错误。为了遵守不改 ComfyUI 核心的约束，不能将更新 Director 等同于完成 VSR。Pika 的具体作者更新入口/权重也需核实。无需另开仓库或修改已部署 Raylight 来掩盖依赖缺失。
