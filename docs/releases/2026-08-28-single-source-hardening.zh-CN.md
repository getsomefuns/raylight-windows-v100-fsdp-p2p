# 1.10.2：单一源码、部署与测试布局加固

## 目标

本补丁把开发仓库、ComfyUI 运行副本、本机测试证据和项目文档的职责彻底分开，避免再次出现多个 Raylight 仓库、散落日志或只在特定目录结构下可运行的测试工具。

## 已实装

- 唯一可编辑 Git 工作树为 `raylight-windows-v100-fsdp-p2p`；`custom_nodes/raylight` 由部署脚本生成且不含 `.git`。
- 部署脚本通过 `git archive` 从不可变的 `HEAD` 对象树生成纯运行副本，避免“检查后、复制前”工作树变化；根目录写入 `.raylight-deployed-commit`，并在部署与 `-VerifyOnly` 时同时校验文件哈希和来源提交。
- 部署替换独立跟踪“旧目录已备份”和“新目录已安装”两种状态；首次部署或覆盖部署的最终校验失败时都会删除无效新目录，并在存在旧版本时自动恢复。新目录通过最终校验后即成为提交点，此后即使旧备份因 Windows 文件锁无法删除，也只保留残留备份并报警，不会撤销已经验证的新部署。
- 测试路径发现同时支持与 ComfyUI 并列的正式工作树，以及位于 `ComfyUI/custom_nodes/raylight` 的部署布局。
- 测试模块改用包内相对导入，避免与 ComfyUI 自带的 `tests` 包冲突。
- FSDP 探针不再写死 `REPO_ROOT.parent/ComfyUI`。
- MiniMax H3 基准工具接受带来源标记的纯部署副本，也接受干净 Git 克隆；纯副本的来源标记不再被当成“文件一定未改”的证明，实际运行文件仍会逐项校验。Git 克隆始终采用自身真实 `HEAD`，忽略可能过期的私有部署标记。
- 纯部署副本的逐文件身份使用 Git blob 比较，而不是把部署字节与 Windows 工作树原始字节直接做 SHA-256；这样仍能发现真实修改，同时不会把 Git 换行过滤或混合 CRLF/LF 误报为部署漂移。
- 旧 `scripts/minimax-h3/deploy-to-comfyui.ps1` 保留为兼容入口，但只转发给统一部署器，不再要求或修改运行目录中的 Git 仓库。
- Ray 本地 runtime working directory 移到系统临时目录，不再在节点源码树创建 `_ray_runtime_env`。
- 历史计划迁入 `docs/superpowers/plans/legacy`；原始本机证据迁入 Git 忽略的 `.local/artifacts`。

## 验证口径

新增回归覆盖正式工作树、部署目录、纯副本来源标记、干净 Git 克隆、启动脚本路径和外置 Ray 工作目录。最终发布还要求完整测试、部署后直接 pytest、P2P/FSDP 探针、节点导入和实际小工作流验证。
