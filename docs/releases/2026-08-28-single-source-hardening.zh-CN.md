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
- 纯部署副本的逐文件身份使用带仓库相对路径的 Git blob 比较，让 `.gitattributes` 与 Git clean filter 正确规范化换行，而不是把部署字节与 Windows 工作树原始字节直接做 SHA-256；这样仍能发现真实修改，同时不会把 CRLF/LF 转换误报为部署漂移。
- 旧 `scripts/minimax-h3/deploy-to-comfyui.ps1` 保留为兼容入口，但只转发给统一部署器，不再要求或修改运行目录中的 Git 仓库。
- Ray 本地 runtime working directory 移到系统临时目录，不再在节点源码树创建 `_ray_runtime_env`。
- 历史计划迁入 `docs/superpowers/plans/legacy`；原始本机证据迁入 Git 忽略的 `.local/artifacts`。

## 验证口径

新增回归覆盖正式工作树、部署目录、纯副本来源标记、干净 Git 克隆、启动脚本路径和外置 Ray 工作目录。最终发布还要求完整测试、部署后直接 pytest、P2P/FSDP 探针、节点导入和实际小工作流验证。

## 最终验收结果

- 唯一工作树完整测试：`444 passed, 1 warning, 30 subtests passed`；部署副本直接测试：`444 passed, 3 warnings, 30 subtests passed`。
- 394 个部署文件与提交 `c0b3f1b51942690638a62c1627761ae260561028` 一致；运行副本无 `.git`，`custom_nodes` 中没有第二份 Raylight 备份目录。
- 双 Tesla V100-SXM2-16GB 处于 TCC 模式；64/128/256/384 MiB CUDA P2P all-gather 全部通过，单向实测约 59.3-61.6 GiB/s。
- 实际 MiniMax H3 REF2VA 验收采用 576x352、24 fps、2 秒请求、Turbo 4 步、FSDP + CPU offload。两个 worker rank 均参与；采样 39.38 秒，整条 prompt 200.25 秒。阶段数据为：预采样 120.19 秒、模型转 GPU 0.93 秒、采样器节点 71.21 秒、VAE 解码 6.47 秒、视频保存 2.38 秒。
- GPU0/GPU1 峰值显存分别为 15949/4603 MiB，峰值利用率 100%/96%；物理内存峰值 95847 MiB，提交内存峰值 134431 MiB，未使用分页文件。
- 输出为 H.264/AAC、576x352、24 fps、56 帧、2.33 秒；视频和音频完整解码通过，56 帧哈希全部唯一，首/中/末帧为有效画面，不是黑屏或噪声。
- 基准结束后 ComfyUI 与两个 Ray worker 均被停止，未残留 Python 计算进程或 GPU 计算显存占用。

PyTorch 2.7 的 Windows legacy TCPStore 仍会打印一次 `WorkStudio:29500` IPv6 候选连接警告；既有隔离测试已确认它会继续使用 IPv4 loopback 完成建组，本次两个 rank 的实际采样和成片也再次证明它不是 P2P 数据面的回退或执行失败。
