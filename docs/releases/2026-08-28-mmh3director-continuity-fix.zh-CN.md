# 1.10.1：MiniMaxH3Director 连续分段条件修复

日期：2026-08-28

## 修复内容

MiniMaxH3Director 在主 ComfyUI 进程中为后一段准备运动续接关键帧；Raylight 则在独立 Ray worker 中执行扩散采样。当连续分段同时包含当前参考图时，ComfyUI 的标准 MiniMax 条件整理会让参考图 latent 覆盖关键帧 latent。布局仍保留全部条件段，最终产生类似 `510` 行条件写入 `4080` 行布局的维度错误。

1.10.1 在 worker 端识别 Director 专用的连续性标记，并且：

- 按“运动续接关键帧在前、当前参考图在后”的顺序恢复视频条件；
- 把续接关键帧重新定位到当前目标视频时间轴；
- 把连续音频参考移动到正确的结束时间；
- 对关键帧数量、参考类型、布局顺序和音频行数执行显式校验；
- 对没有 Director 标记的 MiniMax、LTX、I2V、FL2VA 和 REF2VA 请求保持原对象和原行为。

## 验证结果

- 新增 4 项定向回归测试，覆盖条件合并、关键帧时间、音频时间和普通请求不变性。
- Director/Raylight 相关测试组合：`50 passed`。
- 完整仓库测试：`423 passed, 30 subtests passed`。
- 双 Tesla V100-SXM2-16GB、TCC、FSDP、CUDA P2P/NVLink 实机运行两段 Director 工作流：两段主采样与两段 Refine 均由两个 rank 完成。
- 实机任务总时间约 12 分 47 秒；输出为 1280×736、24 FPS、365 帧、约 15.2 秒。
- 两个保存结果均通过 FFmpeg 完整解码检查。

## 兼容范围

本补丁的已验收 Director 坐标仍为 `AIMixer/ComfyUI_MiniMaxH3_Director` 提交 `d655912`。官方仓库在 2026-08-27 已有后续缓存、内存、FL2V 和 Refine 改造；这些更新未包含在本次发布中，也不声明已经完成兼容验收。

工作流中引用不存在的本地首帧或尾帧仍属于素材配置错误，不经过 Ray worker，也不是本补丁的处理范围。
