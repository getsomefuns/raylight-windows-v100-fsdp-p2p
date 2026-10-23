# Raylight 1.18.3：长参考视频 Attention OOM 与全生命周期内存记录

## 故障与根因

MiniMax H3 Director 的 15 秒 REF2VA + SelfLift 高分辨率采样在双 V100 上进入 Ulysses Attention 后 OOM。旧路径把一次 QKV 投影切成共享同一底层存储的三个视图；单个 Q/K/V 完成通信后，这块完整存储仍不能释放。随后 yunchang `all_to_all` 又创建一个完整 `empty_like` 输出。在每卡已有约 14.5 GiB 物理显存占用时，这个额外约 1.56 GiB 的输出成为直接失败点。

## 修复

1. MiniMax H3 QKV 按 4,096 token 分块写入三个独立存储；每次 Attention 仍只做一轮 FSDP unshard/reshard。
2. Windows 双 rank CUDA P2P 路径复用通信布局张量作为 `all_to_all_single` 输入和输出。既有 128 MiB staging 会先保存待发送子块，再覆盖目标范围，因此不再分配第二个完整输出。
3. xFormers 密集 Attention 将输出写回已消费的 Query 存储，少保留一个完整 context。
4. 非 Windows P2P 后端继续使用 yunchang 原路径，采样数学、模型精度、LoRA 和 Director 参数均未改变。

## 内存记录

启动脚本现在统一设置 `RAYLIGHT_MEMORY_LOG_DIR=<ComfyUI>/user/raylight_memory`。Director 完整生命周期和每次内部采样会以 1 秒周期记录：

- 系统物理内存与 Windows commit/commit limit；
- ComfyUI 主进程和两个 Ray worker 的 RSS、private、pagefile、VMS；
- PyTorch CUDA allocated/reserved；
- `cudaMemGetInfo` 的整卡 used/free。

成功和异常都会写 JSONL 汇总到按日期划分的目录；设置 `RAYLIGHT_MEMORY_RECORD=0` 可关闭。CUDA 峰值在阶段开始前重置，不会混入上一阶段的历史峰值。

## 验证结果

- Windows P2P 原位/非原位双卡探针覆盖到 115,343,360 bytes，所有尺寸 `mismatch=0`、`max_error=0`。
- 全量自动化：`808 passed, 35 subtests passed`。
- 真机验收：362 帧、24 fps、768×1344、双参考图、约 354 帧参考视频、REF2VA、SelfLift 低/高分辨率各 2 步。
- Attention 序列低分辨率约 152,332，高分辨率约 233,224；两名 rank 均完整返回，最终 MP4 为 362 帧，任务状态 `success`。
- 观察到的整卡峰值约为 GPU0 16,083 MiB、GPU1 16,213 MiB；旧实现会在相同位置再申请约 1.56 GiB，本次没有 OOM。
- Director 报告：条件编码 238.2 秒、采样 2,299.2 秒、解码 77.9 秒、缓存 16.7 秒，端到端 46 分钟。

本版本解决的是长参考输入的容量稳定性，不是速度优化。这个 768×1344 极限用例虽然成功，但耗时不适合直接作为生产配置。
