# Windows 双 V100 Raylight CUDA P2P 验收报告

日期：2026-08-13<br>
环境：原生 Windows、2× Tesla V100-SXM2-16GB、TCC、NVLink、PyTorch 2.7.0+cu126<br>
工作流：`LTX2_3_i2v_Raylight_Benchmark.json`<br>
输入：`LTX2_3_i2v_Raylight.jpg`

## 结论

Raylight 的两卡 Ulysses `all_to_all_single` 大张量数据面已从 Windows Gloo 切换为 CUDA IPC/P2P，实测使用 NVLink。Gloo 仍负责 process group、rendezvous 和不在专用后端支持范围内的控制/collective。

正式复用会话的三次端到端耗时为 284.06、285.33、281.78 秒，中位数 284.06 秒。最初以单卡冷生成 454.70 秒比较时得到 37.53% 提升；后续严格单卡冷/热复测显示单卡真正热启动为 316.60 秒。按公平的热对热口径，双卡稳态端到端快 10.28%，**尚未达到至少快 20% 的最终验收线**。

首次冷启动为 490.85 秒，仍慢于单卡。原因是首次必须建立 Ray、创建两个 worker 并加载两份模型。新增会话缓存后，相同初始化配置会复用原 worker；配置变化或 worker 失效才重建。

## 性能证据

| 项目 | 结果 |
|---|---:|
| 单卡旧冷生成基线 | 454.70 秒 |
| 单卡严格冷启动复测 | 519.94 秒 |
| 单卡严格热启动复测 | 316.60 秒 |
| P2P 冷启动 | 490.85 秒 |
| P2P 复用 #1 | 284.06 秒 |
| P2P 复用 #2 | 285.33 秒 |
| P2P 复用 #3 | 281.78 秒 |
| P2P 复用中位数 | 284.06 秒 |
| 相对单卡严格热启动提升 | 10.28% |
| 复用样本极差 | 3.55 秒 |
| 110 MiB Ray actor 有效带宽 | 59.28 GiB/s/方向 |
| 工作流中的 P2P 健康检查 | 约 59.0–59.4 GiB/s/方向 |

复用验收期间 worker PID 始终为同一对进程。日志只出现一次 P2P 建连和健康检查，之后明确打印 `Reusing 2 live Ray workers for unchanged configuration`。

## 正确性和稳定性

- 五次早期 P2P 完整工作流和四次复用版完整工作流均成功；没有 P2P collective 错误。
- 新生成的四个验收视频均可完整解码，规格为 1280×704、25 fps、4.84 秒，包含 AAC 48 kHz 双声道音频。
- 全元素 P2P 探针覆盖两张卡及 516,096、8,388,608、28,835,840、115,343,360 字节真实 LTX 形状；所有位置使用不同值，两个 rank 均为 0 个不匹配、最大误差 0。
- 10,000 次 516 KiB 循环已通过。
- FP16、BF16、FP32、INT32、UINT8 探针均通过。
- 操作序号不一致和缺失对端均在 2.001 秒内失败；失败通信组被毒化，不会继续产生静默错误。
- 自动化单元测试 18/18 通过。

相同固定种子的完整 LTX 视频不是逐帧确定的：P2P 两次固定种子复跑的 SSIM 约 0.57，且服务端历史确认所有关键种子相同。因此视频逐帧哈希不作为通信正确性判据；通信层采用全元素精确比较作为硬门槛。

## 运行配置

- GPU=2
- Ulysses=2、Ring=1、CFG=1、DP=1
- FSDP 关闭
- `sync_ulysses=true`
- `TORCH_EFFICIENT`
- `skip_comm_test=true`
- `clear_vram_after_sampling=true`
- P2P buffer：64 MiB/卡
- P2P 启动最低健康带宽：50 GiB/s/方向
- ComfyUI 必须带 `--disable-cuda-malloc`

启动入口：`<environment-root>/start_raylight_p2p.ps1`

## 支持边界

当前专用后端只承诺原生 Windows TCC、单机双卡、equal-split 同步 CUDA `all_to_all_single`、Ulysses=2、FSDP 关闭。其他 collective、CPU tensor、非等分 split 或不支持的拓扑回退 Gloo。该实现不是 NCCL 的通用替代品，也不把两卡显存自动合成一个 32GB 地址空间；它加速 Raylight 当前 LTX 序列并行的数据交换。

正式工作流与基准工作流均已固化 `skip_comm_test=true`。启动时 P2P 自身会做 110 MiB、100 次、双 actor、全输出检查与最低带宽门槛，因此跳过原 Gloo COMM tester 不会跳过 P2P 健康检查。
