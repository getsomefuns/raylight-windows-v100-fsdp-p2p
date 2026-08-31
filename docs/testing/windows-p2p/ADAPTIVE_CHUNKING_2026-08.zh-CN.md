# O9 Windows CUDA P2P 自适应分块测试记录

日期：2026-08-31

状态：Tasks 1–5 已完成；Gate B 与 Gate D 首轮通过，Gate C 的工作流级系统内存/commit/pagefile 峰值仍待部署后采集。

## 验证目标

证明旧实现把 `all_to_all_single` 的逻辑远端 payload 错误限制为不大于物理 staging buffer，并保存同机、同驱动下 128/256/512 MiB 匹配 buffer 的带宽和 CUDA allocation，供 O9 修改后的正确性、显存与性能验收使用。

## 固定环境

| 项目 | 值 |
|---|---|
| GPU | 2× Tesla V100-SXM2-16GB |
| 模式 | TCC |
| 驱动 | 577.00 |
| Python | 3.10.11，`E:\ComfyUI-py310\Python310\python.exe` |
| PyTorch | 2.7.0+cu126 |
| 源码分支 | `windows-v100-fsdp-p2p` |
| O9 开发前源码 | `309cc9a`（计划提交；transport 与已部署 `70edb4f` 相同） |
| 已部署 Raylight | `70edb4f17175bf1739d98135406d9b0899adcfc8` |
| 空闲显存 | GPU0/GPU1 均为 10 MiB |
| 空闲利用率 | GPU0/GPU1 均为 0% |

基线单元测试命令：

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pytest `
  tests/test_windows_p2p.py `
  tests/test_collective_profile.py `
  tests/test_windows_release_profile.py -q
```

结果：35 passed，10 subtests passed。唯一警告是受当前运行权限影响无法写入 `.pytest_cache`，不影响测试行为；后续正式命令统一追加 `-p no:cacheprovider`。

## 旧失败的精确复现

命令：

```powershell
E:\ComfyUI-py310\Python310\python.exe tests/windows_p2p_probe.py `
  --iterations 1 `
  --capacity-bytes 134217728 `
  --size-bytes 479653888
```

条件：

- 每 rank 物理 staging buffer：134,217,728 字节（128 MiB）。
- all-to-all 完整输入：479,653,888 字节。
- 每 rank 需要从对端接收：239,826,944 字节。

结果：两个 rank 都在任何传输开始前失败：

```text
ValueError: remote payload 239826944 exceeds P2P buffer capacity 134217728
```

判断：这是逻辑容量检查失败，不是 CUDA OOM、NVLink/P2P 失效、rank timeout 或 Gloo 回退。O9 完成后必须使用完全相同命令通过，并报告两个子块。

## 匹配大 buffer 的性能和显存基线

每档 warmup 3 次，正式循环 20 次；每 rank 的远端吞吐按 all-to-all 输入的一半计算。

| staging/rank | 完整输入 | 远端 payload/rank | rank0 GiB/s | rank1 GiB/s | endpoint allocated/rank | payload 后 allocated/rank |
|---:|---:|---:|---:|---:|---:|---:|
| 128 MiB | 256 MiB | 128 MiB | 61.540 | 61.570 | 128 MiB | 640 MiB |
| 256 MiB | 512 MiB | 256 MiB | 62.304 | 62.244 | 256 MiB | 1280 MiB |
| 512 MiB | 1024 MiB | 512 MiB | 62.713 | 62.695 | 512 MiB | 2560 MiB |

结论：

- 匹配大 buffer 的单次基线约为 61.5–62.7 GiB/s/rank。
- endpoint 的 `torch.cuda.memory_allocated()` 精确等于选定 staging 容量，证明 512 MiB 配置会比 128 MiB 配置永久多占 384 MiB/rank。
- O9 Gate D 使用这些同机结果作参考；正式结论仍需每规格至少 5 个交错 session 的中位数和 P95，不能使用本表单次结果直接发布。

## 冻结的真实工作流身份

| 工作流 | SHA256 | 用途 |
|---|---|---|
| `example_workflows/Minimax_H3_I2V_Raylight.json` | `5AFBD4AED78AAB4A6BC5DC0716947118CB2D1B07D148BC4498441A7A68727086` | 历史 239,826,944 字节失败规格来源 |
| `example_workflows/Minimax_H3_REF2VA_Raylight.json` | `1C4841BCEEF4172F5C833EF2878CF91BA3B49BEEFA07E7DCCEFB7D581867D2CA` | MiniMax H3 REF2VA 回归 |
| `E:\ComfyUI-py310\ComfyUI\user\default\workflows\MMH3Director.json` | `97DEFBAFF3F4046D53573D2BB36DBFE01E9B4ECB451ED9481A1AFE7690A08CD4` | 当前 Director/多段回归 |
| `E:\ComfyUI-py310\ComfyUI\user\default\workflows\MMH3Dirctor_Refine.json` | `62F351005AF4A9B244FE105C806A7031BB4A8396C8B8825D67056AF5B2C73E34` | Refine 回归 |
| `E:\ComfyUI-py310\ComfyUI\user\default\LTX2_3_i2v_Raylight.json` | `3687ED37749BB1FF6A11AF4478BF0C6FE77BDBD4A3FA375FF6593A467D9F2A2B` | LTX 5 秒回归 |

正式工作流验收前需复制这些 JSON 到 run artifact，并记录实际 API prompt 中的模型、尺寸、帧数、steps、seed、LoRA、精度和输入素材哈希。若用户在 O9 开发期间改动工作流，必须同时保留本表冻结版本和新版本，不得覆盖基线身份。

## 自适应分块实现后的 synthetic correctness

核心实现提交：`75be33c`。

历史失败命令保持完全相同。修复后结果：

| 项目 | rank 0 | rank 1 |
|---|---:|---:|
| 128 MiB buffer 搬运 239,826,944 字节远端 payload，20 次 | 61.161 GiB/s | 61.184 GiB/s |
| endpoint allocation | 128 MiB | 128 MiB |
| logical calls（含 3 次 warmup） | 23 | 23 |
| data chunks | 46 | 46 |
| next operation id | 70 | 70 |

每次 logical call 精确产生一次完整尺寸握手和两个数据子块，因此 23 次共消耗 69 个内部 operation id。该结果排除了“静默扩容”或“误走单块路径”。

### 64 MiB 固定 buffer 的多规格/多 dtype

四种存储类型分别运行：FP32、FP16、BF16 原始字节、uint8。每个规格先 warmup 3 次，再正式运行 20 次，并验证首字节、每个 64 MiB 边界前后和最终字节。

| 远端 payload/rank | chunks/call | 各 dtype 最低观察值 | 各 dtype 最高观察值 | endpoint allocation/rank |
|---:|---:|---:|---:|---:|
| 64 MiB | 1 | 59.535 GiB/s | 60.195 GiB/s | 64 MiB |
| 128 MiB | 2 | 60.227 GiB/s | 60.325 GiB/s | 64 MiB |
| 256 MiB | 4 | 60.338 GiB/s | 60.424 GiB/s | 64 MiB |
| 512 MiB | 8 | 60.336 GiB/s | 60.447 GiB/s | 64 MiB |
| 768 MiB | 12 | 60.277 GiB/s | 60.504 GiB/s | 64 MiB |

主压力规格使用 64 MiB buffer、512 MiB 远端 payload、uint8 连续正式运行 100 次：两 rank 均约 59.91 GiB/s；103 个 logical calls（含 warmup）记录为 824 个 data chunks，next operation id 为 928，全部边界正确。

BF16 只证明原始字节搬运正确，不代表 V100 支持 BF16 计算。

### 故障和混合 collective

- 第二子块故障注入在 operation 3 把 rank1 发布尺寸从 2,097,152 改为 2,097,151 字节；rank0 在约 3 ms 内失败并 poison group，后续调用在进入 transport 前被拒绝。
- 现有 operation mismatch 和 missing peer 均在 2 秒测试 timeout 内失败；没有放宽正式 10 秒 timeout。
- 同一 Ray actor session 依次完成 all-to-all、同步/异步 all-gather、64/128/256/384 MiB all-gather、两块大 shard 和所有 dtype 尾块；operation id 与可复用 CUDA event 没有互相污染。
- session 结束后两卡均恢复 10 MiB、0% 利用率，无残留 compute process。

## Gate D：匹配大 buffer 与 128 MiB 自动分块

测试方法：每个规格 5 个独立 session；奇数 trial 先跑匹配 buffer，偶数 trial 先跑 128 MiB 分块，降低温度/顺序偏差。每次 3 次 warmup、50 次正式循环，使用两 rank 中较低的吞吐。

| 远端 payload | 模式 | 5 次 GiB/s | 中位数 | 相对匹配模式 |
|---:|---|---|---:|---:|
| 256 MiB | 匹配 256 MiB | 62.415 / 62.405 / 62.344 / 62.340 / 62.378 | 62.378 | 100% |
| 256 MiB | 固定 128 MiB，2 chunks | 61.686 / 61.714 / 61.719 / 61.692 / 61.696 | 61.696 | 98.91% |
| 512 MiB | 匹配 512 MiB | 62.717 / 62.734 / 62.718 / 62.725 / 62.709 | 62.718 | 100% |
| 512 MiB | 固定 128 MiB，4 chunks | 61.464 / 61.777 / 61.785 / 61.774 / 61.411 | 61.774 | 98.50% |

512 MiB 分块的 5 次最大 elapsed 为 0.404632–0.407094 秒，中位数 0.404701 秒；最慢样本仅比中位数高约 0.59%，没有超过 25% 的长尾门槛。两种规格均显著高于计划要求的 85% 吞吐门槛。

相对旧 512 MiB staging，默认 128 MiB endpoint 的 `torch.cuda.memory_allocated()` 每 rank 精确少 384 MiB，双卡共少 768 MiB。逻辑 payload 从 128 MiB 增到 768 MiB 时，endpoint allocation 保持固定；payload tensor 本身随测试规模增长，不计入 staging。

## 启动与健康检查变更

- 仓库启动器默认 staging budget 从 256 MiB 改为 128 MiB。
- 128/256/512 MiB 与 legacy bytes 参数仍保留为高级性能/显存覆盖。
- `-ValidateOnly` 显示 `Larger collectives are auto-chunked: enabled`。
- 默认 worker 健康输入改为 264 MiB，远端 132 MiB，可在模型加载前实际验证 128 MiB 边界。
- 新增环境根薄包装安装器；生成的 `start_raylight_p2p.ps1` 不保存 128/256/512 任一默认值，只把全部参数转发到部署仓库启动器。

## 当前结论

- 旧失败已在空闲双 V100 TCC 环境中稳定复现。
- probe 已能独立指定物理 buffer 和逻辑 payload，并记录 endpoint/payload CUDA allocation。
- transport 已支持大于 capacity 的 equal-split CUDA all-to-all，synthetic Gate B/D 通过。
- Gate C 已证明 staging CUDA allocation 恒定并完成退出回收；系统内存、commit 与 pagefile 峰值将在真实工作流监控中完成最终门禁。
- 下一步提交 Tasks 4–6 代码与记录，然后部署并执行真实 MiniMax H3/LTX Gate E。
