# O9 Windows CUDA P2P 自适应分块测试记录

日期：2026-08-31

状态：Tasks 1–7 与 Gate A–E 已完成；Gate F 发布、部署与最终 smoke 正在收尾。

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

## Gate E：真实工作流验收

所有 O9 正式工作流均使用 128 MiB/rank staging budget、TCC、双 V100、CUDA P2P/NVLink、严格 10 秒协调 timeout；未为任何工作流提高容量、降低分辨率/帧数或关闭 Refine。原始日志、API prompt、监控 CSV 与 benchmark JSON 保存在本机 `.local/artifacts/o9` 或未提交的 `artifacts/o9`，不把视频和大日志提交 Git。

### MiniMax H3 显式 Ray 工作流

| 工作流 | 温度 | 端到端 | 远程采样 | GPU0/1 峰值 MiB | 物理内存峰值 MiB | commit 峰值 MiB | pagefile 峰值 MiB | 结果 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| I2V safe-FP16 Turbo8，1120×768/124 帧 | 冷 | 520.55 s | 359.14 s | 16162 / 14734 | 88947 | 128871 | 0 | PASS |
| 同上 | 热 | 422.54 s | 371.83 s | 14942 / 14734 | 76250 | 92066 | 988 | PASS |
| I2V FP32，历史精确失败 payload | 冷 | 1518.00 s | 1365.77 s | 16178 / 16086 | 89634 | 129243 | 919 | PASS；远端 239,826,944 字节自动 2 chunks |
| REF2VA safe-FP16 Turbo4，1120×768/124 帧 | 冷 | 362.21 s | 201.86 s | 16237 / 15698 | 89947 | 129523 | 848 | PASS |
| 同上 | 热 | 256.92 s | 204.34 s | 15916 / 15708 | 75972 | 92294 | 778 | PASS |

历史 FP32 规格保持旧失败命令中的 479,653,888 字节完整 all-to-all 输入，每 rank 从对端接收 239,826,944 字节。O9 没有扩容 endpoint；每个 logical call 精确记录 2 个 data chunks，证明不是静默改回 256/512 MiB。

### MMH3Director 五段 + Refine

固定工作流为 R2V、5×73=365 帧、1280×736、24 FPS、Turbo4，每段 4 步主采样 + 2 步同模型 Refine。benchmark 为避免 ComfyUI 缓存，冷/热使用 seed 42/43，并在同一 ComfyUI 进程复用相同 worker PID。

| 温度 | 端到端 | 10 次采样合计 | prepare/model-to-GPU | GPU0/1 峰值 MiB | 物理内存峰值 MiB | commit 峰值 MiB | pagefile 峰值 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 冷 | 1445.67 s | 836.13 s | 4.94 / 4.92 s | 15425 / 12840 | 95157 | 136120 | 862 |
| 热 | 1362.48 s | 813.95 s | 2.21 / 2.19 s | 15425 / 12840 | 103353 | 121382 | 854 |

- 冷/热均完成 10 个真实 operation，不是 ComfyUI cache 命中；相同 worker PID `23188/26416` 被复用。
- 每个主采样记录 800 个 all-to-all logical calls，每个 Refine 记录 400 个。该几何最大 all-to-all 完整 payload 为 180,776,960 字节，远端半块低于 128 MiB；最大 all-gather 完整 payload 为 542,330,880 字节，`chunks > calls`，实际跨越 128 MiB 边界。
- 热任务结束后 worker、raylet/子进程均退出，两卡回到 10 MiB；没有每段或每轮增加一份 worker/model/staging。
- O8 的 256 MiB 历史五段同规格结果为 1475.655 秒；O9 冷/热分别约快 2.0%/7.7%。由于历史 seed 与采样时系统状态不同，这只是无明显回归证据，不作为 transport 单独加速比；Gate D 的交错微基准才是容量性能定级依据。

### LTX 2.3 五秒 FSDP

固定 payload 为 1280×704、121 帧、25 FPS、两段 8+3 steps、原 BF16 LoRA 与默认 BF16/FP32 语义。

| 温度 | 端到端 | Sampler 1 | Sampler 2 | GPU0/1 峰值 MiB | 物理内存峰值 MiB | commit 峰值 MiB | pagefile 起/峰/末 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 冷 | 517.13 s | 14.9 s/it | 45.9 s/it | 16224 / 16152 | 69678 | 113116 | 851 / 851 / 765 |
| 热 | 386.36 s | 14.9 s/it | 47.8 s/it | 16234 / 16156 | 54942 | 70055 | 765 / 765 / 765 |

LTX FSDP profile 的最大完整 all-gather payload 为 302,063,616 字节；每段均出现 `chunks > calls`。两张卡采样利用率均达到 100%。最后一个 rank 的重复 profile 文本被 Ray 日志去重，正式传输与完成状态未受影响；1.11.0 已把基准环境固定为 `RAY_DEDUP_LOGS=0`，后续诊断不再丢失重复 rank 文本。

### 媒体与回收

| 输出组 | 规格 | 解码帧 / 唯一帧 | 黑屏区间 | 音频 NaN/Inf | 结论 |
|---|---|---:|---:|---:|---|
| Director 冷/热双 SaveVideo，共 4 个文件 | 1280×736、365 帧、24 FPS、15.21 s | 每个 365 / 365 | 0 | 0 / 0 | PASS |
| LTX 冷/热，共 2 个文件 | 1280×704、121 帧、25 FPS、4.84 s | 每个 121 / 121 | 0 | 0 / 0 | PASS |

所有媒体均可完整 ffprobe/ffmpeg 解码；首/中/尾亮度不是全黑或常量，未出现首帧后乱码。每个 benchmark 退出后两卡均回到 10 MiB，目标 worker PID 不存在。物理内存、commit 与 pagefile 峰值由模型/缓存主导，没有出现与 collective payload 等量、随 chunk 数增长的 host staging。

## 与旧 256/512 MiB 配置的关系

- 相同 transport 微基准已完成严格匹配比较：128 MiB 自动分块保留 256/512 MiB 匹配 buffer 中位带宽的 98.91%/98.50%。
- endpoint allocation 相对 256 MiB 每 rank 少 128 MiB，相对 512 MiB 每 rank少 384 MiB；这是 `torch.cuda.memory_allocated()` 的固定差值。
- 真实工作流峰值常由 activation、临时 full layer、VAE/Text Encoder 和 CUDA workspace 决定，因此 `nvidia-smi` 峰值不会始终完整显示 128/384 MiB 差额。不能把“峰值相近”误解为 staging 没有节省。
- 256/512 MiB 覆盖继续保留用于高级性能实验；它们不再用于修复合法 payload 的容量报错。

## 当前结论

- 旧失败已在空闲双 V100 TCC 环境中稳定复现。
- probe 已能独立指定物理 buffer 和逻辑 payload，并记录 endpoint/payload CUDA allocation。
- transport 已支持大于 capacity 的 equal-split CUDA all-to-all，synthetic Gate B/D 通过。
- Gate C 已证明 staging CUDA allocation 恒定、真实工作流系统内存/commit/pagefile 无 payload 等量 host staging，并完成进程退出回收。
- Gate E 的历史大 payload、I2V、REF2VA、Director 多段/Refine 与 LTX 冷/热全部通过；视频和音频检查通过。
- 下一步完成 Gate F 的版本、文档、完整测试、部署一致性、节点导入、P2P probe 与最终 smoke；全部通过后才解除 O10 阻塞。
