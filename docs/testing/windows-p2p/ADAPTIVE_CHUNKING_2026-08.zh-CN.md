# O9 Windows CUDA P2P 自适应分块测试记录

日期：2026-08-31

状态：Task 1 基线已冻结；transport 尚未修改。

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

## Task 1 结论

- 旧失败已在空闲双 V100 TCC 环境中稳定复现。
- probe 已能独立指定物理 buffer 和逻辑 payload，并记录 endpoint/payload CUDA allocation。
- transport 没有修改；当前大于 capacity 的 all-to-all 仍按预期失败。
- 下一步进入 Task 2，先用失败测试定义两个 rank 的精确分块布局。
