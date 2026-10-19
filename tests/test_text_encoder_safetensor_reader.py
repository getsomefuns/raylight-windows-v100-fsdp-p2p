from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from raylight.text_encoder_parallel.safetensor_reader import NonMmapSafeTensorReader


def test_non_mmap_reader_detaches_tensor_storage_from_checkpoint_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.safetensors"
    expected = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    save_file({"model.weight": expected}, checkpoint)

    with NonMmapSafeTensorReader(checkpoint) as reader:
        assert reader.keys() == ["model.weight"]
        loaded = reader.get_tensor("model.weight")

    renamed = checkpoint.with_suffix(".moved")
    checkpoint.rename(renamed)

    assert torch.equal(loaded, expected)


def test_non_mmap_reader_supports_nvfp4_storage_dtypes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "nvfp4.safetensors"
    qdata = torch.arange(16, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(4, 4)
    block_scale = torch.arange(8, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(2, 4)
    tensor_scale = torch.tensor([0.125], dtype=torch.float32)
    save_file(
        {
            "layer.weight": qdata,
            "layer.weight_scale": block_scale,
            "layer.weight_scale_2": tensor_scale,
        },
        checkpoint,
    )

    with NonMmapSafeTensorReader(checkpoint) as reader:
        assert torch.equal(reader.get_tensor("layer.weight").view(torch.uint8), qdata.view(torch.uint8))
        assert torch.equal(
            reader.get_tensor("layer.weight_scale").view(torch.uint8), block_scale.view(torch.uint8)
        )
        assert torch.equal(reader.get_tensor("layer.weight_scale_2"), tensor_scale)

