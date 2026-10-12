from __future__ import annotations

import sys
from collections import OrderedDict
from types import ModuleType
from unittest import mock

import pytest

from raylight.integrations.mmh3director import bridge
from raylight.integrations.mmh3director.metadata import MMH3DirectorModelMetadata
from raylight.integrations.mmh3director.proxy import RayMMH3DirectorModelProxy
from raylight.integrations.mmh3director import segment_lora
from raylight.integrations.mmh3director.segment_lora import (
    apply_worker_segment_loras,
    make_segment_lora_wrapper,
)


def _proxy():
    return RayMMH3DirectorModelProxy(
        ray_actors={"workers": ("rank-0", "rank-1"), "world_size": 2},
        metadata=MMH3DirectorModelMetadata(
            model_family="minimax_h3",
            sampling_settings={"shift": 12.0, "audio_shift": 3.0},
            latent_channels=32,
            latent_dimensions=3,
            spacial_downscale_ratio=16,
            temporal_downscale_ratio=4,
            noise_scale=1.0,
        ),
    )


def test_segment_lora_wrapper_uses_descriptors_for_proxy_and_original_for_normal_model(tmp_path):
    original = mock.Mock(return_value="local")
    wrapped = make_segment_lora_wrapper(original)
    proxy = _proxy()
    lora_path = tmp_path / "x.safetensors"

    with mock.patch("folder_paths.get_full_path_or_raise", return_value=str(lora_path)) as resolve:
        adapted = wrapped(
            proxy,
            [{"name": "x.safetensors", "strength": 0.5, "active": True}],
            state_dict_cache={"must": "stay unused"},
        )

    resolve.assert_called_once_with("loras", "x.safetensors")
    original.assert_not_called()
    assert adapted is not proxy
    assert adapted.segment_loras == (
        {"path": str(lora_path), "strength_model": 0.5},
    )

    normal = object()
    cache = {}
    assert wrapped(normal, [{"name": "x.safetensors"}], state_dict_cache=cache) == "local"
    original.assert_called_once_with(
        normal,
        [{"name": "x.safetensors"}],
        state_dict_cache=cache,
    )


def test_segment_lora_wrapper_skips_inactive_and_zero_rows_and_caps_stack(tmp_path):
    wrapped = make_segment_lora_wrapper(mock.Mock())
    rows = [
        {"name": "off.safetensors", "strength": 1.0, "active": False},
        {"name": "zero.safetensors", "strength": 0.0, "active": True},
        *(
            {"name": f"lora-{index}.safetensors", "strength": index + 0.25}
            for index in range(10)
        ),
    ]

    with mock.patch(
        "folder_paths.get_full_path_or_raise",
        side_effect=lambda _category, name: str(tmp_path / name),
    ):
        adapted = wrapped(_proxy(), rows)

    assert len(adapted.segment_loras) == 8
    assert all("off.safetensors" not in item["path"] for item in adapted.segment_loras)
    assert all("zero.safetensors" not in item["path"] for item in adapted.segment_loras)


def test_segment_lora_wrapper_reports_missing_model_file():
    wrapped = make_segment_lora_wrapper(mock.Mock())

    with (
        mock.patch(
            "folder_paths.get_full_path_or_raise",
            side_effect=FileNotFoundError("missing"),
        ),
        pytest.raises(ValueError, match="segment LoRA 'missing.safetensors'.*not found"),
    ):
        wrapped(_proxy(), [{"name": "missing.safetensors", "strength": 1.0}])


def test_installer_patches_only_exact_director_aliases_and_restores_them():
    def original(model, rows, *, state_dict_cache=None):
        del model, rows, state_dict_cache

    segment = ModuleType("ComfyUI_MiniMaxH3_Director.director.segment_loras")
    segment.__file__ = "C:/nodes/ComfyUI_MiniMaxH3_Director/director/segment_loras.py"
    segment.apply_segment_loras = original
    executor = ModuleType("ComfyUI_MiniMaxH3_Director.director.executor_core")
    executor.__file__ = "C:/nodes/ComfyUI_MiniMaxH3_Director/director/executor_core.py"
    executor.apply_segment_loras = original
    unrelated = ModuleType("unrelated.segment_loras")
    unrelated.__file__ = "C:/nodes/unrelated/segment_loras.py"
    unrelated.apply_segment_loras = original

    bridge.uninstall_mmh3director_sample_bridge_for_test()
    try:
        with mock.patch.dict(
            sys.modules,
            {
                segment.__name__: segment,
                executor.__name__: executor,
                unrelated.__name__: unrelated,
            },
            clear=False,
        ):
            bridge.install_mmh3director_sample_bridge(require_director=False)

            assert segment.apply_segment_loras is executor.apply_segment_loras
            assert segment.apply_segment_loras is not original
            assert unrelated.apply_segment_loras is original

            bridge.uninstall_mmh3director_sample_bridge_for_test()
            assert segment.apply_segment_loras is original
            assert executor.apply_segment_loras is original
    finally:
        bridge.uninstall_mmh3director_sample_bridge_for_test()


def test_worker_segment_lora_applies_to_temporary_sampling_clone_only():
    base = object()
    patched = object()

    with (
        mock.patch.object(segment_lora, "_load_lora_state", return_value={"lora": 1}),
        mock.patch(
            "raylight.comfy_dist.sd.load_lora_for_models_quantized",
            return_value=patched,
        ) as load,
    ):
        result = apply_worker_segment_loras(
            base,
            [{"path": "x.safetensors", "strength_model": 0.8}],
            is_fsdp=True,
            is_quant=True,
            dynamic_sidecar=False,
        )

    assert result is patched
    assert result is not base
    load.assert_called_once_with(
        base,
        {"lora": 1},
        0.8,
        dynamic_sidecar=False,
        defer_device_move=True,
        fallback_to_patches=False,
    )


def test_worker_segment_lora_state_cache_is_lru_bounded_to_two_files(tmp_path):
    paths = []
    for name in ("a", "b", "c"):
        path = tmp_path / f"{name}.safetensors"
        path.write_bytes(name.encode("ascii"))
        paths.append(path)
    cache = OrderedDict()

    with mock.patch(
        "comfy.utils.load_torch_file",
        side_effect=lambda path, safe_load: {"path": path, "safe": safe_load},
    ) as load:
        first = segment_lora._load_lora_state(str(paths[0]), state_cache=cache)
        assert segment_lora._load_lora_state(str(paths[0]), state_cache=cache) is first
        segment_lora._load_lora_state(str(paths[1]), state_cache=cache)
        segment_lora._load_lora_state(str(paths[2]), state_cache=cache)

    assert load.call_count == 3
    assert len(cache) == 2
    assert {key[0] for key in cache} == {str(paths[1]), str(paths[2])}
