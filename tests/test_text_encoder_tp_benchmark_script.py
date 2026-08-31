import importlib.util
import json
from pathlib import Path

import pytest

from raylight.text_encoder_parallel.baseline import BaselineRecord


SCRIPT = Path(__file__).parents[1] / "scripts" / "minimax-h3" / "benchmark_text_encoder_tp2.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("benchmark_text_encoder_tp2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow() -> dict:
    timeline = {
        "segments": [
            {
                "frameCount": 73,
                "prompt": "A deliberate baseline prompt.",
                "refs": [{"imageFile": "reference.png", "type": "input", "subfolder": ""}],
            }
        ]
    }
    return {
        "nodes": [
            {
                "id": 2,
                "type": "CLIPLoader",
                "widgets_values": ["MMH3\\qwen.safetensors", "minimax", "default"],
            },
            {
                "id": 3,
                "type": "VAELoader",
                "title": "Video VAE",
                "widgets_values": ["MMH3\\video.safetensors"],
            },
            {
                "id": 4,
                "type": "VAELoader",
                "title": "Audio VAE",
                "widgets_values": ["MMH3\\audio.safetensors"],
            },
            {
                "id": 5,
                "type": "MiniMaxH3Director",
                "widgets_values": ["r2v", json.dumps(timeline)],
            },
        ]
    }


def test_loads_current_director_input_and_standard_clip_loader(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(_workflow()), encoding="utf-8")

    baseline = module.load_baseline_spec(path)

    assert baseline.checkpoint == "MMH3\\qwen.safetensors"
    assert baseline.video_vae == "MMH3\\video.safetensors"
    assert baseline.audio_vae == "MMH3\\audio.safetensors"
    assert baseline.prompt == "A deliberate baseline prompt."
    assert baseline.reference_image == "reference.png"
    assert len(baseline.workflow_sha256) == 64


def test_builds_conditioning_only_comfyui_prompt(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(_workflow()), encoding="utf-8")
    baseline = module.load_baseline_spec(path)

    prompt = module.build_conditioning_prompt(baseline, length=90)

    assert {node["class_type"] for node in prompt.values()} == {
        "CLIPLoader",
        "VAELoader",
        "LoadImage",
        "MiniMaxH3ReferenceToVideo",
        "PreviewAny",
    }
    conditioning = next(
        node for node in prompt.values() if node["class_type"] == "MiniMaxH3ReferenceToVideo"
    )
    assert conditioning["inputs"]["length"] == 90
    assert conditioning["inputs"]["prompt"] == "A deliberate baseline prompt."


def test_builds_tp2_conditioning_prompt_with_real_ray_dependency(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(_workflow()), encoding="utf-8")
    baseline = module.load_baseline_spec(path)

    prompt = module.build_conditioning_prompt(baseline, length=90, engine="tp2")

    by_type = {node["class_type"]: node for node in prompt.values()}
    assert "CLIPLoader" not in by_type
    assert by_type["RayInitializer"]["inputs"] == {
        "ray_cluster_address": "local",
        "ray_cluster_namespace": "default",
        "GPU": 2,
        "ulysses_degree": 2,
        "ring_degree": 1,
        "cfg_degree": 1,
        "dp_degree": 1,
        "sync_ulysses": True,
        "clear_vram_after_sampling": False,
        "FSDP": True,
        "FSDP_CPU_OFFLOAD": True,
        "XFuser_attention": "TORCH_EFFICIENT",
        "skip_comm_test": True,
        "use_mmap": True,
    }
    loader = by_type["RayMiniMaxH3DualGPUCLIPLoader"]
    assert loader["inputs"]["execution_mode"] == "tp2"
    assert loader["inputs"]["ray_actors_init"] == ["0", 0]
    conditioning = by_type["MiniMaxH3ReferenceToVideo"]
    assert conditioning["inputs"]["clip"] == ["1", 0]


def test_rejects_unknown_benchmark_engine(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(_workflow()), encoding="utf-8")
    baseline = module.load_baseline_spec(path)

    with pytest.raises(ValueError, match="engine"):
        module.build_conditioning_prompt(baseline, length=90, engine="auto")


def test_event_pair_requires_one_begin_and_one_complete() -> None:
    module = _load_script()
    events = [
        {"event": "text_encoder_begin", "time_ns": 10, "token_count": 12},
        {
            "event": "text_encoder_complete",
            "time_ns": 20,
            "token_count": 12,
            "te_total_s": 1.25,
        },
    ]

    assert module.parse_event_pair(events)["te_total_s"] == 1.25
    with pytest.raises(ValueError, match="complete"):
        module.parse_event_pair(events[:1])


def test_summarizes_exact_tp2_stage_set_and_rejects_failures() -> None:
    module = _load_script()
    events = [
        {
            "event": "text_encoder_stage",
            "stage": stage,
            "elapsed_s": index / 10,
            "failed": False,
        }
        for index, stage in enumerate(module.TP2_STAGE_NAMES, start=1)
    ]

    summary = module.summarize_stage_events(events)

    assert list(summary) == list(module.TP2_STAGE_NAMES)
    assert summary["language_backbone"] > 0
    events[-1]["failed"] = True
    with pytest.raises(ValueError, match="failed"):
        module.summarize_stage_events(events)


def test_summary_reports_cold_and_warm_medians() -> None:
    module = _load_script()
    records = []
    for index, value in enumerate((30.0, 20.0, 10.0), start=1):
        records.append(
            BaselineRecord.from_mapping(
                {
                    "checkpoint": "qwen.safetensors",
                    "workflow_sha256": "a" * 64,
                    "mode": "cold",
                    "run_index": index,
                    "token_count": 100,
                    "te_total_s": value,
                    "gpu0_peak_mib": 10_000 + index,
                    "ram_peak_mib": 20_000 + index,
                    "pagefile_peak_mib": 3_000 + index,
                }
            )
        )
    summary = module.summarize_records(records)

    assert summary["cold"]["runs"] == 3
    assert summary["cold"]["median_te_total_s"] == 20.0
    assert summary["cold"]["peak_gpu0_mib"] == 10_003
    assert summary["warm"]["runs"] == 0
