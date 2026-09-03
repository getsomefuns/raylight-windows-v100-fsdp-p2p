from types import SimpleNamespace

from scripts.runtime.verify_dynamic_vram_runtime import activate_dynamic_vram, validate_runtime_facts


def _candidate_facts(**overrides):
    facts = {
        "torch": "2.8.0+cu128",
        "torchvision": "0.23.0+cu128",
        "torchaudio": "2.8.0+cu128",
        "xformers": "0.0.32.post2",
        "comfy_aimdo": "0.4.13+tcc1",
        "cuda": "12.8",
        "arch_list": ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
        "device_capabilities": [[7, 0], [7, 0]],
        "dynamic_vram": True,
        "xformers_forward": True,
        "aimdo_devices": [0, 1],
        "aimdo_host_buffer_lib": True,
        "aimdo_nvml_fail_closed": True,
        "cuda_ipc_round_trip": True,
    }
    facts.update(overrides)
    return facts


def test_old_runtime_fails_candidate_and_dynamic_vram_contracts():
    facts = _candidate_facts(
        torch="2.7.0+cu126",
        torchvision="0.22.0+cu126",
        torchaudio="2.7.0+cu126",
        xformers="0.0.30",
        comfy_aimdo="0.4.13",
        cuda="12.6",
        dynamic_vram=False,
    )

    errors = validate_runtime_facts(facts, require_dynamic_vram=True)

    assert "torch must be 2.8.0+cu128" in errors
    assert "torchvision must be 0.23.0+cu128" in errors
    assert "torchaudio must be 2.8.0+cu128" in errors
    assert "xformers must be 0.0.32.post2" in errors
    assert "comfy_aimdo must be 0.4.13+tcc1" in errors
    assert "cuda must be 12.8" in errors
    assert "DynamicVRAM is not active" in errors


def test_release_probe_rejects_missing_v100_and_gpu_subprobes():
    facts = _candidate_facts(
        arch_list=["sm_75", "sm_80"],
        device_capabilities=[[7, 0]],
        xformers_forward=False,
        aimdo_devices=[0],
        aimdo_host_buffer_lib=False,
        aimdo_nvml_fail_closed=False,
        cuda_ipc_round_trip=False,
    )

    errors = validate_runtime_facts(facts, require_dynamic_vram=True)

    assert "torch wheel does not contain sm_70" in errors
    assert "both devices must be compute capability 7.0" in errors
    assert "xFormers V100 forward failed" in errors
    assert "AIMDO did not register both devices" in errors
    assert "AIMDO host buffer library is unavailable" in errors
    assert "AIMDO did not fail closed without NVML pressure" in errors
    assert "CUDA IPC round-trip failed" in errors


def test_candidate_runtime_passes_all_release_gates():
    assert validate_runtime_facts(_candidate_facts(), require_dynamic_vram=True) == []


def test_baseline_collection_can_skip_dynamic_vram_requirement():
    facts = _candidate_facts(dynamic_vram=False)

    errors = validate_runtime_facts(facts, require_dynamic_vram=False)

    assert "DynamicVRAM is not active" not in errors


def test_activate_dynamic_vram_mirrors_comfyui_startup_sequence():
    calls = []
    dynamic_patcher = object()
    legacy_patcher = object()
    control = SimpleNamespace(
        init=lambda **kwargs: calls.append(("init", kwargs)),
        init_devices=lambda devices: calls.append(("init_devices", list(devices))) or True,
    )
    model_management = SimpleNamespace(
        get_all_torch_devices=lambda: [SimpleNamespace(index=0), SimpleNamespace(index=1)]
    )
    memory_management = SimpleNamespace(aimdo_enabled=False)
    model_patcher = SimpleNamespace(
        CoreModelPatcher=legacy_patcher,
        ModelPatcherDynamic=dynamic_patcher,
    )

    def load_modules():
        calls.append(("load_modules", None))
        return model_management, memory_management, model_patcher

    active = activate_dynamic_vram(control, load_modules, vram_headroom_gib=0)

    assert active is True
    assert calls == [
        ("init", {"simple_vram_headroom": None, "nvml_pressure": True}),
        ("load_modules", None),
        ("init_devices", [(0, 0), (1, 0)]),
    ]
    assert memory_management.aimdo_enabled is True
    assert model_patcher.CoreModelPatcher is dynamic_patcher
