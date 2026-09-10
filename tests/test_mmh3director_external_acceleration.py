from types import SimpleNamespace

from raylight.integrations.mmh3director.external_acceleration import (
    RAYLIGHT_SOLATTN_CONFIG_KEY,
    RAYLIGHT_TE_SPEED_CONFIG_KEY,
    apply_external_acceleration,
    serialize_external_acceleration,
)


class _TESpeedRuntime:
    threshold = 0.08
    start_percent = 0.10
    end_percent = 0.90
    max_consecutive_skips = 2
    cache_device = "auto"
    block_count = 50
    turbo_variant = "four"


def _sol_override(*args, **kwargs):
    raise AssertionError("main-process Sol callable must not cross the Ray boundary")


_sol_override._solattn_override = True


def test_serialize_external_acceleration_converts_native_objects_to_plain_config():
    options = {
        "transformer_options": {
            "keep": "unchanged",
            "te_speed_minimax_h3_runtime": _TESpeedRuntime(),
            "optimized_attention_override": _sol_override,
            "solattn_sparse": True,
            "solattn_tau": 0.75,
            "solattn_start_percent": 0.2,
            "solattn_end_percent": 0.9,
            "solattn_min_tokens": 1024,
            "solattn_dense_blocks": frozenset({0, 1, 49}),
            "solattn_prefix_tokens": 1024,
            "solattn_topk": 32,
        }
    }

    serialized = serialize_external_acceleration(options)
    transformer = serialized["transformer_options"]

    assert transformer["keep"] == "unchanged"
    assert "te_speed_minimax_h3_runtime" not in transformer
    assert "optimized_attention_override" not in transformer
    assert transformer[RAYLIGHT_TE_SPEED_CONFIG_KEY] == {
        "processing_control_value": 0.08,
        "processing_percent_1": 0.1,
        "processing_percent_2": 0.9,
        "mcs": 2,
        "device": "auto",
        "mode": "4-step LoRA",
    }
    assert transformer[RAYLIGHT_SOLATTN_CONFIG_KEY] == {
        "fp16_safe": False,
        "tau": 0.75,
        "start_percent": 0.2,
        "end_percent": 0.9,
        "min_tokens": 1024,
        "dense_blocks": "0-1,49",
        "h3_prefix_tokens": 1024,
        "topk_blocks": 32,
    }


def test_apply_external_acceleration_replays_nodes_worker_side_without_leaking_config():
    calls = []

    class TE:
        def patch(self, model, **kwargs):
            calls.append(("te", kwargs))
            return (SimpleNamespace(name="te", model_options=model.model_options),)

    class Sol:
        def patch(self, model, **kwargs):
            calls.append(("sol", kwargs))
            return (SimpleNamespace(name="sol", model_options=model.model_options),)

    model = SimpleNamespace(
        name="base",
        model_options={
            "transformer_options": {
                RAYLIGHT_TE_SPEED_CONFIG_KEY: {
                    "processing_control_value": 0.08,
                    "processing_percent_1": 0.1,
                    "processing_percent_2": 0.9,
                    "mcs": 2,
                    "device": "auto",
                    "mode": "4-step LoRA",
                },
                RAYLIGHT_SOLATTN_CONFIG_KEY: {
                    "fp16_safe": False,
                    "tau": 0.75,
                    "start_percent": 0.2,
                    "end_percent": 0.9,
                    "min_tokens": 1024,
                    "dense_blocks": "0-1,49",
                    "h3_prefix_tokens": 1024,
                    "topk_blocks": 32,
                },
            }
        },
    )

    patched = apply_external_acceleration(
        model,
        te_speed_factory=TE,
        solattn_factory=Sol,
    )

    assert patched.name == "sol"
    assert [name for name, _kwargs in calls] == ["te", "sol"]
    assert RAYLIGHT_TE_SPEED_CONFIG_KEY not in patched.model_options["transformer_options"]
    assert RAYLIGHT_SOLATTN_CONFIG_KEY not in patched.model_options["transformer_options"]
    assert calls[1][1]["fp16_safe"] is False

