from unittest import mock

import pytest

from raylight.integrations.mmh3director import nodes as director_nodes
from raylight.integrations.mmh3director.proxy import RayMMH3DirectorModelProxy


def _rank_row(rank, model_family="minimax_h3"):
    return {
        "rank": rank,
        "model_family": model_family,
        "sampling_settings": {"shift": 12.0, "audio_shift": 3.0},
        "latent_channels": 32,
        "latent_dimensions": 3,
        "spacial_downscale_ratio": 16,
        "temporal_downscale_ratio": 4,
        "noise_scale": 1.0,
    }


class _RemoteMethod:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def remote(self):
        self.calls += 1
        return self.result


class _Actor:
    def __init__(self, metadata):
        self.get_mmh3director_proxy_metadata = _RemoteMethod(metadata)


def test_adapter_node_has_one_input_one_output_and_fixed_registration():
    node_class = director_nodes.RayMMH3DirectorModelAdapter

    assert node_class.INPUT_TYPES() == {"required": {"ray_actors": ("RAY_ACTORS",)}}
    assert node_class.RETURN_TYPES == ("MODEL",)
    assert node_class.RETURN_NAMES == ("model",)
    assert node_class.FUNCTION == "adapt_model"
    assert node_class.CATEGORY == "Raylight/integrations/MMH3Director"
    assert director_nodes.NODE_CLASS_MAPPINGS == {
        "RayMMH3DirectorModelAdapter": node_class,
    }
    assert director_nodes.NODE_DISPLAY_NAME_MAPPINGS == {
        "RayMMH3DirectorModelAdapter": "Ray MODEL Adapter (MMH3Director)",
    }


def test_adapter_reads_metadata_from_every_rank_and_returns_weightless_proxy():
    actors = [_Actor(_rank_row(0)), _Actor(_rank_row(1))]
    ray_actors = {
        "workers": actors,
        "world_size": 2,
        "session_id": "session-a",
        "generation_id": 11,
    }

    with (
        mock.patch.object(director_nodes.ray, "get", side_effect=lambda futures: futures),
        mock.patch.object(director_nodes, "install_mmh3director_sample_bridge") as install_bridge,
    ):
        (proxy,) = director_nodes.RayMMH3DirectorModelAdapter().adapt_model(ray_actors)

    assert isinstance(proxy, RayMMH3DirectorModelProxy)
    assert proxy.ray_actors is ray_actors
    assert proxy.session_id == "session-a"
    assert proxy.generation_id == 11
    assert [actor.get_mmh3director_proxy_metadata.calls for actor in actors] == [1, 1]
    install_bridge.assert_called_once_with(require_director=True)


@pytest.mark.parametrize(
    "ray_actors, message",
    [
        ({}, "RAY_ACTORS payload must be a dict with workers"),
        ({"workers": []}, "requires at least one loaded Ray worker"),
        ({"workers": [_Actor(None)]}, "rank 0 has no loaded model metadata"),
        (
            {"workers": [_Actor(_rank_row(0, model_family="ltx"))]},
            "requires MiniMax H3 workers; found ltx",
        ),
    ],
)
def test_adapter_fails_closed_for_invalid_or_non_minimax_workers(ray_actors, message):
    with (
        mock.patch.object(director_nodes.ray, "get", side_effect=lambda futures: futures),
        pytest.raises((TypeError, ValueError, RuntimeError), match=message),
    ):
        director_nodes.RayMMH3DirectorModelAdapter().adapt_model(ray_actors)


def test_adapter_rejects_world_size_that_disagrees_with_worker_count():
    ray_actors = {"workers": [_Actor(_rank_row(0))], "world_size": 2}

    with pytest.raises(ValueError, match="world_size=2.*1 workers"):
        director_nodes.RayMMH3DirectorModelAdapter().adapt_model(ray_actors)
