import pytest

from raylight.text_encoder_parallel.topology import (
    GpuDescriptor,
    PeerTopologyError,
    validate_gpu_descriptors,
)


def _descriptors() -> list[GpuDescriptor]:
    return [
        GpuDescriptor(index=0, name="Tesla V100-SXM2-16GB", driver_model="TCC"),
        GpuDescriptor(index=1, name="Tesla V100-SXM2-16GB", driver_model="TCC"),
    ]


def test_accepts_exactly_two_tcc_v100_with_bidirectional_peer_access() -> None:
    result = validate_gpu_descriptors(_descriptors(), {(0, 1): True, (1, 0): True})

    assert result == (0, 1)


@pytest.mark.parametrize(
    ("descriptors", "peer_access", "message"),
    [
        (_descriptors()[:1], {(0, 1): True, (1, 0): True}, "exactly two"),
        (
            [
                GpuDescriptor(0, "Tesla V100-SXM2-16GB", "TCC"),
                GpuDescriptor(2, "Tesla V100-SXM2-16GB", "TCC"),
            ],
            {(0, 2): True, (2, 0): True},
            "indices",
        ),
        (
            [
                GpuDescriptor(0, "Tesla V100-SXM2-16GB", "TCC"),
                GpuDescriptor(1, "Tesla A100-SXM4-40GB", "TCC"),
            ],
            {(0, 1): True, (1, 0): True},
            "V100",
        ),
        (
            [
                GpuDescriptor(0, "Tesla V100-SXM2-16GB", "TCC"),
                GpuDescriptor(1, "Tesla V100-SXM2-16GB", "WDDM"),
            ],
            {(0, 1): True, (1, 0): True},
            "TCC",
        ),
        (_descriptors(), {(0, 1): True, (1, 0): False}, "bidirectional"),
    ],
)
def test_rejects_unsupported_topology(descriptors, peer_access, message: str) -> None:
    with pytest.raises(PeerTopologyError, match=message):
        validate_gpu_descriptors(descriptors, peer_access)
