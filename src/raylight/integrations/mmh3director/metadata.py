from __future__ import annotations

import copy
from dataclasses import dataclass


_METADATA_FIELDS = (
    "model_family",
    "sampling_settings",
    "latent_channels",
    "latent_dimensions",
    "spacial_downscale_ratio",
    "temporal_downscale_ratio",
    "noise_scale",
)


@dataclass(frozen=True)
class MMH3DirectorModelMetadata:
    model_family: str
    sampling_settings: dict[str, object]
    latent_channels: int
    latent_dimensions: int
    spacial_downscale_ratio: int
    temporal_downscale_ratio: int
    noise_scale: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampling_settings", copy.deepcopy(self.sampling_settings))


def validate_rank_metadata(rows: list[dict]) -> MMH3DirectorModelMetadata:
    """Validate that every worker describes the same lightweight model facade."""

    if not rows:
        raise ValueError("MMH3Director metadata requires at least one rank")

    rows_by_rank: dict[int, dict] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"rank metadata row {row_index} must be a dict")
        if "rank" not in row:
            raise ValueError(f"rank metadata row {row_index} is missing field rank")
        rank = row["rank"]
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ValueError(f"rank metadata row {row_index} has invalid rank {rank!r}")
        if rank in rows_by_rank:
            raise ValueError(f"duplicate rank {rank} in MMH3Director metadata")
        for field in _METADATA_FIELDS:
            if field not in row:
                raise ValueError(f"rank {rank} metadata is missing field {field}")
        rows_by_rank[rank] = row

    ordered_ranks = sorted(rows_by_rank)
    expected_ranks = list(range(len(rows_by_rank)))
    if ordered_ranks != expected_ranks:
        raise ValueError(f"MMH3Director metadata ranks must be contiguous {expected_ranks}; found {ordered_ranks}")

    reference = rows_by_rank[0]
    for rank in ordered_ranks[1:]:
        row = rows_by_rank[rank]
        for field in _METADATA_FIELDS:
            if row[field] != reference[field]:
                raise ValueError(f"rank {rank} metadata field {field} differs from rank 0: {row[field]!r} != {reference[field]!r}")

    return MMH3DirectorModelMetadata(
        model_family=str(reference["model_family"]),
        sampling_settings=copy.deepcopy(reference["sampling_settings"]),
        latent_channels=int(reference["latent_channels"]),
        latent_dimensions=int(reference["latent_dimensions"]),
        spacial_downscale_ratio=int(reference["spacial_downscale_ratio"]),
        temporal_downscale_ratio=int(reference["temporal_downscale_ratio"]),
        noise_scale=None if reference["noise_scale"] is None else float(reference["noise_scale"]),
    )
