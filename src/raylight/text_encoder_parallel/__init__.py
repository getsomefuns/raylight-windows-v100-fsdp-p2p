"""MiniMax H3 text-encoder parallelism experiments."""

from .baseline import BaselineRecord
from .collectives import CudaPeerGroup2, PeerCollectiveStats

__all__ = ["BaselineRecord", "CudaPeerGroup2", "PeerCollectiveStats"]
