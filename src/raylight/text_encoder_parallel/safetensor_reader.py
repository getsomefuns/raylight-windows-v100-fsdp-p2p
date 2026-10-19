from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import struct
import threading
from typing import Iterator

import torch


_SAFETENSOR_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "U16": torch.uint16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "I64": torch.int64,
    "U64": torch.uint64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "C64": torch.complex64,
    "C128": torch.complex128,
}


class NonMmapSafeTensorReader:
    """Read individual safetensors into owned CPU buffers without mmap.

    ``safetensors.safe_open`` uses a memory mapping.  Large, repeatedly opened
    checkpoints can consume the Windows commit limit and eventually crash in
    the Rust/PyTorch storage hand-off instead of raising a Python exception.
    This reader keeps only one requested tensor in flight and returns storage
    backed by a Python bytearray, so closing the reader releases the file
    immediately and never invalidates an already returned tensor.
    """

    def __init__(self, filename: str | Path) -> None:
        self.path = Path(filename)
        self._file = self.path.open("rb", buffering=0)
        self._lock = threading.Lock()
        try:
            prefix = self._file.read(8)
            if len(prefix) != 8:
                raise ValueError(f"invalid safetensors header in {self.path}")
            header_size = struct.unpack("<Q", prefix)[0]
            file_size = self.path.stat().st_size
            if header_size <= 0 or header_size > file_size - 8:
                raise ValueError(f"invalid safetensors header size {header_size} in {self.path}")
            raw_header = self._file.read(header_size)
            if len(raw_header) != header_size:
                raise ValueError(f"incomplete safetensors header in {self.path}")
            header = json.loads(raw_header)
            self._entries = {
                key: value for key, value in header.items() if key != "__metadata__"
            }
            self._data_offset = 8 + header_size
            self._file_size = file_size
            self._validate_entries()
        except BaseException:
            self._file.close()
            raise

    def _validate_entries(self) -> None:
        data_size = self._file_size - self._data_offset
        for key, entry in self._entries.items():
            if not isinstance(entry, dict):
                raise ValueError(f"invalid safetensors entry {key!r} in {self.path}")
            dtype_name = entry.get("dtype")
            if dtype_name not in _SAFETENSOR_DTYPES:
                raise ValueError(f"unsupported safetensors dtype {dtype_name!r} for {key!r}")
            offsets = entry.get("data_offsets")
            shape = entry.get("shape")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
                or offsets[1] > data_size
                or not isinstance(shape, list)
                or not all(isinstance(value, int) and value >= 0 for value in shape)
            ):
                raise ValueError(f"invalid safetensors metadata for {key!r} in {self.path}")
            expected_size = math.prod(shape) * torch.empty(
                (), dtype=_SAFETENSOR_DTYPES[dtype_name]
            ).element_size()
            if offsets[1] - offsets[0] != expected_size:
                raise ValueError(f"invalid safetensors byte size for {key!r} in {self.path}")

    def keys(self) -> list[str]:
        return list(self._entries)

    def get_tensor(self, key: str) -> torch.Tensor:
        if self._file.closed:
            raise RuntimeError("safetensors reader is closed")
        try:
            entry = self._entries[key]
        except KeyError as exc:
            raise KeyError(f"tensor {key!r} is not present in {self.path}") from exc
        start, end = entry["data_offsets"]
        size = end - start
        dtype = _SAFETENSOR_DTYPES[entry["dtype"]]
        shape = tuple(entry["shape"])
        if size == 0:
            return torch.empty(shape, dtype=dtype)

        buffer = bytearray(size)
        view = memoryview(buffer)
        with self._lock:
            self._file.seek(self._data_offset + start)
            offset = 0
            while offset < size:
                count = self._file.readinto(view[offset:])
                if not count:
                    raise EOFError(f"incomplete tensor data for {key!r} in {self.path}")
                offset += count
        return torch.frombuffer(buffer, dtype=dtype).reshape(shape)

    def close(self) -> None:
        with self._lock:
            self._file.close()

    def __enter__(self) -> "NonMmapSafeTensorReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


@contextmanager
def open_text_encoder_safetensors(
    filename: str | Path,
) -> Iterator[NonMmapSafeTensorReader | object]:
    """Use commit-safe reads on Windows and the native mmap reader elsewhere."""

    if os.name == "nt":
        with NonMmapSafeTensorReader(filename) as reader:
            yield reader
        return

    from safetensors import safe_open

    with safe_open(filename, framework="pt", device="cpu") as reader:
        yield reader


__all__ = ["NonMmapSafeTensorReader", "open_text_encoder_safetensors"]
