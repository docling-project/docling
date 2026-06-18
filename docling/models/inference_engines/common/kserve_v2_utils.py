"""Shared helpers for KServe v2 binary tensor encoding and decoding."""

from __future__ import annotations

from typing import Any

import numpy as np


def encode_bytes_element(value: Any) -> bytes:
    """Encode a scalar value to the KServe/Triton BYTES wire format."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray | memoryview | np.bytes_):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return str(value).encode("utf-8")


def encode_bytes_tensor(tensor: np.ndarray) -> bytes:
    """Encode a BYTES tensor as a length-prefixed byte stream."""
    chunks: list[bytes] = []
    for value in tensor.reshape(-1):
        encoded = encode_bytes_element(value)
        chunks.append(len(encoded).to_bytes(4, byteorder="little"))
        chunks.append(encoded)
    return b"".join(chunks)


def decode_bytes_tensor(raw_output: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Decode a length-prefixed BYTES payload to a numpy object array."""
    strings, offset = [], 0
    for _ in range(int(np.prod(shape))):
        if offset + 4 > len(raw_output):
            raise RuntimeError(
                f"Invalid BYTES data: insufficient bytes for length prefix at offset {offset}"
            )
        str_len = int.from_bytes(raw_output[offset : offset + 4], byteorder="little")
        offset += 4
        if offset + str_len > len(raw_output):
            raise RuntimeError(
                f"Invalid BYTES data: insufficient bytes for string of length {str_len} at offset {offset}"
            )
        strings.append(raw_output[offset : offset + str_len])
        offset += str_len
    return np.array(strings, dtype=object).reshape(shape)
