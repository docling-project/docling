# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import hashlib
from io import BytesIO
from pathlib import Path

import pytest

from docling.utils.utils import create_file_hash


@pytest.fixture
def payload() -> bytes:
    return b'{"name": "doc", "texts": [{"text": "first"}]}'


def test_hash_of_a_stream_that_was_written_to_but_not_rewound(payload: bytes):
    """A stream left at its end hashes as empty input, so every document looks the same."""
    stream = BytesIO()
    stream.write(payload)

    assert create_file_hash(stream) == hashlib.sha256(payload).hexdigest()


def test_stream_and_path_hash_the_same_bytes(payload: bytes, tmp_path: Path):
    path = tmp_path / "doc.dclg"
    path.write_bytes(payload)

    assert create_file_hash(path) == create_file_hash(BytesIO(payload))


def test_stream_position_is_left_where_it_was(payload: bytes):
    stream = BytesIO(payload)
    stream.seek(5)

    create_file_hash(stream)

    assert stream.tell() == 5


def test_different_documents_get_different_hashes(payload: bytes):
    other = bytearray(payload)
    other[-1] = ord("o")

    first = BytesIO()
    first.write(payload)
    second = BytesIO()
    second.write(bytes(other))

    assert create_file_hash(first) != create_file_hash(second)
