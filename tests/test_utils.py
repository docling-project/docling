
# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import hashlib
from io import BytesIO
from pathlib import Path

from docling.utils.utils import create_file_hash


def test_create_file_hash_bytesio_at_end():
    content = b"Hello World"
    stream = BytesIO(content)
    stream.seek(0, 2)

    actual = create_file_hash(stream)
    expected = hashlib.sha256(content).hexdigest()

    assert actual == expected
    assert stream.tell() == len(content)


def test_create_file_hash_bytesio_at_middle():
    content = b"Hello World"
    stream = BytesIO(content)
    stream.seek(5)

    actual = create_file_hash(stream)
    expected = hashlib.sha256(content).hexdigest()

    assert actual == expected
    assert stream.tell() == 5


def test_create_file_hash_empty_bytesio():
    stream = BytesIO()

    actual = create_file_hash(stream)
    expected = hashlib.sha256(b"").hexdigest()

    assert actual == expected


def test_create_file_hash_path(tmp_path: Path):
    content = b"Hello World"
    file_path = tmp_path / "example.txt"
    file_path.write_bytes(content)

    actual = create_file_hash(file_path)
    expected = hashlib.sha256(content).hexdigest()

    assert actual == expected
def test_create_file_hash_multiple_chunks():
    content = b"A" * (65536 * 2 + 100)
    stream = BytesIO(content)
    stream.seek(10)

    actual = create_file_hash(stream)
    expected = hashlib.sha256(content).hexdigest()

    assert actual == expected
    assert stream.tell() == 10