# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import hashlib
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Union

import requests
from tqdm import tqdm


def chunkify(iterator, chunk_size):
    """Yield successive chunks of chunk_size from the iterable."""
    # Always take one iterator: re-iterating a tuple or range in islice()
    # would restart it from the beginning for every chunk.
    iterator = iter(iterator)
    for first in iterator:  # Take the first element from the iterator
        yield [first, *list(islice(iterator, chunk_size - 1))]


def create_file_hash(path_or_stream: Union[BytesIO, Path]) -> str:
    """Create a stable page_hash of the path_or_stream of a file"""

    block_size = 65536
    hasher = hashlib.sha256(usedforsecurity=False)

    def _hash_buf(binary_stream):
        buf = binary_stream.read(block_size)  # read and page_hash in chunks
        while len(buf) > 0:
            hasher.update(buf)
            buf = binary_stream.read(block_size)

    if isinstance(path_or_stream, Path):
        with path_or_stream.open("rb") as afile:
            _hash_buf(afile)
    elif isinstance(path_or_stream, BytesIO):
        # The stream can still be sitting where the caller left it, which is the end of the
        # buffer for anyone who just wrote the document into it. Reading from there hashes
        # b"" and every such document ends up with the same hash. Hash the whole buffer
        # instead, and hand the stream back at the position we found it.
        position = path_or_stream.tell()
        path_or_stream.seek(0)
        try:
            _hash_buf(path_or_stream)
        finally:
            path_or_stream.seek(position)

    return hasher.hexdigest()


def create_hash(string: str):
    hasher = hashlib.sha256(usedforsecurity=False)
    hasher.update(string.encode("utf-8"))

    return hasher.hexdigest()


def download_url_with_progress(url: str, progress: bool = False) -> BytesIO:
    buf = BytesIO()
    with requests.get(
        url, stream=True, allow_redirects=True, timeout=(5, 30)
    ) as response:
        total_size = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=(not progress),
        )

        for chunk in response.iter_content(10 * 1024):
            buf.write(chunk)
            progress_bar.update(len(chunk))
        progress_bar.close()

    buf.seek(0)
    return buf


def safe_version(name: str) -> str:
    """Return the installed version of `name`, or `"unknown"` if the
    distribution is not present.

    Slim installs may not have every distribution available (e.g. the
    `docling` meta-package is absent when only `docling-slim` is
    installed; `docling-ibm-models` and `docling-parse` are gated behind
    optional extras).
    """
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
