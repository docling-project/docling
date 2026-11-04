# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import hashlib
from collections.abc import Callable, Sequence
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import TypeVar, Union

import requests
from tqdm import tqdm

_RunT = TypeVar("_RunT")


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
        _hash_buf(path_or_stream)

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


def trim_run_edges(
    runs: Sequence[_RunT],
    *,
    get_text: Callable[[_RunT], str],
    with_text: Callable[[_RunT, str], _RunT],
    is_trimmable: Callable[[_RunT], bool] = lambda run: True,
) -> list[_RunT]:
    """Trim the outer edges of a block of inline runs and keep interior whitespace.

    Leading and trailing runs that hold only whitespace are dropped. The first and the
    last remaining run lose their outer whitespace. Whitespace between runs is never
    changed. Trimmable runs with no text are dropped.

    Args:
        runs: The runs of one block, in document order. The sequence is not changed.
        get_text: Return the text of a run.
        with_text: Return a copy of a run with new text.
        is_trimmable: Return False for a run that must stay as it is, even when it has
            no text. Trimming stops at such a run on either edge.

    Returns:
        A new list with the trimmed runs.
    """
    trimmed = list(runs)

    start = 0
    while start < len(trimmed) and is_trimmable(trimmed[start]):
        head = with_text(trimmed[start], get_text(trimmed[start]).lstrip())
        if get_text(head):
            trimmed[start] = head
            break
        start += 1

    end = len(trimmed)
    while end > start and is_trimmable(trimmed[end - 1]):
        tail = with_text(trimmed[end - 1], get_text(trimmed[end - 1]).rstrip())
        if get_text(tail):
            trimmed[end - 1] = tail
            break
        end -= 1

    return [run for run in trimmed[start:end] if get_text(run) or not is_trimmable(run)]
