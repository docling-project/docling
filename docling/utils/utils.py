import hashlib
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import List, Union

import requests
from tqdm import tqdm


def chunkify(iterator, chunk_size):
    """Yield successive chunks of chunk_size from the iterable."""
    if isinstance(iterator, List):
        iterator = iter(iterator)
    for first in iterator:  # Take the first element from the iterator
        yield [first] + list(islice(iterator, chunk_size - 1))


def create_file_hash(path_or_stream: Union[BytesIO, Path]) -> str:
    """Create a stable page_hash of the path_or_stream of a file"""

    block_size = 65536
    hasher = hashlib.sha256()

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
    hasher = hashlib.sha256()
    hasher.update(string.encode("utf-8"))

    return hasher.hexdigest()


def download_url_with_progress(url: str, progress: bool = False) -> BytesIO:
    buf = BytesIO()
    with requests.get(url, stream=True, allow_redirects=True) as response:
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
