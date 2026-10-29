# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest

from docling.utils.utils import chunkify


@pytest.mark.parametrize(
    "items",
    [
        [1, 2, 3, 4, 5],
        (1, 2, 3, 4, 5),
        range(1, 6),
        iter([1, 2, 3, 4, 5]),
        (i for i in range(1, 6)),
    ],
    ids=["list", "tuple", "range", "iterator", "generator"],
)
def test_chunkify_splits_any_iterable_once(items):
    assert list(chunkify(items, 2)) == [[1, 2], [3, 4], [5]]


def test_chunkify_empty():
    assert list(chunkify((), 3)) == []
