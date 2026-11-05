# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import random
from itertools import combinations_with_replacement

from docling_core.types.doc import BoundingBox

from docling.utils.form_fields import reading_order, text_slots


def _box(left: float, top: float, right: float, bottom: float) -> BoundingBox:
    return BoundingBox(l=left, t=top, r=right, b=bottom)


def test_reading_order_reads_lines_left_to_right_and_columns_top_down() -> None:
    name, date = _box(10, 10, 40, 20), _box(50, 10, 90, 20)
    below = _box(10, 30, 40, 40)
    other_column = _box(60, 30, 90, 40)
    assert reading_order(name, date) == -1
    assert reading_order(date, name) == 1
    assert reading_order(name, below) == -1
    assert reading_order(name, other_column) == 0


def test_text_between_two_fields_goes_between_them() -> None:
    # Name: [___] / Please write in capitals. / Date: [___]
    name, note, date = _box(10, 10, 90, 20), _box(10, 30, 90, 40), _box(10, 50, 90, 60)
    assert text_slots([note], [name, date]) == [1]


def test_text_slots_match_exhaustive_search() -> None:
    rng = random.Random(4496)

    def box() -> BoundingBox:
        left, top = rng.uniform(0, 80), rng.uniform(0, 80)
        return _box(left, top, left + rng.uniform(2, 40), top + rng.uniform(2, 15))

    for _ in range(400):
        texts = [box() for _ in range(rng.randint(0, 5))]
        fields = [box() for _ in range(rng.randint(0, 5))]

        def cost(slots: tuple[int, ...]) -> int:
            return sum(
                (order < 0 and k < slot) or (order > 0 and k >= slot)
                for text, slot in zip(texts, slots)
                for k, field in enumerate(fields)
                for order in [reading_order(text, field)]
            )

        options = list(
            combinations_with_replacement(range(len(fields) + 1), len(texts))
        )
        least = min(cost(o) for o in options)
        # Ties: the latest slots, compared from the last text backwards.
        expected = max((o for o in options if cost(o) == least), key=lambda o: o[::-1])
        assert tuple(text_slots(texts, fields)) == expected
