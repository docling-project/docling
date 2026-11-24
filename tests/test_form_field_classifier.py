# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The classifier keying of form fields, on small synthetic pages.

The trees here are written by hand: one tree that prefers the nearest text
before a value, outside it, and rejects text after it. They stand in for a
fitted model, so the checks exercise the candidates, the choice and the way
the stage uses the result, not the quality of a model.
"""

from pathlib import Path, PurePath
from unittest.mock import MagicMock

import numpy as np
import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfCellRenderingMode,
    PdfTextCell,
    PdfWidget,
    TextCell,
)

from docling.datamodel.base_models import (
    Cluster,
    InputFormat,
    LayoutPrediction,
    Page,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.models.stages.form_field.form_field_model import PdfFormFieldModel
from docling.models.stages.form_field.keying.classifier import FEATURE_NAMES
from docling.models.stages.form_field.keying.trees import TreeEnsemble
from docling.models.stages.form_field.keying.words import trim

HEIGHT = 100.0


def nearest_before() -> TreeEnsemble:
    """Text after the value or over it: no; else the nearer and shorter, the better."""
    f = FEATURE_NAMES.index
    return TreeEnsemble(
        names=tuple(FEATURE_NAMES),
        baseline=0.0,
        feature=np.array(
            [[f("b_right"), f("b_value_in"), 0, f("b_gap"), 0, f("t_words"), 0, 0, 0]]
        ),
        threshold=np.array([[0.0, 0.01, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0]]),
        left=np.array([[1, 3, 0, 5, 0, 7, 0, 0, 0]]),
        right=np.array([[2, 4, 0, 6, 0, 8, 0, 0, 0]]),
        missing_left=np.zeros((1, 9), dtype=bool),
        leaf=np.array([[False, False, True, False, True, False, True, True, True]]),
        value=np.array([[0.0, 0.0, -3.0, 0.0, -3.0, 0.0, 0.0, 2.0, 1.0]]),
    )


def box(left: float, top: float, right: float, bottom: float) -> BoundingBox:
    return BoundingBox(l=left, t=top, r=right, b=bottom)


def text_widget(index: int, bbox: BoundingBox) -> PdfWidget:
    return PdfWidget(
        index=index,
        rect=BoundingRectangle.from_bounding_box(
            bbox.to_bottom_left_origin(page_height=HEIGHT)
        ),
        widget_text="",
        widget_field_type="/Tx",
        widget_field_flags=0,
    )


def word(index: int, text: str, bbox: BoundingBox) -> PdfTextCell:
    return PdfTextCell(
        index=index,
        rect=BoundingRectangle.from_bounding_box(bbox),
        text=text,
        orig=text,
        rendering_mode=PdfCellRenderingMode.FILL_TEXT,
        widget=False,
        font_key="F1",
        font_name="Helvetica",
    )


def line_page(
    words: list[tuple[str, BoundingBox]], widgets: list[BoundingBox]
) -> tuple[Page, Cluster]:
    """One printed line, one text block holding it, and widgets over its blanks."""
    line = BoundingBox.enclosing_bbox([b for _, b in words])
    text = " ".join(t for t, _ in words)
    cell = TextCell(
        index=0,
        rect=BoundingRectangle.from_bounding_box(line),
        text=text,
        orig=text,
        from_ocr=False,
    )
    block = Cluster(id=1, label=DocItemLabel.TEXT, bbox=line, cells=[cell])
    page = Page(page_no=1, size=Size(width=200, height=HEIGHT))
    page.parsed_page = MagicMock(
        widgets=[text_widget(i, b) for i, b in enumerate(widgets)],
        word_cells=[word(i, t, b) for i, (t, b) in enumerate(words)],
        textline_cells=[cell],
    )
    page.predictions.layout = LayoutPrediction(clusters=[block])
    return page, block


def run_stage(page: Page, tmp_path: Path) -> None:
    path = tmp_path / "trees.npz"
    nearest_before().save(path)
    stage = PdfFormFieldModel(enabled=True, keying="classifier", classifier_path=path)
    input_doc = InputDocument.model_construct(
        file=PurePath("input.pdf"),
        document_hash="0" * 64,
        valid=True,
        format=InputFormat.PDF,
    )
    list(stage(ConversionResult(input=input_doc, pages=[page]), [page]))


def keys(page: Page) -> list[str]:
    fields = page.predictions.form_fields
    assert fields is not None
    items = [
        *fields.inline_items.values(),
        *(i for items in fields.form_items.values() for i in items),
        *fields.loose_items,
    ]
    return [item.key_text for item in items]


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("Address:______", "Address:"),
        ("______", None),
        ("Total........", "Total"),
        ("e.g.", "e.g."),  # Single dots are punctuation, not a fill run.
    ],
)
def test_fill_runs_leave_the_key_text(text: str, kept: str | None) -> None:
    trimmed = trim(text, (0.0, 0.0, 70.0, 10.0))
    assert (None if trimmed is None else trimmed[0]) == kept
    if trimmed is not None and kept != text:
        # The box shrinks to the kept characters.
        assert trimmed[1][2] < 70.0


def test_two_keys_of_one_line_key_their_own_values(tmp_path: Path) -> None:
    # The layout keeps the line whole; only the words split it in two keys.
    page, block = line_page(
        [
            ("Name", box(10, 10, 30, 20)),
            ("____", box(32, 10, 70, 20)),
            ("Date", box(72, 10, 90, 20)),
            ("of", box(92, 10, 100, 20)),
            ("birth", box(102, 10, 122, 20)),
            ("____", box(124, 10, 170, 20)),
        ],
        [box(33, 10, 69, 20), box(125, 10, 169, 20)],
    )
    run_stage(page, tmp_path)
    assert sorted(keys(page)) == ["Date of birth", "Name"]
    # Every word of the line is in a key: the line leaves the body once.
    assert page.predictions.layout is not None
    assert block not in page.predictions.layout.clusters


def test_a_line_partly_used_as_a_key_stays_in_the_body(tmp_path: Path) -> None:
    page, block = line_page(
        [
            ("Name", box(10, 10, 30, 20)),
            ("____", box(32, 10, 70, 20)),
            ("(see", box(72, 10, 90, 20)),
            ("instructions)", box(92, 10, 140, 20)),
        ],
        [box(33, 10, 69, 20)],
    )
    run_stage(page, tmp_path)
    assert keys(page) == ["Name"]
    # The rest of the line is not key text, so no part of it is removed.
    assert page.predictions.layout is not None
    assert block in page.predictions.layout.clusters
    assert [c.text for c in block.cells] == ["Name ____ (see instructions)"]


def test_trees_match_scikit_learn_with_missing_features(tmp_path: Path) -> None:
    ensemble = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 5)).astype(np.float32)
    x[rng.random(x.shape) < 0.2] = np.nan
    y = (np.nan_to_num(x[:, 0]) - np.nan_to_num(x[:, 1], nan=1.5) > 0).astype(int)
    model = ensemble.HistGradientBoostingClassifier(max_iter=40, random_state=0)
    model.fit(x, y)
    path = tmp_path / "trees.npz"
    TreeEnsemble.from_sklearn(model, [f"f{i}" for i in range(5)]).save(path)
    test = rng.normal(size=(2000, 5))
    test[rng.random(test.shape) < 0.3] = np.nan
    expected = model.predict_proba(test.astype(np.float32))[:, 1]
    np.testing.assert_allclose(TreeEnsemble.load(path).predict(test), expected)
