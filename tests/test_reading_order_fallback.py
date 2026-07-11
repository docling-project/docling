"""Unit tests for the torch-free reading-order fallback (used when docling-ibm-models is absent)."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pytest
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.base import CoordOrigin, Size
from docling_core.types.doc.document import RefItem

_MODULE = "docling.models.stages.reading_order.readingorder_model"


@pytest.fixture
def fallback_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Reimport the reading-order module with docling-ibm-models unavailable.

    Setting the package (and any loaded submodules) to ``None`` in ``sys.modules`` makes
    ``import docling_ibm_models...`` raise ``ImportError``, so the module binds its torch-free
    fallbacks (geometric reading order + no-op list processing). The original module object is
    restored on teardown so other tests keep the real implementation.
    """
    original = sys.modules.get(_MODULE)
    blocked = [
        name
        for name in list(sys.modules)
        if name == "docling_ibm_models" or name.startswith("docling_ibm_models.")
    ]
    for name in blocked:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "docling_ibm_models", None)
    monkeypatch.delitem(sys.modules, _MODULE, raising=False)

    module = importlib.import_module(_MODULE)
    yield module

    if original is not None:
        sys.modules[_MODULE] = original
    else:
        sys.modules.pop(_MODULE, None)


def _element(
    module: ModuleType,
    *,
    cid: int,
    left: float,
    right: float,
    bottom: float,
    top: float,
    page_no: int = 0,
) -> Any:
    return module.ReadingOrderPageElement(
        cid=cid,
        ref=RefItem.model_validate({"$ref": f"#/{cid}"}),
        text="",
        page_no=page_no,
        page_size=Size(width=100.0, height=100.0),
        label=DocItemLabel.TEXT,
        l=left,
        r=right,
        b=bottom,
        t=top,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )


def test_fallback_orders_top_to_bottom(fallback_module: ModuleType) -> None:
    """Horizontally overlapping elements sort top-first (higher bottom-left ``b``)."""
    upper = _element(fallback_module, cid=0, left=0, right=50, bottom=80, top=90)
    lower = _element(fallback_module, cid=1, left=0, right=50, bottom=10, top=20)
    ordered = fallback_module.ReadingOrderPredictor().predict_reading_order(
        [lower, upper]
    )
    assert [element.cid for element in ordered] == [0, 1]


def test_fallback_orders_left_to_right(fallback_module: ModuleType) -> None:
    """Horizontally disjoint elements sort left-first."""
    left = _element(fallback_module, cid=0, left=0, right=40, bottom=10, top=20)
    right = _element(fallback_module, cid=1, left=60, right=100, bottom=10, top=20)
    ordered = fallback_module.ReadingOrderPredictor().predict_reading_order(
        [right, left]
    )
    assert [element.cid for element in ordered] == [0, 1]


def test_fallback_association_maps_are_empty(fallback_module: ModuleType) -> None:
    """Caption/footnote/merge association is skipped in the torch-free fallback."""
    predictor = fallback_module.ReadingOrderPredictor()
    elements = [_element(fallback_module, cid=0, left=0, right=10, bottom=0, top=10)]
    assert predictor.predict_to_captions(elements) == {}
    assert predictor.predict_to_footnotes(elements) == {}
    assert predictor.predict_merges(elements) == {}


def test_fallback_list_item_processor_is_noop(fallback_module: ModuleType) -> None:
    sentinel = object()
    assert (
        fallback_module.ListItemMarkerProcessor().process_list_item(sentinel)
        is sentinel
    )


def test_fallback_is_active_without_docling_ibm_models(
    fallback_module: ModuleType,
) -> None:
    """The fixture blocks docling-ibm-models, so the bound classes are the local stand-ins."""
    assert sys.modules["docling_ibm_models"] is None
    assert type(fallback_module.ReadingOrderPredictor()).__module__ == _MODULE
    assert fallback_module.ReadingOrderPageElement.__module__ == _MODULE
