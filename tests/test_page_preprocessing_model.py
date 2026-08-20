from types import SimpleNamespace
from typing import cast

import pytest
from docling_core.types.doc import Size

from docling.backend.pdf_backend import PdfPageBackend
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.models.stages.page_preprocessing.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
)


class _InvalidTextBackend:
    def get_segmented_page(self):
        raise UnicodeDecodeError("utf-8", b"\xdf", 0, 1, "invalid start byte")


def _page() -> Page:
    page = Page(page_no=1, size=Size(width=595, height=842))
    page._backend = cast(PdfPageBackend, _InvalidTextBackend())
    return page


def _conversion_result() -> ConversionResult:
    return cast(
        ConversionResult,
        SimpleNamespace(
            confidence=SimpleNamespace(
                pages={1: SimpleNamespace(parse_score=None)},
            )
        ),
    )


def test_native_text_decode_error_falls_back_to_empty_page() -> None:
    model = PagePreprocessingModel(
        PagePreprocessingOptions(
            images_scale=None,
            allow_empty_cells_on_decode_error=True,
        )
    )

    page = model._parse_page_cells(_conversion_result(), _page())

    assert page.parsed_page is not None
    assert page.parsed_page.dimension.width == 595
    assert page.parsed_page.dimension.height == 842
    assert page.cells == []


def test_native_text_decode_error_is_not_hidden_by_default() -> None:
    model = PagePreprocessingModel(PagePreprocessingOptions(images_scale=None))

    with pytest.raises(UnicodeDecodeError):
        model._parse_page_cells(_conversion_result(), _page())
