from pathlib import Path
from typing import Any

import pytest

from docling.backend.docling_parse_backend import (
    DoclingParseDocumentBackend,
    DoclingParsePageBackend,
    ThreadedDoclingParseDocumentBackend,
    ThreadedDoclingParsePageBackend,
)
from docling.datamodel.base_models import BoundingBox, InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.settings import DocumentLimits


@pytest.fixture
def test_doc_path():
    return Path("./tests/data/pdf/2206.01062.pdf")


def _get_backend(pdf_doc):
    in_doc = InputDocument(
        path_or_stream=pdf_doc,
        format=InputFormat.PDF,
        backend=DoclingParseDocumentBackend,
    )

    doc_backend = in_doc._backend
    return doc_backend


def test_text_cell_counts():
    pdf_doc = Path("./tests/data/pdf/redp5110_sampled.pdf")

    doc_backend = _get_backend(pdf_doc)

    for page_index in range(doc_backend.page_count()):
        last_cell_count = None
        for i in range(10):
            page_backend: DoclingParsePageBackend = doc_backend.load_page(0)
            cells = list(page_backend.get_text_cells())

            if last_cell_count is None:
                last_cell_count = len(cells)

            if len(cells) != last_cell_count:
                assert False, (
                    "Loading page multiple times yielded non-identical text cell counts"
                )
            last_cell_count = len(cells)

            # Clean up page backend after each iteration
            page_backend.unload()

    # Explicitly clean up document backend to prevent race conditions in CI
    doc_backend.unload()


def test_get_text_from_rect(test_doc_path):
    doc_backend = _get_backend(test_doc_path)
    page_backend: DoclingParsePageBackend = doc_backend.load_page(0)

    # Get the title text of the DocLayNet paper
    textpiece = page_backend.get_text_in_rect(
        bbox=BoundingBox(l=102, t=77, r=511, b=124)
    )
    ref = "DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis"

    assert textpiece.strip() == ref

    # Explicitly clean up resources
    page_backend.unload()
    doc_backend.unload()


def test_crop_page_image(test_doc_path):
    doc_backend = _get_backend(test_doc_path)
    page_backend: DoclingParsePageBackend = doc_backend.load_page(0)

    # Crop out "Figure 1" from the DocLayNet paper
    page_backend.get_page_image(
        scale=2, cropbox=BoundingBox(l=317, t=246, r=574, b=527)
    )
    # im.show()

    # Explicitly clean up resources
    page_backend.unload()
    doc_backend.unload()


def test_num_pages(test_doc_path):
    doc_backend = _get_backend(test_doc_path)
    assert doc_backend.page_count() == 9

    # Explicitly clean up resources to prevent race conditions in CI
    doc_backend.unload()


def test_iter_pages_default_contract(test_doc_path):
    doc_backend = _get_backend(test_doc_path)

    page_numbers = []
    page_backends = []
    try:
        for index, page_backend in enumerate(doc_backend.iter_pages()):
            page_numbers.append(page_backend.page_no)
            page_backends.append(page_backend)
            if index == 2:
                break
    finally:
        for page_backend in page_backends:
            page_backend.unload()
        doc_backend.unload()

    assert page_numbers == [1, 2, 3]


class _FakeThreadedResult:
    def __init__(
        self,
        *,
        page_number: int,
        success: bool = True,
        page_width: float = 100.0,
        page_height: float = 200.0,
    ) -> None:
        self.page_number = page_number
        self.success = success
        self.page_width = page_width
        self.page_height = page_height
        self.cropboxes: list[BoundingBox | None] = []
        self.scales: list[float] = []

    def get_page(self) -> Any:
        raise AssertionError("get_page() is not expected in this test")

    def get_image(
        self,
        *,
        scale: float | None = None,
        canvas_size=None,
        cropbox: BoundingBox | None = None,
    ):
        from PIL import Image

        assert canvas_size is None
        self.scales.append(1.0 if scale is None else scale)
        self.cropboxes.append(cropbox)
        width = round(self.page_width if cropbox is None else cropbox.width)
        height = round(self.page_height if cropbox is None else cropbox.height)
        scaled_width = max(1, round(width * (1.0 if scale is None else scale)))
        scaled_height = max(1, round(height * (1.0 if scale is None else scale)))
        return Image.new("RGBA", (scaled_width, scaled_height), (255, 255, 255, 255))


class _FakeThreadedParser:
    created: "_FakeThreadedParser | None" = None

    def __init__(self, parser_config=None, decode_config=None) -> None:
        self.parser_config = parser_config
        self.decode_config = decode_config
        self.load_calls: list[list[int] | None] = []
        self.unload_calls: list[str] = []
        _FakeThreadedParser.created = self

    def load(self, path_or_stream, password=None, page_numbers=None) -> str:
        self.load_calls.append(page_numbers)
        return "doc-key"

    def page_count(self, doc_key: str) -> int:
        assert doc_key == "doc-key"
        return 5

    def iterate_results(self):
        yield _FakeThreadedResult(page_number=3)
        yield _FakeThreadedResult(page_number=2)

    def unload(self, doc_key: str) -> bool:
        self.unload_calls.append(doc_key)
        return True


def test_threaded_backend_iterates_requested_pages_and_unloads(
    test_doc_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "docling.backend.docling_parse_backend.DoclingThreadedPdfParser",
        _FakeThreadedParser,
    )

    in_doc = InputDocument(
        path_or_stream=test_doc_path,
        format=InputFormat.PDF,
        backend=ThreadedDoclingParseDocumentBackend,
        limits=DocumentLimits(page_range=(2, 3)),
    )

    doc_backend = in_doc._backend
    assert isinstance(doc_backend, ThreadedDoclingParseDocumentBackend)
    assert doc_backend.page_count() == 5

    page_numbers = [page_backend.page_no for page_backend in doc_backend.iter_pages()]
    assert page_numbers == [3, 2]

    parser = _FakeThreadedParser.created
    assert parser is not None
    assert parser.load_calls == [[2, 3]]

    doc_backend.unload()
    assert parser.unload_calls == ["doc-key"]


def test_threaded_backend_no_page_range_passes_none(
    test_doc_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "docling.backend.docling_parse_backend.DoclingThreadedPdfParser",
        _FakeThreadedParser,
    )

    in_doc = InputDocument(
        path_or_stream=test_doc_path,
        format=InputFormat.PDF,
        backend=ThreadedDoclingParseDocumentBackend,
        # no limits → default page_range (1, sys.maxsize)
    )

    parser = _FakeThreadedParser.created
    assert parser is not None
    assert parser.load_calls == [None]

    in_doc._backend.unload()


def test_threaded_page_backend_delegates_image_access() -> None:
    result = _FakeThreadedResult(page_number=4, page_width=120.0, page_height=90.0)
    page_backend = ThreadedDoclingParsePageBackend(result)
    cropbox = BoundingBox(l=10, t=5, r=40, b=25)

    image = page_backend.get_page_image(scale=2.0, cropbox=cropbox)

    assert page_backend.page_no == 4
    assert page_backend.get_size().width == 120.0
    assert page_backend.get_size().height == 90.0
    assert page_backend.is_valid() is True
    assert image.size == (60, 40)
    assert result.scales == [2.0]
    assert result.cropboxes == [cropbox]
