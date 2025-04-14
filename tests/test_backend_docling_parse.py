from pathlib import Path

import pytest
from docling_core.types.doc import BoundingBox

from docling.backend.docling_parse_backend import (
    DoclingParseDocumentBackend,
    DoclingParsePageBackend,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


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


def test_get_text_from_rect(test_doc_path):
    doc_backend = _get_backend(test_doc_path)
    page_backend: DoclingParsePageBackend = doc_backend.load_page(0)

    # Get the title text of the DocLayNet paper
    textpiece = page_backend.get_text_in_rect(
        bbox=BoundingBox(l=102, t=77, r=511, b=124)
    )
    ref = "DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis"

    assert textpiece.strip() == ref


def test_crop_page_image(test_doc_path):
    doc_backend = _get_backend(test_doc_path)
    page_backend: DoclingParsePageBackend = doc_backend.load_page(0)

    # Crop out "Figure 1" from the DocLayNet paper
    page_backend.get_page_image(
        scale=2, cropbox=BoundingBox(l=317, t=246, r=574, b=527)
    )
    # im.show()


def test_num_pages(test_doc_path):
    doc_backend = _get_backend(test_doc_path)
    doc_backend.page_count() == 9
