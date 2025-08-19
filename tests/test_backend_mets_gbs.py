from pathlib import Path

import pytest

from docling.backend.mets_gbs_backend import MetsGbsDocumentBackend, MetsGbsPageBackend
from docling.datamodel.base_models import BoundingBox, InputFormat
from docling.datamodel.document import InputDocument


@pytest.fixture
def test_doc_path():
    return Path("tests/data/mets_gbs/32044009881525_select.tar.gz")


def _get_backend(pdf_doc):
    in_doc = InputDocument(
        path_or_stream=pdf_doc,
        format=InputFormat.PDF,
        backend=MetsGbsDocumentBackend,
    )

    doc_backend = in_doc._backend
    return doc_backend


def test_process_pages(test_doc_path):
    doc_backend: MetsGbsDocumentBackend = _get_backend(test_doc_path)

    for page_index in range(doc_backend.page_count()):
        page_backend: MetsGbsPageBackend = doc_backend.load_page(page_index)
        list(page_backend.get_text_cells())

        # Clean up page backend after each iteration
        page_backend.unload()

    # Explicitly clean up document backend to prevent race conditions in CI
    doc_backend.unload()


def test_get_text_from_rect(test_doc_path):
    doc_backend: MetsGbsDocumentBackend = _get_backend(test_doc_path)
    page_backend: MetsGbsPageBackend = doc_backend.load_page(0)

    # Get the title text of the DocLayNet paper
    textpiece = page_backend.get_text_in_rect(
        bbox=BoundingBox(l=275, t=263, r=1388, b=311)
    )
    ref = "recently become prevalent that he who speaks"

    assert textpiece.strip() == ref

    # Explicitly clean up resources
    page_backend.unload()
    doc_backend.unload()


def test_crop_page_image(test_doc_path):
    doc_backend: MetsGbsDocumentBackend = _get_backend(test_doc_path)
    page_backend: MetsGbsPageBackend = doc_backend.load_page(0)

    page_backend.get_page_image(
        scale=2, cropbox=BoundingBox(l=270, t=587, r=1385, b=1995)
    )
    # im.show()

    # Explicitly clean up resources
    page_backend.unload()
    doc_backend.unload()


def test_num_pages(test_doc_path):
    doc_backend: MetsGbsDocumentBackend = _get_backend(test_doc_path)
    assert doc_backend.is_valid()
    assert doc_backend.page_count() == 3

    # Explicitly clean up resources to prevent race conditions in CI
    doc_backend.unload()
