from pathlib import Path

import pytest

from docling.backend.mets_gbs_backend import MetsGbsDocumentBackend, MetsGbsPageBackend
from docling.datamodel.backend_options import MetsGbsBackendOptions
from docling.datamodel.base_models import BoundingBox, InputFormat
from docling.datamodel.document import InputDocument


@pytest.fixture
def test_doc_path():
    return Path("tests/data/mets_gbs/32044009881525_select.tar.gz")


def _get_backend(pdf_doc):
    in_doc = InputDocument(
        path_or_stream=pdf_doc,
        format=InputFormat.METS_GBS,
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


def test_max_file_bytes_limit(test_doc_path):
    """Test that max_file_bytes limit is enforced during extraction."""

    options = MetsGbsBackendOptions(max_file_bytes=100)

    with pytest.raises(ValueError, match=r"exceeds.*size limit"):
        InputDocument(
            path_or_stream=test_doc_path,
            format=InputFormat.METS_GBS,
            backend=MetsGbsDocumentBackend,
            backend_options=options,
        )


def test_max_total_bytes_limit(test_doc_path):
    """Test that max_total_bytes limit is enforced across all extractions."""

    options = MetsGbsBackendOptions(
        max_file_bytes=10 * 1024 * 1024,
        max_total_bytes=1000,
    )

    with pytest.raises(ValueError, match="exceeds maximum total extraction size"):
        InputDocument(
            path_or_stream=test_doc_path,
            format=InputFormat.METS_GBS,
            backend=MetsGbsDocumentBackend,
            backend_options=options,
        )


def test_max_member_count_limit(test_doc_path):
    """Test that max_member_count limit is enforced during extraction."""

    options = MetsGbsBackendOptions(max_member_count=2)

    with pytest.raises(ValueError, match="exceeds maximum member count limit"):
        InputDocument(
            path_or_stream=test_doc_path,
            format=InputFormat.METS_GBS,
            backend=MetsGbsDocumentBackend,
            backend_options=options,
        )


def test_limits_with_valid_values(test_doc_path):
    """Test that processing succeeds with generous limits."""
    options = MetsGbsBackendOptions(
        max_file_bytes=10 * 1024 * 1024,  # 10 MB
        max_total_bytes=300 * 1024 * 1024,  # 300 MB
        max_member_count=1000,
    )

    in_doc = InputDocument(
        path_or_stream=test_doc_path,
        format=InputFormat.METS_GBS,
        backend=MetsGbsDocumentBackend,
        backend_options=options,
    )

    assert in_doc.valid
    doc_backend: MetsGbsDocumentBackend = in_doc._backend
    assert doc_backend.is_valid()
    assert doc_backend.page_count() == 3

    page_backend: MetsGbsPageBackend = doc_backend.load_page(0)
    assert page_backend.is_valid()

    page_backend.unload()
    doc_backend.unload()


def test_total_bytes_tracking_across_pages(test_doc_path):
    """Test that total bytes are tracked cumulatively across initialization and page loading.

    This test ensures that when max_total_bytes is larger than max_file_bytes,
    initialization succeeds but page loading eventually fails due to cumulative limit.
    """
    options = MetsGbsBackendOptions(
        max_file_bytes=10 * 1024 * 1024,
        max_total_bytes=20 * 1024,
        max_member_count=1000,
    )

    in_doc = InputDocument(
        path_or_stream=test_doc_path,
        format=InputFormat.METS_GBS,
        backend=MetsGbsDocumentBackend,
        backend_options=options,
    )

    assert in_doc.valid
    doc_backend: MetsGbsDocumentBackend = in_doc._backend
    assert doc_backend.is_valid()

    page_load_failed = False
    for page_index in range(doc_backend.page_count()):
        try:
            page_backend: MetsGbsPageBackend = doc_backend.load_page(page_index)
            page_backend.unload()
        except ValueError as e:
            assert "Total extracted data exceeds maximum limit" in str(e)
            page_load_failed = True
            break

    assert page_load_failed, "Expected page loading to fail due to total bytes limit"
    doc_backend.unload()
