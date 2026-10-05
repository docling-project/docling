# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.mets_gbs_backend import MetsGbsDocumentBackend, MetsGbsPageBackend
from docling.datamodel.backend_options import MetsGbsBackendOptions
from docling.datamodel.base_models import BoundingBox, DocumentStream, InputFormat
from docling.datamodel.document import InputDocument, _DocumentConversionInput
from docling.document_converter import DocumentConverter


@pytest.fixture
def test_doc_path():
    return Path("tests/data/mets_gbs/sources/32044009881525_select.tar.gz")


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


def test_page_missing_coordocr_is_skipped_not_crashed():
    """A page whose METS XML has no 'coordOCR' fptr (e.g. no OCR layer for that page)
    must be reported as invalid, not crash the whole document with an AssertionError.
    """
    path = Path("tests/data/mets_gbs/sources/32044009881525_missing_coordocr.tar.gz")
    doc_backend: MetsGbsDocumentBackend = _get_backend(path)

    assert doc_backend.is_valid()
    assert doc_backend.page_count() == 3

    # Pages 0 and 1 have both image and coordOCR entries and load normally.
    for page_index in (0, 1):
        page_backend: MetsGbsPageBackend = doc_backend.load_page(page_index)
        assert page_backend.is_valid()
        page_backend.unload()

    # Page 2 is missing its coordOCR fptr in the METS XML; loading it must not raise.
    page_backend = doc_backend.load_page(2)
    assert not page_backend.is_valid()

    page_backend.unload()
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


def _repack_with_filler(
    source: Path,
    target: Path,
    head: list[str],
    filler_count: int,
) -> None:
    """Rewrite a METS-GBS archive with the `head` members first, then empty filler
    members, then the remaining original members."""
    with tarfile.open(source, mode="r:gz") as src:
        members = {m.name: (m, src.extractfile(m).read()) for m in src.getmembers()}

    with tarfile.open(target, mode="w:gz") as dst:
        for name in head:
            info, data = members[name]
            dst.addfile(info, BytesIO(data))
        for idx in range(filler_count):
            dst.addfile(tarfile.TarInfo(name=f"filler/{idx:05d}.bin"), BytesIO())
        for name, (info, data) in members.items():
            if name not in head:
                dst.addfile(info, BytesIO(data))


def test_member_limit_enforced_while_iterating(test_doc_path, tmp_path):
    """Page files beyond the member limit are not reached by reading all headers."""
    archive = tmp_path / "filler_before_pages.tar.gz"
    _repack_with_filler(
        test_doc_path, archive, ["HARVARD_32044009881525.xml"], filler_count=2000
    )
    options = MetsGbsBackendOptions(max_member_count=50)

    in_doc = InputDocument(
        path_or_stream=archive,
        format=InputFormat.METS_GBS,
        backend=MetsGbsDocumentBackend,
        backend_options=options,
    )
    assert in_doc.valid
    doc_backend: MetsGbsDocumentBackend = in_doc._backend

    with pytest.raises(ValueError, match="exceeds maximum member count limit"):
        doc_backend.load_page(0)
    assert len(doc_backend._tar.members) <= options.max_member_count + 1
    doc_backend.unload()


def test_members_after_needed_files_are_not_read(test_doc_path, tmp_path):
    """A page loads without reading headers past the files it needs."""
    archive = tmp_path / "filler_after_pages.tar.gz"
    head = [
        "HARVARD_32044009881525.xml",
        "00000010.html",
        "00000010.tif",
        "00000010.txt",
    ]
    _repack_with_filler(test_doc_path, archive, head, filler_count=2000)
    options = MetsGbsBackendOptions(max_member_count=50)

    in_doc = InputDocument(
        path_or_stream=archive,
        format=InputFormat.METS_GBS,
        backend=MetsGbsDocumentBackend,
        backend_options=options,
    )
    assert in_doc.valid
    doc_backend: MetsGbsDocumentBackend = in_doc._backend

    page_backend: MetsGbsPageBackend = doc_backend.load_page(0)
    assert page_backend.is_valid()
    assert len(doc_backend._tar.members) <= len(head)
    page_backend.unload()
    doc_backend.unload()


def test_format_detection_stops_at_member_limit(test_doc_path, tmp_path):
    """Detection gives up once the default member limit is exceeded."""
    archive = tmp_path / "filler_before_mets.tar.gz"
    limit = MetsGbsBackendOptions().max_member_count
    _repack_with_filler(test_doc_path, archive, [], filler_count=limit + 1)

    assert _DocumentConversionInput._detect_mets_gbs(archive) is None
    assert _DocumentConversionInput._detect_mets_gbs(test_doc_path) is not None


def _input_doc(source: Path | DocumentStream) -> InputDocument:
    format_options = DocumentConverter(
        allowed_formats=[InputFormat.METS_GBS]
    ).format_to_options
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[source])
    return next(iter(conv_input.docs(format_options)))


def test_stream_input_matches_path_input(test_doc_path):
    """An archive given as a stream is detected and loaded like the same path."""
    stream = DocumentStream(
        name=test_doc_path.name, stream=BytesIO(test_doc_path.read_bytes())
    )

    from_path = _input_doc(test_doc_path)
    from_stream = _input_doc(stream)

    assert from_stream.format == InputFormat.METS_GBS
    assert from_stream.valid
    assert from_stream.page_count == from_path.page_count
    assert from_stream.document_hash == from_path.document_hash
    from_path._backend.unload()
    from_stream._backend.unload()
