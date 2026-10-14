# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Type

import pytest

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.backend.pypdfium2_backend import (
    PyPdfiumDocumentBackend,
)
from docling.datamodel.backend_options import PdfBackendOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.exceptions import ConversionError


@pytest.fixture
def test_doc_path():
    return Path("./tests/data/pdf_password/sources/2206.01062_pg3.pdf")


@dataclass
class TestOption:
    options: PdfFormatOption
    name: str


def converter_opts_gen() -> Iterable[TestOption]:
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
    )

    backend_options = PdfBackendOptions(password="1234")

    yield TestOption(
        options=PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=PyPdfiumDocumentBackend,
            backend_options=backend_options,
        ),
        name="PyPdfium",
    )

    yield TestOption(
        options=PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=ThreadedDoclingParseDocumentBackend,
            backend_options=backend_options,
        ),
        name="DoclingParse",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("test_options", converter_opts_gen(), ids=lambda o: o.name)
def test_get_text_from_rect(test_doc_path: Path, test_options: TestOption):
    converter = DocumentConverter(
        format_options={InputFormat.PDF: test_options.options}
    )

    res = converter.convert(test_doc_path)
    assert res.status == ConversionStatus.SUCCESS


def _converter_with_password(
    backend_cls: Type[AbstractDocumentBackend], password: Optional[str]
) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=backend_cls,
                backend_options=PdfBackendOptions(password=password),
            )
        }
    )


def _walk_causes(exc: Optional[BaseException]) -> Iterator[BaseException]:
    while exc is not None:
        yield exc
        exc = exc.__cause__


@pytest.mark.parametrize(
    "backend_cls,name,expected_fragment",
    [
        (PyPdfiumDocumentBackend, "PyPdfium", "Incorrect password"),
        (
            ThreadedDoclingParseDocumentBackend,
            "DoclingParse",
            "password-protected",
        ),
    ],
)
@pytest.mark.parametrize(
    "password", [None, "wrong-password"], ids=["no-password", "wrong-password"]
)
def test_encrypted_pdf_error_identifies_encryption(
    test_doc_path: Path,
    backend_cls: Type[AbstractDocumentBackend],
    name: str,
    expected_fragment: str,
    password: Optional[str],
):
    """An encrypted PDF must fail with an error that names password protection.

    Regression test for https://github.com/docling-project/docling/issues/4829:
    the default docling-parse backend used to surface an encrypted PDF with the
    same generic load failure as a corrupt file, so callers could not tell
    "needs a password" apart from "broken file".
    """
    converter = _converter_with_password(backend_cls, password)

    with pytest.raises(ConversionError) as exc_info:
        converter.convert(test_doc_path)

    chain_text = "\n".join(
        f"{type(exc).__name__}: {exc}" for exc in _walk_causes(exc_info.value)
    )
    assert expected_fragment in chain_text
