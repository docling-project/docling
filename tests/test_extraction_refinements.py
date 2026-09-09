# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""R2 (page range on the text channel) and R4 (optional override backend)."""

from types import SimpleNamespace

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.settings import DEFAULT_PAGE_RANGE, DocumentLimits
from docling.document_extractor import (
    DocumentExtractor,
    ExtractionFormatOption,
    _get_default_extraction_option,
)
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline


class _Doc:
    def __init__(self, page_nos):
        self.pages = {p: object() for p in page_nos}

    def export_to_markdown(self, page_no=None):
        return "<all>" if page_no is None else f"<p{page_no}>"


class _DeclBackend(DeclarativeDocumentBackend):
    def __init__(self, doc):
        self._doc = doc

    def convert(self):
        return self._doc

    def is_valid(self):
        return True

    @classmethod
    def supports_pagination(cls):
        return True

    def unload(self):
        pass

    @classmethod
    def supported_formats(cls):
        return set()


def _text_pipeline(doc):
    pipeline = ExtractionVlmPipeline.__new__(ExtractionVlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(markdown_params=None)
    return pipeline


def _get_text(doc, page_range):
    pipeline = _text_pipeline(doc)
    input_doc = SimpleNamespace(
        _backend=_DeclBackend(doc),
        limits=DocumentLimits(page_range=page_range),
    )
    return pipeline._get_text_from_input(input_doc)


def test_text_channel_default_range_serializes_whole_document():
    assert _get_text(_Doc([1, 2, 3]), DEFAULT_PAGE_RANGE) == "<all>"


def test_text_channel_restricts_to_page_range():
    # Only pages inside [2, 3] are serialized, in order.
    assert _get_text(_Doc([1, 2, 3, 4]), (2, 3)) == "<p2>\n\n<p3>"


def test_text_channel_range_ignored_for_markdown_passthrough():
    pipeline = _text_pipeline(None)
    backend = MarkdownDocumentBackend.__new__(MarkdownDocumentBackend)
    backend.markdown = "raw markdown"
    input_doc = SimpleNamespace(
        _backend=backend, limits=DocumentLimits(page_range=(2, 3))
    )
    assert pipeline._get_text_from_input(input_doc) == "raw markdown"


def test_override_without_backend_inherits_default():
    # R4: overriding only pipeline_options must not force restating the backend.
    opts = ExtractionFormatOption(pipeline_cls=ExtractionVlmPipeline)
    assert opts.backend is None

    extractor = DocumentExtractor(
        allowed_formats=[InputFormat.DOCX],
        extraction_format_options={InputFormat.DOCX: opts},
    )
    resolved = extractor.extraction_format_to_options[InputFormat.DOCX]
    assert resolved.backend is _get_default_extraction_option(InputFormat.DOCX).backend
