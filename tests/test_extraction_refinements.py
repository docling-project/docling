# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import VlmExtractionPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.datamodel.settings import DEFAULT_PAGE_RANGE, DocumentLimits
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.document_extractor import (
    DocumentExtractor,
    ExtractionFormatOption,
    _get_default_extraction_option,
)
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline


class _Doc:
    def __init__(self, page_nos: list[int]) -> None:
        self.pages = {p: object() for p in page_nos}

    def export_to_markdown(self, page_no: int | None = None) -> str:
        return "<all>" if page_no is None else f"<p{page_no}>"


class _DeclBackend(DeclarativeDocumentBackend):
    def __init__(self, doc: _Doc) -> None:
        self._doc = doc

    def convert(self) -> Any:
        return self._doc

    def is_valid(self) -> bool:
        return True

    @classmethod
    def supports_pagination(cls) -> bool:
        return True

    def unload(self) -> None:
        pass

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return set()


def _text_pipeline() -> ExtractionVlmPipeline:
    pipeline = ExtractionVlmPipeline.__new__(ExtractionVlmPipeline)
    pipeline.pipeline_options = cast(
        VlmExtractionPipelineOptions, SimpleNamespace(markdown_params=None)
    )
    return pipeline


def _get_text(doc: _Doc, page_range: tuple[int, int]) -> str:
    pipeline = _text_pipeline()
    input_doc = SimpleNamespace(
        _backend=_DeclBackend(doc),
        format=InputFormat.DCLX,
        limits=DocumentLimits(page_range=page_range),
    )
    return pipeline._get_text_from_input(cast(InputDocument, input_doc))


def test_text_channel_default_range_serializes_whole_document() -> None:
    assert _get_text(_Doc([1, 2, 3]), DEFAULT_PAGE_RANGE) == "<all>"


def test_text_channel_restricts_to_page_range() -> None:
    assert _get_text(_Doc([1, 2, 3, 4]), (2, 3)) == "<p2>\n\n<p3>"


def test_override_without_backend_inherits_default() -> None:
    opts = ExtractionFormatOption(pipeline_cls=ExtractionVlmPipeline)
    extractor = DocumentExtractor(
        allowed_formats=[InputFormat.DOCX],
        extraction_format_options={InputFormat.DOCX: opts},
    )
    resolved = extractor.extraction_format_to_options[InputFormat.DOCX]
    assert resolved.backend is _get_default_extraction_option(InputFormat.DOCX).backend


def test_default_extractor_enables_only_supported_formats() -> None:
    extractor = DocumentExtractor()

    assert set(extractor.allowed_formats) == {
        InputFormat.IMAGE,
        InputFormat.PDF,
        InputFormat.DOCX,
        InputFormat.HTML,
        InputFormat.MD,
        InputFormat.DCLX,
    }


@pytest.mark.parametrize("serialized", [False, True])
def test_legacy_inline_options_are_adapted(serialized: bool) -> None:
    legacy = InlineVlmOptions(
        repo_id="numind/NuExtract-2.0-2B",
        prompt="",
        inference_framework=InferenceFramework.TRANSFORMERS,
        response_format=ResponseFormat.PLAINTEXT,
        quantized=True,
        torch_dtype="float16",
        trust_remote_code=True,
        use_kv_cache=False,
    )
    value: Any = legacy.model_dump(mode="json") if serialized else legacy

    with pytest.warns(DeprecationWarning):
        options = VlmExtractionPipelineOptions(vlm_options=value)

    engine = options.vlm_options.engine_options
    assert isinstance(engine, TransformersVlmEngineOptions)
    assert engine.quantized is True
    assert engine.torch_dtype == "float16"
    assert engine.trust_remote_code is True
    assert engine.use_kv_cache is False


def test_document_extractor_imports_without_local_model_dependencies() -> None:
    code = """
import sys
for name in ('torch', 'transformers', 'docling_parse', 'pypdfium2', 'qwen_vl_utils'):
    sys.modules[name] = None
import docling.document_extractor
"""
    subprocess.run([sys.executable, "-c", code], check=True)
