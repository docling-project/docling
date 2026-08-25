# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import re
from collections import OrderedDict

import pytest
from pydantic import AnyUrl, SecretStr

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.extraction import ExtractionResult, ExtractionTemplateType
from docling.datamodel.image_classification_engine_options import (
    TransformersImageClassificationEngineOptions,
)
from docling.datamodel.object_detection_engine_options import (
    TransformersObjectDetectionEngineOptions,
)
from docling.datamodel.picture_classification_options import (
    DocumentPictureClassifierOptions,
)
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    PictureDescriptionBaseOptions,
    PipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, FormatOption
from docling.document_extractor import DocumentExtractor, ExtractionFormatOption
from docling.pipeline.base_extraction_pipeline import BaseExtractionPipeline
from docling.pipeline.base_pipeline import BasePipeline
from docling.utils.pipeline_cache import create_pipeline_options_hash


class _Pipeline(BasePipeline):
    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        return conv_res

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        return conv_res.status

    @classmethod
    def get_default_options(cls) -> PipelineOptions:
        return PipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend) -> bool:
        return True


class _ExtractionPipeline(BaseExtractionPipeline):
    def _extract_data(
        self,
        ext_res: ExtractionResult,
        template: ExtractionTemplateType | None = None,
    ) -> ExtractionResult:
        return ext_res

    def _determine_status(self, ext_res: ExtractionResult) -> ConversionStatus:
        return ext_res.status

    @classmethod
    def get_default_options(cls) -> PipelineOptions:
        return PipelineOptions()


class _FirstPictureOptions(PictureDescriptionBaseOptions):
    pass


class _SecondPictureOptions(PictureDescriptionBaseOptions):
    pass


class _SecretOptions(PipelineOptions):
    credential: SecretStr


class _PatternOptions(PipelineOptions):
    pattern: re.Pattern[str]


def _api_options(url: str, *, reverse_params: bool = False) -> PdfPipelineOptions:
    params = (
        {"metadata": {"b": 2, "a": 1}, "model": "example"}
        if reverse_params
        else {"model": "example", "metadata": {"a": 1, "b": 2}}
    )
    return PdfPipelineOptions(
        picture_description_options=PictureDescriptionApiOptions(
            url=AnyUrl(url),
            params=params,
        )
    )


def test_pipeline_options_hash_includes_concrete_fields_and_normalizes_mappings():
    first = _api_options("https://first.example.com/v1", reverse_params=False)
    reordered = _api_options("https://first.example.com/v1", reverse_params=True)
    different = _api_options("https://second.example.com/v1")

    assert create_pipeline_options_hash(first) == create_pipeline_options_hash(
        reordered
    )
    assert create_pipeline_options_hash(first) != create_pipeline_options_hash(
        different
    )


def test_pipeline_options_hash_includes_concrete_option_types():
    first = PdfPipelineOptions(picture_description_options=_FirstPictureOptions())
    second = PdfPipelineOptions(picture_description_options=_SecondPictureOptions())

    assert create_pipeline_options_hash(first) != create_pipeline_options_hash(second)


def test_pipeline_options_hash_includes_concrete_engine_fields():
    first_classifier = PdfPipelineOptions(
        picture_classification_options=DocumentPictureClassifierOptions(
            engine_options=TransformersImageClassificationEngineOptions(
                torch_dtype="float16"
            )
        )
    )
    second_classifier = PdfPipelineOptions(
        picture_classification_options=DocumentPictureClassifierOptions(
            engine_options=TransformersImageClassificationEngineOptions(
                torch_dtype="float32"
            )
        )
    )
    assert create_pipeline_options_hash(
        first_classifier
    ) != create_pipeline_options_hash(second_classifier)

    first_layout = PdfPipelineOptions(
        layout_options=LayoutObjectDetectionOptions(
            engine_options=TransformersObjectDetectionEngineOptions(
                torch_dtype="float16"
            )
        )
    )
    second_layout = PdfPipelineOptions(
        layout_options=LayoutObjectDetectionOptions(
            engine_options=TransformersObjectDetectionEngineOptions(
                torch_dtype="float32"
            )
        )
    )
    assert create_pipeline_options_hash(first_layout) != create_pipeline_options_hash(
        second_layout
    )


def test_pipeline_options_hash_preserves_pass_through_container_types():
    hashes = {
        create_pipeline_options_hash(
            PdfPipelineOptions(
                ocr_options=RapidOcrOptions(rapidocr_params={"value": value})
            )
        )
        for value in ([1, 2], (1, 2), {1, 2})
    }

    assert len(hashes) == 3


def test_pipeline_options_hash_supports_non_utf8_bytes():
    hashes = {
        create_pipeline_options_hash(
            PdfPipelineOptions(
                ocr_options=RapidOcrOptions(rapidocr_params={"value": value})
            )
        )
        for value in (b"\xff", bytearray(b"\xff"))
    }

    assert len(hashes) == 2


def test_pipeline_options_hash_preserves_lossy_scalar_values():
    assert create_pipeline_options_hash(
        _SecretOptions(credential=SecretStr("alpha"))
    ) != create_pipeline_options_hash(_SecretOptions(credential=SecretStr("bravo")))
    assert create_pipeline_options_hash(
        _PatternOptions(pattern=re.compile("value"))
    ) != create_pipeline_options_hash(
        _PatternOptions(pattern=re.compile("value", re.IGNORECASE))
    )


def test_pipeline_options_hash_does_not_flatten_container_subclasses():
    first = RapidOcrOptions(
        rapidocr_params={"value": OrderedDict((("first", 1), ("second", 2)))}
    )
    second = RapidOcrOptions(
        rapidocr_params={"value": OrderedDict((("second", 2), ("first", 1)))}
    )

    assert create_pipeline_options_hash(
        PdfPipelineOptions(ocr_options=first)
    ) != create_pipeline_options_hash(PdfPipelineOptions(ocr_options=second))


def test_pipeline_options_hash_rejects_cycles():
    cyclic: dict[str, object] = {}
    cyclic["value"] = cyclic
    options = PdfPipelineOptions(ocr_options=RapidOcrOptions(rapidocr_params=cyclic))

    with pytest.raises(ValueError, match="cannot contain cyclic values"):
        create_pipeline_options_hash(options)


def test_cache_projection_does_not_change_public_model_serialization():
    options = PdfPipelineOptions(
        picture_description_options=PictureDescriptionApiOptions(
            url=AnyUrl("https://example.com/v1"),
            headers={"Authorization": "secret"},
        )
    )

    assert "headers" not in options.model_dump()["picture_description_options"]


def test_converter_does_not_reuse_pipeline_for_different_api_options():
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
        format_options={
            InputFormat.PDF: FormatOption(
                pipeline_cls=_Pipeline,
                pipeline_options=_api_options("https://first.example.com/v1"),
                backend=NoOpBackend,
            ),
            InputFormat.DOCX: FormatOption(
                pipeline_cls=_Pipeline,
                pipeline_options=_api_options("https://second.example.com/v1"),
                backend=NoOpBackend,
            ),
        },
    )

    first = converter._get_pipeline(InputFormat.PDF)
    second = converter._get_pipeline(InputFormat.DOCX)

    assert first is not second
    assert len(converter._get_initialized_pipelines()) == 2


def test_extractor_does_not_reuse_pipeline_for_different_api_options():
    extractor = DocumentExtractor(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
        extraction_format_options={
            InputFormat.PDF: ExtractionFormatOption(
                pipeline_cls=_ExtractionPipeline,
                pipeline_options=_api_options("https://first.example.com/v1"),
                backend=NoOpBackend,
            ),
            InputFormat.IMAGE: ExtractionFormatOption(
                pipeline_cls=_ExtractionPipeline,
                pipeline_options=_api_options("https://second.example.com/v1"),
                backend=NoOpBackend,
            ),
        },
    )

    first = extractor._get_pipeline(InputFormat.PDF)
    second = extractor._get_pipeline(InputFormat.IMAGE)

    assert first is not second
    assert len(extractor._initialized_pipelines) == 2
