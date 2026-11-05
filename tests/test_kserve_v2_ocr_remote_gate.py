# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""How the PDF pipeline passes `enable_remote_services` to OCR engines."""

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal, Optional

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    KserveV2OcrOptions,
    OcrOptions,
    PdfPipelineOptions,
)
from docling.exceptions import OperationNotAllowed
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.factories.ocr_factory import OcrFactory
from docling.models.inference_engines.common import KserveV2HttpClient
from docling.models.stages.ocr.kserve_v2_ocr_model import KserveV2OcrModel
from docling.pipeline import standard_pdf_pipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


def _kserve_pipeline_options(enable_remote_services: bool) -> PdfPipelineOptions:
    # The host is never contacted: building the pipeline creates no connection.
    return PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=False,
        enable_remote_services=enable_remote_services,
        ocr_options=KserveV2OcrOptions(
            url="http://kserve.invalid:8000",
            transport="http",
            model_name="ocr",
        ),
    )


def test_kserve_ocr_pipeline_requires_enable_remote_services() -> None:
    with pytest.raises(OperationNotAllowed):
        StandardPdfPipeline(_kserve_pipeline_options(enable_remote_services=False))


def test_kserve_ocr_pipeline_builds_client_with_enable_remote_services() -> None:
    pipeline = StandardPdfPipeline(
        _kserve_pipeline_options(enable_remote_services=True)
    )

    assert isinstance(pipeline.ocr_model, KserveV2OcrModel)
    assert isinstance(pipeline.ocr_model._kserve_client, KserveV2HttpClient)


class _PluginOcrOptions(OcrOptions):
    kind: ClassVar[Literal["test_plugin_ocr"]] = "test_plugin_ocr"


class _PluginOcrModel(BaseOcrModel):
    """OCR plugin written against the base four-argument constructor."""

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: _PluginOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        yield from page_batch

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        return _PluginOcrOptions


@pytest.mark.parametrize("enable_remote_services", [False, True])
def test_ocr_plugin_with_base_constructor_builds_in_pipeline(
    monkeypatch: pytest.MonkeyPatch, enable_remote_services: bool
) -> None:
    factory = OcrFactory()
    factory.register(_PluginOcrModel, "test-plugin", __name__)
    monkeypatch.setattr(
        standard_pdf_pipeline,
        "get_ocr_factory",
        lambda allow_external_plugins=False: factory,
    )

    pipeline = StandardPdfPipeline(
        PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=False,
            enable_remote_services=enable_remote_services,
            ocr_options=_PluginOcrOptions(lang=["en"]),
        )
    )

    assert isinstance(pipeline.ocr_model, _PluginOcrModel)
