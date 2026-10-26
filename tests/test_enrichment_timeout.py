# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from collections.abc import Iterable
from io import BytesIO

from docling_core.types.doc import DoclingDocument, NodeItem

from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    DocItemLabel,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    InputFormat,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import PipelineOptions
from docling.models.base_model import GenericEnrichmentModel
from docling.pipeline.base_pipeline import BasePipeline


class _RecordingEnrichmentModel(GenericEnrichmentModel[NodeItem]):
    def __init__(self):
        self.calls = 0

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return True

    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> NodeItem:
        self.calls += 1
        return element

    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[NodeItem]
    ) -> Iterable[NodeItem]:
        yield from element_batch


class _TimeoutPipeline(BasePipeline):
    def __init__(self, timed_out: bool):
        super().__init__(PipelineOptions())
        self.timed_out = timed_out
        self.model = _RecordingEnrichmentModel()
        self.enrichment_pipe = [self.model]

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        conv_res.document.add_text(label=DocItemLabel.TEXT, text="content")
        if self.timed_out:
            conv_res.errors.append(
                ErrorItem(
                    component_type=DoclingComponentType.PIPELINE,
                    module_name=self.__class__.__name__,
                    error_message="document timed out",
                    category=FailureCategory.TIMEOUT,
                )
            )
        return conv_res

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        if conv_res.errors:
            return ConversionStatus.PARTIAL_SUCCESS
        return ConversionStatus.SUCCESS

    @classmethod
    def get_default_options(cls) -> PipelineOptions:
        return PipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend) -> bool:
        return True


def _input_document() -> InputDocument:
    return InputDocument(
        path_or_stream=BytesIO(b"test"),
        filename="test.pdf",
        format=InputFormat.PDF,
        backend=NoOpBackend,
    )


def test_enrichment_is_skipped_after_document_timeout():
    pipeline = _TimeoutPipeline(timed_out=True)

    result = pipeline.execute(_input_document(), raises_on_error=True)

    assert result.status == ConversionStatus.PARTIAL_SUCCESS
    assert pipeline.model.calls == 0


def test_enrichment_still_runs_without_document_timeout():
    pipeline = _TimeoutPipeline(timed_out=False)

    result = pipeline.execute(_input_document(), raises_on_error=True)

    assert result.status == ConversionStatus.SUCCESS
    assert pipeline.model.calls == 1
