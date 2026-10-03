# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
from docling_core.types.doc import (
    DescriptionMetaField,
    DoclingDocument,
    PictureClassificationMetaField,
    PictureMeta,
)
from docling_core.types.doc.document import PictureClassificationPrediction

from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.models.stages.chart_extraction.granite_vision import (
    ChartExtractionVlmEngineModel,
)
from docling.models.stages.picture_classifier.document_picture_classifier import (
    DocumentPictureClassifier,
)
from docling.pipeline.base_pipeline import ConvertPipeline
from docling.pipeline.vlm_pipeline import VlmPipeline


def test_vlm_chart_extraction_runs_after_picture_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    description_model = object()

    def init_classifier(self, *, enabled: bool, **kwargs) -> None:
        self.enabled = enabled

    def init_chart(self, *, enabled: bool, **kwargs) -> None:
        self.enabled = enabled
        self.engine = None

    monkeypatch.setattr(DocumentPictureClassifier, "__init__", init_classifier)
    monkeypatch.setattr(ChartExtractionVlmEngineModel, "__init__", init_chart)
    monkeypatch.setattr(
        ConvertPipeline,
        "_get_picture_description_model",
        lambda self, artifacts_path=None: description_model,
    )
    monkeypatch.setattr(
        VlmPipeline,
        "_initialize_new_runtime_system",
        lambda self, pipeline_options: None,
    )

    pipeline = VlmPipeline(VlmPipelineOptions(do_chart_extraction=True))

    assert isinstance(pipeline.enrichment_pipe[0], DocumentPictureClassifier)
    assert pipeline.enrichment_pipe[0].enabled is True
    assert pipeline.enrichment_pipe[1] is description_model
    assert isinstance(pipeline.enrichment_pipe[2], ChartExtractionVlmEngineModel)


def test_picture_enrichments_skip_metadata_already_in_vlm_output() -> None:
    doc = DoclingDocument(name="picture")
    picture = doc.add_picture()
    classifier = DocumentPictureClassifier.__new__(DocumentPictureClassifier)
    classifier.enabled = True
    description_model = SimpleNamespace(enabled=True)

    assert classifier.is_processable(doc, picture)
    assert PictureDescriptionBaseModel.is_processable(description_model, doc, picture)

    picture.meta = PictureMeta(
        classification=PictureClassificationMetaField(
            predictions=[PictureClassificationPrediction(class_name="bar_chart")]
        ),
        description=DescriptionMetaField(text="Provided by VLM"),
    )

    assert not classifier.is_processable(doc, picture)
    assert not PictureDescriptionBaseModel.is_processable(
        description_model, doc, picture
    )
