"""Options for picture classification stages."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from docling.datamodel import stage_model_specs
from docling.datamodel.stage_model_specs import (
    ImageClassificationModelSpec,
    ImageClassificationStagePresetMixin,
)
from docling.models.inference_engines.image_classification.base import (
    ImageClassificationEngineOptionsMixin,
)


class DocumentPictureClassifierOptions(
    ImageClassificationStagePresetMixin,
    ImageClassificationEngineOptionsMixin,
    BaseModel,
):
    """Options for configuring the DocumentPictureClassifier stage."""

    kind: ClassVar[str] = "document_picture_classifier"

    model_spec: ImageClassificationModelSpec = Field(
        default_factory=lambda: stage_model_specs.IMAGE_CLASSIFICATION_DOCUMENT_FIGURE.model_spec.model_copy(
            deep=True
        ),
        description="Image-classification model specification for picture classification.",
    )

    @property
    def repo_id(self) -> str:
        return self.model_spec.get_repo_id(self.engine_options.engine_type)

    @property
    def revision(self) -> str:
        return self.model_spec.get_revision(self.engine_options.engine_type)

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


DocumentPictureClassifierOptions.register_preset(
    stage_model_specs.IMAGE_CLASSIFICATION_DOCUMENT_FIGURE
)
