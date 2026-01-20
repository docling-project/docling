import sys
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureClassificationClass,
    PictureClassificationData,
    PictureClassificationMetaField,
    PictureItem,
    PictureMeta,
)
from docling_core.types.doc.document import PictureClassificationPrediction
from PIL import Image
from pydantic import BaseModel

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin
from docling.utils.accelerator_utils import decide_device

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


class DocumentPictureClassifierOptions(BaseModel):
    """
    Options for configuring the DocumentPictureClassifier.
    """

    kind: Literal["document_picture_classifier"] = "document_picture_classifier"
    repo_id: str = "docling-project/DocumentFigureClassifier-v2.0"
    revision: str = "main"

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


class DocumentPictureClassifier(
    BaseItemAndImageEnrichmentModel, HuggingFaceModelDownloadMixin
):
    """
    A model for classifying pictures in documents.

    This class enriches document pictures with predicted classifications
    based on a predefined set of classes.

    Attributes
    ----------
    enabled : bool
        Whether the classifier is enabled for use.
    options : DocumentPictureClassifierOptions
        Configuration options for the classifier.
    document_picture_classifier : DocumentPictureClassifierPredictor
        The underlying prediction model, loaded if the classifier is enabled.

    Methods
    -------
    __init__(enabled, artifacts_path, options, accelerator_options)
        Initializes the classifier with specified configurations.
    is_processable(doc, element)
        Checks if the given element can be processed by the classifier.
    __call__(doc, element_batch)
        Processes a batch of elements and adds classification annotations.
    """

    images_scale = 2

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: DocumentPictureClassifierOptions,
        accelerator_options: AcceleratorOptions,
    ):
        """
        Initializes the DocumentPictureClassifier.

        Parameters
        ----------
        enabled : bool
            Indicates whether the classifier is enabled.
        artifacts_path : Optional[Union[Path, str]],
            Path to the directory containing model artifacts.
        options : DocumentPictureClassifierOptions
            Configuration options for the classifier.
        accelerator_options : AcceleratorOptions
            Options for configuring the device and parallelism.
        """
        self.enabled = enabled
        self.options = options

        if self.enabled:
            self._device = decide_device(accelerator_options.device)

            repo_cache_folder = self.options.repo_cache_folder

            if artifacts_path is None:
                artifacts_path = self.download_models(
                    self.options.repo_id, revision=self.options.revision
                )
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            with _model_init_lock:
                # Image processor
                self._processor = AutoImageProcessor.from_pretrained(
                    artifacts_path, use_fast=True
                )

                # Model
                self._model = AutoModelForImageClassification.from_pretrained(
                    artifacts_path,
                    device_map=self._device,
                )

                if sys.version_info < (3, 14):
                    self._model = torch.compile(self._model)  # type: ignore
                else:
                    self._model.eval()

            self._classes = self._model.config.id2label

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        """
        Determines if the given element can be processed by the classifier.

        Parameters
        ----------
        doc : DoclingDocument
            The document containing the element.
        element : NodeItem
            The element to be checked.

        Returns
        -------
        bool
            True if the element is a PictureItem and processing is enabled; False otherwise.
        """
        return self.enabled and isinstance(element, PictureItem)

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        """
        Processes a batch of elements and enriches them with classification predictions.

        Parameters
        ----------
        doc : DoclingDocument
            The document containing the elements to be processed.
        element_batch : Iterable[ItemAndImageEnrichmentElement]
            A batch of pictures to classify.

        Returns
        -------
        Iterable[NodeItem]
            An iterable of NodeItem objects after processing. The field
            'data.classification' is added containing the classification for each picture.
        """
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        import torch

        images: List[Union[Image.Image, np.ndarray]] = []
        elements: List[PictureItem] = []
        for i, el in enumerate(element_batch):
            assert isinstance(el.item, PictureItem)
            elements.append(el.item)

            raw_image = el.image
            if isinstance(raw_image, Image.Image):
                raw_image = raw_image.convert("RGB")
            elif isinstance(raw_image, np.ndarray):
                raw_image = Image.fromarray(raw_image).convert("RGB")
            else:
                raise TypeError(
                    "Supported input formats are PIL.Image.Image or numpy.ndarray."
                )
            images.append(raw_image)

        inputs = self._processor(images=images, return_tensors="pt")
        # move inputs to the same device as the model
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (batch_size, num_classes)
            probs_batch = logits.softmax(dim=1)  # (batch_size, num_classes)
            probs_batch = probs_batch.cpu().numpy().tolist()

        predictions_batch = []
        for probs_image in probs_batch:
            preds = [(self._classes[i], prob) for i, prob in enumerate(probs_image)]
            preds.sort(key=lambda t: t[1], reverse=True)
            predictions_batch.append(preds)

        for item, output in zip(elements, predictions_batch):
            predicted_classes = [
                PictureClassificationClass(
                    class_name=pred[0],
                    confidence=pred[1],
                )
                for pred in output
            ]

            # FIXME: annotations is deprecated, remove once all consumers use meta.classification
            item.annotations.append(
                PictureClassificationData(
                    provenance="DocumentPictureClassifier",
                    predicted_classes=predicted_classes,
                )
            )

            # Store classification in the new meta field
            predictions = [
                PictureClassificationPrediction(
                    class_name=pred.class_name,
                    confidence=pred.confidence,
                    created_by="DocumentPictureClassifier",
                )
                for pred in predicted_classes
            ]
            classification_data = PictureClassificationMetaField(
                predictions=predictions,
            )

            if item.meta is not None:
                item.meta.classification = classification_data
            else:
                item.meta = PictureMeta(classification=classification_data)

            yield item
