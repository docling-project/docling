from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Union

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

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.picture_classification_options import (
    DocumentPictureClassifierOptions,
)
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.models.inference_engines.image_classification import (
    BaseImageClassificationEngine,
    ImageClassificationEngineInput,
    create_image_classification_engine,
)
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin


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
        self.engine: Optional[BaseImageClassificationEngine] = None
        self._classes: dict[int, str] = {}

        if self.enabled:
            self.engine = create_image_classification_engine(
                options=self.options.engine_options,
                model_spec=self.options.model_spec,
                artifacts_path=artifacts_path,
                accelerator_options=accelerator_options,
            )
            self.engine.initialize()
            self._classes = self.engine.get_label_mapping()

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

        if self.engine is None:
            raise RuntimeError("Picture classifier engine is not initialized.")

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

        engine_input_batch = [
            ImageClassificationEngineInput(image=image) for image in images
        ]
        engine_output_batch = self.engine.predict_batch(engine_input_batch)

        for item, output in zip(elements, engine_output_batch):
            predicted_classes = [
                PictureClassificationClass(
                    class_name=self._classes[label_id],
                    confidence=score,
                )
                for label_id, score in zip(output.label_ids, output.scores)
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

            if item.meta is not None and isinstance(item.meta, PictureMeta):
                item.meta.classification = classification_data
            else:
                item.meta = PictureMeta(classification=classification_data)

            yield item
