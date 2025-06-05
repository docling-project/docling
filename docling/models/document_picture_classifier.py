from collections.abc import Iterable
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureClassificationClass,
    PictureClassificationData,
    PictureItem,
)
from PIL import Image
from pydantic import BaseModel

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.models.base_model import BaseEnrichmentModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device


class DocumentPictureClassifierOptions(BaseModel):
    """
    Options for configuring the DocumentPictureClassifier.

    Attributes
    ----------
    kind : Literal["document_picture_classifier"]
        Identifier for the type of classifier.
    """

    kind: Literal["document_picture_classifier"] = "document_picture_classifier"


class DocumentPictureClassifier(BaseEnrichmentModel):
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

    _model_repo_folder = "ds4sd--DocumentFigureClassifier"
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
            device = decide_device(accelerator_options.device)
            from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor import (
                DocumentFigureClassifierPredictor,
            )

            if artifacts_path is None:
                artifacts_path = self.download_models()
            else:
                artifacts_path = artifacts_path / self._model_repo_folder

            self.document_picture_classifier = DocumentFigureClassifierPredictor(
                artifacts_path=str(artifacts_path),
                device=device,
                num_threads=accelerator_options.num_threads,
            )

    @staticmethod
    def download_models(
        local_dir: Optional[Path] = None, force: bool = False, progress: bool = False
    ) -> Path:
        return download_hf_model(
            repo_id="ds4sd/DocumentFigureClassifier",
            revision="v1.0.1",
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

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
        element_batch: Iterable[NodeItem],
    ) -> Iterable[NodeItem]:
        """
        Processes a batch of elements and enriches them with classification predictions.

        Parameters
        ----------
        doc : DoclingDocument
            The document containing the elements to be processed.
        element_batch : Iterable[NodeItem]
            A batch of pictures to classify.

        Returns
        -------
        Iterable[NodeItem]
            An iterable of NodeItem objects after processing. The field
            'data.classification' is added containing the classification for each picture.
        """
        if not self.enabled:
            for element in element_batch:
                yield element
            return

        images: List[Union[Image.Image, np.ndarray]] = []
        elements: List[PictureItem] = []
        for el in element_batch:
            assert isinstance(el, PictureItem)
            elements.append(el)
            img = el.get_image(doc)
            assert img is not None
            images.append(img)

        outputs = self.document_picture_classifier.predict(images)

        for element, output in zip(elements, outputs):
            element.annotations.append(
                PictureClassificationData(
                    provenance="DocumentPictureClassifier",
                    predicted_classes=[
                        PictureClassificationClass(
                            class_name=pred[0],
                            confidence=pred[1],
                        )
                        for pred in output
                    ],
                )
            )

            yield element
