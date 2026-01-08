# %% [markdown]
# Developing a picture enrichment model (classifier scaffold only).
#
# What this example does
# - Demonstrates how to implement an enrichment model that annotates pictures.
# - Adds a dummy PictureClassificationData entry to each PictureItem.
#
# Important
# - This is a scaffold for development; it does not run a real classifier.
#
# How to run
# - From the repo root: `python docs/examples/develop_picture_enrichment.py`.
#
# Notes
# - Enables picture image generation and sets `images_scale` to improve crops.
# - Extends `StandardPdfPipeline` with a custom enrichment stage.

# %%

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureClassificationMetaField,
    PictureItem,
    PictureMeta,
)
from docling_core.types.doc.document import PictureClassificationPrediction

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.base_model import BaseEnrichmentModel
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


class ExamplePictureClassifierPipelineOptions(PdfPipelineOptions):
    do_picture_classifer: bool = True


class ExamplePictureClassifierEnrichmentModel(BaseEnrichmentModel):
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, PictureItem)

    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[NodeItem]
    ) -> Iterable[Any]:
        if not self.enabled:
            return

        for element in element_batch:
            assert isinstance(element, PictureItem)

            # uncomment this to interactively visualize the image
            # element.get_image(doc).show()  # may block; avoid in headless runs

            # Populate the new meta.classification field
            classification_data = PictureClassificationMetaField(
                predictions=[
                    PictureClassificationPrediction(
                        created_by="example_classifier-0.0.1",
                        class_name="dummy",
                        confidence=0.42,
                    )
                ],
            )

            if element.meta is not None:
                element.meta.classification = classification_data
            else:
                element.meta = PictureMeta(classification=classification_data)

            yield element


class ExamplePictureClassifierPipeline(StandardPdfPipeline):
    def __init__(self, pipeline_options: ExamplePictureClassifierPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: ExamplePictureClassifierPipeline

        self.enrichment_pipe = [
            ExamplePictureClassifierEnrichmentModel(
                enabled=pipeline_options.do_picture_classifer
            )
        ]

    @classmethod
    def get_default_options(cls) -> ExamplePictureClassifierPipelineOptions:
        return ExamplePictureClassifierPipelineOptions()


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"

    pipeline_options = ExamplePictureClassifierPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ExamplePictureClassifierPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    result = doc_converter.convert(input_doc_path)

    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem):
            print(
                f"The model populated the `meta` portion of picture {element.self_ref}:\n{element.meta}"
            )


if __name__ == "__main__":
    main()
