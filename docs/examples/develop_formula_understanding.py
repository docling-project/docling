# WARNING
# This example demonstrates only how to develop a new enrichment model.
# It does not run the actual formula understanding model.

import logging
from collections.abc import Iterable
from pathlib import Path

from docling_core.types.doc import DocItemLabel, DoclingDocument, NodeItem, TextItem

from docling.datamodel.base_models import InputFormat, ItemAndImageEnrichmentElement
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


class ExampleFormulaUnderstandingPipelineOptions(PdfPipelineOptions):
    do_formula_understanding: bool = True


# A new enrichment model using both the document element and its image as input
class ExampleFormulaUnderstandingEnrichmentModel(BaseItemAndImageEnrichmentModel):
    images_scale = 2.6

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return (
            self.enabled
            and isinstance(element, TextItem)
            and element.label == DocItemLabel.FORMULA
        )

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            return

        for enrich_element in element_batch:
            enrich_element.image.show()

            yield enrich_element.item


# How the pipeline can be extended.
class ExampleFormulaUnderstandingPipeline(StandardPdfPipeline):
    def __init__(self, pipeline_options: ExampleFormulaUnderstandingPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: ExampleFormulaUnderstandingPipelineOptions

        self.enrichment_pipe = [
            ExampleFormulaUnderstandingEnrichmentModel(
                enabled=self.pipeline_options.do_formula_understanding
            )
        ]

        if self.pipeline_options.do_formula_understanding:
            self.keep_backend = True

    @classmethod
    def get_default_options(cls) -> ExampleFormulaUnderstandingPipelineOptions:
        return ExampleFormulaUnderstandingPipelineOptions()


# Example main. In the final version, we simply have to set do_formula_understanding to true.
def main():
    logging.basicConfig(level=logging.INFO)

    input_doc_path = Path("./tests/data/pdf/2203.01017v2.pdf")

    pipeline_options = ExampleFormulaUnderstandingPipelineOptions()
    pipeline_options.do_formula_understanding = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ExampleFormulaUnderstandingPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    doc_converter.convert(input_doc_path)


if __name__ == "__main__":
    main()
