import warnings

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


class LegacyStandardPdfPipeline(StandardPdfPipeline):
    def __init__(self, pipeline_options: PdfPipelineOptions) -> None:
        warnings.warn(
            "LegacyStandardPdfPipeline is deprecated; use StandardPdfPipeline.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(pipeline_options)
