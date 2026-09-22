# %% [markdown]
# Granite for Docling 500M: convert a PDF with the doclang-native VLM.
#
# What this example does
# - Uses the `granite_for_docling_500m` preset (Transformers or in-process vLLM).
# - Parses model output as doclang and prints Markdown.
#
# Prerequisites
# - Install Docling with VLM extras.
# - Transformers: a release (or `trust_remote_code` Hub repo) that includes
#   `GraniteForDoclingForConditionalGeneration`.
# - vLLM: a release that registers the Granite for Docling hybrid model.
#
# How to run
# - From the repository root: `python docs/examples/granite_for_docling_500m.py`.

# %%

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

source = "https://arxiv.org/pdf/2501.17887"

# Default: AUTO_INLINE picks Transformers or vLLM from the local environment.
# The 258M `granite_docling` preset remains the CLI/SDK default; this example
# selects the 500M doclang model explicitly.
vlm_options = VlmConvertOptions.from_preset("granite_for_docling_500m")

# Pin Transformers (custom architecture still needs trust_remote_code until
# the class ships on PyPI transformers):
# vlm_options = VlmConvertOptions.from_preset(
#     "granite_for_docling_500m",
#     engine_options=TransformersVlmEngineOptions(trust_remote_code=True),
# )

# Pin in-process vLLM:
# vlm_options = VlmConvertOptions.from_preset(
#     "granite_for_docling_500m",
#     engine_options=VllmVlmEngineOptions(trust_remote_code=True),
# )

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=VlmPipelineOptions(vlm_options=vlm_options),
        ),
    }
)

doc = converter.convert(source=source).document
print(doc.export_to_markdown())
