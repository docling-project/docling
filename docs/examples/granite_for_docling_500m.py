# %% [markdown]
# Granite for Docling 500M: convert a PDF with the doclang-native VLM.
#
# What this example does
# - Uses the `granite_for_docling_500m` preset (Transformers, MLX or in-process vLLM).
# - Parses model output as doclang and prints Markdown.
#
# Prerequisites
# - Install Docling with VLM extras.
# - Transformers: 5.23.0 or later, which includes the native
#   `GraniteDoclingForConditionalGeneration` architecture.
# - vLLM: a version with DeepStack support in its Transformers modeling backend
#   (later than 0.34.0). The same environment needs transformers 5.23.0 or later.
# - MLX (Apple Silicon): mlx-vlm 0.8.1 or later.
#
# How to run
# - From the repository root: `python docs/examples/granite_for_docling_500m.py`.

# %%

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

source = "https://arxiv.org/pdf/2501.17887"

# Default: AUTO_INLINE picks MLX, vLLM or Transformers from the local environment.
# The 258M `granite_docling` preset remains the CLI/SDK default; this example
# selects the 500M doclang model explicitly.
vlm_options = VlmConvertOptions.from_preset("granite_for_docling_500m")

# Extract only tables, with their bounding boxes:
# from docling.datamodel.pipeline_options import DocLangOptions
# from docling.datamodel.vlm_prompts import DocLangTask
# vlm_options = VlmConvertOptions.from_preset(
#     "granite_for_docling_500m",
#     doclang=DocLangOptions(tasks=[DocLangTask.TABLE, DocLangTask.LAYOUT]),
# )

# Pin Transformers:
# vlm_options = VlmConvertOptions.from_preset(
#     "granite_for_docling_500m",
#     engine_options=TransformersVlmEngineOptions(),
# )

# Pin MLX (Apple Silicon):
# vlm_options = VlmConvertOptions.from_preset(
#     "granite_for_docling_500m",
#     engine_options=MlxVlmEngineOptions(),
# )

# Pin in-process vLLM:
# vlm_options = VlmConvertOptions.from_preset(
#     "granite_for_docling_500m",
#     engine_options=VllmVlmEngineOptions(),
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
