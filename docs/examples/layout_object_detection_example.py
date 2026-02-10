# %% [markdown]
#
# What this example does
# - Run layout detection using the object-detection runtime with ONNX
# - Demonstrates the OBJECT_DETECTION_LAYOUT_HERON preset with ONNX Runtime
# - Detects document structure elements (text blocks, tables, figures, etc.)
#
# Requirements
# - Python 3.9+
# - Install Docling: `pip install docling`
# - ONNX Runtime (automatically installed with docling)
#
# How to run (from repo root)
# - `python docs/examples/layout_object_detection_example.py`
#
# ## Example code
# %%

import logging

from docling_core.types.doc.base import ImageRefMode

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
    PdfPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)

_log = logging.getLogger(__name__)


def main():
    # Configure logging to display info messages
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("docling").setLevel(logging.INFO)

    # Use a sample PDF from the test data (path relative to repo root)
    input_doc_path = "tests/data/pdf/2206.01062.pdf"

    # Configure pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    # Enable debug visualization
    settings.debug.visualize_layout = True

    # Create layout options using the ONNX Runtime preset
    # The "layout_heron_default" preset is configured to use ONNX Runtime
    # with the model file "model.onnx" from the HuggingFace repository
    layout_options = LayoutObjectDetectionOptions.from_preset("layout_heron_default")

    # The preset already configures ONNX Runtime as the default engine,
    # but you can override engine options if needed:
    # from docling.datamodel.object_detection_engine_options import (
    #     OnnxRuntimeObjectDetectionEngineOptions,
    # )
    # layout_options.engine_options = OnnxRuntimeObjectDetectionEngineOptions(
    #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    # )

    pipeline_options.layout_options = layout_options

    # Create converter with the configured pipeline
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )

    # Convert the document
    result = converter.convert(input_doc_path)
    if False:
        viz = result.document.get_visualization()
        for k, v in viz.items():
            v.show()
    # Save output
    result.document.save_as_html(
        "layout_object_detection_example.html", image_mode=ImageRefMode.EMBEDDED
    )


if __name__ == "__main__":
    main()
