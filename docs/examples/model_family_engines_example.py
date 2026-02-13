# %% [markdown]
#
# What this example does
# - Demonstrates runtime abstraction for object detection engines
# - Demonstrates runtime abstraction for image classification engines
# - Runs the same layout detection task with both ONNX Runtime and Transformers engines
# - Runs picture classification with the corresponding ONNX Runtime or Transformers engine
# - Shows how to easily switch between inference engines while using the same model
# - Detects document structure elements (text blocks, tables, figures, etc.)
#
# Requirements
# - Python 3.10+
# - Install Docling: `pip install docling[onnxruntime]`
#
# How to run (from repo root)
# - `python docs/examples/model_family_engines_example.py`
#
# ## Example code
# %%

import logging
import sys

from docling_core.types.doc.base import ImageRefMode

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.image_classification_engine_options import (
    OnnxRuntimeImageClassificationEngineOptions,
    TransformersImageClassificationEngineOptions,
)
from docling.datamodel.object_detection_engine_options import (
    OnnxRuntimeObjectDetectionEngineOptions,
    TransformersObjectDetectionEngineOptions,
)
from docling.datamodel.picture_classification_options import (
    DocumentPictureClassifierOptions,
)
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
    PdfPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)

_log = logging.getLogger(__name__)


def is_onnxruntime_available() -> bool:
    """Return True when onnxruntime can be imported in this environment."""
    try:
        import onnxruntime
    except ImportError:
        return False
    return True


def _build_engine_options(engine_kind: str):
    """Build paired layout/classification runtime options from a single selector."""
    if engine_kind == "onnxruntime":
        return (
            "ONNX Runtime",
            OnnxRuntimeObjectDetectionEngineOptions(),
            OnnxRuntimeImageClassificationEngineOptions(),
        )
    if engine_kind == "transformers":
        return (
            "Transformers",
            TransformersObjectDetectionEngineOptions(compile_model=False),
            TransformersImageClassificationEngineOptions(compile_model=False),
        )
    raise ValueError(f"Unsupported engine kind: {engine_kind}")


def run_with_engine(engine_kind: str, input_doc_path: str):
    """Run layout detection and picture classification with paired engines."""
    (
        engine_name,
        layout_engine_options,
        picture_engine_options,
    ) = _build_engine_options(engine_kind)

    _log.info(f"{'=' * 80}")
    _log.info("Running conversion with %s runtime", engine_name)
    _log.info(f"{'=' * 80}\n")

    # Configure pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.do_picture_classification = True
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CPU
    )

    # Create layout options with the specified engine
    layout_options = LayoutObjectDetectionOptions.from_preset("layout_heron_default")
    layout_options.engine_options = layout_engine_options

    pipeline_options.layout_options = layout_options

    # Create picture-classifier options with the same engine family as layout
    picture_classifier_options = DocumentPictureClassifierOptions.from_preset(
        "document_figure_classifier_v2"
    )
    picture_classifier_options.engine_options = picture_engine_options

    pipeline_options.picture_classification_options = picture_classifier_options

    # Create converter with the configured pipeline
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )

    # Convert the document
    result = converter.convert(input_doc_path)

    # Save output with engine-specific filename
    output_filename = f"model_family_engines_{engine_kind}.html"
    result.document.save_as_html(output_filename, image_mode=ImageRefMode.EMBEDDED)
    _log.info(
        "Completed %s run: output=%s, pictures=%d",
        engine_name,
        output_filename,
        len(result.document.pictures),
    )

    return result


def main():
    # Configure logging to display info messages
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("docling").setLevel(logging.INFO)

    # Use a sample PDF from the test data (path relative to repo root)
    input_doc_path = "tests/data/pdf/2206.01062.pdf"

    # Run 1: ONNX Runtime Engine (if available in the current environment)
    if is_onnxruntime_available():
        # Uses automatic device selection via pipeline accelerator options
        run_with_engine("onnxruntime", input_doc_path)
    else:
        _log.warning(
            "Skipping ONNX engine run: onnxruntime is not available for Python %d.%d. "
            "Use Python < 3.14 and install `docling[onnxruntime]`.",
            sys.version_info.major,
            sys.version_info.minor,
        )

    # Run 2: Transformers Engine
    # Uses PyTorch with HuggingFace Transformers and automatic device selection
    run_with_engine("transformers", input_doc_path)


if __name__ == "__main__":
    main()
