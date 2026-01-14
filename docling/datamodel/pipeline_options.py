import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Dict, List, Literal, Optional, Union

from docling_core.types.doc import PictureClassificationLabel
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
)
from typing_extensions import deprecated

from docling.datamodel import asr_model_specs, vlm_model_specs

# Import the following for backwards compatibility
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.layout_model_specs import (
    DOCLING_LAYOUT_EGRET_LARGE,
    DOCLING_LAYOUT_EGRET_MEDIUM,
    DOCLING_LAYOUT_EGRET_XLARGE,
    DOCLING_LAYOUT_HERON,
    DOCLING_LAYOUT_HERON_101,
    DOCLING_LAYOUT_V2,
    LayoutModelConfig,
)
from docling.datamodel.pipeline_options_asr_model import (
    InlineAsrOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.datamodel.vlm_model_specs import (
    GRANITE_VISION_OLLAMA as granite_vision_vlm_ollama_conversion_options,
    GRANITE_VISION_TRANSFORMERS as granite_vision_vlm_conversion_options,
    NU_EXTRACT_2B_TRANSFORMERS,
    SMOLDOCLING_MLX as smoldocling_vlm_mlx_conversion_options,
    SMOLDOCLING_TRANSFORMERS as smoldocling_vlm_conversion_options,
    VlmModelType,
)

_log = logging.getLogger(__name__)


class BaseOptions(BaseModel):
    """Base class for options."""

    kind: ClassVar[str]


class TableFormerMode(str, Enum):
    """Operating modes for TableFormer table structure extraction model.

    Controls the trade-off between processing speed and extraction accuracy.
    Choose based on your performance requirements and document complexity.

    Attributes:
        FAST: Fast mode prioritizes speed over precision. Suitable for simple tables or high-volume
            processing.
        ACCURATE: Accurate mode provides higher quality results with slower processing. Recommended for complex
            tables and production use.
    """

    FAST = "fast"
    ACCURATE = "accurate"


class BaseTableStructureOptions(BaseOptions):
    """Base options for table structure models."""


class TableStructureOptions(BaseTableStructureOptions):
    """Configuration for table structure extraction using the TableFormer model."""

    kind: ClassVar[str] = "docling_tableformer"
    do_cell_matching: bool = Field(
        default=True,
        description=(
            "Enable cell matching to align detected table cells with their content. When enabled, the model "
            "attempts to match table structure predictions with actual cell content for improved accuracy."
        ),
    )
    mode: TableFormerMode = Field(
        default=TableFormerMode.ACCURATE,
        description=(
            "Table structure extraction mode. `accurate` provides higher quality results with slower processing, "
            "while `fast` prioritizes speed over precision. Recommended: `accurate` for production use."
        ),
    )


class OcrOptions(BaseOptions):
    """OCR options."""

    lang: Annotated[
        List[str],
        Field(
            description="List of OCR languages to use. The format must match the values of the OCR engine of choice.",
            examples=[["deu", "eng"]],
        ),
    ]
    force_full_page_ocr: bool = Field(
        default=False,
        description="If enabled, a full-page OCR is always applied.",
        examples=[False],
    )
    bitmap_area_threshold: float = Field(
        default=0.05,
        description="Percentage of the page area for a bitmap to be processed with OCR.",
        examples=[0.05, 0.1],
    )


class OcrAutoOptions(OcrOptions):
    """Automatic OCR engine selection based on system availability."""

    kind: ClassVar[Literal["auto"]] = "auto"
    lang: List[str] = Field(
        default=[],
        description=(
            "The automatic OCR engine will use the default values of the engine. Please specify the engine "
            "explicitly to change the language selection."
        ),
    )


class RapidOcrOptions(OcrOptions):
    """Configuration for RapidOCR engine with multiple backend support.

    See Also:
        - https://rapidai.github.io/RapidOCRDocs/install_usage/api/RapidOCR/
        - https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/#__tabbed_3_4
    """

    kind: ClassVar[Literal["rapidocr"]] = "rapidocr"
    # English and chinese are the most commly used models and have been tested with RapidOCR.
    lang: List[str] = Field(
        default=["english", "chinese"],
        description=(
            "List of OCR languages. Note: RapidOCR does not currently support language selection; "
            "this parameter is reserved for future compatibility. See RapidOCR documentation for supported languages."
        ),
    )
    backend: Literal["onnxruntime", "openvino", "paddle", "torch"] = Field(
        default="onnxruntime",
        description=(
            "Inference backend for RapidOCR. Options: `onnxruntime` (default, cross-platform), `openvino` (Intel), "
            "`paddle` (PaddlePaddle), `torch` (PyTorch). Choose based on your hardware and available libraries."
        ),
    )
    text_score: float = Field(
        default=0.5,
        description=(
            "Minimum confidence score for text detection. Text regions with scores below this threshold are "
            "filtered out. Range: 0.0-1.0. Lower values detect more text but may include false positives."
        ),
    )
    use_det: Optional[bool] = Field(
        default=None,
        description="Enable text detection stage. If None, uses RapidOCR default behavior.",
    )
    use_cls: Optional[bool] = Field(
        default=None,
        description="Enable text direction classification stage. If None, uses RapidOCR default behavior.",
    )
    use_rec: Optional[bool] = Field(
        default=None,
        description="Enable text recognition stage. If None, uses RapidOCR default behavior.",
    )
    print_verbose: bool = Field(
        default=False,
        description="Enable verbose logging output from RapidOCR for debugging purposes.",
    )
    det_model_path: Optional[str] = Field(
        default=None,
        description="Custom path to text detection model. If None, uses default RapidOCR model.",
    )
    cls_model_path: Optional[str] = Field(
        default=None,
        description="Custom path to text classification model. If None, uses default RapidOCR model.",
    )
    rec_model_path: Optional[str] = Field(
        default=None,
        description="Custom path to text recognition model. If None, uses default RapidOCR model.",
    )
    rec_keys_path: Optional[str] = Field(
        default=None,
        description="Custom path to recognition keys file. If None, uses default RapidOCR keys.",
    )
    rec_font_path: Optional[str] = Field(
        default=None,
        description="Deprecated. Use font_path instead.",
        deprecated=True,
    )
    font_path: Optional[str] = Field(
        default=None,
        description="Custom path to font file for text rendering in visualization.",
    )
    rapidocr_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional parameters to pass through to RapidOCR engine. Use this to override or extend "
            "default RapidOCR configuration with engine-specific options."
        ),
    )
    model_config = ConfigDict(
        extra="forbid",
    )


class EasyOcrOptions(OcrOptions):
    """Configuration for EasyOCR engine."""

    kind: ClassVar[Literal["easyocr"]] = "easyocr"
    lang: List[str] = Field(
        default=["fr", "de", "es", "en"],
        description=(
            "List of language codes for OCR. EasyOCR supports 80+ languages. Use ISO 639-1 codes "
            "(e.g., `en`, `fr`, `de`). Multiple languages can be specified for multilingual documents."
        ),
    )
    use_gpu: Optional[bool] = Field(
        default=None,
        description=(
            "Enable GPU acceleration for EasyOCR. If None, automatically detects and uses GPU if available. "
            "Set to False to force CPU-only processing."
        ),
    )
    confidence_threshold: float = Field(
        default=0.5,
        description=(
            "Minimum confidence score for text recognition. Text with confidence below this threshold is filtered out. "
            "Range: 0.0-1.0. Lower values include more text but may reduce accuracy."
        ),
    )
    model_storage_directory: Optional[str] = Field(
        default=None,
        description=(
            "Directory path for storing downloaded EasyOCR models. If None, uses default EasyOCR cache location. "
            "Useful for offline environments or custom model management."
        ),
    )
    recog_network: Optional[str] = Field(
        default="standard",
        description=(
            "Recognition network architecture to use. Options: `standard` (default, balanced), `craft` (higher "
            "accuracy). Different networks may perform better on specific document types."
        ),
    )
    download_enabled: bool = Field(
        default=True,
        description=(
            "Allow automatic download of EasyOCR models on first use. Disable for offline environments "
            "where models must be pre-installed."
        ),
    )
    suppress_mps_warnings: bool = Field(
        default=True,
        description=(
            "Suppress Metal Performance Shaders (MPS) warnings on macOS. Reduces console noise when using "
            "Apple Silicon GPUs with EasyOCR."
        ),
    )
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )


class TesseractCliOcrOptions(OcrOptions):
    """Configuration for Tesseract OCR via command-line interface."""

    kind: ClassVar[Literal["tesseract"]] = "tesseract"
    lang: List[str] = Field(
        default=["fra", "deu", "spa", "eng"],
        description=(
            "List of Tesseract language codes. Use 3-letter ISO 639-2 codes (e.g., `eng`, `fra`, `deu`). "
            "Multiple languages enable multilingual OCR. Requires corresponding Tesseract language data files."
        ),
    )
    tesseract_cmd: str = Field(
        default="tesseract",
        description=(
            "Command or path to Tesseract executable. Use `tesseract` if in system PATH, or provide full path "
            "for custom installations (e.g., `/usr/local/bin/tesseract`)."
        ),
    )
    path: Optional[str] = Field(
        default=None,
        description=(
            "Path to Tesseract data directory containing language files. If None, uses Tesseract's default "
            "TESSDATA_PREFIX location."
        ),
    )
    psm: Optional[int] = Field(
        default=None,
        description=(
            "Page Segmentation Mode for Tesseract. Values 0-13 control how Tesseract segments the page. "
            "Common values: 3 (auto), 6 (uniform block), 11 (sparse text). If None, uses Tesseract default."
        ),
    )
    model_config = ConfigDict(
        extra="forbid",
    )


class TesseractOcrOptions(OcrOptions):
    """Configuration for Tesseract OCR via Python bindings (tesserocr)."""

    kind: ClassVar[Literal["tesserocr"]] = "tesserocr"
    lang: List[str] = Field(
        default=["fra", "deu", "spa", "eng"],
        description=(
            "List of Tesseract language codes. Use 3-letter ISO 639-2 codes (e.g., `eng`, `fra`, `deu`). "
            "Multiple languages enable multilingual OCR. Requires corresponding Tesseract language data files."
        ),
    )
    path: Optional[str] = Field(
        default=None,
        description=(
            "Path to Tesseract data directory containing language files. If None, uses Tesseract's default "
            "TESSDATA_PREFIX location."
        ),
    )
    psm: Optional[int] = Field(
        default=None,
        description=(
            "Page Segmentation Mode for Tesseract. Values 0-13 control how Tesseract segments the page. "
            "Common values: 3 (auto), 6 (uniform block), 11 (sparse text). If None, uses Tesseract default."
        ),
    )
    model_config = ConfigDict(
        extra="forbid",
    )


class OcrMacOptions(OcrOptions):
    """Configuration for native macOS OCR using Vision framework."""

    kind: ClassVar[Literal["ocrmac"]] = "ocrmac"
    lang: List[str] = Field(
        default=["fr-FR", "de-DE", "es-ES", "en-US"],
        description=(
            "List of language locale codes for macOS OCR. Use format `language-REGION` (e.g., `en-US`, `fr-FR`). "
            "Leverages native macOS Vision framework for OCR on Apple platforms."
        ),
    )
    recognition: str = Field(
        default="accurate",
        description=(
            "Recognition accuracy level. Options: `accurate` (higher quality, slower) or `fast` (lower quality, "
            "faster). Choose based on speed vs. accuracy requirements."
        ),
    )
    framework: str = Field(
        default="vision",
        description=(
            "macOS framework to use for OCR. Currently supports `vision` (Apple Vision framework). "
            "Future versions may support additional frameworks."
        ),
    )
    model_config = ConfigDict(
        extra="forbid",
    )


class PictureDescriptionBaseOptions(BaseOptions):
    """Base configuration for picture description models."""

    batch_size: int = Field(
        default=8,
        description=(
            "Number of images to process in a single batch during picture description. Higher values improve "
            "throughput but increase memory usage. Adjust based on available GPU/CPU memory."
        ),
    )
    scale: float = Field(
        default=2.0,
        description=(
            "Scaling factor for image resolution before processing. Higher values (e.g., 2.0) provide more detail "
            "for the vision model but increase processing time and memory. Range: 0.5-4.0 typical."
        ),
    )
    picture_area_threshold: float = Field(
        default=0.05,
        description=(
            "Minimum picture area as fraction of page area (0.0-1.0) to trigger description. Pictures smaller than "
            "this threshold are skipped. Use lower values (e.g., 0.01) to describe small images."
        ),
    )
    classification_allow: Optional[list[PictureClassificationLabel]] = Field(
        default=None,
        description=(
            "List of picture classification labels to allow for description. Only pictures classified with these "
            "labels will be processed. If None, all picture types are allowed unless explicitly denied. Use to "
            "focus description on specific image types (e.g., diagrams, charts)."
        ),
    )
    classification_deny: Optional[list[PictureClassificationLabel]] = Field(
        default=None,
        description=(
            "List of picture classification labels to exclude from description. Pictures classified with these "
            "labels will be skipped. If None, no picture types are denied unless not in allow list. Use to "
            "exclude unwanted image types (e.g., decorative images, logos)."
        ),
    )
    classification_min_confidence: float = Field(
        default=0.0,
        description=(
            "Minimum classification confidence score (0.0-1.0) required for a picture to be processed. Pictures "
            "with classification confidence below this threshold are skipped. Higher values ensure only "
            "confidently classified images are described. Range: 0.0 (no filtering) to 1.0 (maximum confidence)."
        ),
    )


class PictureDescriptionApiOptions(PictureDescriptionBaseOptions):
    """Configuration for API-based picture description services."""

    kind: ClassVar[Literal["api"]] = "api"
    url: AnyUrl = Field(
        default=AnyUrl("http://localhost:8000/v1/chat/completions"),
        description=(
            "API endpoint URL for picture description service. Must be OpenAI-compatible chat completions endpoint. "
            "Default points to local server; update for cloud services or custom deployments."
        ),
    )
    headers: Dict[str, str] = Field(
        default={},
        description=(
            "HTTP headers to include in API requests. Use for authentication or custom headers required by your API "
            "service."
        ),
        examples=[{"Authorization": "Bearer TOKEN"}],
    )
    params: Dict[str, Any] = Field(
        default={},
        description=(
            "Additional query parameters to include in API requests. Service-specific parameters for customizing "
            "API behavior beyond standard options."
        ),
    )
    timeout: float = Field(
        default=20.0,
        description=(
            "Maximum time in seconds to wait for API response before timing out. Increase for slow networks or "
            "complex image descriptions. Recommended: 10-60 seconds."
        ),
    )
    concurrency: int = Field(
        default=1,
        description=(
            "Number of concurrent API requests allowed. Higher values improve throughput but may hit API rate limits. "
            "Adjust based on API service quotas and network capacity."
        ),
    )
    prompt: str = Field(
        default="Describe this image in a few sentences.",
        description=(
            "Prompt template sent to the vision model for image description. Customize to guide the model's output "
            "style, detail level, or focus."
        ),
        examples=["Provide a technical description of this diagram"],
    )
    provenance: str = Field(
        default="",
        description=(
            "Provenance information to track the source or method of picture descriptions. Used for metadata "
            "and auditing purposes in the output document."
        ),
    )


class PictureDescriptionVlmOptions(PictureDescriptionBaseOptions):
    """Configuration for inline vision-language models for picture description."""

    kind: ClassVar[Literal["vlm"]] = "vlm"
    repo_id: Annotated[
        str,
        Field(
            description=(
                "HuggingFace model repository ID for the vision-language model. "
                "Must be a model capable of image-to-text generation for picture descriptions."
            ),
            examples=[
                "HuggingFaceTB/SmolVLM-256M-Instruct",
                "ibm-granite/granite-vision-3.3-2b",
            ],
        ),
    ]
    prompt: str = Field(
        default="Describe this image in a few sentences.",
        description=(
            "Prompt template for the vision model. Customize to control description style, detail level, or focus."
        ),
        examples=[
            "What is shown in this image?",
            "Provide a detailed technical description",
        ],
    )
    generation_config: Dict[str, Any] = Field(
        default=dict(max_new_tokens=200, do_sample=False),
        description=(
            "HuggingFace generation configuration for text generation. Controls output length, sampling strategy, "
            "temperature, etc. See: "
            "https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig"
        ),
    )

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


# SmolVLM
smolvlm_picture_description = PictureDescriptionVlmOptions(
    repo_id="HuggingFaceTB/SmolVLM-256M-Instruct"
)
"""Pre-configured SmolVLM model options for picture description.

Uses the HuggingFace SmolVLM-256M-Instruct model, a lightweight vision-language model
optimized for generating natural language descriptions of images.
"""

# GraniteVision
granite_picture_description = PictureDescriptionVlmOptions(
    repo_id="ibm-granite/granite-vision-3.3-2b",
    prompt="What is shown in this image?",
)
"""Pre-configured Granite Vision model options for picture description.

Uses IBM's Granite Vision 3.3-2B model with a custom prompt for generating
detailed descriptions of image content.
"""


# Define an enum for the backend options
class PdfBackend(str, Enum):
    """Available PDF parsing backends for document processing.

    Different backends offer varying levels of text extraction quality, layout preservation,
    and processing speed. Choose based on your document complexity and quality requirements.

    Attributes:
        PYPDFIUM2: Standard PDF parser using PyPDFium2 library. Fast and reliable for basic text extraction.
        DLPARSE_V1: Docling Parse v1 backend with enhanced layout analysis and structure preservation.
        DLPARSE_V2: Docling Parse v2 backend with improved table detection and complex layout handling.
        DLPARSE_V4: Docling Parse v4 backend (latest) with advanced features and best accuracy for complex documents.
    """

    PYPDFIUM2 = "pypdfium2"
    DLPARSE_V1 = "dlparse_v1"
    DLPARSE_V2 = "dlparse_v2"
    DLPARSE_V4 = "dlparse_v4"


# Define an enum for the ocr engines
@deprecated(
    "Use get_ocr_factory().registered_kind to get a list of registered OCR engines."
)
class OcrEngine(str, Enum):
    """Available OCR (Optical Character Recognition) engines for text extraction from images.

    Each engine has different characteristics in terms of accuracy, speed, language support,
    and platform compatibility. Choose based on your specific requirements.

    Attributes:
        AUTO: Automatically select the best available OCR engine based on platform and installed libraries.
        EASYOCR: Deep learning-based OCR supporting 80+ languages with GPU acceleration.
        TESSERACT_CLI: Tesseract OCR via command-line interface (requires system installation).
        TESSERACT: Tesseract OCR via Python bindings (tesserocr library).
        OCRMAC: Native macOS Vision framework OCR (Apple platforms only).
        RAPIDOCR: Lightweight OCR with multiple backend options (ONNX, OpenVINO, PaddlePaddle).
    """

    AUTO = "auto"
    EASYOCR = "easyocr"
    TESSERACT_CLI = "tesseract_cli"
    TESSERACT = "tesseract"
    OCRMAC = "ocrmac"
    RAPIDOCR = "rapidocr"


class PipelineOptions(BaseOptions):
    """Base configuration for document processing pipelines."""

    document_timeout: Optional[float] = Field(
        default=None,
        description=(
            "Maximum processing time in seconds before aborting document conversion. When exceeded, the pipeline "
            "stops processing and returns partial results with PARTIAL_SUCCESS status. If None, no timeout is "
            "enforced. Recommended: 90-120 seconds for production systems."
        ),
        examples=[10.0, 20.0],
    )
    accelerator_options: AcceleratorOptions = Field(
        default=AcceleratorOptions(),
        description=(
            "Hardware acceleration configuration for model inference. Controls GPU device selection, memory "
            "management, and execution optimization settings for layout, OCR, and table structure models."
        ),
    )
    enable_remote_services: bool = Field(
        default=False,
        description=(
            "Allow pipeline to call external APIs or cloud services during processing. Required for API-based "
            "picture description models. Disabled by default for security and offline operation."
        ),
        examples=[False],
    )
    allow_external_plugins: bool = Field(
        default=False,
        description=(
            "Allow loading external third-party plugins for OCR, layout, table structure, or picture description "
            "models. Enables custom model implementations via plugin system. Disabled by default for security."
        ),
        examples=[False],
    )
    artifacts_path: Optional[Union[Path, str]] = Field(
        default=None,
        description=(
            "Local directory containing pre-downloaded model artifacts (weights, configs). If None, models are "
            "fetched from remote sources on first use. Use `docling-tools models download` to pre-fetch artifacts "
            "for offline operation or faster initialization."
        ),
        examples=["./artifacts", "/tmp/docling_outputs"],
    )


class ConvertPipelineOptions(PipelineOptions):
    """Base configuration for document conversion pipelines."""

    do_picture_classification: bool = Field(
        default=False,
        description=(
            "Enable picture classification to categorize images by type (photo, diagram, chart, etc.). "
            "Useful for downstream processing that requires image type awareness."
        ),
    )
    do_picture_description: bool = Field(
        default=False,
        description=(
            "Enable automatic generation of textual descriptions for pictures using vision-language models. "
            "Descriptions are added to the document for accessibility and searchability."
        ),
    )
    picture_description_options: PictureDescriptionBaseOptions = Field(
        default=smolvlm_picture_description,
        description=(
            "Configuration for picture description model. Specifies which vision model to use (API or inline) "
            "and model-specific parameters. Only applicable when `do_picture_description=True`."
        ),
    )


class PaginatedPipelineOptions(ConvertPipelineOptions):
    """Configuration for pipelines processing paginated documents."""

    images_scale: float = Field(
        default=1.0,
        description=(
            "Scaling factor for generated images. Higher values produce higher resolution but increase processing time "
            "and storage requirements. Recommended values: 1.0 (standard quality), 2.0 (high resolution), 0.5 (lower "
            "resolution for previews)."
        ),
    )
    generate_page_images: bool = Field(
        default=False,
        description=(
            "Generate rendered page images during extraction. Creates PNG representations of each page for visual "
            "preview, validation, or downstream image-based machine learning tasks."
        ),
    )
    generate_picture_images: bool = Field(
        default=False,
        description=(
            "Extract and save embedded images from the document. Exports individual images (figures, photos, diagrams, "
            "charts) found in the document as separate image files for downstream use."
        ),
    )


class VlmPipelineOptions(PaginatedPipelineOptions):
    """Pipeline configuration for vision-language model based document processing."""

    generate_page_images: bool = Field(
        default=True,
        description=(
            "Generate page images for VLM processing. Required for vision-language models to analyze document pages. "
            "Automatically enabled in VLM pipeline."
        ),
    )
    force_backend_text: bool = Field(
        default=False,
        description=(
            "Force use of backend's native text extraction instead of VLM predictions. When enabled, bypasses VLM "
            "text detection and uses embedded text from the document directly."
        ),
    )
    vlm_options: Union[InlineVlmOptions, ApiVlmOptions] = Field(
        default=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
        description=(
            "Vision-Language Model configuration for document understanding. Specifies which VLM to use (inline or "
            "API) and model-specific parameters for vision-based document processing."
        ),
    )


class BaseLayoutOptions(BaseOptions):
    """Base options for layout models."""

    keep_empty_clusters: bool = Field(
        default=False,
        description=(
            "Retain empty clusters in layout analysis results. When False, clusters without content are removed. "
            "Enable for debugging or when empty regions are semantically important."
        ),
    )
    skip_cell_assignment: bool = Field(
        default=False,
        description=(
            "Skip assignment of cells to table structures during layout analysis. When True, cells are detected "
            "but not associated with tables. Use for performance optimization when table structure is not needed."
        ),
    )


class LayoutOptions(BaseLayoutOptions):
    """Options for layout processing."""

    kind: ClassVar[str] = "docling_layout_default"
    create_orphan_clusters: bool = Field(
        default=True,
        description=(
            "Create clusters for orphaned elements not assigned to any structure. When True, isolated text or "
            "elements are grouped into their own clusters. Recommended for complete document coverage."
        ),
    )
    model_spec: LayoutModelConfig = Field(
        default=DOCLING_LAYOUT_HERON,
        description=(
            "Layout model configuration specifying which model to use for document layout analysis. Options include "
            "DOCLING_LAYOUT_HERON (default, balanced), DOCLING_LAYOUT_EGRET_* (higher accuracy), etc."
        ),
    )


class AsrPipelineOptions(PipelineOptions):
    """Configuration options for the Automatic Speech Recognition (ASR) pipeline.

    This pipeline processes audio files and converts speech to text using Whisper-based models.
    Supports various audio formats (MP3, WAV, FLAC, etc.) and video files with audio tracks.
    """

    asr_options: InlineAsrOptions = Field(
        default=asr_model_specs.WHISPER_TINY,
        description=(
            "Automatic Speech Recognition (ASR) model configuration for audio transcription. Specifies which "
            "ASR model to use (e.g., Whisper variants) and model-specific parameters for speech-to-text conversion."
        ),
    )


class VlmExtractionPipelineOptions(PipelineOptions):
    """Options for extraction pipeline."""

    vlm_options: InlineVlmOptions = Field(
        default=NU_EXTRACT_2B_TRANSFORMERS,
        description=(
            "Vision-Language Model (VLM) configuration for structured information extraction. Specifies which VLM "
            "to use and its parameters for extracting structured data from documents using vision models."
        ),
    )


class PdfPipelineOptions(PaginatedPipelineOptions):
    """Configuration options for the PDF document processing pipeline.

    Notes:
        - Enabling multiple features (OCR, table structure, formulas) increases the processing time significantly.
            Enable only necessary features for your use case.
        - For production systems processing large document volumes, implement a timeout protection (for instance, 90-120
            seconds via `document_timeout` parameter).
        - OCR requires a system installation of engines (Tesseract, EasyOCR). Verify the installation before enabling
            OCR via `do_ocr=True`.
        - RapidOCR has known issues with read-only filesystems (e.g., Databricks). Consider Tesseract or alternative
            backends for distributed systems.

    See Also:
        - `examples/pipeline_options_advanced.py`: Comprehensive configuration examples.
    """

    do_table_structure: bool = Field(
        default=True,
        description=(
            "Enable table structure extraction and reconstruction. Detects table regions, extracts cell content with "
            "row/column relationships, and reconstructs the logical table structure for downstream processing."
        ),
    )
    do_ocr: bool = Field(
        default=True,
        description=(
            "Enable Optical Character Recognition for scanned or image-based PDFs. Replaces or supplements "
            "programmatic text extraction with OCR-detected text. Required for scanned documents with no embedded "
            "text layer. Note: OCR significantly increases processing time."
        ),
    )
    do_code_enrichment: bool = Field(
        default=False,
        description=(
            "Enable specialized processing for code blocks. Applies code-aware OCR and formatting to improve accuracy "
            "of programming language snippets, terminal output, and structured code content."
        ),
    )
    do_formula_enrichment: bool = Field(
        default=False,
        description=(
            "Enable mathematical formula recognition and LaTeX conversion. Uses specialized models to detect and "
            "extract mathematical expressions, converting them to LaTeX format for accurate representation."
        ),
    )
    force_backend_text: bool = Field(
        default=False,
        description=(
            "Force use of PDF backend's native text extraction instead of layout model predictions. When enabled, "
            "bypasses the layout model's text detection and uses the embedded text from the PDF file directly. Useful "
            "for PDFs with reliable programmatic text layers."
        ),
    )
    table_structure_options: BaseTableStructureOptions = Field(
        default=TableStructureOptions(),
        description=(
            "Configuration for table structure extraction. Controls table detection accuracy, cell matching behavior, "
            "and table formatting. Only applicable when `do_table_structure=True`."
        ),
    )
    ocr_options: OcrOptions = Field(
        default=OcrAutoOptions(),
        description=(
            "Configuration for OCR engine. Specifies which OCR engine to use (Tesseract, EasyOCR, RapidOCR, etc.) "
            "and engine-specific settings. Only applicable when `do_ocr=True`."
        ),
    )
    layout_options: BaseLayoutOptions = Field(
        default=LayoutOptions(),
        description=(
            "Configuration for document layout analysis model. Controls layout detection behavior including cluster "
            "creation for orphaned elements, cell assignment to table structures, and handling of empty regions. "
            "Specifies which layout model to use (default: Heron)."
        ),
    )
    images_scale: float = Field(
        default=1.0,
        description=(
            "Scaling factor for generated images. Higher values produce higher resolution but increase processing time "
            "and storage requirements. Recommended values: 1.0 (standard quality), 2.0 (high resolution), 0.5 (lower "
            "resolution for previews)."
        ),
    )
    generate_page_images: bool = Field(
        default=False,
        description=(
            "Generate rendered page images during extraction. Creates PNG representations of each page for visual "
            "preview, validation, or downstream image-based machine learning tasks."
        ),
    )
    generate_picture_images: bool = Field(
        default=False,
        description=(
            "Extract and save embedded images from the PDF. Exports individual images (figures, photos, diagrams, "
            "charts) found in the document as separate image files for downstream use."
        ),
    )
    generate_table_images: bool = Field(
        default=False,
        deprecated=(
            "This field is deprecated. Use `generate_page_images=True` and call `TableItem.get_image()` to extract "
            "table images from page images."
        ),
    )
    generate_parsed_pages: bool = Field(
        default=False,
        description=(
            "Retain intermediate parsed page representations after processing. When enabled, keeps detailed page-level "
            "parsing data structures for debugging or advanced post-processing. Increases memory usage. Automatically "
            "disabled after document assembly unless explicitly enabled."
        ),
    )

    ### Arguments for threaded PDF pipeline with batching and backpressure control

    # Batch sizes for different stages
    ocr_batch_size: int = Field(
        default=4,
        description=(
            "Batch size for OCR processing stage in threaded pipeline. Pages are grouped and processed together to "
            "improve throughput. Higher values increase GPU/CPU utilization but require more memory. Only used by "
            "`StandardPdfPipeline` (threaded mode)."
        ),
    )
    layout_batch_size: int = Field(
        default=4,
        description=(
            "Batch size for layout analysis stage in threaded pipeline. Pages are grouped and processed together by "
            "the layout model. Higher values improve throughput but increase memory usage. Only used by "
            "`StandardPdfPipeline` (threaded mode)."
        ),
    )
    table_batch_size: int = Field(
        default=4,
        description=(
            "Batch size for table structure extraction stage in threaded pipeline. Tables from multiple pages are "
            "processed together. Higher values improve throughput but increase memory usage. Only used by "
            "`StandardPdfPipeline` (threaded mode)."
        ),
    )

    # Timing control
    batch_polling_interval_seconds: float = Field(
        default=0.5,
        description=(
            "Polling interval in seconds for batch collection in threaded pipeline stages. Each stage waits up to "
            "this duration to accumulate items before processing. Lower values reduce latency but may decrease "
            "batching efficiency. Only used by `StandardPdfPipeline` (threaded mode)."
        ),
    )
    # Backpressure and queue control
    queue_max_size: int = Field(
        default=100,
        description=(
            "Maximum queue size for inter-stage communication in threaded pipeline. Limits the number of items "
            "buffered between processing stages to prevent memory overflow. When full, upstream stages block until "
            "space is available. Only used by `StandardPdfPipeline` (threaded mode)."
        ),
    )


class ProcessingPipeline(str, Enum):
    """Available document processing pipeline types for different use cases.

    Each pipeline is optimized for specific document types and processing requirements.
    Select the appropriate pipeline based on your input format and desired output.

    Attributes:
        LEGACY: Legacy pipeline for backward compatibility with older document processing workflows.
        STANDARD: Standard pipeline for general document processing (PDF, DOCX, images, etc.) with layout analysis.
        VLM: Vision-Language Model pipeline for advanced document understanding using multimodal AI models.
        ASR: Automatic Speech Recognition pipeline for audio and video transcription to text.
    """

    LEGACY = "legacy"
    STANDARD = "standard"
    VLM = "vlm"
    ASR = "asr"


class ThreadedPdfPipelineOptions(PdfPipelineOptions):
    """Pipeline options for the threaded PDF pipeline with batching and backpressure control"""
