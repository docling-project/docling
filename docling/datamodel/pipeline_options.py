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
    """Modes for the TableFormer model."""

    FAST = "fast"
    ACCURATE = "accurate"


class BaseTableStructureOptions(BaseOptions):
    """Base options for table structure models."""


class TableStructureOptions(BaseTableStructureOptions):
    """Options for the table structure."""

    kind: ClassVar[str] = "docling_tableformer"
    do_cell_matching: bool = (
        True
        # True:  Matches predictions back to PDF cells. Can break table output if PDF cells
        #        are merged across table columns.
        # False: Let table structure model define the text cells, ignore PDF cells.
    )
    mode: TableFormerMode = TableFormerMode.ACCURATE


class OcrOptions(BaseOptions):
    """OCR options."""

    lang: Annotated[
        List[str],
        Field(
            description="List of OCR languages to use. The format must match the values of the OCR engine of choice.",
            examples=[["deu", "eng"]],
        ),
    ]

    force_full_page_ocr: Annotated[
        bool,
        Field(
            description="If enabled, a full-page OCR is always applied.",
            examples=[False],
        ),
    ] = False

    bitmap_area_threshold: Annotated[
        float,
        Field(
            description="Percentage of the page area for a bitmap to be processed with OCR.",
            examples=[0.05, 0.1],
        ),
    ] = 0.05


class OcrAutoOptions(OcrOptions):
    """Options for pick OCR engine automatically."""

    kind: ClassVar[Literal["auto"]] = "auto"
    lang: Annotated[
        List[str],
        Field(
            description="The automatic OCR engine will use the default values of the engine. Please specify the engine explicitly to change the language selection.",
        ),
    ] = []


class RapidOcrOptions(OcrOptions):
    """Options for the RapidOCR engine."""

    kind: ClassVar[Literal["rapidocr"]] = "rapidocr"

    # English and chinese are the most commly used models and have been tested with RapidOCR.
    lang: List[str] = [
        "english",
        "chinese",
    ]
    # However, language as a parameter is not supported by rapidocr yet
    # and hence changing this options doesn't affect anything.

    # For more details on supported languages by RapidOCR visit
    # https://rapidai.github.io/RapidOCRDocs/blog/2022/09/28/%E6%94%AF%E6%8C%81%E8%AF%86%E5%88%AB%E8%AF%AD%E8%A8%80/

    # For more details on the following options visit
    # https://rapidai.github.io/RapidOCRDocs/install_usage/api/RapidOCR/

    # https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/#__tabbed_3_4
    backend: Literal["onnxruntime", "openvino", "paddle", "torch"] = "onnxruntime"
    text_score: float = 0.5  # same default as rapidocr

    use_det: Optional[bool] = None  # same default as rapidocr
    use_cls: Optional[bool] = None  # same default as rapidocr
    use_rec: Optional[bool] = None  # same default as rapidocr

    print_verbose: bool = False  # same default as rapidocr

    det_model_path: Optional[str] = None  # same default as rapidocr
    cls_model_path: Optional[str] = None  # same default as rapidocr
    rec_model_path: Optional[str] = None  # same default as rapidocr
    rec_keys_path: Optional[str] = None  # same default as rapidocr
    rec_font_path: Optional[str] = None  # Deprecated, please use font_path instead
    font_path: Optional[str] = None  # same default as rapidocr

    # Dictionary to overwrite or pass-through additional parameters
    rapidocr_params: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        extra="forbid",
    )


class EasyOcrOptions(OcrOptions):
    """Options for the EasyOCR engine."""

    kind: ClassVar[Literal["easyocr"]] = "easyocr"
    lang: List[str] = ["fr", "de", "es", "en"]

    use_gpu: Optional[bool] = None

    confidence_threshold: float = 0.5

    model_storage_directory: Optional[str] = None
    recog_network: Optional[str] = "standard"
    download_enabled: bool = True

    suppress_mps_warnings: bool = True

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )


class TesseractCliOcrOptions(OcrOptions):
    """Options for the TesseractCli engine."""

    kind: ClassVar[Literal["tesseract"]] = "tesseract"
    lang: List[str] = ["fra", "deu", "spa", "eng"]
    tesseract_cmd: str = "tesseract"
    path: Optional[str] = None
    psm: Optional[int] = (
        None  # Page Segmentation Mode (0-13), defaults to tesseract's default
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class TesseractOcrOptions(OcrOptions):
    """Options for the Tesseract engine."""

    kind: ClassVar[Literal["tesserocr"]] = "tesserocr"
    lang: List[str] = ["fra", "deu", "spa", "eng"]
    path: Optional[str] = None
    psm: Optional[int] = (
        None  # Page Segmentation Mode (0-13), defaults to tesseract's default
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class OcrMacOptions(OcrOptions):
    """Options for the Mac OCR engine."""

    kind: ClassVar[Literal["ocrmac"]] = "ocrmac"
    lang: List[str] = ["fr-FR", "de-DE", "es-ES", "en-US"]
    recognition: str = "accurate"
    framework: str = "vision"

    model_config = ConfigDict(
        extra="forbid",
    )


class PictureDescriptionBaseOptions(BaseOptions):
    batch_size: int = 8
    scale: float = 2

    picture_area_threshold: float = (
        0.05  # percentage of the area for a picture to processed with the models
    )
    classification_allow: Optional[List[PictureClassificationLabel]] = None
    classification_deny: Optional[List[PictureClassificationLabel]] = None
    classification_min_confidence: float = 0.0


class PictureDescriptionApiOptions(PictureDescriptionBaseOptions):
    kind: ClassVar[Literal["api"]] = "api"

    url: AnyUrl = AnyUrl("http://localhost:8000/v1/chat/completions")
    headers: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    timeout: float = 20
    concurrency: int = 1

    prompt: str = "Describe this image in a few sentences."
    provenance: str = ""


class PictureDescriptionVlmOptions(PictureDescriptionBaseOptions):
    kind: ClassVar[Literal["vlm"]] = "vlm"

    repo_id: str
    prompt: str = "Describe this image in a few sentences."
    # Config from here https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
    generation_config: Dict[str, Any] = dict(max_new_tokens=200, do_sample=False)

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


# SmolVLM
smolvlm_picture_description = PictureDescriptionVlmOptions(
    repo_id="HuggingFaceTB/SmolVLM-256M-Instruct"
)

# GraniteVision
granite_picture_description = PictureDescriptionVlmOptions(
    repo_id="ibm-granite/granite-vision-3.3-2b",
    prompt="What is shown in this image?",
)


# Define an enum for the backend options
class PdfBackend(str, Enum):
    """Enum of valid PDF backends."""

    PYPDFIUM2 = "pypdfium2"
    DLPARSE_V1 = "dlparse_v1"
    DLPARSE_V2 = "dlparse_v2"
    DLPARSE_V4 = "dlparse_v4"


# Define an enum for the ocr engines
@deprecated(
    "Use get_ocr_factory().registered_kind to get a list of registered OCR engines."
)
class OcrEngine(str, Enum):
    """Enum of valid OCR engines."""

    AUTO = "auto"
    EASYOCR = "easyocr"
    TESSERACT_CLI = "tesseract_cli"
    TESSERACT = "tesseract"
    OCRMAC = "ocrmac"
    RAPIDOCR = "rapidocr"


class PipelineOptions(BaseOptions):
    """Base pipeline options."""

    document_timeout: Annotated[
        Optional[float],
        Field(
            description="Maximum allowed processing time for a document before timing out. If None, no timeout is enforced.",
            examples=[10.0, 20.0],
        ),
    ] = None

    accelerator_options: Annotated[
        AcceleratorOptions,
        Field(
            description="Configuration options for hardware acceleration (e.g., GPU or optimized execution settings).",
        ),
    ] = AcceleratorOptions()

    enable_remote_services: Annotated[
        bool,
        Field(
            description="Enable calling external APIs or cloud services during pipeline execution.",
            examples=[False],
        ),
    ] = False

    allow_external_plugins: Annotated[
        bool,
        Field(
            description="Allow loading external third-party plugins or modules. Disabled by default for safety.",
            examples=[False],
        ),
    ] = False

    artifacts_path: Annotated[
        Optional[Union[Path, str]],
        Field(
            description="Filesystem path where pipeline artifacts should be stored. If None, artifacts will be fetched. You can use the utility `docling-tools models download` to pre-fetch the model artifacts.",
            examples=["./artifacts", "/tmp/docling_outputs"],
        ),
    ] = None


class ConvertPipelineOptions(PipelineOptions):
    """Base convert pipeline options."""

    do_picture_classification: bool = False  # True: classify pictures in documents

    do_picture_description: bool = False  # True: run describe pictures in documents
    picture_description_options: PictureDescriptionBaseOptions = (
        smolvlm_picture_description
    )


class PaginatedPipelineOptions(ConvertPipelineOptions):
    images_scale: float = 1.0
    generate_page_images: bool = False
    generate_picture_images: bool = False


class VlmPipelineOptions(PaginatedPipelineOptions):
    generate_page_images: bool = True
    force_backend_text: bool = (
        False  # (To be used with vlms, or other generative models)
    )
    # If True, text from backend will be used instead of generated text
    vlm_options: Union[InlineVlmOptions, ApiVlmOptions] = (
        vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
    )


class BaseLayoutOptions(BaseOptions):
    """Base options for layout models."""

    keep_empty_clusters: bool = (
        False  # Whether to keep clusters that contain no text cells
    )
    skip_cell_assignment: bool = (
        False  # Skip cell-to-cluster assignment for VLM-only processing
    )


class LayoutOptions(BaseLayoutOptions):
    """Options for layout processing."""

    kind: ClassVar[str] = "docling_layout_default"
    create_orphan_clusters: bool = True  # Whether to create clusters for orphaned cells
    model_spec: LayoutModelConfig = DOCLING_LAYOUT_HERON


class AsrPipelineOptions(PipelineOptions):
    asr_options: Union[InlineAsrOptions] = asr_model_specs.WHISPER_TINY


class VlmExtractionPipelineOptions(PipelineOptions):
    """Options for extraction pipeline."""

    vlm_options: Union[InlineVlmOptions] = NU_EXTRACT_2B_TRANSFORMERS


class PdfPipelineOptions(PaginatedPipelineOptions):
    """
    Configuration options for PDF document processing pipeline.

    Controls the behavior of document extraction including table structure recognition,
    OCR processing, image generation, and formula extraction. These options determine
    which processing steps are applied and how resources are allocated.

    Attributes:
        do_table_structure: Enable table structure extraction. Identifies table boundaries,
            extracts cell content, and reconstructs table structure with rows/columns.
            Default: True.

        do_ocr: Enable Optical Character Recognition for scanned or image-based PDFs.
            Replaces or supplements programmatic text extraction with OCR-detected text.
            Required for scanned documents with no embedded text layer. Note: OCR
            significantly increases processing time. Default: True.

        do_code_enrichment: Enable specialized OCR for code blocks. Applies code-specific
            recognition to improve accuracy of programming language snippets, terminal
            output, and structured code. Default: False.

        do_formula_enrichment: Enable mathematical formula extraction with LaTeX output.
            Detects mathematical expressions and converts them to LaTeX format.
            Default: False.

        generate_page_images: Generate rendered page images during extraction. Creates
            PNG representations of each page for preview, validation, or image-based
            ML tasks. Inherited from PaginatedPipelineOptions. Default: False.

        generate_picture_images: Extract embedded images from the PDF. Saves individual
            images (figures, photos, diagrams) found in the document as separate files.
            Inherited from PaginatedPipelineOptions. Default: False.

        images_scale: Scaling factor for generated images. Higher values produce higher
            resolution but increase processing time and storage. Recommended values:
            1.0 (standard), 2.0 (high resolution), 0.5 (lower resolution preview).
            Inherited from PaginatedPipelineOptions. Default: 1.0.

        ocr_options: Configuration for OCR engine. Specifies which OCR engine to use
            (Tesseract, EasyOCR, RapidOCR, etc.) and engine-specific settings.
            Only applicable when do_ocr=True. Default: None (auto-selects engine).

        table_structure_options: Configuration for table structure extraction. Controls
            table detection accuracy, cell matching behavior, and table formatting.
            Only applicable when do_table_structure=True. Default: None (uses defaults).

    Notes:
        - Enabling multiple features (OCR, table structure, formulas) increases processing
          time significantly. Enable only necessary features for your use case.
        - For production systems processing large document volumes, implement timeout
          protection (90-120 seconds recommended via document_timeout parameter).
        - OCR requires system installation of engines (Tesseract, EasyOCR). Verify
          installation before enabling do_ocr=True.
        - RapidOCR has known issues with read-only filesystems (e.g., Databricks).
          Consider Tesseract or alternative backends for distributed systems.

    Examples:
        Basic digital PDF extraction (no OCR)::

            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat

            # Fast extraction for digital PDFs
            options = PdfPipelineOptions()
            options.do_ocr = False
            options.do_table_structure = True

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=options)
                }
            )
            result = converter.convert("document.pdf")

        Scanned PDF with OCR::

            from docling.datamodel.pipeline_options import TesseractOcrOptions

            options = PdfPipelineOptions()
            options.do_ocr = True
            options.ocr_options = TesseractOcrOptions()
            options.generate_page_images = True
            options.images_scale = 2.0

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=options)
                }
            )

        Scientific paper with tables and formulas::

            options = PdfPipelineOptions()
            options.do_ocr = False  # Digital PDF
            options.do_table_structure = True
            options.do_formula_enrichment = True
            options.table_structure_options = TableStructureOptions()
            options.table_structure_options.mode = TableFormerMode.ACCURATE

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=options)
                }
            )

        Databricks-compatible configuration::

            # Avoid RapidOCR filesystem issues in distributed environments
            options = PdfPipelineOptions()
            options.do_ocr = False  # Disabled for read-only site-packages
            options.do_table_structure = True
            options.generate_page_images = True
            options.generate_picture_images = True

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=options)
                }
            )

    See Also:
        PaginatedPipelineOptions: Base class with image generation options
        OcrOptions: OCR engine configuration base class
        TableStructureOptions: Table extraction configuration
        DocumentConverter: Main conversion interface
    """
    do_table_structure: bool = True  # True: perform table structure extraction
    do_ocr: bool = True  # True: perform OCR, replace programmatic PDF text
    do_code_enrichment: bool = False  # True: perform code OCR
    do_formula_enrichment: bool = False  # True: perform formula OCR, return Latex code
    force_backend_text: bool = (
        False  # (To be used with vlms, or other generative models)
    )
    # If True, text from backend will be used instead of generated text

    table_structure_options: BaseTableStructureOptions = TableStructureOptions()
    ocr_options: OcrOptions = OcrAutoOptions()
    layout_options: BaseLayoutOptions = LayoutOptions()

    images_scale: float = 1.0
    generate_page_images: bool = False
    generate_picture_images: bool = False
    generate_table_images: bool = Field(
        default=False,
        deprecated=(
            "Field `generate_table_images` is deprecated. "
            "To obtain table images, set `PdfPipelineOptions.generate_page_images = True` "
            "before conversion and then use the `TableItem.get_image` function."
        ),
    )

    generate_parsed_pages: bool = False

    ### Arguments for threaded PDF pipeline with batching and backpressure control

    # Batch sizes for different stages
    ocr_batch_size: int = 4
    layout_batch_size: int = 4
    table_batch_size: int = 4

    # Timing control
    batch_polling_interval_seconds: float = 0.5

    # Backpressure and queue control
    queue_max_size: int = 100


class ProcessingPipeline(str, Enum):
    LEGACY = "legacy"
    STANDARD = "standard"
    VLM = "vlm"
    ASR = "asr"


class ThreadedPdfPipelineOptions(PdfPipelineOptions):
    """Pipeline options for the threaded PDF pipeline with batching and backpressure control"""
