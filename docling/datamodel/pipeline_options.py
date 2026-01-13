import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Dict, List, Literal, Optional, Union

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

    do_chart_extraction: bool = (
        False  # True: extract data in tabular format from bar-, pie and line-charts
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
    """Options for the PDF pipeline."""

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
