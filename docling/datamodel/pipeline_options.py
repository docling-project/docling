import logging
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
)
from typing_extensions import deprecated

# Import the following for backwards compatibility
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.datamodel.vlm_model_specs import (
    GRANITE_VISION_OLLAMA as granite_vision_vlm_ollama_conversion_options,
    GRANITE_VISION_TRANSFORMERS as granite_vision_vlm_conversion_options,
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


class TableStructureOptions(BaseModel):
    """Options for the table structure."""

    do_cell_matching: bool = (
        True
        # True:  Matches predictions back to PDF cells. Can break table output if PDF cells
        #        are merged across table columns.
        # False: Let table structure model define the text cells, ignore PDF cells.
    )
    mode: TableFormerMode = TableFormerMode.ACCURATE


class OcrOptions(BaseOptions):
    """OCR options."""

    lang: List[str]
    force_full_page_ocr: bool = False  # If enabled a full page OCR is always applied
    bitmap_area_threshold: float = (
        0.05  # percentage of the area for a bitmap to processed with OCR
    )


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

    text_score: float = 0.5  # same default as rapidocr

    use_det: Optional[bool] = None  # same default as rapidocr
    use_cls: Optional[bool] = None  # same default as rapidocr
    use_rec: Optional[bool] = None  # same default as rapidocr

    print_verbose: bool = False  # same default as rapidocr

    det_model_path: Optional[str] = None  # same default as rapidocr
    cls_model_path: Optional[str] = None  # same default as rapidocr
    rec_model_path: Optional[str] = None  # same default as rapidocr
    rec_keys_path: Optional[str] = None  # same default as rapidocr

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

    model_config = ConfigDict(
        extra="forbid",
    )


class TesseractOcrOptions(OcrOptions):
    """Options for the Tesseract engine."""

    kind: ClassVar[Literal["tesserocr"]] = "tesserocr"
    lang: List[str] = ["fra", "deu", "spa", "eng"]
    path: Optional[str] = None

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
    repo_id="ibm-granite/granite-vision-3.1-2b-preview",
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
@deprecated("Use ocr_factory.registered_enum")
class OcrEngine(str, Enum):
    """Enum of valid OCR engines."""

    EASYOCR = "easyocr"
    TESSERACT_CLI = "tesseract_cli"
    TESSERACT = "tesseract"
    OCRMAC = "ocrmac"
    RAPIDOCR = "rapidocr"


class PipelineOptions(BaseModel):
    """Base pipeline options."""

    create_legacy_output: bool = (
        True  # This default will be set to False on a future version of docling
    )
    document_timeout: Optional[float] = None
    accelerator_options: AcceleratorOptions = AcceleratorOptions()
    enable_remote_services: bool = False
    allow_external_plugins: bool = False


class PaginatedPipelineOptions(PipelineOptions):
    artifacts_path: Optional[Union[Path, str]] = None

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
        smoldocling_vlm_conversion_options
    )


class PdfPipelineOptions(PaginatedPipelineOptions):
    """Options for the PDF pipeline."""

    do_table_structure: bool = True  # True: perform table structure extraction
    do_ocr: bool = True  # True: perform OCR, replace programmatic PDF text
    do_code_enrichment: bool = False  # True: perform code OCR
    do_formula_enrichment: bool = False  # True: perform formula OCR, return Latex code
    do_picture_classification: bool = False  # True: classify pictures in documents
    do_picture_description: bool = False  # True: run describe pictures in documents
    force_backend_text: bool = (
        False  # (To be used with vlms, or other generative models)
    )
    # If True, text from backend will be used instead of generated text

    table_structure_options: TableStructureOptions = TableStructureOptions()
    ocr_options: OcrOptions = EasyOcrOptions()
    picture_description_options: PictureDescriptionBaseOptions = (
        smolvlm_picture_description
    )

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


class PdfPipeline(str, Enum):
    STANDARD = "standard"
    VLM = "vlm"
