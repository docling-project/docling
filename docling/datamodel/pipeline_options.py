import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import deprecated

_log = logging.getLogger(__name__)


class AcceleratorDevice(str, Enum):
    """Devices to run model inference"""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class AcceleratorOptions(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCLING_", env_nested_delimiter="_", populate_by_name=True
    )

    num_threads: int = 4
    device: Union[str, AcceleratorDevice] = "auto"
    cuda_use_flash_attention2: bool = False

    @field_validator("device")
    def validate_device(cls, value):
        # "auto", "cpu", "cuda", "mps", or "cuda:N"
        if value in {d.value for d in AcceleratorDevice} or re.match(
            r"^cuda(:\d+)?$", value
        ):
            return value
        raise ValueError(
            "Invalid device option. Use 'auto', 'cpu', 'mps', 'cuda', or 'cuda:N'."
        )

    @model_validator(mode="before")
    @classmethod
    def check_alternative_envvars(cls, data: Any) -> Any:
        r"""
        Set num_threads from the "alternative" envvar OMP_NUM_THREADS.
        The alternative envvar is used only if it is valid and the regular envvar is not set.

        Notice: The standard pydantic settings mechanism with parameter "aliases" does not provide
        the same functionality. In case the alias envvar is set and the user tries to override the
        parameter in settings initialization, Pydantic treats the parameter provided in __init__()
        as an extra input instead of simply overwriting the evvar value for that parameter.
        """
        if isinstance(data, dict):
            input_num_threads = data.get("num_threads")
            # Check if to set the num_threads from the alternative envvar
            if input_num_threads is None:
                docling_num_threads = os.getenv("DOCLING_NUM_THREADS")
                omp_num_threads = os.getenv("OMP_NUM_THREADS")
                if docling_num_threads is None and omp_num_threads is not None:
                    try:
                        data["num_threads"] = int(omp_num_threads)
                    except ValueError:
                        _log.error(
                            "Ignoring misformatted envvar OMP_NUM_THREADS '%s'",
                            omp_num_threads,
                        )
        return data


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
    ]  # However, language as a parameter is not supported by rapidocr yet and hence changing this options doesn't affect anything.
    # For more details on supported languages by RapidOCR visit https://rapidai.github.io/RapidOCRDocs/blog/2022/09/28/%E6%94%AF%E6%8C%81%E8%AF%86%E5%88%AB%E8%AF%AD%E8%A8%80/

    # For more details on the following options visit https://rapidai.github.io/RapidOCRDocs/install_usage/api/RapidOCR/
    text_score: float = 0.5  # same default as rapidocr

    use_det: Optional[bool] = None  # same default as rapidocr
    use_cls: Optional[bool] = None  # same default as rapidocr
    use_rec: Optional[bool] = None  # same default as rapidocr

    # class Device(Enum):
    #     CPU = "CPU"
    #     CUDA = "CUDA"
    #     DIRECTML = "DIRECTML"
    #     AUTO = "AUTO"

    # device: Device = Device.AUTO  # Default value is AUTO

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

    prompt: str = "Describe this image in a few sentences. Extract semantic components from the image (objects, actions, colors, etc.) and generate a star-structure graph with the image as the central node. Link surrounding nodes with labeled relationships to form a unique ontology-style concept map and add this to the description of the image."
    provenance: str = ""


class PictureDescriptionVlmOptions(PictureDescriptionBaseOptions):
    kind: ClassVar[Literal["vlm"]] = "vlm"

    repo_id: str
    prompt: str = "Describe this image in a few sentences. Extract semantic components from the image (objects, actions, colors, etc.) and generate a star-structure graph with the image as the central node. Link surrounding nodes with labeled relationships to form a unique ontology-style concept map and add this to the description of the image."
    # Config from here https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
    generation_config: Dict[str, Any] = dict(max_new_tokens=200, do_sample=False)

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


smolvlm_picture_description = PictureDescriptionVlmOptions(
    repo_id="HuggingFaceTB/SmolVLM-256M-Instruct"
)
# phi_picture_description = PictureDescriptionVlmOptions(repo_id="microsoft/Phi-3-vision-128k-instruct")
granite_picture_description = PictureDescriptionVlmOptions(
    repo_id="ibm-granite/granite-vision-3.1-2b-preview",
    prompt="What is shown in this image? Extract semantic components from the image (objects, actions, colors, etc.) and generate a star-structure graph with the image as the central node. Link surrounding nodes with labeled relationships to form a unique ontology-style concept map and add this to the description of the image.",
)


class BaseVlmOptions(BaseModel):
    kind: str
    prompt: str


class ResponseFormat(str, Enum):
    DOCTAGS = "doctags"
    MARKDOWN = "markdown"


class InferenceFramework(str, Enum):
    MLX = "mlx"
    TRANSFORMERS = "transformers"
    OPENAI = "openai"


class HuggingFaceVlmOptions(BaseVlmOptions):
    kind: Literal["hf_model_options"] = "hf_model_options"

    repo_id: str
    load_in_8bit: bool = True
    llm_int8_threshold: float = 6.0
    quantized: bool = False

    inference_framework: InferenceFramework
    response_format: ResponseFormat

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


class ApiVlmOptions(BaseVlmOptions):
    kind: Literal["api_model_options"] = "api_model_options"

    url: AnyUrl = AnyUrl(
        "http://localhost:11434/v1/chat/completions"
    )  # Default to ollama
    headers: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    scale: float = 2.0
    timeout: float = 60
    response_format: ResponseFormat


smoldocling_vlm_mlx_conversion_options = HuggingFaceVlmOptions(
    repo_id="ds4sd/SmolDocling-256M-preview-mlx-bf16",
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    inference_framework=InferenceFramework.MLX,
)


smoldocling_vlm_conversion_options = HuggingFaceVlmOptions(
    repo_id="ds4sd/SmolDocling-256M-preview",
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    inference_framework=InferenceFramework.TRANSFORMERS,
)

granite_vision_vlm_conversion_options = HuggingFaceVlmOptions(
    repo_id="ibm-granite/granite-vision-3.1-2b-preview",
    # prompt="OCR the full page to markdown.",
    prompt="OCR this image.",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS,
)

granite_vision_vlm_ollama_conversion_options = ApiVlmOptions(
    url=AnyUrl("http://localhost:11434/v1/chat/completions"),
    params={"model": "granite3.2-vision:2b"},
    prompt="OCR the full page to markdown.",
    scale=1.0,
    timeout=120,
    response_format=ResponseFormat.MARKDOWN,
)


class VlmModelType(str, Enum):
    SMOLDOCLING = "smoldocling"
    GRANITE_VISION = "granite_vision"
    GRANITE_VISION_OLLAMA = "granite_vision_ollama"


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
    vlm_options: Union[HuggingFaceVlmOptions, ApiVlmOptions] = (
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
    do_table_description: bool = False # True: run describe tables in documents
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
