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

from docling.datamodel import asr_model_specs

# Import the following for backwards compatibility





_log = logging.getLogger(__name__)





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





# Define an enum for the backend options
class PdfBackend(str, Enum):
    """Enum of valid PDF backends."""

    PYPDFIUM2 = "pypdfium2"
    DLPARSE_V1 = "dlparse_v1"
    DLPARSE_V2 = "dlparse_v2"
    DLPARSE_V4 = "dlparse_v4"


# Define an enum for the ocr engines



class PipelineOptions(BaseModel):
    """Base pipeline options."""

    create_legacy_output: bool = (
        True  # This default will be set to False on a future version of docling
    )
    document_timeout: Optional[float] = None
    accelerator_options: AcceleratorOptions = AcceleratorOptions()
    enable_remote_services: bool = False
    allow_external_plugins: bool = False





class VlmPipelineOptions(PipelineOptions):
    artifacts_path: Optional[Union[Path, str]] = None

    
