# Define the input options for the API
import warnings
from typing import Annotated, Any, Optional, Union

from docling_core.types.doc import ImageRefMode, PictureClassificationLabel
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat, OutputFormat

# Import new engine system (available in docling>=2.73.0)
from docling.datamodel.pipeline_options import (
    CodeFormulaVlmOptions,
    PdfBackend,
    PictureDescriptionBaseOptions,
    PictureDescriptionVlmEngineOptions,
    ProcessingPipeline,
    TableFormerMode,
    TableStructureOptions,
    VlmConvertOptions,
)

# Import legacy types for backwards compatibility
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
    TransformersModelType,
)
from docling.datamodel.settings import (
    DEFAULT_PAGE_RANGE,
    PageRange,
)


class PictureDescriptionLocal(BaseModel):
    repo_id: Annotated[
        str,
        Field(
            description="Repository id from the Hugging Face Hub.",
            examples=[
                "HuggingFaceTB/SmolVLM-256M-Instruct",
                "ibm-granite/granite-vision-3.3-2b",
            ],
        ),
    ]
    prompt: Annotated[
        str,
        Field(
            description="Prompt used when calling the vision-language model.",
            examples=[
                "Describe this image in a few sentences.",
                "This is a figure from a document. Provide a detailed description of it.",
            ],
        ),
    ] = "Describe this image in a few sentences."
    generation_config: Annotated[
        dict[str, Any],
        Field(
            description="Config from https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig",
            examples=[{"max_new_tokens": 200, "do_sample": False}],
        ),
    ] = {"max_new_tokens": 200, "do_sample": False}
    classification_allow: Annotated[
        Optional[list[PictureClassificationLabel]],
        Field(
            description="Only describe pictures whose predicted class is in this allow-list."
        ),
    ] = None
    classification_deny: Annotated[
        Optional[list[PictureClassificationLabel]],
        Field(
            description="Do not describe pictures whose predicted class is in this deny-list."
        ),
    ] = None
    classification_min_confidence: Annotated[
        float,
        Field(
            description="Minimum classification confidence required before a picture can be described.",
            ge=0.0,
            le=1.0,
        ),
    ] = 0.0


class PictureDescriptionApi(BaseModel):
    url: Annotated[
        AnyUrl,
        Field(
            description="Endpoint which accepts openai-api compatible requests.",
            examples=[
                AnyUrl(
                    "http://localhost:8000/v1/chat/completions"
                ),  # example of a local vllm api
                AnyUrl(
                    "http://localhost:1234/v1/chat/completions"
                ),  # example of lm studio
                AnyUrl(
                    "http://localhost:11434/v1/chat/completions"
                ),  # example of ollama
            ],
        ),
    ]
    headers: Annotated[
        dict[str, str],
        Field(
            description="Headers used for calling the API endpoint. For example, it could include authentication headers."
        ),
    ] = {}
    params: Annotated[
        dict[str, Any],
        Field(
            description="Model parameters.",
            examples=[
                {  # on vllm
                    "model": "HuggingFaceTB/SmolVLM-256M-Instruct",
                    "max_completion_tokens": 200,
                },
                {  # on vllm
                    "model": "ibm-granite/granite-vision-3.3-2b",
                    "max_completion_tokens": 200,
                },
                {  # on ollama
                    "model": "granite3.2-vision:2b"
                },
            ],
        ),
    ] = {}
    timeout: Annotated[float, Field(description="Timeout for the API request.")] = 20
    concurrency: Annotated[
        PositiveInt,
        Field(
            description="Maximum number of concurrent requests to the API.",
            examples=[1],
        ),
    ] = 1
    prompt: Annotated[
        str,
        Field(
            description="Prompt used when calling the vision-language model.",
            examples=[
                "Describe this image in a few sentences.",
                "This is a figures from a document. Provide a detailed description of it.",
            ],
        ),
    ] = "Describe this image in a few sentences."
    classification_allow: Annotated[
        Optional[list[PictureClassificationLabel]],
        Field(
            description="Only describe pictures whose predicted class is in this allow-list."
        ),
    ] = None
    classification_deny: Annotated[
        Optional[list[PictureClassificationLabel]],
        Field(
            description="Do not describe pictures whose predicted class is in this deny-list."
        ),
    ] = None
    classification_min_confidence: Annotated[
        float,
        Field(
            description="Minimum classification confidence required before a picture can be described.",
            ge=0.0,
            le=1.0,
        ),
    ] = 0.0


class VlmModelLocal(BaseModel):
    repo_id: Annotated[
        str,
        Field(
            description="Repository id from the Hugging Face Hub.",
        ),
    ]
    prompt: Annotated[
        str,
        Field(
            description="Prompt used when calling the vision-language model.",
            examples=[
                "Convert this page to docling.",
                "Convert this page to markdown. Do not miss any text and only output the bare markdown!",
            ],
        ),
    ] = "Convert this page to docling."
    scale: Annotated[float, Field(description="Scale factor of the images used.")] = 2.0
    response_format: Annotated[
        ResponseFormat, Field(description="Type of response generated by the model.")
    ]
    inference_framework: Annotated[
        InferenceFramework, Field(description="Inference framework to use.")
    ]
    transformers_model_type: Annotated[
        TransformersModelType,
        Field(description="Type of transformers auto-model to use."),
    ] = TransformersModelType.AUTOMODEL
    extra_generation_config: Annotated[
        dict[str, Any],
        Field(
            description="Config from https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig",
            examples=[{"max_new_tokens": 800, "do_sample": False}],
        ),
    ] = {"max_new_tokens": 800, "do_sample": False}
    temperature: Annotated[
        float,
        Field(
            description="Temperature parameter controlling the reproducibility of the result.",
            examples=[0.0, 1.0],
        ),
    ] = 0.0

    @staticmethod
    def from_docling(options: InlineVlmOptions):
        return VlmModelLocal.model_validate(options.model_dump())


class VlmModelApi(BaseModel):
    url: Annotated[
        AnyUrl,
        Field(
            description="Endpoint which accepts openai-api compatible requests.",
            examples=[
                AnyUrl(
                    "http://localhost:8000/v1/chat/completions"
                ),  # example of a local vllm api
                AnyUrl(
                    "http://localhost:1234/v1/chat/completions"
                ),  # example of lm studio
            ],
        ),
    ]
    headers: Annotated[
        dict[str, str],
        Field(
            description="Headers used for calling the API endpoint. For example, it could include authentication headers."
        ),
    ] = {}
    params: Annotated[
        dict[str, Any],
        Field(
            description="Model parameters.",
            examples=[
                {  # on vllm
                    "model": "ibm-granite/granite-docling-258M",
                    "max_completion_tokens": 800,
                },
                {  # on vllm
                    "model": "ibm-granite/granite-vision-3.3-2b",
                    "max_completion_tokens": 800,
                },
            ],
        ),
    ] = {}
    timeout: Annotated[float, Field(description="Timeout for the API request.")] = 60
    concurrency: Annotated[
        PositiveInt,
        Field(
            description="Maximum number of concurrent requests to the API.",
            examples=[1],
        ),
    ] = 1
    prompt: Annotated[
        str,
        Field(
            description="Prompt used when calling the vision-language model.",
            examples=[
                "Convert this page to docling.",
                "Convert this page to markdown. Do not miss any text and only output the bare markdown!",
            ],
        ),
    ] = "Convert this page to docling."
    scale: Annotated[float, Field(description="Scale factor of the images used.")] = 2.0
    response_format: Annotated[
        ResponseFormat, Field(description="Type of response generated by the model.")
    ]
    temperature: Annotated[
        float,
        Field(
            description="Temperature parameter controlling the reproducibility of the result.",
            examples=[0.0, 1.0],
        ),
    ] = 0.0


class ConvertDocumentsOptions(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    from_formats: Annotated[
        list[InputFormat],
        Field(
            description=(
                "Input format(s) to convert from. String or list of strings. "
                f"Allowed values: {', '.join([v.value for v in InputFormat])}. "
                "Optional, defaults to all formats."
            ),
            examples=[[v.value for v in InputFormat]],
        ),
    ] = list(InputFormat)

    to_formats: Annotated[
        list[OutputFormat],
        Field(
            description=(
                "Output format(s) to convert to. String or list of strings. "
                f"Allowed values: {', '.join([v.value for v in OutputFormat])}. "
                "Optional, defaults to Markdown."
            ),
            examples=[
                [OutputFormat.MARKDOWN],
                [OutputFormat.MARKDOWN, OutputFormat.JSON],
                [v.value for v in OutputFormat],
            ],
        ),
    ] = [OutputFormat.MARKDOWN]

    image_export_mode: Annotated[
        ImageRefMode,
        Field(
            description=(
                "Image export mode for the document (in case of JSON,"
                " Markdown or HTML). "
                f"Allowed values: {', '.join([v.value for v in ImageRefMode])}. "
                "Optional, defaults to Embedded."
            ),
            examples=[ImageRefMode.EMBEDDED.value],
            # pattern="embedded|placeholder|referenced",
        ),
    ] = ImageRefMode.EMBEDDED

    do_ocr: Annotated[
        bool,
        Field(
            description=(
                "If enabled, the bitmap content will be processed using OCR. "
                "Boolean. Optional, defaults to true"
            ),
            # examples=[True],
        ),
    ] = True

    force_ocr: Annotated[
        bool,
        Field(
            description=(
                "If enabled, replace existing text with OCR-generated "
                "text over content. Boolean. Optional, defaults to false."
            ),
            # examples=[False],
        ),
    ] = False

    ocr_engine: Annotated[
        str,
        Field(
            description=(
                "DEPRECATED: Use ocr_preset instead. The OCR engine to use. String. "
            ),
            deprecated=True,
        ),
    ] = "auto"

    ocr_lang: Annotated[
        Optional[list[str]],
        Field(
            description=(
                "List of languages used by the OCR engine. "
                "Note that each OCR engine has "
                "different values for the language names. String or list of strings. "
                "Optional, defaults to empty."
            ),
            examples=[["fr", "de", "es", "en"]],
        ),
    ] = None

    ocr_preset: Annotated[
        str,
        Field(
            default="auto",
            description="Preset ID for OCR engine.",
            examples=["auto", "easyocr", "tesseract"],
        ),
    ] = "auto"

    ocr_custom_config: Annotated[
        Optional[dict[str, Any]],
        Field(
            default=None,
            description=(
                "Custom configuration for OCR engine. Use this to specify "
                "engine-specific options beyond ocr_lang. "
                "Each OCR engine kind has its own configuration schema."
            ),
            examples=[
                {
                    "kind": "easyocr",
                    "lang": ["en", "fr"],
                    "use_gpu": True,
                    "confidence_threshold": 0.5,
                    "force_full_page_ocr": False,
                },
                {
                    "kind": "tesseract_cli",
                    "lang": ["eng", "deu"],
                    "force_full_page_ocr": False,
                },
            ],
        ),
    ] = None

    pdf_backend: Annotated[
        PdfBackend,
        Field(
            description=(
                "The PDF backend to use. String. "
                f"Allowed values: {', '.join([v.value for v in PdfBackend])}. "
                f"Optional, defaults to {PdfBackend.DOCLING_PARSE.value}."
            ),
            examples=[PdfBackend.DOCLING_PARSE],
        ),
    ] = PdfBackend.DOCLING_PARSE

    table_mode: Annotated[
        TableFormerMode,
        Field(
            description=(
                "Mode to use for table structure, String. "
                f"Allowed values: {', '.join([v.value for v in TableFormerMode])}. "
                "Optional, defaults to accurate."
            ),
            examples=[TableStructureOptions().mode],
            # pattern="fast|accurate",
        ),
    ] = TableStructureOptions().mode

    table_cell_matching: Annotated[
        bool,
        Field(
            description="If true, matches table cells predictions back to PDF cells. Can break table output if PDF cells are merged across table columns. If false, let table structure model define the text cells, ignore PDF cells.",
            examples=[TableStructureOptions().do_cell_matching],
        ),
    ] = TableStructureOptions().do_cell_matching

    pipeline: Annotated[
        ProcessingPipeline,
        Field(description="Choose the pipeline to process PDF or image files."),
    ] = ProcessingPipeline.STANDARD

    page_range: Annotated[
        PageRange,
        Field(
            description="Only convert a range of pages. The page number starts at 1.",
            examples=[DEFAULT_PAGE_RANGE, (1, 4)],
        ),
    ] = DEFAULT_PAGE_RANGE

    document_timeout: Annotated[
        Optional[float],
        Field(
            description="The timeout for processing each document, in seconds.",
        ),
    ] = None

    abort_on_error: Annotated[
        bool,
        Field(
            description=(
                "Abort on error if enabled. Boolean. Optional, defaults to false."
            ),
            # examples=[False],
        ),
    ] = False

    do_table_structure: Annotated[
        bool,
        Field(
            description=(
                "If enabled, the table structure will be extracted. "
                "Boolean. Optional, defaults to true."
            ),
            examples=[True],
        ),
    ] = True

    include_images: Annotated[
        bool,
        Field(
            description=(
                "If enabled, images will be extracted from the document. "
                "Boolean. Optional, defaults to true."
            ),
            examples=[True],
        ),
    ] = True

    images_scale: Annotated[
        float,
        Field(
            description="Scale factor for images. Float. Optional, defaults to 2.0.",
            examples=[2.0],
        ),
    ] = 2.0

    md_page_break_placeholder: Annotated[
        str,
        Field(
            description="Add this placeholder between pages in the markdown output.",
            examples=["<!-- page-break -->", ""],
        ),
    ] = ""

    do_code_enrichment: Annotated[
        bool,
        Field(
            description=(
                "If enabled, perform OCR code enrichment. "
                "Boolean. Optional, defaults to false."
            ),
            examples=[False],
        ),
    ] = False

    do_formula_enrichment: Annotated[
        bool,
        Field(
            description=(
                "If enabled, perform formula OCR, return LaTeX code. "
                "Boolean. Optional, defaults to false."
            ),
            examples=[False],
        ),
    ] = False

    do_picture_classification: Annotated[
        bool,
        Field(
            description=(
                "If enabled, classify pictures in documents. "
                "Boolean. Optional, defaults to false."
            ),
            examples=[False],
        ),
    ] = False

    do_chart_extraction: Annotated[
        bool,
        Field(
            description=(
                "If enabled, extract numeric data from charts. "
                "Boolean. Optional, defaults to false."
            ),
            examples=[False],
        ),
    ] = False

    do_picture_description: Annotated[
        bool,
        Field(
            description=(
                "If enabled, describe pictures in documents. "
                "Boolean. Optional, defaults to false."
            ),
            examples=[False],
        ),
    ] = False

    picture_description_area_threshold: Annotated[
        float,
        Field(
            description="Minimum percentage of the area for a picture to be processed with the models.",
            examples=[PictureDescriptionBaseOptions().picture_area_threshold],
        ),
    ] = PictureDescriptionBaseOptions().picture_area_threshold

    picture_description_local: Annotated[
        Optional[PictureDescriptionLocal],
        Field(
            deprecated=True,
            description="DEPRECATED: Options for running a local vision-language model in the picture description. The parameters refer to a model hosted on Hugging Face. This parameter is mutually exclusive with picture_description_api. Please migrate to picture_description_preset or picture_description_custom_config.",
            examples=[
                PictureDescriptionLocal(repo_id="ibm-granite/granite-vision-3.2-2b"),
                PictureDescriptionLocal(repo_id="HuggingFaceTB/SmolVLM-256M-Instruct"),
            ],
        ),
    ] = None

    picture_description_api: Annotated[
        Optional[PictureDescriptionApi],
        Field(
            deprecated=True,
            description="DEPRECATED: API details for using a vision-language model in the picture description. This parameter is mutually exclusive with picture_description_local. Please migrate to picture_description_preset or picture_description_custom_config.",
            examples=[
                PictureDescriptionApi(
                    url=AnyUrl("http://localhost:1234/v1/chat/completions"),
                    params={"model": "granite3.2-vision:2b"},
                ),
                PictureDescriptionApi(
                    url=AnyUrl("http://localhost:11434/v1/chat/completions"),
                    params={"model": "granite3.2-vision:2b"},
                ),
            ],
        ),
    ] = None

    vlm_pipeline_model: Annotated[
        Optional[vlm_model_specs.VlmModelType],
        Field(
            deprecated=True,
            description="DEPRECATED: Preset of local and API models for the vlm pipeline. This parameter is mutually exclusive with vlm_pipeline_model_local and vlm_pipeline_model_api. Use the other options for more parameters. Please migrate to vlm_pipeline_preset or vlm_pipeline_custom_config.",
            examples=[vlm_model_specs.VlmModelType.GRANITEDOCLING],
        ),
    ] = None
    vlm_pipeline_model_local: Annotated[
        Optional[VlmModelLocal],
        Field(
            deprecated=True,
            description="DEPRECATED: Options for running a local vision-language model for the vlm pipeline. The parameters refer to a model hosted on Hugging Face. This parameter is mutually exclusive with vlm_pipeline_model_api and vlm_pipeline_model. Please migrate to vlm_pipeline_preset or vlm_pipeline_custom_config.",
            examples=[
                VlmModelLocal.from_docling(vlm_model_specs.GRANITEDOCLING_TRANSFORMERS),
                VlmModelLocal.from_docling(vlm_model_specs.GRANITEDOCLING_MLX),
                VlmModelLocal.from_docling(vlm_model_specs.GRANITE_VISION_TRANSFORMERS),
            ],
        ),
    ] = None

    vlm_pipeline_model_api: Annotated[
        Optional[VlmModelApi],
        Field(
            deprecated=True,
            description="DEPRECATED: API details for using a vision-language model for the vlm pipeline. This parameter is mutually exclusive with vlm_pipeline_model_local and vlm_pipeline_model. Please migrate to vlm_pipeline_preset or vlm_pipeline_custom_config.",
            examples=[
                VlmModelApi(
                    url=AnyUrl("http://localhost:1234/v1/chat/completions"),
                    params={"model": "ibm-granite/granite-docling-258M-mlx"},
                    response_format=ResponseFormat.DOCTAGS,
                    prompt="Convert this page to docling.",
                )
            ],
        ),
    ] = None

    # === NEW: Preset Selection OR Custom Config ===

    # Option 1: Use preset (recommended)
    vlm_pipeline_preset: Annotated[
        Optional[str],
        Field(
            default=None,
            description='Preset ID to use (e.g., "default", "granite_docling"). '
            'Use "default" for stable, admin-controlled configuration.',
            examples=["default", "granite_docling"],
        ),
    ] = None

    picture_description_preset: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Preset ID for picture description.",
            examples=["default", "smolvlm", "granite_vision"],
        ),
    ] = None

    code_formula_preset: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Preset ID for code/formula extraction.",
            examples=["default"],
        ),
    ] = None

    # Option 2: Custom configuration (if allowed by config)
    vlm_pipeline_custom_config: Annotated[
        Optional[Union[VlmConvertOptions, dict]],
        Field(
            default=None,
            description="Custom VLM configuration including model spec and engine options. "
            "Only available if admin allows it. Must include 'model_spec' and 'engine_options'.",
        ),
    ] = None

    picture_description_custom_config: Annotated[
        Optional[Union[PictureDescriptionVlmEngineOptions, dict]],
        Field(
            default=None,
            description="Custom picture description configuration including model spec and engine options.",
        ),
    ] = None

    code_formula_custom_config: Annotated[
        Optional[Union[CodeFormulaVlmOptions, dict]],
        Field(
            default=None,
            description="Custom code/formula extraction configuration including model spec and engine options.",
        ),
    ] = None

    # === NEW: Kind Selection for Pipeline Stages ===

    # Table Structure Configuration
    table_structure_preset: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Preset ID for table structure detection.",
            examples=[
                "default",
                "tableformer_v1_accurate",
                "tableformer_v1_fast",
                "tableformer_v2",
            ],
        ),
    ] = None

    table_structure_custom_config: Annotated[
        Optional[dict[str, Any]],
        Field(
            default=None,
            description=(
                "Custom configuration for table structure model. Use this to specify a "
                "non-default kind with its options. The 'kind' field in the config dict "
                "determines which table structure implementation to use. "
                "If not specified, uses the default kind with preset configuration."
            ),
            examples=[
                {
                    "kind": "custom_table_model",
                    "model_path": "/path/to/model",
                    "confidence_threshold": 0.8,
                },
                {
                    "kind": "docling_tableformer",
                    "mode": "fast",
                    "do_cell_matching": False,
                },
            ],
        ),
    ] = None

    # Layout Configuration
    layout_custom_config: Annotated[
        Optional[dict[str, Any]],
        Field(
            default=None,
            description=(
                "Custom configuration for layout model. Use this to specify a "
                "non-default kind with its options. The 'kind' field in the config dict "
                "determines which layout implementation to use. "
                "If not specified, uses the default kind with preset configuration."
            ),
            examples=[
                {
                    "kind": "docling_layout_default",
                    "keep_empty_clusters": False,
                    "skip_cell_assignment": False,
                    "create_orphan_clusters": True,
                },
                {
                    "kind": "layout_object_detection",
                    "keep_empty_clusters": False,
                    "skip_cell_assignment": False,
                },
            ],
        ),
    ] = None

    # Layout preset field
    layout_preset: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Preset ID for layout detection.",
            examples=["default", "docling_layout_default"],
        ),
    ] = None

    # Picture Classification Configuration
    picture_classification_preset: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Preset ID for picture classification.",
            examples=["default", "document_figure_classifier_v2"],
        ),
    ] = None

    picture_classification_custom_config: Annotated[
        Optional[dict[str, Any]],
        Field(
            default=None,
            description=(
                "Custom configuration for picture classification. Use this to specify "
                "custom options for the picture classifier. "
                "The configuration should match DocumentPictureClassifierOptions schema."
            ),
            examples=[
                {
                    "kind": "document_picture_classifier",
                },
            ],
        ),
    ] = None

    # Field validators for deprecated fields - trigger warnings on assignment
    @field_validator("picture_description_api", mode="before")
    @classmethod
    def validate_picture_description_api(cls, v):
        """Emit deprecation warning when picture_description_api is set."""
        if v is not None:
            warnings.warn(
                "picture_description_api is deprecated. "
                "Please migrate to picture_description_preset or "
                "picture_description_custom_config.",
                DeprecationWarning,
                stacklevel=2,
            )
        return v

    @field_validator("picture_description_local", mode="before")
    @classmethod
    def validate_picture_description_local(cls, v):
        """Emit deprecation warning when picture_description_local is set."""
        if v is not None:
            warnings.warn(
                "picture_description_local is deprecated. "
                "Please migrate to picture_description_preset or "
                "picture_description_custom_config.",
                DeprecationWarning,
                stacklevel=2,
            )
        return v

    @field_validator("vlm_pipeline_model", mode="before")
    @classmethod
    def validate_vlm_pipeline_model(cls, v):
        """Emit deprecation warning when vlm_pipeline_model is set."""
        if v is not None:
            warnings.warn(
                "vlm_pipeline_model is deprecated. "
                "Please migrate to vlm_pipeline_preset or vlm_pipeline_custom_config.",
                DeprecationWarning,
                stacklevel=2,
            )
        return v

    @field_validator("vlm_pipeline_model_local", mode="before")
    @classmethod
    def validate_vlm_pipeline_model_local(cls, v):
        """Emit deprecation warning when vlm_pipeline_model_local is set."""
        if v is not None:
            warnings.warn(
                "vlm_pipeline_model_local is deprecated. "
                "Please migrate to vlm_pipeline_preset or vlm_pipeline_custom_config.",
                DeprecationWarning,
                stacklevel=2,
            )
        return v

    @field_validator("vlm_pipeline_model_api", mode="before")
    @classmethod
    def validate_vlm_pipeline_model_api(cls, v):
        """Emit deprecation warning when vlm_pipeline_model_api is set."""
        if v is not None:
            warnings.warn(
                "vlm_pipeline_model_api is deprecated. "
                "Please migrate to vlm_pipeline_preset or vlm_pipeline_custom_config.",
                DeprecationWarning,
                stacklevel=2,
            )
        return v

    @model_validator(mode="after")
    def picture_description_exclusivity(self) -> Self:
        # Validate picture description options
        if (
            self.picture_description_local is not None
            and self.picture_description_api is not None
        ):
            raise ValueError(
                "The parameters picture_description_local and picture_description_api are mutually exclusive, only one of them can be set."
            )

        return self

    @model_validator(mode="after")
    def vlm_model_exclusivity(self) -> Self:
        # Validate vlm model options
        num_not_nan = sum(
            opt is not None
            for opt in (
                self.vlm_pipeline_model,
                self.vlm_pipeline_model_local,
                self.vlm_pipeline_model_api,
            )
        )
        if num_not_nan > 1:
            raise ValueError(
                "The parameters vlm_pipeline_model, vlm_pipeline_model_local and vlm_pipeline_model_api are mutually exclusive, only one of them can be set."
            )

        return self

    @model_validator(mode="after")
    def validate_vlm_pipeline_options(self) -> Self:
        """Ensure preset and custom config are mutually exclusive for VLM pipeline."""
        if self.vlm_pipeline_preset and self.vlm_pipeline_custom_config:
            raise ValueError(
                "Cannot specify both vlm_pipeline_preset and vlm_pipeline_custom_config. "
                "Please use one or the other."
            )

        # Check if using legacy fields with new fields
        legacy_set = (
            self.vlm_pipeline_model is not None
            or self.vlm_pipeline_model_local is not None
            or self.vlm_pipeline_model_api is not None
        )
        new_set = (
            self.vlm_pipeline_preset is not None
            or self.vlm_pipeline_custom_config is not None
        )

        if legacy_set and new_set:
            raise ValueError(
                "Cannot mix legacy VLM options (vlm_pipeline_model*) with new options "
                "(vlm_pipeline_preset/custom_config). Please use only one approach."
            )

        # Note: Deprecation warnings are now emitted by field validators
        # when the fields are set, not here in the model validator

        return self

    @model_validator(mode="after")
    def validate_picture_description_options(self) -> Self:
        """Ensure preset and custom config are mutually exclusive for picture description."""
        if self.picture_description_preset and self.picture_description_custom_config:
            raise ValueError(
                "Cannot specify both picture_description_preset and "
                "picture_description_custom_config."
            )

        # Check if using legacy fields with new fields
        legacy_set = (
            self.picture_description_local is not None
            or self.picture_description_api is not None
        )
        new_set = (
            self.picture_description_preset is not None
            or self.picture_description_custom_config is not None
        )

        if legacy_set and new_set:
            raise ValueError(
                "Cannot mix legacy picture description options (picture_description_local/api) "
                "with new options (picture_description_preset/custom_config). "
                "Please use only one approach."
            )

        # Note: Deprecation warnings are now emitted by field validators
        # when the fields are set, not here in the model validator

        return self

    @model_validator(mode="after")
    def validate_code_formula_options(self) -> Self:
        """Ensure preset and custom config are mutually exclusive for code/formula."""
        if self.code_formula_preset and self.code_formula_custom_config:
            raise ValueError(
                "Cannot specify both code_formula_preset and code_formula_custom_config."
            )

        return self

    @model_validator(mode="after")
    def validate_layout_options(self) -> Self:
        """Ensure preset and custom config are mutually exclusive for layout."""
        if self.layout_preset and self.layout_custom_config:
            raise ValueError(
                "Cannot specify both layout_preset and layout_custom_config."
            )
        return self

    @model_validator(mode="after")
    def validate_picture_classification_options(self) -> Self:
        """Ensure preset and custom config are mutually exclusive for picture classification."""
        if (
            self.picture_classification_preset
            and self.picture_classification_custom_config
        ):
            raise ValueError(
                "Cannot specify both picture_classification_preset and "
                "picture_classification_custom_config."
            )
        return self

    @model_validator(mode="after")
    def validate_ocr_options(self) -> Self:
        """Handle deprecated ocr_engine and sync to ocr_preset."""
        # If ocr_engine is explicitly set (not default), sync to ocr_preset
        if (
            hasattr(self, "__pydantic_fields_set__")
            and "ocr_engine" in self.__pydantic_fields_set__
            and "ocr_preset" not in self.__pydantic_fields_set__
        ):
            warnings.warn(
                "ocr_engine is deprecated and will be removed in a future version. "
                "Use ocr_preset instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Sync ocr_engine value to ocr_preset only if ocr_preset wasn't explicitly set
            object.__setattr__(self, "ocr_preset", self.ocr_engine)

        # Ensure preset and custom_config are mutually exclusive
        if self.ocr_preset != "auto" and self.ocr_custom_config:
            raise ValueError("Cannot specify both ocr_preset and ocr_custom_config.")

        return self
