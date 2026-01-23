import logging
import re
from io import BytesIO
from pathlib import Path
from typing import List, Union, cast

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    DocItem,
    DoclingDocument,
    ImageRef,
    PictureItem,
    ProvenanceItem,
    TextItem,
)
from docling_core.types.doc.base import (
    BoundingBox,
    Size,
)
from docling_core.types.doc.document import DocTagsDocument
from PIL import Image as PILImage

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
    DeclarativeDocumentBackend,
)
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.models.vlm_pipeline_models.api_vlm_model import ApiVlmModel
from docling.models.vlm_pipeline_models.hf_transformers_model import (
    HuggingFaceTransformersVlmModel,
)
from docling.models.vlm_pipeline_models.mlx_model import HuggingFaceMlxModel
from docling.pipeline.base_pipeline import PaginatedPipeline
from docling.utils.deepseekocr_utils import parse_deepseekocr_markdown
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


class VlmPipeline(PaginatedPipeline):
    def __init__(self, pipeline_options: VlmPipelineOptions):
        super().__init__(pipeline_options)
        self.keep_backend = True

        self.pipeline_options: VlmPipelineOptions

        # force_backend_text = False - use text that is coming from VLM response
        # force_backend_text = True - get text from backend using bounding boxes predicted by SmolDocling doctags
        self.force_backend_text = (
            pipeline_options.force_backend_text
            and pipeline_options.vlm_options.response_format == ResponseFormat.DOCTAGS
        )

        self.keep_images = self.pipeline_options.generate_page_images

        if isinstance(pipeline_options.vlm_options, ApiVlmOptions):
            self.build_pipe = [
                ApiVlmModel(
                    enabled=True,  # must be always enabled for this pipeline to make sense.
                    enable_remote_services=self.pipeline_options.enable_remote_services,
                    vlm_options=cast(ApiVlmOptions, self.pipeline_options.vlm_options),
                ),
            ]
        elif isinstance(self.pipeline_options.vlm_options, InlineVlmOptions):
            vlm_options = cast(InlineVlmOptions, self.pipeline_options.vlm_options)
            if vlm_options.inference_framework == InferenceFramework.MLX:
                self.build_pipe = [
                    HuggingFaceMlxModel(
                        enabled=True,  # must be always enabled for this pipeline to make sense.
                        artifacts_path=self.artifacts_path,
                        accelerator_options=pipeline_options.accelerator_options,
                        vlm_options=vlm_options,
                    ),
                ]
            elif vlm_options.inference_framework == InferenceFramework.TRANSFORMERS:
                self.build_pipe = [
                    HuggingFaceTransformersVlmModel(
                        enabled=True,  # must be always enabled for this pipeline to make sense.
                        artifacts_path=self.artifacts_path,
                        accelerator_options=pipeline_options.accelerator_options,
                        vlm_options=vlm_options,
                    ),
                ]
            elif vlm_options.inference_framework == InferenceFramework.VLLM:
                from docling.models.vlm_pipeline_models.vllm_model import VllmVlmModel

                self.build_pipe = [
                    VllmVlmModel(
                        enabled=True,  # must be always enabled for this pipeline to make sense.
                        artifacts_path=self.artifacts_path,
                        accelerator_options=pipeline_options.accelerator_options,
                        vlm_options=vlm_options,
                    ),
                ]
            else:
                raise ValueError(
                    f"Could not instantiate the right type of VLM pipeline: {vlm_options.inference_framework}"
                )

        self.enrichment_pipe = [
            # Other models working on `NodeItem` elements in the DoclingDocument
        ]

    def initialize_page(self, conv_res: ConversionResult, page: Page) -> Page:
        with TimeRecorder(conv_res, "page_init"):
            images_scale = self.pipeline_options.images_scale
            if images_scale is not None:
                page._default_image_scale = images_scale
            page._backend = conv_res.input._backend.load_page(page.page_no - 1)  # type: ignore
            if page._backend is not None and page._backend.is_valid():
                page.size = page._backend.get_size()

                if self.force_backend_text:
                    page.parsed_page = page._backend.get_segmented_page()

        return page

    def extract_text_from_backend(
        self, page: Page, bbox: Union[BoundingBox, None]
    ) -> str:
        # Convert bounding box normalized to 0-100 into page coordinates for cropping
        text = ""
        if bbox:
            if page.size:
                if page._backend:
                    text = page._backend.get_text_in_rect(bbox)
        return text

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):
            if (
                self.pipeline_options.vlm_options.response_format
                == ResponseFormat.DOCTAGS
            ):
                conv_res.document = self._turn_dt_into_doc(conv_res)

            elif (
                self.pipeline_options.vlm_options.response_format
                == ResponseFormat.DEEPSEEKOCR_MARKDOWN
            ):
                conv_res.document = self._parse_deepseekocr_markdown(conv_res)

            elif (
                self.pipeline_options.vlm_options.response_format
                == ResponseFormat.MARKDOWN
            ):
                conv_res.document = self._convert_text_with_backend(
                    conv_res, InputFormat.MD, MarkdownDocumentBackend
                )

            elif (
                self.pipeline_options.vlm_options.response_format == ResponseFormat.HTML
            ):
                conv_res.document = self._convert_text_with_backend(
                    conv_res, InputFormat.HTML, HTMLDocumentBackend
                )

            else:
                raise RuntimeError(
                    f"Unsupported VLM response format {self.pipeline_options.vlm_options.response_format}"
                )

            # Generate images of the requested element types
            if self.pipeline_options.generate_picture_images:
                scale = self.pipeline_options.images_scale
                for element, _level in conv_res.document.iterate_items():
                    if not isinstance(element, DocItem) or len(element.prov) == 0:
                        continue
                    if (
                        isinstance(element, PictureItem)
                        and self.pipeline_options.generate_picture_images
                    ):
                        page_ix = element.prov[0].page_no - 1
                        page = conv_res.pages[page_ix]
                        assert page.size is not None
                        assert page.image is not None

                        crop_bbox = (
                            element.prov[0]
                            .bbox.scaled(scale=scale)
                            .to_top_left_origin(page_height=page.size.height * scale)
                        )

                        cropped_im = page.image.crop(crop_bbox.as_tuple())
                        element.image = ImageRef.from_pil(
                            cropped_im, dpi=int(72 * scale)
                        )

        return conv_res

    def _turn_dt_into_doc(self, conv_res) -> DoclingDocument:
        doctags_list = []
        image_list = []
        for page in conv_res.pages:
            predicted_doctags = ""
            img = PILImage.new("RGB", (1, 1), "rgb(255,255,255)")
            if page.predictions.vlm_response:
                predicted_doctags = page.predictions.vlm_response.text
            if page.image:
                img = page.image
            image_list.append(img)
            doctags_list.append(predicted_doctags)

        doctags_list_c = cast(List[Union[Path, str]], doctags_list)
        image_list_c = cast(List[Union[Path, PILImage.Image]], image_list)
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
            doctags_list_c, image_list_c
        )
        conv_res.document = DoclingDocument.load_from_doctags(
            doctag_document=doctags_doc
        )

        # If forced backend text, replace model predicted text with backend one
        if page.size:
            if self.force_backend_text:
                scale = self.pipeline_options.images_scale
                for element, _level in conv_res.document.iterate_items():
                    if not isinstance(element, TextItem) or len(element.prov) == 0:
                        continue
                    crop_bbox = (
                        element.prov[0]
                        .bbox.scaled(scale=scale)
                        .to_top_left_origin(page_height=page.size.height * scale)
                    )
                    txt = self.extract_text_from_backend(page, crop_bbox)
                    element.text = txt
                    element.orig = txt

        return conv_res.document

    def _parse_deepseekocr_markdown(
        self, conv_res: ConversionResult
    ) -> DoclingDocument:
        """Parse DeepSeek OCR markdown with label[[x1, y1, x2, y2]] format.

        Labels supported:
        - text: Standard body text
        - title: Main document or section titles
        - sub_title: Secondary headings or sub-headers
        - table: Tabular data
        - table_caption: Descriptive text for tables
        - figure: Image-based elements or diagrams
        - figure_caption: Titles or descriptions for figures/images
        - header / footer: Content at top or bottom margins of pages
        """
        page_docs = []

        for pg_idx, page in enumerate(conv_res.pages):
            predicted_text = ""
            if page.predictions.vlm_response:
                predicted_text = page.predictions.vlm_response.text

            assert page.size is not None

            # Parse single page using the utility function
            # Pass vlm_options.scale to convert bboxes from scaled image coords to original PDF coords
            page_doc = parse_deepseekocr_markdown(
                content=predicted_text,
                original_page_size=page.size,
                page_no=pg_idx + 1,
                filename=conv_res.input.file.name or "file",
                page_image=page.image,
            )
            page_docs.append(page_doc)

        # Add page metadata and concatenate
        return self._add_page_metadata_and_concatenate(page_docs, conv_res)

    def _extract_code_block(self, text: str) -> str:
        """
        Extracts text from markdown code blocks (enclosed in triple backticks).
        If no code blocks are found, returns the original text.

        Args:
            text (str): Input text that may contain markdown code blocks

        Returns:
            str: Extracted code if code blocks exist, otherwise original text
        """
        # Regex pattern to match content between triple backticks
        # This handles multiline content and optional language specifier
        pattern = r"^```(?:\w*\n)?(.*?)```(\n)*$"

        # Search with DOTALL flag to match across multiple lines
        mtch = re.search(pattern, text, re.DOTALL)

        if mtch:
            # Return only the content of the first capturing group
            return mtch.group(1)
        else:
            # No code blocks found, return original text
            return text

    def _add_page_metadata_and_concatenate(
        self,
        page_docs: List[DoclingDocument],
        conv_res: ConversionResult,
    ) -> DoclingDocument:
        """
        Add page metadata to page documents and concatenate them.

        Args:
            page_docs: List of page documents to process
            conv_res: Conversion result containing page information

        Returns:
            DoclingDocument: Concatenated document with page metadata
        """
        for pg_idx, (page_doc, page) in enumerate(zip(page_docs, conv_res.pages)):
            # Add page metadata to the page document before concatenation
            if page.image is not None:
                pg_width = page.image.width
                pg_height = page.image.height
            else:
                pg_width = 1
                pg_height = 1

            page_doc.add_page(
                page_no=pg_idx + 1,
                size=Size(width=pg_width, height=pg_height),
                image=ImageRef.from_pil(image=page.image, dpi=72)
                if page.image
                else None,
            )

        # Concatenate all page documents to preserve hierarchy
        return DoclingDocument.concatenate(docs=page_docs)

    def _convert_text_with_backend(
        self,
        conv_res: ConversionResult,
        input_format: InputFormat,
        backend_class: type[DeclarativeDocumentBackend],
    ) -> DoclingDocument:
        """
        Convert text-based formats (Markdown, HTML) into DoclingDocument using a backend.

        Args:
            conv_res: The conversion result containing pages with VLM predictions
            input_format: The format type (MD or HTML)
            backend_class: The backend class to use for conversion

        Returns:
            DoclingDocument: The assembled document
        """
        page_docs = []

        for pg_idx, page in enumerate(conv_res.pages):
            predicted_text = ""
            if page.predictions.vlm_response:
                predicted_text = page.predictions.vlm_response.text + "\n\n"

            # Extract content from code blocks if present
            predicted_text = self._extract_code_block(text=predicted_text)

            # Convert text to document using specified backend
            response_bytes = BytesIO(predicted_text.encode("utf8"))
            out_doc = InputDocument(
                path_or_stream=response_bytes,
                filename=conv_res.input.file.name,
                format=input_format,
                backend=backend_class,
            )
            backend = backend_class(
                in_doc=out_doc,
                path_or_stream=response_bytes,
            )
            page_doc = backend.convert()

            # Modify provenance in place for all items in the page document
            for item, level in page_doc.iterate_items(
                with_groups=True,
                traverse_pictures=True,
                included_content_layers=set(ContentLayer),
            ):
                if isinstance(item, DocItem):
                    item.prov = [
                        ProvenanceItem(
                            page_no=pg_idx + 1,
                            bbox=BoundingBox(
                                t=0.0, b=0.0, l=0.0, r=0.0
                            ),  # FIXME: would be nice not to have to "fake" it
                            charspan=[0, 0],
                        )
                    ]

            page_docs.append(page_doc)

        # Add page metadata and concatenate
        return self._add_page_metadata_and_concatenate(page_docs, conv_res)

    @classmethod
    def get_default_options(cls) -> VlmPipelineOptions:
        return VlmPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return isinstance(backend, PdfDocumentBackend)
