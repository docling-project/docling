import logging
import re
import time
import warnings
from collections.abc import Generator
from io import BytesIO
from typing import cast

from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer
from docling_core.types.doc import (
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
from typing_extensions import override

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
    DeclarativeDocumentBackend,
)
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend, iter_pdf_page_backends
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    InputFormat,
    Page,
    VlmStopReason,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    BaseVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.datamodel.settings import settings

# VlmResponseFormat is actually ResponseFormat from pipeline_options_vlm_model
# No need to import it separately as it's already imported above
from docling.models.stages.vlm_convert.vlm_convert_model import VlmConvertModel
from docling.models.vlm_pipeline_models.api_vlm_model import ApiVlmModel
from docling.models.vlm_pipeline_models.hf_transformers_model import (
    HuggingFaceTransformersVlmModel,
)
from docling.models.vlm_pipeline_models.mlx_model import HuggingFaceMlxModel
from docling.pipeline.base_pipeline import PaginatedPipeline
from docling.utils.deepseekocr_utils import parse_deepseekocr_markdown
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)
_DOCLANG_OPEN_RE = re.compile(r"<doclang(?:\s[^>]*)?>")


class VlmPipeline(PaginatedPipeline):
    def __init__(self, pipeline_options: VlmPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: VlmPipelineOptions

        # Check if using new VlmConvertOptions
        if isinstance(pipeline_options.vlm_options, VlmConvertOptions):
            self._initialize_new_runtime_system(pipeline_options)
        else:
            self._initialize_legacy_vlm_models(pipeline_options)

        self.enrichment_pipe: list = [
            # Other models working on `NodeItem` elements in the DoclingDocument
        ]

    def _initialize_new_runtime_system(
        self, pipeline_options: VlmPipelineOptions
    ) -> None:
        """Initialize pipeline with new VlmConvertOptions and runtime system.

        Args:
            pipeline_options: Pipeline configuration with VlmConvertOptions
        """
        vlm_convert_options = cast(VlmConvertOptions, pipeline_options.vlm_options)

        # Determine response format from model spec
        response_format = vlm_convert_options.model_spec.response_format

        # force_backend_text = False - use text that is coming from VLM response
        # force_backend_text = True - get text from backend using bounding boxes predicted by SmolDocling doctags
        self.force_backend_text = (
            vlm_convert_options.force_backend_text
            and response_format == ResponseFormat.DOCTAGS
        )

        self.keep_images = self.pipeline_options.generate_page_images

        # Use new VlmConvertModel stage
        self.build_pipe = [
            VlmConvertModel(
                enabled=True,
                enable_remote_services=self.pipeline_options.enable_remote_services,
                artifacts_path=self.artifacts_path,
                options=vlm_convert_options,
                accelerator_options=self.pipeline_options.accelerator_options,
            ),
        ]

        _log.info("Using new VlmConvertModel with runtime system")

    def _initialize_legacy_vlm_models(
        self, pipeline_options: VlmPipelineOptions
    ) -> None:
        """Initialize pipeline with legacy InlineVlmOptions or ApiVlmOptions.

        Args:
            pipeline_options: Pipeline configuration with legacy VLM options

        Note:
            This method is deprecated and will be removed in a future version.
        """
        # Legacy path - using old InlineVlmOptions or ApiVlmOptions
        warnings.warn(
            "Using legacy VLM options (InlineVlmOptions/ApiVlmOptions) is deprecated. "
            "Please migrate to VlmConvertOptions with preset system. "
            "Example: VlmConvertOptions.from_preset('smoldocling')",
            DeprecationWarning,
            stacklevel=3,
        )

        # force_backend_text = False - use text that is coming from VLM response
        # force_backend_text = True - get text from backend using bounding boxes predicted by SmolDocling doctags
        self.force_backend_text = (
            pipeline_options.force_backend_text
            and pipeline_options.vlm_options.response_format == ResponseFormat.DOCTAGS  # type: ignore[union-attr]
        )

        self.keep_images = self.pipeline_options.generate_page_images

        if isinstance(pipeline_options.vlm_options, ApiVlmOptions):
            self.build_pipe = [
                ApiVlmModel(
                    enabled=True,
                    enable_remote_services=self.pipeline_options.enable_remote_services,
                    vlm_options=cast(ApiVlmOptions, self.pipeline_options.vlm_options),
                ),
            ]
        elif isinstance(self.pipeline_options.vlm_options, InlineVlmOptions):
            vlm_options = cast(InlineVlmOptions, self.pipeline_options.vlm_options)
            if vlm_options.inference_framework == InferenceFramework.MLX:
                self.build_pipe = [
                    HuggingFaceMlxModel(
                        enabled=True,
                        artifacts_path=self.artifacts_path,
                        accelerator_options=pipeline_options.accelerator_options,
                        vlm_options=vlm_options,
                    ),
                ]
            elif vlm_options.inference_framework == InferenceFramework.TRANSFORMERS:
                self.build_pipe = [
                    HuggingFaceTransformersVlmModel(
                        enabled=True,
                        artifacts_path=self.artifacts_path,
                        accelerator_options=pipeline_options.accelerator_options,
                        vlm_options=vlm_options,
                    ),
                ]
            elif vlm_options.inference_framework == InferenceFramework.VLLM:
                from docling.models.vlm_pipeline_models.vllm_model import VllmVlmModel

                self.build_pipe = [
                    VllmVlmModel(
                        enabled=True,
                        artifacts_path=self.artifacts_path,
                        accelerator_options=pipeline_options.accelerator_options,
                        vlm_options=vlm_options,
                    ),
                ]
            else:
                raise ValueError(
                    f"Could not instantiate the right type of VLM pipeline: {vlm_options.inference_framework}"
                )

    def initialize_page(self, conv_res: ConversionResult, page: Page) -> Page:
        assert isinstance(conv_res.input._backend, PdfDocumentBackend)
        page._backend = conv_res.input._backend.load_page(page.page_no - 1)
        return self._initialize_page(conv_res, page)

    def _initialize_page(self, conv_res: ConversionResult, page: Page) -> Page:
        with TimeRecorder(conv_res, "page_init"):
            images_scale = self.pipeline_options.images_scale
            if images_scale is not None:
                page._default_image_scale = images_scale
            if page._backend is not None and page._backend.is_valid():
                page.size = page._backend.get_size()

                if self.force_backend_text:
                    page.parsed_page = page._backend.get_segmented_page()

        return page

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        if not isinstance(conv_res.input._backend, PdfDocumentBackend):
            raise RuntimeError(
                f"The selected backend {type(conv_res.input._backend).__name__} for "
                f"{conv_res.input.file} is not a PDF backend."
            )

        start_page, end_page = conv_res.input.limits.page_range
        expected_page_nos = list(
            range(max(1, start_page), min(conv_res.input.page_count, end_page) + 1)
        )
        pages_by_no = {page_no: Page(page_no=page_no) for page_no in expected_page_nos}
        page_documents: dict[int, DoclingDocument] = {}
        processed_page_nos: set[int] = set()
        failed_page_nos: set[int] = set()
        page_iterator = iter_pdf_page_backends(
            conv_res.input._backend, expected_page_nos
        )
        assert isinstance(page_iterator, Generator)
        page_batch: list[Page] = []
        started_at = time.monotonic()

        try:
            for page_backend in page_iterator:
                page = pages_by_no.get(page_backend.page_no)
                if page is None:
                    page_backend.unload()
                    continue
                page._backend = page_backend
                if not page_backend.is_valid():
                    failed_page_nos.add(page.page_no)
                    conv_res.errors.append(
                        ErrorItem(
                            component_type=DoclingComponentType.DOCUMENT_BACKEND,
                            module_name=type(page_backend).__name__,
                            error_message="Page failed to parse.",
                            category=FailureCategory.BACKEND_FAILURE,
                            page_no=page.page_no,
                        )
                    )
                    self._release_page_resources(page)
                    continue
                page_batch.append(self._initialize_page(conv_res, page))
                if len(page_batch) < settings.perf.page_batch_size:
                    continue

                self._process_page_batch(
                    conv_res=conv_res,
                    page_batch=page_batch,
                    page_documents=page_documents,
                    processed_page_nos=processed_page_nos,
                )
                page_batch = []
                if self._document_timed_out(
                    conv_res=conv_res,
                    elapsed=time.monotonic() - started_at,
                    processed_pages=len(processed_page_nos),
                    total_pages=len(pages_by_no),
                ):
                    break

            if page_batch:
                self._process_page_batch(
                    conv_res=conv_res,
                    page_batch=page_batch,
                    page_documents=page_documents,
                    processed_page_nos=processed_page_nos,
                )
                self._document_timed_out(
                    conv_res=conv_res,
                    elapsed=time.monotonic() - started_at,
                    processed_pages=len(processed_page_nos),
                    total_pages=len(pages_by_no),
                )
        finally:
            page_iterator.close()
            for page in pages_by_no.values():
                self._release_page_resources(page)

        if not any(
            error.category == FailureCategory.TIMEOUT for error in conv_res.errors
        ):
            for page_no in (
                set(expected_page_nos) - processed_page_nos - failed_page_nos
            ):
                conv_res.errors.append(
                    ErrorItem(
                        component_type=DoclingComponentType.DOCUMENT_BACKEND,
                        module_name=type(conv_res.input._backend).__name__,
                        error_message="Page was not returned by the PDF backend.",
                        category=FailureCategory.BACKEND_FAILURE,
                        page_no=page_no,
                    )
                )

        conv_res.pages = [
            pages_by_no[page_no]
            for page_no in expected_page_nos
            if page_no in processed_page_nos
        ]
        conv_res.document = self._concatenate_page_documents(
            [
                (page_no, page_documents[page_no])
                for page_no in expected_page_nos
                if page_no in page_documents
            ]
        )
        return conv_res

    def _process_page_batch(
        self,
        *,
        conv_res: ConversionResult,
        page_batch: list[Page],
        page_documents: dict[int, DoclingDocument],
        processed_page_nos: set[int],
    ) -> None:
        try:
            for page in self._apply_on_pages(conv_res, page_batch):
                if page.size is None:
                    continue
                page_documents[page.page_no] = self._finalize_page_document(
                    conv_res, page
                )
                processed_page_nos.add(page.page_no)
        finally:
            for page in page_batch:
                self._release_page_resources(page)

    def _document_timed_out(
        self,
        *,
        conv_res: ConversionResult,
        elapsed: float,
        processed_pages: int,
        total_pages: int,
    ) -> bool:
        timeout = self.pipeline_options.document_timeout
        if timeout is None or elapsed <= timeout:
            return False

        message = (
            f"Document processing timeout: exceeded {timeout:.3f}s limit after "
            f"{elapsed:.3f}s. Processed {processed_pages}/{total_pages} pages."
        )
        _log.warning(message)
        conv_res.errors.append(
            ErrorItem(
                component_type=DoclingComponentType.PIPELINE,
                module_name=self.__class__.__name__,
                error_message=message,
                category=FailureCategory.TIMEOUT,
            )
        )
        conv_res.status = ConversionStatus.PARTIAL_SUCCESS
        return True

    @staticmethod
    def _release_page_resources(page: Page) -> None:
        if page._backend is not None:
            page._backend.unload()
            page._backend = None
        page._image_cache = {}
        page.parsed_page = None

    def extract_text_from_backend(self, page: Page, bbox: BoundingBox | None) -> str:
        # Convert bounding box normalized to 0-100 into page coordinates for cropping
        text = ""
        if bbox:
            if page.size:
                if page._backend:
                    text = page._backend.get_text_in_rect(bbox)
        return text

    @override
    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        """Determine conversion status accounting for VLM stop reasons.

        Extends the base implementation to detect partial failures from VLM
        inference, such as truncated output (LENGTH) or filtered content
        (CONTENT_FILTERED).
        """
        status = conv_res.status
        if status in {ConversionStatus.PENDING, ConversionStatus.STARTED}:
            status = ConversionStatus.SUCCESS

        for page in conv_res.pages:
            vlm_response = page.predictions.vlm_response
            if vlm_response is None:
                conv_res.errors.append(
                    ErrorItem(
                        component_type=DoclingComponentType.PIPELINE,
                        module_name=self.__class__.__name__,
                        error_message="No VLM prediction.",
                        category=FailureCategory.INFERENCE_FAILURE,
                        page_no=page.page_no,
                    )
                )
                status = ConversionStatus.PARTIAL_SUCCESS
            elif vlm_response.stop_reason in (
                VlmStopReason.LENGTH,
                VlmStopReason.CONTENT_FILTERED,
            ):
                conv_res.errors.append(
                    ErrorItem(
                        component_type=DoclingComponentType.PIPELINE,
                        module_name=self.__class__.__name__,
                        error_message="VLM output incomplete "
                        f"(stop_reason={vlm_response.stop_reason.value}).",
                        category=FailureCategory.INFERENCE_FAILURE,
                        page_no=page.page_no,
                    )
                )
                status = ConversionStatus.PARTIAL_SUCCESS

        if status == ConversionStatus.SUCCESS and conv_res.errors:
            status = ConversionStatus.PARTIAL_SUCCESS
        return status

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        return conv_res

    def _response_format(self) -> ResponseFormat:
        vlm_options = self.pipeline_options.vlm_options
        if isinstance(vlm_options, VlmConvertOptions):
            return vlm_options.model_spec.response_format
        assert isinstance(vlm_options, BaseVlmOptions)
        return vlm_options.response_format

    def _finalize_page_document(
        self, conv_res: ConversionResult, page: Page
    ) -> DoclingDocument:
        response_format = self._response_format()
        response = page.predictions.vlm_response
        predicted_text = response.text if response is not None else ""
        image = page.image or PILImage.new("RGB", (1, 1), "white")
        assert page.size is not None

        if response_format == ResponseFormat.DOCTAGS:
            document = self._doctags_page_document(predicted_text, image)
        elif response_format == ResponseFormat.DOCLANG:
            document = self._doclang_page_document(conv_res, page, predicted_text)
        elif response_format == ResponseFormat.DEEPSEEKOCR_MARKDOWN:
            document = parse_deepseekocr_markdown(
                content=predicted_text,
                original_page_size=page.size,
                page_no=page.page_no,
                filename=conv_res.input.file.name or "file",
                page_image=page.image,
            )
        elif response_format == ResponseFormat.UNLIMITED_OCR_MARKDOWN:
            from docling.utils.deepseekocr_utils import parse_unlimited_ocr_markdown

            document = parse_unlimited_ocr_markdown(
                content=predicted_text,
                original_page_size=page.size,
                page_no=page.page_no,
                filename=conv_res.input.file.name or "file",
                page_image=page.image,
            )
        elif response_format == ResponseFormat.MARKDOWN:
            document = self._convert_text_page(
                conv_res, page, InputFormat.MD, MarkdownDocumentBackend
            )
        elif response_format == ResponseFormat.HTML:
            document = self._convert_text_page(
                conv_res, page, InputFormat.HTML, HTMLDocumentBackend
            )
        elif response_format == ResponseFormat.CHANDRA_HTML:
            from docling.utils.chandra_utils import parse_chandra_html

            document = parse_chandra_html(
                content=predicted_text,
                original_page_size=page.size,
                page_no=page.page_no,
                filename=conv_res.input.file.name or "file",
                page_image=page.image,
            )
        elif response_format == ResponseFormat.DOTS_JSON:
            document = self._dots_page_document(
                conv_res, page, predicted_text, page.image
            )
        else:
            raise RuntimeError(f"Unsupported VLM response format {response_format}")

        self._finalize_page_output(document, page)
        return document

    def _extract_doclang_fragment(self, text: str) -> str | None:
        """Extract the first <doclang>...</doclang> fragment from text."""
        if not text or not _DOCLANG_OPEN_RE.search(text):
            return None
        start = text.find("<doclang")
        if start < 0:
            return None
        end = text.find("</doclang>", start)
        if end < 0:
            return None
        return text[start : end + len("</doclang>")]

    def _doclang_page_document(
        self, conv_res: ConversionResult, page: Page, predicted_text: str
    ) -> DoclingDocument:
        fragment = self._extract_doclang_fragment(predicted_text)
        if fragment is None:
            conv_res.errors.append(
                ErrorItem(
                    component_type=DoclingComponentType.PIPELINE,
                    module_name=self.__class__.__name__,
                    error_message="No <doclang> XML fragment found in VLM response.",
                    category=FailureCategory.INFERENCE_FAILURE,
                    page_no=page.page_no,
                )
            )
            return DoclingDocument(name=f"page_{page.page_no}")

        try:
            return DocLangDocDeserializer().deserialize_str(fragment)
        except Exception as exc:
            conv_res.errors.append(
                ErrorItem(
                    component_type=DoclingComponentType.PIPELINE,
                    module_name=self.__class__.__name__,
                    error_message=f"DoclangDeserializer failed: {exc}",
                    category=FailureCategory.BACKEND_FAILURE,
                    page_no=page.page_no,
                )
            )
            return DoclingDocument(name=f"page_{page.page_no}")

    @staticmethod
    def _doctags_page_document(
        predicted_text: str, image: PILImage.Image
    ) -> DoclingDocument:
        return DoclingDocument.load_from_doctags(
            doctag_document=DocTagsDocument.from_doctags_and_image_pairs(
                [predicted_text], [image]
            )
        )

    def _turn_dt_into_doc(self, conv_res: ConversionResult) -> DoclingDocument:
        page_documents: list[tuple[int, DoclingDocument]] = []
        for page in conv_res.pages:
            response = page.predictions.vlm_response
            document = self._doctags_page_document(
                response.text if response is not None else "",
                page.image or PILImage.new("RGB", (1, 1), "white"),
            )
            self._finalize_page_output(document, page)
            page_documents.append((page.page_no, document))
        return self._concatenate_page_documents(page_documents)

    def _dots_page_document(
        self,
        conv_res: ConversionResult,
        page: Page,
        predicted_text: str,
        page_image: PILImage.Image | None,
    ) -> DoclingDocument:
        from docling.utils.dots_utils import parse_dots_json
        from docling.utils.vlm_utils import compute_qwen2vl_image_size

        vlm_options = self.pipeline_options.vlm_options
        if isinstance(vlm_options, (VlmConvertOptions, BaseVlmOptions)):
            vlm_scale = vlm_options.scale
            vlm_max_size = vlm_options.max_size
        else:
            raise TypeError(
                "DOTS JSON parsing requires VlmConvertOptions or BaseVlmOptions, "
                f"got {type(vlm_options).__name__}."
            )

        assert page.size is not None
        inference_image = page.get_image(scale=vlm_scale, max_size=vlm_max_size)
        model_image_size = None
        if inference_image is not None:
            model_image_size = compute_qwen2vl_image_size(
                width=inference_image.width,
                height=inference_image.height,
                scale=1.0,
                max_size=None,
            )
        return parse_dots_json(
            content=predicted_text,
            original_page_size=page.size,
            page_no=page.page_no,
            filename=conv_res.input.file.name or "file",
            page_image=page_image,
            model_image_size=model_image_size,
        )

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

    def _convert_text_page(
        self,
        conv_res: ConversionResult,
        page: Page,
        input_format: InputFormat,
        backend_class: type[DeclarativeDocumentBackend],
    ) -> DoclingDocument:
        response = page.predictions.vlm_response
        predicted_text = self._extract_code_block(
            text=f"{response.text}\n\n" if response is not None else ""
        )
        response_bytes = BytesIO(predicted_text.encode("utf8"))
        out_doc = InputDocument(
            path_or_stream=response_bytes,
            filename=conv_res.input.file.name,
            format=input_format,
            backend=backend_class,
        )
        backend = backend_class(in_doc=out_doc, path_or_stream=response_bytes)
        try:
            document = backend.convert()
        finally:
            backend.unload()

        for item, _level in document.iterate_items(
            with_groups=True,
            traverse_pictures=True,
            included_content_layers=set(ContentLayer),
        ):
            if isinstance(item, DocItem):
                item.prov = [
                    ProvenanceItem(
                        page_no=page.page_no,
                        bbox=BoundingBox(t=0.0, b=0.0, l=0.0, r=0.0),
                        charspan=(0, 0),
                    )
                ]
        return document

    def _finalize_page_output(self, document: DoclingDocument, page: Page) -> None:
        if not document.pages:
            document.add_page(page_no=1, size=page.size or Size(width=1, height=1))
        else:
            page_item = next(iter(document.pages.values()))
            page_item.page_no = 1
            document.pages = {1: page_item}
        for item, _level in document.iterate_items():
            if isinstance(item, DocItem):
                for provenance in item.prov:
                    provenance.page_no = 1

        image = page.image
        if self.force_backend_text:
            assert page.size is not None
            scale = self.pipeline_options.images_scale
            for element, _level in document.iterate_items():
                if not isinstance(element, TextItem) or not element.prov:
                    continue
                crop_bbox = (
                    element.prov[0]
                    .bbox.scaled(scale=scale)
                    .to_top_left_origin(page_height=page.size.height * scale)
                )
                text = self.extract_text_from_backend(page, crop_bbox)
                element.text = text
                element.orig = text

        if self.pipeline_options.generate_picture_images and image is not None:
            assert page.size is not None
            scale = self.pipeline_options.images_scale
            for element, _level in document.iterate_items():
                if not isinstance(element, PictureItem) or not element.prov:
                    continue
                crop_bbox = (
                    element.prov[0]
                    .bbox.scaled(scale=scale)
                    .to_top_left_origin(page_height=page.size.height * scale)
                )
                element.image = ImageRef.from_pil(
                    image.crop(crop_bbox.as_tuple()), dpi=int(72 * scale)
                )

        for page_item in document.pages.values():
            page_item.image = (
                ImageRef.from_pil(
                    image=image, dpi=int(72 * self.pipeline_options.images_scale)
                )
                if self.pipeline_options.generate_page_images and image is not None
                else None
            )

    @staticmethod
    def _concatenate_page_documents(
        page_documents: list[tuple[int, DoclingDocument]],
    ) -> DoclingDocument:
        if not page_documents:
            return DoclingDocument(name="")

        document = DoclingDocument.concatenate(
            docs=[page_document for _, page_document in page_documents]
        )
        page_no_map = {
            current_page_no: requested_page_no
            for current_page_no, (requested_page_no, _) in zip(
                sorted(document.pages), page_documents
            )
        }
        document.pages = {
            page_no_map[page_no]: page_item
            for page_no, page_item in document.pages.items()
        }
        for page_no, page_item in document.pages.items():
            page_item.page_no = page_no
        for item, _level in document.iterate_items():
            if isinstance(item, DocItem):
                for provenance in item.prov:
                    provenance.page_no = page_no_map[provenance.page_no]
        return document

    @classmethod
    def get_default_options(cls) -> VlmPipelineOptions:
        return VlmPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return isinstance(backend, PdfDocumentBackend)
