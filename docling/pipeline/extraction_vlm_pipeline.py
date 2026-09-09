# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import json
import logging
import time
from collections.abc import Generator

from docling_core.types.doc import DoclingDocument
from PIL.Image import Image

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend, iter_pdf_page_backends
from docling.backend.xml.doclang_archive_backend import DocLangArchiveBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    InputFormat,
    VlmPrediction,
    VlmStopReason,
)
from docling.datamodel.document import InputDocument
from docling.datamodel.extraction import (
    ContentItem,
    ExtractedPageData,
    ExtractionResult,
    ExtractionTemplateType,
    ImageContentItem,
    TextContentItem,
)
from docling.datamodel.extraction_options import (
    ChannelSelection,
    ExtractionVlmOptions,
)
from docling.datamodel.pipeline_options import (
    PipelineOptions,
    VlmExtractionPipelineOptions,
)
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
from docling.models.base_model import BaseVlmModel, SupportsContentExtraction
from docling.models.extraction.api_extraction_model import ApiExtractionVlmModel
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.pipeline.base_extraction_pipeline import BaseExtractionPipeline

_log = logging.getLogger(__name__)


class ExtractionVlmPipeline(BaseExtractionPipeline):
    def __init__(self, pipeline_options: VlmExtractionPipelineOptions):
        super().__init__(pipeline_options)

        self.pipeline_options: VlmExtractionPipelineOptions
        vlm_options: ExtractionVlmOptions = pipeline_options.vlm_options
        self.vlm_model: BaseVlmModel

        engine_type = vlm_options.engine_options.engine_type
        if VlmEngineType.is_api_variant(engine_type):
            self.vlm_model = ApiExtractionVlmModel(
                enabled=True,
                enable_remote_services=pipeline_options.enable_remote_services,
                vlm_options=vlm_options,
            )
        else:
            from docling.models.extraction.transformers_extraction_model import (
                TransformersExtractionModel,
            )

            self.vlm_model = TransformersExtractionModel(
                enabled=True,
                artifacts_path=self.artifacts_path,
                accelerator_options=pipeline_options.accelerator_options,
                vlm_options=vlm_options,
            )

    def _extract_data(
        self,
        ext_res: ExtractionResult,
        template: ExtractionTemplateType | None = None,
    ) -> ExtractionResult:
        prompt = self._build_prompt(template)
        channel = self._resolve_channel(ext_res.input)

        if channel == ChannelSelection.TEXT:
            self._extract_via_text(ext_res, prompt)
        else:
            self._extract_per_page(
                ext_res,
                prompt,
                include_text=channel == ChannelSelection.IMAGE_AND_TEXT,
            )

        ext_res.pages.sort(key=lambda page: page.page_no)
        return ext_res

    def _resolve_channel(self, input_doc: InputDocument) -> ChannelSelection:
        """Resolve a channel supported by both the source and model."""
        backend = input_doc._backend
        offers_image = isinstance(backend, PdfDocumentBackend)
        offers_text = isinstance(backend, DeclarativeDocumentBackend)
        if isinstance(backend, DocLangArchiveBackend):
            doc = backend.convert()
            start_page, end_page = input_doc.limits.page_range
            pages = [
                doc.pages[page_no]
                for page_no in sorted(doc.pages)
                if start_page <= page_no <= end_page
            ]
            offers_image = bool(pages) and all(
                page.image is not None and page.image.pil_image is not None
                for page in pages
            )

        selection = self.pipeline_options.input_channels
        spec = self.pipeline_options.vlm_options.model_spec
        accepts_image = spec.accepts_image
        accepts_text = spec.accepts_text

        if selection == ChannelSelection.AUTO:
            if offers_image and accepts_image:
                resolved = ChannelSelection.IMAGE
            elif offers_text and accepts_text:
                resolved = ChannelSelection.TEXT
            else:
                raise ValueError(
                    f"No channel works for format {input_doc.format} with model "
                    f"'{spec.name}': format offers "
                    f"{'image' if offers_image else ''}"
                    f"{'+' if offers_image and offers_text else ''}"
                    f"{'text' if offers_text else ''}, model accepts "
                    f"{'image' if accepts_image else ''}"
                    f"{'+' if accepts_image and accepts_text else ''}"
                    f"{'text' if accepts_text else ''}."
                )
        else:
            resolved = selection

        needs_image = resolved in (
            ChannelSelection.IMAGE,
            ChannelSelection.IMAGE_AND_TEXT,
        )
        needs_text = resolved in (
            ChannelSelection.TEXT,
            ChannelSelection.IMAGE_AND_TEXT,
        )
        if needs_image and not offers_image:
            raise ValueError(
                f"{resolved.value} channel requested but format {input_doc.format} "
                f"does not offer page images."
            )
        if needs_text and not offers_text:
            raise ValueError(
                f"{resolved.value} channel requested but format {input_doc.format} "
                f"does not offer a text payload."
            )
        if needs_image and not accepts_image:
            raise ValueError(
                f"{resolved.value} channel requested but model '{spec.name}' does "
                f"not accept an image payload."
            )
        if needs_text and not accepts_text:
            raise ValueError(
                f"{resolved.value} channel requested but model '{spec.name}' does "
                f"not accept a text payload."
            )
        return resolved

    def _extract_via_text(self, ext_res: ExtractionResult, prompt: str) -> None:
        text = self._get_text_from_input(ext_res.input)
        request: list[ContentItem] = [TextContentItem(text=text)]
        assert isinstance(self.vlm_model, SupportsContentExtraction)
        try:
            predictions = list(self.vlm_model.process([request], prompt))
        except Exception as e:
            _log.error(f"Error processing text document: {e}")
            ext_res.pages.append(
                ExtractedPageData(page_no=1, extracted_data=None, errors=[str(e)])
            )
            return

        ext_res.pages.append(self._prediction_to_page_data(1, predictions, ext_res))

    def _get_text_from_input(self, input_doc: InputDocument) -> str:
        """Serialize the selected source text and page range as Markdown."""
        backend = input_doc._backend
        if input_doc.format == InputFormat.MD:
            from docling.backend.md_backend import MarkdownDocumentBackend

            assert isinstance(backend, MarkdownDocumentBackend)
            return backend.markdown

        assert isinstance(backend, DeclarativeDocumentBackend)
        doc = backend.convert()

        start_page, end_page = input_doc.limits.page_range
        pages: set[int] | None = None
        if (start_page, end_page) != DEFAULT_PAGE_RANGE and doc.pages:
            pages = {p for p in doc.pages if start_page <= p <= end_page}
        return self._serialize_doc(doc, pages=pages)

    def _serialize_doc(
        self, doc: DoclingDocument, pages: set[int] | None = None
    ) -> str:
        """Serialize a document or page subset to Markdown."""
        params = self.pipeline_options.markdown_params
        if params is None:
            if pages is None:
                return doc.export_to_markdown()
            return "\n\n".join(
                doc.export_to_markdown(page_no=page_no) for page_no in sorted(pages)
            )

        if pages is not None:
            params = params.model_copy(update={"pages": set(pages)})

        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

        return MarkdownDocSerializer(doc=doc, params=params).serialize().text

    def _extract_per_page(
        self, ext_res: ExtractionResult, prompt: str, *, include_text: bool
    ) -> None:
        """Extract one result per page image."""
        doc: DoclingDocument | None = None
        if include_text:
            backend = ext_res.input._backend
            assert isinstance(backend, DeclarativeDocumentBackend)
            doc = backend.convert()

        images = self._get_images_from_input(ext_res.input)
        processed_image = False
        started_at = time.monotonic()
        try:
            for page_number, image in images:
                processed_image = True
                try:
                    if include_text:
                        assert doc is not None
                        assert isinstance(self.vlm_model, SupportsContentExtraction)
                        request: list[ContentItem] = [
                            ImageContentItem(image=image),
                            TextContentItem(
                                text=self._serialize_doc(doc, pages={page_number})
                            ),
                        ]
                        predictions = list(self.vlm_model.process([request], prompt))
                    else:
                        predictions = list(
                            self.vlm_model.process_images([image], prompt)
                        )
                    page_data = self._prediction_to_page_data(
                        page_number, predictions, ext_res
                    )
                except Exception as e:
                    _log.error(f"Error processing page {page_number}: {e}")
                    page_data = ExtractedPageData(
                        page_no=page_number,
                        extracted_data=None,
                        errors=[str(e)],
                    )
                ext_res.pages.append(page_data)

                timeout = self.pipeline_options.document_timeout
                elapsed = time.monotonic() - started_at
                if timeout is not None and elapsed > timeout:
                    message = (
                        "Document processing timeout: exceeded "
                        f"{timeout:.3f}s limit after {elapsed:.3f}s."
                    )
                    _log.warning(message)
                    ext_res.errors.append(
                        ErrorItem(
                            component_type=DoclingComponentType.PIPELINE,
                            module_name=self.__class__.__name__,
                            error_message=message,
                            category=FailureCategory.TIMEOUT,
                        )
                    )
                    ext_res.status = ConversionStatus.PARTIAL_SUCCESS
                    break
        finally:
            images.close()

        if not processed_image:
            ext_res.status = ConversionStatus.FAILURE
            ext_res.errors.append(
                ErrorItem(
                    component_type=DoclingComponentType.PIPELINE,
                    module_name=self.__class__.__name__,
                    error_message="No images found in document",
                    category=FailureCategory.BACKEND_FAILURE,
                )
            )

    def _prediction_to_page_data(
        self,
        page_no: int,
        predictions: list[VlmPrediction],
        ext_res: ExtractionResult,
    ) -> ExtractedPageData:
        if not predictions:
            return ExtractedPageData(
                page_no=page_no,
                extracted_data=None,
                errors=["No extraction result from VLM model"],
            )

        prediction = predictions[0]
        if prediction.stop_reason in {
            VlmStopReason.LENGTH,
            VlmStopReason.STOP_SEQUENCE,
        }:
            ext_res.status = ConversionStatus.PARTIAL_SUCCESS

        errors: list[str] = []
        extracted_data = None
        try:
            extracted_data = json.loads(prediction.text)
        except json.JSONDecodeError as exc:
            errors.append(f"Model returned invalid JSON: {exc.msg}")
        if extracted_data is not None and not isinstance(extracted_data, dict):
            errors.append("Model returned JSON that is not an object")
            extracted_data = None
        if prediction.stop_reason == VlmStopReason.CONTENT_FILTERED:
            errors.append("Model output was filtered by the API provider")

        return ExtractedPageData(
            page_no=page_no,
            extracted_data=extracted_data,
            raw_text=prediction.text,
            errors=errors,
        )

    def _determine_status(self, ext_res: ExtractionResult) -> ConversionStatus:
        if not any(page.extracted_data is not None for page in ext_res.pages):
            return ConversionStatus.FAILURE
        if (
            ext_res.status == ConversionStatus.PARTIAL_SUCCESS
            or ext_res.errors
            or any(page.errors for page in ext_res.pages)
        ):
            return ConversionStatus.PARTIAL_SUCCESS
        return ConversionStatus.SUCCESS

    def _get_images_from_input(
        self, input_doc: InputDocument
    ) -> Generator[tuple[int, Image], None, None]:
        """Yield ``(page_no, image)`` for each page the source offers.

        PDF/IMAGE pages are rendered on the fly and released before advancing;
        DCLX pages carry images restored from the archive on the DoclingDocument.
        """
        backend = input_doc._backend
        if isinstance(backend, DocLangArchiveBackend):
            yield from self._iter_dclx_page_images(input_doc, backend)
            return

        page_iterator = None
        try:
            assert isinstance(backend, PdfDocumentBackend)
            page_count = backend.page_count()
            start_page, end_page = input_doc.limits.page_range
            _log.info(
                f"Processing pages {start_page}-{end_page} of {page_count} total pages for extraction"
            )
            page_nos = range(max(1, start_page), min(page_count, end_page) + 1)
            page_iterator = iter_pdf_page_backends(backend, page_nos)
            for page_backend in page_iterator:
                page_image = None
                try:
                    if not page_backend.is_valid():
                        _log.warning(
                            f"Page {page_backend.page_no} backend is not valid"
                        )
                        continue
                    render_scale = self.pipeline_options.vlm_options.scale
                    max_size = self.pipeline_options.vlm_options.max_size
                    if max_size is not None:
                        page_size = page_backend.get_size()
                        render_scale = min(
                            render_scale,
                            max_size / max(page_size.width, page_size.height),
                        )
                    page_image = page_backend.get_page_image(scale=render_scale)
                    yield page_backend.page_no, page_image
                except Exception as e:
                    _log.error(f"Error loading page {page_backend.page_no}: {e}")
                finally:
                    if page_image is not None:
                        page_image.close()
                    page_backend.unload()

        except Exception as e:
            _log.error(f"Error getting images from input document: {e}")
        finally:
            if isinstance(page_iterator, Generator):
                page_iterator.close()

    def _iter_dclx_page_images(
        self, input_doc: InputDocument, backend: DocLangArchiveBackend
    ) -> Generator[tuple[int, Image], None, None]:
        """Yield page images restored from a DCLX archive, within the page range."""
        doc = backend.convert()
        start_page, end_page = input_doc.limits.page_range
        for page_no in sorted(doc.pages):
            if not (start_page <= page_no <= end_page):
                continue
            page = doc.pages[page_no]
            if page.image is None or page.image.pil_image is None:
                raise RuntimeError(f"DCLX page {page_no} has no restored image")
            image = page.image.pil_image
            max_size = self.pipeline_options.vlm_options.max_size
            if max_size is None or max(image.size) <= max_size:
                yield page_no, image
                continue
            resized = image.copy()
            try:
                resized.thumbnail((max_size, max_size))
                yield page_no, resized
            finally:
                resized.close()

    def _build_prompt(self, template: ExtractionTemplateType | None) -> str:
        if template is None:
            return "Extract all text and structured information from this document. Return as JSON."

        return self.pipeline_options.vlm_options.build_extraction_prompt(template)

    @classmethod
    def get_default_options(cls) -> PipelineOptions:
        return VlmExtractionPipelineOptions()
