# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace

from docling_core.types.doc import DocItem, DoclingDocument
from PIL.Image import Image

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    VlmPrediction,
    VlmStopReason,
)
from docling.datamodel.document import InputDocument
from docling.datamodel.extraction import (
    ContentItem,
    DocumentExtractionResult,
    DocumentScope,
    ExtractionItem,
    ExtractionScope,
    ExtractionTarget,
    ExtractionTemplateType,
    ImageContentItem,
    PageScope,
    TextContentItem,
    VlmInferenceMetadata,
)
from docling.datamodel.extraction_options import ChannelSelection, ExtractionVlmOptions
from docling.datamodel.pipeline_options import (
    PipelineOptions,
    VlmExtractionPipelineOptions,
)
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
from docling.models.base_model import BaseVlmModel, SupportsContentExtraction
from docling.models.extraction.api_extraction_model import ApiExtractionVlmModel
from docling.models.extraction.prompt_utils import (
    _prepare_normalized_target,
    _PreparedTarget,
    prepare_legacy_target,
    prepare_output_target,
    prepared_image_prompt,
)
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.pipeline.base_extraction_pipeline import BaseExtractionPipeline


@dataclass(frozen=True)
class ExtractionChunk:
    scope: ExtractionScope
    content: list[ContentItem]


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
        ext_res: DocumentExtractionResult,
        target: ExtractionTarget | str | None = None,
    ) -> DocumentExtractionResult:
        started_at = time.monotonic()
        prepared = (
            _prepare_normalized_target(
                target, self.pipeline_options.vlm_options.model_spec
            )
            if isinstance(target, ExtractionTarget)
            else self._prepare_target(target)
        )
        prepared = prepare_output_target(
            prepared,
            self.pipeline_options.vlm_options.output_mode,
            self.pipeline_options.vlm_options.engine_options.engine_type,
        )
        chunks = self._iter_extraction_chunks(ext_res, prepared, started_at)
        try:
            for chunk in chunks:
                remaining = self._remaining_time(started_at)
                if remaining is not None and remaining <= 0:
                    ext_res.items.append(
                        self._failed_item(
                            chunk.scope, prepared, "Document processing timeout"
                        )
                    )
                    self._record_timeout(ext_res)
                    continue
                predictions: list[VlmPrediction] = []
                try:
                    assert isinstance(self.vlm_model, SupportsContentExtraction)
                    # Consume lazy predictions while the producer still owns the current page.
                    predictions.extend(
                        self.vlm_model.process(
                            [chunk.content],
                            replace(prepared, request_timeout=remaining),
                        )
                    )
                    item = self._prediction_to_item(chunk.scope, predictions, prepared)
                except Exception as exc:
                    item = self._prediction_to_item(
                        chunk.scope, predictions, prepared, inference_error=str(exc)
                    )
                ext_res.items.append(item)
                remaining = self._remaining_time(started_at)
                if remaining is not None and remaining <= 0:
                    self._record_timeout(ext_res)
        finally:
            chunks.close()
        # Some sequential backends yield pages out of numerical order.
        ext_res.items.sort(
            key=lambda item: (
                item.scope.page_no if isinstance(item.scope, PageScope) else 0
            )
        )
        return ext_res

    def _remaining_time(self, started_at: float) -> float | None:
        timeout = self.pipeline_options.document_timeout
        return None if timeout is None else timeout - (time.monotonic() - started_at)

    def _record_timeout(self, result: DocumentExtractionResult) -> None:
        if not any(
            error.category == FailureCategory.TIMEOUT for error in result.errors
        ):
            result.errors.append(
                ErrorItem(
                    component_type=DoclingComponentType.PIPELINE,
                    module_name=self.__class__.__name__,
                    error_message="Document processing timeout: selected input may be incomplete; local generation is not preempted.",
                    category=FailureCategory.TIMEOUT,
                )
            )

    def _resolve_channel(
        self, input_doc: InputDocument, doc: DoclingDocument | None = None
    ) -> ChannelSelection:
        backend = input_doc._backend
        if doc is None and isinstance(backend, DeclarativeDocumentBackend):
            doc = backend.convert()
        paginated = isinstance(backend, PdfDocumentBackend) or bool(
            doc is not None and doc.pages
        )
        offers_text = isinstance(
            backend, (PdfDocumentBackend, DeclarativeDocumentBackend)
        )
        offers_image = isinstance(backend, PdfDocumentBackend)
        if doc is not None and doc.pages:
            start, end = input_doc.limits.page_range
            offers_image = any(
                page.image is not None
                for page_no, page in doc.pages.items()
                if start <= page_no <= end
            )
        selection = self.pipeline_options.input_channels
        spec = self.pipeline_options.vlm_options.model_spec
        if selection == ChannelSelection.AUTO:
            if paginated and offers_image and spec.accepts_image:
                resolved = ChannelSelection.IMAGE
            elif offers_text and spec.accepts_text:
                resolved = ChannelSelection.TEXT
            else:
                raise ValueError(
                    f"No channel works for format {input_doc.format} with model '{spec.name}'"
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
        if needs_image and not paginated:
            raise ValueError(
                f"{resolved.value} channel requested but source does not offer page images: unpaginated sources require text"
            )
        # A paginated document with missing images is handled as scoped load failures.
        if needs_text and not offers_text:
            raise ValueError(
                f"{resolved.value} channel requested but source does not offer a text payload"
            )
        if needs_image and not spec.accepts_image:
            raise ValueError(
                f"{resolved.value} channel requested but model '{spec.name}' does not accept an image payload"
            )
        if needs_text and not spec.accepts_text:
            raise ValueError(
                f"{resolved.value} channel requested but model '{spec.name}' does not accept a text payload"
            )
        return resolved

    def _iter_extraction_chunks(
        self,
        result: DocumentExtractionResult,
        target: _PreparedTarget,
        started_at: float,
    ) -> Generator[ExtractionChunk, None, None]:
        input_doc = result.input
        backend = input_doc._backend
        doc = (
            backend.convert()
            if isinstance(backend, DeclarativeDocumentBackend)
            else None
        )
        pdf = isinstance(backend, PdfDocumentBackend)
        paginated = pdf or bool(doc is not None and doc.pages)
        if doc is not None and len(doc.pages) > input_doc.limits.max_num_pages:
            raise ValueError("Converted document exceeds the max_num_pages input limit")
        if not paginated:
            if target.target is None:
                raise ValueError(
                    "Legacy template= requires page-representable input; use target= for unpaginated sources"
                )
            if input_doc.limits.page_range != DEFAULT_PAGE_RANGE:
                raise ValueError(
                    "Unpaginated sources do not support a non-default page range"
                )
        channel = self._resolve_channel(input_doc, doc)
        needs_text = channel in (ChannelSelection.TEXT, ChannelSelection.IMAGE_AND_TEXT)
        start, end = input_doc.limits.page_range
        if pdf:
            assert isinstance(backend, PdfDocumentBackend)
            selected = list(range(max(1, start), min(backend.page_count(), end) + 1))
        elif paginated:
            assert doc is not None
            selected = [n for n in sorted(doc.pages) if start <= n <= end]
        else:
            selected = []
        attribution_error = None
        if paginated and needs_text and doc is not None:
            for item, _ in doc.iterate_items():
                if isinstance(item, DocItem) and (
                    not item.prov
                    or len({prov.page_no for prov in item.prov}) != 1
                    or any(prov.page_no not in doc.pages for prov in item.prov)
                ):
                    attribution_error = "Unsupported page attribution: document content lacks reliable single-page provenance"
                    break
        if not paginated:
            scope = DocumentScope()
            try:
                remaining = self._remaining_time(started_at)
                if remaining is not None and remaining <= 0:
                    self._record_timeout(result)
                    raise TimeoutError("Document processing timeout")
                assert doc is not None
                text = (
                    backend.markdown
                    if isinstance(backend, MarkdownDocumentBackend)
                    else self._serialize_doc(doc)
                )
                if not text.strip():
                    raise ValueError("No text found in unpaginated document")
                yield ExtractionChunk(scope, [TextContentItem(text=text)])
            except Exception as exc:
                result.items.append(self._failed_item(scope, target, str(exc)))
            return
        if not selected:
            raise ValueError("No pages selected for extraction")
        pending = set(selected)
        page_index = 0
        page_iterator = None
        if pdf:
            assert isinstance(backend, PdfDocumentBackend)
            if not backend.supports_random_page_access:
                page_iterator = backend.iter_pages()
        try:
            while pending:
                page_backend = None
                while selected[page_index] not in pending:
                    page_index += 1
                page_no = selected[page_index]
                scope = PageScope(page_no=page_no)
                try:
                    remaining = self._remaining_time(started_at)
                    if remaining is not None and remaining <= 0:
                        self._record_timeout(result)
                        raise TimeoutError(
                            "Document processing timeout: selected page not processed"
                        )
                    if attribution_error:
                        raise ValueError(attribution_error)
                    if pdf:
                        assert isinstance(backend, PdfDocumentBackend)
                        if page_iterator is None:
                            page_backend = backend.load_page(page_no - 1)
                        else:
                            page_backend = next(page_iterator)
                            page_no = page_backend.page_no
                            if page_no not in pending:
                                continue
                            scope = PageScope(page_no=page_no)
                        if not page_backend.is_valid():
                            raise ValueError(
                                f"Page {page_no} backend is not valid: {page_backend.get_error_message()}"
                            )
                    with self._page_content(
                        page_no, page_backend, doc, channel
                    ) as content:
                        pending.remove(page_no)
                        yield ExtractionChunk(scope, content)
                except StopIteration:
                    for missing in sorted(pending):
                        result.items.append(
                            self._failed_item(
                                PageScope(page_no=missing),
                                target,
                                "Selected page is missing from backend",
                            )
                        )
                    pending.clear()
                except Exception as exc:
                    result.items.append(self._failed_item(scope, target, str(exc)))
                    pending.discard(page_no)
                finally:
                    if page_backend is not None:
                        page_backend.unload()
        finally:
            if isinstance(page_iterator, Generator):
                page_iterator.close()

    @contextmanager
    def _page_content(
        self,
        page_no: int,
        page_backend: PdfPageBackend | None,
        doc: DoclingDocument | None,
        channel: ChannelSelection,
    ) -> Generator[list[ContentItem], None, None]:
        image: Image | None = None
        owned_image = False
        needs_image = channel in (
            ChannelSelection.IMAGE,
            ChannelSelection.IMAGE_AND_TEXT,
        )
        needs_text = channel in (ChannelSelection.TEXT, ChannelSelection.IMAGE_AND_TEXT)
        content: list[ContentItem] = []
        try:
            if needs_image:
                if page_backend is not None:
                    scale = self.pipeline_options.vlm_options.scale
                    max_size = self.pipeline_options.vlm_options.max_size
                    if max_size is not None:
                        size = page_backend.get_size()
                        scale = min(scale, max_size / max(size.width, size.height))
                    image = page_backend.get_page_image(scale=scale)
                    owned_image = True
                else:
                    assert doc is not None
                    page = doc.pages[page_no]
                    if page.image is None or page.image.pil_image is None:
                        raise ValueError(f"Page {page_no} has no restored image")
                    image = page.image.pil_image
                    max_size = self.pipeline_options.vlm_options.max_size
                    if max_size is not None and max(image.size) > max_size:
                        image = image.copy()
                        owned_image = True
                        image.thumbnail((max_size, max_size))
                content.append(ImageContentItem(image=image))
            if needs_text:
                if page_backend is not None:
                    text = "\n".join(
                        cell.text for cell in page_backend.get_text_cells()
                    )
                else:
                    assert doc is not None
                    text = self._serialize_doc(doc, pages={page_no})
                if not text.strip() and not needs_image:
                    raise ValueError(f"Page {page_no} has no text content")
                content.append(TextContentItem(text=text))
            yield content
        finally:
            if image is not None and owned_image:
                image.close()

    def _item_error(self, scope: ExtractionScope, message: str) -> ErrorItem:
        """A scoped item-level failure; page_no carries the scope for free."""
        return ErrorItem(
            component_type=DoclingComponentType.MODEL,
            module_name=self.__class__.__name__,
            error_message=message,
            category=FailureCategory.INFERENCE_FAILURE,
            page_no=scope.page_no if isinstance(scope, PageScope) else None,
        )

    def _failed_item(
        self, scope: ExtractionScope, target: _PreparedTarget, message: str
    ) -> ExtractionItem:
        return ExtractionItem(
            scope=scope,
            errors=[self._item_error(scope, message)],
            validation_status="not_run"
            if target.validator is not None
            else "not_requested",
        )

    def _prediction_to_item(
        self,
        scope: ExtractionScope,
        predictions: list[VlmPrediction],
        target: _PreparedTarget,
        *,
        inference_error: str | None = None,
    ) -> ExtractionItem:
        if len(predictions) != 1:
            return self._failed_item(
                scope,
                target,
                inference_error
                or "Expected exactly one extraction result from VLM model",
            )
        prediction = predictions[0]
        item = ExtractionItem(
            scope=scope,
            raw_text=prediction.text,
            inference_metadata=VlmInferenceMetadata(
                generation_time=prediction.generation_time,
                num_tokens=prediction.num_tokens,
                usage=prediction.usage,
                stop_reason=prediction.stop_reason,
            ),
            validation_status="not_run"
            if target.validator is not None
            else "not_requested",
        )
        if inference_error is not None:
            item.errors.append(self._item_error(scope, inference_error))
            return item
        try:
            data = json.loads(
                prediction.text, parse_constant=self._reject_json_constant
            )
            json.dumps(data, allow_nan=False)
            if not isinstance(data, dict):
                raise ValueError("Model returned JSON that is not an object")
        except ValueError as exc:
            item.errors.append(
                self._item_error(scope, f"Model returned invalid JSON: {exc}")
            )
            return item
        if prediction.stop_reason == VlmStopReason.CONTENT_FILTERED:
            item.errors.append(
                self._item_error(scope, "Model output was filtered by the API provider")
            )
            return item
        if target.validator is not None:
            try:
                errors = list(target.validator.iter_errors(data))
            except Exception as exc:
                item.errors.append(
                    self._item_error(scope, f"Schema validation could not run: {exc}")
                )
                return item
            item.validation_status = "failed" if errors else "passed"
            item.errors.extend(
                self._item_error(
                    scope, f"Schema validation at {error.json_path}: {error.message}"
                )
                for error in errors
            )
            if errors:
                return item
        item.extracted_data = data
        return item

    @staticmethod
    def _reject_json_constant(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value {value}")

    def _determine_status(self, ext_res: DocumentExtractionResult) -> ConversionStatus:
        if not any(item.extracted_data is not None for item in ext_res.items):
            return ConversionStatus.FAILURE
        if ext_res.errors or any(
            item.errors
            or (
                item.inference_metadata is not None
                and item.inference_metadata.stop_reason
                in {VlmStopReason.LENGTH, VlmStopReason.STOP_SEQUENCE}
            )
            for item in ext_res.items
        ):
            return ConversionStatus.PARTIAL_SUCCESS
        return ConversionStatus.SUCCESS

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

    def _prepare_target(
        self, template: ExtractionTemplateType | None
    ) -> _PreparedTarget:
        if template is None:
            return prepared_image_prompt(
                "Extract all text and structured information from this document. Return as JSON.",
                self.pipeline_options.vlm_options.model_spec,
            )
        return prepare_legacy_target(
            template, self.pipeline_options.vlm_options.model_spec
        )

    @classmethod
    def get_default_options(cls) -> PipelineOptions:
        return VlmExtractionPipelineOptions()
