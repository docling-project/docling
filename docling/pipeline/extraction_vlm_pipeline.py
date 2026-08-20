import inspect
import json
import logging
import time
from collections.abc import Generator
from typing import Optional

from PIL.Image import Image
from pydantic import BaseModel

from docling.backend.pdf_backend import PdfDocumentBackend, iter_pdf_page_backends
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
    VlmStopReason,
)
from docling.datamodel.document import InputDocument
from docling.datamodel.extraction import (
    ExtractedPageData,
    ExtractionResult,
    ExtractionTemplateType,
)
from docling.datamodel.pipeline_options import (
    PipelineOptions,
    VlmExtractionPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.models.extraction.transformers_extraction_model import (
    TransformersExtractionModel,
)
from docling.pipeline.base_extraction_pipeline import BaseExtractionPipeline
from docling.utils.accelerator_utils import decide_device

_log = logging.getLogger(__name__)


class ExtractionVlmPipeline(BaseExtractionPipeline):
    def __init__(self, pipeline_options: VlmExtractionPipelineOptions):
        super().__init__(pipeline_options)

        self.accelerator_options = pipeline_options.accelerator_options
        self.pipeline_options: VlmExtractionPipelineOptions

        self.vlm_model = TransformersExtractionModel(
            enabled=True,
            artifacts_path=self.artifacts_path,
            accelerator_options=self.accelerator_options,
            vlm_options=pipeline_options.vlm_options,
            prompt_style=pipeline_options.extraction_prompt_style,
        )

    def _extract_data(
        self,
        ext_res: ExtractionResult,
        template: Optional[ExtractionTemplateType] = None,
    ) -> ExtractionResult:
        """Extract data using the VLM model."""
        try:
            images = self._get_images_from_input(ext_res.input)
            if template is not None:
                prompt = self._serialize_template(template)
            else:
                prompt = "Extract all text and structured information from this document. Return as JSON."

            processed_image = False
            started_at = time.monotonic()
            try:
                for page_number, image in images:
                    processed_image = True
                    try:
                        predictions = list(
                            self.vlm_model.process_images([image], prompt)
                        )
                        if predictions:
                            extracted_text = predictions[0].text
                            extracted_data = None
                            vlm_stop_reason: VlmStopReason = predictions[0].stop_reason
                            if vlm_stop_reason in {
                                VlmStopReason.LENGTH,
                                VlmStopReason.STOP_SEQUENCE,
                            }:
                                ext_res.status = ConversionStatus.PARTIAL_SUCCESS

                            try:
                                extracted_data = json.loads(extracted_text)
                            except (json.JSONDecodeError, ValueError):
                                pass

                            page_data = ExtractedPageData(
                                page_no=page_number,
                                extracted_data=extracted_data,
                                raw_text=extracted_text,
                            )
                        else:
                            page_data = ExtractedPageData(
                                page_no=page_number,
                                extracted_data=None,
                                errors=["No extraction result from VLM model"],
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

            ext_res.pages.sort(key=lambda page: page.page_no)

        except Exception as e:
            _log.error(f"Error during extraction: {e}")
            ext_res.errors.append(
                ErrorItem(
                    component_type=DoclingComponentType.PIPELINE,
                    module_name=self.__class__.__name__,
                    error_message=str(e),
                    category=FailureCategory.UNKNOWN,
                )
            )

        return ext_res

    def _determine_status(self, ext_res: ExtractionResult) -> ConversionStatus:
        """Determine the status based on extraction results."""
        if ext_res.pages and not any(page.errors for page in ext_res.pages):
            return (
                ConversionStatus.PARTIAL_SUCCESS
                if ext_res.status == ConversionStatus.PARTIAL_SUCCESS
                else ConversionStatus.SUCCESS
            )
        else:
            return ConversionStatus.FAILURE

    def _get_images_from_input(
        self, input_doc: InputDocument
    ) -> Generator[tuple[int, Image], None, None]:
        """Yield one rendered page at a time and release it before advancing."""
        page_iterator = None
        try:
            backend = input_doc._backend
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
                    page_image = page_backend.get_page_image(
                        scale=self.pipeline_options.vlm_options.scale
                    )
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

    def _serialize_template(self, template: ExtractionTemplateType) -> str:
        """Serialize template to string based on its type."""
        if isinstance(template, str):
            return template
        elif isinstance(template, dict):
            return json.dumps(template, indent=2)
        elif isinstance(template, BaseModel):
            return template.model_dump_json(indent=2)
        elif inspect.isclass(template) and issubclass(template, BaseModel):
            from polyfactory.factories.pydantic_factory import ModelFactory

            class ExtractionTemplateFactory(ModelFactory[template]):  # type: ignore
                __use_examples__ = True  # prefer Field(examples=...) when present
                __use_defaults__ = True  # use field defaults instead of random values
                __check_model__ = (
                    True  # setting the value to avoid deprecation warnings
                )

            return ExtractionTemplateFactory.build().model_dump_json(indent=2)  # type: ignore
        else:
            raise ValueError(f"Unsupported template type: {type(template)}")

    @classmethod
    def get_default_options(cls) -> PipelineOptions:
        return VlmExtractionPipelineOptions()
