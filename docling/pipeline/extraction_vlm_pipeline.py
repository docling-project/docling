import inspect
import json
import logging
from typing import Optional

from PIL.Image import Image
from pydantic import BaseModel

from docling.backend.abstract_backend import PaginatedDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import ConversionStatus, ErrorItem, VlmStopReason
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
from docling.models.extraction.nuextract_transformers_model import (
    NuExtractTransformersModel,
)
from docling.pipeline.base_extraction_pipeline import BaseExtractionPipeline
from docling.utils.accelerator_utils import decide_device

_log = logging.getLogger(__name__)


class ExtractionVlmPipeline(BaseExtractionPipeline):
    def __init__(self, pipeline_options: VlmExtractionPipelineOptions):
        super().__init__(pipeline_options)

        # Initialize VLM model with default options
        self.accelerator_options = pipeline_options.accelerator_options
        self.pipeline_options: VlmExtractionPipelineOptions

        # Create VLM model instance
        self.vlm_model = NuExtractTransformersModel(
            enabled=True,
            artifacts_path=self.artifacts_path,  # Will download automatically
            accelerator_options=self.accelerator_options,
            vlm_options=pipeline_options.vlm_options,
        )

    def _extract_data(
        self,
        ext_res: ExtractionResult,
        template: Optional[ExtractionTemplateType] = None,
    ) -> ExtractionResult:
        """Extract data using the VLM model."""
        try:
            # Get images from input document using the backend
            images = self._get_images_from_input(ext_res.input)
            if not images:
                ext_res.status = ConversionStatus.FAILURE
                ext_res.errors.append(
                    ErrorItem(
                        component_type="extraction_pipeline",
                        module_name=self.__class__.__name__,
                        error_message="No images found in document",
                    )
                )
                return ext_res

            # Use provided template or default prompt
            if template is not None:
                prompt = self._serialize_template(template)
            else:
                prompt = "Extract all text and structured information from this document. Return as JSON."

            # Process all images with VLM model
            start_page, end_page = ext_res.input.limits.page_range
            for i, image in enumerate(images):
                # Calculate the actual page number based on the filtered range
                page_number = start_page + i
                try:
                    predictions = list(self.vlm_model.process_images([image], prompt))

                    if predictions:
                        # Parse the extracted text as JSON if possible, otherwise use as-is
                        extracted_text = predictions[0].text
                        extracted_data = None
                        vlm_stop_reason: VlmStopReason = predictions[0].stop_reason
                        if (
                            vlm_stop_reason == VlmStopReason.LENGTH
                            or vlm_stop_reason == VlmStopReason.STOP_SEQUENCE
                        ):
                            ext_res.status = ConversionStatus.PARTIAL_SUCCESS

                        try:
                            extracted_data = json.loads(extracted_text)
                        except (json.JSONDecodeError, ValueError):
                            # If not valid JSON, keep extracted_data as None
                            pass

                        # Create page data with proper structure
                        page_data = ExtractedPageData(
                            page_no=page_number,
                            extracted_data=extracted_data,
                            raw_text=extracted_text,  # Always populate raw_text
                        )
                        ext_res.pages.append(page_data)
                    else:
                        # Add error page data
                        page_data = ExtractedPageData(
                            page_no=page_number,
                            extracted_data=None,
                            errors=["No extraction result from VLM model"],
                        )
                        ext_res.pages.append(page_data)

                except Exception as e:
                    _log.error(f"Error processing page {page_number}: {e}")
                    page_data = ExtractedPageData(
                        page_no=page_number, extracted_data=None, errors=[str(e)]
                    )
                    ext_res.pages.append(page_data)

        except Exception as e:
            _log.error(f"Error during extraction: {e}")
            ext_res.errors.append(
                ErrorItem(
                    component_type="extraction_pipeline",
                    module_name=self.__class__.__name__,
                    error_message=str(e),
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

    def _get_images_from_input(self, input_doc: InputDocument) -> list[Image]:
        """Extract images from input document using the backend."""
        images = []

        try:
            backend = input_doc._backend

            assert isinstance(backend, PdfDocumentBackend)
            # Use the backend's pagination interface
            page_count = backend.page_count()

            # Respect page range limits, following the same pattern as PaginatedPipeline
            start_page, end_page = input_doc.limits.page_range
            _log.info(
                f"Processing pages {start_page}-{end_page} of {page_count} total pages for extraction"
            )

            for page_num in range(page_count):
                # Only process pages within the specified range (0-based indexing)
                if start_page - 1 <= page_num <= end_page - 1:
                    try:
                        page_backend = backend.load_page(page_num)
                        if page_backend.is_valid():
                            # Get page image at a reasonable scale
                            page_image = page_backend.get_page_image(
                                scale=self.pipeline_options.vlm_options.scale
                            )
                            images.append(page_image)
                        else:
                            _log.warning(f"Page {page_num + 1} backend is not valid")
                    except Exception as e:
                        _log.error(f"Error loading page {page_num + 1}: {e}")

        except Exception as e:
            _log.error(f"Error getting images from input document: {e}")

        return images

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
