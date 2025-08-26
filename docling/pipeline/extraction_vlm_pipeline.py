import inspect
import json
import logging
from pathlib import Path
from typing import Optional

from PIL.Image import Image
from pydantic import BaseModel

from docling.backend.abstract_backend import PaginatedDocumentBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ExtractionResult, InputDocument
from docling.datamodel.pipeline_options import VlmExtractionPipelineOptions
from docling.models.vlm_models_inline.nuextract_transformers_model import (
    NuExtractTransformersModel,
)
from docling.pipeline.base_extraction_pipeline import BaseExtractionPipeline
from docling.types import ExtractionTemplateType
from docling.utils.accelerator_utils import decide_device

_log = logging.getLogger(__name__)


class ExtractionVlmPipeline(BaseExtractionPipeline):
    def __init__(self, pipeline_options: VlmExtractionPipelineOptions):
        super().__init__(pipeline_options)

        # Initialize VLM model with default options
        self.accelerator_options = pipeline_options.accelerator_options

        # Create VLM model instance
        self.vlm_model = NuExtractTransformersModel(
            enabled=True,
            artifacts_path=pipeline_options.artifacts_path,  # Will download automatically
            accelerator_options=self.accelerator_options,
            vlm_options=self.pipeline_options.vlm_options,
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
                ext_res.data = {"error": "No images found in document"}
                return ext_res

            # Use provided template or default prompt
            if template is not None:
                prompt = self._serialize_template(template)
            else:
                prompt = "Extract all text and structured information from this document. Return as JSON."

            # Process all images with VLM model
            all_extracted_data = []
            for i, image in enumerate(images):
                predictions = list(self.vlm_model.process_images([image], prompt))

                if predictions:
                    # Parse the extracted text as JSON if possible, otherwise use as-is
                    extracted_text = predictions[0].text
                    try:
                        import json

                        extracted_data = json.loads(extracted_text)
                        extracted_data["page"] = i + 1  # Add page number
                    except (json.JSONDecodeError, ValueError):
                        # If not valid JSON, store as text
                        extracted_data = {
                            "page": i + 1,
                            "extracted_text": extracted_text,
                        }

                    all_extracted_data.append(extracted_data)
                else:
                    all_extracted_data.append(
                        {"page": i + 1, "error": "No extraction result"}
                    )

            # Combine all page results
            if len(all_extracted_data) == 1:
                ext_res.data = all_extracted_data[0]
            else:
                ext_res.data = {
                    "pages": all_extracted_data,
                    "total_pages": len(all_extracted_data),
                }

        except Exception as e:
            _log.error(f"Error during extraction: {e}")
            ext_res.data = {"error": str(e)}

        return ext_res

    def _determine_status(self, ext_res: ExtractionResult) -> ConversionStatus:
        """Determine the status based on extraction results."""
        if ext_res.data and "error" not in ext_res.data:
            return ConversionStatus.SUCCESS
        else:
            return ConversionStatus.FAILURE

    def _get_images_from_input(self, input_doc: InputDocument) -> list[Image]:
        """Extract images from input document using the backend."""
        images = []

        try:
            backend = input_doc._backend

            assert isinstance(backend, PaginatedDocumentBackend)
            # Use the backend's pagination interface
            page_count = backend.page_count()
            _log.info(f"Processing {page_count} pages for extraction")

            for page_num in range(page_count):
                try:
                    page_backend = backend.load_page(page_num)
                    if page_backend.is_valid():
                        # Get page image at a reasonable scale
                        page_image = page_backend.get_page_image(
                            scale=self.pipeline_options.vlm_options.scale
                        )
                        images.append(page_image)
                    else:
                        _log.warning(f"Page {page_num} backend is not valid")
                except Exception as e:
                    _log.error(f"Error loading page {page_num}: {e}")

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

            class ExtractionTemplateFactory(ModelFactory[template]):
                __use_examples__ = True  # prefer Field(examples=...) when present
                __use_defaults__ = True  # use field defaults instead of random values

            return ExtractionTemplateFactory.build().model_dump_json(indent=2)
        else:
            raise ValueError(f"Unsupported template type: {type(template)}")

    @classmethod
    def get_default_options(cls) -> VlmExtractionPipelineOptions:
        return VlmExtractionPipelineOptions()
