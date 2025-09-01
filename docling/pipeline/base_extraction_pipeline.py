import logging
from abc import ABC, abstractmethod
from typing import Optional

from docling.datamodel.base_models import ConversionStatus, ErrorItem
from docling.datamodel.document import InputDocument
from docling.datamodel.extraction import ExtractionResult, ExtractionTemplateType
from docling.datamodel.pipeline_options import BaseOptions

_log = logging.getLogger(__name__)


class BaseExtractionPipeline(ABC):
    def __init__(self, pipeline_options: BaseOptions):
        self.pipeline_options = pipeline_options

    def execute(
        self,
        in_doc: InputDocument,
        raises_on_error: bool,
        template: Optional[ExtractionTemplateType] = None,
    ) -> ExtractionResult:
        ext_res = ExtractionResult(input=in_doc)

        try:
            ext_res = self._extract_data(ext_res, template)
            ext_res.status = self._determine_status(ext_res)
        except Exception as e:
            ext_res.status = ConversionStatus.FAILURE
            error_item = ErrorItem(
                component_type="extraction_pipeline",
                module_name=self.__class__.__name__,
                error_message=str(e),
            )
            ext_res.errors.append(error_item)
            if raises_on_error:
                raise e

        return ext_res

    @abstractmethod
    def _extract_data(
        self,
        ext_res: ExtractionResult,
        template: Optional[ExtractionTemplateType] = None,
    ) -> ExtractionResult:
        """Subclass must populate ext_res.pages/errors and return the result."""
        raise NotImplementedError

    @abstractmethod
    def _determine_status(self, ext_res: ExtractionResult) -> ConversionStatus:
        """Subclass must decide SUCCESS/PARTIAL_SUCCESS/FAILURE based on ext_res."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_default_options(cls) -> BaseOptions:
        pass
