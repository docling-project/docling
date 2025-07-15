import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterable, Dict, Optional, Set, Tuple

from docling.backend.pdf_backend import PdfPageBackend
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import PipelineOptions

_log = logging.getLogger(__name__)


@dataclass
class DocumentTracker:
    """Tracks document processing state for resource management"""

    doc_id: str
    total_pages: int
    processed_pages: int = 0
    page_backends: Dict[int, PdfPageBackend] = field(
        default_factory=dict
    )  # page_no -> backend
    conv_result: Optional[ConversionResult] = None


class AsyncPipeline(ABC):
    """Base class for async pipeline implementations"""

    def __init__(self, pipeline_options: PipelineOptions):
        self.pipeline_options = pipeline_options
        self.keep_images = False
        self.keep_backend = False

    @abstractmethod
    async def execute_stream(
        self, input_docs: AsyncIterable[InputDocument]
    ) -> AsyncIterable[ConversionResult]:
        """Process multiple documents with cross-document batching"""

    async def execute_single(
        self, in_doc: InputDocument, raises_on_error: bool = True
    ) -> ConversionResult:
        """Process a single document - for backward compatibility"""

        async def single_doc_stream():
            yield in_doc

        async for result in self.execute_stream(single_doc_stream()):
            return result

        # Should never reach here
        raise RuntimeError("No result produced for document")
