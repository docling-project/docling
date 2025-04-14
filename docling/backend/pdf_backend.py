from abc import ABC, abstractmethod
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Optional, Set, Union

from docling_core.types.doc import BoundingBox, Size
from docling_core.types.doc.page import SegmentedPdfPage, TextCell
from PIL import Image

from docling.backend.abstract_backend import PaginatedDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


class PdfPageBackend(ABC):
    @abstractmethod
    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        pass

    @abstractmethod
    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        pass

    @abstractmethod
    def get_text_cells(self) -> Iterable[TextCell]:
        pass

    @abstractmethod
    def get_bitmap_rects(self, float: int = 1) -> Iterable[BoundingBox]:
        pass

    @abstractmethod
    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        pass

    @abstractmethod
    def get_size(self) -> Size:
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        pass

    @abstractmethod
    def unload(self):
        pass


class PdfDocumentBackend(PaginatedDocumentBackend):
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        if self.input_format is not InputFormat.PDF:
            if self.input_format is InputFormat.IMAGE:
                buf = BytesIO()
                img = Image.open(self.path_or_stream)
                img.save(buf, "PDF")
                buf.seek(0)
                self.path_or_stream = buf
            else:
                raise RuntimeError(
                    f"Incompatible file format {self.input_format} was passed to a PdfDocumentBackend."
                )

    @abstractmethod
    def load_page(self, page_no: int) -> PdfPageBackend:
        pass

    @abstractmethod
    def page_count(self) -> int:
        pass

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        return {InputFormat.PDF}

    @classmethod
    def supports_pagination(cls) -> bool:
        return True
