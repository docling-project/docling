from abc import ABC, abstractmethod
from docling_core.types import DoclingDocument


class BaseEnricher(ABC):
    @abstractmethod
    async def enrich(self, document: DoclingDocument, **kwargs) -> DoclingDocument:
        ...
