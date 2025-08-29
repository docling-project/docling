import abc
from collections.abc import Iterable
from typing import Any

from docling.datamodel.document import ConversionResult


class BasePostprocessor(abc.ABC):
    """Base interface for post-processing Docling conversion results.

    Implementations may index to vector stores, export to custom sinks, etc.
    """

    @abc.abstractmethod
    def process(self, conv_results: Iterable[ConversionResult], **kwargs: Any) -> Any:
        """Consume an iterator of ConversionResult and perform side-effects.

        Returns an optional result object (implementation-defined).
        """
        raise NotImplementedError


