from abc import ABC, abstractmethod
from typing import FrozenSet


class LibraryHandler(ABC):
    @property
    @abstractmethod
    def environments(self) -> FrozenSet[str]:
        pass

    @property
    @abstractmethod
    def macros(self) -> FrozenSet[str]:
        pass

    @abstractmethod
    def handle_environment(self, *args, **kwargs):
        pass
