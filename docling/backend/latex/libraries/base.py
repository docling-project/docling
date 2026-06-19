from abc import ABC, abstractmethod


class LibraryHandler(ABC):
    @property
    @abstractmethod
    def environments(self) -> frozenset[str]:
        pass

    @property
    @abstractmethod
    def macros(self) -> frozenset[str]:
        pass

    @abstractmethod
    def handle_environment(self, *args, **kwargs):
        pass
