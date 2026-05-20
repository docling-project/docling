from __future__ import annotations

from typing import ClassVar, Literal, Protocol, runtime_checkable

BackendName = Literal["pdf_sdk", "rhwp", "libreoffice"]
BACKEND_NAMES: tuple[BackendName, ...] = ("pdf_sdk", "rhwp", "libreoffice")


@runtime_checkable
class HwpToPdfConverter(Protocol):
    name: ClassVar[BackendName]

    def is_available(self) -> bool: ...

    def convert(self, file_path: str) -> str | None: ...
