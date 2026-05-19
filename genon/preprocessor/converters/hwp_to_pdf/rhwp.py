from __future__ import annotations

import logging
from typing import ClassVar

from .availability import rhwp_available
from .base import BackendName

_log = logging.getLogger(__name__)


class RhwpConverter:
    name: ClassVar[BackendName] = "rhwp"

    def is_available(self) -> bool:
        return rhwp_available()

    def convert(self, file_path: str) -> str | None:
        _log.warning(
            "[hwp_to_pdf:rhwp] backend stub invoked but rhwp CLI integration is not implemented yet "
            "(see issue #199, PR2). file=%s",
            file_path,
        )
        return None
