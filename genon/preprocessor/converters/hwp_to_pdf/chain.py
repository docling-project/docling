from __future__ import annotations

import logging
from typing import Sequence

from .base import HwpToPdfConverter

_log = logging.getLogger(__name__)


class ConverterChain:
    def __init__(self, converters: Sequence[HwpToPdfConverter]):
        self._converters = list(converters)

    @property
    def backends(self) -> list[str]:
        return [c.name for c in self._converters]

    def try_each(self, file_path: str) -> str | None:
        if not self._converters:
            _log.error("[hwp_to_pdf] no backends configured; cannot convert %s", file_path)
            return None

        _log.info("[hwp_to_pdf] chain start file=%s order=%s", file_path, self.backends)
        for c in self._converters:
            _log.info("[hwp_to_pdf] try backend=%s file=%s", c.name, file_path)
            try:
                result = c.convert(file_path)
            except Exception as e:
                _log.warning("[hwp_to_pdf] backend=%s raised: %s", c.name, e, exc_info=True)
                result = None
            if result:
                _log.info("[hwp_to_pdf] success backend=%s -> %s", c.name, result)
                return result
            _log.warning("[hwp_to_pdf] backend=%s returned no output", c.name)

        _log.error("[hwp_to_pdf] all backends failed file=%s tried=%s", file_path, self.backends)
        return None
