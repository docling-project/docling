# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod


class RenderEngine(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def render(self, *args, **kwargs):
        pass
