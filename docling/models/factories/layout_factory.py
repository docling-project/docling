# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from docling.models.base_layout_model import BaseLayoutModel
from docling.models.factories.base_factory import BaseFactory


class LayoutFactory(BaseFactory[BaseLayoutModel]):
    def __init__(self, *args, **kwargs):
        super().__init__("layout_engines", *args, **kwargs)
