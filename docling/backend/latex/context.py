from dataclasses import dataclass
from typing import Optional

from docling_core.types.doc.common.formatting import Formatting
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.node import NodeItem
from docling_core.types.doc.labels import DocItemLabel


@dataclass
class ParseContext:
    doc: DoclingDocument
    parent: NodeItem | None = None
    formatting: Formatting | None = None
    text_label: DocItemLabel | None = None
