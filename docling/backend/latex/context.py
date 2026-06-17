from dataclasses import dataclass
from typing import Optional

from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    Formatting,
    NodeItem,
)


@dataclass
class ParseContext:
    doc: DoclingDocument
    parent: NodeItem | None = None
    formatting: Formatting | None = None
    text_label: DocItemLabel | None = None
