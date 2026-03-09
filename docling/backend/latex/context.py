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
    parent: Optional[NodeItem] = None
    formatting: Optional[Formatting] = None
    text_label: Optional[DocItemLabel] = None
