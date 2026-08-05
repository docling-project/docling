import logging
from collections.abc import Iterable
from typing import List, Optional, Tuple

from docling_core.types.doc import DoclingDocument, NodeItem, TableItem
from PIL import Image
from pydantic import BaseModel, ConfigDict

from docling.datamodel.document import ConversionResult
from docling.models.base_model import GenericEnrichmentModel

_log = logging.getLogger(__name__)


class TableFormulaEnrichmentElement(BaseModel):
    """A table together with the cell images that may contain formulas."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: TableItem
    cell_crops: List[Tuple[int, Image.Image]]


class TableCellFormulaVlmModel(GenericEnrichmentModel[TableFormulaEnrichmentElement]):
    """Runs formula recognition over PDF table cells.

    ``TableItem`` is yielded by ``DoclingDocument.iterate_items()``, so this
    stage receives the table and descends into ``data.table_cells`` itself.
    No change to the enrichment loop or to existing models is required.

    Cell crops cannot reuse ``BaseItemAndImageEnrichmentModel``: a ``TableCell``
    carries a ``bbox`` but no ``prov``, and is not a ``DocItem``.
    """

    # Aligned with CodeFormulaVlmModel so crops match the model's training data.
    images_scale = 1.67  # 120 dpi
    expansion_factor = 0.18
    elements_batch_size = 2

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, TableItem)

    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> Optional[TableFormulaEnrichmentElement]:
        if not self.is_processable(doc=conv_res.document, element=element):
            return None
        assert isinstance(element, TableItem)

        if not element.prov:
            return None  # no page geometry: nothing to crop

        page_no = element.prov[0].page_no
        page_ix = page_no - conv_res.pages[0].page_no
        page = conv_res.pages[page_ix]

        crops: List[Tuple[int, Image.Image]] = []
        for idx, cell in enumerate(element.data.table_cells):
            if cell.bbox is None:
                continue
            bbox = cell.bbox.expand_by_scale(
                self.expansion_factor, self.expansion_factor
            )
            image = page.get_image(scale=self.images_scale, cropbox=bbox)
            if image is not None:
                crops.append((idx, image))

        if not crops:
            return None
        return TableFormulaEnrichmentElement(item=element, cell_crops=crops)

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[TableFormulaEnrichmentElement],
    ) -> Iterable[NodeItem]:
        # Plumbing only: formula detection and rich-cell construction are not
        # implemented yet, pending design confirmation on the linked issue.
        for element in element_batch:
            yield element.item
