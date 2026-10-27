# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Type

from docling_core.types.doc import DocItemLabel

from docling.datamodel.base_models import Cluster, Page, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import BaseTableStructureOptions
from docling.models.base_model import BaseModelWithOptions, BasePageModel


def degenerate_structure_reason(otsl_seq: list[str], max_steps: int) -> str | None:
    """Explain why a predicted OTSL sequence cannot describe a table, or None.

    The structure decoder is autoregressive with a fixed step budget. On some
    crops it never emits a row break (``nl``) and repeats one cell token until
    the budget runs out. The cell matcher then turns that into a one-row grid
    and silently drops nearly every text cell the layout stage assigned to the
    table (issue #3002). A complete table always ends with a row break, so a
    sequence that stops within a couple of tokens of the budget without one
    was cut off.
    """
    if not otsl_seq:
        return None
    if "nl" not in otsl_seq:
        return "the predicted structure has no row break"
    if len(otsl_seq) >= max_steps - 2 and otsl_seq[-1] != "nl":
        return f"the predicted structure hit the {max_steps}-step decode limit"
    return None


def keep_table_text_as_child(table_cluster: Cluster, page: Page) -> None:
    """Make sure the text under a table cluster survives a discarded structure.

    The reading-order stage emits a table's nested clusters only when the
    table has no grid, so a table whose structure was discarded keeps its text
    through them. A table cluster without nested clusters gets one text child
    holding the page cells inside its box.
    """
    if table_cluster.children:
        return
    cells = [
        cell
        for cell in page.cells
        if cell.text.strip()
        and cell.rect.to_bounding_box().intersection_over_self(table_cluster.bbox) > 0.5
    ]
    if not cells:
        return
    layout = page.predictions.layout
    used_ids = [table_cluster.id]
    if layout is not None:
        used_ids.extend(c.id for c in layout.clusters)
        used_ids.extend(ch.id for c in layout.clusters for ch in c.children)
    table_cluster.children = [
        Cluster(
            id=max(used_ids) + 1,
            label=DocItemLabel.TEXT,
            bbox=table_cluster.bbox,
            confidence=table_cluster.confidence,
            cells=cells,
        )
    ]


class BaseTableStructureModel(BasePageModel, BaseModelWithOptions, ABC):
    """Shared interface for table structure models."""

    enabled: bool

    @classmethod
    @abstractmethod
    def get_options_type(cls) -> Type[BaseTableStructureOptions]:
        """Return the options type supported by this table model."""

    @abstractmethod
    def predict_tables(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[TableStructurePrediction]:
        """Produce table structure predictions for the provided pages."""

    def __call__(
        self,
        conv_res: ConversionResult,
        page_batch: Iterable[Page],
    ) -> Iterable[Page]:
        if not getattr(self, "enabled", True):
            yield from page_batch
            return

        pages = list(page_batch)
        predictions = self.predict_tables(conv_res, pages)

        for page, prediction in zip(pages, predictions):
            page.predictions.tablestructure = prediction
            yield page
