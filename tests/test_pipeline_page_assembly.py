# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from docling_core.types.doc import (
    BoundingBox,
    DoclingDocument,
    FormItem,
    KeyValueItem,
    ProvenanceItem,
    Size,
)
from docling_core.types.doc.document import GraphCell, GraphCellLabel, GraphData

from docling.pipeline.base_pipeline import BasePipeline


@pytest.mark.parametrize("item_type", [KeyValueItem, FormItem])
def test_page_assembly_preserves_graph_cell_source_pages(item_type):
    page_documents = []
    for page_no in (5, 9):
        document = DoclingDocument(name="page")
        document.add_page(page_no=1, size=Size(width=100, height=100))
        provenance = ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=10, t=10, r=90, b=90),
            charspan=(0, 0),
        )
        graph = GraphData(
            cells=[
                GraphCell(
                    label=GraphCellLabel.KEY,
                    cell_id=0,
                    text="Page",
                    orig="Page",
                    prov=provenance.model_copy(deep=True),
                ),
                GraphCell(
                    label=GraphCellLabel.VALUE,
                    cell_id=1,
                    text=str(page_no),
                    orig=str(page_no),
                ),
            ],
        )
        if item_type is KeyValueItem:
            document.add_key_values(graph=graph, prov=provenance)
        else:
            document.add_form(graph=graph, prov=provenance)
        page_documents.append((page_no, document))

    assembled = BasePipeline._concatenate_page_documents(page_documents)

    assert sorted(assembled.pages) == [5, 9]
    items = [item for item, _ in assembled.iterate_items()]
    assert len(items) == 2
    for item, page_no in zip(items, (5, 9)):
        assert isinstance(item, item_type)
        assert item.prov[0].page_no == page_no
        assert item.graph.cells[0].prov is not None
        assert item.graph.cells[0].prov.page_no == page_no
        assert item.graph.cells[1].prov is None
