from __future__ import annotations

from typing import Protocol

from bs4 import BeautifulSoup, Tag
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.page import BoundingRectangle, TextCell
from docling_core.types.doc.utils import parse_otsl_table_content

from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import (
    Cluster,
    Page,
    Table,
    TableStructurePrediction,
    VlmPrediction,
)


class VlmTaskInterpreter(Protocol):
    def interpret(
        self, page: Page, cluster: Cluster, prediction: VlmPrediction
    ) -> None: ...


class PlainTextInterpreter(VlmTaskInterpreter):
    def interpret(
        self, page: Page, cluster: Cluster, prediction: VlmPrediction
    ) -> None:
        text = prediction.text.strip()
        if not text:
            return
        # Attach as a single TextCell to the corresponding cluster
        cluster.cells = [
            TextCell(
                index=0,  # TODO: add index, could break stuff.
                text=text,
                orig=text,
                from_ocr=True,
                rect=BoundingRectangle.from_bounding_box(cluster.bbox),
            )
        ]


class HtmlTableInterpreter(VlmTaskInterpreter):
    def interpret(
        self, page: Page, cluster: Cluster, prediction: VlmPrediction
    ) -> None:
        # Only process table-like clusters; otherwise, no-op
        if cluster.label != DocItemLabel.TABLE:
            return

        html = prediction.text.strip()
        if not html:
            return

        # Ensure the HTML is wrapped in <table> tags if missing
        if "<table" not in html.lower():
            html = f"<table>{html}</table>"

        soup = BeautifulSoup(html, "html.parser")
        table_tag: Tag | None = soup.find("table")  # type: ignore[assignment]

        if table_tag is None:
            return

        data = HTMLDocumentBackend.parse_table_data(table_tag)

        if data is None:
            return

        # Create or update the TableStructurePrediction for this page
        if page.predictions.tablestructure is None:
            page.predictions.tablestructure = TableStructurePrediction()

        # Find the target cluster reference on this page to attach to Table object
        target_cluster = cluster

        tbl = Table(
            otsl_seq=[],
            table_cells=data.table_cells,
            num_rows=data.num_rows,
            num_cols=data.num_cols,
            id=cluster.id,
            page_no=page.page_no,
            cluster=target_cluster,
            label=target_cluster.label,
        )

        page.predictions.tablestructure.table_map[cluster.id] = tbl


class OtslTableInterpreter(VlmTaskInterpreter):
    """Interprets OTSL table predictions from VLM models."""

    def interpret(
        self, page: Page, cluster: Cluster, prediction: VlmPrediction
    ) -> None:
        # Only process table-like clusters; otherwise, no-op
        if cluster.label != DocItemLabel.TABLE:
            return

        otsl_content = prediction.text.strip()
        if not otsl_content:
            return

        try:
            data = parse_otsl_table_content(otsl_content)
        except Exception:
            return

        # Create or update the TableStructurePrediction for this page
        if page.predictions.tablestructure is None:
            page.predictions.tablestructure = TableStructurePrediction()

        # Find the target cluster reference on this page to attach to Table object
        target_cluster = cluster

        tbl = Table(
            otsl_seq=[],
            table_cells=data.table_cells,
            num_rows=data.num_rows,
            num_cols=data.num_cols,
            id=cluster.id,
            page_no=page.page_no,
            cluster=target_cluster,
            label=target_cluster.label,
        )

        page.predictions.tablestructure.table_map[cluster.id] = tbl
