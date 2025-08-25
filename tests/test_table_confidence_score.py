import pytest
import numpy as np
from docling.models.table_confidence_model import TableConfidenceModel, TableConfidenceOptions


class BBox:
    def __init__(self, x, y, width, height):
        self.l = x
        self.t = y
        self.r = x + width
        self.b = y + height

    @property
    def x(self):
        return self.l

    @property
    def y(self):
        return self.t

    @property
    def width(self):
        return self.r - self.l

    @property
    def height(self):
        return self.b - self.t


class Cell:
    def __init__(self, text, bbox, row, column):
        self.text = text
        self.bbox = bbox
        self.row = row
        self.column = column


class Cluster:
    def __init__(self, confidence):
        self.confidence = confidence


class Table:
    def __init__(self, table_cells, num_rows, num_cols, cluster=None, otsl_seq=""):
        self.table_cells = table_cells
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cluster = cluster
        self.otsl_seq = otsl_seq


class TableStructure:
    def __init__(self, table_map):
        self.table_map = table_map


class Predictions:
    def __init__(self, tablestructure):
        self.tablestructure = tablestructure
        self.confidence_scores = type("ConfidenceScores", (), {"tables": {}})()


class Page:
    def __init__(self, predictions):
        self.predictions = predictions


class TableConfidenceOptions:
    pass


@pytest.fixture
def table_good():
    return Table(
        table_cells=[
            Cell(text="A", bbox=BBox(0,0,10,10), row=1, column=1),
            Cell(text="B", bbox=BBox(10,0,10,10), row=1, column=2)
        ],
        num_rows=1,
        num_cols=2,
        cluster=Cluster(confidence=0.8),
        otsl_seq="T{good}"
    )


@pytest.fixture
def table_empty():
    return Table(
        table_cells=[],
        num_rows=0,
        num_cols=0,
        cluster=None,
        otsl_seq=""
    )


@pytest.fixture
def table_partial():
    return Table(
        table_cells=[
            Cell(text="1", bbox=BBox(0,0,10,10), row=1, column=1),
            Cell(text="", bbox=BBox(10,0,10,10), row=1, column=2),
            Cell(text="3", bbox=BBox(0,10,10,10), row=2, column=1)
        ],
        num_rows=2,
        num_cols=2,
        cluster=Cluster(confidence=0.7),
        otsl_seq="T{partial}"
    )


@pytest.fixture
def table_single_col():
    return Table(
        table_cells=[
            Cell(text="X", bbox=BBox(0,0,10,10), row=1, column=1),
            Cell(text="Y", bbox=BBox(0,10,10,10), row=2, column=1),
            Cell(text="", bbox=BBox(0,20,10,10), row=3, column=1)
        ],
        num_rows=3,
        num_cols=1,
        cluster=Cluster(confidence=0.6),
        otsl_seq="T{single_col}"
    )


@pytest.fixture
def table_sparse():
    return Table(
        table_cells=[
            Cell(text="A", bbox=BBox(0,0,10,10), row=1, column=1),
            Cell(text="B", bbox=BBox(10,0,10,10), row=1, column=2),
            Cell(text="C", bbox=BBox(0,20,10,10), row=3, column=1),
            Cell(text="D", bbox=BBox(10,20,10,10), row=3, column=2)
        ],
        num_rows=3,
        num_cols=2,
        cluster=Cluster(confidence=0.9),
        otsl_seq="T{sparse}"
    )


@pytest.fixture
def table_merged():
    # Merged header table
    return Table(
        table_cells=[
            Cell(text="Header (spans 2 cols)", bbox=BBox(0,0,20,10), row=1, column=1),
            Cell(text="A", bbox=BBox(0,10,10,10), row=2, column=1),
            Cell(text="B", bbox=BBox(10,10,10,10), row=2, column=2),
            Cell(text="Footer (spans 2 cols)", bbox=BBox(0,20,20,10), row=3, column=1)
        ],
        num_rows=3,
        num_cols=2,
        cluster=Cluster(confidence=0.85),
        otsl_seq="T{merged}"
    )


@pytest.fixture
def table_header():
    return Table(
        table_cells=[
            Cell(text="Header (spans 2 cols)", bbox=BBox(0,0,20,10), row=1, column=1),
            Cell(text="A", bbox=BBox(0,10,10,10), row=2, column=1),
            Cell(text="B", bbox=BBox(10,10,10,10), row=2, column=2),
            Cell(text="Footer (spans 2 cols)", bbox=BBox(0,20,20,10), row=3, column=1)
        ],
        num_rows=3,
        num_cols=2,
        cluster=Cluster(confidence=0.85),
        otsl_seq="T{merged}"
    )


@pytest.fixture
def table_very_sparse():
    return Table(
        table_cells=[
            Cell(text="Only1", bbox=BBox(0,0,10,10), row=1, column=1)
        ],
        num_rows=3,
        num_cols=3,
        cluster=Cluster(confidence=0.4),
        otsl_seq="T{very_sparse}"
    )


@pytest.fixture
def table_overlap():
    # Overlapping cells
    return Table(
        table_cells=[
            Cell(text="A", bbox=BBox(0,0,10,10), row=1, column=1),
            Cell(text="B", bbox=BBox(5,0,10,10), row=1, column=2) # Overlaps with A
        ],
        num_rows=1,
        num_cols=2,
        cluster=None,
        otsl_seq="T{overlap}"
    )


@pytest.fixture
def table_overlapping_cells():
    return Table(
        table_cells=[
            Cell(text="A", bbox=BBox(0,0,10,10), row=1, column=1),
            Cell(text="B", bbox=BBox(5,0,10,10), row=1, column=2)  # Overlaps with A
        ],
        num_rows=1,
        num_cols=2,
        cluster=None,
        otsl_seq="T{overlap}"
    )


@pytest.fixture
def table_ragged():
    # Non-rectangular / ragged table
    return Table(
        table_cells=[
            Cell(text="1", bbox=BBox(0,0,10,10), row=1, column=1),
            Cell(text="2", bbox=BBox(10,0,10,10), row=1, column=2),
            Cell(text="3", bbox=BBox(0,10,10,10), row=2, column=1)  # row 2 has only 1 cell
        ],
        num_rows=2,
        num_cols=2,
        cluster=None,
        otsl_seq="T{ragged}"
    )


@pytest.fixture
def table_row_merged():
    # Multi-row merged cells
    return Table(
        table_cells=[
            Cell(text="Header (2 rows)", bbox=BBox(0,0,20,20), row=1, column=1),
            Cell(text="Body A", bbox=BBox(0,20,10,10), row=3, column=1),
            Cell(text="Body B", bbox=BBox(10,20,10,10), row=3, column=2)
        ],
        num_rows=3,
        num_cols=2,
        cluster=None,
        otsl_seq="T{row_merged}"
    )


@pytest.fixture
def table_variable_conf():
    # Cluster with varying confidence (simulate cell-level confidence)
    return Table(
        table_cells=[
            Cell(text="A", bbox=BBox(0,0,10,10), row=1, column=1),
            Cell(text="B", bbox=BBox(10,0,10,10), row=1, column=2)
        ],
        num_rows=1,
        num_cols=2,
        cluster=None,  # Could be ignored in model if using cell-level confidence
        otsl_seq="T{variable_conf}"
    )


@pytest.fixture
def table_header_only():
    # Only header
    return Table(
        table_cells=[
            Cell(text="Header1", bbox=BBox(0,0,10,10), row=1, column=1),
            Cell(text="Header2", bbox=BBox(10,0,10,10), row=1, column=2)
        ],
        num_rows=1,
        num_cols=2,
        cluster=None,
        otsl_seq="T{header_only}"
    )



@pytest.fixture
def table_footer_only():
    # Only footer
    return Table(
        table_cells=[
            Cell(text="Footer1", bbox=BBox(0,10,10,10), row=1, column=1),
            Cell(text="Footer2", bbox=BBox(10,10,10,10), row=1, column=2)
        ],
        num_rows=1,  # Corrected from 2 to 1
        num_cols=2,
        cluster=None,
        otsl_seq="T{footer_only}"
    )


@pytest.fixture
def table_symbols():
    # Non-text content (symbols)
    return Table(
        table_cells=[
            Cell(text="✓", bbox=BBox(0,0,10,10), row=1, column=1),
            Cell(text="✗", bbox=BBox(10,0,10,10), row=1, column=2)
        ],
        num_rows=1,
        num_cols=2,
        cluster=None,
        otsl_seq="T{symbols}"
    )


@pytest.fixture
def table_large():
    # Extreme sizes
    return Table(
        table_cells=[Cell(text=str(i+j*10), bbox=BBox(i*10,j*10,10,10), row=j+1, column=i+1) for j in range(50) for i in range(20)],
        num_rows=50,
        num_cols=20,
        cluster=None,
        otsl_seq="T{large}"
    )


@pytest.fixture
def table_small():
    return Table(
        table_cells=[Cell(text="1", bbox=BBox(0,0,10,10), row=1, column=1)],
        num_rows=1,
        num_cols=1,
        cluster=None,
        otsl_seq="T{small}"
    )


@pytest.fixture
def table_irregular_bbox():
    # Irregular bounding boxes
    return Table(
        table_cells=[
            Cell(text="Zero width", bbox=BBox(0,0,0,10), row=1, column=1),
            Cell(text="Negative coords", bbox=BBox(-5,0,10,10), row=1, column=2)
        ],
        num_rows=1,
        num_cols=2,
        cluster=None,
        otsl_seq="T{irregular_bbox}"
    )


@pytest.fixture
def table_overlapping_cells():
    return Table(
        table_cells=[
            Cell(text="A", bbox=BBox(0,0,10,10), row=1, column=1),
            Cell(text="B", bbox=BBox(5,0,10,10), row=1, column=2)  # Overlaps with A
        ],
        num_rows=1,
        num_cols=2,
        cluster=None,
        otsl_seq="T{overlap}"
    )


def test_table_confidence_scores_with_edge_cases(
    table_good,
    table_empty,
    table_partial,
    table_single_col,
    table_sparse,
    table_merged,
    table_very_sparse,
    table_overlap,
    table_ragged,
    table_row_merged,
    table_variable_conf,
    table_header_only,
    table_footer_only,
    table_symbols,
    table_large,
    table_small,
    table_irregular_bbox,
    table_overlapping_cells,
):
    model = TableConfidenceModel(options=TableConfidenceOptions(), enabled=True)

    page = Page(
        predictions=Predictions(
            tablestructure=TableStructure(
                table_map={
                    "good": table_good,
                    "empty": table_empty,
                    "partial": table_partial,
                    "single_col": table_single_col,
                    "sparse": table_sparse,
                    "merged": table_merged,
                    "very_sparse": table_very_sparse,
                    "overlap": table_overlap,
                    "ragged": table_ragged,
                    "row_merged": table_row_merged,
                    "variable_conf": table_variable_conf,
                    "header_only": table_header_only,
                    "footer_only": table_footer_only,
                    "symbols": table_symbols,
                    "large": table_large,
                    "small": table_small,
                    "irregular_bbox": table_irregular_bbox,
                    "table_overlapping_cells": table_overlapping_cells
                }
            )
        )
    )

    pages = list(model(conv_res=None, page_batch=[page]))
    
    for table_id, table in pages[0].predictions.tablestructure.table_map.items():
        scores = pages[0].predictions.confidence_scores.tables[table_id]

       # All scores should be within the [0, 1] range after adjustment
        assert 0.0 <= scores.structure_score <= 1.0
        assert 0.0 <= scores.cell_text_score <= 1.0
        assert 0.0 <= scores.completeness_score <= 1.0
        assert 0.0 <= scores.layout_score <= 1.0

    scores = pages[0].predictions.confidence_scores.tables

    # Structure Score Asserts
    assert scores["good"].structure_score > scores["partial"].structure_score
    assert scores["good"].structure_score > scores["single_col"].structure_score
    assert scores["partial"].structure_score > scores["empty"].structure_score
    assert scores["single_col"].structure_score > scores["empty"].structure_score
    assert scores["single_col"].structure_score <= scores["good"].structure_score
    assert scores["good"].structure_score > scores["very_sparse"].structure_score
    assert scores["good"].structure_score > scores["overlap"].structure_score

    # Cell Text Score Asserts
    # A table with text should score higher than one with symbols or empty cells.
    assert scores["good"].cell_text_score > scores["symbols"].cell_text_score
    assert scores["good"].cell_text_score > scores["empty"].cell_text_score

    # Completeness Score Asserts
    # A full table should have higher completeness than a partial or very sparse one.
    assert scores["good"].completeness_score > scores["partial"].completeness_score
    assert scores["empty"].completeness_score == 0.0
    assert scores["very_sparse"].completeness_score == 0.0
    assert scores["sparse"].completeness_score > 0.0
    assert scores["very_sparse"].completeness_score <= scores["sparse"].completeness_score

    # Layout Score Assert
    assert scores["good"].layout_score > scores["irregular_bbox"].layout_score
