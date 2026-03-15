import math
import logging
import numpy as np
import re
from typing import Iterable, List, Optional, Dict
from collections import defaultdict
from pydantic import BaseModel, Field, computed_field

from docling.datamodel.base_models import (
    QualityGrade,
    PageConfidenceScores as BasePageConfidenceScores,
    ConfidenceReport as BaseConfidenceReport,
)

from docling.datamodel.base_models import Table, ScoreValue, BoundingBox
from docling.datamodel.document import ConversionResult, Page
from docling.models.base_model import BasePageModel
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class TableConfidenceOptions(BaseModel):
    """Placeholder for future configuration options for the model."""
    # Order: [structure_score, cell_text_score, completeness_score, layout_score]
    weights: List[float] = Field(default_factory=lambda: [0.3, 0.3, 0.2, 0.2])


class TableConfidenceScores(BaseModel):
    """Holds the individual confidence scores for a single table."""
    structure_score: float = np.nan
    cell_text_score: float = np.nan
    completeness_score: float = np.nan
    layout_score: float = np.nan
    options: TableConfidenceOptions

    def total_table_score(self, weights: List[float] = None) -> float:
        """
        Calculates the weighted average of the individual confidence scores 
        to produce a single total score.
        """
        scores = [
            self.structure_score,
            self.cell_text_score,
            self.completeness_score,
            self.layout_score,
        ]

        weights = self.options.weights
        valid = [(s, w) for s, w in zip(scores, weights) if not math.isnan(s)]
        if not valid:
            return np.nan

        scores, weights = zip(*valid)
        weights = [w / sum(weights) for w in weights]
        return float(np.average(scores, weights=weights))

    def _score_to_grade(self, score: float) -> QualityGrade:
        if score < 0.5:
            return QualityGrade.POOR
        elif score < 0.8:
            return QualityGrade.FAIR
        elif score < 0.9:
            return QualityGrade.GOOD
        else:
            return QualityGrade.EXCELLENT

    @property
    def grade(self, weights: List[float] = None) -> QualityGrade:
        return self._score_to_grade(self.total_table_score(weights=weights))


class PageConfidenceScores(BasePageConfidenceScores):
    tables: Dict[int, TableConfidenceScores] = Field(default_factory=dict)

    @property
    def table_score(self) -> float:
        if not self.tables:
            return np.nan
        return float(np.nanmean([t.total_table_score for t in self.tables.values()]))
    
    @classmethod
    def from_base(cls, base: BasePageConfidenceScores) -> "PageConfidenceScores":
        """
        Convert a base PageConfidenceScores into the extended version.
        Preserves existing fields and initializes `tables` empty.
        """
        return cls(
            parse_score=base.parse_score,
            layout_score=base.layout_score,
            ocr_score=base.ocr_score,
            # old base had table_score as a scalar — drop into tables if needed
            tables={},
        )


class ConfidenceReport(BaseConfidenceReport):
    pages: Dict[int, PageConfidenceScores] = Field(
        default_factory=lambda: defaultdict(PageConfidenceScores)
    )

    # Document-level fields
    ocr_score: float = np.nan
    table_score: float = np.nan
    layout_score: float = np.nan
    parse_score: float = np.nan

    @classmethod
    def from_base(cls, base: BaseConfidenceReport) -> "ConfidenceReport":
        return cls(
            pages={pid: PageConfidenceScores.from_base(p) for pid, p in base.pages.items()},
            ocr_score=getattr(base, "ocr_score", np.nan),
            table_score=getattr(base, "table_score", np.nan),
            layout_score=getattr(base, "layout_score", np.nan),
            parse_score=getattr(base, "parse_score", np.nan),
        )


def _calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (BoundingBox): The first bounding box.
        box2 (BoundingBox): The second bounding box.

    Returns:
        float: The IoU score, a value between 0.0 and 1.0.
    """

    x1, y1 = box1.l, box1.t
    x2, y2 = box2.l, box2.t
    w1, h1 = box1.r - box1.l, box1.b - box1.t
    w2, h2 = box2.r - box2.l, box2.b - box2.t
    
    # Calculate the intersection coordinates
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    
    intersection = x_overlap * y_overlap
    
    # Calculate the union
    union = (w1 * h1) + (w2 * h2) - intersection
    if union == 0:
        return 0.0
    
    return intersection / union


def _adjust_scores(scores: List[float], method: str = "sigmoid") -> List[float]:
    """
    Adjusts a list of confidence scores using a mathematical transformation.

    Args:
        scores (List[float]): A list of scores to adjust.
        method (str): The adjustment method. Options are "sigmoid", "sqrt", or "linear".

    Returns:
        List[float]: The list of adjusted scores.
    """
    adjusted = []

    for s in scores:
        s = max(0.0, min(1.0, s))  # Clamp between 0 and 1
        if method == "sigmoid":
            adjusted.append(1 / (1 + math.exp(-12 * (s - 0.5))))
        elif method == "sqrt":
            adjusted.append(math.sqrt(s))
        else:  # linear
            adjusted.append(s)

    return adjusted


class TableConfidenceModel(BasePageModel):
    """
    Model to score the confidence of detected tables in a document page.

    Each table receives a score based on its structure, text quality, completeness, and layout.
    """
    def __init__(self, options: TableConfidenceOptions, enabled: bool = True, adjustment_method: str = "sigmoid"):
        """
        Initializes the TableConfidenceModel.

        Args:
            options (TableConfidenceOptions): Configuration options for the model.
            enabled (bool): Whether the model is enabled. If False, it will not run.
            adjustment_method (str): The method to use for adjusting final scores.
        """
        super().__init__()
        self.options = options
        self.enabled = enabled
        self.adjustment_method = adjustment_method
    
    def compute_page_scores(self, conv_res: ConversionResult, pages: Iterable[Page]) -> Dict[int, PageConfidenceScores]:
        """
        Computes table confidence scores for each page.

        Args:
            pages (Iterable[Page]): List of pages to process.

        Returns:
            Dict[int, PageConfidenceScores]: Mapping of page_no → confidence scores.
        """
        page_scores: Dict[int, PageConfidenceScores] = {}

        for page in pages:
            with TimeRecorder(conv_res, "table_confidence_score"):
                if not page.predictions or not getattr(page.predictions, "tablestructure", None):
                    continue

                table_map = getattr(page.predictions.tablestructure, "table_map", {})
                page_conf = PageConfidenceScores()

                for table_id, table in table_map.items():
                    table_score = self.calculate_scores(table)
                    page_conf.tables[table_id] = table_score

                if page_conf.tables:
                    page_scores[page.page_no] = page_conf

        return page_scores
        
    def __call__(self, conv_res: ConversionResult, pages: Iterable[Page] = None) -> Iterable[Page]:
        """
        Processes a batch of pages to calculate confidence scores for each table.

        Args:
            conv_res (ConversionResult)
            pages (Iterable[Page], optional): Defaults to all pages in conv_res.

        Returns:
            Dict[int, PageConfidenceScores]: page_no → scores
        """
        if pages is None:
            pages = conv_res.pages

        return self.compute_page_scores(conv_res, pages)


    def calculate_scores(self, detected_table: Table) -> TableConfidenceScores:
        """
        Orchestrates the calculation of all four individual confidence scores.

        This function acts as the central hub, calling dedicated helper methods for each
        scoring component (structure, text, completeness, and layout) and aggregating
        their results into a single comprehensive `TableConfidenceScores` object.

        Args:
            detected_table (Table): The table object to be scored. It must contain
                                    the necessary attributes like `table_cells`,
                                    `num_rows`, `num_cols`, and `cluster`.

        Returns:
            TableConfidenceScores: An object containing all calculated scores, each
                                   wrapped in a `ScoreValue` object.
        """
        scores = TableConfidenceScores(options=self.options)

        scores.structure_score = self._calculate_structure_score(detected_table)
        scores.cell_text_score = self._calculate_cell_text_score(detected_table)
        scores.completeness_score = self._calculate_completeness_score(detected_table)
        scores.layout_score = self._calculate_layout_score(detected_table)
        
        return scores
        
    def _calculate_structure_score(self, detected_table: Table) -> ScoreValue:
        """
        Evaluates the structural integrity of the table, focusing on how well its cells form a coherent grid.

        The score is a heuristic calculated as follows:
        1. **Base Score**: Starts with the table's cluster confidence (from the detection model). If not present, a default of 0.5 is used. For very large tables (>100 cells), a higher base score of 0.95 is assigned, assuming they are likely valid.
        2. **Overlap Penalty**: A penalty is applied for every pair of cells with an IoU (Intersection over Union) of more than 10%. The penalty is scaled by the IoU, so greater overlap results in a larger penalty.
        3. **Grid Bonus**: A bonus of 0.1 is added if the table has a multi-row or multi-column structure, as this suggests a more legitimate grid than a single-row/column list.

        The final score is clamped between 0.0 and 1.0 and then adjusted using the specified adjustment method (e.g., sigmoid).

        Returns:
            ScoreValue: The calculated structure score.
        """
        # Base score: Use cluster confidence if available, otherwise apply a heuristic based on table size.
        if detected_table.cluster and detected_table.cluster.confidence:
            score = detected_table.cluster.confidence
        elif detected_table.num_rows * detected_table.num_cols > 100:
            score = 0.95  # Assume high confidence for large tables
        else:
            score = 0.5  # Default score for smaller, unconfident tables

        # Penalty for excessive cell overlap. Overlaps suggest a poor grid or merged cells.
        overlap_penalty = 0.0
        cells = detected_table.table_cells
        if cells:
            num_cells = len(cells)
            for i in range(num_cells):
                for j in range(i + 1, num_cells):
                    iou = _calculate_iou(cells[i].bbox, cells[j].bbox)
                    if iou > 0.1:
                        overlap_penalty += iou * 0.3  # Penalize based on the degree of overlap

        score = max(0.0, score - overlap_penalty)

        # Bonus for a multi-row or multi-column structure, as these are more likely to be legitimate tables.
        if detected_table.num_rows > 1 or detected_table.num_cols > 1:
            score += 0.1

        # Apply final adjustments and return the score.
        adjusted = _adjust_scores([score], method=self.adjustment_method)[0]
        return ScoreValue(adjusted)

    def _calculate_cell_text_score(self, detected_table: Table) -> ScoreValue:
        """
        Evaluates the accuracy and consistency of the text recognized within each cell.

        The score is a heuristic blend of multiple factors:
        1. **Base Score**: The table's cluster confidence. If not present, a default of 0.5 is used.
        2. **Overlap Penalty**: A small penalty is applied for cell overlaps, as they can lead to corrupted text extraction.
        3. **Consistency Penalty**: A penalty of 0.05 is applied if a column contains both numeric and non-numeric text, which is a strong sign of a parsing or OCR error.
        4. **Text Quality Heuristic**: A helper function, `_calculate_text_quality`, scores each cell's text based on content (e.g., numeric codes vs. descriptive text). The average of these scores is blended with the base score (50/50 weighted average).
        5. **Grid Bonus**: A bonus of 0.05 is added for tables with a non-trivial grid.

        The final blended score is clamped and adjusted.

        Returns:
            ScoreValue: The calculated cell text score.
        """
        score = detected_table.cluster.confidence if detected_table.cluster else 0.5

        def _calculate_text_quality(text: str) -> float:
            """Calculates a heuristic score for text quality based on content."""
            if not text or not text.strip():
                return 0.0

            norm = re.sub(r"\s+", " ", text.lower())

            # Heuristic for numeric IDs or codes (assumed high quality)
            if re.match(r"^[0-9]+(-[0-9]+)*(\s*and\s*[0-9]+)*$", norm):
                return 1.0
            
            # Heuristics for descriptive text based on length and alphanumeric ratio
            alpha_ratio = sum(c.isalpha() for c in norm) / max(len(norm), 1)

            if len(norm) > 50:
                return max(0.8, 0.9 * alpha_ratio + 0.1)
            elif len(norm) > 20:
                return max(0.7, 0.9 * alpha_ratio + 0.1)
            else:
                return max(0.6, 0.7 * alpha_ratio + 0.3)

        # Base score from cluster confidence or a default value
        score = detected_table.cluster.confidence if detected_table.cluster else 0.5

        # Apply a penalty for cell overlaps, which can indicate text extraction issues
        overlap_penalty = sum(
            0.05 for i, c1 in enumerate(detected_table.table_cells)
            for c2 in detected_table.table_cells[i+1:]
            if _calculate_iou(c1.bbox, c2.bbox) > 0.2
        )
        score = max(0.0, score - overlap_penalty)

        # Penalize columns with mixed data types (e.g., numbers and text)
        for col in range(1, detected_table.num_cols + 1):
            col_texts = [
                c.text for c in detected_table.table_cells
                if c.start_col_offset_idx <= col <= c.end_col_offset_idx
            ]
            if col_texts:
                has_number = any(str(t).replace('.', '', 1).isdigit() for t in col_texts)
                has_text = any(not str(t).replace('.', '', 1).isdigit() for t in col_texts)
                if has_number and has_text:
                    score = max(0.0, score - 0.05)
        
        # Calculate the average text quality score for all cells
        text_scores = [_calculate_text_quality(c.text) for c in detected_table.table_cells]
        if text_scores:
            avg_text_score = np.mean(text_scores)
        else:
            avg_text_score = 0.0

         # Blend the base score with the heuristic text quality score
        blended_score = 0.5 * score + 0.5 * avg_text_score

        # Apply a bonus for non-trivial grids
        if detected_table.num_rows > 1 and detected_table.num_cols > 1:
            blended_score += 0.05

        #  Finalize and return the adjusted score
        final_score = min(1.0, blended_score)
        adjusted = _adjust_scores([final_score], method=self.adjustment_method)[0]
        return ScoreValue(adjusted)

    def _calculate_completeness_score(self, detected_table: Table) -> ScoreValue:
        """
        Measures the completeness of a table based on its fill ratio and column/row density.

        The score is a heuristic calculated as follows:
        1. **Base Score**: The ratio of filled cells to the total number of expected cells in the grid.
           - For standard grids, this is `num_rows * num_cols`.
           - For tables with merged cells, this is the sum of the spans of all detected cells.
           - For single-row or single-column tables, this is the ratio of filled cells to the total number of detected cells.
           The filled count is tolerant of invisible OCR characters.
        2. **Sparse Penalty**: A penalty of 0.1 is applied for each column or row with a fill density below 10%. This helps penalize tables that have large, empty "ghost" rows or columns that were incorrectly detected.

        The final score is clamped to a minimum of 0.0.

        Returns:
            ScoreValue: The calculated completeness score.
        """
        # Handle empty table case
        if not detected_table.table_cells:
            return ScoreValue(0.0)

        # Count the number of cells that contain non-empty, OCR-tolerant text.
        filled_count = sum(
            (getattr(c, 'row_span', 1) or 1) * (getattr(c, 'col_span', 1) or 1)
            for c in detected_table.table_cells
            if c.text and c.text.strip().replace('\u200b', '')
        )


        # Determine the total expected number of cells, accounting for merged cells and table shape.
        has_merged_cells = any(getattr(c, 'row_span', 1) > 1 or getattr(c, 'col_span', 1) > 1 for c in detected_table.table_cells)

        if detected_table.num_rows <= 1 or detected_table.num_cols <= 1:
            # For single-row/column tables, completeness is the filled ratio of detected cells
            total_expected_cells = max(1, len(detected_table.table_cells))
        elif has_merged_cells:
            # For tables with merged cells, total expected cells is the sum of all cell spans
            total_expected_cells = sum(getattr(c, 'row_span', 1) * getattr(c, 'col_span', 1) for c in detected_table.table_cells)
        else:
            # For a standard grid, total expected cells is num_rows * num_cols
            total_expected_cells = detected_table.num_rows * detected_table.num_cols

        if total_expected_cells == 0:
            return ScoreValue(0.0)
        
        base_score = filled_count / total_expected_cells

        # Apply penalties for sparse rows and columns, which can indicate poor extraction.
        total_penalty = 0.0

        if detected_table.num_cols > 1:
            sparse_cols = sum(
                1 for col in range(1, detected_table.num_cols + 1)
                if sum(
                    1 for c in detected_table.table_cells
                    if c.start_col_offset_idx <= col <= c.end_col_offset_idx and c.text
                    and c.text.strip().replace('\u200b','')
                ) / max(1, detected_table.num_rows) < 0.1
            )
            total_penalty += (sparse_cols / detected_table.num_cols) * 0.1

        if detected_table.num_rows > 1:
            sparse_rows = sum(
                1 for row in range(detected_table.num_rows)  # 0-based
                if sum(
                    1 for c in detected_table.table_cells
                    if c.start_row_offset_idx <= row <= c.end_row_offset_idx
                    and c.text and c.text.strip().replace('\u200b','')
                ) / max(1, detected_table.num_cols) < 0.1
            )
            total_penalty += (sparse_rows / detected_table.num_rows) * 0.1

        final_score = max(0.0, base_score - total_penalty)
        return ScoreValue(final_score)

    def _calculate_layout_score(self, detected_table: Table) -> ScoreValue:
        """
        Measures the table's visual and structural integrity, including alignment and OTSL markers.

        This score awards points for positive indicators like OTSL markers, a clear grid, and consistent
        column alignment, while penalizing for irregular bounding boxes or other layout issues.

        This score is a heuristic calculated as follows:
        1. **Base Score**: A default of 0.5.
        2. **OTSL Bonus**: A bonus of 0.25 is added if the table has a recognized Open Table Structure Language (OTSL) sequence, which indicates a good structural model.
        3. **Grid Bonus**: A bonus of 0.25 is added if the table has a clear grid with both rows and columns.
        4. **Alignment Bonus**: A bonus of up to 0.3 is awarded based on the proportion of columns where cells have a consistent horizontal starting position (low standard deviation of `x` coordinates).
        5. **Row Height Consistency Bonus**: A bonus of up to 0.2 is given for tables with uniform row heights, calculated using the standard deviation of cell heights.
        6. **Penalties**:
           - **Overlap**: A penalty of 0.4 for tables with detected overlaps.
           - **Fill Ratio**: A penalty of 0.2 if the fill ratio is below 30%.
           - **Row Height Variation**: A penalty of 0.2 for tables with highly inconsistent row heights.
           - **Irregular BBoxes**: A significant penalty of 0.6 is applied if any cell has an invalid bounding box (e.g., zero or negative width/height).

        The final score is clamped to the range [0.0, 1.0] and adjusted.

        Returns:
            ScoreValue: The calculated layout score.
        """
        # Start with a base score and check for irregular bounding boxes
        has_irregular_bboxes = False
        for cell in detected_table.table_cells:
            if cell.bbox.width <= 0 or cell.bbox.height <= 0 or cell.bbox.l < 0 or cell.bbox.t < 0:
                has_irregular_bboxes = True
                break
        layout_score = 0.5

        # Bonus for the presence of a known table structure language (OTSL)
        if detected_table.otsl_seq and "T{" in detected_table.otsl_seq:
            layout_score += 0.25
        
        # Bonus for having a clear grid with both rows and columns
        if detected_table.num_rows > 0 and detected_table.num_cols > 0:
            layout_score += 0.25
        
        # Calculate bonus for consistent column alignment
        aligned_fraction = 0.0
        if detected_table.num_cols > 1:
            consistent_columns = 0
            for col in range(detected_table.num_cols):  # zero-based index
                col_x_coords = [
                    c.bbox.l  # use left edge for alignment
                    for c in detected_table.table_cells
                    if c.start_col_offset_idx <= col <= c.end_col_offset_idx
                ]
                if len(col_x_coords) > 1 and np.std(col_x_coords) < 5:
                    consistent_columns += 1
            aligned_fraction = consistent_columns / detected_table.num_cols
            layout_score += 0.3 * aligned_fraction
        
        # Calculate bonus for consistent row heights
        row_heights = [c.bbox.height for c in detected_table.table_cells]
        if row_heights:
            max_h = max(row_heights)
            if max_h > 0:
                norm_std = min(np.std(row_heights) / max_h, 1.0)
                layout_score += 0.2 * (1 - norm_std)
        
        # Apply penalties for known layout issues
        if getattr(detected_table, "has_overlaps", False):
            layout_score -= 0.4
        if getattr(detected_table, "fill_ratio", 1.0) < 0.3:
            layout_score -= 0.2
        if row_heights and max_h > 0 and (np.std(row_heights) / max_h) > 0.5:
            layout_score -= 0.2
        if has_irregular_bboxes:
            layout_score -= 0.6  # Heavy penalty for invalid bounding boxes
        
        # Clamp the score to the valid range and apply the final adjustment
        layout_score = max(0.0, min(1.0, layout_score))
        if self.adjustment_method:
            layout_score = _adjust_scores([layout_score], method=self.adjustment_method)[0]
        return ScoreValue(layout_score)


def main():
    from pathlib import Path
    from docling.datamodel.pipeline_options import TableStructureOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    data_folder = Path(__file__).parent / "../../tests/data"
    input_pdf_path = data_folder / "pdf/2206.01062.pdf"

    # Configure table structure to use the ACCURATE mode
    table_options = TableStructureOptions(
        table_former_mode=TableFormerMode.ACCURATE
    )

    # 1. Define the pipeline options, ensuring table structure is enabled.
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        force_full_page_ocr=True,
        do_table_structure=True,
        table_structure_options=table_options,
        # ocr_options = EasyOcrOptions(force_full_page_ocr=True)
    )
    
    # 2. Instantiate the DocumentConverter with the pipeline options.
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    # 3. Run the conversion.
    conv_result: ConversionResult = doc_converter.convert(source=input_pdf_path)

    table_options = TableConfidenceOptions(
        # If not specified, uses default weights
        # Order: [structure_score, cell_text_score, completeness_score, layout_score]
        weights=[0.25, 0.25, 0.25, 0.25]
    )

    # 4. Compute Table Conversion Confidence Scores as post-processing step.
    table_conf_model = TableConfidenceModel(
        options=table_options,
        enabled=True,
        adjustment_method="sigmoid"
    )

    # 5. Get table confidence mapping
    table_scores = table_conf_model(conv_result)

    for page_no, page_conf in table_scores.items():
        print(f"Page {page_no}:")
        for table_id, t_score in page_conf.tables.items():
            print(f"Table {table_id}: scores=<{t_score}>, grade={t_score.grade}")


if __name__ == "__main__":
    main()
