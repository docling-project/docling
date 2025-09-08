## Table Confidence Model Documentation

This document explains the ``TableConfidenceModel`` used for scoring tables detected in document pages.

### Overview

The ``TableConfidenceModel`` evaluates detected tables and assigns multiple confidence scores to quantify aspects such as structure, text quality, completeness, and layout. The scoring system helps downstream processes filter low-quality tables and weight them appropriately for further processing.

The model uses heuristics to detect issues such as overlapping cells, mixed content types, sparse rows/columns, and irregular layouts. Each score is then adjusted using configurable methods such as ``sigmoid``, ``sqrt``, or ``linear``.

### Model Configuration

**TableConfidenceOptions**

``TableConfidenceOptions`` is a placeholder for future configuration options, allowing customization of scoring behavior, thresholds, and weighting.

**TableConfidenceModel**

The ``TableConfidenceModel`` is the main class responsible for calculating table confidence scores.
 - ``enabled``: Whether the model actively scores tables.
 - ``adjustment_method``: Determines how raw scores are adjusted.

When the model is called on a batch of pages, it iterates through each table and calculates the following scores:
 - ``Structure Score``: Measures grid integrity and penalizes overlapping cells. Larger tables and multi-row/column tables can receive a bonus.
 - ``Cell Text Score``: Evaluates the consistency and quality of cell text, penalizing mixed data types within a column and overlapping cells, while rewarding high-quality or structured text.
- ``Completeness Score``: Determines how fully populated a table is by comparing filled cells to expected cells, penalizing sparse rows or columns and accounting for merged cells.
- ``Layout Score``: Assesses visual and structural integrity, including grid presence, column alignment, row height consistency, OTSL sequences, and bounding box validity, applying both bonuses and penalties.

Scores are clamped to a 0-1 range and adjusted according to the specified method.

### How Scores Are Calculated

The model uses the following heuristics:
 - ``Structure``: Starts with a base score from the detection model, subtracts penalties for overlapping cells, and adds bonuses for multi-row or multi-column structures.
 - ``Cell Text``: Begins with a base confidence, applies penalties for overlaps and mixed content types, evaluates text quality heuristically based on content and character composition, and blends this with the base score.
 - ``Completeness``: Computes the ratio of filled cells to total expected cells, considering merged cells and table shape, with penalties for sparsely populated rows or columns.
 - ``Layout``: Combines visual cues, such as grid presence, alignment, row height consistency, and OTSL sequences, while penalizing irregular bounding boxes, overlaps, low fill ratios, and inconsistent row heights.

These four scores together provide a comprehensive assessment of table quality.

### Table and Page Data Structures

**Table**
- Table extends BasePageElement and represents a detected table on a page.
- **Key attributes**:
   - ``num_rows`` and ``num_cols``: dimensions of the table grid.
   - ``table_cells``: list of TableCell objects representing individual cells.
   - ``detailed_scores``: holds a TableConfidenceScores object with all calculated confidence scores.
   - ``otsl_seq``: stores the Open Table Structure Language sequence for layout evaluation.

This structure is what the ``TableConfidenceModel`` consumes to calculate scores.

**TableConfidenceScores**
- Stores the four individual confidence scores:
    1. structure_score
    2. cell_text_score
    3. completeness_score
    4. layout_score
- Property ``total_table_score`` provides a weighted average using default weights:
    1. Structure: 0.3
    2. Cell Text: 0.3
    3. Completeness: 0.2
    4. Layout: 0.2

**PageConfidenceScores**
- Aggregates confidence for an entire page.
- Contains per-table scores (tables: Dict[int, TableConfidenceScores]).
- Provides computed properties:
   - ``table_score``: average across all tables on the page.
   - ``mean_score``: average across OCR, layout, parsing, and tables.
   - ``mean_grade`` and ``low_grade``: map numeric scores to qualitative grades (POOR, FAIR, GOOD, EXCELLENT).

**ConfidenceReport**
- Aggregates confidence across the whole document.
- Holds pages: Dict[int, PageConfidenceScores] and document-level scores (mean_score, table_score, etc.).

### Usage Example

The following example shows how to configure a document conversion pipeline to enable table structure extraction and run the ``TableConfidenceModel``:

```
# 1. Configure table structure to use the ACCURATE mode
table_options = TableStructureOptions(
    table_former_mode=TableFormerMode.ACCURATE
)

# 2. Define the pipeline options, ensuring table structure is enabled.
pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    force_full_page_ocr=True,   
    do_table_structure=True, # Crucial for the table confidence model to run
    table_structure_options=table_options,
)

# 3. Instantiate the DocumentConverter with the pipeline options.
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

# 4. Run the conversion.
conv_result: ConversionResult = doc_converter.convert(source=pdf_path)

# 5. Access table confidence scores.
for page in conv_result.pages:
    if page.predictions.confidence_scores and page.predictions.confidence_scores.tables:
        for table_id, score in page.predictions.confidence_scores.tables.items():
            print(f"Page {page.page_no}, Table {table_id} scores: {score}")
```
