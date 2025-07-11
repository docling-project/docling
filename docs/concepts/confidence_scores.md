## Introduction

**Confidence scores** were introduced in [v2.34.0](https://github.com/docling-project/docling/releases/tag/v2.34.0) to help users understand how well a conversion performed and guide decisions about post-processing workflows. They are available in the [`confidence`](../../reference/document_converter/#docling.document_converter.ConversionResult.confidence) field of the [`ConversionResult`](../../reference/document_converter/#docling.document_converter.ConversionResult) object returned by the document converter.

## Purpose

Complex layouts, poor scan quality, or challenging formatting can lead to suboptimal document conversion results that may require additional attention or alternative conversion pipelines.

Confidence scores provide a quantitative assessment of document conversion quality. Each confidence report includes a **numerical score** (0.0 to 1.0) measuring conversion accuracy, and a **quality grade** (poor, fair, good, excellent) for quick interpretation.

Use cases for confidence scores include:

- Identify documents requiring manual review after the conversion
- Adjust conversion pipelines to the most appropriate for each document type
- Set confidence thresholds for unattended batch conversions
- Catch potential conversion issues early in your workflow

## Concepts

### Scores and grades

A confidence report contains *scores* and *grades*:

- **Scores**: Numerical values between 0.0 and 1.0, where higher values indicate better conversion quality
- **Grades**: Categorical quality assessments based on score thresholds:
  - `POOR`: Score < 0.5
  - `FAIR`: Score < 0.8  
  - `GOOD`: Score < 0.9
  - `EXCELLENT`: Score ≥ 0.9

### Types of scores

Each confidence report includes four component scores:

- **`layout_score`**: Overall quality of text content extraction
- **`ocr_score`**: Quality of OCR-extracted content
- **`parse_score`**: 10th percentile score of text cells (emphasizes problem areas)
- **`table_score`**: Table extraction quality *(not yet implemented)*

### Summary scores

Two aggregate scores provide overall document quality assessment:

- **`mean_score`**: Average of the four component scores
- **`low_score`**: 5th percentile score (highlights worst-performing areas)

Both summary scores include corresponding `mean_grade` and `low_grade` fields for quick quality assessment.

### Page-level vs document-level

Confidence scores are calculated at two levels:

- **Page-level**: Individual scores for each page, stored in the `pages` field
- **Document-level**: Overall scores for the entire document, calculated as averages of page-level scores and stored in fields equally named in the root [`ConfidenceReport`](h../../reference/document_converter/#docling.document_converter.ConversionResult.confidence)

!!! note "Document-level scores"

    For most use cases, users should safely focus on the document-level `mean_score` / `mean_grade` and `low_score` / `low_grade` fields to assess overall conversion quality.

### Example

![confidence_scores](../assets/confidence_scores.png)

