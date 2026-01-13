"""Advanced PipelineOptions Configuration Examples.

This module demonstrates various ways to configure PipelineOptions for different
document processing scenarios, from basic usage to advanced production setups.

Examples are based on real-world clinical document processing and enterprise
data pipeline implementations.
"""

from pathlib import Path

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrOptions,
    PipelineOptions,
    TableFormerMode,
    TableStructureOptions,
    TesseractCliOcrOptions,
    TesseractOcrModel,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# ============================================================================
# Example 1: Basic Document Conversion
# ============================================================================


def example_basic_conversion():
    """Basic document conversion with default pipeline options.

    This is the simplest configuration for converting digital PDF documents
    that don't require OCR or special table handling.
    """
    input_doc = Path("./sample.pdf")

    # Use default pipeline options
    pipeline_options = PipelineOptions()
    pipeline_options.do_ocr = False  # Digital PDF, no OCR needed
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
            )
        }
    )

    result = converter.convert(input_doc)

    # Access the document
    result.document.export_to_markdown()
    print(f"Converted document with {len(result.document.pages)} pages")

    return result


# ============================================================================
# Example 2: High-Accuracy OCR for Scanned Documents
# ============================================================================


def example_ocr_heavy_pipeline():
    """Configuration optimized for scanned documents requiring OCR.

    Use this when processing:
    - Scanned PDFs or images
    - Historical documents
    - Low-quality document scans
    - Multi-language documents
    """
    pipeline_options = PipelineOptions()

    # Enable OCR with high accuracy settings
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    # Configure OCR for best quality
    pipeline_options.ocr_options = OcrOptions(
        use_gpu=True,  # Enable GPU acceleration if available
        lang=["eng", "fra"],  # Multiple language support
        bitmap_area_threshold=0.05,
        tesseract_cli_options=TesseractCliOcrOptions(
            force_full_page_ocr=True,  # Process entire page for better accuracy
            tesseract_cmd="tesseract",
            model=TesseractOcrModel.ACCURATE,  # Use most accurate model
        ),
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return converter


# ============================================================================
# Example 3: Table-Focused Document Processing
# ============================================================================


def example_table_extraction_pipeline():
    """Optimized configuration for documents with complex table structures.

    Suitable for:
    - Financial reports with tabular data
    - Clinical trial protocols with result tables
    - Research papers with data tables
    - Spreadsheet-heavy documents
    """
    pipeline_options = PipelineOptions()

    # Enable table structure extraction with high accuracy
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True,  # Match cell content to structure
        mode=TableFormerMode.ACCURATE,  # Use accurate model for complex tables
    )

    # Disable OCR if tables are already in digital format
    pipeline_options.do_ocr = False

    # Generate table images for quality verification
    pipeline_options.generate_table_images = True
    pipeline_options.artifacts_path = Path("./table_artifacts")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return converter


# ============================================================================
# Example 4: Performance-Optimized High-Volume Processing
# ============================================================================


def example_fast_processing_pipeline():
    """Balanced configuration for high-volume document processing.

    Optimized for:
    - Batch processing large document sets
    - Real-time document ingestion pipelines
    - Cost-sensitive cloud deployments
    - Scenarios where speed outweighs perfect accuracy
    """
    pipeline_options = PipelineOptions()

    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    # Use faster models for better throughput
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST

    # Configure OCR for speed
    pipeline_options.ocr_options = OcrOptions(
        use_gpu=True,
        lang=["eng"],
        tesseract_cli_options=TesseractCliOcrOptions(
            force_full_page_ocr=False,  # Cell-level OCR is faster
            model=TesseractOcrModel.FAST,
        ),
    )

    # Disable image generation to save storage and processing time
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.generate_table_images = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return converter


# ============================================================================
# Example 5: Clinical Document Processing (Production-Grade)
# ============================================================================


def example_clinical_document_pipeline():
    """Production configuration for clinical and regulatory documents.

    Features:
    - High accuracy for regulatory compliance
    - Comprehensive artifact generation for audit trails
    - Table extraction for clinical data
    - Multi-language OCR support

    Use case: Processing clinical trial protocols, informed consent forms,
    and regulatory submission documents.
    """
    pipeline_options = PipelineOptions()

    # Enable all processing features
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    # High accuracy table extraction
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True, mode=TableFormerMode.ACCURATE
    )

    # Comprehensive OCR configuration
    pipeline_options.ocr_options = OcrOptions(
        use_gpu=True,
        lang=["eng"],  # Add more languages as needed
        bitmap_area_threshold=0.05,
        tesseract_cli_options=TesseractCliOcrOptions(
            force_full_page_ocr=True, model=TesseractOcrModel.ACCURATE
        ),
    )

    # Generate all artifacts for compliance and debugging
    pipeline_options.artifacts_path = Path("./clinical_artifacts")
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return converter


# ============================================================================
# Example 6: Debugging and Quality Assurance Configuration
# ============================================================================


def example_debug_pipeline():
    """Configuration for debugging and quality assurance workflows.

    Generates comprehensive artifacts for:
    - Troubleshooting processing issues
    - Quality verification
    - Model performance evaluation
    - Training data generation
    """
    pipeline_options = PipelineOptions()

    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    # Generate all possible artifacts
    artifacts_dir = Path("./debug_artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    pipeline_options.artifacts_path = artifacts_dir
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True

    # Use accurate models for best quality assessment
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.ocr_options.tesseract_cli_options.model = (
        TesseractOcrModel.ACCURATE
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return converter


# ============================================================================
# Example 7: Custom Artifact Paths for Organized Output
# ============================================================================


def example_organized_artifacts():
    """Configure organized artifact storage for multi-document processing.

    Recommended practices for managing artifacts:
    - Separate directories by document type or batch
    - Include timestamps in artifact paths
    - Implement cleanup strategies for old artifacts
    """
    from datetime import datetime

    # Create timestamped artifact directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_base = Path(f"./artifacts_{timestamp}")

    pipeline_options = PipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    # Organize artifacts by type
    pipeline_options.artifacts_path = artifacts_base / "processed"
    pipeline_options.generate_page_images = True
    pipeline_options.generate_table_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"Artifacts will be saved to: {artifacts_base}")
    return converter


# ============================================================================
# Example 8: Multi-Language Document Processing
# ============================================================================


def example_multilingual_pipeline():
    """Configuration for processing multi-language documents.

    Supports documents containing:
    - Multiple languages in one document
    - Mixed scripts (Latin, Cyrillic, etc.)
    - International regulatory documents
    """
    pipeline_options = PipelineOptions()

    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    # Configure OCR for multiple languages
    pipeline_options.ocr_options = OcrOptions(
        use_gpu=True,
        lang=[
            "eng",
            "fra",
            "deu",
            "spa",
            "ita",
        ],  # English, French, German, Spanish, Italian
        tesseract_cli_options=TesseractCliOcrOptions(
            force_full_page_ocr=True, model=TesseractOcrModel.ACCURATE
        ),
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return converter


# ============================================================================
# Example 9: Batch Processing with Progress Tracking
# ============================================================================


def example_batch_processing():
    """Process multiple documents with consistent configuration."""
    input_files = [
        Path("./documents/report1.pdf"),
        Path("./documents/report2.pdf"),
        Path("./documents/report3.pdf"),
    ]

    pipeline_options = PipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    results = []
    for idx, doc_path in enumerate(input_files, 1):
        print(f"Processing {idx}/{len(input_files)}: {doc_path.name}")
        result = converter.convert(doc_path)
        results.append(result)

    return results


# ============================================================================
# Example 10: Conditional Processing Based on Document Type
# ============================================================================


def example_adaptive_pipeline(document_path: Path):
    """Adapt pipeline configuration based on document characteristics.

    This example shows how to dynamically configure the pipeline based
    on document analysis or metadata.
    """
    # Analyze document to determine if OCR is needed
    # This is a simplified example - actual implementation would inspect the PDF
    needs_ocr = document_path.suffix.lower() in [".jpg", ".png", ".tiff"]
    is_table_heavy = "financial" in document_path.stem.lower()

    pipeline_options = PipelineOptions()

    if needs_ocr:
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = OcrOptions(
            use_gpu=True,
            tesseract_cli_options=TesseractCliOcrOptions(
                model=TesseractOcrModel.ACCURATE
            ),
        )
    else:
        pipeline_options.do_ocr = False

    if is_table_heavy:
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE, do_cell_matching=True
        )
        pipeline_options.generate_table_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return converter


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    print("Docling Pipeline Options Configuration Examples")
    print("=" * 60)

    # Example 1: Basic conversion
    print("\n1. Basic Conversion")
    print("   Use case: Digital PDFs without OCR requirements")
    # result = example_basic_conversion()

    # Example 2: OCR-heavy pipeline
    print("\n2. OCR-Heavy Pipeline")
    print("   Use case: Scanned documents and images")
    # converter = example_ocr_heavy_pipeline()

    # Example 3: Table extraction
    print("\n3. Table Extraction Pipeline")
    print("   Use case: Financial reports and data-heavy documents")
    # converter = example_table_extraction_pipeline()

    # Example 4: Fast processing
    print("\n4. Performance-Optimized Pipeline")
    print("   Use case: High-volume batch processing")
    # converter = example_fast_processing_pipeline()

    # Example 5: Clinical documents
    print("\n5. Clinical Document Pipeline")
    print("   Use case: Regulatory and compliance-focused processing")
    # converter = example_clinical_document_pipeline()

    print("\nRefer to function docstrings for detailed configuration options.")
