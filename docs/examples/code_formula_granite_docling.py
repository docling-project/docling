"""Example: Comparing CodeFormula models for code and formula extraction.

This example demonstrates how to use both the default CodeFormulaV2 model
and the new Granite Docling model for extracting code blocks and mathematical
formulas from PDF documents, allowing you to compare their outputs.
"""

from pathlib import Path

from docling_core.types.doc import CodeItem, FormulaItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    CodeFormulaVlmOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def extract_with_preset(preset_name: str, input_doc: Path):
    """Extract code and formulas using a specific preset.

    Args:
        preset_name: Name of the preset to use ('default' or 'granite_docling')
        input_doc: Path to the input PDF document

    Returns:
        The converted document
    """
    print(f"\n{'=' * 60}")
    print(f"Processing with preset: {preset_name}")
    print(f"{'=' * 60}\n")

    # Create options with the specified preset
    code_formula_options = CodeFormulaVlmOptions.from_preset(preset_name)

    # Display preset information
    print(f"Model: {code_formula_options.model_spec.name}")
    print(f"Repo ID: {code_formula_options.model_spec.default_repo_id}")
    print(f"Scale: {code_formula_options.scale}")
    print(f"Max tokens: {code_formula_options.model_spec.max_new_tokens}")
    print()

    # Configure the PDF pipeline to use code/formula enrichment
    pipeline_options = PdfPipelineOptions(
        do_code_enrichment=True,
        do_formula_enrichment=True,
        code_formula_options=code_formula_options,
    )

    # Create converter with the configured options
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert the document
    result = converter.convert(input_doc)
    doc = result.document

    # Print extracted code blocks
    code_blocks = [
        item for item, _ in doc.iterate_items() if isinstance(item, CodeItem)
    ]
    print(f"Code blocks found: {len(code_blocks)}")
    for i, item in enumerate(code_blocks, 1):
        print(f"\n  Code block {i}:")
        print(f"    Language: {item.code_language}")
        print(f"    Text: {item.text[:100]}{'...' if len(item.text) > 100 else ''}")

    # Print extracted formulas
    formulas = [
        item for item, _ in doc.iterate_items() if isinstance(item, FormulaItem)
    ]
    print(f"\nFormulas found: {len(formulas)}")
    for i, item in enumerate(formulas, 1):
        print(f"\n  Formula {i}:")
        print(f"    Text: {item.text[:100]}{'...' if len(item.text) > 100 else ''}")

    return doc


def main():
    """Main function to compare both presets."""
    input_doc = Path("tests/data/pdf/code_and_formula.pdf")

    if not input_doc.exists():
        print(f"Error: Input file not found: {input_doc}")
        print("Please provide a valid PDF file with code and formulas.")
        return

    print("Comparing CodeFormula presets for code and formula extraction")
    print(f"Input document: {input_doc}")

    # Extract with default CodeFormulaV2 model
    extract_with_preset("default", input_doc)

    # Extract with Granite Docling model
    extract_with_preset("granite_docling", input_doc)

    print(f"\n{'=' * 60}")
    print("Comparison complete!")
    print(f"{'=' * 60}")
    print("\nBoth presets have been tested. You can compare the outputs above.")
    print("\nKey differences:")
    print("- Default: Uses specialized CodeFormulaV2 model")
    print(
        "- Granite Docling: Uses IBM Granite-Docling-258M with extended context (8192 tokens)"
    )


if __name__ == "__main__":
    main()
