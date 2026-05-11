#!/usr/bin/env python3
"""Reproduce docling #3428: Chunk overflow with landscape tables"""

import tempfile
from pathlib import Path
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.service_options import HybridChunkerOptions
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.document import ConversionResult

def main():
    # Create test PDF with landscape table (mock for now)
    print("Creating test document with landscape table...")
    
    # Setup options
    pdf_opts = PdfPipelineOptions()
    chunk_opts = HybridChunkerOptions(
        max_tokens=8192,
        use_markdown_tables=True,
        merge_peers=True
    )
    
    # Create pipeline
    pipeline = SimplePipeline(
        pdf_options=pdf_opts,
        chunk_options=chunk_opts
    )
    
    # TODO: Need actual PDF test data
    print("Issue reproduction requires actual PDF test data")
    print("The bug happens when converting PDFs with landscape tables")

if __name__ == "__main__":
    main()
