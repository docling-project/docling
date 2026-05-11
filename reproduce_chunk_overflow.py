#!/usr/bin/env python3
"""Reproduce docling #3428: Chunk overflow with landscape tables"""

import tempfile
import sys
from pathlib import Path

# Add local path to import docling
sys.path.insert(0, '/tmp/docling')

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.service_options import HybridChunkerOptions
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.document import ConversionResult

def main():
    """Test chunking with a landscape table scenario"""
    
    # Create test PDF with landscape table content
    print("Testing chunk overflow scenario...")
    
    # Setup chunking options like in the issue
    chunk_opts = HybridChunkerOptions(
        max_tokens=8192,  # As mentioned in the issue
        use_markdown_tables=True,  # Trigger the bug
        merge_peers=True
    )
    
    print(f"Chunk options: max_tokens={chunk_opts.max_tokens}, use_markdown_tables={chunk_opts.use_markdown_tables}")
    
    # Create a mock landscape table scenario
    # The issue: when converting landscape tables in non-landscape mode,
    # table headers get repeated, causing chunk overflow
    
    print("Issue: Landscape tables in non-landscape mode cause header repetition")
    print("Root cause: MarkdownTableSerializer serializes landscape tables incorrectly")
    print("Expected: Chunk stays within max_tokens limit")
    print("Actual: Chunk exceeds 8192 tokens due to repeated headers")
    
    # This would require actual PDF data to fully reproduce
    print("Full reproduction needs actual PDF test data with landscape tables")
    
    return True

if __name__ == "__main__":
    main()
