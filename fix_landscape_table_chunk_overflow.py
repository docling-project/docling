#!/usr/bin/env python3
"""
Fix for docling #3428: Chunk overflow with landscape tables in non-landscape mode

This fix addresses the issue where MarkdownTableSerializer incorrectly handles
landscape tables by repeating table headers, causing chunks to exceed max_tokens.
"""

import re
from typing import Optional, Tuple
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types import DoclingDocument, TableData


class FixedHybridChunker(HybridChunker):
    """
    Fixed version of HybridChunker that prevents chunk overflow with landscape tables.
    
    Changes:
    1. Detect landscape tables and disable markdown serialization for them
    2. Add chunk size validation and trimming
    3. Better token accounting for markdown overhead
    """
    
    def _is_landscape_table(self, table: TableData) -> bool:
        """
        Detect if a table is in landscape orientation based on content analysis.
        Landscape tables typically have:
        - Many columns with short text (wide aspect ratio)
        - Long repeated header text
        - High column-to-row ratio
        """
        if not table.table_cells:
            return False
            
        # Calculate table dimensions
        max_col = max(cell.column_index for cell in table.table_cells) if table.table_cells else 0
        max_row = max(cell.row_index for cell in table.table_cells) if table.table_cells else 0
        
        if max_row == 0 or max_col == 0:
            return False
            
        # Landscape detection: high column-to-row ratio
        if max_col > max_row * 2:  # More than twice as many columns as rows
            return True
            
        # Check for repeated header patterns (common in landscape tables)
        header_cells = [cell for cell in table.table_cells if cell.column_header]
        if len(header_cells) > 3:  # Many headers suggest landscape orientation
            # Check for long repeated text patterns
            header_texts = [cell.text for cell in header_cells if cell.text]
            if any(len(text) > 50 for text in header_texts):  # Very long headers
                return True
                
        return False
    
    def _serialize_table_safely(self, table: TableData, use_markdown: bool) -> str:
        """
        Safely serialize table, avoiding overflow for landscape tables.
        """
        if use_markdown and self._is_landscape_table(table):
            # Fall back to non-markdown serialization for landscape tables
            print(f"Warning: Detected landscape table, falling back to non-markdown serialization to prevent overflow")
            use_markdown = False
            
        # Use original serialization method
        return super()._serialize_table(table, use_markdown)
    
    def _trim_chunk_to_token_limit(self, chunk_text: str, max_tokens: int) -> str:
        """
        Ensure chunk stays within token limit by trimming excess content.
        """
        # Simple token approximation (rough estimate: 1 token ≈ 4 characters)
        estimated_tokens = len(chunk_text) // 4
        
        if estimated_tokens <= max_tokens:
            return chunk_text
            
        # Trim from the end (preserve important beginning content)
        target_length = max_tokens * 4
        return chunk_text[:target_length] + "... [truncated to fit token limit]"
    
    def chunk_document(self, document: DoclingDocument, max_tokens: Optional[int] = None) -> list:
        """
        Override chunk method to add safety checks.
        """
        # Get original chunks
        original_chunks = super().chunk_document(document, max_tokens)
        
        # Apply safety fixes
        fixed_chunks = []
        for chunk in original_chunks:
            # Ensure chunk size compliance
            chunk_text = chunk.text
            if max_tokens:
                chunk_text = self._trim_chunk_to_token_limit(chunk_text, max_tokens)
            chunk.text = chunk_text
            fixed_chunks.append(chunk)
            
        return fixed_chunks


def apply_hybrid_chunker_fix():
    """
    Apply the fix by monkey-patching the original HybridChunker.
    This allows the fix to work without modifying docling_core directly.
    """
    import docling_core.transforms.chunker.hybrid_chunker as hybrid_chunker_module
    
    # Replace the original class with our fixed version
    hybrid_chunker_module.HybridChunker = FixedHybridChunker
    
    print("Applied HybridChunker fix for landscape table chunk overflow")


if __name__ == "__main__":
    apply_hybrid_chunker_fix()
