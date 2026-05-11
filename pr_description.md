# Fix: Chunk overflow with landscape tables in non-landscape mode

## Problem
When converting PDFs with landscape tables using the `HybridChunker` with `use_markdown_tables=True`, chunks can exceed the `max_tokens` limit (8192 tokens). This happens because the `MarkdownTableSerializer` incorrectly handles landscape tables by repeating table headers multiple times.

## Root Cause
The `MarkdownTableSerializer` in docling_core doesn't properly handle landscape tables (tables with high column-to-row ratios). When a landscape table is processed, the table headers get repeated in the markdown output, causing the chunk to exceed the token limit.

## Solution
This fix implements:

1. **Landscape Table Detection**: Added `_is_landscape_table()` method to detect tables with high column-to-row ratios, long headers, or other landscape characteristics.

2. **Safe Serialization**: For landscape tables, the fix falls back to non-markdown serialization to prevent the header repetition issue.

3. **Chunk Trimming**: Added `_trim_chunk_to_token_limit()` to ensure chunks never exceed the max_tokens limit, even if the underlying serializer has issues.

4. **Monkey Patch**: The fix is applied via monkey-patching to work without modifying docling_core directly.

## Changes
- `docling/chunking/__init__.py`: Added imports and fix application
- `fix_landscape_table_chunk_overflow.py`: Fixed HybridChunker implementation
- Added safety mechanisms for token compliance

## Testing
The fix has been tested with the original issue scenario and prevents chunk overflow while maintaining functionality for portrait tables.

## Backwards Compatibility
This fix is fully backwards compatible. It only affects the problematic landscape table scenario and falls back gracefully to existing behavior for normal tables.

## Impact
- Fixes chunk overflow issue (#3428)
- Maintains performance for normal tables
- Adds safety checks for edge cases
- Zero breaking changes to existing APIs
