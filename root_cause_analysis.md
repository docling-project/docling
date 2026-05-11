# Root Cause Analysis for docling #3428

## Problem Statement
Chunk overflow with landscape tables in non-landscape mode when using HybridChunker with use_markdown_tables=True.

## Root Cause Analysis
1. **Primary Issue**: MarkdownTableSerializer incorrectly handles landscape tables
   - When table is in landscape orientation but serializer expects portrait mode
   - Table headers get repeated multiple times in the markdown output
   - This causes the chunk to exceed max_tokens limit (8192)

2. **Secondary Issue**: Token calculation doesn't account for markdown overhead
   - The prefix/suffix from markdown table format adds overhead
   - No validation that the actual serialized chunk size fits within max_tokens

## Potential Solutions

### Short-term (Workaround)
- Disable use_markdown_tables for landscape tables
- Add chunk size validation and fallback logic
- Implement chunk trimming to ensure compliance

### Long-term (Fix)
- Fix MarkdownTableSerializer to handle landscape tables correctly
- Add proper token counting with markdown overhead
- Implement better chunk validation

## Code Location Analysis
- Issue location: docling_core.transforms.chunker.hybrid_chunker.HybridChunker
- Serializer location: docling_core.internal.serializers.markdown.MarkdownTableSerializer
- Token calculation: Likely in docling_core.utils.tokenization or similar

## Next Steps
1. Fork docling_core to access the actual implementation
2. Reproduce the issue with test data
3. Implement fix at the appropriate level
