#!/usr/bin/env python3
"""
Example demonstrating how to use HybridChunker with markdown text directly.

This example shows the solution to the issue:
"Can I use HybridChunker directly with markdown text instead of converting it to a docling document first?"

Answer: Yes! Now you can use the utility functions provided in docling.chunking.utils
"""

def main():
    # Sample markdown text to chunk
    markdown_text = """# IBM Company Overview

IBM is one of the world's largest technology companies.

## History

IBM was founded in 1911 as the Computing-Tabulating-Recording Company (CTR). It was renamed "International Business Machines" in 1924.

## Key Products

IBM has been responsible for several technological innovations:

- Automated teller machine (ATM)
- Dynamic random-access memory (DRAM)
- The floppy disk
- The hard disk drive
- The magnetic stripe card

## Modern Era

Today, IBM focuses on:

1. Cloud computing services
2. Artificial intelligence (Watson)
3. Quantum computing research
4. Enterprise software solutions

IBM continues to be a major player in the technology industry with operations worldwide.
"""

    print("=== Solution 1: Using utility functions ===")

    # Method 1: Convert text to DoclingDocument first, then chunk
    from docling.chunking import HybridChunker
    from docling.chunking.utils import markdown_to_docling_document

    # Convert markdown to DoclingDocument
    doc = markdown_to_docling_document(markdown_text)
    print(f"✓ Converted markdown to DoclingDocument: {doc.name}")

    # Create chunker and chunk the document
    chunker = HybridChunker()
    chunks = list(chunker.chunk(dl_doc=doc))

    print(f"✓ Generated {len(chunks)} chunks using HybridChunker")

    # Display first few chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  Text: {chunk.text[:100]}...")
        print(f"  Length: {len(chunk.text)} characters")

    print("\n" + "="*60)
    print("=== Solution 2: One-step text chunking ===")

    # Method 2: Direct text chunking in one step
    from docling.chunking.utils import chunk_text_with_hybrid_chunker

    # Chunk text directly
    chunks_direct = list(chunk_text_with_hybrid_chunker(markdown_text))

    print(f"✓ Generated {len(chunks_direct)} chunks directly from text")

    # Display first few chunks
    for i, chunk in enumerate(chunks_direct[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  Text: {chunk.text[:100]}...")
        if hasattr(chunk, "meta") and hasattr(chunk.meta, "headings"):
            print(f"  Headings: {chunk.meta.headings}")

    print("\n" + "="*60)
    print("=== Solution 3: Custom chunker configuration ===")

    # Method 3: Using custom chunker configuration
    chunks_custom = list(chunk_text_with_hybrid_chunker(
        markdown_text,
        chunker_kwargs={
            # You can pass any HybridChunker arguments here
            # "tokenizer": "sentence-transformers/all-MiniLM-L6-v2",
            # "merge_peers": True,
        }
    ))

    print(f"✓ Generated {len(chunks_custom)} chunks with custom configuration")

    print("\n" + "="*60)
    print("=== Comparison with traditional approach ===")

    # Traditional approach (for comparison)
    from io import BytesIO

    from docling.document_converter import DocumentConverter

    # This is how you would traditionally need to do it:
    # 1. Create a temporary file or stream
    # 2. Use DocumentConverter
    # 3. Extract the document
    # 4. Then chunk it

    print("Traditional approach requires these steps:")
    print("1. Create BytesIO stream or temporary file")
    print("2. Use DocumentConverter to convert")
    print("3. Extract DoclingDocument from result")
    print("4. Create HybridChunker and chunk the document")
    print("\nWith our solution, you can do it in just 1-2 lines!")

    print("\n" + "="*60)
    print("=== Summary ===")
    print("✓ Problem solved! You can now use HybridChunker with markdown text directly.")
    print("✓ Three approaches available:")
    print("  1. markdown_to_docling_document() + HybridChunker.chunk()")
    print("  2. chunk_text_with_hybrid_chunker() - one-step solution")
    print("  3. text_to_docling_document() for other text formats")
    print("✓ No need to manually create InputDocument or use DocumentConverter")
    print("✓ Preserves all HybridChunker functionality and configuration options")


if __name__ == "__main__":
    main()
