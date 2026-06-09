Here is the Korean document you need to analyze:

<document>
{{raw_text}}
</document>

Your task is to extract and organize all structural elements from this document into a hierarchical table of contents. Korean documents often have mixed structures where some sections follow formal regulatory patterns (제x장/절/관/조) while others use general section numbering and headers.

## Analysis Process

Before generating the final table of contents, work through the document systematically in `<analysis>` tags. It's OK for this section to be quite long. Follow these steps:

1. **Document Title Extraction**: Quote the main document title exactly as it appears at the beginning of the document.

2. **Structural Marker Identification**: Scan through the document and quote all the key structural markers you find, such as:
   - Formal regulatory patterns: 제x장, 제x절, 제x관, 제x조
   - General section patterns: numbered headers (1., 2., etc.), lettered headers (가., 나., etc.)
   - Special sections: 부칙, 별지, 별표, etc.

3. **Systematic Section Extraction**: Work through the document from beginning to end, extracting each structural element in order:
   - For each main section, quote the exact title as it appears
   - For each subsection, quote the exact title and note which main section it belongs under
   - For each article/item, quote the exact title and note its parent section
   - Include any appendices, attachments, and addenda

4. **Hierarchy Building**: For each extracted element, explicitly note:
   - What level it should be at (main section, subsection, sub-subsection, etc.)
   - What its parent section is (if any)
   - What numbering it should receive in the final TOC (1., 1.1., 1.1.1., etc.)

5. **Structure Verification**: Review your extracted elements to ensure:
   - All structural elements are captured in document order
   - The hierarchy makes logical sense
   - No elements are duplicated or missed

## Output Requirements

After your analysis, generate the table of contents with this exact format:

```
<toc>
TITLE:<document title>
1. <first main section title>
1.1. <first subsection title>
1.1.1. <first sub-subsection title>
1.2. <second subsection title>
2. <second main section title>
2.1. <subsection under second main section>
3. <third main section title>
</toc>
```

## Formatting Guidelines

- Start with `TITLE:` followed by the document title
- Use hierarchical decimal numbering (1, 1.1, 1.1.1, etc.)
- Follow each number with a space and the original title exactly as it appears
- Maintain the document's logical hierarchy
- Include appendices, attachments, and addenda as separate top-level items
- Extract titles exactly as they appear - do not include explanatory content
- Handle both formal regulatory structures and general section headers
- Wrap the entire table of contents in `<toc></toc>` tags
