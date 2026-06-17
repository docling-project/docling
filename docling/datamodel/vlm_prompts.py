"""Shared VLM prompt constants owned by the datamodel layer."""

CHANDRA_ALLOWED_TAGS = (
    "['math', 'br', 'i', 'b', 'u', 'del', 'sup', 'sub', 'table', 'tr', 'td', "
    "'p', 'th', 'div', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'ul', 'ol', 'li', "
    "'input', 'a', 'span', 'img', 'hr', 'tbody', 'small', 'caption', 'strong', "
    "'thead', 'big', 'code', 'chem']"
)
CHANDRA_ALLOWED_ATTRS = (
    "['class', 'colspan', 'rowspan', 'display', 'checked', 'type', 'border', "
    "'value', 'style', 'href', 'alt', 'align', 'data-bbox', 'data-label']"
)
CHANDRA_PROMPT_ENDING = (
    f"Only use these tags {CHANDRA_ALLOWED_TAGS}, "
    f"and these attributes {CHANDRA_ALLOWED_ATTRS}.\n\n"
    "Guidelines:\n"
    "* Inline math: Surround math with <math>...</math> tags. Math expressions "
    "should be rendered in KaTeX-compatible LaTeX. Use display for block math.\n"
    "* Tables: Use colspan and rowspan attributes to match table structure.\n"
    "* Formatting: Maintain consistent formatting with the image, including spacing, "
    "indentation, subscripts/superscripts, and special characters.\n"
    "* Images: Include a description of any images in the alt attribute of an <img> tag. "
    "Do not fill out the src property. Describe in detail inside the div tag. "
    "Also convert charts to high fidelity data, and convert diagrams to mermaid.\n"
    "* Forms: Mark checkboxes and radio buttons properly.\n"
    "* Text: join lines together properly into paragraphs using <p>...</p> tags. "
    "Use <br> tags for line breaks within paragraphs, but only when absolutely "
    "necessary to maintain meaning.\n"
    "* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.\n"
    "* Lists: Preserve indents and proper list markers.\n"
    "* Use the simplest possible HTML structure that accurately represents the content "
    "of the block.\n"
    "* Make sure the text is accurate and easy for a human to read and interpret. "
    "Reading order should be correct and natural."
)
CHANDRA_OCR_LAYOUT_PROMPT = (
    "OCR this image to HTML, arranged as layout blocks. Each layout block should be "
    "a div with the data-bbox attribute representing the bounding box of the block in "
    "x0 y0 x1 y1 format. Bboxes are normalized 0-1000. The data-label attribute is "
    "the label for the block.\n\n"
    "Use the following labels:\n"
    "- Caption\n- Footnote\n- Equation-Block\n- List-Group\n- Page-Header\n"
    "- Page-Footer\n- Image\n- Section-Header\n- Table\n- Text\n- Complex-Block\n"
    "- Code-Block\n- Form\n- Table-Of-Contents\n- Figure\n- Chemical-Block\n"
    "- Diagram\n- Bibliography\n- Blank-Page\n\n" + CHANDRA_PROMPT_ENDING
)

DOTS_LAYOUT_PROMPT = (
    "Please output the layout information from the PDF image, including each layout "
    "element's bbox, its category, and the corresponding text content within the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', "
    "'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', "
    "'Text', 'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n\n"
    "5. Final Output: The entire output must be a single JSON object."
)
