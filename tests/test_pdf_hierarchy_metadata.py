import re
from pathlib import Path

from docling_core.types.doc.document import SectionHeaderItem

from docling.document_converter import DocumentConverter

results_path = Path(__file__).parent / "results"
sample_path = Path(__file__).parent / "data" / "pdf"


def test_convert():
    ref_output = """  Some kind of text document
  1. Introduction
    1.1 Background
    1.2 Purpose
  2. Main Content
    2.1 Section One
      2.1.1 Subsection
      2.1.2 Another Subsection
    2.2 Section Two
  3. Conclusion"""

    refs = ref_output.split("\n")
    ref_tuples = [(len(re.match(r"^ *", s).group(0)) // 2, s.lstrip(" ")) for s in refs]
    found_i = 0

    source = (
        sample_path / "sample_document_hierarchical.pdf"
    )  # document per local path or URL
    converter = DocumentConverter()
    result = converter.convert(source)
    for el, level in result.document.iterate_items():
        if isinstance(el, SectionHeaderItem):
            if el.text == ref_tuples[found_i][1]:
                assert level == ref_tuples[found_i][0]
                found_i += 1
            else:
                assert level == ref_tuples[found_i][0] + 1
