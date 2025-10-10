# patched_pypdfium_backend.py
import re
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend as OriginalBackend

def remove_duplicate_characters(text: str) -> str:
    # Clean up punctuation / small duplicates
    text = re.sub(r'([,;:])\s*\1+', r'\1', text)
    text = re.sub(r'([a-zA-Z])\s*,\s*\1', r'\1', text)
    return text

class PyPdfiumDocumentBackend(OriginalBackend):
    def load_page(self, page_no):
        page = super().load_page(page_no)
        original_compute = page._compute_text_cells

        def patched_compute():
            cells = original_compute()
            # Merge small adjacent fragments if needed
            merged_cells = []
            prev_cell = None
            for cell in cells:
                if prev_cell and abs(cell.rect.to_bounding_box().l - prev_cell.rect.to_bounding_box().r) < 5:
                    # Merge text
                    prev_cell.text += cell.text
                    prev_cell.orig += cell.orig
                else:
                    merged_cells.append(cell)
                    prev_cell = cell
            # Apply regex cleanup
            for cell in merged_cells:
                cell.text = remove_duplicate_characters(cell.text)
            return merged_cells

        page._compute_text_cells = patched_compute
        return page
