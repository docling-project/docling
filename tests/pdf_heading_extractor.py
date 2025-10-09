import fitz
from typing import List, Dict

def extract_headings(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    headings = []

    for page_number, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b["type"] == 0:
                for line in b["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        size = span["size"]
                        bold = "bold" in span["font"].lower()
                        all_caps = text.isupper()
                        if text:
                            headings.append({
                                "text": text,
                                "size": size,
                                "bold": bold,
                                "all_caps": all_caps,
                                "page": page_number
                            })

    if not headings:
        return []

    # Determine likely headings: largest 2 font sizes OR all caps OR bold, ignoring tiny text
    unique_sizes = sorted({h["size"] for h in headings}, reverse=True)
    top_sizes = unique_sizes[:2]

    headings_filtered = [
        h for h in headings
        if (h['size'] in top_sizes or h['all_caps'] or h['bold']) and len(h['text']) > 3
    ]

    if not headings_filtered:
        return []

    # Map sizes to levels (largest = 1)
    size_to_level = {size: i+1 for i, size in enumerate(unique_sizes)}
    for h in headings_filtered:
        h["level"] = size_to_level.get(h["size"], 3)  # default level 3 if size not in top_sizes

    structured_headings = []
    current_chapter = ""
    current_section = ""

    for h in headings_filtered:
        if h["level"] == 1:
            current_chapter = h["text"]
        elif h["level"] == 2:
            current_section = h["text"]
        elif h["level"] >= 3:
            structured_headings.append({
                "chapter": current_chapter,
                "section": current_section,
                "title": h["text"],
                "page": h["page"]
            })

    return structured_headings


# Example usage
if __name__ == "__main__":
    pdf_file = r"C:\Users\Admin\Desktop\demo\docling\docling\cli\sample.pdf"
    chunks = extract_headings(pdf_file)
    if not chunks:
        print("No headings found!")
    else:
        for c in chunks:
            print(f"Chapter: {c['chapter']}, Section: {c['section']}, Title: {c['title']}, Page: {c['page']}")
