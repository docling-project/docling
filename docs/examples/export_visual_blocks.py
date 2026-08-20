# %% [markdown]
# Export a clean reader-oriented Markdown document from a research PDF.
#
# What this example does
# - Uses Docling for semantic structure, tables, and detected pictures.
# - Uses the source PDF geometry to keep formulas, figures, tables, and charts in
#   their original positions.
# - Suppresses OCR/layout text that lies inside a detected visual region, avoiding
#   the duplicated chart labels and garbled reading order that can otherwise occur.
# - Falls back to PDF vector/raster geometry for captioned figures and tables that
#   are not emitted as Docling visual items.
#
# Prerequisites
# - Install `pymupdf` in addition to Docling: `pip install pymupdf`.
#
# How to run
# - From the repo root: `python docs/examples/export_visual_blocks.py`.
# - Use `--input path/to/paper.pdf` and `--output scratch/reader-export` to
#   export another document.
# - Use `--pages 3 4` for a fast representative-page export while testing.
# - The Markdown document and cropped assets are written to `scratch/reader-export/`.
#
# Input document
# - Defaults to `tests/data/pdf/sources/2206.01062.pdf`. Change `input_pdf` as needed.

# %%

from __future__ import annotations

import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
from docling_core.types.doc import PictureItem, TableItem, TextItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

CAPTION_RE = re.compile(r"^(table|figure|fig\\.?)\\s+\\d+\\s*[:.]", re.IGNORECASE)
OVERLAP_THRESHOLD = 0.80
ASSET_SCALE = 2.0


def label_name(item: object) -> str:
    """Return a stable label string for a Docling item."""
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label)).lower()


def item_text(item: object) -> str:
    return re.sub(r"\\s+", " ", str(getattr(item, "text", "") or "")).strip()


def item_bbox(
    item: object, pdf: fitz.Document
) -> tuple[int, tuple[float, float, float, float]] | None:
    """Return a source-PDF top-left bounding box for a Docling item."""
    provenance = getattr(item, "prov", [])
    if not provenance:
        return None
    source = provenance[0]
    bbox = source.bbox
    if source.page_no < 1 or source.page_no > len(pdf):
        return None
    top, bottom = bbox.t, bbox.b
    if "BOTTOM" in str(getattr(bbox, "coord_origin", "TOPLEFT")).upper():
        page_height = pdf[source.page_no - 1].rect.height
        top, bottom = page_height - top, page_height - bottom
    return source.page_no, (bbox.l, min(top, bottom), bbox.r, max(top, bottom))


def overlap_ratio(
    inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]
) -> float:
    """Return the fraction of *inner* covered by *outer*."""
    left = max(inner[0], outer[0])
    top = max(inner[1], outer[1])
    right = min(inner[2], outer[2])
    bottom = min(inner[3], outer[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area = max(0.000001, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return intersection / area


def crop_item(
    pdf: fitz.Document, item: object, destination: Path, padding: float = 4
) -> bool:
    """Crop a Docling provenance box directly from the source PDF."""
    provenance = getattr(item, "prov", [])
    if not provenance:
        return False

    source = provenance[0]
    page = pdf[source.page_no - 1]
    bbox = source.bbox
    top, bottom = bbox.t, bbox.b
    origin = str(getattr(bbox, "coord_origin", "TOPLEFT")).upper()
    if "BOTTOM" in origin:
        top, bottom = page.rect.height - top, page.rect.height - bottom
    clip = fitz.Rect(bbox.l, min(top, bottom), bbox.r, max(top, bottom))
    # `fitz.Rect` overloads addition to expand a rectangle by these offsets.
    clip = clip + (-padding, -padding, padding, padding)  # noqa: RUF005
    clip &= page.rect
    if clip.is_empty or clip.get_area() <= 0:
        return False
    page.get_pixmap(
        matrix=fitz.Matrix(ASSET_SCALE, ASSET_SCALE), clip=clip, alpha=False
    ).save(destination)
    return True


def merge_visual_regions(page: fitz.Page) -> list[fitz.Rect]:
    """Find raster and vector regions, including line-built tables and charts."""
    regions: list[fitz.Rect] = []
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing["rect"])
        if max(rect.width, rect.height) < 4:
            continue
        # PDF table borders are commonly zero-width or zero-height paths.
        if rect.width < 2:
            rect.x0 -= 1
            rect.x1 += 1
        if rect.height < 2:
            rect.y0 -= 1
            rect.y1 += 1
        regions.append(rect)
    for image in page.get_images(full=True):
        regions.extend(
            rect
            for rect in page.get_image_rects(image[0])
            if rect.width >= 4 and rect.height >= 4
        )

    groups: list[fitz.Rect] = []
    for region in sorted(regions, key=lambda rect: (rect.y0, rect.x0)):
        for index, group in enumerate(groups):
            if fitz.Rect(
                group.x0 - 18, group.y0 - 18, group.x1 + 18, group.y1 + 18
            ).intersects(region):
                groups[index] = group | region
                break
        else:
            groups.append(region)

    changed = True
    while changed:
        changed = False
        merged: list[fitz.Rect] = []
        for region in groups:
            for index, group in enumerate(merged):
                if fitz.Rect(
                    group.x0 - 18, group.y0 - 18, group.x1 + 18, group.y1 + 18
                ).intersects(region):
                    merged[index] = group | region
                    changed = True
                    break
            else:
                merged.append(region)
        groups = merged
    return [region for region in groups if region.get_area() >= 120]


def fallback_visuals(
    pdf: fitz.Document,
    known_visuals: dict[int, list[tuple[float, float, float, float]]],
) -> Iterable[tuple[str, str, int, fitz.Rect]]:
    """Yield captioned source-PDF visuals missed by the semantic conversion."""
    for page_number, page in enumerate(pdf, start=1):
        regions = merge_visual_regions(page)
        if not regions:
            continue
        captions = []
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            caption = re.sub(r"\\s+", " ", text or "").strip()
            match = CAPTION_RE.match(caption)
            if match:
                captions.append(
                    (match.group(1).lower(), caption, fitz.Rect(x0, y0, x1, y1))
                )
        for marker, caption, caption_box in captions:
            kind = "table" if marker == "table" else "figure"
            before = [region for region in regions if region.y1 <= caption_box.y0 + 16]
            after = [region for region in regions if region.y0 >= caption_box.y1 - 16]
            candidates = (after if kind == "table" else before) or (
                before if kind == "table" else after
            )
            if not candidates:
                continue
            region = min(
                candidates,
                key=lambda rect: min(
                    abs(rect.y0 - caption_box.y1), abs(caption_box.y0 - rect.y1)
                ),
            )
            source_box = (region.x0, region.y0, region.x1, region.y1)
            if any(
                overlap_ratio(source_box, known) >= OVERLAP_THRESHOLD
                for known in known_visuals[page_number]
            ):
                continue
            yield kind, caption, page_number, region


def write_visual_markdown(
    lines: list[str], asset_name: str, caption: str, page_number: int
) -> None:
    lines.extend(
        (
            f"![{caption or 'visual'}](assets/{asset_name})",
            "",
            f"*{caption or f'Visual on page {page_number}'}*",
            "",
        )
    )


def main(
    input_pdf: Path, output_dir: Path, page_range: tuple[int, int] | None = None
) -> None:
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = ASSET_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    # Preserve source visuals directly; enrichment models are not needed to
    # position a chart, table, or formula in the reading document.
    pipeline_options.do_ocr = False
    pipeline_options.do_formula_enrichment = False
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    if page_range:
        document = converter.convert(input_pdf, page_range=page_range).document
    else:
        document = converter.convert(input_pdf).document
    source_pdf = fitz.open(input_pdf)

    visual_boxes: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for visual in [*document.pictures, *document.tables]:
        location = item_bbox(visual, source_pdf)
        if location:
            visual_boxes[location[0]].append(location[1])

    fallback_items = list(fallback_visuals(source_pdf, visual_boxes))

    def reading_key(entry: tuple[str, object]) -> tuple[int, float, float, int]:
        entry_type, value = entry
        if entry_type == "fallback":
            _kind, _caption, page_number, region = value  # type: ignore[misc]
            return page_number, region.y0, region.x0, 1
        location = item_bbox(value, source_pdf)
        if not location:
            return 0, 0, 0, 0
        page_number, (left, top, _right, _bottom) = location
        return page_number, top, left, 0

    entries: list[tuple[str, object]] = [
        ("docling", item) for item, _level in document.iterate_items()
    ]
    entries.extend(("fallback", item) for item in fallback_items)
    entries.sort(key=reading_key)

    lines = [f"# {input_pdf.stem}", ""]
    written_captions: set[str] = set()
    visual_count = 0
    formula_count = 0

    for entry_type, value in entries:
        if entry_type == "fallback":
            kind, caption, page_number, region = value  # type: ignore[misc]
            visual_count += 1
            asset_name = f"source-{kind}-{visual_count}.png"
            page = source_pdf[page_number - 1]
            page.get_pixmap(
                matrix=fitz.Matrix(ASSET_SCALE, ASSET_SCALE), clip=region, alpha=False
            ).save(assets_dir / asset_name)
            written_captions.add(caption)
            write_visual_markdown(lines, asset_name, caption, page_number)
            continue

        item = value
        label = label_name(item)
        text = item_text(item)
        location = item_bbox(item, source_pdf)

        if isinstance(item, (PictureItem, TableItem)):
            visual_count += 1
            kind = "table" if isinstance(item, TableItem) else "figure"
            caption = item.caption_text(document).strip()
            if caption:
                written_captions.add(caption)
            asset_name = f"{kind}-{visual_count}.png"
            item.get_image(document).save(assets_dir / asset_name, "PNG")
            write_visual_markdown(
                lines, asset_name, caption, location[0] if location else 0
            )
            if isinstance(item, TableItem):
                lines.extend(
                    (item.export_to_dataframe(document).to_markdown(index=False), "")
                )
            continue

        if not isinstance(item, TextItem) or not text:
            continue
        if text in written_captions:
            continue
        if (
            label != "caption"
            and location
            and any(
                overlap_ratio(location[1], visual) >= OVERLAP_THRESHOLD
                for visual in visual_boxes[location[0]]
            )
        ):
            # The visual crop is authoritative for its chart labels, table cells,
            # and diagram glyphs. Do not emit them again as reading text.
            continue
        if label in {"title", "document_title"}:
            lines.extend((f"# {text}", ""))
        elif label in {"section_header", "heading"}:
            lines.extend((f"## {text}", ""))
        elif label == "formula":
            formula_count += 1
            asset_name = f"formula-{formula_count}.png"
            if crop_item(source_pdf, item, assets_dir / asset_name):
                write_visual_markdown(
                    lines, asset_name, "Formula", location[0] if location else 0
                )
            else:
                lines.extend(("```text", text, "```", ""))
        else:
            lines.extend((text, ""))

    markdown_path = output_dir / f"{input_pdf.stem}-reader.md"
    markdown_path.write_text("\\n".join(lines), encoding="utf-8")
    print(f"Wrote reader Markdown to {markdown_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Export a visual-safe reader Markdown file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tests/data/pdf/sources/2206.01062.pdf"),
        help="PDF to convert.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scratch/reader-export"),
        help="Directory for the Markdown file and visual assets.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        help="Inclusive 1-based page range to convert.",
    )
    arguments = parser.parse_args()
    main(
        arguments.input,
        arguments.output,
        tuple(arguments.pages) if arguments.pages else None,
    )
