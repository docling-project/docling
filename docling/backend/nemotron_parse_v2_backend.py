# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Parser for NVIDIA Nemotron Parse 2.0 native output."""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    PictureMeta,
    ProvenanceItem,
    Size,
    TableData,
    TabularChartMetaField,
)
from PIL import Image as PILImage

from docling.backend.latex_backend import LatexDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

_MODEL_CANVAS = Size(width=1664, height=2048)
_REGION_PATTERN = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>"
    r"(.*?)"
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>"
    r"<class_([^>]+)>",
    re.DOTALL,
)
_HEADING_PATTERN = re.compile(r"^\s*(#{1,6})\s+")
_TEXT_LABELS: dict[str, DocItemLabel] = {
    "Text": DocItemLabel.TEXT,
    "Caption": DocItemLabel.CAPTION,
    "Footnote": DocItemLabel.FOOTNOTE,
    "Page-header": DocItemLabel.PAGE_HEADER,
    "Page-footer": DocItemLabel.PAGE_FOOTER,
    "Bibliography": DocItemLabel.REFERENCE,
    "Formula": DocItemLabel.FORMULA,
}


@dataclass(frozen=True)
class NemotronParseV2Region:
    """One decoded Nemotron Parse 2.0 layout region."""

    label: str
    bbox: tuple[float, float, float, float]
    text: str


class _MarkdownTextParser(HTMLParser):
    """Flatten rendered Markdown HTML while retaining explicit line breaks."""

    _BLOCK_TAGS = {"blockquote", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "br":
            self._append_break()

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._BLOCK_TAGS:
            self._append_break()

    def handle_data(self, data: str) -> None:
        self._parts.append(re.sub(r"\s+", " ", data))

    def _append_break(self) -> None:
        if self._parts and not self._parts[-1].endswith("\n"):
            self._parts[-1] = self._parts[-1].rstrip()
            self._parts.append("\n")

    def get_text(self) -> str:
        lines = (line.strip() for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line)


def extract_nemotron_parse_v2_regions(content: str) -> list[NemotronParseV2Region]:
    """Extract regions from Nemotron's native coordinate/class envelope."""
    regions: list[NemotronParseV2Region] = []
    for match in _REGION_PATTERN.finditer(content):
        x1, y1, text, x2, y2, label = match.groups()
        if label == "Inline-formula":
            label = "Formula"
        regions.append(
            NemotronParseV2Region(
                label=label,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                text=_remove_nemotron_formatting(text),
            )
        )
    return regions


def transform_nemotron_bbox(
    bbox: tuple[float, float, float, float],
    *,
    inference_image_size: Size,
    original_page_size: Size,
) -> BoundingBox:
    """Undo Nemotron's centered padding and map the box to page coordinates."""
    image_width = inference_image_size.width
    image_height = inference_image_size.height
    target_width = _MODEL_CANVAS.width
    target_height = _MODEL_CANVAS.height

    aspect_ratio = image_width / image_height
    resized_width = image_width
    resized_height = image_height
    if resized_height > target_height:
        resized_height = target_height
        resized_width = int(resized_height * aspect_ratio)
    if resized_width > target_width:
        resized_width = target_width
        resized_height = int(resized_width / aspect_ratio)

    pad_left = max(0, target_width - resized_width) // 2
    pad_top = max(0, target_height - resized_height) // 2
    x_scale = original_page_size.width / resized_width
    y_scale = original_page_size.height / resized_height

    left = ((bbox[0] * target_width) - pad_left) * x_scale
    right = ((bbox[2] * target_width) - pad_left) * x_scale
    top = ((bbox[1] * target_height) - pad_top) * y_scale
    bottom = ((bbox[3] * target_height) - pad_top) * y_scale

    return BoundingBox(
        l=max(0, min(original_page_size.width, left)),
        t=max(0, min(original_page_size.height, top)),
        r=max(0, min(original_page_size.width, right)),
        b=max(0, min(original_page_size.height, bottom)),
        coord_origin=CoordOrigin.TOPLEFT,
    )


def _remove_nemotron_formatting(text: str) -> str:
    return (
        text.replace("<tbc>", "")
        .replace(r"\<|unk|\>", "")
        .replace(r"\unknown", "")
        .strip()
    )


def _convert_with_backend(
    content: str,
    *,
    input_format: InputFormat,
    backend_class: type[LatexDocumentBackend] | type[MarkdownDocumentBackend],
) -> DoclingDocument:
    stream = BytesIO(f"{content}\n".encode())
    in_doc = InputDocument(
        path_or_stream=stream,
        filename=f"nemotron.{input_format.value}",
        format=input_format,
        backend=backend_class,
    )
    backend = backend_class(in_doc=in_doc, path_or_stream=stream)
    try:
        return backend.convert()
    finally:
        backend.unload()


def _markdown_to_text(markdown: str) -> str:
    if not markdown.strip():
        return ""
    try:
        import marko

        parser = _MarkdownTextParser()
        parser.feed(marko.convert(markdown))
        return parser.get_text()
    except Exception as exc:
        _log.warning("Failed to parse Nemotron Markdown: %s", exc)
        return markdown.strip()


def _markdown_cell_to_text(markdown: str) -> str:
    """Parse Markdown cell content using Docling's Markdown backend."""
    try:
        document = _convert_with_backend(
            markdown,
            input_format=InputFormat.MD,
            backend_class=MarkdownDocumentBackend,
        )
        return document.export_to_text().strip()
    except Exception as exc:
        _log.warning("Failed to parse Markdown in Nemotron table cell: %s", exc)
        return markdown.strip()


def _remove_trailing_empty_rows(table_data: TableData) -> None:
    while table_data.num_rows > 0:
        if any(cell.text.strip() for cell in table_data.grid[-1]):
            return
        table_data.remove_rows([table_data.num_rows - 1])


def _table_data_from_latex(latex: str) -> TableData:
    try:
        document = _convert_with_backend(
            latex,
            input_format=InputFormat.LATEX,
            backend_class=LatexDocumentBackend,
        )
        if document.tables:
            table_data = deepcopy(document.tables[0].data)
            _remove_trailing_empty_rows(table_data)
            for cell in table_data.table_cells:
                cell.text = _markdown_cell_to_text(cell.text)
                cell.column_header = cell.start_row_offset_idx == 0
            return table_data
    except Exception as exc:
        _log.warning("Failed to parse Nemotron LaTeX table: %s", exc)
    return TableData(num_rows=0, num_cols=0, table_cells=[])


def _table_data_from_markdown(markdown: str) -> TableData | None:
    try:
        document = _convert_with_backend(
            markdown,
            input_format=InputFormat.MD,
            backend_class=MarkdownDocumentBackend,
        )
        if document.tables:
            return deepcopy(document.tables[0].data)
    except Exception as exc:
        _log.warning("Failed to parse Nemotron chart Markdown: %s", exc)
    return None


def parse_nemotron_parse_v2(
    content: str,
    original_page_size: Size,
    inference_image_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: PILImage.Image | None = None,
) -> DoclingDocument:
    """Parse one page of Nemotron Parse 2.0 native output."""
    origin = DocumentOrigin(filename=filename, mimetype="text/plain", binary_hash=0)
    document = DoclingDocument(name=filename.rsplit(".", 1)[0], origin=origin)
    image_dpi = 72
    if page_image is not None:
        image_dpi = int(72 * page_image.width / original_page_size.width)
    document.add_page(
        page_no=page_no,
        size=original_page_size,
        image=(
            ImageRef.from_pil(image=page_image, dpi=image_dpi)
            if page_image is not None
            else None
        ),
    )

    current_list_group = None
    for region in extract_nemotron_parse_v2_regions(content):
        bbox = transform_nemotron_bbox(
            region.bbox,
            inference_image_size=inference_image_size,
            original_page_size=original_page_size,
        )
        provenance = ProvenanceItem(page_no=page_no, bbox=bbox, charspan=(0, 0))
        raw_text = region.text

        if region.label == "List-item":
            if current_list_group is None:
                current_list_group = document.add_list_group()
            document.add_list_item(
                text=_markdown_to_text(raw_text),
                orig=raw_text,
                parent=current_list_group,
                prov=provenance,
            )
            continue

        current_list_group = None
        if region.label == "Picture":
            document.add_picture(prov=provenance)
        elif region.label == "Table":
            document.add_table(data=_table_data_from_latex(raw_text), prov=provenance)
        elif region.label == "Chart":
            table_data = _table_data_from_markdown(raw_text)
            picture = document.add_picture(prov=provenance)
            if table_data is not None:
                picture.meta = PictureMeta(
                    tabular_chart=TabularChartMetaField(chart_data=table_data)
                )
        elif region.label == "Title":
            document.add_title(
                text=_markdown_to_text(raw_text), orig=raw_text, prov=provenance
            )
        elif region.label == "Section-header":
            heading_match = _HEADING_PATTERN.match(raw_text)
            level = len(heading_match.group(1)) if heading_match is not None else 1
            document.add_heading(
                text=_markdown_to_text(raw_text),
                orig=raw_text,
                level=level,
                prov=provenance,
            )
        else:
            label = _TEXT_LABELS.get(region.label, DocItemLabel.TEXT)
            text = (
                raw_text
                if label == DocItemLabel.FORMULA
                else _markdown_to_text(raw_text)
            )
            document.add_text(
                label=label,
                text=text,
                orig=raw_text,
                prov=provenance,
            )

    return document
