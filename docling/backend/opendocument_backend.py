"""Backends to parse OpenDocument formats (ODT, ODS, ODP).

The backends leverage the ``odfdo`` library (https://github.com/jdum/odfdo) to
read the underlying XML structure and translate it into a :class:`DoclingDocument`.

The conventions used here mirror the corresponding Microsoft Office backends
(:mod:`docling.backend.msword_backend`, :mod:`docling.backend.mspowerpoint_backend`,
:mod:`docling.backend.msexcel_backend`).

Known gaps to improve:
- rich text styling is not preserved yet;
- ODP table extraction and ordered-list markers are incomplete;
- ODS conversion focuses on cell/table content rather than spreadsheet rendering
  fidelity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    Formatting,
    GroupLabel,
    ImageRef,
    NodeItem,
    PictureClassificationLabel,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    ProvenanceItem,
    RichTableCell,
    Script,
    Size,
    TableCell,
    TableData,
    TableItem,
    TabularChartMetaField,
)
from PIL import Image as PILImage
from typing_extensions import override

from docling.backend.abstract_backend import (
    DeclarativeDocumentBackend,
    PaginatedDocumentBackend,
)
from docling.datamodel.backend_options import OdsBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_ODFDO_AVAILABLE: bool = False
_ODFDO_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - import-time guard
    from odfdo import (
        Document as OdfDocument,
        DrawPage,
        Frame,
        Header,
        List as OdfList,
        ListItem,
        Paragraph,
        Section,
        Table as OdfTable,
    )

    _ODFDO_AVAILABLE = True
except ImportError as e:  # pragma: no cover - import-time guard
    _ODFDO_IMPORT_ERROR = e

_log = logging.getLogger(__name__)

_INSTALL_HINT = (
    "The 'odfdo' package is required to process OpenDocument files. "
    "Install it with `pip install 'docling-slim[format-opendocument]'`."
)

_ODF_CHART_CLASS_TO_PICTURE_CLASSIFICATION = {
    "chart:bar": PictureClassificationLabel.BAR_CHART,
    "chart:line": PictureClassificationLabel.LINE_CHART,
    "chart:circle": PictureClassificationLabel.PIE_CHART,
    "chart:pie": PictureClassificationLabel.PIE_CHART,
    "chart:scatter": PictureClassificationLabel.SCATTER_PLOT,
}


@dataclass
class _OdfListState:
    group: NodeItem
    last_item: NodeItem | None
    enumerated: bool
    counter: int


@dataclass
class _OdfTextRun:
    text: str
    formatting: Formatting | None = None


def _load_odf_document(
    path_or_stream: BytesIO | Path, document_hash: str
) -> OdfDocument:
    """Load an ODF document from a path or in-memory stream."""
    try:
        if isinstance(path_or_stream, BytesIO):
            return OdfDocument(path_or_stream)
        return OdfDocument(str(path_or_stream))
    except Exception as e:
        raise RuntimeError(
            f"OpenDocument backend could not load document with hash {document_hash}"
        ) from e


class _OdfBaseBackend(DeclarativeDocumentBackend):
    """Shared loading / validation logic for ODT, ODS and ODP backends."""

    _odf_type: str = ""  # "text", "spreadsheet" or "presentation"

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: OdsBackendOptions | None = None,
    ) -> None:
        if not _ODFDO_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _ODFDO_IMPORT_ERROR
        super().__init__(in_doc, path_or_stream, options)
        self.path_or_stream: BytesIO | Path = path_or_stream
        self.valid: bool = False
        self.odf_obj: OdfDocument = _load_odf_document(
            path_or_stream, self.document_hash
        )
        if self._odf_type and self.odf_obj.get_type() != self._odf_type:
            raise RuntimeError(
                f"Expected an OpenDocument {self._odf_type!r} but got "
                f"{self.odf_obj.get_type()!r}"
            )
        self.valid = True

    @override
    def is_valid(self) -> bool:
        return self.valid

    @override
    def unload(self):
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()
        self.path_or_stream = None


def _find_true_data_bounds(table: OdfTable) -> tuple[int, int, int, int]:
    """Find the true data boundaries (min/max rows and columns) in an ODS table.

    This function scans all cells to find the smallest rectangular region that contains
    all non-empty cells or merged cell ranges, similar to the Excel backend approach.

    Args:
        table: The ODF table to analyze.

    Returns:
        A tuple (min_row, max_row, min_col, max_col) representing the 0-based indices
        of the data region. If the table is empty, returns (0, 0, 0, 0).
    """
    min_row, min_col = None, None
    max_row, max_col = 0, 0

    # Scan all rows and cells to find non-empty cells
    for row_idx, row in enumerate(table.traverse()):
        for col_idx, cell in enumerate(row.traverse()):
            # Check if cell has content (value or is part of a span)
            if _odf_cell_has_content(cell) or cell.tag == "table:covered-table-cell":
                if min_row is None:
                    min_row = row_idx
                if min_col is None or col_idx < min_col:
                    min_col = col_idx
                max_row = max(max_row, row_idx)
                max_col = max(max_col, col_idx)

            # Also check for cells with spans (they define data regions)
            if cell.tag != "table:covered-table-cell":
                attrs = cell.attributes
                row_span = int(attrs.get("table:number-rows-spanned") or 1)
                col_span = int(attrs.get("table:number-columns-spanned") or 1)
                if row_span > 1 or col_span > 1:
                    if min_row is None:
                        min_row = row_idx
                    if min_col is None or col_idx < min_col:
                        min_col = col_idx
                    max_row = max(max_row, row_idx + row_span - 1)
                    max_col = max(max_col, col_idx + col_span - 1)

    # If no data found, return empty bounds
    if min_row is None or min_col is None:
        return (0, 0, 0, 0)

    return (min_row, max_row, min_col, max_col)


def _clean_odf_text_lines(text: str) -> list[str]:
    return [line for line in (line.strip() for line in text.splitlines()) if line]


def _formatting_or_none(formatting: Formatting) -> Formatting | None:
    return None if formatting == Formatting() else formatting


def _copy_formatting(formatting: Formatting | None) -> Formatting:
    if formatting is None:
        return Formatting()
    return formatting.model_copy()


def _is_bold_weight(value: str) -> bool:
    if value == "bold":
        return True
    if value in {"normal", ""}:
        return False
    try:
        return int(value) >= 600
    except ValueError:
        return False


def _formatting_from_odf_text_style(
    odf_obj: OdfDocument | None,
    style_name: str | None,
    base_formatting: Formatting | None = None,
) -> Formatting | None:
    formatting = _copy_formatting(base_formatting)
    if odf_obj is None or style_name is None:
        return _formatting_or_none(formatting)

    style = odf_obj.get_style("text", style_name)
    if style is None:
        return _formatting_or_none(formatting)

    props = style.get_properties() or {}
    font_weight = next(
        (
            props[name]
            for name in (
                "fo:font-weight",
                "style:font-weight-asian",
                "style:font-weight-complex",
            )
            if name in props
        ),
        None,
    )
    if font_weight is not None:
        formatting.bold = _is_bold_weight(font_weight)

    font_style = next(
        (
            props[name]
            for name in (
                "fo:font-style",
                "style:font-style-asian",
                "style:font-style-complex",
            )
            if name in props
        ),
        None,
    )
    if font_style is not None:
        formatting.italic = font_style in {"italic", "oblique"}

    underline_style = props.get("style:text-underline-style")
    if underline_style is not None:
        formatting.underline = underline_style != "none"

    line_through = props.get("style:text-line-through-style") or props.get(
        "style:text-line-through-type"
    )
    if line_through is not None:
        formatting.strikethrough = line_through != "none"

    text_position = props.get("style:text-position")
    if text_position is not None:
        if text_position.startswith("super"):
            formatting.script = Script.SUPER
        elif text_position.startswith("sub"):
            formatting.script = Script.SUB
        else:
            formatting.script = Script.BASELINE

    return _formatting_or_none(formatting)


def _odf_text_runs(
    element: Any,
    odf_obj: OdfDocument | None,
    inherited_formatting: Formatting | None = None,
) -> list[_OdfTextRun]:
    style_name = getattr(element, "attributes", {}).get("text:style-name")
    formatting = _formatting_from_odf_text_style(
        odf_obj, style_name, inherited_formatting
    )
    tag = getattr(element, "tag", None)
    if tag == "text:line-break":
        text = getattr(element, "text", "\n") or "\n"
        text_recursive = getattr(element, "text_recursive", "")
        if text_recursive.startswith(text):
            text = text_recursive
        return [_OdfTextRun(text=text, formatting=formatting)]
    if tag == "text:tab":
        return [_OdfTextRun(text="\t", formatting=formatting)]

    runs: list[_OdfTextRun] = []
    text = getattr(element, "text", "")
    if text:
        runs.append(_OdfTextRun(text=text, formatting=formatting))

    for child in getattr(element, "children", []):
        runs.extend(_odf_text_runs(child, odf_obj, formatting))

    if not runs and not getattr(element, "children", []):
        text_recursive = getattr(element, "text_recursive", "")
        if text_recursive:
            runs.append(_OdfTextRun(text=text_recursive, formatting=formatting))

    return runs


def _normalize_odf_text_runs(runs: list[_OdfTextRun]) -> list[_OdfTextRun]:
    merged_runs: list[_OdfTextRun] = []
    for run in runs:
        if run.text == "":
            continue
        if merged_runs and merged_runs[-1].formatting == run.formatting:
            merged_runs[-1].text += run.text
        else:
            merged_runs.append(_OdfTextRun(text=run.text, formatting=run.formatting))

    while merged_runs and merged_runs[0].text.strip() == "":
        merged_runs.pop(0)
    if merged_runs:
        merged_runs[0].text = merged_runs[0].text.lstrip()

    while merged_runs and merged_runs[-1].text.strip() == "":
        merged_runs.pop()
    if merged_runs:
        merged_runs[-1].text = merged_runs[-1].text.rstrip()

    return [run for run in merged_runs if run.text]


def _odf_text_from_runs(runs: list[_OdfTextRun]) -> str:
    runs = _normalize_odf_text_runs(runs)
    return "".join(run.text for run in runs).strip()


def _add_odf_text_runs(
    doc: DoclingDocument,
    runs: list[_OdfTextRun],
    *,
    label: DocItemLabel,
    parent: NodeItem | None,
    content_layer: ContentLayer | None,
) -> NodeItem | None:
    runs = _normalize_odf_text_runs(runs)
    if not runs:
        return None
    if len(runs) == 1:
        return doc.add_text(
            label=label,
            parent=parent,
            text=runs[0].text,
            content_layer=content_layer,
            formatting=runs[0].formatting,
        )

    inline_group = doc.add_inline_group(parent=parent, content_layer=content_layer)
    for run in runs:
        doc.add_text(
            label=label,
            parent=inline_group,
            text=run.text,
            content_layer=content_layer,
            formatting=run.formatting,
        )
    return inline_group


def _add_odf_heading(
    doc: DoclingDocument,
    element: Header,
    *,
    parent: NodeItem | None,
    content_layer: ContentLayer | None,
    odf_obj: OdfDocument | None,
) -> None:
    level = element.get_attribute_integer("text:outline-level") or 1
    runs = _odf_text_runs(element, odf_obj)
    runs = _normalize_odf_text_runs(runs)
    text = _odf_text_from_runs(runs)
    if not text:
        return
    if len(runs) == 1:
        doc.add_heading(
            parent=parent,
            text=text,
            level=max(1, level),
            content_layer=content_layer,
            formatting=runs[0].formatting,
        )
        return

    inline_group = doc.add_inline_group(parent=parent, content_layer=content_layer)
    for run in runs:
        doc.add_heading(
            parent=inline_group,
            text=run.text,
            level=max(1, level),
            content_layer=content_layer,
            formatting=run.formatting,
        )


def _odf_paragraph_style_names(
    odf_obj: OdfDocument | None, element: Paragraph
) -> set[str]:
    style_names: set[str] = set()
    style_name = element.attributes.get("text:style-name")
    if style_name is not None:
        style_names.add(style_name)

    if odf_obj is None or style_name is None:
        return style_names

    style = odf_obj.get_style("paragraph", style_name)
    if style is None:
        return style_names

    parent_style_name = style.attributes.get("style:parent-style-name")
    if parent_style_name is not None:
        style_names.add(parent_style_name)
    display_name = style.attributes.get("style:display-name")
    if display_name is not None:
        style_names.add(display_name)
    return style_names


def _add_odf_paragraph(
    doc: DoclingDocument,
    element: Paragraph,
    *,
    parent: NodeItem | None,
    content_layer: ContentLayer | None,
    odf_obj: OdfDocument | None,
) -> None:
    chart_count = _add_odf_charts(doc, element, parent, content_layer, odf_obj)
    images = element.get_images()
    image_count = _add_odf_images(
        doc,
        images,
        parent,
        content_layer,
        odf_obj,
        skip_object_replacements=chart_count > 0,
    )
    runs = _odf_text_runs(element, odf_obj)
    text = _odf_text_from_runs(runs)
    if images:
        stripped_text = _strip_odf_image_reference_text(text, images).strip()
        if stripped_text != text:
            runs = [_OdfTextRun(text=stripped_text)] if stripped_text else []
            text = stripped_text
    if image_count > 0 and _odf_text_is_generated_image_references(text, images):
        return
    if chart_count > 0 and ("ObjectReplacements" in text or not text):
        return

    style_names = _odf_paragraph_style_names(odf_obj, element)
    if "Title" in style_names:
        _add_odf_text_runs(
            doc,
            runs,
            label=DocItemLabel.TITLE,
            parent=parent,
            content_layer=content_layer,
        )
    elif "Subtitle" in style_names:
        text = _odf_text_from_runs(runs)
        if text:
            doc.add_heading(
                parent=parent,
                text=text,
                level=1,
                content_layer=content_layer,
                formatting=runs[0].formatting if len(runs) == 1 else None,
            )
    else:
        _add_odf_text_runs(
            doc,
            runs,
            label=DocItemLabel.TEXT,
            parent=parent,
            content_layer=content_layer,
        )


def _odf_element_text_lines(element: Any) -> list[str]:
    if isinstance(element, OdfList):
        lines: list[str] = []
        for child in element.children:
            if isinstance(child, ListItem):
                lines.extend(_odf_element_text_lines(child))
        return lines

    if isinstance(element, ListItem):
        lines: list[str] = []
        for child in element.children:
            lines.extend(_odf_element_text_lines(child))
        if lines:
            return lines
        return _clean_odf_text_lines(element.text_recursive)

    if isinstance(element, (Header, Paragraph)):
        return _clean_odf_text_lines(element.text_recursive)

    child_lines: list[str] = []
    for child in getattr(element, "children", []):
        child_lines.extend(_odf_element_text_lines(child))
    if child_lines:
        return child_lines

    return _clean_odf_text_lines(element.text_recursive)


def _odf_list_item_content(
    item: ListItem, *, flatten_nested_text: bool = True
) -> tuple[str, list[OdfList]]:
    text_parts: list[str] = []
    nested: list[OdfList] = []
    for child in item.children:
        if isinstance(child, OdfList):
            nested.append(child)
        elif isinstance(child, Paragraph):
            text_parts.extend(_clean_odf_text_lines(child.text_recursive))
    if not text_parts and (flatten_nested_text or not nested):
        text_parts.extend(_clean_odf_text_lines(item.text_recursive))
    return " ".join(text_parts), nested


def _odf_list_item_text_runs(
    item: ListItem,
    odf_obj: OdfDocument | None,
    *,
    flatten_nested_text: bool = True,
) -> list[_OdfTextRun]:
    runs: list[_OdfTextRun] = []
    has_nested = False
    for child in item.children:
        if isinstance(child, OdfList):
            has_nested = True
        elif isinstance(child, Paragraph):
            runs.extend(_odf_text_runs(child, odf_obj))
    if not runs and (flatten_nested_text or not has_nested):
        text = _odf_text_from_runs(_odf_text_runs(item, odf_obj))
        if text:
            runs.append(_OdfTextRun(text=text))
    return _normalize_odf_text_runs(runs)


def _odf_list_starts_with_empty_nested_item(
    odf_list: OdfList, *, flatten_nested_text: bool
) -> bool:
    for child in odf_list.children:
        if not isinstance(child, ListItem):
            continue
        text, nested = _odf_list_item_content(
            child, flatten_nested_text=flatten_nested_text
        )
        return text == "" and any(
            _odf_list_has_renderable_content(
                item, flatten_nested_text=flatten_nested_text
            )
            for item in nested
        )
    return False


def _odf_list_has_direct_item_text(
    odf_list: OdfList, *, flatten_nested_text: bool
) -> bool:
    for child in odf_list.children:
        if not isinstance(child, ListItem):
            continue
        text, _nested = _odf_list_item_content(
            child, flatten_nested_text=flatten_nested_text
        )
        if text:
            return True
    return False


def _odf_list_has_renderable_content(
    odf_list: OdfList, *, flatten_nested_text: bool = True
) -> bool:
    for child in odf_list.children:
        if not isinstance(child, ListItem):
            continue
        text, nested = _odf_list_item_content(
            child, flatten_nested_text=flatten_nested_text
        )
        if text or any(
            _odf_list_has_renderable_content(
                item, flatten_nested_text=flatten_nested_text
            )
            for item in nested
        ):
            return True
    return False


def _odf_list_level_style(
    odf_obj: OdfDocument | None, odf_list: OdfList, level: int
) -> Any | None:
    if odf_obj is None:
        return None

    style_name = odf_list.attributes.get("text:style-name")
    if style_name is None:
        return None

    style = odf_obj.get_style("list", style_name)
    if style is None:
        return None

    return style.get_level_style(level)


def _odf_list_level_is_enumerated(
    odf_obj: OdfDocument | None,
    odf_list: OdfList,
    level: int,
    fallback: bool,
) -> bool:
    level_style = _odf_list_level_style(odf_obj, odf_list, level)
    if level_style is None:
        return fallback
    return level_style.tag == "text:list-level-style-number"


def _odf_list_start_value(
    odf_obj: OdfDocument | None,
    odf_list: OdfList,
    level: int,
) -> int:
    level_style = _odf_list_level_style(odf_obj, odf_list, level)
    if level_style is None:
        return 1

    start_value = level_style.attributes.get("text:start-value")
    if start_value is None:
        return 1

    try:
        return max(1, int(start_value))
    except ValueError:
        return 1


def _odf_list_marker(
    counter: int,
    odf_obj: OdfDocument | None,
    odf_list: OdfList,
    level: int,
) -> str:
    level_style = _odf_list_level_style(odf_obj, odf_list, level)
    suffix = "."
    if level_style is not None:
        suffix = level_style.attributes.get("style:num-suffix") or suffix
    return f"{counter}{suffix}"


def _odf_table_has_content(table: OdfTable) -> bool:
    for row in table.traverse():
        for cell in row.traverse():
            if cell.tag == "table:covered-table-cell":
                return True
            if _odf_cell_has_content(cell):
                return True
    return False


def _odf_cell_has_rich_content(cell: Any) -> bool:
    if _odf_cell_has_images(cell):
        return True

    non_empty_paragraphs = 0
    for child in cell.children:
        if isinstance(child, OdfList):
            if _odf_list_has_renderable_content(child):
                return True
        elif isinstance(child, Header):
            if _clean_odf_text_lines(child.text_recursive):
                return True
        elif isinstance(child, Paragraph):
            if _clean_odf_text_lines(child.text_recursive):
                non_empty_paragraphs += 1
            if child.get_images():
                return True
        elif isinstance(child, OdfTable):
            if _odf_table_has_content(child):
                return True

    return non_empty_paragraphs > 1 or (cell.value is None and non_empty_paragraphs > 0)


def _odf_cell_child_text(cell: Any) -> str:
    lines: list[str] = []
    for child in cell.children:
        lines.extend(_odf_element_text_lines(child))
    return "\n".join(lines)


def _odf_cell_text(cell: Any) -> str:
    child_text = _odf_cell_child_text(cell)
    if _odf_cell_has_rich_content(cell):
        return _strip_odf_image_reference_text(child_text, cell.get_images())

    if cell.value is not None:
        return str(cell.value)
    if child_text:
        return child_text
    if cell.children:
        return ""

    return "\n".join(_clean_odf_text_lines(cell.text_recursive))


def _odf_cell_has_images(cell: Any) -> bool:
    return len(cell.get_images()) > 0


def _odf_cell_has_content(cell: Any) -> bool:
    return _odf_cell_text(cell) != "" or _odf_cell_has_images(cell)


def _odf_cell_is_rich(cell: Any) -> bool:
    return _odf_cell_has_rich_content(cell)


def _image_ref_from_odf_image(
    odf_obj: OdfDocument | None, image: Any
) -> ImageRef | None:
    image_url = _odf_image_href(image)
    if not _odf_image_can_be_bitmap(image, image_url):
        return None

    image_data: bytes | None = None
    get_data = getattr(image, "get_data", None)
    if callable(get_data):
        image_data = get_data()

    if image_data is None and odf_obj is not None and image_url:
        try:
            image_data = odf_obj.get_part(image_url)
        except Exception:
            image_data = None

    if image_data is None and image_url:
        image_path = Path(image_url)
        if image_path.is_file():
            image_data = image_path.read_bytes()

    if image_data is None:
        return None

    pil_image = PILImage.open(BytesIO(image_data))
    pil_image.load()
    return ImageRef.from_pil(image=pil_image, dpi=72)


def _odf_image_href(image: Any) -> str | None:
    return getattr(image, "url", None) or getattr(image, "attributes", {}).get(
        "xlink:href"
    )


def _odf_image_can_be_bitmap(image: Any, image_url: str | None) -> bool:
    mime_type = getattr(image, "attributes", {}).get("draw:mime-type")
    if mime_type is not None:
        return mime_type.startswith("image/") and mime_type != "image/svg+xml"

    if image_url is None:
        return True

    suffix = Path(image_url).suffix.lower()
    if suffix in {".pdf", ".svg", ".emf", ".wmf"}:
        return False
    return suffix in {
        "",
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }


def _odf_text_is_generated_image_references(text: str, images: list[Any]) -> bool:
    return _strip_odf_image_reference_text(text, images).strip() == ""


def _strip_odf_image_reference_text(text: str, images: list[Any]) -> str:
    remaining = text
    for image in images:
        href = _odf_image_href(image)
        if href is None:
            continue
        href = href.strip()
        refs = {href, href.removeprefix("./")}
        for ref in refs:
            remaining = remaining.replace(f"({ref})", "")

    return remaining


def _add_odf_images(
    doc: DoclingDocument,
    images: list[Any],
    parent: NodeItem,
    content_layer: ContentLayer | None,
    odf_obj: OdfDocument | None,
    *,
    skip_object_replacements: bool = False,
) -> int:
    image_count = 0
    for image in images:
        image_url = _odf_image_href(image)
        if skip_object_replacements and image_url is not None:
            if image_url.removeprefix("./").startswith("ObjectReplacements/"):
                continue
        try:
            image_ref = _image_ref_from_odf_image(odf_obj, image)
        except Exception as e:
            _log.debug("Could not extract OpenDocument image: %s", e)
            image_ref = None
        if image_ref is None:
            continue
        doc.add_picture(parent=parent, image=image_ref, content_layer=content_layer)
        image_count += 1
    return image_count


def _add_odf_child(
    doc: DoclingDocument,
    element: Any,
    *,
    parent: NodeItem | None,
    content_layer: ContentLayer | None,
    odf_obj: OdfDocument | None,
) -> _OdfListState | None:
    if isinstance(element, Header):
        _add_odf_heading(
            doc,
            element,
            parent=parent,
            content_layer=content_layer,
            odf_obj=odf_obj,
        )
    elif isinstance(element, Paragraph) and not isinstance(element, Header):
        _add_odf_paragraph(
            doc,
            element,
            parent=parent,
            content_layer=content_layer,
            odf_obj=odf_obj,
        )
    elif isinstance(element, OdfList):
        return _add_odf_list(
            doc,
            element,
            parent=parent,
            content_layer=content_layer,
            odf_obj=odf_obj,
            enumerated=False,
            flatten_nested_text=False,
        )
    elif isinstance(element, OdfTable):
        _add_table_from_odf(
            doc,
            element,
            parent=parent,
            content_layer=content_layer,
            odf_obj=odf_obj,
        )
    elif isinstance(element, Section):
        _add_odf_children(
            doc,
            element.children,
            parent=parent,
            content_layer=content_layer,
            odf_obj=odf_obj,
        )
    elif isinstance(element, Frame):
        chart_count = _add_odf_charts(doc, element, parent, content_layer, odf_obj)
        _add_odf_images(
            doc,
            element.get_images(),
            parent,
            content_layer,
            odf_obj,
            skip_object_replacements=chart_count > 0,
        )
    else:
        get_images = getattr(element, "get_images", None)
        if callable(get_images):
            _add_odf_images(doc, get_images(), parent, content_layer, odf_obj)
        else:
            _log.debug(
                "Ignoring ODF element with tag: %s", getattr(element, "tag", None)
            )
    return None


def _add_odf_children(
    doc: DoclingDocument,
    elements: list[Any],
    *,
    parent: NodeItem | None,
    content_layer: ContentLayer | None,
    odf_obj: OdfDocument | None,
) -> None:
    previous_list_state: _OdfListState | None = None
    for element in elements:
        if isinstance(element, OdfList):
            previous_list_state = _add_odf_list(
                doc,
                element,
                parent=parent,
                content_layer=content_layer,
                odf_obj=odf_obj,
                enumerated=False,
                continued_state=previous_list_state,
                flatten_nested_text=False,
            )
        else:
            previous_list_state = None
            _add_odf_child(
                doc,
                element,
                parent=parent,
                content_layer=content_layer,
                odf_obj=odf_obj,
            )


def _embedded_odf_content_path(href: str) -> str:
    return f"{href.removeprefix('./').rstrip('/')}/content.xml"


def _odf_chart_classification(chart_content: Any) -> PictureClassificationLabel:
    for chart in chart_content.get_elements("descendant::chart:chart"):
        chart_class = chart.attributes.get("chart:class")
        if chart_class in _ODF_CHART_CLASS_TO_PICTURE_CLASSIFICATION:
            return _ODF_CHART_CLASS_TO_PICTURE_CLASSIFICATION[chart_class]

    for series in chart_content.get_elements("descendant::chart:series"):
        chart_class = series.attributes.get("chart:class")
        if chart_class in _ODF_CHART_CLASS_TO_PICTURE_CLASSIFICATION:
            return _ODF_CHART_CLASS_TO_PICTURE_CLASSIFICATION[chart_class]

    return PictureClassificationLabel.OTHER_CHART


def _chart_data_from_frame(
    frame: Frame, odf_obj: OdfDocument | None
) -> tuple[TableData, PictureClassificationLabel] | None:
    if odf_obj is None:
        return None

    object_href: str | None = None
    for child in frame.children:
        if getattr(child, "tag", None) == "draw:object":
            object_href = child.attributes.get("xlink:href")
            break
    if object_href is None:
        return None

    try:
        chart_content = odf_obj.get_part(_embedded_odf_content_path(object_href))
    except Exception:
        return None
    chart_classification = _odf_chart_classification(chart_content)
    for table in chart_content.get_elements("descendant::table:table"):
        if isinstance(table, OdfTable) and table.name == "local-table":
            table_data = _table_data_from_odf(table)
            if table_data is not None:
                return table_data, chart_classification
    return None


def _add_odf_charts(
    doc: DoclingDocument,
    element: Any,
    parent: NodeItem | None,
    content_layer: ContentLayer | None,
    odf_obj: OdfDocument | None,
) -> int:
    chart_count = 0
    frames = [element] if isinstance(element, Frame) else []
    get_frames = getattr(element, "get_frames", None)
    if callable(get_frames):
        frames.extend(get_frames())
    else:
        frames.extend(
            child
            for child in getattr(element, "children", [])
            if isinstance(child, Frame)
        )
    for frame in frames:
        chart_result = _chart_data_from_frame(frame, odf_obj)
        if chart_result is None:
            continue
        chart_data, chart_classification = chart_result
        chart = doc.add_picture(parent=parent, content_layer=content_layer)
        chart.label = DocItemLabel.PICTURE
        chart.meta = PictureMeta(
            classification=PictureClassificationMetaField(
                predictions=[
                    PictureClassificationPrediction(class_name=chart_classification)
                ]
            ),
            tabular_chart=TabularChartMetaField(chart_data=chart_data),
        )
        chart_count += 1
    return chart_count


def _add_odf_list(
    doc: DoclingDocument,
    odf_list: OdfList,
    parent: NodeItem,
    content_layer: ContentLayer | None,
    odf_obj: OdfDocument | None,
    enumerated: bool = False,
    level: int = 1,
    continued_state: _OdfListState | None = None,
    flatten_nested_text: bool = True,
) -> _OdfListState | None:
    if not _odf_list_has_renderable_content(
        odf_list, flatten_nested_text=flatten_nested_text
    ):
        return None

    style_enumerated = _odf_list_level_is_enumerated(
        odf_obj, odf_list, level, fallback=enumerated
    )
    should_continue = (
        continued_state is not None
        and continued_state.last_item is not None
        and _odf_list_starts_with_empty_nested_item(
            odf_list, flatten_nested_text=flatten_nested_text
        )
    )
    if not should_continue and not _odf_list_has_direct_item_text(
        odf_list, flatten_nested_text=flatten_nested_text
    ):
        for child in odf_list.children:
            if not isinstance(child, ListItem):
                continue
            _text, nested = _odf_list_item_content(
                child, flatten_nested_text=flatten_nested_text
            )
            for nested_list in nested:
                _add_odf_list(
                    doc,
                    nested_list,
                    parent=parent,
                    content_layer=content_layer,
                    odf_obj=odf_obj,
                    enumerated=style_enumerated,
                    level=level + 1,
                    flatten_nested_text=flatten_nested_text,
                )
        return None

    if should_continue and continued_state is not None:
        list_group = continued_state.group
        current_enumerated = continued_state.enumerated
        counter = continued_state.counter
        previous_item = continued_state.last_item
    else:
        list_group = doc.add_list_group(
            name="list", parent=parent, content_layer=content_layer
        )
        current_enumerated = style_enumerated
        counter = _odf_list_start_value(odf_obj, odf_list, level) - 1
        previous_item = None

    for child in odf_list.children:
        if not isinstance(child, ListItem):
            continue
        text, nested = _odf_list_item_content(
            child, flatten_nested_text=flatten_nested_text
        )
        nested = [
            item
            for item in nested
            if _odf_list_has_renderable_content(
                item, flatten_nested_text=flatten_nested_text
            )
        ]
        if not text and not nested:
            continue
        if not text:
            nested_parent = previous_item or list_group
            for nested_list in nested:
                _add_odf_list(
                    doc,
                    nested_list,
                    parent=nested_parent,
                    content_layer=content_layer,
                    odf_obj=odf_obj,
                    enumerated=style_enumerated,
                    level=level + 1,
                    flatten_nested_text=flatten_nested_text,
                )
            continue
        counter += 1
        marker = (
            _odf_list_marker(counter, odf_obj, odf_list, level)
            if current_enumerated
            else ""
        )
        runs = _odf_list_item_text_runs(
            child,
            odf_obj,
            flatten_nested_text=flatten_nested_text,
        )
        if len(runs) <= 1:
            item = doc.add_list_item(
                marker=marker,
                enumerated=current_enumerated,
                parent=list_group,
                text=text,
                content_layer=content_layer,
                formatting=runs[0].formatting if runs else None,
            )
        else:
            item = doc.add_list_item(
                marker=marker,
                enumerated=current_enumerated,
                parent=list_group,
                text="",
                content_layer=content_layer,
            )
            inline_group = doc.add_inline_group(
                parent=item, content_layer=content_layer
            )
            for run in runs:
                doc.add_text(
                    label=DocItemLabel.TEXT,
                    parent=inline_group,
                    text=run.text,
                    content_layer=content_layer,
                    formatting=run.formatting,
                )
        previous_item = item
        for nested_list in nested:
            _add_odf_list(
                doc,
                nested_list,
                parent=item,
                content_layer=content_layer,
                odf_obj=odf_obj,
                enumerated=style_enumerated,
                level=level + 1,
                flatten_nested_text=flatten_nested_text,
            )
    return _OdfListState(
        group=list_group,
        last_item=previous_item,
        enumerated=current_enumerated,
        counter=counter,
    )


def _add_rich_cell_children(
    doc: DoclingDocument,
    cell: Any,
    parent: NodeItem,
    content_layer: ContentLayer | None,
    odf_obj: OdfDocument | None,
) -> None:
    for child in cell.children:
        _add_odf_child(
            doc,
            child,
            parent=parent,
            content_layer=content_layer,
            odf_obj=odf_obj,
        )


def _add_table_from_odf(
    doc: DoclingDocument,
    table: OdfTable,
    parent: NodeItem | None,
    *,
    min_row: int | None = None,
    max_row: int | None = None,
    min_col: int | None = None,
    max_col: int | None = None,
    prov: ProvenanceItem | None = None,
    content_layer: ContentLayer | None = None,
    odf_obj: OdfDocument | None = None,
) -> TableItem | None:
    if min_row is None or max_row is None or min_col is None or max_col is None:
        min_row, max_row, min_col, max_col = _find_true_data_bounds(table)

    height = max_row - min_row + 1
    width = max_col - min_col + 1
    if width == 0 or height == 0:
        return None

    data = TableData(num_rows=height, num_cols=width, table_cells=[])
    table_item = doc.add_table(
        parent=parent,
        data=data,
        prov=prov,
        content_layer=content_layer,
    )

    for row_idx, row in enumerate(table.traverse()):
        if row_idx < min_row or row_idx > max_row:
            continue

        for col_idx, cell in enumerate(row.traverse()):
            if col_idx < min_col or col_idx > max_col:
                continue

            if cell.tag == "table:covered-table-cell":
                continue

            attrs = cell.attributes
            row_span = int(attrs.get("table:number-rows-spanned") or 1)
            col_span = int(attrs.get("table:number-columns-spanned") or 1)
            adjusted_row = row_idx - min_row
            adjusted_col = col_idx - min_col
            text = _odf_cell_text(cell)
            cell_kwargs = {
                "text": text,
                "row_span": row_span,
                "col_span": col_span,
                "start_row_offset_idx": adjusted_row,
                "end_row_offset_idx": adjusted_row + row_span,
                "start_col_offset_idx": adjusted_col,
                "end_col_offset_idx": adjusted_col + col_span,
                "column_header": adjusted_row == 0,
                "row_header": False,
            }

            if _odf_cell_is_rich(cell):
                group = doc.add_group(
                    label=GroupLabel.UNSPECIFIED,
                    name=f"rich_cell_group_{len(doc.tables) - 1}_{adjusted_col}_{adjusted_row}",
                    parent=table_item,
                    content_layer=content_layer,
                )
                _add_rich_cell_children(
                    doc,
                    cell,
                    parent=group,
                    content_layer=content_layer,
                    odf_obj=odf_obj,
                )
                table_cell = RichTableCell(**cell_kwargs, ref=group.get_ref())
            else:
                table_cell = TableCell(**cell_kwargs)

            doc.add_table_cell(table_item=table_item, cell=table_cell)

    return table_item


def _table_region_has_rich_cell(
    table: OdfTable,
    min_row: int,
    max_row: int,
    min_col: int,
    max_col: int,
) -> bool:
    for row_idx, row in enumerate(table.traverse()):
        if row_idx < min_row or row_idx > max_row:
            continue
        for col_idx, cell in enumerate(row.traverse()):
            if col_idx < min_col or col_idx > max_col:
                continue
            if cell.tag != "table:covered-table-cell" and _odf_cell_is_rich(cell):
                return True
    return False


def _table_data_from_odf(
    table: OdfTable,
    min_row: int | None = None,
    max_row: int | None = None,
    min_col: int | None = None,
    max_col: int | None = None,
) -> TableData | None:
    """Convert an ODF table to a :class:`TableData` object.

    This function finds the true data boundaries and only processes cells within
    that region, avoiding the inclusion of large numbers of empty cells that may
    exist beyond the actual data.

    Args:
        table: The ODF table to convert.
        min_row: Optional minimum row index (0-based). If None, will be computed.
        max_row: Optional maximum row index (0-based). If None, will be computed.
        min_col: Optional minimum column index (0-based). If None, will be computed.
        max_col: Optional maximum column index (0-based). If None, will be computed.

    Returns ``None`` when the table has no rows or columns.
    """
    # Find the actual data boundaries if not provided
    if min_row is None or max_row is None or min_col is None or max_col is None:
        min_row, max_row, min_col, max_col = _find_true_data_bounds(table)

    # Calculate the dimensions of the actual data region
    height = max_row - min_row + 1
    width = max_col - min_col + 1

    if width == 0 or height == 0:
        return None

    cells: list[TableCell] = []

    # Only process rows and columns within the data bounds
    for row_idx, row in enumerate(table.traverse()):
        if row_idx < min_row or row_idx > max_row:
            continue

        for col_idx, cell in enumerate(row.traverse()):
            if col_idx < min_col or col_idx > max_col:
                continue

            if cell.tag == "table:covered-table-cell":
                # Spanned-over cells are skipped; the anchoring cell carries the span.
                continue

            attrs = cell.attributes
            row_span = int(attrs.get("table:number-rows-spanned") or 1)
            col_span = int(attrs.get("table:number-columns-spanned") or 1)
            text = _odf_cell_text(cell)

            # Adjust cell coordinates to be relative to the data region
            adjusted_row = row_idx - min_row
            adjusted_col = col_idx - min_col

            cells.append(
                TableCell(
                    text=text,
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=adjusted_row,
                    end_row_offset_idx=adjusted_row + row_span,
                    start_col_offset_idx=adjusted_col,
                    end_col_offset_idx=adjusted_col + col_span,
                    column_header=adjusted_row == 0,
                    row_header=False,
                )
            )

    return TableData(num_rows=height, num_cols=width, table_cells=cells)


class OdtDocumentBackend(_OdfBaseBackend):
    """Backend for OpenDocument Text (``.odt``) files."""

    _odf_type = "text"

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.ODT}

    @override
    def convert(self) -> DoclingDocument:
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/vnd.oasis.opendocument.text",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        if not self.is_valid():
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend failed to init."
            )

        self._walk(self.odf_obj.body.children, parent=None, doc=doc)
        return doc

    def _walk(
        self,
        elements: list[Any],
        parent: NodeItem | None,
        doc: DoclingDocument,
    ) -> None:
        _add_odf_children(
            doc,
            elements,
            parent=parent,
            content_layer=None,
            odf_obj=self.odf_obj,
        )


class OdpDocumentBackend(_OdfBaseBackend, PaginatedDocumentBackend):
    """Backend for OpenDocument Presentation (``.odp``) files."""

    _odf_type = "presentation"

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return True

    @override
    def page_count(self) -> int:
        if not self.is_valid():
            return 0
        return len(self.odf_obj.body.get_draw_pages())

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.ODP}

    @override
    def convert(self) -> DoclingDocument:
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/vnd.oasis.opendocument.presentation",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        if not self.is_valid():
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend failed to init."
            )

        for slide_idx, page in enumerate(self.odf_obj.body.get_draw_pages()):
            slide_name = page.name or f"slide-{slide_idx + 1}"
            slide_group = doc.add_group(
                name=f"slide-{slide_idx}",
                label=GroupLabel.CHAPTER,
                parent=None,
            )
            if not self._slide_has_visible_title(page):
                doc.add_text(
                    label=DocItemLabel.TITLE,
                    parent=slide_group,
                    text=slide_name,
                )
            self._walk_slide(page, parent=slide_group, doc=doc)
        return doc

    def _walk_slide(
        self, page: DrawPage, parent: NodeItem, doc: DoclingDocument
    ) -> None:
        seen_text_content = False
        for element in page.children:
            if getattr(element, "tag", None) in {"anim:par", "presentation:notes"}:
                continue
            has_text = bool(
                _clean_odf_text_lines(getattr(element, "text_recursive", ""))
            )
            is_title = self._is_slide_title_element(element, not seen_text_content)
            if has_text:
                seen_text_content = True
            if isinstance(element, Frame):
                self._walk_slide_frame(
                    element,
                    parent=parent,
                    doc=doc,
                    is_title=is_title,
                )
            else:
                self._walk_textbox_children(
                    element.children,
                    parent=parent,
                    doc=doc,
                    is_title=is_title,
                )

    @staticmethod
    def _slide_has_visible_title(page: DrawPage) -> bool:
        seen_text_content = False
        for element in page.children:
            if getattr(element, "tag", None) in {"anim:par", "presentation:notes"}:
                continue
            if OdpDocumentBackend._is_slide_title_element(
                element, not seen_text_content
            ):
                return True
            if _clean_odf_text_lines(getattr(element, "text_recursive", "")):
                seen_text_content = True
        return False

    @staticmethod
    def _is_slide_title_element(element: Any, is_first_text_content: bool) -> bool:
        attrs = getattr(element, "attributes", {})
        if attrs.get("presentation:class") == "title":
            return True
        return is_first_text_content and getattr(element, "tag", None) == (
            "draw:custom-shape"
        )

    def _walk_slide_frame(
        self,
        frame: Frame,
        parent: NodeItem,
        doc: DoclingDocument,
        *,
        is_title: bool,
    ) -> None:
        chart_count = _add_odf_charts(
            doc,
            frame,
            parent=parent,
            content_layer=None,
            odf_obj=self.odf_obj,
        )

        for tbl in frame.get_elements("descendant::table:table"):
            _add_table_from_odf(
                doc,
                tbl,
                parent=parent,
                odf_obj=self.odf_obj,
            )

        _add_odf_images(
            doc,
            frame.get_images(),
            parent,
            None,
            self.odf_obj,
            skip_object_replacements=chart_count > 0,
        )

        for textbox in frame.get_elements("descendant::draw:text-box"):
            self._walk_textbox_children(
                textbox.children,
                parent=parent,
                doc=doc,
                is_title=is_title,
            )

    def _walk_textbox_children(
        self,
        elements: list[Any],
        parent: NodeItem,
        doc: DoclingDocument,
        *,
        is_title: bool = False,
    ) -> None:
        previous_list_state: _OdfListState | None = None
        for el in elements:
            if isinstance(el, Header):
                previous_list_state = None
                _add_odf_heading(
                    doc,
                    el,
                    parent=parent,
                    content_layer=None,
                    odf_obj=self.odf_obj,
                )
            elif isinstance(el, Paragraph):
                previous_list_state = None
                _add_odf_text_runs(
                    doc,
                    _odf_text_runs(el, self.odf_obj),
                    label=DocItemLabel.TITLE if is_title else DocItemLabel.TEXT,
                    parent=parent,
                    content_layer=None,
                )
            elif isinstance(el, OdfList):
                previous_list_state = _add_odf_list(
                    doc,
                    el,
                    parent=parent,
                    content_layer=None,
                    odf_obj=self.odf_obj,
                    enumerated=False,
                    continued_state=previous_list_state,
                    flatten_nested_text=False,
                )


class OdsDocumentBackend(_OdfBaseBackend, PaginatedDocumentBackend):
    """Backend for OpenDocument Spreadsheet (``.ods``) files.

    Each sheet becomes a separate page. The backend can detect multiple disconnected
    tables within a sheet and optionally treat singleton cells as text items (e.g., titles).
    """

    _odf_type = "spreadsheet"

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: OdsBackendOptions | None = None,
    ) -> None:
        if options is None:
            options = OdsBackendOptions()
        super().__init__(in_doc, path_or_stream, options)

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return True

    @override
    def page_count(self) -> int:
        if not self.is_valid():
            return 0
        sheet_names_filter: list[str] | None = (
            self.options.sheet_names
            if isinstance(self.options, OdsBackendOptions)
            else None
        )
        if sheet_names_filter is None:
            return len(self.odf_obj.body.tables)
        return sum(
            1 for table in self.odf_obj.body.tables if table.name in sheet_names_filter
        )

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.ODS}

    @override
    def convert(self) -> DoclingDocument:
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/vnd.oasis.opendocument.spreadsheet",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        if not self.is_valid():
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend failed to init."
            )

        sheet_names_filter: list[str] | None = (
            self.options.sheet_names
            if isinstance(self.options, OdsBackendOptions)
            else None
        )

        page_no = 0
        for sheet_idx, table in enumerate(self.odf_obj.body.tables):
            if sheet_names_filter is not None and table.name not in sheet_names_filter:
                _log.debug(f"Skipping sheet {sheet_idx}: {table.name} (filtered out)")
                continue

            page_no += 1
            _log.info(f"Processing sheet {sheet_idx}: {table.name} as page {page_no}")

            # Add page for this sheet
            page = doc.add_page(page_no=page_no, size=Size(width=0, height=0))

            # Determine content layer based on sheet visibility
            content_layer = self._get_sheet_content_layer(table)

            sheet_group = doc.add_group(
                parent=None,
                label=GroupLabel.SECTION,
                name=f"sheet: {table.name}",
                content_layer=content_layer,
            )

            # Convert table data with provenance
            self._convert_sheet_table(doc, table, sheet_group, page_no, content_layer)

            # Extract images from the sheet
            self._find_images_in_sheet(doc, table, sheet_group, page_no, content_layer)

            # Calculate and set page size based on content
            width, height = self._find_page_size(doc, page_no)
            page.size = Size(width=width, height=height)

            _log.debug("Processed ODS sheet %s as page %s", table.name, page_no)
        return doc

    def _convert_sheet_table(
        self,
        doc: DoclingDocument,
        table: OdfTable,
        parent: NodeItem,
        page_no: int,
        content_layer: ContentLayer | None,
    ) -> None:
        """Convert an ODS table and add it to the document with provenance.

        This method finds all disconnected data regions in the sheet and creates
        separate tables for each. Singleton cells can optionally be treated as text.
        """
        # Find all data tables in the sheet
        data_tables = self._find_data_tables_in_sheet(table)

        treat_singleton_as_text = (
            isinstance(self.options, OdsBackendOptions)
            and self.options.treat_singleton_as_text
        )

        for data_table in data_tables:
            min_row, max_row, min_col, max_col = data_table["bounds"]
            table_data = data_table["data"]
            has_rich_content = _table_region_has_rich_cell(
                table, min_row, max_row, min_col, max_col
            )

            # Check if this is a singleton (1x1 table)
            if (
                treat_singleton_as_text
                and len(table_data.table_cells) == 1
                and not has_rich_content
            ):
                # Treat as text item instead of table
                cell = table_data.table_cells[0]
                doc.add_text(
                    text=cell.text,
                    label=DocItemLabel.TEXT,
                    parent=parent,
                    prov=ProvenanceItem(
                        page_no=page_no,
                        charspan=(0, 0),
                        bbox=BoundingBox.from_tuple(
                            (min_col, min_row, max_col + 1, max_row + 1),
                            origin=CoordOrigin.TOPLEFT,
                        ),
                    ),
                    content_layer=content_layer,
                )
            else:
                # Add as table with provenance information
                _add_table_from_odf(
                    doc,
                    table,
                    parent,
                    min_row=min_row,
                    max_row=max_row,
                    min_col=min_col,
                    max_col=max_col,
                    prov=ProvenanceItem(
                        page_no=page_no,
                        charspan=(0, 0),
                        bbox=BoundingBox.from_tuple(
                            (min_col, min_row, max_col + 1, max_row + 1),
                            origin=CoordOrigin.TOPLEFT,
                        ),
                    ),
                    content_layer=content_layer,
                    odf_obj=self.odf_obj,
                )

    def _find_data_tables_in_sheet(
        self, table: OdfTable
    ) -> list[dict[str, tuple[int, int, int, int] | TableData]]:
        """Find all disconnected data tables in an ODS sheet using flood-fill.

        Returns a list of dictionaries, each containing:
        - 'bounds': (min_row, max_row, min_col, max_col)
        - 'data': TableData object
        """
        import collections

        # Get the overall data bounds
        overall_min_row, overall_max_row, overall_min_col, overall_max_col = (
            _find_true_data_bounds(table)
        )

        # Check if we found any data
        if (
            overall_min_row == 0
            and overall_max_row == 0
            and overall_min_col == 0
            and overall_max_col == 0
        ):
            first_cell = table.get_cell("A1")
            if not _odf_cell_has_content(first_cell):
                return []

        GAP_TOLERANCE = cast(OdsBackendOptions, self.options).gap_tolerance
        tables: list[dict[str, tuple[int, int, int, int] | TableData]] = []
        visited: set[tuple[int, int]] = set()

        # Build a map of cell contents for quick lookup
        cell_map: dict[tuple[int, int], bool] = {}
        for row_idx, row in enumerate(table.traverse()):
            for col_idx, cell in enumerate(row.traverse()):
                has_data = _odf_cell_has_content(cell) or (
                    cell.tag == "table:covered-table-cell"
                )
                cell_map[(row_idx, col_idx)] = has_data

        # Helper: Check if a cell has content
        def has_content(r: int, c: int) -> bool:
            if (
                r < overall_min_row
                or r > overall_max_row
                or c < overall_min_col
                or c > overall_max_col
            ):
                return False
            return cell_map.get((r, c), False)

        # Scan for table starts
        for ri in range(overall_min_row, overall_max_row + 1):
            for ci in range(overall_min_col, overall_max_col + 1):
                if (ri, ci) in visited:
                    continue

                if not has_content(ri, ci):
                    continue

                # Found a new table start - use flood fill to find its bounds
                table_cells: set[tuple[int, int]] = set()
                queue = collections.deque([(ri, ci)])
                table_cells.add((ri, ci))

                min_r, max_r = ri, ri
                min_c, max_c = ci, ci

                # Phase 1: Flood Fill
                while queue:
                    curr_r, curr_c = queue.popleft()

                    # Update bounds
                    min_r = min(min_r, curr_r)
                    max_r = max(max_r, curr_r)
                    min_c = min(min_c, curr_c)
                    max_c = max(max_c, curr_c)

                    # Check neighbors in 4 directions with gap tolerance
                    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

                    for dr, dc in directions:
                        for step in range(1, GAP_TOLERANCE + 2):
                            nr, nc = curr_r + (dr * step), curr_c + (dc * step)

                            if (nr, nc) in table_cells:
                                break

                            if has_content(nr, nc):
                                table_cells.add((nr, nc))
                                queue.append((nr, nc))
                                break

                # Mark all cells in this table as visited
                visited.update(table_cells)

                # Phase 2: Extract data for this table region
                # Create a sub-table with just this region
                data = _table_data_from_odf(
                    table, min_row=min_r, max_row=max_r, min_col=min_c, max_col=max_c
                )

                if data is not None:
                    tables.append(
                        {"bounds": (min_r, max_r, min_c, max_c), "data": data}
                    )

        return tables

    def _find_images_in_sheet(
        self,
        doc: DoclingDocument,
        table: OdfTable,
        parent: NodeItem,
        page_no: int,
        content_layer: ContentLayer | None,
    ) -> None:
        """Find and extract images from an ODS sheet."""
        try:
            # Get all images in the table
            images = table.get_images()
            for img in images:
                try:
                    # Get the image data
                    image_data = img.get_data()
                    if image_data:
                        # Convert to PIL Image
                        from io import BytesIO

                        pil_image = PILImage.open(BytesIO(image_data))

                        # Try to get position information
                        # ODF images are typically anchored to cells
                        # For now, use a default position
                        anchor = (0, 0, 1, 1)

                        doc.add_picture(
                            parent=parent,
                            image=ImageRef.from_pil(image=pil_image, dpi=72),
                            caption=None,
                            prov=ProvenanceItem(
                                page_no=page_no,
                                charspan=(0, 0),
                                bbox=BoundingBox.from_tuple(
                                    anchor, origin=CoordOrigin.TOPLEFT
                                ),
                            ),
                            content_layer=content_layer,
                        )
                except Exception as e:
                    _log.debug(f"Could not extract image from ODS sheet: {e}")
        except Exception as e:
            _log.debug(f"Could not find images in ODS sheet: {e}")

    @staticmethod
    def _find_page_size(doc: DoclingDocument, page_no: int) -> tuple[float, float]:
        """Calculate page size based on the bounding boxes of all items on the page."""
        left: float = -1.0
        top: float = -1.0
        right: float = -1.0
        bottom: float = -1.0

        for item, _ in doc.iterate_items(traverse_pictures=True, page_no=page_no):
            if not isinstance(item, DocItem):
                continue
            for provenance in item.prov:
                if provenance.bbox is None:
                    continue
                bbox = provenance.bbox
                left = min(left, bbox.l) if left != -1 else bbox.l
                right = max(right, bbox.r) if right != -1 else bbox.r
                top = min(top, bbox.t) if top != -1 else bbox.t
                bottom = max(bottom, bbox.b) if bottom != -1 else bbox.b

        # Return dimensions, defaulting to (0, 0) if no items found
        if left == -1 or right == -1:
            return (0.0, 0.0)
        return (right - left, bottom - top)

    @staticmethod
    def _get_sheet_content_layer(table: OdfTable) -> ContentLayer | None:
        """Determine if a sheet is hidden and should be marked as invisible."""
        # Check if the table has a display attribute indicating it's hidden
        # ODF uses table:display="false" for hidden sheets
        display = table.get_attribute("table:display")
        if display == "false":
            return ContentLayer.INVISIBLE
        return None
