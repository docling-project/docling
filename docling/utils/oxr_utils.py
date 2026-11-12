# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Convert SoMark OXR parse results into ``DoclingDocument`` objects.

``parse_oxr_result`` consumes a saved OXR parse response. It does not run OXR
or import its SDK. ``run_oxr_parse`` only posts one document to an OXR service
and returns the JSON result that ``parse_oxr_result`` accepts.

OXR returns one JSON entry for each detected region, already in reading order,
with pixel coordinates on the rendered page canvas. The adapter keeps those
coordinates and their top-left origin; it does not rescale them to PDF points.
A region that crosses the canvas edge is clamped to the canvas.

Limits:

- OXR reports no heading depth, so section levels are normalized per document
  instead of recovered.
- A picture entry names a file on the OXR host. An image is attached only when
  ``load_local_images`` is set and the file exists.
- ``Chemical Structure`` and ``Stamp`` have no dedicated Docling label. They
  map to a formula and to a picture (or text) respectively.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import requests
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from docling.utils.chandra_utils import _add_html_table
from docling.utils.vlm_utils import (
    normalize_section_header_levels,
    strip_math_delimiters,
)

_log = logging.getLogger(__name__)

__all__ = ["parse_oxr_result", "run_oxr_parse"]

_OXR_PARSE_ENDPOINT: Final[str] = "/parse/sync"
_DEFAULT_FILENAME: Final[str] = "oxr_result.json"
_PICTURE_DPI: Final[int] = 72

_TEXT_LABELS: Final[dict[str, DocItemLabel]] = {
    "Text": DocItemLabel.TEXT,
    "Caption": DocItemLabel.CAPTION,
    "Footnote": DocItemLabel.FOOTNOTE,
    "Page-header": DocItemLabel.PAGE_HEADER,
    "Page-footer": DocItemLabel.PAGE_FOOTER,
}
_FORMULA_LABELS: Final[frozenset[str]] = frozenset({"Formula", "Chemical Equation"})
_IMAGE_URL_PREFIXES: Final[tuple[str, ...]] = ("http://", "https://", "data:")

_MARKDOWN_IMAGE = re.compile(r"^\s*!\[[^\]]*\]\(\s*(?P<target>[^)\s]+)\s*\)\s*$")
_CODE_FENCE = re.compile(
    r"^(?P<fence>`{3,}|~{3,})[^\n]*\n(?P<body>.*?)\n?(?P=fence)\s*$", re.DOTALL
)
_TABLE_SEPARATOR = re.compile(r":?-+:?")


class _OxrPageSize(BaseModel):
    """Rendered page canvas reported by OXR, in pixels."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    w: float = Field(gt=0)
    h: float = Field(gt=0)


class _OxrBlock(BaseModel):
    """Validated subset of one serialized OXR region.

    ``idx`` is not modeled: OXR keeps reading order in the list, and cleanup can
    leave gaps in the index.
    """

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    type: str
    bbox: tuple[float, float, float, float]
    content: str = ""
    format: str = "text"

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("type must not be empty")
        return value

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if not all(math.isfinite(coord) for coord in value):
            raise ValueError("bbox coordinates must be finite")
        return value


class _OxrPage(BaseModel):
    """Validated subset of one serialized OXR page."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    page_num: int = Field(ge=0)
    page_size: _OxrPageSize
    blocks: list[_OxrBlock] = Field(default_factory=list)


@dataclass
class _BuildState:
    load_local_images: bool
    base_dir: Path | None
    title_added: bool = False


def _as_mapping(content: Mapping[str, Any] | str | bytes) -> Mapping[str, Any]:
    if isinstance(content, (str, bytes)):
        try:
            payload: Any = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid OXR JSON") from exc
    elif isinstance(content, Mapping):
        payload = content
    else:
        raise TypeError(
            "content must be a JSON object mapping, JSON string, or JSON bytes"
        )
    if not isinstance(payload, Mapping):
        raise ValueError("OXR JSON must contain an object")
    return payload


def _log_reported_warnings(payload: Mapping[str, Any]) -> None:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        return
    for warning in warnings:
        _log.warning("OXR reported: %s", warning)


def _unwrap_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the parse result of a sync response, async query, or saved result."""
    _log_reported_warnings(payload)
    code = payload.get("code")
    if isinstance(code, int) and code != 0:
        message = payload.get("message") or "unknown error"
        raise ValueError(f"OXR service returned code {code}: {message}")

    data = payload.get("data")
    if not isinstance(data, Mapping):
        return payload

    _log_reported_warnings(data)
    result = data.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("OXR JSON has no parse result; the task may still be running")
    return result


def _extract_pages(result: Mapping[str, Any]) -> list[_OxrPage]:
    outputs = result.get("outputs")
    if isinstance(outputs, Mapping):
        json_output = outputs.get("json")
        if not isinstance(json_output, Mapping):
            raise ValueError(
                "OXR result has no 'json' output; request output_formats=json"
            )
        raw_pages = json_output.get("pages")
    else:
        raw_pages = result.get("pages")

    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("OXR JSON contains no pages")

    pages = [_OxrPage.model_validate(page) for page in raw_pages]
    page_numbers = [page.page_num for page in pages]
    if len(set(page_numbers)) != len(page_numbers):
        raise ValueError("OXR page_num values must be unique")
    return sorted(pages, key=lambda page: page.page_num)


def _resolve_filename(result: Mapping[str, Any], filename: str | None) -> str:
    if filename:
        return filename
    reported = result.get("file_name")
    if isinstance(reported, str) and reported.strip():
        # OXR services can run on another operating system; normalize both
        # separators before deriving the source filename.
        return Path(reported.replace("\\", "/")).name
    return _DEFAULT_FILENAME


def _make_provenance(
    block: _OxrBlock, *, page: _OxrPage, page_no: int
) -> ProvenanceItem:
    width = page.page_size.w
    height = page.page_size.h
    left, top, right, bottom = block.bbox
    clamped = (
        min(max(left, 0.0), width),
        min(max(top, 0.0), height),
        min(max(right, 0.0), width),
        min(max(bottom, 0.0), height),
    )
    if clamped != (left, top, right, bottom):
        _log.debug(
            "Clamped OXR bbox %s to the page canvas %sx%s", block.bbox, width, height
        )

    clamped_left, clamped_top, clamped_right, clamped_bottom = clamped
    if clamped_right <= clamped_left or clamped_bottom <= clamped_top:
        raise ValueError(
            f"bbox {block.bbox} does not intersect the page canvas {width}x{height}"
        )
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(
            l=clamped_left,
            t=clamped_top,
            r=clamped_right,
            b=clamped_bottom,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        charspan=(0, 0),
    )


def _image_reference(content: str) -> str | None:
    """Return the target of a Markdown image, or ``None`` for other content."""
    match = _MARKDOWN_IMAGE.match(content)
    if match is None:
        return None
    return match.group("target")


def _is_local_reference(target: str) -> bool:
    return not target.startswith(_IMAGE_URL_PREFIXES)


def _load_picture(target: str, state: _BuildState) -> ImageRef | None:
    if not state.load_local_images:
        return None
    path = Path(target)
    if not path.is_absolute() and state.base_dir is not None:
        path = state.base_dir / path
    if not path.is_file():
        _log.warning("OXR picture file not found: %s", path)
        return None
    try:
        with Image.open(path) as image:
            return ImageRef.from_pil(image.convert("RGB"), dpi=_PICTURE_DPI)
    except OSError as exc:
        _log.warning("OXR picture file could not be read: %s (%s)", path, exc)
        return None


def _add_title(
    doc: DoclingDocument, text: str, prov: ProvenanceItem, state: _BuildState
) -> None:
    if not state.title_added:
        doc.add_title(text=text, prov=prov)
        state.title_added = True
        return
    doc.add_heading(text=text, level=1, prov=prov)


def _add_formula(doc: DoclingDocument, text: str, prov: ProvenanceItem) -> bool:
    latex = strip_math_delimiters(text.strip())
    if not latex:
        return False
    doc.add_formula(text=latex, prov=prov)
    return True


def _code_text(markdown: str) -> str:
    match = _CODE_FENCE.match(markdown.strip())
    if match is None:
        return markdown.strip()
    return match.group("body").strip()


def _markdown_table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _markdown_table_data(markdown: str) -> TableData | None:
    """Parse a pipe table without the optional Markdown dependency."""
    lines = [
        line.strip()
        for line in markdown.strip().splitlines()
        if line.strip().startswith("|")
    ]
    if len(lines) < 2:
        return None

    header_seen = False
    rows: list[list[str]] = []
    for line in lines:
        cells = _markdown_table_cells(line.replace("\\|", "\x00"))
        if all(_TABLE_SEPARATOR.fullmatch(cell) for cell in cells):
            header_seen = True
            continue
        rows.append([cell.replace("\x00", "|") for cell in cells])
    if not rows:
        return None

    num_cols = max(len(row) for row in rows)
    table_cells: list[TableCell] = []
    for row_idx, row in enumerate(rows):
        for col_idx in range(num_cols):
            table_cells.append(
                TableCell(
                    text=row[col_idx] if col_idx < len(row) else "",
                    row_span=1,
                    col_span=1,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + 1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    column_header=header_seen and row_idx == 0,
                    row_header=False,
                )
            )
    return TableData(num_rows=len(rows), num_cols=num_cols, table_cells=table_cells)


def _add_markdown_table(
    doc: DoclingDocument, content: str, prov: ProvenanceItem
) -> bool:
    data = _markdown_table_data(content)
    if data is None:
        return False
    doc.add_table(data=data, prov=prov)
    return True


def _add_table(doc: DoclingDocument, block: _OxrBlock, prov: ProvenanceItem) -> bool:
    content = block.content.strip()
    if not content:
        return False
    if block.format == "html" or "<table" in content.casefold():
        return _add_html_table(doc, content, prov) is not None
    if block.format == "markdown" or content.startswith("|"):
        return _add_markdown_table(doc, content, prov)
    return False


def _add_text(doc: DoclingDocument, text: str, prov: ProvenanceItem) -> None:
    text = text.strip()
    if text:
        doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)


def _add_picture(
    doc: DoclingDocument, block: _OxrBlock, prov: ProvenanceItem, state: _BuildState
) -> None:
    target = _image_reference(block.content)
    image = None
    if target is not None and _is_local_reference(target):
        image = _load_picture(target, state)
    elif target is not None:
        # A remote reference cannot be loaded into the document without a size,
        # so the picture keeps its geometry only.
        _log.debug("OXR picture reference is remote and is not loaded: %s", target)
    doc.add_picture(image=image, prov=prov)
    if target is None:
        # Content that is not an image reference can be a description or OCR
        # text. Keep it as adjacent text instead of dropping it.
        _add_text(doc, block.content, prov)


def _add_block(
    doc: DoclingDocument,
    block: _OxrBlock,
    *,
    page: _OxrPage,
    page_no: int,
    state: _BuildState,
) -> None:
    # A picture is a visual region even without recognized content. Every other
    # type needs content to produce an item.
    if block.type != "Picture" and not block.content.strip():
        return

    prov = _make_provenance(block, page=page, page_no=page_no)
    kind = block.type

    if kind == "Title":
        _add_title(doc, block.content.strip(), prov, state)
        return
    if kind == "Table":
        if not _add_table(doc, block, prov):
            _log.warning("OXR table could not be parsed; keeping it as text")
            _add_text(doc, block.content, prov)
        return
    if kind in _FORMULA_LABELS:
        if not _add_formula(doc, block.content, prov):
            _add_text(doc, block.content, prov)
        return
    if kind == "Code Block":
        doc.add_code(text=_code_text(block.content), prov=prov)
        return
    if kind == "Picture":
        _add_picture(doc, block, prov, state)
        return
    if kind == "Stamp":
        # A stamp can be an image or plain text. Only the image form is a
        # picture; the text form would otherwise gain an empty picture item.
        if _image_reference(block.content) is not None:
            _add_picture(doc, block, prov, state)
        else:
            _add_text(doc, block.content, prov)
        return
    if kind == "Chemical Structure":
        if _image_reference(block.content) is not None:
            _add_picture(doc, block, prov, state)
        elif not _add_formula(doc, block.content, prov):
            _add_text(doc, block.content, prov)
        return

    doc_label = _TEXT_LABELS.get(kind)
    if doc_label is None:
        _log.warning("Unsupported OXR block type %r; mapping it to text", kind)
        doc_label = DocItemLabel.TEXT
    doc.add_text(label=doc_label, text=block.content.strip(), prov=prov)


def parse_oxr_result(
    content: Mapping[str, Any] | str | bytes,
    *,
    filename: str | None = None,
    page_no: int | None = None,
    load_local_images: bool = False,
    base_dir: Path | None = None,
) -> DoclingDocument:
    """Convert one saved OXR parse result into a ``DoclingDocument``.

    Args:
        content: A full sync response, an async query response, the bare
            ``result`` object, or a bare ``{"pages": [...]}`` payload, given as
            a mapping, a JSON string, or JSON bytes.
        filename: Optional source filename override. Otherwise the OXR
            ``file_name`` is used.
        page_no: Optional 1-based page number override. OXR page numbers are
            0-based, so the default is ``page_num + 1``. It is only allowed for
            a single-page payload.
        load_local_images: Attach picture images only when this is set. OXR
            picture entries name files on the OXR host, which a converting host
            must opt in to read.
        base_dir: Directory for relative picture paths.

    Returns:
        A ``DoclingDocument`` with typed items, page furniture labels, and
        provenance in the OXR page canvas coordinate space.

    Raises:
        TypeError: If ``content`` is not a supported input type.
        ValueError: If the payload, page scope, geometry, or service envelope is
            invalid.
        pydantic.ValidationError: If a page or region does not match the OXR
            schema.
    """
    result = _unwrap_result(_as_mapping(content))
    pages = _extract_pages(result)
    if page_no is not None and page_no < 1:
        raise ValueError("page_no is 1-based and must be greater than zero")
    if page_no is not None and len(pages) > 1:
        raise ValueError("page_no can only override a single-page OXR payload")

    resolved_filename = _resolve_filename(result, filename)
    origin = DocumentOrigin(
        filename=resolved_filename,
        mimetype="application/json",
        binary_hash=0,
    )
    doc = DoclingDocument(name=Path(resolved_filename).stem, origin=origin)
    state = _BuildState(load_local_images=load_local_images, base_dir=base_dir)

    for page in pages:
        resolved_page_no = page_no if page_no is not None else page.page_num + 1
        doc.add_page(
            page_no=resolved_page_no,
            size=Size(width=page.page_size.w, height=page.page_size.h),
        )
        for block in page.blocks:
            _add_block(doc, block, page=page, page_no=resolved_page_no, state=state)

    # OXR has no heading depth, so levels are normalized per document instead
    # of recovered from the source.
    normalize_section_header_levels(doc)
    return doc


def run_oxr_parse(
    source: Path,
    *,
    base_url: str,
    api_key: str = "dummy",
    output_formats: Sequence[str] = ("json",),
    keep_header_footer: bool = False,
    timeout: float = 600.0,
    session: requests.Session | None = None,
) -> Mapping[str, Any]:
    """Send one document to an OXR parse service and return its parse result.

    Args:
        source: Local PDF or image file to parse.
        base_url: OXR service base URL, such as ``http://localhost:8000/v1``.
        api_key: Value for the OXR ``api_key`` form field. Self-hosted OXR
            accepts any value.
        output_formats: OXR output formats to request. ``json`` is required by
            :func:`parse_oxr_result`.
        keep_header_footer: Ask OXR to keep page headers and footers.
        timeout: Request timeout in seconds.
        session: Optional ``requests`` session, for reuse or testing.

    Returns:
        The ``result`` mapping of the OXR response.

    Raises:
        requests.HTTPError: If the service answers with an error status.
        ValueError: If the response envelope is invalid.
    """
    endpoint = base_url.rstrip("/") + _OXR_PARSE_ENDPOINT
    data: dict[str, Any] = {
        "api_key": api_key,
        "output_formats": list(output_formats),
    }
    if keep_header_footer:
        data["feature_config"] = json.dumps({"keep_header_footer": True})

    http = requests.Session() if session is None else session
    try:
        with source.open("rb") as handle:
            response = http.post(
                endpoint,
                data=data,
                files={"file": (source.name, handle, "application/octet-stream")},
                timeout=timeout,
            )
    finally:
        if session is None:
            http.close()

    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError("OXR service returned a non-object JSON payload")
    return _unwrap_result(payload)
