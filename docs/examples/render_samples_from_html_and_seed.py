import argparse
import logging
import math
import multiprocessing as mp
import os
import random
import re
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from pydantic import BaseModel as PydanticBaseModel
from tqdm import tqdm

from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import ImageRef, TableItem
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption

"""
uv run python docs/examples/render_samples_from_html_and_seed.py \
  --input-html-path <html_root> \
  --seed-image-path <seed_png_root> \
  --out-dir <output_root>
"""

_log = logging.getLogger(__name__)
PARTITION_SIZE = 1_000_000
TABLE_OVERFLOW_MAX_WIDTH_MULTIPLIER = 3.0
TABLE_OVERFLOW_MAX_ATTEMPTS = 5
TABLE_OVERFLOW_MIN_WIDTH_STEP = 64
TABLE_EXTRACT_MAX_PAD_RATIO = 0.08
TABLE_EXTRACT_MAX_PAD_PX = 48
TABLE_EXTRACT_MIN_PAD_PX = 2
TABLE_EXTRACT_TABLE_SAFETY_PX = 1
HEAD_RE = re.compile(rb"(<head\b[^>]*>)(.*?)(</head\s*>)", re.IGNORECASE | re.DOTALL)
TITLE_RE = re.compile(rb"<title\b[^>]*>.*?</title\s*>", re.IGNORECASE | re.DOTALL)
_WORKER_CONVERTERS: dict[tuple[int, int, str], DocumentConverter] = {}

_TABLE_CELL_OVERFLOW_CHECK_JS = """
() => {
  const EPS = 0.75;
  const cells = Array.from(document.querySelectorAll("table td, table th"));
  const tables = Array.from(document.querySelectorAll("table"));
  let overflowCount = 0;
  let tableLeft = null;
  let tableTop = null;
  let tableRight = null;
  let tableBottom = null;

  const isVisibleRect = (rect) => rect && (rect.width > 0 || rect.height > 0);
  const isInside = (inner, outer) => (
    inner.left >= outer.left - EPS &&
    inner.right <= outer.right + EPS &&
    inner.top >= outer.top - EPS &&
    inner.bottom <= outer.bottom + EPS
  );

  for (const cell of cells) {
    const cellRect = cell.getBoundingClientRect();
    if (!isVisibleRect(cellRect)) {
      continue;
    }
    const cellStyle = window.getComputedStyle(cell);
    if (cellStyle.display === "none" || cellStyle.visibility === "hidden") {
      continue;
    }

    let overflow = (
      (cell.scrollWidth || 0) > (cell.clientWidth || 0) + 1 ||
      (cell.scrollHeight || 0) > (cell.clientHeight || 0) + 1
    );

    if (!overflow) {
      const descendants = Array.from(cell.querySelectorAll("*"));
      for (const node of descendants) {
        const nodeStyle = window.getComputedStyle(node);
        if (nodeStyle.display === "none" || nodeStyle.visibility === "hidden") {
          continue;
        }
        const rect = node.getBoundingClientRect();
        if (!isVisibleRect(rect)) {
          continue;
        }
        if (!isInside(rect, cellRect)) {
          overflow = true;
          break;
        }
      }
    }

    if (!overflow) {
      const walker = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const textNode = walker.currentNode;
        if (!textNode || !textNode.textContent || !textNode.textContent.trim()) {
          continue;
        }
        const range = document.createRange();
        range.selectNodeContents(textNode);
        const rects = Array.from(range.getClientRects());
        for (const rect of rects) {
          if (!isVisibleRect(rect)) {
            continue;
          }
          if (!isInside(rect, cellRect)) {
            overflow = true;
            break;
          }
        }
        if (overflow) {
          break;
        }
      }
    }

    if (overflow) {
      overflowCount += 1;
    }
  }

  for (const table of tables) {
    const tableStyle = window.getComputedStyle(table);
    if (tableStyle.display === "none" || tableStyle.visibility === "hidden") {
      continue;
    }
    const rect = table.getBoundingClientRect();
    if (!isVisibleRect(rect)) {
      continue;
    }
    const l = rect.left + window.scrollX;
    const t = rect.top + window.scrollY;
    const r = l + rect.width;
    const b = t + rect.height;
    tableLeft = tableLeft === null ? l : Math.min(tableLeft, l);
    tableTop = tableTop === null ? t : Math.min(tableTop, t);
    tableRight = tableRight === null ? r : Math.max(tableRight, r);
    tableBottom = tableBottom === null ? b : Math.max(tableBottom, b);
  }

  const tableBounds = (
    tableLeft === null ||
    tableTop === null ||
    tableRight === null ||
    tableBottom === null
  )
    ? null
    : { left: tableLeft, top: tableTop, right: tableRight, bottom: tableBottom };

  return {
    tableCellCount: cells.length,
    overflowCellCount: overflowCount,
    hasOverflow: overflowCount > 0,
    tableBounds,
  };
}
"""
TABLE_TAG_RE = re.compile(rb"<table\b", re.IGNORECASE)


def _iter_html_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        dir_path = Path(dirpath)
        for filename in filenames:
            if filename.lower().endswith(".html"):
                yield dir_path / filename


def _iter_png_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        dir_path = Path(dirpath)
        for filename in filenames:
            if filename.lower().endswith(".png"):
                yield dir_path / filename


def _extract_seed_key_from_html_name(stem: str) -> Optional[str]:
    seed_pos = stem.find("seed_")
    if seed_pos < 0:
        return None
    start = seed_pos + len("seed_")
    if start >= len(stem):
        return None

    tail = stem[start:]
    delim_pos = tail.find("__")
    if delim_pos >= 0:
        tail = tail[:delim_pos]

    return tail if tail else None


def _build_seed_index(seed_image_path: Path) -> tuple[dict[str, list[Path]], list[tuple[str, Path]]]:
    by_stem: dict[str, list[Path]] = {}
    entries: list[tuple[str, Path]] = []
    for png_path in _iter_png_files(seed_image_path):
        stem = png_path.stem
        by_stem.setdefault(stem, []).append(png_path)
        entries.append((stem, png_path))

    for stem_paths in by_stem.values():
        stem_paths.sort(key=lambda p: str(p))
    entries.sort(key=lambda item: (item[0], str(item[1])))
    return by_stem, entries


def _find_seed_png(
    seed_key: str,
    by_stem: dict[str, list[Path]],
    entries: list[tuple[str, Path]],
) -> Optional[Path]:
    exact = by_stem.get(seed_key)
    if exact:
        return exact[0]

    partial_candidates = [
        png_path for stem, png_path in entries if seed_key in stem or stem in seed_key
    ]
    if not partial_candidates:
        return None
    partial_candidates.sort(key=lambda p: (abs(len(p.stem) - len(seed_key)), str(p)))
    return partial_candidates[0]


def _build_html_options(
    sample_source_uri: Path, render_page_width: int, render_page_height: int
) -> HTMLBackendOptions:
    return HTMLBackendOptions(
        render_page=True,
        render_page_width=max(1, int(render_page_width)),
        render_page_height=max(1, int(render_page_height)),
        render_device_scale=1.0,
        render_page_orientation="portrait",
        render_print_media=True,
        render_wait_until="networkidle",
        render_wait_ms=500,
        render_full_page=True,
        render_dpi=96,
        page_padding=0,
        enable_local_fetch=True,
        fetch_images=True,
        source_uri=sample_source_uri.resolve(),
    )


def _strip_title_from_head(html_bytes: bytes) -> tuple[bytes, bool]:
    head_match = HEAD_RE.search(html_bytes)
    if head_match is None:
        return html_bytes, False

    head_start, head_end = head_match.span()
    head_open, head_content, head_close = head_match.groups()
    sanitized_head_content, removed_count = TITLE_RE.subn(b"", head_content)
    if removed_count == 0:
        return html_bytes, False

    sanitized_head = b"".join((head_open, sanitized_head_content, head_close))
    sanitized_html = b"".join(
        (html_bytes[:head_start], sanitized_head, html_bytes[head_end:])
    )
    return sanitized_html, True


def _prepare_html_without_head_title(input_path: Path) -> tuple[Path, Optional[Path]]:
    original_html = input_path.read_bytes()
    sanitized_html, removed = _strip_title_from_head(original_html)
    if not removed:
        return input_path, None

    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".html",
        prefix=f".{input_path.stem}.notitle.",
        dir=input_path.parent,
        delete=False,
    ) as temp_html:
        temp_html.write(sanitized_html)
        temp_path = Path(temp_html.name)
        return temp_path, temp_path


def _html_has_table(html_path: Path) -> bool:
    try:
        return TABLE_TAG_RE.search(html_path.read_bytes()) is not None
    except Exception:
        return True


def _detect_table_cell_overflow_with_browser(
    browser: Any, html_path: Path, render_page_width: int, render_page_height: int
) -> dict[str, Any]:
    context = browser.new_context(
        viewport={
            "width": max(1, int(render_page_width)),
            "height": max(1, int(render_page_height)),
        },
        java_script_enabled=False,
        offline=True,
        service_workers="block",
    )
    try:
        page = context.new_page()
        page.emulate_media(media="print")
        page.goto(html_path.resolve().as_uri(), wait_until="load")
        data = page.evaluate(_TABLE_CELL_OVERFLOW_CHECK_JS)
    finally:
        context.close()

    return {
        "has_overflow": bool(data.get("hasOverflow", False)),
        "table_cell_count": int(data.get("tableCellCount", 0)),
        "overflow_cell_count": int(data.get("overflowCellCount", 0)),
        "table_bounds": data.get("tableBounds"),
    }


def _find_autofit_render_width_for_table_cells(
    html_path: Path, render_page_width: int, render_page_height: int
) -> dict[str, Any]:
    if not _html_has_table(html_path):
        return {
            "render_page_width": max(1, int(render_page_width)),
            "attempts": 0,
            "auto_fit_applied": False,
            "table_cell_count": 0,
            "overflow_cell_count": 0,
            "table_bounds": None,
            "has_overflow": False,
        }

    initial_width = max(1, int(render_page_width))
    current_width = initial_width
    max_width = max(
        initial_width + TABLE_OVERFLOW_MIN_WIDTH_STEP,
        int(round(initial_width * TABLE_OVERFLOW_MAX_WIDTH_MULTIPLIER)),
    )

    attempts = 0
    auto_fit_applied = False
    last_result: dict[str, Any] = {
        "has_overflow": False,
        "table_cell_count": 0,
        "overflow_cell_count": 0,
    }

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for table-cell overflow auto-fit. "
            "Install it with 'pip install \"docling[htmlrender]\"' and run "
            "'playwright install'."
        ) from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            while True:
                attempts += 1
                last_result = _detect_table_cell_overflow_with_browser(
                    browser=browser,
                    html_path=html_path,
                    render_page_width=current_width,
                    render_page_height=render_page_height,
                )
                if not last_result["has_overflow"]:
                    return {
                        "render_page_width": current_width,
                        "attempts": attempts,
                        "auto_fit_applied": auto_fit_applied,
                        "table_cell_count": last_result["table_cell_count"],
                        "overflow_cell_count": last_result["overflow_cell_count"],
                        "table_bounds": last_result.get("table_bounds"),
                        "has_overflow": False,
                    }

                if attempts >= TABLE_OVERFLOW_MAX_ATTEMPTS or current_width >= max_width:
                    return {
                        "render_page_width": None,
                        "attempts": attempts,
                        "auto_fit_applied": auto_fit_applied
                        or current_width != initial_width,
                        "table_cell_count": last_result["table_cell_count"],
                        "overflow_cell_count": last_result["overflow_cell_count"],
                        "table_bounds": last_result.get("table_bounds"),
                        "has_overflow": True,
                    }

                next_width = max(
                    current_width + TABLE_OVERFLOW_MIN_WIDTH_STEP,
                    int(round(current_width * 1.2)),
                )
                next_width = min(max_width, next_width)
                if next_width <= current_width:
                    return {
                        "render_page_width": None,
                        "attempts": attempts,
                        "auto_fit_applied": auto_fit_applied
                        or current_width != initial_width,
                        "table_cell_count": last_result["table_cell_count"],
                        "overflow_cell_count": last_result["overflow_cell_count"],
                        "table_bounds": last_result.get("table_bounds"),
                        "has_overflow": True,
                    }
                current_width = next_width
                auto_fit_applied = True
        finally:
            browser.close()


def _average_size(sizes: list[tuple[int, int]]) -> Optional[tuple[int, int]]:
    if not sizes:
        return None
    avg_width = max(1, int(round(sum(width for width, _ in sizes) / len(sizes))))
    avg_height = max(1, int(round(sum(height for _, height in sizes) / len(sizes))))
    return avg_width, avg_height


def _write_unresolved_seed_report(
    report_path: Path, input_html_path: Path, unresolved_items: list[dict[str, Any]]
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        report_file.write("# seed_key_crop\treason\thtml_path\n")
        for item in sorted(unresolved_items, key=lambda row: str(row["input_path"])):
            input_path = Path(item["input_path"])
            try:
                html_path = input_path.relative_to(input_html_path)
            except ValueError:
                html_path = input_path
            seed_key = item.get("seed_key")
            seed_crop = str(seed_key) if seed_key else "<none>"
            reason = str(item.get("reason", "unknown"))
            report_file.write(f"{seed_crop}\t{reason}\t{html_path}\n")


def _iter_bounding_boxes(obj: Any, visited: Optional[set[int]] = None):
    if visited is None:
        visited = set()
    if obj is None:
        return
    if isinstance(obj, BoundingBox):
        yield obj
        return

    obj_id = id(obj)
    if obj_id in visited:
        return

    if isinstance(obj, PydanticBaseModel):
        visited.add(obj_id)
        for field_name in obj.__class__.model_fields:
            yield from _iter_bounding_boxes(getattr(obj, field_name, None), visited)
        return

    if isinstance(obj, dict):
        visited.add(obj_id)
        for value in obj.values():
            yield from _iter_bounding_boxes(value, visited)
        return

    if isinstance(obj, (list, tuple, set)):
        visited.add(obj_id)
        for value in obj:
            yield from _iter_bounding_boxes(value, visited)
        return


def _shift_bbox_for_crop_in_place(
    bbox: BoundingBox,
    crop_left: int,
    crop_top: int,
    old_page_height: float,
    new_page_width: float,
    new_page_height: float,
) -> None:
    original_origin = bbox.coord_origin
    top_left = (
        bbox.to_top_left_origin(page_height=old_page_height)
        if original_origin != CoordOrigin.TOPLEFT
        else bbox
    )

    shifted_l = top_left.l - crop_left
    shifted_r = top_left.r - crop_left
    shifted_t = top_left.t - crop_top
    shifted_b = top_left.b - crop_top

    shifted_l = max(0.0, min(float(new_page_width), shifted_l))
    shifted_r = max(0.0, min(float(new_page_width), shifted_r))
    shifted_t = max(0.0, min(float(new_page_height), shifted_t))
    shifted_b = max(0.0, min(float(new_page_height), shifted_b))
    if shifted_r < shifted_l:
        shifted_l, shifted_r = shifted_r, shifted_l
    if shifted_b < shifted_t:
        shifted_t, shifted_b = shifted_b, shifted_t

    if original_origin == CoordOrigin.TOPLEFT:
        bbox.l = shifted_l
        bbox.r = shifted_r
        bbox.t = shifted_t
        bbox.b = shifted_b
        bbox.coord_origin = CoordOrigin.TOPLEFT
        return

    shifted_top_left = BoundingBox(
        l=shifted_l,
        t=shifted_t,
        r=shifted_r,
        b=shifted_b,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    shifted_original = shifted_top_left.to_bottom_left_origin(page_height=new_page_height)
    bbox.l = shifted_original.l
    bbox.r = shifted_original.r
    bbox.t = shifted_original.t
    bbox.b = shifted_original.b
    bbox.coord_origin = shifted_original.coord_origin


def _collect_table_bboxes_top_left(doc: Any, page_no: int, page_height: float) -> list[BoundingBox]:
    boxes: list[BoundingBox] = []
    try:
        iterator = doc.iterate_items(page_no=page_no)
    except Exception:
        iterator = []

    for item, _ in iterator:
        if not isinstance(item, TableItem):
            continue

        for prov in getattr(item, "prov", []) or []:
            if getattr(prov, "page_no", None) == page_no and getattr(prov, "bbox", None):
                boxes.append(prov.bbox.to_top_left_origin(page_height=page_height))

        table_data = getattr(item, "data", None)
        if table_data is None:
            continue
        for cell in getattr(table_data, "table_cells", []) or []:
            cell_bbox = getattr(cell, "bbox", None)
            if cell_bbox is None:
                continue
            boxes.append(cell_bbox.to_top_left_origin(page_height=page_height))

    return boxes


def _table_bbox_from_fit_result(table_bounds: Any) -> Optional[BoundingBox]:
    if not isinstance(table_bounds, dict):
        return None
    try:
        left = float(table_bounds["left"])
        top = float(table_bounds["top"])
        right = float(table_bounds["right"])
        bottom = float(table_bounds["bottom"])
    except Exception:
        return None
    if not math.isfinite(left) or not math.isfinite(top):
        return None
    if not math.isfinite(right) or not math.isfinite(bottom):
        return None
    if right <= left or bottom <= top:
        return None
    return BoundingBox(
        l=left,
        t=top,
        r=right,
        b=bottom,
        coord_origin=CoordOrigin.TOPLEFT,
    )


def _compute_randomized_crop_box(
    table_bbox: BoundingBox, page_width: int, page_height: int
) -> tuple[int, int, int, int]:
    table_w = max(1, int(math.ceil(table_bbox.r - table_bbox.l)))
    table_h = max(1, int(math.ceil(table_bbox.b - table_bbox.t)))

    max_pad_x = min(TABLE_EXTRACT_MAX_PAD_PX, int(round(table_w * TABLE_EXTRACT_MAX_PAD_RATIO)))
    max_pad_y = min(TABLE_EXTRACT_MAX_PAD_PX, int(round(table_h * TABLE_EXTRACT_MAX_PAD_RATIO)))
    max_pad_x = max(TABLE_EXTRACT_MIN_PAD_PX, max_pad_x)
    max_pad_y = max(TABLE_EXTRACT_MIN_PAD_PX, max_pad_y)
    min_pad_x = min(TABLE_EXTRACT_MIN_PAD_PX, max_pad_x)
    min_pad_y = min(TABLE_EXTRACT_MIN_PAD_PX, max_pad_y)

    pad_left = random.randint(min_pad_x, max_pad_x)
    pad_right = random.randint(min_pad_x, max_pad_x)
    pad_top = random.randint(min_pad_y, max_pad_y)
    pad_bottom = random.randint(min_pad_y, max_pad_y)

    if pad_left == pad_right and max_pad_x > min_pad_x:
        pad_right = min(max_pad_x, pad_right + 1)
    if pad_top == pad_bottom and max_pad_y > min_pad_y:
        pad_bottom = min(max_pad_y, pad_bottom + 1)

    crop_left = max(0, int(math.floor(table_bbox.l)) - pad_left)
    crop_top = max(0, int(math.floor(table_bbox.t)) - pad_top)
    crop_right = min(page_width, int(math.ceil(table_bbox.r)) + pad_right)
    crop_bottom = min(page_height, int(math.ceil(table_bbox.b)) + pad_bottom)

    if crop_right <= crop_left:
        crop_right = min(page_width, crop_left + 1)
    if crop_bottom <= crop_top:
        crop_bottom = min(page_height, crop_top + 1)

    return crop_left, crop_top, crop_right, crop_bottom


def _apply_table_extract_crop(
    doc: Any, preferred_table_bbox: Optional[BoundingBox] = None
) -> dict[str, Any]:
    page_no = 1
    page = doc.pages.get(page_no)
    if page is None or page.image is None or page.image.pil_image is None:
        return {"applied": False, "reason": "missing_page_image"}

    page_image = page.image.pil_image
    page_width, page_height = page_image.size
    if page_width <= 1 or page_height <= 1:
        return {"applied": False, "reason": "invalid_page_size"}

    old_page_height = float(page.size.height if page.size else page_height)
    table_bbox = preferred_table_bbox
    if table_bbox is None:
        table_boxes = _collect_table_bboxes_top_left(
            doc=doc, page_no=page_no, page_height=old_page_height
        )
        if not table_boxes:
            return {"applied": False, "reason": "no_table_bbox"}
        table_bbox = BoundingBox(
            l=min(box.l for box in table_boxes),
            t=min(box.t for box in table_boxes),
            r=max(box.r for box in table_boxes),
            b=max(box.b for box in table_boxes),
            coord_origin=CoordOrigin.TOPLEFT,
        )

    if (
        table_bbox.l < -0.5
        or table_bbox.t < -0.5
        or table_bbox.r > float(page_width) + 0.5
        or table_bbox.b > float(page_height) + 0.5
    ):
        return {"applied": False, "reason": "table_not_fully_visible_on_page"}

    table_left = max(
        0.0,
        min(float(page_width), table_bbox.l - TABLE_EXTRACT_TABLE_SAFETY_PX),
    )
    table_right = min(
        float(page_width),
        max(0.0, table_bbox.r + TABLE_EXTRACT_TABLE_SAFETY_PX),
    )
    table_top = max(
        0.0,
        min(float(page_height), table_bbox.t - TABLE_EXTRACT_TABLE_SAFETY_PX),
    )
    table_bottom = min(
        float(page_height),
        max(0.0, table_bbox.b + TABLE_EXTRACT_TABLE_SAFETY_PX),
    )
    if table_right <= table_left or table_bottom <= table_top:
        return {"applied": False, "reason": "degenerate_table_bbox"}

    table_bbox_for_crop = BoundingBox(
        l=table_left,
        t=table_top,
        r=table_right,
        b=table_bottom,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    crop_left, crop_top, crop_right, crop_bottom = _compute_randomized_crop_box(
        table_bbox=table_bbox_for_crop,
        page_width=page_width,
        page_height=page_height,
    )
    required_left = max(0, int(math.floor(table_bbox_for_crop.l)))
    required_top = max(0, int(math.floor(table_bbox_for_crop.t)))
    required_right = min(page_width, int(math.ceil(table_bbox_for_crop.r)))
    required_bottom = min(page_height, int(math.ceil(table_bbox_for_crop.b)))
    if (
        crop_left > required_left
        or crop_top > required_top
        or crop_right < required_right
        or crop_bottom < required_bottom
    ):
        return {"applied": False, "reason": "crop_cuts_table"}

    new_width = max(1, crop_right - crop_left)
    new_height = max(1, crop_bottom - crop_top)

    cropped_image = page_image.crop((crop_left, crop_top, crop_right, crop_bottom))
    current_dpi = int(getattr(page.image, "dpi", 96) or 96)
    page.image = ImageRef.from_pil(image=cropped_image, dpi=current_dpi)
    if page.size is not None:
        page.size.width = float(new_width)
        page.size.height = float(new_height)

    for bbox in _iter_bounding_boxes(doc):
        _shift_bbox_for_crop_in_place(
            bbox=bbox,
            crop_left=crop_left,
            crop_top=crop_top,
            old_page_height=old_page_height,
            new_page_width=float(new_width),
            new_page_height=float(new_height),
        )

    return {
        "applied": True,
        "crop_left": crop_left,
        "crop_top": crop_top,
        "crop_right": crop_right,
        "crop_bottom": crop_bottom,
        "cropped_width": new_width,
        "cropped_height": new_height,
    }


def _get_worker_converter(
    input_path: Path, render_page_width: int, render_page_height: int
) -> DocumentConverter:
    parent_key = str(input_path.parent.resolve())
    cache_key = (render_page_width, render_page_height, parent_key)
    converter = _WORKER_CONVERTERS.get(cache_key)
    if converter is not None:
        return converter

    html_options = _build_html_options(
        sample_source_uri=input_path,
        render_page_width=render_page_width,
        render_page_height=render_page_height,
    )
    converter = DocumentConverter(
        format_options={InputFormat.HTML: HTMLFormatOption(backend_options=html_options)}
    )
    _WORKER_CONVERTERS[cache_key] = converter
    return converter


def _convert_one(
    input_path_str: str,
    json_path_str: str,
    png_path_str: str,
    render_page_width: int,
    render_page_height: int,
    table_extract: bool,
) -> dict[str, Any]:
    input_path = Path(input_path_str)
    json_path = Path(json_path_str)
    png_path = Path(png_path_str)
    temp_html_path: Optional[Path] = None

    try:
        if json_path.exists():
            return {
                "ok": True,
                "file": input_path.name,
                "elapsed": 0.0,
                "skipped": True,
            }

        html_for_conversion, temp_html_path = _prepare_html_without_head_title(input_path)
        fit_result = _find_autofit_render_width_for_table_cells(
            html_path=html_for_conversion,
            render_page_width=render_page_width,
            render_page_height=render_page_height,
        )
        fit_width = fit_result.get("render_page_width")
        if fit_result.get("has_overflow") or fit_width is None:
            return {
                "ok": True,
                "file": input_path.name,
                "elapsed": 0.0,
                "skipped": True,
                "skip_reason": "table_cell_overflow",
                "autofit_attempts": int(fit_result.get("attempts", 0)),
                "auto_fit_applied": bool(fit_result.get("auto_fit_applied", False)),
                "table_cell_count": int(fit_result.get("table_cell_count", 0)),
                "overflow_cell_count": int(fit_result.get("overflow_cell_count", 0)),
                "requested_width": int(render_page_width),
            }

        effective_width = int(fit_width)
        converter = _get_worker_converter(
            input_path=input_path,
            render_page_width=effective_width,
            render_page_height=render_page_height,
        )

        start = time.perf_counter()
        res = converter.convert(html_for_conversion)
        elapsed = time.perf_counter() - start

        doc = res.document

        table_extract_result: Optional[dict[str, Any]] = None
        if table_extract:
            preferred_table_bbox = _table_bbox_from_fit_result(
                fit_result.get("table_bounds")
            )
            table_extract_result = _apply_table_extract_crop(
                doc, preferred_table_bbox=preferred_table_bbox
            )
            if not table_extract_result.get("applied"):
                return {
                    "ok": True,
                    "file": input_path.name,
                    "elapsed": 0.0,
                    "skipped": True,
                    "skip_reason": "table_extract_failed",
                    "table_extract_reason": str(
                        table_extract_result.get("reason", "unknown")
                    ),
                    "requested_width": int(render_page_width),
                    "target_width": effective_width,
                }

        json_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.parent.mkdir(parents=True, exist_ok=True)

        page = doc.pages[1]
        rendered_width: Optional[int] = None
        rendered_height: Optional[int] = None
        if page.image and page.image.pil_image:
            rendered_width, rendered_height = page.image.pil_image.size
            page.image.pil_image.save(png_path)

        doc.save_as_json(json_path)

        return {
            "ok": True,
            "file": input_path.name,
            "elapsed": elapsed,
            "skipped": False,
            "rendered_width": rendered_width,
            "rendered_height": rendered_height,
            "requested_width": int(render_page_width),
            "target_width": effective_width,
            "target_height": render_page_height,
            "autofit_attempts": int(fit_result.get("attempts", 0)),
            "auto_fit_applied": bool(fit_result.get("auto_fit_applied", False)),
            "table_cell_count": int(fit_result.get("table_cell_count", 0)),
            "overflow_cell_count": int(fit_result.get("overflow_cell_count", 0)),
            "table_extract_applied": bool(table_extract_result and table_extract_result.get("applied")),
            "table_extract_reason": (
                str(table_extract_result.get("reason"))
                if table_extract_result and not table_extract_result.get("applied")
                else None
            ),
            "cropped_width": (
                int(table_extract_result["cropped_width"])
                if table_extract_result and table_extract_result.get("applied")
                else None
            ),
            "cropped_height": (
                int(table_extract_result["cropped_height"])
                if table_extract_result and table_extract_result.get("applied")
                else None
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "file": input_path.name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if temp_html_path is not None:
            try:
                temp_html_path.unlink(missing_ok=True)
            except Exception:
                _log.warning(
                    "Failed to remove temporary sanitized HTML: %s", temp_html_path
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render HTML samples into Docling JSON + PNG while matching render width "
            "to seed PNG width inferred from filename."
        )
    )
    parser.add_argument(
        "--input-html-path",
        required=True,
        type=Path,
        help="Root folder containing HTML files (scanned recursively).",
    )
    parser.add_argument(
        "--seed-image-path",
        required=True,
        type=Path,
        help="Root folder containing seed PNG files (scanned recursively).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output root folder for mirrored JSON and images.",
    )
    parser.add_argument(
        "--same-output-folder",
        action="store_true",
        help=(
            "Write JSON and PNG side-by-side in the same mirrored folder instead of "
            "separate json/ and images/ subfolders."
        ),
    )
    parser.add_argument(
        "--table-extract",
        action="store_true",
        help=(
            "After conversion, crop the rendered page image to table area with "
            "small random per-side padding and shift all Docling bboxes accordingly."
        ),
    )
    args = parser.parse_args()

    input_html_path = args.input_html_path
    seed_image_path = args.seed_image_path
    out_dir = args.out_dir
    same_output_folder = args.same_output_folder
    table_extract = args.table_extract
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Starting HTML conversion process with seed image resolution matching.")
    print(f"Input HTML directory: {input_html_path}")
    print(f"Seed image directory: {seed_image_path}")
    print(f"Output root: {out_dir}")
    print(
        "Output layout: "
        f"{'shared folder (json+png)' if same_output_folder else 'separate json/ and images/ subfolders'}"
    )
    print(f"Table extract crop: {table_extract}")

    seed_by_stem, seed_entries = _build_seed_index(seed_image_path)
    print(f"Indexed {len(seed_entries)} seed PNG files.")
    if not seed_entries:
        print(f"No PNG files found in {seed_image_path}")
        return

    total_html_files = 0
    for _ in _iter_html_files(input_html_path):
        total_html_files += 1
    if total_html_files == 0:
        print(f"No HTML files found in {input_html_path}")
        return

    timings: list[float] = []
    failed_files: list[Path] = []
    unresolved_seed_files: list[Path] = []
    unresolved_seed_items: list[dict[str, Any]] = []
    unreadable_seed_files: list[Path] = []
    table_overflow_skipped_files: list[Path] = []
    table_extract_failed_skipped_files: list[Path] = []
    successful_render_sizes: list[tuple[int, int]] = []
    seed_size_cache: dict[Path, tuple[int, int]] = {}
    html_without_seeds_path = out_dir / "html_without_seeds.txt"
    max_workers = min(
        16, max(1, int(os.environ.get("DOCLING_HTML_WORKERS", os.cpu_count() or 1)))
    )
    partition_size = max(
        1,
        int(os.environ.get("DOCLING_PARTITION_SIZE", PARTITION_SIZE)),
    )
    use_partitions = total_html_files > partition_size
    max_in_flight = max_workers * 4
    print(f"Discovered {total_html_files} HTML files.")
    print(f"Using {max_workers} worker process(es)")
    print(f"Partition size: {partition_size} files per part")
    print(f"Partitions enabled: {use_partitions}")
    print(f"Max in-flight jobs: {max_in_flight}")

    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
        futures: dict[Any, tuple[Path, bool]] = {}
        scanned_count = 0
        submitted_count = 0
        skipped_count = 0
        success_count = 0
        fallback_submitted_count = 0
        fallback_success_count = 0
        table_overflow_skipped_count = 0
        table_extract_applied_count = 0
        table_extract_failed_skipped_count = 0
        started_announced = False
        first_result_announced = False
        with tqdm(
            total=0,
            desc="HTML conversions",
            unit="file",
            dynamic_ncols=True,
        ) as pbar:
            file_index = 0

            def _handle_done_future(future: Any, input_path: Path, is_fallback: bool) -> None:
                nonlocal success_count, fallback_success_count, table_overflow_skipped_count
                nonlocal table_extract_applied_count, table_extract_failed_skipped_count
                nonlocal first_result_announced
                pbar.update(1)
                try:
                    result = future.result()
                except Exception as exc:
                    failed_files.append(input_path)
                    _log.exception("Worker crashed for %s: %s", input_path, exc)
                    tqdm.write(f"{input_path.name}: FAILED (worker crash: {exc})")
                    pbar.set_postfix(
                        scanned=scanned_count,
                        queued=submitted_count,
                        skipped=skipped_count,
                        ok=success_count,
                        failed=len(failed_files),
                        left=max(0, submitted_count - pbar.n),
                    )
                    return

                if result.get("ok"):
                    if result.get("skipped"):
                        skip_reason = str(result.get("skip_reason", "already_converted"))
                        if skip_reason == "table_cell_overflow":
                            table_overflow_skipped_count += 1
                            if len(table_overflow_skipped_files) < 200:
                                table_overflow_skipped_files.append(input_path)
                            overflow_cells = int(result.get("overflow_cell_count", 0))
                            table_cells = int(result.get("table_cell_count", 0))
                            attempts = int(result.get("autofit_attempts", 0))
                            tqdm.write(
                                f"{result['file']}: skipped (table cell overflow "
                                f"{overflow_cells}/{table_cells} after {attempts} attempt(s))"
                            )
                        elif skip_reason == "table_extract_failed":
                            table_extract_failed_skipped_count += 1
                            if len(table_extract_failed_skipped_files) < 200:
                                table_extract_failed_skipped_files.append(input_path)
                            reason = str(result.get("table_extract_reason", "unknown"))
                            tqdm.write(
                                f"{result['file']}: skipped (table-extract failed: {reason})"
                            )
                        else:
                            tqdm.write(f"{result['file']}: skipped (already converted)")
                    else:
                        success_count += 1
                        if is_fallback:
                            fallback_success_count += 1
                        elapsed = float(result["elapsed"])
                        timings.append(elapsed)
                        rendered_w = result.get("rendered_width")
                        rendered_h = result.get("rendered_height")
                        requested_w = result.get("requested_width")
                        target_w = result.get("target_width")
                        target_h = result.get("target_height")
                        auto_fit_applied = bool(result.get("auto_fit_applied", False))
                        autofit_attempts = int(result.get("autofit_attempts", 0))
                        if table_extract:
                            if bool(result.get("table_extract_applied", False)):
                                table_extract_applied_count += 1
                        avg_width = rendered_w if rendered_w else target_w
                        avg_height = rendered_h if rendered_h else target_h
                        if isinstance(avg_width, int) and isinstance(avg_height, int):
                            if avg_width > 0 and avg_height > 0:
                                successful_render_sizes.append((avg_width, avg_height))
                        if auto_fit_applied and requested_w and target_w:
                            tqdm.write(
                                f"{result['file']}: converted in {elapsed:.3f}s "
                                f"(autofit_w={target_w}, seed_w={requested_w}, "
                                f"attempts={autofit_attempts})"
                            )
                        elif rendered_w and target_w and rendered_w != target_w:
                            tqdm.write(
                                f"{result['file']}: converted in {elapsed:.3f}s "
                                f"(target_w={target_w}, rendered_w={rendered_w})"
                            )
                        elif table_extract and result.get("table_extract_applied"):
                            cropped_w = result.get("cropped_width")
                            cropped_h = result.get("cropped_height")
                            tqdm.write(
                                f"{result['file']}: converted in {elapsed:.3f}s "
                                f"(table_extract={cropped_w}x{cropped_h})"
                            )
                        else:
                            tqdm.write(f"{result['file']}: converted in {elapsed:.3f}s")
                else:
                    failed_files.append(input_path)
                    _log.error(
                        "Failed to convert %s\n%s",
                        input_path,
                        result.get("traceback", result.get("error", "unknown error")),
                    )
                    tqdm.write(
                        f"{result['file']}: FAILED ({result.get('error', 'unknown error')})"
                    )

                if not first_result_announced:
                    tqdm.write("Workers are active. First conversion result received.")
                    first_result_announced = True
                if pbar.n % 1000 == 0:
                    tqdm.write(
                        "Progress update: "
                        f"scanned={scanned_count}, submitted={submitted_count}, "
                        f"completed={pbar.n}, skipped={skipped_count}, "
                        f"ok={success_count}, failed={len(failed_files)}, "
                        f"no_seed={len(unresolved_seed_files)}, "
                        f"bad_seed={len(unreadable_seed_files)}, "
                        f"table_ovf_skip={table_overflow_skipped_count}, "
                        f"table_extract_skip={table_extract_failed_skipped_count}, "
                        f"table_extract={table_extract_applied_count}, "
                        f"in_flight={len(futures)}"
                    )

                pbar.set_postfix(
                    scanned=scanned_count,
                    queued=submitted_count,
                    skipped=skipped_count,
                    ok=success_count,
                    failed=len(failed_files),
                    no_seed=len(unresolved_seed_files),
                    bad_seed=len(unreadable_seed_files),
                    fallback=fallback_success_count,
                    ovf_skip=table_overflow_skipped_count,
                    tbl_crop_skip=table_extract_failed_skipped_count,
                    tbl_crop=table_extract_applied_count,
                    left=max(0, submitted_count - pbar.n),
                )

            for input_path in _iter_html_files(input_html_path):
                scanned_count += 1
                file_index += 1
                rel_dir = input_path.parent.relative_to(input_html_path)
                if use_partitions:
                    part_no = ((file_index - 1) // partition_size) + 1
                    base_root = out_dir / f"part{part_no}"
                else:
                    base_root = out_dir
                mirrored_root = base_root / rel_dir if rel_dir != Path(".") else base_root
                if same_output_folder:
                    json_path = mirrored_root / f"{input_path.stem}.json"
                    png_path = mirrored_root / f"{input_path.stem}.png"
                else:
                    json_dir = mirrored_root / "json"
                    png_dir = mirrored_root / "images"
                    json_path = json_dir / f"{input_path.stem}.json"
                    png_path = png_dir / f"{input_path.stem}.png"

                if json_path.exists():
                    skipped_count += 1
                    if scanned_count % 5000 == 0:
                        pbar.set_postfix(
                            scanned=scanned_count,
                            queued=submitted_count,
                            skipped=skipped_count,
                            ok=success_count,
                            failed=len(failed_files),
                            no_seed=len(unresolved_seed_files),
                            bad_seed=len(unreadable_seed_files),
                            left=max(0, submitted_count - pbar.n),
                        )
                    continue

                seed_key = _extract_seed_key_from_html_name(input_path.stem)
                if seed_key is None:
                    unresolved_seed_files.append(input_path)
                    unresolved_seed_items.append(
                        {
                            "input_path": input_path,
                            "seed_key": None,
                            "reason": "no_seed_key_in_name",
                            "json_path": json_path,
                            "png_path": png_path,
                        }
                    )
                    continue

                seed_png = _find_seed_png(seed_key, seed_by_stem, seed_entries)
                if seed_png is None:
                    unresolved_seed_files.append(input_path)
                    unresolved_seed_items.append(
                        {
                            "input_path": input_path,
                            "seed_key": seed_key,
                            "reason": "no_matching_seed_image",
                            "json_path": json_path,
                            "png_path": png_path,
                        }
                    )
                    continue

                try:
                    if seed_png not in seed_size_cache:
                        with Image.open(seed_png) as im:
                            seed_size_cache[seed_png] = (int(im.width), int(im.height))
                    seed_width, seed_height = seed_size_cache[seed_png]
                except Exception:
                    unreadable_seed_files.append(input_path)
                    _log.exception("Could not read seed image size for %s", seed_png)
                    continue

                future = executor.submit(
                    _convert_one,
                    str(input_path),
                    str(json_path),
                    str(png_path),
                    seed_width,
                    seed_height,
                    table_extract,
                )
                futures[future] = (input_path, False)
                submitted_count += 1
                if submitted_count <= 100 or submitted_count % 1000 == 0:
                    pbar.total = submitted_count
                    pbar.refresh()
                if not started_announced:
                    tqdm.write("Conversion started. First job submitted.")
                    started_announced = True
                if scanned_count % 100000 == 0:
                    tqdm.write(
                        "Scan update: "
                        f"scanned={scanned_count}, submitted={submitted_count}, "
                        f"skipped={skipped_count}, in_flight={len(futures)}"
                    )

                if len(futures) >= max_in_flight:
                    done, _ = wait(
                        set(futures.keys()),
                        return_when=FIRST_COMPLETED,
                    )
                    for done_future in done:
                        done_input_path, done_is_fallback = futures.pop(done_future)
                        _handle_done_future(done_future, done_input_path, done_is_fallback)

            while futures:
                done, _ = wait(
                    set(futures.keys()),
                    return_when=FIRST_COMPLETED,
                )
                for done_future in done:
                    done_input_path, done_is_fallback = futures.pop(done_future)
                    _handle_done_future(done_future, done_input_path, done_is_fallback)

            _write_unresolved_seed_report(
                report_path=html_without_seeds_path,
                input_html_path=input_html_path,
                unresolved_items=unresolved_seed_items,
            )
            tqdm.write(
                "Wrote unresolved-seed report: "
                f"{html_without_seeds_path} ({len(unresolved_seed_items)} entries)"
            )

            fallback_size = _average_size(successful_render_sizes)
            if fallback_size is None and seed_size_cache:
                fallback_size = _average_size(list(seed_size_cache.values()))
            if fallback_size is None:
                try:
                    with Image.open(seed_entries[0][1]) as im:
                        fallback_size = (max(1, int(im.width)), max(1, int(im.height)))
                except Exception:
                    fallback_size = (1024, 1448)
                    _log.exception(
                        "Could not infer fallback render size from seed images; using %sx%s",
                        fallback_size[0],
                        fallback_size[1],
                    )

            if unresolved_seed_items:
                fallback_width, fallback_height = fallback_size
                tqdm.write(
                    "Submitting unresolved HTML files with fallback size "
                    f"{fallback_width}x{fallback_height}."
                )
                for item in unresolved_seed_items:
                    input_path = Path(item["input_path"])
                    json_path = Path(item["json_path"])
                    png_path = Path(item["png_path"])

                    if json_path.exists():
                        skipped_count += 1
                        continue

                    future = executor.submit(
                        _convert_one,
                        str(input_path),
                        str(json_path),
                        str(png_path),
                        fallback_width,
                        fallback_height,
                        table_extract,
                    )
                    futures[future] = (input_path, True)
                    submitted_count += 1
                    fallback_submitted_count += 1
                    if (
                        submitted_count <= 100
                        or submitted_count % 1000 == 0
                        or fallback_submitted_count <= 100
                    ):
                        pbar.total = submitted_count
                        pbar.refresh()
                    if len(futures) >= max_in_flight:
                        done, _ = wait(
                            set(futures.keys()),
                            return_when=FIRST_COMPLETED,
                        )
                        for done_future in done:
                            done_input_path, done_is_fallback = futures.pop(done_future)
                            _handle_done_future(
                                done_future, done_input_path, done_is_fallback
                            )

                while futures:
                    done, _ = wait(
                        set(futures.keys()),
                        return_when=FIRST_COMPLETED,
                    )
                    for done_future in done:
                        done_input_path, done_is_fallback = futures.pop(done_future)
                        _handle_done_future(done_future, done_input_path, done_is_fallback)

    print(
        f"Scanned {scanned_count} files. "
        f"Submitted {submitted_count}. "
        f"Fallback submitted {fallback_submitted_count}. "
        f"Skipped existing {skipped_count}. "
        f"Table-overflow skipped {table_overflow_skipped_count}. "
        f"Table-extract applied {table_extract_applied_count}. "
        f"Table-extract skipped {table_extract_failed_skipped_count}. "
        f"No-seed {len(unresolved_seed_files)}. "
        f"Bad-seed {len(unreadable_seed_files)}."
    )
    print(f"Unresolved seed report: {html_without_seeds_path}")

    if timings:
        avg_time = sum(timings) / len(timings)
        print(f"Average conversion time: {avg_time:.3f}s across {len(timings)} samples")
    if fallback_submitted_count:
        print(
            "Fallback conversion success: "
            f"{fallback_success_count}/{fallback_submitted_count}"
        )
    if table_extract:
        print(
            "Table-extract crop applied: "
            f"{table_extract_applied_count} (skipped: {table_extract_failed_skipped_count})"
        )
    if table_extract_failed_skipped_files:
        print(
            "Examples skipped for table-extract "
            f"(showing up to {min(50, len(table_extract_failed_skipped_files))}):"
        )
        for skipped_file in table_extract_failed_skipped_files[:50]:
            print(f" - {skipped_file}")
    print(f"Skipped due to persistent table cell overflow: {table_overflow_skipped_count}")
    if failed_files:
        print(f"Failed files: {len(failed_files)}")
        for failed_path in failed_files:
            print(f" - {failed_path}")
    if table_overflow_skipped_files:
        print(
            "Examples skipped for table overflow "
            f"(showing up to {min(50, len(table_overflow_skipped_files))}):"
        )
        for overflow_file in table_overflow_skipped_files[:50]:
            print(f" - {overflow_file}")
    if unresolved_seed_files:
        print(f"Files without matching seed image: {len(unresolved_seed_files)}")
        for missing_seed_file in unresolved_seed_files[:50]:
            print(f" - {missing_seed_file}")
    if unreadable_seed_files:
        print(f"Files with unreadable seed image: {len(unreadable_seed_files)}")
        for bad_seed_file in unreadable_seed_files[:50]:
            print(f" - {bad_seed_file}")


if __name__ == "__main__":
    main()
