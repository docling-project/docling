"""Iterate over PDFs in a directory, load them with DoclingParseDocumentBackend,
and extract text cells and page images for every page."""

import argparse
import gc
import json
import logging
import os
from pathlib import Path

import psutil

from docling.backend.docling_parse_backend import (
    DoclingParseDocumentBackend,
    DoclingParsePageBackend,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


_PROC = psutil.Process(os.getpid())


def _rss_mb() -> float:
    """Return current process RSS in MiB."""
    return _PROC.memory_info().rss / (1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the PDF files to process.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. If provided, page images (PNG) and "
            "text cells (JSON) are written here, one subdirectory per PDF."
        ),
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=1.5,
        help="Scale factor for rendered page images (default: 1.5).",
    )
    parser.add_argument(
        "--glob",
        default="*.pdf",
        help="Glob pattern to match files in input_dir (default: '*.pdf').",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help=(
            "Path to the page-count cache JSON (default: "
            "'<input_dir>/.docling_page_counts.json'). Entries are keyed by "
            "absolute path and validated against file size and mtime."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the page-count cache and always re-parse to count pages.",
    )
    return parser.parse_args()


def _count_pages(pdf_path: Path) -> int:
    in_doc = InputDocument(
        path_or_stream=pdf_path,
        format=InputFormat.PDF,
        backend=DoclingParseDocumentBackend,
    )
    doc_backend: DoclingParseDocumentBackend = in_doc._backend
    try:
        if not doc_backend.is_valid():
            return 0
        return doc_backend.page_count()
    finally:
        doc_backend.unload()


def _load_cache(cache_file: Path) -> dict[str, dict]:
    if not cache_file.is_file():
        return {}
    try:
        return json.loads(cache_file.read_text())
    except (OSError, json.JSONDecodeError) as e:
        _log.warning("Ignoring unreadable cache %s: %s", cache_file, e)
        return {}


def _save_cache(cache_file: Path, cache: dict[str, dict]) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=2, sort_keys=True))
    except OSError as e:
        _log.warning("Failed to write cache %s: %s", cache_file, e)


def collect_pdfs_by_page_count(
    pdfs: list[Path],
    cache_file: Path | None,
) -> list[tuple[Path, int]]:
    """Return [(pdf_path, page_count), ...] sorted by descending page count,
    using cache_file (if given) to skip re-parsing unchanged files."""
    cache: dict[str, dict] = _load_cache(cache_file) if cache_file else {}
    results: list[tuple[Path, int]] = []
    cache_dirty = False

    for pdf_path in pdfs:
        abs_key = str(pdf_path.resolve())
        stat = pdf_path.stat()
        entry = cache.get(abs_key)
        if (
            entry is not None
            and entry.get("size") == stat.st_size
            and entry.get("mtime_ns") == stat.st_mtime_ns
            and isinstance(entry.get("page_count"), int)
        ):
            page_count = entry["page_count"]
            _log.info("  %s: %d page(s) [cached]", pdf_path.name, page_count)
        else:
            _log.info("  %s: counting pages...", pdf_path.name)
            try:
                page_count = _count_pages(pdf_path)
            except Exception:
                _log.exception("Failed to count pages for %s", pdf_path.name)
                continue
            _log.info("  %s: %d page(s)", pdf_path.name, page_count)
            if cache_file is not None:
                cache[abs_key] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "page_count": page_count,
                }
                cache_dirty = True

        results.append((pdf_path, page_count))

    if cache_file is not None and cache_dirty:
        _save_cache(cache_file, cache)

    results.sort(key=lambda item: (-item[1], item[0].name))
    return results


def process_pdf(
    pdf_path: Path,
    output_dir: Path | None,
    scale: float,
) -> None:
    _log.info("Processing %s", pdf_path.name)

    in_doc = InputDocument(
        path_or_stream=pdf_path,
        format=InputFormat.PDF,
        backend=DoclingParseDocumentBackend,
    )
    doc_backend: DoclingParseDocumentBackend = in_doc._backend

    if not doc_backend.is_valid():
        _log.warning("Skipping invalid document: %s", pdf_path.name)
        doc_backend.unload()
        return

    pdf_out_dir: Path | None = None
    if output_dir is not None:
        pdf_out_dir = output_dir / pdf_path.stem
        pdf_out_dir.mkdir(parents=True, exist_ok=True)

    num_pages = doc_backend.page_count()
    _log.info("  %d page(s)", num_pages)

    for page_no in range(num_pages):
        rss_before = _rss_mb()
        page_backend: DoclingParsePageBackend = doc_backend.load_page(page_no)
        try:
            text_cells = list(page_backend.get_text_cells())
            page_image = page_backend.get_page_image(scale=scale)
            rss_loaded = _rss_mb()

            _log.info(
                "  page %d/%d: %d text cell(s), image size=%s, "
                "RSS before=%.1f MiB, loaded=%.1f MiB (+%.1f)",
                page_no + 1,
                num_pages,
                len(text_cells),
                page_image.size,
                rss_before,
                rss_loaded,
                rss_loaded - rss_before,
            )

            """
            if pdf_out_dir is not None:
                image_path = pdf_out_dir / f"page_{page_no + 1:04d}.png"
                page_image.save(image_path)

                cells_path = pdf_out_dir / f"page_{page_no + 1:04d}_cells.json"
                cells_payload = [cell.model_dump(mode="json") for cell in text_cells]
                cells_path.write_text(json.dumps(cells_payload, indent=2))
            """
        finally:
            page_backend.unload()
            del page_backend
            gc.collect()
            rss_after = _rss_mb()
            _log.info(
                "  page %d/%d: RSS after unload=%.1f MiB (delta vs before=%+.1f)",
                page_no + 1,
                num_pages,
                rss_after,
                rss_after - rss_before,
            )

    doc_backend.unload()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(args.input_dir.glob(args.glob))
    if not pdfs:
        _log.warning(
            "No files matched '%s' in %s", args.glob, args.input_dir
        )
        return

    _log.info("Found %d PDF file(s) in %s", len(pdfs), args.input_dir)

    if args.no_cache:
        cache_file: Path | None = None
    else:
        cache_file = (
            args.cache_file
            if args.cache_file is not None
            else args.input_dir / ".docling_page_counts.json"
        )

    _log.info("Collecting page counts...")
    ordered = collect_pdfs_by_page_count(pdfs, cache_file)

    _log.info("Processing order (by descending page count):")
    for pdf_path, page_count in ordered:
        _log.info("  %5d pages  %s", page_count, pdf_path.name)

    for pdf_path, _ in ordered:
        try:
            process_pdf(pdf_path, args.output_dir, args.scale)
        except Exception:
            _log.exception("Failed to process %s", pdf_path.name)


if __name__ == "__main__":
    main()
