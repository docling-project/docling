"""Iterate over PDFs in a directory, load them with
ThreadedDoclingParseDocumentBackend, and extract text cells and page images for
every page."""

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import psutil

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.settings import DocumentLimits

_log = logging.getLogger(__name__)


_PROC = psutil.Process(os.getpid())


def _memory_metrics_mb() -> dict[str, float]:
    """Return available process memory counters in MiB."""
    info = _PROC.memory_info()
    metrics = {
        "rss_mb": info.rss / (1024 * 1024),
        "vms_mb": info.vms / (1024 * 1024),
    }

    try:
        full_info = _PROC.memory_full_info()
    except (psutil.AccessDenied, AttributeError):
        return metrics

    if hasattr(full_info, "uss"):
        metrics["uss_mb"] = full_info.uss / (1024 * 1024)
    if hasattr(full_info, "pss"):
        metrics["pss_mb"] = full_info.pss / (1024 * 1024)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        help="Local directory containing PDF files to process.",
    )
    source_group.add_argument(
        "-r",
        "--repo-id",
        help=(
            "Hugging Face dataset repo ID. The script downloads the dataset "
            "snapshot and iterates over its 'pdf/' subfolder."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face dataset revision, branch, or commit.",
    )
    parser.add_argument(
        "--mode",
        choices=("throughput", "memory"),
        default="throughput",
        help=(
            "Run either a throughput measurement or a memory profiling run. "
            "Memory mode writes the JSONL file consumed by plot_memory_metrics.py."
        ),
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
    parser.add_argument(
        "--parser-threads",
        type=int,
        default=8,
        help="Optional number of docling-parse parser threads.",
    )
    parser.add_argument(
        "--release-native-memory-every-n-pages",
        type=int,
        default=128,
        help="Release native parser memory after every N decoded pages.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional global limit for the total number of pages to iterate.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("memory-metrics.jsonl"),
        help=(
            "JSONL file for memory mode. This remains compatible with "
            "plot_memory_metrics.py."
        ),
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON summary report path. Defaults to "
            "'iterate-pdf-pages-<mode>-report.json'."
        ),
    )
    return parser.parse_args()


def _default_report_file(mode: str) -> Path:
    return Path(f"iterate-pdf-pages-{mode}-report.json")


def _make_backend_options(
    parser_threads: int | None,
    release_native_memory_every_n_pages: int,
) -> ThreadedDoclingParseBackendOptions:
    return ThreadedDoclingParseBackendOptions(
        parser_threads=parser_threads,
        release_native_memory_every_n_pages=release_native_memory_every_n_pages,
    )


def _resolve_input_dir(args: argparse.Namespace) -> Path:
    if args.input_dir is not None:
        return args.input_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Hugging Face dataset support requires 'huggingface_hub' to be installed."
        ) from exc

    assert args.repo_id is not None
    snapshot_dir = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns="pdf/**",
        )
    )
    pdf_dir = snapshot_dir / "pdf"
    if not pdf_dir.is_dir():
        raise SystemExit(
            f"Dataset '{args.repo_id}' does not contain a 'pdf/' subfolder."
        )
    return pdf_dir


def _count_pages(
    pdf_path: Path,
    parser_threads: int | None,
    release_native_memory_every_n_pages: int,
) -> int:
    in_doc = InputDocument(
        path_or_stream=pdf_path,
        format=InputFormat.PDF,
        backend=ThreadedDoclingParseDocumentBackend,
        backend_options=_make_backend_options(
            parser_threads,
            release_native_memory_every_n_pages,
        ),
    )
    doc_backend: ThreadedDoclingParseDocumentBackend = in_doc._backend
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
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _log.warning("Ignoring unreadable cache %s: %s", cache_file, e)
        return {}


def _save_cache(cache_file: Path, cache: dict[str, dict]) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(cache, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as e:
        _log.warning("Failed to write cache %s: %s", cache_file, e)


def _append_metrics(metrics_file: Path | None, payload: dict[str, Any]) -> None:
    if metrics_file is None:
        return

    try:
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with metrics_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, sort_keys=True))
            fp.write("\n")
    except OSError as e:
        _log.warning("Failed to append metrics to %s: %s", metrics_file, e)


def _write_report(report_file: Path, payload: dict[str, Any]) -> None:
    try:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as e:
        _log.warning("Failed to write report %s: %s", report_file, e)


def collect_pdfs_by_page_count(
    pdfs: list[Path],
    cache_file: Path | None,
    parser_threads: int | None,
    release_native_memory_every_n_pages: int,
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
                page_count = _count_pages(
                    pdf_path,
                    parser_threads,
                    release_native_memory_every_n_pages,
                )
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
    parser_threads: int | None,
    release_native_memory_every_n_pages: int,
    processed_pages: int,
    target_total_pages: int,
    max_pages: int | None,
    mode: str,
    metrics_file: Path | None,
    run_start_time: float,
) -> tuple[int, int]:
    _log.info("Processing %s", pdf_path.name)

    limits: DocumentLimits | None = None
    if max_pages is not None:
        remaining_pages = max_pages - processed_pages
        if remaining_pages <= 0:
            return processed_pages, 0
        limits = DocumentLimits(page_range=(1, remaining_pages))

    in_doc = InputDocument(
        path_or_stream=pdf_path,
        format=InputFormat.PDF,
        backend=ThreadedDoclingParseDocumentBackend,
        backend_options=_make_backend_options(
            parser_threads,
            release_native_memory_every_n_pages,
        ),
        limits=limits,
    )
    doc_backend: ThreadedDoclingParseDocumentBackend = in_doc._backend

    try:
        if not doc_backend.is_valid():
            _log.warning("Skipping invalid document: %s", pdf_path.name)
            return processed_pages, 0

        pdf_out_dir: Path | None = None
        if output_dir is not None:
            pdf_out_dir = output_dir / pdf_path.stem
            pdf_out_dir.mkdir(parents=True, exist_ok=True)

        num_pages = doc_backend.page_count()
        failed_pages = 0
        _log.info("  %d page(s)", num_pages)

        for page_backend in doc_backend.iter_pages():
            page_no = page_backend.page_no
            processed_pages += 1
            page_success = False
            memory_before: dict[str, float] | None = None
            rss_before: float | None = None
            try:
                if mode == "memory":
                    memory_before = _memory_metrics_mb()
                    rss_before = memory_before["rss_mb"]

                text_cells = list(page_backend.get_text_cells())
                page_image = page_backend.get_page_image(scale=scale)
                page_success = True

                if mode == "memory":
                    assert memory_before is not None
                    assert rss_before is not None
                    memory_loaded = _memory_metrics_mb()
                    rss_loaded = memory_loaded["rss_mb"]
                    _log.info(
                        "  page %d/%d: %d text cell(s), image size=%s, "
                        "total page %d/%d, "
                        "RSS before=%.1f MiB, loaded=%.1f MiB (+%.1f)",
                        page_no,
                        num_pages,
                        len(text_cells),
                        page_image.size,
                        processed_pages,
                        target_total_pages,
                        rss_before,
                        rss_loaded,
                        rss_loaded - rss_before,
                    )
                    _append_metrics(
                        metrics_file,
                        {
                            "doc_page_count": num_pages,
                            "elapsed_seconds": time.monotonic() - run_start_time,
                            "event": "loaded",
                            "image_height": page_image.size[1],
                            "image_width": page_image.size[0],
                            "page_no": page_no,
                            "pdf": pdf_path.name,
                            "pdf_path": str(pdf_path),
                            "memory_before_mb": memory_before,
                            "memory_loaded_mb": memory_loaded,
                            "rss_before_mb": rss_before,
                            "rss_loaded_delta_mb": rss_loaded - rss_before,
                            "rss_loaded_mb": rss_loaded,
                            "success": True,
                            "target_total_pages": target_total_pages,
                            "text_cell_count": len(text_cells),
                            "total_page_no": processed_pages,
                        },
                    )
                else:
                    _log.info(
                        "  page %d/%d: %d text cell(s), image size=%s, total page %d/%d",
                        page_no,
                        num_pages,
                        len(text_cells),
                        page_image.size,
                        processed_pages,
                        target_total_pages,
                    )

                """
                if pdf_out_dir is not None:
                    image_path = pdf_out_dir / f"page_{page_no:04d}.png"
                    page_image.save(image_path)

                    cells_path = pdf_out_dir / f"page_{page_no:04d}_cells.json"
                    cells_payload = [cell.model_dump(mode="json") for cell in text_cells]
                    cells_path.write_text(json.dumps(cells_payload, indent=2))
                """
            except Exception:
                failed_pages += 1
                _log.exception(
                    "  page %d/%d: failed to parse/render", page_no, num_pages
                )
            finally:
                page_backend.unload()
                if mode == "memory":
                    del page_backend
                    gc.collect()
                    assert memory_before is not None
                    assert rss_before is not None
                    memory_after = _memory_metrics_mb()
                    rss_after = memory_after["rss_mb"]
                    _append_metrics(
                        metrics_file,
                        {
                            "doc_page_count": num_pages,
                            "elapsed_seconds": time.monotonic() - run_start_time,
                            "event": "after_unload",
                            "page_no": page_no,
                            "pdf": pdf_path.name,
                            "pdf_path": str(pdf_path),
                            "memory_after_mb": memory_after,
                            "memory_before_mb": memory_before,
                            "rss_after_delta_mb": rss_after - rss_before,
                            "rss_after_mb": rss_after,
                            "rss_before_mb": rss_before,
                            "success": page_success,
                            "target_total_pages": target_total_pages,
                            "total_page_no": processed_pages,
                        },
                    )
                    _log.info(
                        "  page %d/%d: total page %d/%d, RSS after unload=%.1f MiB "
                        "(delta vs before=%+.1f)",
                        page_no,
                        num_pages,
                        processed_pages,
                        target_total_pages,
                        rss_after,
                        rss_after - rss_before,
                    )

        if failed_pages:
            _log.warning(
                "Completed %s with %d failed page(s) out of %d.",
                pdf_path.name,
                failed_pages,
                num_pages,
            )
        return processed_pages, failed_pages
    finally:
        doc_backend.unload()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_start_time = time.monotonic()
    args = parse_args()
    input_dir = _resolve_input_dir(args)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file: Path | None = args.metrics_file if args.mode == "memory" else None
    if metrics_file is not None:
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text("", encoding="utf-8")
        _log.info("Writing per-page metrics to %s", metrics_file)

    report_file = args.report_file or _default_report_file(args.mode)

    pdfs = sorted(input_dir.glob(args.glob))
    if not pdfs:
        _log.warning("No files matched '%s' in %s", args.glob, input_dir)
        return

    _log.info("Found %d PDF file(s) in %s", len(pdfs), input_dir)
    _log.info("Running in %s mode", args.mode)

    if args.no_cache:
        cache_file: Path | None = None
    else:
        cache_file = (
            args.cache_file
            if args.cache_file is not None
            else input_dir / ".docling_page_counts.json"
        )

    _log.info("Collecting page counts...")
    ordered = collect_pdfs_by_page_count(
        pdfs,
        cache_file,
        args.parser_threads,
        args.release_native_memory_every_n_pages,
    )
    total_available_pages = sum(page_count for _, page_count in ordered)
    target_total_pages = (
        min(total_available_pages, args.max_pages)
        if args.max_pages is not None
        else total_available_pages
    )

    _log.info("Processing order (by descending page count):")
    for pdf_path, page_count in ordered:
        _log.info("  %5d pages  %s", page_count, pdf_path.name)

    processed_pages = 0
    failed_pages = 0
    for pdf_path, _ in ordered:
        if args.max_pages is not None and processed_pages >= args.max_pages:
            _log.info(
                "Reached max page limit (%d/%d). Stopping.",
                processed_pages,
                target_total_pages,
            )
            break
        try:
            processed_pages, pdf_failed_pages = process_pdf(
                pdf_path,
                args.output_dir,
                args.scale,
                args.parser_threads,
                args.release_native_memory_every_n_pages,
                processed_pages=processed_pages,
                target_total_pages=target_total_pages,
                max_pages=args.max_pages,
                mode=args.mode,
                metrics_file=metrics_file,
                run_start_time=run_start_time,
            )
            failed_pages += pdf_failed_pages
        except Exception:
            _log.exception("Failed to process %s", pdf_path.name)

    elapsed_seconds = time.monotonic() - run_start_time
    report = {
        "elapsed_seconds": elapsed_seconds,
        "failed_pages": failed_pages,
        "glob": args.glob,
        "input_dir": str(input_dir),
        "max_pages": args.max_pages,
        "metrics_file": str(metrics_file) if metrics_file is not None else None,
        "mode": args.mode,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "pages_per_second": (
            processed_pages / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        ),
        "parser_threads": args.parser_threads,
        "processed_pages": processed_pages,
        "release_native_memory_every_n_pages": args.release_native_memory_every_n_pages,
        "repo_id": args.repo_id,
        "report_file": str(report_file),
        "revision": args.revision,
        "scale": args.scale,
        "target_total_pages": target_total_pages,
        "total_available_pages": total_available_pages,
    }
    _write_report(report_file, report)
    _log.info(
        "Processed %d/%d page(s) in %.2fs (%.2f pages/s). Report: %s",
        processed_pages,
        target_total_pages,
        elapsed_seconds,
        report["pages_per_second"],
        report_file,
    )


if __name__ == "__main__":
    main()
