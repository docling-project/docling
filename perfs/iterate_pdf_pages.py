"""Iterate over PDFs in a directory, load them with
ThreadedDoclingParseDocumentBackend, and extract text cells and page images for
every page."""

import argparse
import contextlib
import gc
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psutil
from docling_parse.pdf_parser import PageMaterializationConfig, RenderConfig
from docling_parse.pdf_parsers import DecodePageConfig

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.settings import DocumentLimits

_log = logging.getLogger(__name__)


_PROC = psutil.Process(os.getpid())
DEFAULT_REPO_ID = "docling-project/performance-dataset-bo767"
DEFAULT_THREADS = "1,2,4,8,12,16"


class _DefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default in (None, argparse.SUPPRESS):
            return action.help or ""
        return super()._get_help_string(action)


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


def _parse_threads(value: str) -> list[int]:
    thread_counts: list[int] = []
    for raw_count in value.split(","):
        count_text = raw_count.strip()
        if not count_text:
            raise argparse.ArgumentTypeError(
                "thread counts must be comma-separated positive integers"
            )
        try:
            count = int(count_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid thread count {count_text!r}"
            ) from exc
        if count <= 0:
            raise argparse.ArgumentTypeError(
                f"thread count must be positive, got {count}"
            )
        thread_counts.append(count)
    return thread_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_DefaultsHelpFormatter,
        add_help=False,
    )
    input_group = parser.add_argument_group("input source (choose one)")
    source_group = input_group.add_mutually_exclusive_group()
    source_group.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        help="Local directory containing PDF files to process.",
    )
    source_group.add_argument(
        "-r",
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=(
            "Hugging Face dataset repo ID. The script downloads the dataset "
            "snapshot and iterates over its 'pdf/' subfolder."
        ),
    )
    options_group = parser.add_argument_group("options")
    options_group.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )
    options_group.add_argument(
        "--revision",
        default=None,
        help=(
            "Hugging Face dataset revision, branch, or commit. "
            "Default: use the dataset default revision."
        ),
    )
    options_group.add_argument(
        "--mode",
        choices=("throughput", "memory"),
        default="throughput",
        help=(
            "Run either a throughput measurement or a memory profiling run. "
            "Memory mode writes the JSONL file consumed by plot_memory_metrics.py."
        ),
    )
    options_group.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. If provided, page images (PNG) and "
            "text cells (JSON) are written here, one subdirectory per PDF."
            " Default: do not write page outputs."
        ),
    )
    options_group.add_argument(
        "-s",
        "--scale",
        type=float,
        default=1,
        help="Scale factor for rendered page images.",
    )
    options_group.add_argument(
        "--glob",
        default="*.pdf",
        help="Glob pattern to match files in input_dir.",
    )
    options_group.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help=(
            "Path to the page-count cache JSON (default: "
            "'<input_dir>/.docling_page_counts.json'). Entries are keyed by "
            "absolute path and validated against file size and mtime."
        ),
    )
    options_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the page-count cache and always re-parse to count pages.",
    )
    options_group.add_argument(
        "--threads",
        type=_parse_threads,
        default=_parse_threads(DEFAULT_THREADS),
        metavar="THREADS",
        help=(
            "Comma-separated list of docling-parse parser thread counts to test. "
            f"Default: {DEFAULT_THREADS}."
        ),
    )
    options_group.add_argument(
        "--release-native-memory-every-n-pages",
        type=int,
        default=128,
        help="Release native parser memory after every N decoded pages.",
    )
    options_group.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Global limit for the total number of pages to iterate. Default: no limit.",
    )
    options_group.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("memory-metrics.jsonl"),
        help=(
            "JSONL file for memory mode. This remains compatible with "
            "plot_memory_metrics.py."
        ),
    )
    options_group.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help=(
            "JSON summary report path. Default: "
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


def _format_config_table(rows: list[tuple[str, object]]) -> str:
    parameter_width = max(len(parameter) for parameter, _ in rows)
    value_width = max(len(str(value)) for _, value in rows)
    lines = [
        f"{'parameter':<{parameter_width}}  {'value':<{value_width}}",
        f"{'-' * parameter_width}  {'-' * value_width}",
    ]
    lines.extend(
        f"{parameter:<{parameter_width}}  {value}" for parameter, value in rows
    )
    return "\n".join(lines)


def _make_display_decode_config(
    release_native_memory_every_n_pages: int,
) -> DecodePageConfig:
    config = DecodePageConfig()
    config.keep_char_cells = True
    config.keep_shapes = True
    config.keep_bitmaps = True
    config.create_word_cells = True
    config.create_line_cells = True
    config.enforce_same_font = True
    config.release_native_memory_every_n_pages = release_native_memory_every_n_pages

    # docling-parse moved this setting between config objects across versions.
    try:
        config.materialize_bitmap_bytes = False
    except AttributeError:
        pass

    return config


def _log_docling_parse_config(
    *,
    document_count: int,
    total_pages: int,
    mode: str,
    thread_counts: list[int],
    release_native_memory_every_n_pages: int,
    scale: float,
) -> None:
    decode_config = _make_display_decode_config(
        release_native_memory_every_n_pages=release_native_memory_every_n_pages,
    )
    materialization_config = PageMaterializationConfig()
    materialization_config.materialize_char_cells = False
    materialization_config.materialize_word_cells = True
    materialization_config.materialize_line_cells = True
    materialization_config.materialize_shapes = False
    materialization_config.materialize_bitmaps = True
    materialization_config.materialize_bitmap_bytes = False
    render_config = RenderConfig()
    render_config.scale = scale

    _log.info("Benchmark: %d documents, %d total pages", document_count, total_pages)
    _log.info("Mode: %s", mode)
    _log.info("Thread counts to test: %s", thread_counts)
    _log.info("Render scale: %s", scale)
    _log.info(
        "Decode config:\n%s",
        _format_config_table(
            [
                ("page_boundary", decode_config.page_boundary),
                ("do_sanitization", decode_config.do_sanitization),
                ("keep_char_cells", decode_config.keep_char_cells),
                ("keep_shapes", decode_config.keep_shapes),
                ("keep_bitmaps", decode_config.keep_bitmaps),
                ("max_num_lines", decode_config.max_num_lines),
                ("max_num_bitmaps", decode_config.max_num_bitmaps),
                ("create_word_cells", decode_config.create_word_cells),
                ("create_line_cells", decode_config.create_line_cells),
                ("enforce_same_font", decode_config.enforce_same_font),
                (
                    "horizontal_cell_tolerance",
                    decode_config.horizontal_cell_tolerance,
                ),
                (
                    "word_space_width_factor_for_merge",
                    decode_config.word_space_width_factor_for_merge,
                ),
                (
                    "line_space_width_factor_for_merge",
                    decode_config.line_space_width_factor_for_merge,
                ),
                (
                    "line_space_width_factor_for_merge_with_space",
                    decode_config.line_space_width_factor_for_merge_with_space,
                ),
                ("do_thread_safe", decode_config.do_thread_safe),
                (
                    "release_native_memory_every_n_pages",
                    decode_config.release_native_memory_every_n_pages,
                ),
                ("keep_glyphs", decode_config.keep_glyphs),
                ("keep_qpdf_warnings", decode_config.keep_qpdf_warnings),
            ]
        ),
    )
    _log.info(
        "Materialization config:\n%s",
        _format_config_table(
            [
                (
                    "materialize_char_cells",
                    materialization_config.materialize_char_cells,
                ),
                (
                    "materialize_word_cells",
                    materialization_config.materialize_word_cells,
                ),
                (
                    "materialize_line_cells",
                    materialization_config.materialize_line_cells,
                ),
                ("materialize_shapes", materialization_config.materialize_shapes),
                ("materialize_bitmaps", materialization_config.materialize_bitmaps),
                (
                    "materialize_bitmap_bytes",
                    materialization_config.materialize_bitmap_bytes,
                ),
            ]
        ),
    )
    _log.info(
        "Render config:\n%s",
        _format_config_table(
            [
                ("render_text", render_config.render_text),
                ("draw_text_bbox", render_config.draw_text_bbox),
                ("draw_text_basepoint", render_config.draw_text_basepoint),
                (
                    "fit_glyph_bbox_to_target",
                    render_config.fit_glyph_bbox_to_target,
                ),
                ("resolve_fonts", render_config.resolve_fonts),
                ("font_similarity_cutoff", render_config.font_similarity_cutoff),
                ("scale", render_config.scale),
                ("canvas_width", render_config.canvas_width),
                ("canvas_height", render_config.canvas_height),
            ]
        ),
    )


@contextlib.contextmanager
def _suppress_huggingface_output() -> Iterator[None]:
    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    progress_bars_were_disabled = are_progress_bars_disabled()
    previous_levels = {
        logger_name: logging.getLogger(logger_name).level
        for logger_name in ("huggingface_hub", "httpx")
    }

    disable_progress_bars()
    for logger_name in previous_levels:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    try:
        yield
    finally:
        if not progress_bars_were_disabled:
            enable_progress_bars()
        for logger_name, level in previous_levels.items():
            logging.getLogger(logger_name).setLevel(level)


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
    with _suppress_huggingface_output():
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


def _metrics_file_for_thread_count(
    metrics_file: Path | None,
    thread_count: int,
    multiple_thread_counts: bool,
) -> Path | None:
    if metrics_file is None or not multiple_thread_counts:
        return metrics_file
    return metrics_file.with_name(
        f"{metrics_file.stem}-threads-{thread_count}{metrics_file.suffix}"
    )


def _format_duration(seconds: float) -> str:
    seconds_int = max(0, round(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


def _estimated_total_duration(
    *,
    start_time: float,
    completed: int,
    total: int,
) -> str:
    if completed <= 0 or total <= 0:
        return "?"
    elapsed_seconds = time.monotonic() - start_time
    return _format_duration(elapsed_seconds / (completed / total))


class _ThroughputProgressBar:
    def __init__(self, *, total: int, desc: str, width: int = 39) -> None:
        self.total = total
        self.desc = desc
        self.width = width
        self.n = 0
        self.start_time = time.monotonic()

    def __enter__(self) -> "_ThroughputProgressBar":
        self._write()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    def update(self, increment: int = 1) -> None:
        self.n += increment
        self._write()

    def _write(self) -> None:
        percentage = (self.n / self.total * 100) if self.total else 0.0
        filled = round(self.width * self.n / self.total) if self.total else 0
        bar = "#" * filled + " " * (self.width - filled)
        elapsed_seconds = time.monotonic() - self.start_time
        rate_text = (
            f"{self.n / elapsed_seconds:.1f}/s"
            if self.n > 0 and elapsed_seconds
            else "?/s"
        )
        total_text = _estimated_total_duration(
            start_time=self.start_time,
            completed=self.n,
            total=self.total,
        )
        sys.stdout.write(
            "\r"
            f"  {self.desc}: [{bar}] {self.n}/{self.total} {percentage:3.1f}% "
            f"{rate_text} elapsed: {_format_duration(elapsed_seconds)} [sec] "
            f"total: {total_text} [sec]"
        )
        sys.stdout.flush()


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
    verbose: bool = False,
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
            if verbose:
                _log.info("  %s: %d page(s) [cached]", pdf_path.name, page_count)
        else:
            if verbose:
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
            if verbose:
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
    progress_bar: Any | None = None,
) -> tuple[int, int]:
    if mode == "memory":
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
        if mode == "memory":
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
                if mode == "memory":
                    _log.exception(
                        "  page %d/%d: failed to parse/render", page_no, num_pages
                    )
            finally:
                if progress_bar is not None:
                    progress_bar.update(1)
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
            if mode == "memory":
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
        format="%(message)s",
    )
    args = parse_args()
    input_dir = _resolve_input_dir(args)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

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
        args.threads[0],
        args.release_native_memory_every_n_pages,
        verbose=args.mode == "memory",
    )
    total_available_pages = sum(page_count for _, page_count in ordered)
    target_total_pages = (
        min(total_available_pages, args.max_pages)
        if args.max_pages is not None
        else total_available_pages
    )
    _log_docling_parse_config(
        document_count=len(ordered),
        total_pages=total_available_pages,
        mode=args.mode,
        thread_counts=args.threads,
        release_native_memory_every_n_pages=args.release_native_memory_every_n_pages,
        scale=args.scale,
    )

    if args.mode == "memory":
        _log.info("Processing order (by descending page count):")
        for pdf_path, page_count in ordered:
            _log.info("  %5d pages  %s", page_count, pdf_path.name)

    runs: list[dict[str, Any]] = []
    multiple_thread_counts = len(args.threads) > 1
    for thread_count in args.threads:
        run_start_time = time.monotonic()
        metrics_file = _metrics_file_for_thread_count(
            args.metrics_file if args.mode == "memory" else None,
            thread_count,
            multiple_thread_counts,
        )
        if metrics_file is not None:
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            metrics_file.write_text("", encoding="utf-8")
            _log.info("Writing per-page metrics to %s", metrics_file)

        print(f"Running threaded renderer with {thread_count} threads ...", flush=True)
        processed_pages = 0
        failed_pages = 0
        with _ThroughputProgressBar(
            total=target_total_pages,
            desc="rendering",
        ) as progress_bar:
            for pdf_path, _ in ordered:
                if args.max_pages is not None and processed_pages >= args.max_pages:
                    break
                try:
                    processed_pages, pdf_failed_pages = process_pdf(
                        pdf_path,
                        args.output_dir,
                        args.scale,
                        thread_count,
                        args.release_native_memory_every_n_pages,
                        processed_pages=processed_pages,
                        target_total_pages=target_total_pages,
                        max_pages=args.max_pages,
                        mode=args.mode,
                        metrics_file=metrics_file,
                        run_start_time=run_start_time,
                        progress_bar=progress_bar,
                    )
                    failed_pages += pdf_failed_pages
                except Exception:
                    failed_pages += 1
                    if args.mode == "memory":
                        _log.exception("Failed to process %s", pdf_path.name)

        elapsed_seconds = time.monotonic() - run_start_time
        run_report = {
            "elapsed_seconds": elapsed_seconds,
            "failed_pages": failed_pages,
            "metrics_file": str(metrics_file) if metrics_file is not None else None,
            "pages_per_second": (
                processed_pages / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
            ),
            "processed_pages": processed_pages,
            "threads": thread_count,
        }
        runs.append(run_report)
        error_text = f" ({failed_pages} errors)" if failed_pages else ""
        print(f"  threads={thread_count}: {elapsed_seconds:.3f}s{error_text}", flush=True)

    report = {
        "glob": args.glob,
        "input_dir": str(input_dir),
        "max_pages": args.max_pages,
        "mode": args.mode,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "release_native_memory_every_n_pages": args.release_native_memory_every_n_pages,
        "repo_id": args.repo_id,
        "report_file": str(report_file),
        "revision": args.revision,
        "runs": runs,
        "scale": args.scale,
        "target_total_pages": target_total_pages,
        "threads": args.threads,
        "total_available_pages": total_available_pages,
    }
    _write_report(report_file, report)
    print(f"Wrote summary report to {report_file}", flush=True)


if __name__ == "__main__":
    main()
