import multiprocessing as mp
import os
import json
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from docling_core.types.doc import ImageRefMode
from tqdm import tqdm

from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption
from docling.utils.visualization import draw_clusters

_log = logging.getLogger(__name__)
_WORKER_CONVERTER: DocumentConverter | None = None
_WORKER_OUT_DIR: Path | None = None
_WORKER_OUT_DIR_PNG: Path | None = None
_WORKER_OUT_DIR_VIZ: Path | None = None

# Requires Playwright to be installed locally.


def _build_html_options(sample_source_uri: Path) -> HTMLBackendOptions:
    return HTMLBackendOptions(
        render_page=True,
        # render_page_width=1588,
        # ender_page_height=2246,
        render_page_width=794,
        render_page_height=100,
        render_device_scale=2.0,
        # render_page_height=1123,
        render_page_orientation="portrait",
        render_print_media=True,
        render_wait_until="networkidle",
        render_wait_ms=500,
        render_full_page=True,
        render_dpi=144,
        page_padding=16,
        enable_local_fetch=True,
        fetch_images=True,
        source_uri=sample_source_uri.resolve(),
    )


def _done_marker_path(input_path: Path, out_dir: Path) -> Path:
    return out_dir / f"{input_path.stem}.done"


def _is_already_converted(input_path: Path, out_dir: Path) -> bool:
    # Keep legacy JSON-only skip behavior and add a dedicated completion marker for MT runs.
    return _done_marker_path(input_path, out_dir).exists() or (
        out_dir / f"{input_path.stem}.json"
    ).exists()


def _init_worker(
    sample_source_uri: str, out_dir: str, out_dir_png: str, out_dir_viz: str
) -> None:
    global _WORKER_CONVERTER, _WORKER_OUT_DIR, _WORKER_OUT_DIR_PNG, _WORKER_OUT_DIR_VIZ

    _WORKER_OUT_DIR = Path(out_dir)
    _WORKER_OUT_DIR_PNG = Path(out_dir_png)
    _WORKER_OUT_DIR_VIZ = Path(out_dir_viz)
    html_options = _build_html_options(Path(sample_source_uri))
    _WORKER_CONVERTER = DocumentConverter(
        format_options={
            InputFormat.HTML: HTMLFormatOption(backend_options=html_options)
        }
    )


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"
    tmp_path.write_text(text)
    tmp_path.replace(path)


def _convert_one(input_path_str: str) -> dict[str, Any]:
    input_path = Path(input_path_str)
    if (
        _WORKER_CONVERTER is None
        or _WORKER_OUT_DIR is None
        or _WORKER_OUT_DIR_PNG is None
        or _WORKER_OUT_DIR_VIZ is None
    ):
        raise RuntimeError("Worker not initialized")

    try:
        start = time.perf_counter()
        res = _WORKER_CONVERTER.convert(input_path)
        elapsed = time.perf_counter() - start

        doc = res.document
        viz_pages = doc.get_visualization()
        viz_pages2 = doc.get_visualization(viz_mode="key_value")

        stem = res.input.file.stem
        json_path = _WORKER_OUT_DIR / f"{stem}.json"
        _write_text_atomic(json_path, json.dumps(doc.export_to_dict()))

        page = doc.pages[1]
        if page.image and page.image.pil_image:
            page.image.pil_image.save(_WORKER_OUT_DIR_PNG / f"{stem}_page_{1}.png")

        page_viz = viz_pages[1]
        page_viz.save(_WORKER_OUT_DIR_VIZ / f"{stem}_page_{1}_viz.png")

        page_viz = viz_pages2[1]
        page_viz.save(_WORKER_OUT_DIR_VIZ / f"{stem}_page_{1}_viz_kvp.png")

        _write_text_atomic(_done_marker_path(input_path, _WORKER_OUT_DIR), "ok\n")
        return {
            "ok": True,
            "file": input_path.name,
            "elapsed": elapsed,
            "viz_pages": len(viz_pages),
        }
    except Exception as exc:
        return {
            "ok": False,
            "file": input_path.name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    input_html_path = Path("input_dir_to_html/")
    out_dir = Path("ouput_dir/json")
    out_dir_png = Path("ouput_dir/png")
    out_dir_viz = Path("ouput_dir/viz")

    input_paths = sorted([file for file in input_html_path.iterdir() if file.is_file()])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_png.mkdir(parents=True, exist_ok=True)
    out_dir_viz.mkdir(parents=True, exist_ok=True)

    if not input_paths:
        print(f"No input files found in {input_html_path}")
        return

    pending_input_paths = [
        input_path
        for input_path in input_paths
        if not _is_already_converted(input_path, out_dir)
    ]
    skipped_count = len(input_paths) - len(pending_input_paths)

    print(
        f"Found {len(input_paths)} files. "
        f"Skipping {skipped_count} already converted. "
        f"Remaining: {len(pending_input_paths)}."
    )

    if not pending_input_paths:
        return

    timings: list[float] = []
    failed_files: list[Path] = []
    max_workers = min(4, max(1, int(os.environ.get("DOCLING_HTML_WORKERS", os.cpu_count() or 1))))
    print(f"Using {max_workers} worker process(es)")

    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
        initializer=_init_worker,
        initargs=(
            str(pending_input_paths[0]),
            str(out_dir),
            str(out_dir_png),
            str(out_dir_viz),
        ),
    ) as executor:
        futures = {
            executor.submit(_convert_one, str(input_path)): input_path
            for input_path in pending_input_paths
        }

        success_count = 0
        with tqdm(
            total=len(pending_input_paths),
            desc="HTML conversions",
            unit="file",
        ) as pbar:
            for future in as_completed(futures):
                input_path = futures[future]
                pbar.update(1)
                try:
                    result = future.result()
                except Exception as exc:
                    failed_files.append(input_path)
                    _log.exception("Worker crashed for %s: %s", input_path, exc)
                    tqdm.write(f"{input_path.name}: FAILED (worker crash: {exc})")
                    pbar.set_postfix(
                        ok=success_count,
                        failed=len(failed_files),
                        left=len(pending_input_paths) - pbar.n,
                    )
                    continue

                if result.get("ok"):
                    success_count += 1
                    elapsed = float(result["elapsed"])
                    timings.append(elapsed)
                    tqdm.write(
                        f"{result['file']}: converted in {elapsed:.3f}s "
                        f"({result['viz_pages']} viz pages)"
                    )
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

                pbar.set_postfix(
                    ok=success_count,
                    failed=len(failed_files),
                    left=len(pending_input_paths) - pbar.n,
                )

    if timings:
        avg_time = sum(timings) / len(timings)
        print(f"Average conversion time: {avg_time:.3f}s across {len(timings)} samples")
    if failed_files:
        print(f"Failed files: {len(failed_files)}")
        for failed_path in failed_files:
            print(f" - {failed_path}")


if __name__ == "__main__":
    main()
