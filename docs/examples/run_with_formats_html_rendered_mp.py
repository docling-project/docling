import logging
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from docling_core.types.doc import DoclingDocument, FloatingItem
from tqdm import tqdm

from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption

_log = logging.getLogger(__name__)
_WORKER_CONVERTER: DocumentConverter | None = None
PARTITION_SIZE = 1_000_000
INPUT_DIR = Path("input_dir_to_html")
OUT_DIR = Path("output_dir")
KEEP_PAGE_IMAGE_IN_JSON = True

# Requires Playwright to be installed locally.


def _build_html_options(sample_source_uri: Path) -> HTMLBackendOptions:
    return HTMLBackendOptions(
        render_page=True,
        render_page_width=794,
        render_page_height=1126,
        render_device_scale=2.0,
        render_page_orientation="portrait",
        render_print_media=True,
        render_wait_until="networkidle",
        render_wait_ms=500,
        render_full_page=True,
        render_dpi=144,
        page_padding=10,
        enable_local_fetch=True,
        fetch_images=True,
        source_uri=sample_source_uri.resolve(),
    )


def _init_worker(sample_source_uri: str) -> None:
    global _WORKER_CONVERTER

    html_options = _build_html_options(Path(sample_source_uri))
    _WORKER_CONVERTER = DocumentConverter(
        format_options={
            InputFormat.HTML: HTMLFormatOption(backend_options=html_options)
        }
    )


def _convert_one(
    input_path_str: str, json_path_str: str, png_path_str: str
) -> dict[str, Any]:
    input_path = Path(input_path_str)
    json_path = Path(json_path_str)
    png_path = Path(png_path_str)
    if _WORKER_CONVERTER is None:
        raise RuntimeError("Worker not initialized")

    try:
        # Avoid duplicate work if another process already wrote this result.
        if json_path.exists():
            return {
                "ok": True,
                "file": input_path.name,
                "elapsed": 0.0,
                "skipped": True,
            }

        start = time.perf_counter()
        res = _WORKER_CONVERTER.convert(input_path)
        elapsed = time.perf_counter() - start

        doc = res.document

        json_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        # doc.save_as_json(json_path, image_mode=ImageRefMode.PLACEHOLDER)

        page = doc.pages[1]
        if page.image and page.image.pil_image:
            page.image.pil_image.save(png_path)

        if not KEEP_PAGE_IMAGE_IN_JSON:
            doc = _drop_pictures(doc)
        doc.save_as_json(json_path)

        return {
            "ok": True,
            "file": input_path.name,
            "elapsed": elapsed,
            "skipped": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "file": input_path.name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _iter_html_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        dir_path = Path(dirpath)
        for filename in filenames:
            if filename.lower().endswith(".html"):
                yield dir_path / filename


def _drop_pictures(doc: DoclingDocument):
    for item, _ in doc.iterate_items(with_groups=False, traverse_pictures=True):
        if isinstance(item, FloatingItem):
            item.image = None
    for page in doc.pages.values():
        page.image = None
    return doc


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting HTML conversion process.")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output root: {OUT_DIR}")

    sample_source_uri: Path | None = None
    total_html_files = 0
    for candidate in _iter_html_files(INPUT_DIR):
        total_html_files += 1
        if sample_source_uri is None:
            sample_source_uri = candidate
    if sample_source_uri is None:
        print(f"No HTML files found in {INPUT_DIR}")
        return

    timings: list[float] = []
    failed_files: list[Path] = []
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
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
        initializer=_init_worker,
        initargs=(str(sample_source_uri),),
    ) as executor:
        futures: dict[Any, Path] = {}
        scanned_count = 0
        submitted_count = 0
        skipped_count = 0
        success_count = 0
        started_announced = False
        first_result_announced = False
        with tqdm(
            total=0,
            desc="HTML conversions",
            unit="file",
            dynamic_ncols=True,
        ) as pbar:
            file_index = 0

            def _handle_done_future(future: Any, input_path: Path) -> None:
                nonlocal success_count, first_result_announced
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
                        tqdm.write(f"{result['file']}: skipped (already converted)")
                    else:
                        success_count += 1
                        elapsed = float(result["elapsed"])
                        timings.append(elapsed)
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
                        f"in_flight={len(futures)}"
                    )

                pbar.set_postfix(
                    scanned=scanned_count,
                    queued=submitted_count,
                    skipped=skipped_count,
                    ok=success_count,
                    failed=len(failed_files),
                    left=max(0, submitted_count - pbar.n),
                )

            for input_path in _iter_html_files(INPUT_DIR):
                scanned_count += 1
                file_index += 1
                rel_dir = input_path.parent.relative_to(INPUT_DIR)
                if use_partitions:
                    part_no = ((file_index - 1) // partition_size) + 1
                    base_root = OUT_DIR / f"part{part_no}"
                else:
                    base_root = OUT_DIR
                mirrored_root = (
                    base_root / rel_dir if rel_dir != Path(".") else base_root
                )
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
                            left=max(0, submitted_count - pbar.n),
                        )
                    if scanned_count % 100000 == 0:
                        tqdm.write(
                            "Scan update: "
                            f"scanned={scanned_count}, submitted={submitted_count}, "
                            f"skipped={skipped_count}"
                        )
                    continue

                future = executor.submit(
                    _convert_one,
                    str(input_path),
                    str(json_path),
                    str(png_path),
                )
                futures[future] = input_path
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
                        done_input_path = futures.pop(done_future)
                        _handle_done_future(done_future, done_input_path)

            while futures:
                done, _ = wait(
                    set(futures.keys()),
                    return_when=FIRST_COMPLETED,
                )
                for done_future in done:
                    done_input_path = futures.pop(done_future)
                    _handle_done_future(done_future, done_input_path)

    print(
        f"Scanned {scanned_count} files. "
        f"Submitted {submitted_count}. "
        f"Skipped existing {skipped_count}."
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
