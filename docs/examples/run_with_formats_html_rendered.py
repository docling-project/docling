import json
import logging
import time
from pathlib import Path

from docling_core.types.doc import ImageRefMode

from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption
from docling.utils.visualization import draw_clusters

_log = logging.getLogger(__name__)

# Requires Playwright to be installed locally.


def main() -> None:
    input_html_path = Path("input_dir_to_html/")
    out_dir = Path("ouput_dir/json")
    out_dir_png = Path("ouput_dir/png")
    out_dir_viz = Path("ouput_dir/viz")

    input_paths = sorted([file for file in input_html_path.iterdir() if file.is_file()])

    html_options = HTMLBackendOptions(
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
        source_uri=input_paths[0].resolve(),
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.HTML: HTMLFormatOption(backend_options=html_options)
        }
    )
    timings: list[float] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_png.mkdir(parents=True, exist_ok=True)
    out_dir_viz.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        start = time.perf_counter()
        res = converter.convert(input_path)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        print(f"Converted in {elapsed:.3f}s")

        doc = res.document
        viz_pages = doc.get_visualization()
        viz_pages2 = doc.get_visualization(viz_mode="key_value")
        print(len(viz_pages))
        with (out_dir / f"{res.input.file.stem}.json").open("w") as fp:
            fp.write(json.dumps(doc.export_to_dict()))

        page = doc.pages[1]
        if page.image and page.image.pil_image:
            page.image.pil_image.save(
                out_dir_png / f"{res.input.file.stem}_page_{1}.png"
            )

        page_viz = viz_pages[1]
        page_viz.save(out_dir_viz / f"{res.input.file.stem}_page_{1}_viz.png")

        page_viz = viz_pages2[1]
        page_viz.save(out_dir_viz / f"{res.input.file.stem}_page_{1}_viz_kvp.png")

    if timings:
        avg_time = sum(timings) / len(timings)
        print(f"Average conversion time: {avg_time:.3f}s across {len(timings)} samples")


if __name__ == "__main__":
    main()
