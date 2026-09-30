"""Import a saved PaddleOCR-VL result into a DoclingDocument.

The external PaddleOCR-VL pipeline must already have run. This example neither
imports Paddle/PaddleX nor downloads or executes a model. It targets the
PaddleOCR-VL 1.6 result schema produced by PaddleX 3.7.2, accepting either a
bare single-page payload or the ``{"res": payload}`` wrapper written by
the ``result.json`` property. ``save_to_json()`` writes the bare payload, and
the official hosted service's per-page ``prunedResult`` is also accepted.

Current adapter boundaries:

- geometry refers to the processed-page pixel canvas;
- polygon shapes are represented by their rectangular ``block_bbox``;
- one page is converted per call, without Paddle ``restructure_pages()`` parity;
- ordinary saved JSON does not embed page or crop image bytes;
- provider-added Markdown heading and centered-caption wrappers are normalized
  when ``model_settings.format_block_content`` is enabled;
- nonempty picture/chart/seal ``block_content`` is preserved as adjacent text,
  because provider switches can make it OCR text, chart rows, or markup rather
  than a natural-language picture description.

Run with the bundled official-service fixture:

    python docs/examples/paddleocr_vl_result_to_docling.py

Run with your own saved result:

    python docs/examples/paddleocr_vl_result_to_docling.py result.json
"""

import argparse
from pathlib import Path

from docling.utils.paddleocr_vl_utils import parse_paddleocr_vl_result

_DEFAULT_INPUT = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "json_paddleocr_vl"
    / "self_authored_page.paddleocr-vl-1.6.aistudio-pruned.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a saved PaddleOCR-VL 1.6 / PaddleX 3.7.2 single-page "
            "JSON result to Markdown and Docling JSON."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=_DEFAULT_INPUT,
        help=(
            "Path to the saved result JSON "
            "(default: bundled official AI Studio fixture)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scratch"),
        help="Output directory (default: scratch)",
    )
    parser.add_argument(
        "--filename",
        help="Override the source filename stored in the DoclingDocument",
    )
    parser.add_argument(
        "--page-no",
        type=int,
        help="Override the 1-based page number stored in provenance",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    document = parse_paddleocr_vl_result(
        input_path.read_text(encoding="utf-8"),
        filename=args.filename,
        page_no=args.page_no,
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{input_path.stem}.md"
    docling_json_path = output_dir / f"{input_path.stem}.docling.json"

    markdown_path.write_text(document.export_to_markdown(), encoding="utf-8")
    document.save_as_json(docling_json_path)

    print(f"Markdown: {markdown_path}")
    print(f"Docling JSON: {docling_json_path}")


if __name__ == "__main__":
    main()
