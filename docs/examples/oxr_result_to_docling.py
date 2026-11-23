"""Import a SoMark OXR parse result into a DoclingDocument.

Two modes:

- Convert one saved OXR result JSON, also through `DocumentConverter`.
- Parse one document with a running OXR service, then convert its result.

What this example does:

- keeps OXR page geometry, element labels, and reading order;
- maps headers, footers, formulas, tables, code, and pictures;
- writes Markdown and Docling JSON.

Run with the built-in sample:

    python docs/examples/oxr_result_to_docling.py

Run with your own saved result:

    python docs/examples/oxr_result_to_docling.py result.oxr.json

Parse and convert through a local OXR service:

    python docs/examples/oxr_result_to_docling.py document.pdf \
        --base-url http://localhost:8000/v1
"""

import argparse
import json
import tempfile
from pathlib import Path

from docling.utils.oxr_utils import parse_oxr_result, run_oxr_parse

_SAMPLE_RESULT = {
    "code": 0,
    "message": "success",
    "warnings": [],
    "data": {
        "task_id": "example",
        "error": None,
        "metadata": {},
        "result": {
            "file_name": "sample.pdf",
            "metadata": {"page_num": 1, "file_type": ".pdf"},
            "outputs": {
                "markdown": "",
                "json": {
                    "pages": [
                        {
                            "page_num": 0,
                            "page_size": {"w": 800, "h": 1000},
                            "blocks": [
                                {
                                    "type": "Title",
                                    "bbox": [40, 40, 760, 100],
                                    "content": "OXR Example",
                                    "format": "text",
                                },
                                {
                                    "type": "Text",
                                    "bbox": [40, 120, 760, 220],
                                    "content": (
                                        "A saved OXR parse result converts without "
                                        "a running OXR service."
                                    ),
                                    "format": "text",
                                },
                            ],
                        }
                    ]
                },
            },
        },
    },
}


def _write_sample(directory: Path) -> Path:
    """Write the built-in OXR result next to the caller's output."""
    path = directory / "sample.oxr.json"
    path.write_text(json.dumps(_SAMPLE_RESULT, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a saved OXR parse result, or an OXR service response, to "
            "Markdown and Docling JSON."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Saved OXR result JSON, or a PDF/image when --base-url is set "
        "(default: a small built-in sample)",
    )
    parser.add_argument(
        "--base-url",
        help="Parse the input with this OXR service first, for example "
        "http://localhost:8000/v1",
    )
    parser.add_argument(
        "--api-key",
        default="dummy",
        help="OXR api_key form field (default: dummy; self-hosted OXR accepts any value)",
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
    parser.add_argument(
        "--load-local-images",
        action="store_true",
        help="Read picture files named by the OXR result, relative to the JSON file",
    )
    args = parser.parse_args()

    sample_dir = tempfile.TemporaryDirectory()
    input_path: Path = (
        args.input if args.input is not None else _write_sample(Path(sample_dir.name))
    )
    if args.base_url is not None:
        result = run_oxr_parse(input_path, base_url=args.base_url, api_key=args.api_key)
        document = parse_oxr_result(
            result,
            filename=args.filename or input_path.name,
            page_no=args.page_no,
        )
    else:
        document = parse_oxr_result(
            input_path.read_bytes(),
            filename=args.filename,
            page_no=args.page_no,
            load_local_images=args.load_local_images,
            base_dir=input_path.parent,
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
