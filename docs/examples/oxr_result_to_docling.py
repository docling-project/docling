"""Import a SoMark OXR parse result into a DoclingDocument.

Two modes:

- Convert one saved OXR result JSON, also through `DocumentConverter`.
- Parse one document with a running OXR service, then convert its result.

What this example does:

- keeps OXR page geometry, element labels, and reading order;
- maps headers, footers, formulas, tables, code, and pictures;
- writes Markdown and Docling JSON.

Run with the bundled fixture:

    python docs/examples/oxr_result_to_docling.py

Run with your own saved result:

    python docs/examples/oxr_result_to_docling.py result.oxr.json

Parse and convert through a local OXR service:

    python docs/examples/oxr_result_to_docling.py document.pdf \
        --base-url http://localhost:8000/v1
"""

import argparse
from pathlib import Path

from docling.utils.oxr_utils import parse_oxr_result, run_oxr_parse

_DEFAULT_INPUT = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "json_oxr"
    / "self_authored_pages.oxr.json"
)


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
        default=_DEFAULT_INPUT,
        help="Saved OXR result JSON, or a PDF/image when --base-url is set "
        "(default: bundled synthetic fixture)",
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

    input_path: Path = args.input
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
