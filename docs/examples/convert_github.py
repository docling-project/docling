# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Retrieve a complete GitHub conversation and export it as DCLX.

Install docling (or docling-slim[format-web]), install gh, and run gh auth login.
Usage: python docs/examples/convert_github.py https://github.com/OWNER/REPO/pull/1
"""

import argparse
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling.utils.github import fetch_github_conversation, github_document_stream


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    export = fetch_github_conversation(args.url, timeout=180)
    source = export.conversation.source
    name = f"{source.owner}-{source.repository}-{source.kind}-{source.number}"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{name}.github.json").write_text(
        export.model_dump_json(indent=2), encoding="utf-8"
    )
    document = (
        DocumentConverter(allowed_formats=[InputFormat.JSON_GITHUB])
        .convert(github_document_stream(export))
        .document
    )
    document.save_as_doclang_archive(
        args.output / f"{name}.dclx", add_named_groups=True
    )


if __name__ == "__main__":
    main()
