# %% [markdown]
# Convert a DOCX file with tracked changes (Word "Suggestions").
#
# What this example does
# - Shows the three track_changes modes: accept, reject, raw.
# - "accept" (default) includes inserted text and drops deleted text — the final document.
# - "reject" includes deleted text and drops inserted text — the original document.
# - "raw" includes both; text items carry change_type="inserted" or change_type="deleted"
#   so serializers and downstream consumers can decide how to render them.
#
# Prerequisites
# - Install Docling with DOCX support: `pip install "docling[format-docx]"`
#
# How to run
# - From the repository root: `python docs/examples/docx_track_changes.py`
#
# CLI equivalent
# - `docling my_doc.docx --docx-track-changes raw`

# %%

import logging
from pathlib import Path

from docling.datamodel.backend_options import MsWordBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, WordFormatOption

logging.basicConfig(level=logging.WARNING)


def convert(input_path: Path, mode: str) -> str:
    converter = DocumentConverter(
        format_options={
            InputFormat.DOCX: WordFormatOption(
                backend_options=MsWordBackendOptions(track_changes=mode)
            )
        }
    )
    result = converter.convert(input_path)
    return result.document.export_to_markdown()


def main() -> None:
    input_path = Path("tests/data/docx/word_sample.docx")

    for mode in ("accept", "reject", "raw"):
        md = convert(input_path, mode)
        out = Path(f"scratch/track_changes_{mode}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        print(f"[{mode}] {len(md)} chars -> {out}")


if __name__ == "__main__":
    main()
