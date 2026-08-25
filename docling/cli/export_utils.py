import re
from pathlib import Path

import typer
from docling_core.types.doc import ImageRefMode

from docling.datamodel.base_models import OutputFormat
from docling.datamodel.settings import PageRange

_OUTPUT_FORMATS_NOT_SUPPORTING_IMAGE_EMBEDDING = frozenset(
    {
        OutputFormat.TEXT,
        OutputFormat.DOCTAGS,
        OutputFormat.VTT,
        OutputFormat.DOCLANG,
        OutputFormat.CHUNKS,
    }
)


def _should_generate_export_images(
    image_export_mode: ImageRefMode,
    to_formats: list[OutputFormat],
) -> bool:
    return image_export_mode != ImageRefMode.PLACEHOLDER and any(
        to_format not in _OUTPUT_FORMATS_NOT_SUPPORTING_IMAGE_EMBEDDING
        for to_format in to_formats
    )


def _export_flags_from_formats(to_formats: list[OutputFormat]) -> dict[str, bool]:
    """Expand requested output formats into the per-format export booleans.

    The returned keys match the ``export_*`` keyword arguments of
    ``docling.cli.main.export_documents`` so the result can be splatted into it.
    Shared by the local ``convert`` command and the remote ``convert-remote``
    command to keep their output selection identical.
    """
    return {
        "export_json": OutputFormat.JSON in to_formats,
        "export_yaml": OutputFormat.YAML in to_formats,
        "export_html": OutputFormat.HTML in to_formats,
        "export_html_split_page": OutputFormat.HTML_SPLIT_PAGE in to_formats,
        "export_md": OutputFormat.MARKDOWN in to_formats,
        "export_txt": OutputFormat.TEXT in to_formats,
        "export_doctags": OutputFormat.DOCTAGS in to_formats,
        "export_vtt": OutputFormat.VTT in to_formats,
        "export_doclang": OutputFormat.DOCLANG in to_formats,
        "export_dclx": OutputFormat.DCLX in to_formats,
        "export_chunks": OutputFormat.CHUNKS in to_formats,
    }


def _parse_page_range(raw: str | None) -> PageRange | None:
    """Parse a ``--page-range`` value like ``1-4`` (or a single page ``4``).

    Page numbers start at 1. Returns ``None`` when no range is given so the
    caller's default (all pages) applies. Shared by the local ``convert``
    command and the remote ``convert-remote`` command.
    """
    if raw is None:
        return None
    text = raw.strip()
    try:
        if "-" in text:
            start_str, end_str = text.split("-", 1)
            start, end = int(start_str), int(end_str)
        else:
            start = end = int(text)
    except ValueError:
        raise typer.BadParameter(
            f"Invalid --page-range {raw!r}. Use START-END (e.g. 1-4) or a single page.",
        )
    if start < 1 or end < start:
        raise typer.BadParameter(
            f"Invalid --page-range {raw!r}. Page numbers start at 1 and END must be "
            ">= START.",
        )
    return (start, end)


def _split_list(raw: str | None) -> list[str] | None:
    """Split a comma/semicolon-separated CLI value, dropping blanks.

    Stripping matters: `--ocr-lang "en, de"` must yield `de`, not `" de"`.
    """
    if raw is None:
        return None
    return [item.strip() for item in re.split(r"[;,]", raw) if item.strip()]


def _is_empty_output(path: Path) -> bool:
    try:
        return not path.exists() or path.stat().st_size == 0
    except OSError:
        return True
