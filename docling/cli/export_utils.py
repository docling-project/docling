import re
from pathlib import Path

from docling_core.types.doc import ImageRefMode

from docling.datamodel.base_models import OutputFormat

_OUTPUT_FORMATS_NOT_SUPPORTING_IMAGE_EMBEDDING = frozenset(
    {
        OutputFormat.TEXT,
        OutputFormat.DOCTAGS,
        OutputFormat.VTT,
        OutputFormat.DOCLANG,
        OutputFormat.DCLX,
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
    }


def _split_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return re.split(r"[;,]", raw)


def _is_empty_output(path: Path) -> bool:
    try:
        return not path.exists() or path.stat().st_size == 0
    except OSError:
        return True
