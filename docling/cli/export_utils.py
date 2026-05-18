import re
from pathlib import Path

from docling_core.types.doc import ImageRefMode

from docling.datamodel.base_models import OutputFormat

_OUTPUT_FORMATS_NOT_SUPPORTING_IMAGE_EMBEDDING = frozenset(
    {
        OutputFormat.TEXT,
        OutputFormat.DOCTAGS,
        OutputFormat.VTT,
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


def _split_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return re.split(r"[;,]", raw)


def _is_empty_output(path: Path) -> bool:
    try:
        return not path.exists() or path.stat().st_size == 0
    except OSError:
        return True
