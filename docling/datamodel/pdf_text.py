from collections.abc import Iterable

from docling_core.types.doc.page import PdfCellRenderingMode, PdfTextCell, TextCell

_INVISIBLE_RENDERING_MODES = frozenset(
    {PdfCellRenderingMode.INVISIBLE, PdfCellRenderingMode.ONLY_CLIPPING}
)


def is_render_mode_invisible(cell: TextCell) -> bool:
    """Return whether a PDF text cell uses a rendering mode that paints no ink."""
    return (
        isinstance(cell, PdfTextCell)
        and cell.rendering_mode in _INVISIBLE_RENDERING_MODES
    )


def split_by_render_mode_visibility(
    cells: Iterable[TextCell],
) -> tuple[list[TextCell], list[TextCell]]:
    """Split cells into visible and render-mode-invisible lists, preserving order."""
    visible: list[TextCell] = []
    invisible: list[TextCell] = []
    for cell in cells:
        (invisible if is_render_mode_invisible(cell) else visible).append(cell)
    return visible, invisible
