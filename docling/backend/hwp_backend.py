import json
import logging
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from typing_extensions import override

from docling.backend.abstract_backend import (
    DeclarativeDocumentBackend,
    PaginatedDocumentBackend,
)
from docling.datamodel.backend_options import HwpBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class HwpDocumentBackend(DeclarativeDocumentBackend, PaginatedDocumentBackend):
    """Convert HWP/HWPX files through the external rhwp CLI.

    The current implementation consumes `rhwp export-render-tree`, which exposes
    page-local bounding boxes and text runs. This gives Docling a layout-grounded
    first pass without introducing a Python/Rust extension dependency.
    """

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        options: Optional[HwpBackendOptions] = None,
    ) -> None:
        hwp_options = options or HwpBackendOptions()
        super().__init__(
            in_doc,
            path_or_stream=path_or_stream,
            options=hwp_options,
        )
        self.hwp_options = hwp_options
        self._doc_or_err = self._get_doc_or_err()

    @override
    def is_valid(self) -> bool:
        return isinstance(self._doc_or_err, DoclingDocument)

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return True

    @override
    def page_count(self) -> int:
        if isinstance(self._doc_or_err, DoclingDocument):
            return len(self._doc_or_err.pages)
        return 0

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.HWP}

    @override
    def convert(self) -> DoclingDocument:
        if isinstance(self._doc_or_err, DoclingDocument):
            return self._doc_or_err

        raise self._doc_or_err

    def _get_doc_or_err(self) -> Union[DoclingDocument, Exception]:
        try:
            with tempfile.TemporaryDirectory(prefix="docling-hwp-") as tmp:
                tmp_path = Path(tmp)
                input_path = self._materialize_input(tmp_path)
                render_tree_dir = tmp_path / "render-tree"
                render_tree_dir.mkdir()

                self._run_rhwp_export_render_tree(input_path, render_tree_dir)
                return self._convert_render_tree_dir(render_tree_dir)
        except Exception as exc:
            return exc

    def _materialize_input(self, tmp_path: Path) -> Path:
        if isinstance(self.path_or_stream, Path):
            return self.path_or_stream

        if isinstance(self.path_or_stream, BytesIO):
            suffix = self.file.suffix if self.file.suffix in {".hwp", ".hwpx"} else ".hwp"
            input_path = tmp_path / f"input{suffix}"
            input_path.write_bytes(self.path_or_stream.getvalue())
            return input_path

        raise RuntimeError(f"Unexpected HWP input type: {type(self.path_or_stream)!r}")

    def _run_rhwp_export_render_tree(self, input_path: Path, output_dir: Path) -> None:
        cmd = [
            self.hwp_options.rhwp_binary,
            "export-render-tree",
            str(input_path),
            "-o",
            str(output_dir),
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                check=False,
                text=True,
                timeout=self.hwp_options.export_timeout,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "The rhwp executable was not found. Install rhwp or set "
                "HwpBackendOptions.rhwp_binary to its path."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "rhwp export-render-tree timed out after "
                f"{self.hwp_options.export_timeout}s"
            ) from exc

        if completed.returncode != 0:
            err = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"rhwp export-render-tree failed: {err}")

    def _convert_render_tree_dir(self, render_tree_dir: Path) -> DoclingDocument:
        paths = sorted(render_tree_dir.glob("render_tree_*.json"))
        if not paths:
            raise RuntimeError("rhwp did not produce any render-tree JSON files.")

        doc = DoclingDocument(name=self.file.stem or "file")
        doc.origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/octet-stream",
            binary_hash=_document_hash_as_int(self.document_hash),
        )

        for page_no, path in enumerate(paths, start=1):
            root = json.loads(path.read_text(encoding="utf-8"))
            bbox = root.get("bbox", {})
            page_w = _float_or_default(bbox.get("w"), 0.0)
            page_h = _float_or_default(bbox.get("h"), 0.0)
            doc.add_page(page_no=page_no, size=Size(width=page_w, height=page_h))
            self._add_nodes(doc, root.get("children", []), page_no=page_no)

        return doc

    def _add_nodes(
        self,
        doc: DoclingDocument,
        nodes: list[dict[str, Any]],
        *,
        page_no: int,
        furniture_label: Optional[DocItemLabel] = None,
    ) -> None:
        for node in nodes:
            node_type = node.get("type")
            if node_type == "Header":
                self._add_nodes(
                    doc,
                    node.get("children", []),
                    page_no=page_no,
                    furniture_label=DocItemLabel.PAGE_HEADER,
                )
            elif node_type == "Footer":
                self._add_nodes(
                    doc,
                    node.get("children", []),
                    page_no=page_no,
                    furniture_label=DocItemLabel.PAGE_FOOTER,
                )
            elif node_type == "Table":
                self._add_table(doc, node, page_no=page_no)
            elif node_type == "Image":
                doc.add_picture(prov=_provenance(node, page_no))
            elif node_type == "Equation":
                doc.add_formula(text="", prov=_provenance(node, page_no))
            elif node_type == "TextLine":
                text = _node_text(node)
                if text:
                    label = furniture_label or DocItemLabel.TEXT
                    doc.add_text(
                        label=label,
                        text=text,
                        prov=_provenance(node, page_no, text=text),
                        content_layer=ContentLayer.FURNITURE
                        if furniture_label
                        else None,
                    )
            elif node_type == "TextRun":
                text = str(node.get("text", ""))
                if text:
                    doc.add_text(
                        label=furniture_label or DocItemLabel.TEXT,
                        text=text,
                        prov=_provenance(node, page_no, text=text),
                        content_layer=ContentLayer.FURNITURE
                        if furniture_label
                        else None,
                    )
            else:
                self._add_nodes(
                    doc,
                    node.get("children", []),
                    page_no=page_no,
                    furniture_label=furniture_label,
                )

    def _add_table(
        self, doc: DoclingDocument, node: dict[str, Any], *, page_no: int
    ) -> None:
        cells = _table_cells(node, page_no=page_no)
        num_rows = max(
            _int_or_default(node.get("rows"), 0),
            max((cell.end_row_offset_idx for cell in cells), default=0),
        )
        num_cols = max(
            _int_or_default(node.get("cols"), 0),
            max((cell.end_col_offset_idx for cell in cells), default=0),
        )
        doc.add_table(
            data=TableData(table_cells=cells, num_rows=num_rows, num_cols=num_cols),
            prov=_provenance(node, page_no),
        )


def _table_cells(node: dict[str, Any], *, page_no: int) -> list[TableCell]:
    cells: list[TableCell] = []
    for cell_node in _iter_nodes(node):
        if cell_node.get("type") != "Cell":
            continue
        row = _int_or_default(cell_node.get("row"), 0)
        col = _int_or_default(cell_node.get("col"), 0)
        cells.append(
            TableCell(
                bbox=_bbox(cell_node),
                text=_node_text(cell_node),
                row_span=1,
                col_span=1,
                start_row_offset_idx=row,
                end_row_offset_idx=row + 1,
                start_col_offset_idx=col,
                end_col_offset_idx=col + 1,
            )
        )
    return cells


def _iter_nodes(node: dict[str, Any]):
    yield node
    for child in node.get("children", []):
        yield from _iter_nodes(child)


def _node_text(node: dict[str, Any]) -> str:
    if node.get("type") == "TextRun":
        return str(node.get("text", ""))
    return "".join(_node_text(child) for child in node.get("children", []))


def _provenance(
    node: dict[str, Any], page_no: int, *, text: str = ""
) -> Optional[ProvenanceItem]:
    bbox = _bbox(node)
    if bbox is None:
        return None
    return ProvenanceItem(page_no=page_no, bbox=bbox, charspan=(0, len(text)))


def _bbox(node: dict[str, Any]) -> Optional[BoundingBox]:
    raw = node.get("bbox")
    if not isinstance(raw, dict):
        return None

    x = _float_or_default(raw.get("x"), 0.0)
    y = _float_or_default(raw.get("y"), 0.0)
    w = _float_or_default(raw.get("w"), 0.0)
    h = _float_or_default(raw.get("h"), 0.0)
    return BoundingBox.from_tuple((x, y, x + w, y + h), origin=CoordOrigin.TOPLEFT)


def _document_hash_as_int(value: str) -> int:
    try:
        return int(value, 16) & 0xFFFFFFFFFFFFFFFF
    except ValueError:
        return 0


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
