import json
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path

from docling_core.types.doc import CoordOrigin, DocItemLabel, TableItem

from docling.datamodel.backend_options import HwpBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import _DocumentConversionInput
from docling.document_converter import DocumentConverter, HwpFormatOption


def _fake_render_tree() -> dict:
    return {
        "type": "Page",
        "bbox": {"x": 0, "y": 0, "w": 600, "h": 800},
        "children": [
            {
                "type": "Body",
                "bbox": {"x": 40, "y": 50, "w": 520, "h": 700},
                "children": [
                    {
                        "type": "TextLine",
                        "bbox": {"x": 50, "y": 80, "w": 180, "h": 20},
                        "children": [
                            {
                                "type": "TextRun",
                                "bbox": {"x": 50, "y": 80, "w": 90, "h": 20},
                                "text": "Hello ",
                                "pi": 0,
                            },
                            {
                                "type": "TextRun",
                                "bbox": {"x": 140, "y": 80, "w": 90, "h": 20},
                                "text": "HWP",
                                "pi": 0,
                            },
                        ],
                    },
                    {
                        "type": "Table",
                        "bbox": {"x": 50, "y": 130, "w": 200, "h": 40},
                        "rows": 1,
                        "cols": 2,
                        "children": [
                            {
                                "type": "Cell",
                                "bbox": {"x": 50, "y": 130, "w": 100, "h": 40},
                                "row": 0,
                                "col": 0,
                                "children": [
                                    {
                                        "type": "TextRun",
                                        "bbox": {
                                            "x": 60,
                                            "y": 140,
                                            "w": 20,
                                            "h": 16,
                                        },
                                        "text": "A",
                                    }
                                ],
                            },
                            {
                                "type": "Cell",
                                "bbox": {"x": 150, "y": 130, "w": 100, "h": 40},
                                "row": 0,
                                "col": 1,
                                "children": [
                                    {
                                        "type": "TextRun",
                                        "bbox": {
                                            "x": 160,
                                            "y": 140,
                                            "w": 20,
                                            "h": 16,
                                        },
                                        "text": "B",
                                    }
                                ],
                            },
                        ],
                    },
                ],
            }
        ],
    }


def _patch_rhwp(monkeypatch):
    def fake_run(cmd, **kwargs):
        assert cmd[:2] == ["rhwp-test", "export-render-tree"]
        output_dir = Path(cmd[cmd.index("-o") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "render_tree_001.json").write_text(
            json.dumps(_fake_render_tree()), encoding="utf-8"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("docling.backend.hwp_backend.subprocess.run", fake_run)


def _converter() -> DocumentConverter:
    return DocumentConverter(
        allowed_formats=[InputFormat.HWP],
        format_options={
            InputFormat.HWP: HwpFormatOption(
                backend_options=HwpBackendOptions(rhwp_binary="rhwp-test")
            )
        },
    )


def test_hwp_backend_converts_render_tree_from_path(tmp_path: Path, monkeypatch):
    _patch_rhwp(monkeypatch)
    doc_path = tmp_path / "sample.hwp"
    doc_path.write_bytes(b"fake-hwp")

    result = _converter().convert(doc_path)

    assert result.input.format == InputFormat.HWP
    assert result.document.pages[1].size.width == 600
    assert result.document.pages[1].size.height == 800
    assert result.document.texts[0].label == DocItemLabel.TEXT
    assert result.document.texts[0].text == "Hello HWP"
    prov = result.document.texts[0].prov[0]
    assert prov.page_no == 1
    assert prov.bbox.coord_origin == CoordOrigin.TOPLEFT
    assert prov.bbox.as_tuple() == (50.0, 80.0, 230.0, 100.0)

    assert len(result.document.tables) == 1
    table = result.document.tables[0]
    assert isinstance(table, TableItem)
    assert table.data.num_rows == 1
    assert table.data.num_cols == 2
    assert [[cell.text for cell in row] for row in table.data.grid] == [["A", "B"]]


def test_hwp_backend_converts_stream(monkeypatch):
    _patch_rhwp(monkeypatch)
    stream = DocumentStream(name="sample.hwpx", stream=BytesIO(b"fake-hwpx"))

    result = _converter().convert(stream)

    assert result.input.format == InputFormat.HWP
    assert result.document.export_to_markdown().startswith("Hello HWP")


def test_hwp_guess_format_by_extension(tmp_path: Path):
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    hwp_path = tmp_path / "sample.hwp"
    hwp_path.write_bytes(b"fake-hwp")
    assert dci._guess_format(hwp_path) == InputFormat.HWP

    hwpx_path = tmp_path / "sample.hwpx"
    with zipfile.ZipFile(hwpx_path, "w") as zf:
        zf.writestr("Contents/content.hpf", "<package/>")
    assert dci._guess_format(hwpx_path) == InputFormat.HWP

    stream_bytes = BytesIO()
    with zipfile.ZipFile(stream_bytes, "w") as zf:
        zf.writestr("Contents/content.hpf", "<package/>")
    stream_bytes.seek(0)
    stream = DocumentStream(name="sample.hwpx", stream=stream_bytes)
    assert dci._guess_format(stream) == InputFormat.HWP
