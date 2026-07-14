from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import _DocumentConversionInput
from docling.document_converter import DocumentConverter

DOCLANG_XML = """<doclang>
  <heading>DocLang Title</heading>
  <text>Hello world</text>
  <table>
    <fcel/><text>H1</text><fcel/><text>H2</text><nl/>
    <fcel/><text>C1</text><fcel/><text>C2</text><nl/>
  </table>
</doclang>
"""


def _write_dclx_archive(path: Path, *, xml: str = DOCLANG_XML) -> None:
    DocumentConverter(allowed_formats=[InputFormat.XML_DOCLANG]).convert_string(
        xml,
        format=InputFormat.XML_DOCLANG,
        name=f"{path.stem}.dclg.xml",
    ).document.save_as_doclang_archive(filename=path)


def test_dclx_backend_converts_path(tmp_path: Path):
    archive_path = tmp_path / "sample.dclx"
    _write_dclx_archive(archive_path)

    result = DocumentConverter(allowed_formats=[InputFormat.DCLX]).convert(archive_path)

    assert result.input.format == InputFormat.DCLX
    assert result.document.texts[0].label == DocItemLabel.TITLE
    assert result.document.texts[0].text == "DocLang Title"
    assert result.document.texts[1].text == "Hello world"
    assert len(result.document.tables) == 1


def test_dclx_backend_converts_stream(tmp_path: Path):
    archive_path = tmp_path / "sample.dclx"
    _write_dclx_archive(archive_path)
    stream = DocumentStream(
        name="sample.dclx",
        stream=BytesIO(archive_path.read_bytes()),
    )

    result = DocumentConverter(allowed_formats=[InputFormat.DCLX]).convert(stream)

    assert result.input.format == InputFormat.DCLX
    assert result.document.export_to_markdown().startswith("# DocLang Title")


def test_dclx_guess_format_by_extension(tmp_path: Path):
    dci = _DocumentConversionInput(path_or_stream_iterator=[])
    archive_path = tmp_path / "sample.dclx"
    _write_dclx_archive(archive_path)

    assert dci._guess_format(archive_path) == InputFormat.DCLX

    stream = DocumentStream(
        name="sample.dclx",
        stream=BytesIO(archive_path.read_bytes()),
    )
    assert dci._guess_format(stream) == InputFormat.DCLX


def test_dclx_not_guessed_without_dclx_extension(tmp_path: Path):
    dci = _DocumentConversionInput(path_or_stream_iterator=[])
    archive_path = tmp_path / "sample.dclx"
    _write_dclx_archive(archive_path)

    zip_path = tmp_path / "archive.zip"
    zip_path.write_bytes(archive_path.read_bytes())
    assert dci._guess_format(zip_path) is None

    extensionless_path = tmp_path / "archive_no_ext"
    extensionless_path.write_bytes(archive_path.read_bytes())
    assert dci._guess_format(extensionless_path) is None

    stream = DocumentStream(
        name="archive.zip",
        stream=BytesIO(zip_path.read_bytes()),
    )
    assert dci._guess_format(stream) is None


ROUNDTRIP_GT_PATHS = [
    Path("tests/data/md_deepseek/groundtruth/deepseek_simple.md.json"),
]


@pytest.mark.parametrize("gt_path", ROUNDTRIP_GT_PATHS)
def test_dclx_roundtrip_from_groundtruth(gt_path: Path, tmp_path: Path):
    original_doc = DoclingDocument.load_from_json(gt_path)
    archive_path = tmp_path / f"{gt_path.stem}.dclx"
    original_doc.save_as_doclang_archive(filename=archive_path)

    result = DocumentConverter(allowed_formats=[InputFormat.DCLX]).convert(archive_path)
    roundtrip_doc = result.document

    assert roundtrip_doc.export_to_markdown() == original_doc.export_to_markdown()
