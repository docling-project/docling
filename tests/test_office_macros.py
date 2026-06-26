import logging
import zipfile
from io import BytesIO

from docling.utils.office_utils import warn_if_macros


def _make_zip(members: dict[str, bytes]) -> BytesIO:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    buf.seek(0)
    return buf


def test_macro_file_logs_warning(caplog):
    buf = _make_zip(
        {
            "word/document.xml": b"<doc/>",
            "word/vbaProject.bin": b"fake vba content",
        }
    )
    with caplog.at_level(logging.WARNING, logger="docling.utils.office_utils"):
        warn_if_macros(buf)
    assert any("vbaProject.bin" in r.message for r in caplog.records)


def test_clean_file_no_warning(caplog):
    buf = _make_zip({"word/document.xml": b"<doc/>"})
    with caplog.at_level(logging.WARNING, logger="docling.utils.office_utils"):
        warn_if_macros(buf)
    assert not caplog.records


def test_bytesio_rewound_after_check():
    buf = _make_zip({"word/vbaProject.bin": b"vba"})
    warn_if_macros(buf)
    assert buf.tell() == 0


def test_bytesio_rewound_on_clean_file():
    buf = _make_zip({"word/document.xml": b"<doc/>"})
    warn_if_macros(buf)
    assert buf.tell() == 0


def test_non_zip_does_not_raise():
    buf = BytesIO(b"this is not a zip file at all")
    warn_if_macros(buf)  # must not raise
    assert buf.tell() == 0


def test_path_based_macro_detection(tmp_path, caplog):
    p = tmp_path / "macro.docx"
    buf = _make_zip({"word/vbaProject.bin": b"vba"})
    p.write_bytes(buf.read())

    with caplog.at_level(logging.WARNING, logger="docling.utils.office_utils"):
        warn_if_macros(p)
    assert any("vbaProject.bin" in r.message for r in caplog.records)


def test_path_based_clean_file(tmp_path, caplog):
    p = tmp_path / "clean.docx"
    buf = _make_zip({"word/document.xml": b"<doc/>"})
    p.write_bytes(buf.read())

    with caplog.at_level(logging.WARNING, logger="docling.utils.office_utils"):
        warn_if_macros(p)
    assert not caplog.records
