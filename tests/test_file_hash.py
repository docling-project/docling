# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import hashlib
from io import BytesIO

from docling_core.types.doc import DocItemLabel, DoclingDocument

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter
from docling.utils.utils import create_file_hash

DATA = b"x,y\n1,2\n"
EXPECTED = hashlib.sha256(DATA).hexdigest()


def test_create_file_hash_path_and_stream_agree(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(DATA)

    assert create_file_hash(path) == EXPECTED
    assert create_file_hash(BytesIO(DATA)) == EXPECTED


def test_create_file_hash_stream_not_rewound():
    stream = BytesIO()
    stream.write(DATA)  # position is now at the end

    assert create_file_hash(stream) == EXPECTED


def test_create_file_hash_keeps_stream_position():
    stream = BytesIO(DATA)
    stream.seek(3)

    assert create_file_hash(stream) == EXPECTED
    assert stream.tell() == 3
    assert create_file_hash(stream) == EXPECTED


def test_create_file_hash_different_streams_differ():
    first = BytesIO()
    first.write(b"a,b\n1,2\n")
    second = BytesIO()
    second.write(b"c,d\n3,4\n")

    assert create_file_hash(first) != create_file_hash(second)


def _doclang_stream(text: str) -> BytesIO:
    doc = DoclingDocument(name="doc")
    doc.add_text(label=DocItemLabel.TEXT, text=text)
    stream = BytesIO()
    stream.write(doc.export_to_doclang().encode("utf-8"))  # not rewound
    return stream


def test_converter_hashes_doclang_stream_that_was_not_rewound():
    # Format detection returns early for .dclg/.dclx names, before the
    # rewind that the content sniffing of other formats does.
    converter = DocumentConverter(allowed_formats=[InputFormat.XML_DOCLANG])
    first = _doclang_stream("first")
    second = _doclang_stream("second")
    expected_first = hashlib.sha256(first.getvalue()).hexdigest()

    result_first = converter.convert(DocumentStream(name="a.dclg", stream=first))
    result_second = converter.convert(DocumentStream(name="b.dclg", stream=second))

    assert result_first.input.document_hash == expected_first
    assert result_first.input.document_hash != result_second.input.document_hash
    assert (
        result_first.document.origin.binary_hash
        != result_second.document.origin.binary_hash
    )
