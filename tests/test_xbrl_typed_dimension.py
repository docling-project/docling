# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression tests for typed-dimension facts in the XBRL backend.

A typed dimension (``xbrldi:typedMember``) has no member QName by design —
arelle leaves ``ModelDimensionValue.memberQname`` ``None`` and carries the
value on the ``typedMember`` element instead.  The backend used to
dereference ``memberQname.localName`` unconditionally, so every conversion
of a valid typed-dimension fact crashed with
``AttributeError: 'NoneType' object has no attribute 'localName'`` (issue #4437).

The fixtures under ``tests/data/xbrl/sources/typed-dimension`` are a
self-contained, minimal taxonomy: the standard XBRL 2.1 schemas are vendored
locally so the test does not depend on arelle's bundled web cache or on
remote schema resolution (which is disabled in CI via
``enable_remote_fetch=False``).
"""

from pathlib import Path

from docling.backend.xml.xbrl_backend import XBRLDocumentBackend
from docling.datamodel.backend_options import XBRLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_DATA_DIR = Path(__file__).parent / "data" / "xbrl" / "typed-dimension"
_TAXONOMY_DIR = _DATA_DIR / "taxonomy"
_INSTANCE = _DATA_DIR / "typed-dimension.xml"


def _convert():
    options = XBRLBackendOptions(
        taxonomy=_TAXONOMY_DIR,
        enable_local_fetch=True,
        enable_remote_fetch=False,
    )
    in_doc = InputDocument(
        path_or_stream=_INSTANCE,
        format=InputFormat.XML_XBRL,
        backend=XBRLDocumentBackend,
        backend_options=options,
        filename=_INSTANCE.name,
    )
    backend = XBRLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=_INSTANCE,
        options=options,
    )
    doc = backend.convert()
    return doc, backend


def test_typed_dimension_fact_converts() -> None:
    """A valid typed-dimension instance must convert without crashing.

    Before the fix, ``convert()`` raised
    ``AttributeError: 'NoneType' object has no attribute 'localName'``
    while rendering the fact's dimension cells.
    """
    doc, backend = _convert()

    # The fact itself is still rendered into the key-value cells.
    cell_texts = [cell.text for cell in backend._cells]
    assert "Revenue" in cell_texts
    assert "value: 123" in cell_texts
    assert doc.name == "typed-dimension"
