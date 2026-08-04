"""Generate the synthetic ``.pages`` fixtures under ``tests/data/pages/sources``.

Apple's Pages app is macOS/iOS-only, so the fixtures in this repository are built
from the published container layout rather than exported from Pages itself. Every
layer the backend actually inspects is reproduced faithfully:

* the ZIP container and its member names,
* the ``Index/*.iwa`` chunk framing (1-byte compression tag + 3-byte little-endian
  length, wrapping a *raw* — unframed, CRC-less — Snappy block),
* the ``QuickLook/Preview.pdf`` render that Pages embeds when "Include preview in
  document" is enabled.

The protobuf payload inside the IWA is a minimal hand-encoded ``TSP.ArchiveInfo``
stand-in, not real Pages document content. That is sufficient for the preview-PDF
backend, which never decodes it. Native IWA parsing will need fixtures exported
from a real copy of Pages.

Run with ``python scripts/make_iwork_pages_fixtures.py``; it rewrites the fixtures
in place and needs no third-party packages.
"""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

OUT_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "data" / "pages" / "sources"
)

# Snappy is only needed to build the IWA payload. cramjam is the runtime choice
# for Phase 2; here we emit a literal-only Snappy block by hand so that fixture
# generation stays dependency-free.
_MAX_SNAPPY_LITERAL = 1 << 16


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _snappy_literal_block(payload: bytes) -> bytes:
    """Encode ``payload`` as a raw Snappy block using a single literal run.

    A literal-only block is a valid, if uncompressed, Snappy stream: the format
    permits any tag sequence that reproduces the input. Keeping the encoder to
    one branch avoids pulling a compressor into the build step.
    """
    if len(payload) >= _MAX_SNAPPY_LITERAL:
        raise ValueError("fixture payloads are expected to stay well under 64 KiB")

    out = bytearray(_varint(len(payload)))
    length = len(payload)
    if length <= 60:
        out.append((length - 1) << 2)
    else:
        # 2-byte little-endian length, tag 61 in the upper 6 bits.
        out.append((61 << 2) | 0x00)
        out.append((length - 1) & 0xFF)
        out.append(((length - 1) >> 8) & 0xFF)
    out += payload
    return bytes(out)


def _archive_info(object_id: int, message_type: int, payload_len: int) -> bytes:
    """Hand-encode a minimal ``TSP.ArchiveInfo`` protobuf message."""
    message_info = (
        b"\x08"
        + _varint(message_type)  # field 1: type
        + b"\x18"
        + _varint(payload_len)  # field 3: length
    )
    return (
        b"\x08"
        + _varint(object_id)  # field 1: identifier
        + b"\x12"
        + _varint(len(message_info))
        + message_info  # field 2: message_infos
    )


def build_iwa(object_id: int, message_type: int, payload: bytes) -> bytes:
    """Build a single-chunk IWA file around ``payload``."""
    info = _archive_info(object_id, message_type, len(payload))
    uncompressed = _varint(len(info)) + info + payload
    compressed = _snappy_literal_block(uncompressed)

    if len(compressed) > 0xFFFFFF:
        raise ValueError("chunk exceeds the 3-byte IWA length field")
    header = b"\x00" + len(compressed).to_bytes(3, "little")
    return header + compressed


def build_pdf(pages: list[list[str]]) -> bytes:
    """Build a minimal single-font PDF with one text block per page.

    Written by hand so the fixtures stay byte-stable and the build step keeps no
    PDF-producing dependency.
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_id = 3 + 2 * len(pages)
    page_ids = [3 + 2 * i for i in range(len(pages))]

    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = b" ".join(b"%d 0 R" % pid for pid in page_ids)
    add(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(pages)))

    for lines in pages:
        stream_parts = [b"BT", b"/F1 18 Tf", b"72 700 Td", b"22 TL"]
        for line in lines:
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            stream_parts.append(b"(" + escaped.encode("ascii") + b") Tj T*")
        stream_parts.append(b"ET")
        stream = b"\n".join(stream_parts)

        add(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
            % (font_id, len(objects) + 2)
        )
        add(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))

    add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % index + body + b"\nendobj\n"

    xref_offset = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_offset,
    )
    return bytes(out)


# A 1x1 white JPEG, standing in for the QuickLook thumbnail Pages writes.
_THUMBNAIL_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300ff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffc2000b08000100"
    "0101011100ffc40014000100000000000000000000000000000009"
    "ffda0008010100000010d27fffd9"
)

_LEGACY_INDEX_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<sl:document xmlns:sl="http://developer.apple.com/namespaces/sl"
             xmlns:sf="http://developer.apple.com/namespaces/sf"
             sl:version="92008.102325">
  <sf:text-body>
    <sf:p>Legacy iWork 09 body text.</sf:p>
  </sf:text-body>
</sl:document>
"""


def _write_pages_file(
    path: Path,
    *,
    members: dict[str, bytes],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            # Fixed timestamp keeps regeneration byte-stable.
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)


def _modern_members() -> dict[str, bytes]:
    """Members of a Pages 5+ (2013 onwards) container.

    The layout mirrors a real modern document, member for member: content lives
    in ``Index/*.iwa``, previews are root-level JPEGs, and there is **no**
    ``QuickLook/Preview.pdf`` — Apple dropped that after iWork '09.
    """
    return {
        "Data/1129414749_329x494-small-14.jpeg": _THUMBNAIL_JPEG,
        # 10000 is TP.DocumentArchive in the reverse-engineered iWork mapping.
        "Index/Document.iwa": build_iwa(1, 10000, b"\x0a\x0cPages fixture"),
        "Index/ViewState.iwa": build_iwa(2, 11000, b"\x0a\x04view"),
        "Index/DocumentStylesheet.iwa": build_iwa(3, 2034, b"\x0a\x06styles"),
        "Index/DocumentMetadata.iwa": build_iwa(4, 10011, b"\x0a\x04meta"),
        "Index/Metadata.iwa": build_iwa(5, 11006, b"\x0a\x08metadata"),
        "Metadata/Properties.plist": (
            b'<?xml version="1.0" encoding="UTF-8"?>\n'
            b'<plist version="1.0"><dict>'
            b"<key>fileFormatVersion</key><string>14.2</string>"
            b"</dict></plist>\n"
        ),
        "Metadata/DocumentIdentifier": b"00000000-0000-0000-0000-00000000FEED",
        "Metadata/BuildVersionHistory.plist": (
            b'<?xml version="1.0" encoding="UTF-8"?>\n'
            b'<plist version="1.0"><array>'
            b"<string>Pages 14.2 (Fixture)</string>"
            b"</array></plist>\n"
        ),
        "preview.jpg": _THUMBNAIL_JPEG,
        "preview-micro.jpg": _THUMBNAIL_JPEG,
        "preview-web.jpg": _THUMBNAIL_JPEG,
    }


def main() -> None:
    preview = build_pdf(
        [
            ["Docling Pages fixture", "First page body text."],
            ["Second page heading", "Second page body text."],
        ]
    )

    # Pages 5+ (2013 onwards): IWA content, root-level JPEG previews, no PDF.
    # This is what essentially every Pages document in the wild looks like.
    _write_pages_file(OUT_DIR / "pages_modern.pages", members=_modern_members())

    # iWork '09: gzipped index.xml plus the QuickLook preview PDF that Apple
    # stopped writing after this release.
    _write_pages_file(
        OUT_DIR / "pages_legacy09.pages",
        members={
            "index.xml.gz": gzip.compress(_LEGACY_INDEX_XML, mtime=0),
            "QuickLook/Preview.pdf": preview,
            "QuickLook/Thumbnail.jpg": _THUMBNAIL_JPEG,
        },
    )

    # iWork '09 saved with "Include preview in document" unchecked.
    _write_pages_file(
        OUT_DIR / "pages_legacy09_no_preview.pages",
        members={"index.xml.gz": gzip.compress(_LEGACY_INDEX_XML, mtime=0)},
    )

    for path in sorted(OUT_DIR.glob("*.pages")):
        print(
            f"wrote {path.relative_to(OUT_DIR.parent.parent.parent)} ({path.stat().st_size} bytes)"
        )


if __name__ == "__main__":
    main()
