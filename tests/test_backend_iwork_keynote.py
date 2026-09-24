# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the Apple Keynote (``.key``) document backend.

Test Data Attribution
---------------------
``keynote_2013.key``, ``keynote_2018.key``, ``keynote_iwork09.key`` and
``keynote_iwork09_textboxes.key`` are ``testKeynote2013.key``,
``testKeynote2018.key``, ``testKeynote.key`` and ``testTextBoxes.key`` from the
Apache Tika test corpus, licensed under the Apache License 2.0. They are genuine
Apple Keynote output, and between them cover all three container layouts
Keynote has used: ``keynote_2013.key`` stores its content as ``Index/*.iwa``,
``keynote_2018.key`` zips that index a second time into
``Presentation.key/Index.zip``, and the other two use the iWork '09
``index.apxl`` layout. Conveniently, ``keynote_2013.key`` and
``keynote_iwork09.key`` hold the same source deck, so the two readers can be
checked against each other.

See https://github.com/apache/tika (``tika-parser-apple-module`` test resources).
"""

import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import ContentLayer, DocItemLabel, GroupLabel
from docling_core.types.doc.items.group import GroupItem
from docling_core.types.doc.items.text import ListItem, TextItem

from docling.backend.iwork.content import Geometry
from docling.backend.iwork.keynote_content import (
    SLIDE_ROW_TOLERANCE,
    reading_order,
)
from docling.backend.iwork_backend import IWorkKeynoteDocumentBackend
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import InputDocument, _DocumentConversionInput
from docling.document_converter import DocumentConverter
from docling.exceptions import DocumentLoadError

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

SOURCES = Path("./tests/data/keynote/sources")
KEYNOTE_2013 = SOURCES / "keynote_2013.key"
KEYNOTE_2018 = SOURCES / "keynote_2018.key"
KEYNOTE_IWORK09 = SOURCES / "keynote_iwork09.key"
KEYNOTE_IWORK09_TEXTBOXES = SOURCES / "keynote_iwork09_textboxes.key"

GROUNDTRUTH = Path("./tests/data/keynote/groundtruth")

# Every fixture, each of which converts and so has a stored groundtruth.
CONVERTIBLE = [
    KEYNOTE_2013,
    KEYNOTE_2018,
    KEYNOTE_IWORK09,
    KEYNOTE_IWORK09_TEXTBOXES,
]

# The two fixtures below hold the same source deck, one per container generation.
_SHARED_DECK = [KEYNOTE_2013, KEYNOTE_IWORK09]

_TITLES = ("A sample presentation", "Slide 1", "Slide 3")

_BODY_SENTENCE = "Some random text for the sake of testability."


def _backend(
    path: Path, options: IWorkBackendOptions | None = None
) -> IWorkKeynoteDocumentBackend:
    in_doc = InputDocument(
        path_or_stream=path,
        format=InputFormat.IWORK_KEYNOTE,
        backend=IWorkKeynoteDocumentBackend,
        backend_options=options,
    )
    backend = in_doc._backend
    assert isinstance(backend, IWorkKeynoteDocumentBackend)
    return backend


def test_detects_keynote_from_path_and_named_stream():
    """`.key` is a ZIP, so detection must not stop at ``application/zip``."""
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[])

    assert conv_input._guess_format(KEYNOTE_2013) == InputFormat.IWORK_KEYNOTE

    stream = DocumentStream(name="deck.key", stream=BytesIO(KEYNOTE_2013.read_bytes()))
    assert conv_input._guess_format(stream) == InputFormat.IWORK_KEYNOTE


def test_extensionless_keynote_stream_is_not_claimed():
    """Without the extension a Keynote container is indistinguishable from Pages
    and Numbers, so the backend must not claim it rather than guess wrong."""
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[])
    stream = DocumentStream(name="blob", stream=BytesIO(KEYNOTE_2013.read_bytes()))

    assert conv_input._guess_format(stream) is None


@pytest.mark.parametrize("source", _SHARED_DECK, ids=lambda path: path.name)
def test_slide_titles_and_body_are_extracted(source: Path):
    """Both container generations hold the same deck, so the independent IWA and
    XML readers must recover the same titles and the same body text."""
    text = _backend(source).convert().export_to_markdown()

    for title in _TITLES:
        assert title in text
    assert _BODY_SENTENCE in text


def test_nested_index_presentation_is_read():
    """Keynote 2018 and later flatten the package and zip the index a second
    time, which is the layout most Keynote files in circulation now use."""
    names = set(zipfile.ZipFile(KEYNOTE_2018).namelist())
    assert "Presentation.key/Index.zip" in names, (
        "fixture no longer exercises the nested-index path"
    )
    assert not any(name.startswith("Index/") for name in names)

    text = _backend(KEYNOTE_2018).convert().export_to_markdown()

    assert "Libreoffice 6.2" in text
    assert "Windows 10 x86" in text


def test_a_slide_becomes_a_chapter_group_and_a_page():
    """The shape the PowerPoint and ODP backends give a presentation: one
    chapter group and one page per slide, sized as the deck is laid out."""
    doc = _backend(KEYNOTE_2013).convert()

    groups = [
        item
        for item in doc.body.children
        if isinstance(resolved := item.resolve(doc), GroupItem)
        and resolved.label == GroupLabel.CHAPTER
    ]
    assert len(groups) == 3
    assert [group.resolve(doc).name for group in groups] == [
        "slide-0",
        "slide-1",
        "slide-2",
    ]

    assert sorted(doc.pages) == [1, 2, 3]
    assert doc.pages[1].size.width == 1024
    assert doc.pages[1].size.height == 768


def test_widescreen_slide_size_is_read():
    """The 2018 fixture is 16:9, so the size comes from the show rather than
    from the 4:3 default."""
    doc = _backend(KEYNOTE_2018).convert()

    assert doc.pages[1].size.width == 1920
    assert doc.pages[1].size.height == 1080


@pytest.mark.parametrize("source", _SHARED_DECK, ids=lambda path: path.name)
def test_a_slide_title_is_labelled_as_a_title(source: Path):
    """The title comes from the placeholder Keynote reserves for it, not from a
    paragraph style called "Title"; theme style names are localised."""
    doc = _backend(source).convert()

    titles = [
        item.text
        for item, _ in doc.iterate_items()
        if isinstance(item, TextItem) and item.label == DocItemLabel.TITLE
    ]
    assert titles == list(_TITLES)


def test_a_localised_theme_still_yields_a_title():
    """The 2018 fixture names its theme styles in Indonesian — "Judul" for
    Title — which is exactly what a style-name mapping would miss."""
    doc = _backend(KEYNOTE_2018).convert()

    titles = [
        item.text
        for item, _ in doc.iterate_items()
        if isinstance(item, TextItem) and item.label == DocItemLabel.TITLE
    ]
    assert titles == ["Libreoffice 6.2", "Test running..."]


def test_bullets_become_list_items():
    """Keynote leaves the list style to the theme rather than naming it on the
    paragraph, so the style's inheritance chain has to be followed to find it."""
    doc = _backend(KEYNOTE_2018).convert()

    items = [item.text for item, _ in doc.iterate_items() if isinstance(item, ListItem)]
    assert items == [
        "I save in Apple Key App as default .key",
        "Open in Libreoffice 6.2",
        "Windows 10 x86",
    ]


@pytest.mark.parametrize("source", _SHARED_DECK, ids=lambda path: path.name)
def test_a_theme_inherited_bullet_is_found_in_either_container(source: Path):
    """Both generations hand the ladder down rather than copying it onto the
    style the text names, and both have to be followed to the end.

    The 2013 container spells the chain as a parent reference on the style's
    ``TSS`` super, the '09 one as ``sf:parent-ident``. Reading only the style in
    force finds an empty ladder in both and drops the bullet — which is what the
    2013 reader used to do, while the '09 reader kept it.
    """
    doc = _backend(source).convert()

    items = [item.text for item, _ in doc.iterate_items() if isinstance(item, ListItem)]
    assert _BODY_SENTENCE in items


def test_the_two_generations_disagree_only_where_their_themes_do():
    """The remaining difference between the two fixtures is in the documents.

    Slide 1's subtitle is a list item in the '09 file and plain text in the 2013
    one, because the themes differ rather than the readers: the 2013 style chain
    ends at a ladder that labels no rung at all, so Keynote draws no bullet
    there, while the '09 chain ends at one whose first rung is a bullet. Pinned
    so that the difference is not mistaken later for the reader bug it looks
    like.
    """
    subtitle = "For the Apache Tika project"

    modern = _backend(KEYNOTE_2013).convert()
    legacy = _backend(KEYNOTE_IWORK09).convert()

    def labelled(doc) -> str:
        item = next(
            item
            for item, _ in doc.iterate_items()
            if isinstance(item, TextItem) and item.text == subtitle
        )
        return type(item).__name__

    assert labelled(modern) == "TextItem"
    assert labelled(legacy) == "ListItem"


@pytest.mark.parametrize("source", _SHARED_DECK, ids=lambda path: path.name)
def test_a_table_on_a_slide_is_read(source: Path):
    """A slide's table is the same TST archive, and the same sf:tabular-model,
    that a Pages document embeds — including the cell it leaves empty."""
    doc = _backend(source).convert()

    assert len(doc.tables) == 1
    grid = doc.tables[0].data.grid
    assert [cell.text for cell in grid[0]] == ["Cell one", "Cell two", "Cell three"]
    # The first cell of the last row holds a number, which is not guessed at —
    # but it keeps its place, or the two text cells after it shift left.
    assert [cell.text for cell in grid[2]] == ["", "Cell eight", "5/5/1985"]


@pytest.mark.parametrize("source", _SHARED_DECK, ids=lambda path: path.name)
def test_presenter_notes_go_to_the_notes_layer(source: Path):
    """Notes are not shown when the deck is presented, so they stay out of the
    reading order, which is where the PowerPoint backend puts them too."""
    doc = _backend(source).convert()

    assert "A nice note" not in doc.export_to_markdown()

    notes = [
        item.text
        for item, _ in doc.iterate_items(included_content_layers={ContentLayer.NOTES})
        if isinstance(item, TextItem)
    ]
    assert "A nice note" in notes


@pytest.mark.parametrize("source", _SHARED_DECK, ids=lambda path: path.name)
def test_a_comment_is_read_once_and_not_as_body_text(source: Path):
    """Keynote draws a comment as a note stuck to the slide, so it arrives as a
    drawable carrying a copy of its own text — which must not be emitted as
    body text as well."""
    doc = _backend(source).convert()

    assert "A nice comment" not in doc.export_to_markdown()

    said = [
        item.text
        for item, _ in doc.iterate_items(included_content_layers={ContentLayer.NOTES})
        if isinstance(item, TextItem) and "A nice comment" in item.text
    ]
    assert len(said) == 1


def test_the_2013_comment_records_its_author():
    """The modern container keeps the comment proper in a TSD.CommentStorage
    beside the shape; the '09 one records no author at all."""
    doc = _backend(KEYNOTE_2013).convert()

    said = [
        item.text
        for item, _ in doc.iterate_items(included_content_layers={ContentLayer.NOTES})
        if isinstance(item, TextItem) and "A nice comment" in item.text
    ]
    assert said == ["[author: Author]: A nice comment"]


def test_text_boxes_come_out_in_reading_order():
    """Keynote stores a slide's drawables in the order they were stacked. This
    fixture stores text1, text2, text3 but lays them out with text3 at the top,
    so stored order is the wrong reading order."""
    doc = _backend(KEYNOTE_IWORK09_TEXTBOXES).convert()

    texts = [
        item.text
        for item, _ in doc.iterate_items()
        if isinstance(item, TextItem) and item.text.startswith("text")
    ]
    assert texts == ["text3", "text1", "text2"]


def test_reading_order_bands_a_row_and_reads_it_across():
    """Drawables within the row tolerance of the one before them share a row and
    are read left to right; the band may drift further than the tolerance as
    long as each step stays inside it."""
    drift = SLIDE_ROW_TOLERANCE / 2
    placed = [
        Geometry(left=300, top=100, width=10, height=10),
        Geometry(left=100, top=100 + drift, width=10, height=10),
        Geometry(left=200, top=100 + 2 * drift, width=10, height=10),
        Geometry(left=500, top=400, width=10, height=10),
    ]

    assert reading_order(placed) == [1, 2, 0, 3]


def test_reading_order_keeps_unplaced_drawables_last_and_in_order():
    """A drawable Keynote did not position has nowhere to sort to, so it falls
    to the end of the slide rather than to the top of it."""
    placed = [
        None,
        Geometry(left=10, top=500, width=10, height=10),
        None,
        Geometry(left=10, top=100, width=10, height=10),
    ]

    assert reading_order(placed) == [3, 1, 0, 2]


def test_master_slide_content_is_not_repeated():
    """What a master draws belongs to every slide using it rather than to any
    one of them, so the master's layer is not followed."""
    doc = _backend(KEYNOTE_IWORK09_TEXTBOXES).convert()

    text = doc.export_to_markdown()
    assert "text1" in text
    # The theme's master slides carry their own placeholder prompts; none of
    # them is content of this deck.
    assert "Bullets" not in text
    assert "Double-click to edit" not in text


def test_the_slide_number_placeholder_is_not_emitted():
    """A slide number is page furniture, and its placeholder holds nothing but
    the U+FFFC the number is substituted for."""
    doc = _backend(KEYNOTE_2013).convert()

    assert "￼" not in doc.export_to_markdown()


def test_items_carry_the_slide_they_came_from():
    """A paginated backend whose items had no provenance would leave every page
    empty, so each item records its page and the box of the drawable holding it."""
    doc = _backend(KEYNOTE_2013).convert()

    pages = {
        item.prov[0].page_no
        for item, _ in doc.iterate_items()
        if isinstance(item, TextItem) and item.prov
    }
    assert pages == {1, 2, 3}

    title = next(
        item
        for item, _ in doc.iterate_items()
        if isinstance(item, TextItem) and item.text == "Slide 1"
    )
    bbox = title.prov[0].bbox
    assert (bbox.l, bbox.t) == (100, 20)
    assert (bbox.r, bbox.b) == (924, 212)


def test_page_count_is_the_number_of_slides():
    backend = _backend(KEYNOTE_2013)

    assert backend.supports_pagination()
    assert backend.page_count() == 3


def test_keynote_backend_accepts_a_stream():
    stream = BytesIO(KEYNOTE_2013.read_bytes())
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.IWORK_KEYNOTE,
        backend=IWorkKeynoteDocumentBackend,
        filename="deck.key",
    )
    backend = in_doc._backend
    assert isinstance(backend, IWorkKeynoteDocumentBackend)

    assert _BODY_SENTENCE in backend.convert().export_to_markdown()


def _construct(
    path: Path, options: IWorkBackendOptions | None = None
) -> IWorkKeynoteDocumentBackend:
    """Build the backend directly, so a refusal surfaces instead of being caught.

    ``InputDocument`` swallows a backend that fails to load and marks itself
    invalid, which is right for a conversion run and useless for a test about
    why the load failed.
    """
    return IWorkKeynoteDocumentBackend(
        InputDocument(
            path_or_stream=path,
            format=InputFormat.IWORK_KEYNOTE,
            backend=IWorkKeynoteDocumentBackend,
        ),
        path,
        options,
    )


def _nested_key(
    tmp_path: Path, members: dict[str, bytes], *, encrypted: bool = False
) -> Path:
    """Build a ``.key`` in the layout Keynote 2018 and later write.

    The index is a ZIP inside the container, which is what makes it worth
    testing separately from the flat one: the stored size of that member says
    nothing about what the archive inside it holds.

    Args:
        tmp_path: Where to write the container.
        members: The inner archive's members and their contents.
        encrypted: Whether to mark the inner members as encrypted.

    Returns:
        The path of the container.
    """
    inner = BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as index:
        for name, payload in members.items():
            index.writestr(name, payload)
    raw = _flagged_as_encrypted(inner.getvalue()) if encrypted else inner.getvalue()

    path = tmp_path / "nested.key"
    with zipfile.ZipFile(path, "w") as container:
        container.writestr("Presentation.key/Index.zip", raw)
    return path


def _flagged_as_encrypted(raw: bytes) -> bytes:
    """Set the ZIP encryption flag on every member of an archive.

    ``zipfile`` recomputes the general-purpose flags as it writes, so the bit
    has to be set afterwards — in the local header and in the central directory
    alike, since both are read back.
    """
    patched = bytearray(raw)
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        at = patched.find(signature)
        while at != -1:
            patched[at + offset] |= 0x1
            at = patched.find(signature, at + 1)
    return bytes(patched)


def test_a_nested_index_is_held_to_the_total_size_limit(tmp_path: Path):
    """Everything a nested-index presentation holds is inside the inner archive,
    so bounding only the outer one bounds nothing: a container of a few KiB can
    expand to hundreds of megabytes, and the reader keeps every payload it
    decodes."""
    deck = _nested_key(
        tmp_path, {f"Index/Slide-{n}.iwa": b"\0" * (1 << 20) for n in range(8)}
    )
    assert deck.stat().st_size < 100 * 1024, "the container should be small"

    with pytest.raises(DocumentLoadError, match="max_total_bytes"):
        _construct(deck, IWorkBackendOptions(max_total_bytes=1024 * 1024))


def test_an_encrypted_nested_index_is_reported_as_password_protected(tmp_path: Path):
    """The advice to remove the password is the same whichever layout the
    container uses, so the encryption scan has to reach the inner archive too —
    otherwise this surfaces as a decompression failure instead."""
    deck = _nested_key(tmp_path, {"Index/Document.iwa": b"\0" * 64}, encrypted=True)

    with pytest.raises(DocumentLoadError, match="password-protected"):
        _construct(deck)


def test_a_zip_that_is_not_a_presentation_is_refused(tmp_path: Path):
    other_zip = tmp_path / "not_really.key"
    with zipfile.ZipFile(other_zip, "w") as archive:
        archive.writestr("ppt/presentation.xml", "<p:presentation/>")

    with pytest.raises(DocumentLoadError, match="does not look like a Keynote"):
        _construct(other_zip)


def test_a_file_that_is_not_a_zip_is_refused(tmp_path: Path):
    broken = tmp_path / "broken.key"
    broken.write_bytes(b"this is not a zip archive")

    with pytest.raises(DocumentLoadError, match="not a readable ZIP container"):
        _construct(broken)


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ({"max_member_count": 1}, "max_member_count"),
        ({"max_total_bytes": 1024}, "max_total_bytes"),
        ({"max_file_bytes": 64}, "max_file_bytes"),
    ],
    ids=["members", "total", "file"],
)
def test_archive_limits_are_enforced(option: dict[str, int], message: str):
    """The container is attacker-controlled, so each limit has to bite before
    the IWA archives are decompressed."""
    with pytest.raises(DocumentLoadError, match=message):
        _construct(KEYNOTE_2013, IWorkBackendOptions(**option))


def test_the_nested_index_is_held_to_the_same_limits():
    """The inner archive is where a nested-index presentation keeps everything,
    so bounding only the outer one would bound nothing."""
    with pytest.raises(DocumentLoadError, match="max_file_bytes"):
        _construct(KEYNOTE_2018, IWorkBackendOptions(max_file_bytes=64))


@pytest.mark.parametrize("source", CONVERTIBLE, ids=lambda path: path.name)
def test_conversion_matches_the_groundtruth(source: Path):
    """Pin the whole conversion of every fixture, so a change in any part of the
    backend shows up as a reviewable diff rather than passing unnoticed.

    The Markdown is the reading order a caller gets by default; the serialized
    ``DoclingDocument`` is what carries the rest — the slide grouping, the
    pages, the labels and list grouping, and the notes and comments that live
    outside the body layer.
    """
    doc = (
        DocumentConverter(allowed_formats=[InputFormat.IWORK_KEYNOTE])
        .convert(source)
        .document
    )
    groundtruth = GROUNDTRUTH / source.name

    assert verify_export(
        doc.export_to_markdown(), str(groundtruth) + ".md", generate=GEN_TEST_DATA
    ), f"export to markdown failed on {source}"

    assert verify_document(doc, str(groundtruth) + ".json", generate=GEN_TEST_DATA), (
        f"DoclingDocument verification failed on {source}"
    )
