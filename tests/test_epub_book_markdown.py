import re
from pathlib import Path

import pytest
import yaml
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    GroupLabel,
    ProvenanceItem,
)

from docling.backend.epub_serializer import (
    EpubDocument,
    EpubMarkdownDocSerializer,
    EpubMarkdownParams,
    EpubMetadata,
)


def _book_document() -> DoclingDocument:
    doc = DoclingDocument(name="alice")
    doc.add_title("Alice's Adventures in Wonderland")
    doc.add_heading("CHAPTER I. Dîner à Oxford", level=1)
    doc.add_text(DocItemLabel.TEXT, "Down the rabbit-hole.")
    doc.add_heading("CHAPTER II. The Pool of Tears", level=1)
    doc.add_text(DocItemLabel.TEXT, "Curiouser and curiouser!")
    return doc


def _metadata() -> EpubMetadata:
    return EpubMetadata(
        title="Alice's Adventures in Wonderland",
        authors=["Lewis Carroll"],
        published="2008-06-27",
        language="en",
        source_file="alice.epub",
    )


def _serialize(params: EpubMarkdownParams) -> str:
    return (
        EpubMarkdownDocSerializer(doc=_book_document(), params=params).serialize().text
    )


def test_book_options_off_preserve_default_markdown() -> None:
    doc = _book_document()

    expected = MarkdownDocSerializer(doc=doc).serialize().text
    actual = (
        EpubMarkdownDocSerializer(
            doc=doc,
            params=EpubMarkdownParams(metadata=_metadata()),
        )
        .serialize()
        .text
    )

    assert actual == expected


def test_book_frontmatter_serializes_available_metadata() -> None:
    metadata = _metadata().model_copy(update={"published": None})

    markdown = _serialize(EpubMarkdownParams(metadata=metadata, book_frontmatter=True))
    frontmatter = markdown.split("---\n", maxsplit=2)[1]
    parsed = yaml.safe_load(frontmatter)

    assert parsed == {
        "title": "Alice's Adventures in Wonderland",
        "authors": ["Lewis Carroll"],
        "language": "en",
        "source_file": "alice.epub",
    }


def test_chapter_byte_offsets_seek_to_utf8_heading() -> None:
    markdown = _serialize(EpubMarkdownParams(metadata=_metadata(), chapter_index=True))
    parsed = yaml.safe_load(markdown.split("---\n", maxsplit=2)[1])
    markdown_bytes = markdown.encode("utf-8")

    assert [chapter["title"] for chapter in parsed["chapters"]] == [
        "CHAPTER I. Dîner à Oxford",
        "CHAPTER II. The Pool of Tears",
    ]
    for chapter in parsed["chapters"]:
        heading = f"## {chapter['title']}".encode()
        offset = chapter["byte"]
        assert markdown_bytes[offset : offset + len(heading)] == heading


def test_chapter_line_offsets_are_one_based_and_numeric_fields_are_fixed_width() -> (
    None
):
    markdown = _serialize(EpubMarkdownParams(metadata=_metadata(), chapter_index=True))
    parsed = yaml.safe_load(markdown.split("---\n", maxsplit=2)[1])
    lines = markdown.splitlines()

    for chapter in parsed["chapters"]:
        assert lines[chapter["line"] - 1] == f"## {chapter['title']}"

    numeric_fields = re.findall(r"^    (?:line|byte): (.{10})$", markdown, re.MULTILINE)
    assert len(numeric_fields) == 4
    assert all(
        value.startswith(" ") and value.strip().isdigit() for value in numeric_fields
    )


def test_chapter_index_output_is_deterministic() -> None:
    params = EpubMarkdownParams(metadata=_metadata(), chapter_index=True)

    first = _serialize(params)
    second = _serialize(params)

    assert second.encode("utf-8") == first.encode("utf-8")


def test_frontmatter_size_is_invariant_across_offset_digit_widths() -> None:
    frontmatter_sizes: set[int] = set()
    offset_digit_widths: set[int] = set()

    for prefix_size in (1, 2_000, 1_000_000):
        doc = DoclingDocument(name="offset-width")
        doc.add_text(DocItemLabel.TEXT, "x" * prefix_size)
        doc.add_heading("Chapter", level=1)
        markdown = (
            EpubMarkdownDocSerializer(
                doc=doc,
                params=EpubMarkdownParams(metadata=_metadata(), chapter_index=True),
            )
            .serialize()
            .text
        )
        frontmatter_end = markdown.index("---\n\n") + len("---\n\n")
        chapter = yaml.safe_load(markdown.split("---\n", maxsplit=2)[1])["chapters"][0]

        frontmatter_sizes.add(len(markdown[:frontmatter_end].encode()))
        offset_digit_widths.add(len(str(chapter["byte"])))

    assert len(frontmatter_sizes) == 1
    assert offset_digit_widths == {3, 4, 7}


def test_chapter_index_only_includes_headings_at_top_level_part_boundaries() -> None:
    doc = DoclingDocument(name="nested")
    section = doc.add_group(label=GroupLabel.SECTION, name="section")
    doc.add_text(DocItemLabel.TEXT, "See ## Nested heading", parent=section)
    doc.add_heading("Nested heading", level=1, parent=section)

    markdown = (
        EpubMarkdownDocSerializer(
            doc=doc,
            params=EpubMarkdownParams(metadata=_metadata(), chapter_index=True),
        )
        .serialize()
        .text
    )
    parsed = yaml.safe_load(markdown.split("---\n", maxsplit=2)[1])

    assert parsed["chapters"] == []


def test_empty_chapter_index_serializes_as_a_list() -> None:
    doc = DoclingDocument(name="empty")
    doc.add_text(DocItemLabel.TEXT, "No headings here.")

    markdown = (
        EpubMarkdownDocSerializer(
            doc=doc,
            params=EpubMarkdownParams(metadata=_metadata(), chapter_index=True),
        )
        .serialize()
        .text
    )
    parsed = yaml.safe_load(markdown.split("---\n", maxsplit=2)[1])

    assert parsed["chapters"] == []


def test_chapter_offsets_account_for_page_break_replacement() -> None:
    first_page = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox(l=0, t=0, r=1, b=1),
        charspan=(0, 1),
    )
    second_page = ProvenanceItem(
        page_no=2,
        bbox=BoundingBox(l=0, t=0, r=1, b=1),
        charspan=(2, 3),
    )
    doc = DoclingDocument(name="pages")
    doc.add_heading("One", level=1, prov=first_page)
    doc.add_text(DocItemLabel.TEXT, "body", prov=first_page)
    doc.add_heading("Two", level=1, prov=second_page)

    markdown = (
        EpubMarkdownDocSerializer(
            doc=doc,
            params=EpubMarkdownParams(
                metadata=_metadata(),
                chapter_index=True,
                page_break_placeholder="<PB>",
            ),
        )
        .serialize()
        .text
    )
    parsed = yaml.safe_load(markdown.split("---\n", maxsplit=2)[1])
    markdown_bytes = markdown.encode()

    for chapter in parsed["chapters"]:
        assert markdown_bytes[chapter["byte"] :].startswith(
            f"## {chapter['title']}".encode()
        )
        assert markdown.splitlines()[chapter["line"] - 1] == f"## {chapter['title']}"


def test_saved_chapter_offsets_are_independent_of_platform_newlines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = EpubDocument.from_document(
        _book_document(),
        metadata=_metadata(),
        internal_links_rewritten=True,
    )
    output = tmp_path / "book.md"

    def write_text_with_windows_newlines(
        path: Path, text: str, encoding: str | None = None, **_: object
    ) -> int:
        encoded = text.replace("\n", "\r\n").encode(encoding or "utf-8")
        return path.write_bytes(encoded)

    monkeypatch.setattr(Path, "write_text", write_text_with_windows_newlines)
    document.save_as_book_markdown(output, chapter_index=True)

    saved = output.read_bytes()
    parsed = yaml.safe_load(saved.decode().split("---", maxsplit=2)[1])
    for chapter in parsed["chapters"]:
        assert saved[chapter["byte"] :].startswith(f"## {chapter['title']}".encode())
