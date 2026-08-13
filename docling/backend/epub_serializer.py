"""Markdown serialization helpers for book-shaped EPUB documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.types.doc import (
    DoclingDocument,
    ImageRefMode,
    SectionHeaderItem,
)
from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import Self, override

_OFFSET_WIDTH = 10


class EpubMetadata(BaseModel):
    """Book metadata extracted from an EPUB package document."""

    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    published: str | None = None
    language: str | None = None
    source_file: str | None = None


class EpubDocument(DoclingDocument):
    """A Docling document with transient metadata from an EPUB package."""

    _epub_metadata: EpubMetadata = PrivateAttr(default_factory=EpubMetadata)
    _internal_links_rewritten: bool = PrivateAttr(default=False)

    @property
    def epub_metadata(self) -> EpubMetadata:
        """Return a defensive copy of metadata extracted from the EPUB package."""
        return self._epub_metadata.model_copy(deep=True)

    @classmethod
    def from_document(
        cls,
        document: DoclingDocument,
        *,
        metadata: EpubMetadata,
        internal_links_rewritten: bool = False,
    ) -> Self:
        """Create an EPUB document without changing the serialized document model."""
        epub_document = cls.model_validate(document.model_dump())
        epub_document._epub_metadata = metadata.model_copy(deep=True)
        epub_document._internal_links_rewritten = internal_links_rewritten
        return epub_document

    def export_to_book_markdown(
        self,
        *,
        book_frontmatter: bool = False,
        chapter_index: bool = False,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        compact_tables: bool = False,
    ) -> str:
        """Export book Markdown; chapter offsets are UTF-8 bytes and 1-based lines."""
        self._ensure_book_export_ready(
            book_frontmatter=book_frontmatter,
            chapter_index=chapter_index,
        )
        serializer = EpubMarkdownDocSerializer(
            doc=self,
            params=EpubMarkdownParams(
                book_frontmatter=book_frontmatter,
                chapter_index=chapter_index,
                metadata=self._epub_metadata,
                image_mode=image_mode,
                compact_tables=compact_tables,
            ),
        )
        return serializer.serialize().text

    def save_as_book_markdown(
        self,
        filename: str | Path,
        *,
        artifacts_dir: Path | None = None,
        book_frontmatter: bool = False,
        chapter_index: bool = False,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        compact_tables: bool = False,
    ) -> None:
        """Save book Markdown while honoring Docling's image reference modes."""
        self._ensure_book_export_ready(
            book_frontmatter=book_frontmatter,
            chapter_index=chapter_index,
        )
        output_path = Path(filename)
        resolved_artifacts_dir, reference_path = self._get_output_paths(
            output_path, artifacts_dir
        )
        if image_mode == ImageRefMode.REFERENCED:
            resolved_artifacts_dir.mkdir(parents=True, exist_ok=True)

        document = self._make_copy_with_refmode(
            resolved_artifacts_dir,
            image_mode,
            page_no=None,
            reference_path=reference_path,
        )
        serializer = EpubMarkdownDocSerializer(
            doc=document,
            params=EpubMarkdownParams(
                book_frontmatter=book_frontmatter,
                chapter_index=chapter_index,
                metadata=self._epub_metadata,
                image_mode=image_mode,
                compact_tables=compact_tables,
            ),
        )
        output_path.write_bytes(serializer.serialize().text.encode("utf-8"))

    def _ensure_book_export_ready(
        self, *, book_frontmatter: bool, chapter_index: bool
    ) -> None:
        if (book_frontmatter or chapter_index) and not self._internal_links_rewritten:
            raise ValueError(
                "EPUB book Markdown requires conversion with "
                "EpubBackendOptions(rewrite_internal_links=True)."
            )


class EpubMarkdownParams(MarkdownParams):
    """Opt-in parameters for EPUB book Markdown serialization."""

    book_frontmatter: bool = Field(default=False, exclude=True)
    chapter_index: bool = Field(default=False, exclude=True)
    metadata: EpubMetadata | None = Field(default=None, exclude=True)


class _ChapterPosition(BaseModel):
    title: str
    line: int
    byte: int


class EpubMarkdownDocSerializer(MarkdownDocSerializer):
    """Serialize EPUB Markdown with optional metadata and chapter navigation."""

    params: EpubMarkdownParams

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        body_result = super().serialize_doc(parts=parts, **kwargs)
        if not (self.params.book_frontmatter or self.params.chapter_index):
            return body_result

        chapters = self._collect_chapters(parts) if self.params.chapter_index else []
        preliminary = self._render_frontmatter(chapters)
        byte_shift = len(preliminary.encode("utf-8"))
        line_shift = preliminary.count("\n")
        shifted_chapters = [
            chapter.model_copy(
                update={
                    "byte": chapter.byte + byte_shift,
                    "line": chapter.line + line_shift,
                }
            )
            for chapter in chapters
        ]
        frontmatter = self._render_frontmatter(shifted_chapters)
        if len(frontmatter.encode("utf-8")) != byte_shift:
            raise ValueError("EPUB frontmatter changed size after applying offsets")

        return create_ser_result(
            text=f"{frontmatter}{body_result.text}",
            span_source=parts,
        )

    def _collect_chapters(
        self, parts: list[SerializationResult]
    ) -> list[_ChapterPosition]:
        chapters: list[_ChapterPosition] = []
        byte_offset = 0
        line_number = 1
        has_content = False

        for part in parts:
            if not part.text:
                continue
            part_text = self._finalize_part_text(part.text)
            if has_content:
                byte_offset += len(b"\n\n")
                line_number += 2

            first_item = part.spans[0].item if part.spans else None
            if isinstance(first_item, SectionHeaderItem) and first_item.level == 1:
                chapters.append(
                    _ChapterPosition(
                        title=first_item.text,
                        line=line_number,
                        byte=byte_offset,
                    )
                )

            byte_offset += len(part_text.encode("utf-8"))
            line_number += part_text.count("\n")
            has_content = True

        return chapters

    def _finalize_part_text(self, text: str) -> str:
        """Apply body-level substitutions before measuring a serialized part."""
        if not self.requires_page_break():
            return text

        page_separator = self.params.page_break_placeholder or ""
        for full_match, _, _ in self._get_page_breaks(text=text):
            text = text.replace(full_match, page_separator)
        return text

    def _render_frontmatter(self, chapters: list[_ChapterPosition]) -> str:
        metadata = self.params.metadata
        lines = ["---"]
        if metadata is not None:
            fields: tuple[tuple[str, str | list[str] | None], ...] = (
                ("title", metadata.title),
                ("authors", metadata.authors or None),
                ("published", metadata.published),
                ("language", metadata.language),
                ("source_file", metadata.source_file),
            )
            for key, value in fields:
                if value is not None:
                    lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")

        if self.params.chapter_index:
            lines.append("chapters:" if chapters else "chapters: []")
            for chapter in chapters:
                lines.extend(
                    [
                        f"  - title: {json.dumps(chapter.title, ensure_ascii=False)}",
                        f"    line: {chapter.line:>{_OFFSET_WIDTH}}",
                        f"    byte: {chapter.byte:>{_OFFSET_WIDTH}}",
                    ]
                )

        lines.append("---")
        return "\n".join(lines) + "\n\n"
