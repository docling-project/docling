# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Final, Optional, Union

from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    Formatting,
    GroupItem,
    GroupLabel,
    ImageRef,
    ListItem,
    NodeItem,
    TableCell,
    TableData,
    TextItem,
)
from pydantic import AnyUrl, TypeAdapter

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.utils.image_resource_loader import ImageResourceLoader
from docling.datamodel.backend_options import AsciiDocBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError
from docling.utils._inline_runs import trim_run_edges
from docling.utils.code_language import CodeLanguageLabel, detect_code_language
from docling.utils.text_decoding import decode_text

_log = logging.getLogger(__name__)

# A line starting with exactly two slashes is a comment and never body content.
# A line of four or more slashes opens a delimited comment block: every line up
# to an identical line is a comment.
_LINE_COMMENT_RE = re.compile(r"^//(?!/)")
_COMMENT_BLOCK_DELIMITER_RE = re.compile(r"/{4,}")
# The "start" attribute of an ordered list, e.g. "[start=4]" or
# "[loweralpha,start=4]", sets the number of its first item.
_LIST_START_ATTR_RE = re.compile(
    r'^\[(?:[^\]]*,)?\s*start\s*=\s*"?(\d+)"?\s*(?:,[^\]]*)?\]$'
)

# Cell format specifier that may precede a "|" delimiter, e.g. "^.^h" in
# "^.^h|Header": span (3*, 2+, .2+, 2.3+), alignment (<, ^, >, .^), style
# (a/d/e/h/l/m/s). AsciiDoc writes the span as [colspan][.rowspan] followed by
# "+" or "*", and either number may be omitted, so ".2+" is a rowspan on its
# own. Requiring at least one of the two keeps a bare "+" from matching.
_CELL_SPEC: Final = r"(?:(?:\d+(?:\.\d+)?|\.\d+)[*+])*[<^>]?(?:\.[<^>])?[adehlms]?"
# The span part of a cell specifier: "2+" spans two columns, ".3+" three rows,
# "2.3+" both. A "*" factor duplicates the cell instead and is not a span.
_CELL_SPAN_RE: Final = re.compile(r"(?P<col>\d+)?(?:\.(?P<row>\d+))?\+")
# The duplication factor of a cell specifier: "3*" writes the cell three times.
# Asciidoctor takes one span or one factor per cell and keeps "2*3+" as text;
# Docling drops it as a specifier, as it did before the factor was read.
_CELL_REPEAT_RE: Final = re.compile(r"(?P<repeat>\d+)\*(?![\d.])")
# An ordered marker with a letter has one letter only ("a.", "B."), so a line
# that starts with an abbreviation such as "Dr." or "Fig." stays text.
_LIST_ITEM_PATTERN: Final = r"^(\s*)(\*+|-|\.+|\d+\.|[a-zA-Z]\.)\s+(.*)"

# URL schemes that Asciidoctor turns into links without a macro name.
_URL_SCHEME: Final = r"(?:https?|file|ftp|irc)://"
# Asciidoctor only links a URL at the start of the text, after whitespace or
# after one of >()[];"' so that a URL inside a word or an attribute stays text.
_URL_START: Final = r"""(?<![^\s>()\[\];"'])"""
# The attribute list of a link, between its brackets. A bracket in it must be
# escaped, so an unclosed "[" ends the scan at the next one, not at the end of
# the line, and a line with many of them still converts in linear time.
_LINK_ATTRS: Final = r"(?:[^\[\]\\]|\\.)*"
# A link target is a URL, or a relative path such as "docs/install.html".
_LINK_TARGET: Final = TypeAdapter(AnyUrl | Path)

# Links, then inline formatting marks. Links come first so that the "_" and
# "*" in a URL are not read as formatting marks. The link forms follow
# Asciidoctor: a URL with link text in brackets, a URL in angle brackets, a
# bare URL (which ends before trailing punctuation), and the link: and mailto:
# macros. Formatting marks are unconstrained (doubled, valid anywhere) before
# constrained (single, valid only at word boundaries, so "snake_case" and
# "2*3*4" stay plain text), as in Asciidoctor. A constrained span does not
# contain its own mark, so each scan stops at the next mark on the line.
_INLINE_FORMAT_RE: Final = re.compile(
    rf"{_URL_START}(?P<url>{_URL_SCHEME}[^\s\[\]]+)\[(?P<url_text>{_LINK_ATTRS})\]"
    rf"|<(?P<angle_url>{_URL_SCHEME}[^\s<>\[\]]+)>"
    rf"|{_URL_START}(?P<bare_url>{_URL_SCHEME}[^\s\[\]<]*[^\s,.?!\[\]<)])"
    r"|(?<![\w\\])(?:link|(?P<mailto>mailto)):"
    rf"(?P<macro_target>[^:\s\[][^\s\[]*)\[(?P<macro_text>{_LINK_ATTRS})\]"
    r"|``(?P<umono>.+?)``"
    r"|\*\*(?P<ubold>.+?)\*\*"
    r"|__(?P<uitalic>.+?)__"
    r"|(?<!\w)`(?P<mono>[^`\s](?:[^`]*?[^`\s])?)`(?!\w)"
    r"|(?<!\w)\*(?P<bold>[^*\s](?:[^*]*?[^*\s])?)\*(?!\w)"
    r"|(?<!\w)_(?P<italic>[^_\s](?:[^_]*?[^_\s])?)_(?!\w)"
)


@dataclass(frozen=True)
class _InlineRun:
    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False
    hyperlink: str | None = None


@dataclass(frozen=True)
class _LiteralBlock:
    text: str
    language: str | None = None  # set for [source,lang] listing blocks


@dataclass
class _TableCellSource:
    """One table cell as written: its text, spans and duplication factor."""

    text: str
    col_span: int = 1
    row_span: int = 1
    repeat: int = 1


@dataclass(frozen=True)
class _CommentBlock:
    """An AsciiDoc comment, with its // or //// delimiters removed.

    One block per comment: a run of consecutive // lines is one comment, and
    so is everything between a pair of //// delimiters.
    """

    text: str


class AsciiDocBackend(DeclarativeDocumentBackend):
    def __init__(
        self,
        in_doc: "InputDocument",
        path_or_stream: Union[BytesIO, Path],
        options: Optional[AsciiDocBackendOptions] = None,
    ):
        if options is None:
            options = AsciiDocBackendOptions()
        super().__init__(in_doc, path_or_stream, options)

        self.path_or_stream = path_or_stream
        self.options: AsciiDocBackendOptions
        self._image_loader = ImageResourceLoader(
            enable_local_fetch=options.enable_local_fetch,
            enable_remote_fetch=options.enable_remote_fetch,
        )

        # A leading BOM is dropped. Kept, it prefixes the first line, so a
        # document title ("= Title") is no longer recognized as one and the BOM
        # reaches the output.
        try:
            self.lines = decode_text(self.path_or_stream, options.encoding).split("\n")
            self.valid = True

        except DocumentLoadError:
            # Already carries a message naming what could not be decoded.
            raise
        except Exception as e:
            raise DocumentLoadError(
                f"Could not initialize AsciiDoc backend for file with hash {self.document_hash}."
            ) from e
        return

    def is_valid(self) -> bool:
        return self.valid

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    def unload(self):
        return

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.ASCIIDOC}

    def convert(self) -> DoclingDocument:
        """
        Parses the ASCII into a structured document model.
        """

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="text/asciidoc",
            binary_hash=self.document_hash,
        )

        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        doc = self._parse(doc)

        return doc

    def _flush_blank_line(
        self,
        doc: DoclingDocument,
        caption_data: list[str],
        text_data: list[str],
        parent: Union[GroupItem, None],
    ) -> tuple[list[str], list[str]]:
        """Flush both accumulators on a blank line.

        Plain text accumulated before a block title is emitted first to
        preserve reading order, then any pending caption is emitted as a
        bold paragraph.
        """
        text_data = self._flush_text_data(doc=doc, text_data=text_data, parent=parent)
        if caption_data:
            caption_data = self._flush_caption_as_paragraph(doc, caption_data, parent)
        return caption_data, text_data

    def _flush_caption_as_paragraph(
        self,
        doc: DoclingDocument,
        caption_data: list[str],
        parent: Union[GroupItem, None],
    ) -> list[str]:
        """Emit a pending block title as a bold paragraph.

        Used for block titles whose target is not a floating item (picture,
        table, or code block), which have no caption slot.
        """
        self._flush_text_data(doc=doc, text_data=caption_data, parent=parent, bold=True)
        return []

    def _parse(self, doc: DoclingDocument):
        """Orchestrate parsing and populate `doc` from the source lines.

        Handles titles, section headers, text paragraphs, lists, tables,
        pictures, literal (code) blocks, and block titles.

        Block titles (AsciiDoc lines that start with `.` immediately
        followed by text, e.g. `.Procedure`) are treated as follows:

        * FloatingItem targets (picture, table, code block): the block
          title is created as a `CAPTION`-labelled `TextItem` and
          attached to the floating item via its `caption` parameter, so it
          participates in the standard caption relationship.
        * All other targets (lists, paragraphs, ...): the block title is
          emitted as a bold `PARAGRAPH` `TextItem` inserted
          immediately before the element it precedes.  This preserves
          reading order while acknowledging that non-floating items (e.g.
          `GroupItem`) have no caption slot.
        """

        in_list = False
        in_table = False
        in_delimited_table = False

        text_data: list[str] = []
        table_data: list[list[_TableCellSource]] = []
        caption_data: list[str] = []
        last_list_item: ListItem | None = None
        list_continuation = False
        # The marker kind of the list group at each level ("." for ordered,
        # "*" or "-" for bullets), for the list being parsed. Reset when a new
        # list starts, so a level that holds a section header is never mistaken
        # for a list.
        list_kinds: dict[int, str] = {}
        # The "start" value of a block attribute line, kept until the block
        # it applies to, and the next number of each ordered list that does
        # not start at 1, by the reference of its group.
        pending_list_start: int | None = None
        next_list_numbers: dict[str, int] = {}

        parents: dict[int, Union[GroupItem, None]] = {}
        indents: dict[int, Union[GroupItem, None]] = {}

        for i in range(10):
            parents[i] = None
            indents[i] = None

        for block in self._iter_blocks(self.lines):
            if not isinstance(block, str):
                pending_list_start = None
                text_data, caption_data, in_list, last_list_item, list_continuation = (
                    self._add_block(
                        doc=doc,
                        block=block,
                        text_data=text_data,
                        caption_data=caption_data,
                        in_list=in_list,
                        last_list_item=last_list_item,
                        list_continuation=list_continuation,
                        parents=parents,
                    )
                )
                continue

            line = block
            stripped_line = line.strip()
            if in_delimited_table and self._continue_table(table_data, line):
                continue

            in_list, last_list_item, list_continuation = self._close_list_if_needed(
                line=line,
                in_list=in_list,
                parents=parents,
                last_list_item=last_list_item,
                list_continuation=list_continuation,
                is_continuation_block=False,
            )

            # Block attributes and anchors (e.g. "[cols=\"2,1\"]", "[NOTE]",
            # "[[some-id]]") are metadata that applies to the block that
            # follows; they are never body content, so skip them.
            if (
                re.match(r"^\[\[.+\]\]$", stripped_line)
                or re.match(r"^\[[^\[\]]+\]$", stripped_line)
                or re.match(r"^:[\w.-]+:(\s.*)?$", stripped_line)
            ):
                start_match = _LIST_START_ATTR_RE.match(stripped_line)
                pending_list_start = (
                    int(start_match.group(1)) if start_match else pending_list_start
                )
                continue

            list_start, pending_list_start = self._take_list_start(
                line, pending_list_start
            )

            # Title
            if self._is_title(line):
                item = self._parse_title(line)
                level = item["level"]

                parents[level] = doc.add_text(
                    text=item["text"], label=DocItemLabel.TITLE
                )

            # Section headers
            elif self._is_section_header(line):
                # A heading ends the paragraph being accumulated: without
                # this flush the text before the heading is appended to the
                # text after it and lands in the wrong section.
                text_data = self._flush_text_data(
                    doc=doc,
                    text_data=text_data,
                    parent=self._get_current_parent(parents),
                )
                item = self._parse_section_header(line)
                level = item["level"]

                ancestor = next(
                    (parents[k] for k in range(level - 1, -1, -1) if parents[k]),
                    None,
                )
                parents[level] = doc.add_heading(
                    text=item["text"], level=item["level"], parent=ancestor
                )
                for k, v in parents.items():
                    if k > level:
                        parents[k] = None

            # Lists
            elif self._is_list_item(line):
                _log.debug(f"line: {line}")
                item = self._parse_list_item(line)
                _log.debug(f"parsed list-item: {item}")

                level = self._get_current_level(parents)

                if not in_list:
                    in_list = True
                    # GroupItem has no caption slot; emit the pending block
                    # title as a bold paragraph immediately before the list.
                    if caption_data:
                        self._flush_text_data(
                            doc=doc,
                            text_data=caption_data,
                            parent=parents[level],
                            bold=True,
                        )
                        caption_data = []

                    list_group = doc.add_group(
                        parent=parents[level], name="list", label=GroupLabel.LIST
                    )
                    parents[level + 1] = list_group
                    indents[level + 1] = item["indent"]
                    list_kinds = {level + 1: self._list_kind(item)}
                    if list_start is not None:
                        next_list_numbers[list_group.self_ref] = list_start

                elif in_list and item["indent"] > indents[level]:
                    # A nested list belongs to the item above it. As a sibling
                    # of the items, it is counted as an item of the outer list,
                    # and the next ordered item is numbered one too high.
                    parents[level + 1] = doc.add_group(
                        parent=last_list_item, name="list", label=GroupLabel.LIST
                    )
                    indents[level + 1] = item["indent"]
                    list_kinds[level + 1] = self._list_kind(item)

                elif in_list and item["indent"] < indents[level]:
                    while level > 0 and item["indent"] < indents[level]:
                        # Only pop the current level if there is an outer group
                        # to fall back to; otherwise keep it as the list root.
                        if indents[level - 1] is None:
                            break
                        parents[level] = None
                        indents[level] = None
                        level -= 1

                self._match_list_kind(
                    doc=doc,
                    item=item,
                    parents=parents,
                    indents=indents,
                    list_kinds=list_kinds,
                    last_list_item=last_list_item,
                )

                list_parent = self._get_current_parent(parents)
                last_list_item = doc.add_list_item(
                    item["text"],
                    enumerated=item["numbered"],
                    marker=self._list_item_marker(item, list_parent, next_list_numbers),
                    parent=list_parent,
                )
                list_continuation = False

            elif in_list and stripped_line == "+":
                list_continuation = True
                continue

            # Tables
            elif line.strip() == "|===":  # start or end of table
                if in_table:
                    self._add_table_if_nonempty(
                        doc, table_data, caption_data, self._get_current_parent(parents)
                    )
                    caption_data = []
                    in_table = False
                    in_delimited_table = False
                    table_data = []
                else:
                    in_table = True
                    in_delimited_table = True

            elif self._is_table_line(line):  # within a table
                in_table = True
                table_data.append(self._parse_table_cells(line))

            elif in_table and not self._is_table_line(line):  # end of table
                self._add_table_if_nonempty(
                    doc, table_data, caption_data, self._get_current_parent(parents)
                )
                caption_data = []
                in_table = False
                table_data = []

            # Picture
            elif self._is_picture(line):
                caption_data, in_list, last_list_item = self._add_picture_item(
                    doc=doc,
                    line=line,
                    caption_data=caption_data,
                    in_list=in_list,
                    last_list_item=last_list_item,
                    parents=parents,
                )
                list_continuation = False

            # Caption
            elif self._is_caption(line) and len(caption_data) == 0:
                item = self._parse_caption(line)
                caption_data.append(item["text"])

            elif len(line.strip()) > 0 and len(caption_data) > 0:
                # A block title is a single line, so this line starts the
                # paragraph it titles. A paragraph has no caption slot.
                parent = self._get_current_parent(parents)
                text_data = self._flush_text_data(
                    doc=doc, text_data=text_data, parent=parent
                )
                caption_data = self._flush_caption_as_paragraph(
                    doc, caption_data, parent
                )
                item = self._parse_text(line)
                text_data.append(item["text"])

            elif len(line.strip()) == 0:
                caption_data, text_data = self._flush_blank_line(
                    doc,
                    caption_data,
                    text_data,
                    self._get_current_parent(parents),
                )

            elif len(line.strip()) > 0:  # allow multiline texts
                item = self._parse_text(line)
                text_data.append(item["text"])

        text_data = self._flush_text_data(
            doc=doc, text_data=text_data, parent=self._get_current_parent(parents)
        )

        if in_table:
            # An unclosed table still consumes its pending caption below.
            self._add_table_if_nonempty(
                doc, table_data, caption_data, self._get_current_parent(parents)
            )
        elif caption_data:
            # A block title never followed by a floating element must still
            # reach the document.
            self._flush_caption_as_paragraph(
                doc, caption_data, self._get_current_parent(parents)
            )

        return doc

    @staticmethod
    def _get_current_level(parents):
        for k, v in parents.items():
            if v is None and k > 0:
                return k - 1

        return 0

    @staticmethod
    def _get_current_parent(parents):
        for k, v in parents.items():
            if v is None and k > 0:
                return parents[k - 1]

        return None

    _SOURCE_ATTR_RE = re.compile(r"^\[source(?:,\s*([\w+#.-]+))?[^\]]*\]$")
    _CONTENT_BLOCK_DELIMITERS = ("====", "****", "____", "--", "++++")

    @staticmethod
    def _has_matching_closer(lines: list[str], open_idx: int, delimiter: str) -> bool:
        """Return True if `delimiter` appears on a line of its own after `open_idx`."""
        for j in range(open_idx + 1, len(lines)):
            if lines[j].strip() == delimiter:
                return True
        return False

    @staticmethod
    def _is_line_comment(line: str) -> bool:
        """Is `line` a // comment line rather than text or a //// block delimiter?"""
        return _LINE_COMMENT_RE.match(line) is not None

    @staticmethod
    def _make_comment(comment_lines: list[str]) -> _CommentBlock | None:
        """Build a comment block, or None when the comment carries no text."""
        text = "\n".join(comment_lines).strip()
        return _CommentBlock(text=text) if text else None

    @classmethod
    def _iter_blocks(
        cls, lines: list[str]
    ) -> Iterator[str | _LiteralBlock | _CommentBlock]:
        block_data: list[str] | None = None
        block_delimiter: str | None = None
        block_language: str | None = None
        comment_data: list[str] | None = None
        comment_delimiter: str | None = None
        # Content block delimiters still waiting for their closer, innermost
        # last. The lines between a pair run through this loop like any other,
        # so a comment inside such a block is a comment and not body text.
        open_content_blocks: list[str] = []

        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            stripped = line.strip()

            if block_data is not None:
                # Inside a verbatim block: only the matching closer ends it,
                # and a "//" line here is content rather than a comment.
                if stripped == block_delimiter:
                    yield _LiteralBlock(
                        text="\n".join(block_data), language=block_language
                    )
                    block_data = None
                    block_delimiter = None
                    block_language = None
                else:
                    block_data.append(line.rstrip("\r\n"))
                i += 1
                continue

            # A comment block hides every construct, so it is checked ahead of
            # the block openers: a "----" between //// delimiters is comment
            # text, not the start of a listing.
            if comment_data is not None:
                if stripped == comment_delimiter:
                    comment = cls._make_comment(comment_data)
                    if comment is not None:
                        yield comment
                    comment_data = None
                    comment_delimiter = None
                else:
                    comment_data.append(line.rstrip("\r\n"))
                i += 1
                continue

            if _COMMENT_BLOCK_DELIMITER_RE.fullmatch(stripped):
                comment_data = []
                comment_delimiter = stripped
                i += 1
                continue

            if cls._is_line_comment(line):
                # Consecutive // lines read as one comment, so take the run.
                comment_lines = []
                while i < n and cls._is_line_comment(lines[i]):
                    comment_lines.append(lines[i].rstrip("\r\n")[2:].strip())
                    i += 1
                comment = cls._make_comment(comment_lines)
                if comment is not None:
                    yield comment
                continue

            if stripped in {"....", "----"}:
                # "...." is a literal block; "----" a listing block. A
                # "[source,lang]" attribute consumed just before a "----"
                # labels the listing's language.
                block_data = []
                block_delimiter = stripped
                block_language = None
                i += 1
                continue

            if open_content_blocks and stripped == open_content_blocks[-1]:
                # Closer of the innermost content block. Yield a trailing blank
                # so the following text starts a fresh paragraph.
                open_content_blocks.pop()
                yield ""
                i += 1
                continue

            if stripped in cls._CONTENT_BLOCK_DELIMITERS:
                if cls._has_matching_closer(lines, i, stripped):
                    open_content_blocks.append(stripped)
                else:
                    # Stray delimiter (no closer): break the paragraph, do not
                    # render as text. The line must be consumed either way;
                    # leaving `i` on it loops over the same line forever.
                    yield ""
                i += 1
                continue

            source_attr = cls._SOURCE_ATTR_RE.match(stripped)
            if source_attr is not None and i + 1 < n and lines[i + 1].strip() == "----":
                block_data = []
                block_delimiter = "----"
                block_language = source_attr.group(1)
                i += 2
                continue

            yield line
            i += 1

        if block_data is not None:
            yield _LiteralBlock(text="\n".join(block_data), language=block_language)
        if comment_data is not None:
            # An unclosed //// comments out the rest of the document, which is
            # how Asciidoctor reads it.
            comment = cls._make_comment(comment_data)
            if comment is not None:
                yield comment

    def _add_picture_item(
        self,
        *,
        doc: DoclingDocument,
        line: str,
        caption_data: list[str],
        in_list: bool,
        last_list_item: ListItem | None,
        parents: dict[int, GroupItem | None],
    ) -> tuple[list[str], bool, ListItem | None]:
        """Add a picture with its pending caption, if any."""
        caption: Optional[TextItem] = None
        if len(caption_data) > 0:
            text, hyperlink = self._caption_text(caption_data)
            caption = doc.add_text(
                text=text, label=DocItemLabel.CAPTION, hyperlink=hyperlink
            )

        caption_data = []

        item = self._parse_picture(line)

        image: Optional[ImageRef] = None
        if "uri" in item and self.options.fetch_images:
            base_path = (
                str(self.options.source_uri)
                if self.options.source_uri is not None
                else None
            )
            image = self._image_loader.load_image_ref(item["uri"], base_path)
        doc.add_picture(
            image=image,
            caption=caption,
            parent=last_list_item if in_list else self._get_current_parent(parents),
        )
        return caption_data, in_list, last_list_item

    def _add_block(
        self,
        *,
        doc: DoclingDocument,
        block: _LiteralBlock | _CommentBlock,
        text_data: list[str],
        caption_data: list[str],
        in_list: bool,
        last_list_item: ListItem | None,
        list_continuation: bool,
        parents: dict[int, GroupItem | None],
    ) -> tuple[list[str], list[str], bool, ListItem | None, bool]:
        """Add a delimited block and return the parser state after it.

        A comment is markup, not content: it neither ends the list nor the
        paragraph it sits in, so the state passes through unchanged. A
        literal block flushes the pending text and captions first.
        """
        if isinstance(block, _CommentBlock):
            self._add_comment_block(
                doc=doc,
                block=block,
                in_list=in_list,
                last_list_item=last_list_item,
                parents=parents,
            )
            return text_data, caption_data, in_list, last_list_item, list_continuation
        return self._add_literal_block(
            doc=doc,
            block=block,
            text_data=text_data,
            caption_data=caption_data,
            in_list=in_list,
            last_list_item=last_list_item,
            list_continuation=list_continuation,
            parents=parents,
        )

    def _add_comment_block(
        self,
        *,
        doc: DoclingDocument,
        block: _CommentBlock,
        in_list: bool,
        last_list_item: ListItem | None,
        parents: dict[int, GroupItem | None],
    ) -> None:
        """Add a comment to the notes content layer.

        A comment between two lines of one paragraph lands just before that
        paragraph, since flushing the text to place it would split the body
        text in two.
        """
        if not self.options.capture_comments:
            return
        doc.add_text(
            text=block.text,
            label=DocItemLabel.TEXT,
            content_layer=ContentLayer.NOTES,
            parent=(last_list_item if in_list else self._get_current_parent(parents)),
        )

    def _add_literal_block(
        self,
        *,
        doc: DoclingDocument,
        block: _LiteralBlock,
        text_data: list[str],
        caption_data: list[str],
        in_list: bool,
        last_list_item: ListItem | None,
        list_continuation: bool,
        parents: dict[int, GroupItem | None],
    ) -> tuple[list[str], list[str], bool, ListItem | None, bool]:
        """Add a literal/listing block, flushing pending text and captions."""
        in_list, last_list_item, list_continuation = self._close_list_if_needed(
            line="<literal-block>",
            in_list=in_list,
            parents=parents,
            last_list_item=last_list_item,
            list_continuation=list_continuation,
            is_continuation_block=True,
        )
        text_data = self._flush_text_data(
            doc=doc,
            text_data=text_data,
            parent=self._get_current_parent(parents),
        )
        caption: Optional[TextItem] = None
        if caption_data:
            text, hyperlink = self._caption_text(caption_data)
            caption = doc.add_text(
                text=text, label=DocItemLabel.CAPTION, hyperlink=hyperlink
            )
            caption_data = []
        code_language = (
            detect_code_language(block.text, hint=block.language)
            if block.language is not None
            else None
        )
        doc.add_code(
            text=block.text,
            code_language=code_language,
            caption=caption,
            parent=(last_list_item if in_list else self._get_current_parent(parents)),
        )
        return text_data, caption_data, in_list, last_list_item, False

    @classmethod
    def _close_list_if_needed(
        cls,
        *,
        line: str,
        in_list: bool,
        parents: dict[int, GroupItem | None],
        last_list_item: ListItem | None,
        list_continuation: bool,
        is_continuation_block: bool,
    ) -> tuple[bool, ListItem | None, bool]:
        if (
            not in_list
            or cls._is_list_item(line)
            or line.strip() in {"", "+"}
            or (list_continuation and (is_continuation_block or cls._is_picture(line)))
        ):
            return in_list, last_list_item, list_continuation

        # Close every level of the list, not only the innermost one. Otherwise
        # the text after a nested list is added to the outer list group.
        level = cls._get_current_level(parents)
        while (
            isinstance(group := parents[level], GroupItem)
            and group.label == GroupLabel.LIST
        ):
            parents[level] = None
            level -= 1
        return False, None, False

    @classmethod
    def _match_list_kind(
        cls,
        *,
        doc: DoclingDocument,
        item: dict,
        parents: dict[int, GroupItem | None],
        indents: dict[int, GroupItem | None],
        list_kinds: dict[int, str],
        last_list_item: ListItem | None,
    ) -> None:
        """Keep items with different list markers out of the same list group.

        AsciiDoc nests lists by marker, not by indentation. The item returns to
        the closest enclosing list with its marker at the same indent, or else
        opens a nested list under the item above it.
        """
        level = cls._get_current_level(parents)
        kind = cls._list_kind(item)
        if list_kinds.get(level) == kind:
            return
        outer = level - 1
        while outer in list_kinds and indents[outer] == item["indent"]:
            if list_kinds[outer] == kind:
                for closed in range(outer + 1, level + 1):
                    parents[closed] = None
                    indents[closed] = None
                return
            outer -= 1
        parents[level + 1] = doc.add_group(
            parent=last_list_item, name="list", label=GroupLabel.LIST
        )
        indents[level + 1] = item["indent"]
        list_kinds[level + 1] = kind

    @classmethod
    def _take_list_start(
        cls, line: str, pending_list_start: int | None
    ) -> tuple[int | None, int | None]:
        """Return the list start that applies to this line, and the one to keep.

        Block attributes apply to the next block, past blank lines and its
        block title.
        """
        if line.strip() and not cls._is_caption(line):
            return pending_list_start, None
        return pending_list_start, pending_list_start

    @staticmethod
    def _list_item_marker(
        item: dict,
        parent: NodeItem | None,
        next_list_numbers: dict[str, int],
    ) -> str | None:
        """Return the marker of a list item, numbered from the list start."""
        if (
            item["numbered"]
            and parent is not None
            and parent.self_ref in next_list_numbers
        ):
            number = next_list_numbers[parent.self_ref]
            next_list_numbers[parent.self_ref] = number + 1
            return f"{number}."
        return item["marker"] if item["marker"][:-1].isdigit() else None

    @staticmethod
    def _list_kind(item: dict) -> str:
        # All ordered markers ("." and "1.") count as one kind; a bullet's kind
        # is its marker character, so "*" and "-" lists nest like Asciidoctor.
        return "." if item["numbered"] else item["marker"][0]

    @staticmethod
    def _caption_text(
        caption_data: list[str],
    ) -> tuple[str, AnyUrl | Path | None]:
        """Return the text and hyperlink of a caption.

        A caption is one text item, so runs cannot be kept apart. The
        hyperlink is kept only if every non-blank run has the same one.
        """
        runs = AsciiDocBackend._parse_inline(" ".join(caption_data))
        text = "".join(run.text for run in runs).strip()
        targets = {run.hyperlink for run in runs if run.text.strip()}
        if len(targets) == 1 and (target := targets.pop()) is not None:
            return text, _LINK_TARGET.validate_python(target)
        return text, None

    @staticmethod
    def _join_lines(lines: list[str]) -> str:
        """Join the lines of a paragraph with spaces.

        A line ending in " +" is an AsciiDoc hard line break, so the next line
        starts on a new line, as with a hard break in Markdown.
        """
        text = ""
        for i, line in enumerate(lines):
            if i == len(lines) - 1:
                text += line
            elif line.endswith(" +"):
                text += line[:-2].rstrip() + "\n"
            else:
                text += line + " "
        return text

    @staticmethod
    def _flush_text_data(
        *,
        doc: DoclingDocument,
        text_data: list[str],
        parent: GroupItem | None,
        bold: bool = False,
    ) -> list[str]:
        if len(text_data) == 0:
            return []
        text = AsciiDocBackend._join_lines(text_data)
        runs = AsciiDocBackend._parse_inline(text, bold=bold)
        if not any(
            run.bold != bold or run.italic or run.code or run.hyperlink for run in runs
        ):
            doc.add_text(
                text="".join(run.text for run in runs) if bold else text,
                label=DocItemLabel.PARAGRAPH,
                parent=parent,
                formatting=Formatting(bold=True) if bold else None,
            )
            return []
        # Mixed formatting and links need one item per run, kept together in an
        # inline group like the Markdown backend does.
        # Runs keep their boundary whitespace (docling-core joins them with ""); a
        # code run's edge whitespace moves out into a plain run of its own.
        spaced: list[_InlineRun] = []
        for run in runs:
            core = run.text.strip()
            if not run.code or not core:
                spaced.append(replace(run, code=False))
                continue
            lead, _, trail = run.text.partition(core)
            for part in (_InlineRun(lead), replace(run, text=core), _InlineRun(trail)):
                if part.text:
                    spaced.append(part)
        trimmed = trim_run_edges(
            spaced,
            get_text=lambda run: run.text,
            with_text=lambda run, text: replace(run, text=text),
        )
        group = doc.add_inline_group(parent=parent)
        for run in trimmed:
            hyperlink = (
                _LINK_TARGET.validate_python(run.hyperlink) if run.hyperlink else None
            )
            if run.code:
                doc.add_code(text=run.text, parent=group, hyperlink=hyperlink)
            else:
                doc.add_text(
                    text=run.text,
                    label=DocItemLabel.TEXT,
                    parent=group,
                    formatting=(
                        Formatting(bold=run.bold, italic=run.italic)
                        if run.bold or run.italic
                        else None
                    ),
                    hyperlink=hyperlink,
                )
        return []

    @staticmethod
    def _parse_inline(
        text: str,
        bold: bool = False,
        italic: bool = False,
        hyperlink: str | None = None,
    ) -> list[_InlineRun]:
        """Split text into runs of plain, bold, italic, monospace and link text.

        `hyperlink` is set inside the text of a link. A link has no links in
        it, so a link form found there stays text of the outer link.
        """
        runs: list[_InlineRun] = []
        position = 0
        for match in _INLINE_FORMAT_RE.finditer(text):
            if match.start() > position:
                runs.append(
                    _InlineRun(
                        text[position : match.start()],
                        bold=bold,
                        italic=italic,
                        hyperlink=hyperlink,
                    )
                )
            if (link := AsciiDocBackend._match_link(match)) is not None:
                target, link_text = link
                if hyperlink is not None:
                    runs.append(
                        _InlineRun(match.group(0), bold, italic, hyperlink=hyperlink)
                    )
                elif link_text is None:
                    # No link text: the target itself is shown, as written.
                    shown = match.group("macro_target") or target
                    runs.append(_InlineRun(shown, bold, italic, hyperlink=target))
                    # Blank brackets have no link text, but their whitespace stays.
                    blank = match.group("url_text") or match.group("macro_text") or ""
                    if blank.isspace():
                        runs.append(_InlineRun(blank, bold, italic))
                else:
                    runs.extend(
                        AsciiDocBackend._parse_inline(link_text, bold, italic, target)
                    )
            elif (code := match.group("umono") or match.group("mono")) is not None:
                runs.append(_InlineRun(code, code=True, hyperlink=hyperlink))
            elif (inner := match.group("ubold") or match.group("bold")) is not None:
                runs.extend(
                    AsciiDocBackend._parse_inline(inner, True, italic, hyperlink)
                )
            else:
                inner = match.group("uitalic") or match.group("italic")
                runs.extend(AsciiDocBackend._parse_inline(inner, bold, True, hyperlink))
            position = match.end()
        if position < len(text):
            runs.append(
                _InlineRun(
                    text[position:], bold=bold, italic=italic, hyperlink=hyperlink
                )
            )
        return runs

    @staticmethod
    def _match_link(match: re.Match[str]) -> tuple[str, str | None] | None:
        """Return the target and the link text of a link match.

        The link text is None when the link has none, and then Asciidoctor
        shows the target. Returns None when the match is not a link.
        """
        if url := match.group("url"):
            return url, AsciiDocBackend._link_text(match.group("url_text"))
        if url := match.group("angle_url") or match.group("bare_url"):
            return url, None
        if target := match.group("macro_target"):
            if match.group("mailto"):
                target = f"mailto:{target}"
            return target, AsciiDocBackend._link_text(match.group("macro_text"))
        return None

    @staticmethod
    def _link_text(attributes: str) -> str | None:
        """Get the link text from the attribute list in the brackets of a link.

        As in Asciidoctor, named attributes (e.g. `window=_blank`) can follow
        the text, the text is then quoted if it contains a comma, and a final
        `^` (open in a new window) is not part of the text.
        """
        text = re.sub(r"\\([\[\]])", r"\1", attributes)
        if "=" in text:
            quoted = re.match(r'\s*"((?:[^"\\]|\\.)*)"\s*(?:,|$)', text)
            if quoted:
                text = quoted.group(1).replace('\\"', '"')
            else:
                text = text.split(",", 1)[0].strip()
                if re.match(r"[A-Za-z_][\w-]*\s*=", text):
                    # Only named attributes, no link text.
                    text = ""
        if text.endswith("^"):
            text = text[:-1]
        return text if text.strip() else None

    #   =========   Title
    @staticmethod
    def _is_title(line):
        return re.match(r"^= ", line)

    @staticmethod
    def _strip_closing_marker(text: str, marker: str) -> str:
        """Drop the closing marks of a symmetric title such as "== Title ==".

        Asciidoctor accepts a title closed by the same marks that open it,
        separated by whitespace, and leaves them out of the title text.
        """
        return re.sub(rf"[ \t]+{re.escape(marker)}$", "", text)

    @staticmethod
    def _parse_title(line):
        text = AsciiDocBackend._strip_closing_marker(line[2:].strip(), "=")
        return {"type": "title", "text": text, "level": 0}

    #   =========   Section headers
    @staticmethod
    def _is_section_header(line):
        return re.match(r"^==+\s+", line)

    @staticmethod
    def _parse_section_header(line):
        match = re.match(r"^(=+)\s+(.*)", line)

        marker = match.group(1)  # The list marker (e.g., "*", "-", "1.")
        text = match.group(2)  # The actual text of the list item

        header_level = marker.count("=")  # number of '=' represents level
        return {
            "type": "header",
            "level": header_level - 1,
            "text": AsciiDocBackend._strip_closing_marker(text.strip(), marker),
        }

    #   =========   Lists
    @staticmethod
    def _is_list_item(line):
        return re.match(_LIST_ITEM_PATTERN, line)

    @staticmethod
    def _parse_list_item(line):
        """Extract the item marker (number or bullet symbol) and the text of the item."""

        match = re.match(_LIST_ITEM_PATTERN, line)
        if match:
            indent = match.group(1)
            marker = match.group(2)  # The list marker (e.g., "*", "-", "1.")
            text = match.group(3)  # The actual text of the list item
            indent_width = len(indent)
            if marker.startswith((".", "*")):
                indent_width += len(marker) - 1

            if marker.startswith("*") or marker == "-":
                return {
                    "type": "list_item",
                    "marker": marker,
                    "text": text.strip(),
                    "numbered": False,
                    "indent": indent_width,
                }
            else:
                return {
                    "type": "list_item",
                    "marker": marker,
                    "text": text.strip(),
                    "numbered": True,
                    "indent": indent_width,
                }
        else:
            # Fallback if no match
            return {
                "type": "list_item",
                "marker": "-",
                "text": line,
                "numbered": False,
                "indent": 0,
            }

    #   =========   Tables
    @staticmethod
    def _is_table_line(line):
        # A cell may be the only one on its line: vertical AsciiDoc tables
        # commonly write one cell per line, so a second "|" is not required.
        return re.match(rf"^{_CELL_SPEC}\|", line) is not None

    @classmethod
    def _continue_table(
        cls, table_data: list[list[_TableCellSource]], line: str
    ) -> bool:
        """Consume a line between |=== delimiters that is not a row.

        A blank line only separates rows, and a line without a leading "|"
        continues the cell above it. Returns False for any other line.
        """
        stripped = line.strip()
        if stripped == "|===" or cls._is_table_line(line):
            return False
        if stripped and not table_data:
            return False
        if stripped:
            cell = table_data[-1][-1]
            cell.text = f"{cell.text} {stripped}".lstrip()
        return True

    @staticmethod
    def _parse_table_cells(line: str) -> list[_TableCellSource]:
        """Split a table line into its cells, reading each cell's spans.

        A specifier glued to a "|" (e.g. "^.^h", "2+") belongs to the cell that
        "|" opens. It is anchored to whitespace, so content ending in a style
        letter (e.g. "Eth") survives. A "\\|" is a pipe in the cell text and
        loses its backslash.
        """
        pipes = [match.start() for match in re.finditer(r"(?<!\\)\|", line)]
        specs: list[str] = []
        texts: list[str] = []
        for index, pipe in enumerate(pipes):
            start = pipes[index - 1] + 1 if index else 0
            segment = line[start:pipe]
            anchor = r"(?:^|\s)" if index == 0 else r"\s"
            match = re.search(rf"{anchor}({_CELL_SPEC})$", segment)
            spec = match.group(1) if match else ""
            specs.append(spec)
            if index:
                texts.append(segment[: len(segment) - len(spec)])
        if pipes:
            texts.append(line[pipes[-1] + 1 :])

        cells = []
        for spec, text in zip(specs, texts):
            span = _CELL_SPAN_RE.match(spec)
            repeat = _CELL_REPEAT_RE.match(spec)
            cells.append(
                _TableCellSource(
                    text=text.replace("\\|", "|").strip(),
                    col_span=int(span.group("col") or 1) if span else 1,
                    row_span=int(span.group("row") or 1) if span else 1,
                    repeat=int(repeat.group("repeat")) if repeat else 1,
                )
            )
        return cells

    @staticmethod
    def _add_table_if_nonempty(
        doc: DoclingDocument,
        table_data: list[list[_TableCellSource]],
        caption_data: list[str],
        parent: NodeItem | None,
    ) -> None:
        if not table_data:
            return

        caption = None
        if caption_data:
            text, hyperlink = AsciiDocBackend._caption_text(caption_data)
            caption = doc.add_text(
                text=text, label=DocItemLabel.CAPTION, hyperlink=hyperlink
            )

        data = AsciiDocBackend._populate_table_as_grid(table_data)
        doc.add_table(data=data, parent=parent, caption=caption)

    @staticmethod
    def _group_table_rows(
        table_data: list[list[_TableCellSource]],
    ) -> list[list[_TableCellSource]]:
        """Group the cells of a table into its rows.

        Asciidoctor takes the column count from the first line of the table and
        fills each row with that many columns, so a row may be written on
        several lines (one cell per line is common). A line that holds a whole
        row on its own stays one row, also when it is wider. Columns that a
        rowspan from a row above covers count as filled. A cell with a "3*"
        factor counts as three cells, expanded only here so that a line
        continuing its text applies to every copy.
        """
        if not table_data:
            return []
        table_data = [
            [replace(cell, repeat=1) for cell in line for _ in range(cell.repeat)]
            for line in table_data
        ]
        num_cols = sum(cell.col_span for cell in table_data[0])
        rows: list[list[_TableCellSource]] = []
        covered: dict[int, int] = {}
        row: list[_TableCellSource] = []
        width = 0

        def close_row() -> None:
            nonlocal row, width
            for cell in row:
                for offset in range(1, cell.row_span):
                    target = len(rows) + offset
                    covered[target] = covered.get(target, 0) + cell.col_span
            rows.append(row)
            row, width = [], 0

        for line_cells in table_data:
            needed = num_cols - covered.get(len(rows), 0)
            if not row and sum(cell.col_span for cell in line_cells) >= needed:
                row = list(line_cells)
                close_row()
                continue
            for cell in line_cells:
                row.append(cell)
                width += cell.col_span
                if width >= num_cols - covered.get(len(rows), 0):
                    close_row()
        if row:
            close_row()
        return rows

    @classmethod
    def _populate_table_as_grid(
        cls, table_data: list[list[_TableCellSource]]
    ) -> TableData:
        """Place each cell on the grid, with its spans.

        A cell takes the first column of its row that no cell to its left and
        no rowspan from a row above already covers. A rowspan is cut at the
        last row of the table.
        """
        rows = cls._group_table_rows(table_data)
        num_rows = len(rows)
        occupied: set[tuple[int, int]] = set()
        cells: list[TableCell] = []
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for source in row:
                while (row_idx, col_idx) in occupied:
                    col_idx += 1
                row_span = min(source.row_span, num_rows - row_idx)
                col_span = source.col_span
                for r in range(row_idx, row_idx + row_span):
                    for c in range(col_idx, col_idx + col_span):
                        occupied.add((r, c))
                cells.append(
                    TableCell(
                        text=source.text,
                        row_span=row_span,
                        col_span=col_span,
                        start_row_offset_idx=row_idx,
                        end_row_offset_idx=row_idx + row_span,
                        start_col_offset_idx=col_idx,
                        end_col_offset_idx=col_idx + col_span,
                        column_header=row_idx == 0,
                        row_header=False,
                    )
                )
                col_idx += col_span

        num_cols = max((cell.end_col_offset_idx for cell in cells), default=0)
        return TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)

    #   =========   Pictures
    @staticmethod
    def _is_picture(line):
        return re.match(r"^image::", line)

    @staticmethod
    def _parse_picture(line):
        """
        Parse an image macro, extracting its path and attributes.
        Syntax: image::path/to/image.png[Alt Text, width=200, height=150, align=center]
        """
        mtch = re.match(r"^image::(.+)\[(.*)\]$", line)
        if mtch:
            picture_path = mtch.group(1).strip()
            attributes = mtch.group(2).split(",")
            picture_info = {"type": "picture", "uri": picture_path}

            # Extract optional attributes (alt text, width, height, alignment)
            if attributes:
                alt_parts = [attributes[0].strip()] if attributes[0] else [""]
                for attr in attributes[1:]:
                    if "=" in attr:
                        key, value = attr.split("=", 1)
                        picture_info[key.strip()] = value.strip()
                    else:
                        alt_parts.append(attr.strip())
                picture_info["alt"] = ", ".join(alt_parts)

            return picture_info

        return {"type": "picture", "uri": line}

    #   =========   Captions
    @staticmethod
    def _is_caption(line):
        return re.match(r"^\.(\S.*)", line)

    @staticmethod
    def _parse_caption(line):
        mtch = re.match(r"^\.(.+)", line)
        if mtch:
            text = mtch.group(1)
            return {"type": "caption", "text": text}

        return {"type": "caption", "text": ""}

    #   =========   Plain text
    @staticmethod
    def _parse_text(line):
        return {"type": "text", "text": line.strip()}
