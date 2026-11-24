# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""GitHub HTML blocks can contain blank lines in nested preformatted snippets.

This module is loaded only during GitHub conversion, after the Markdown backend
has checked its optional dependencies.
"""

import re
from collections.abc import Callable
from html.parser import HTMLParser

from bs4 import BeautifulSoup
from docling_core.types.doc import DocItemLabel, DoclingDocument, InlineGroup, ListItem
from marko import Markdown
from marko.block import BlockElement, Document, HTMLBlock, Paragraph
from marko.element import Element
from marko.helpers import MarkoExtension
from marko.inline import Image, InlineElement, Link
from marko.source import Source
from typing_extensions import override

from docling.backend.md_backend import MarkdownDocumentBackend


class _HTMLBoundary(HTMLParser):
    def __init__(self, tag: str) -> None:
        super().__init__(convert_charrefs=False)
        self.tag = tag
        self.depth = 0
        self.started = False

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == self.tag:
            self.depth += 1
            self.started = True

    @override
    def handle_endtag(self, tag: str) -> None:
        if tag == self.tag:
            self.depth -= 1


class _GitHubHTMLBlock(HTMLBlock):
    override = True

    @classmethod
    def match(cls, source: Source) -> int | bool:
        matched = super().match(source)
        line = source.next_line() or ""
        tag = (
            re.match(r"(?i) {0,3}<(details|table|div)(?:\s|>)", line)
            if matched
            else None
        )
        source.context.github_html_tag = (
            tag.group(1).lower() if tag is not None else None
        )
        return matched

    @classmethod
    def parse(cls, source: Source) -> str:
        tag = source.context.github_html_tag
        if tag is None:
            return super().parse(source)
        boundary = _HTMLBoundary(tag)
        lines: list[str] = []
        while not source.exhausted:
            line = source.next_line()
            if line is None:
                break
            lines.append(line)
            boundary.feed(line)
            source.consume()
            if boundary.started and boundary.depth <= 0:
                break
        return "".join(lines)


class GitHubMarkdownBackend(MarkdownDocumentBackend):
    link_resolver: Callable[[str], str] | None = None

    @override
    def _prepare_markdown_ast(self, element: Document) -> None:
        if self.link_resolver is None:
            return
        pending: list[Element] = [element]
        while pending:
            node = pending.pop()
            if isinstance(node, Image):
                # Associate alt text with the picture instead of leaving it as
                # a neighbouring paragraph. The original syntax stays in metadata.
                def alt_text(element: Element) -> str:
                    if not isinstance(element, BlockElement | InlineElement):
                        return ""
                    children = element.children
                    return (
                        children
                        if isinstance(children, str)
                        else "".join(alt_text(child) for child in children)
                    )

                node.title = node.title or alt_text(node)
                node.children = []
            elif isinstance(node, Link):
                node.dest = self.link_resolver(node.dest)
            if isinstance(node, HTMLBlock):
                soup = BeautifulSoup(node.body, "html.parser")
                for anchor in soup.find_all("a", href=True):
                    target = anchor.get("href")
                    if isinstance(target, str):
                        anchor["href"] = self.link_resolver(target)
                node.body = soup.decode(formatter="minimal")
            if isinstance(node, BlockElement | InlineElement) and not isinstance(
                node.children, str
            ):
                pending.extend(node.children)

    @property
    def _task_marker_prefix(self) -> str:
        return f"docling-task-{self.document_hash}-"

    @override
    def _get_list_item_marker(self, paragraph: Paragraph, marker: str) -> str:
        # Marko stores task-list state on the first paragraph of the list item.
        checked = getattr(paragraph, "checked", None)
        if checked is None:
            return marker
        # The Markdown-to-HTML round trip preserves the list marker.
        return f"{self._task_marker_prefix}{int(checked)}:{marker}"

    @override
    def convert(self) -> DoclingDocument:
        doc = super().convert()
        prefix = self._task_marker_prefix
        for item in list(doc.texts):
            if not isinstance(item, ListItem) or not item.marker.startswith(prefix):
                continue
            state, _, item.marker = item.marker.removeprefix(prefix).partition(":")
            group = item.children[0].resolve(doc) if item.children else None
            if not isinstance(group, InlineGroup):
                group = doc.add_inline_group(parent=item)
                item.children.insert(0, item.children.pop())
            doc.add_text(
                label=DocItemLabel.CHECKBOX_SELECTED
                if state == "1"
                else DocItemLabel.CHECKBOX_UNSELECTED,
                text=item.text,
                orig=item.orig,
                formatting=item.formatting,
                hyperlink=item.hyperlink,
                parent=group,
            )
            group.children.insert(0, group.children.pop())
            item.text = item.orig = ""
            item.formatting = item.hyperlink = None
        return doc

    @override
    def _make_markdown_parser(self) -> Markdown:
        parser = super()._make_markdown_parser()
        parser.use(MarkoExtension(elements=[_GitHubHTMLBlock]))
        return parser
