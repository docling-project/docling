# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Render parsed wikitext to the HTML structures consumed by Docling."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from html import escape, unescape
from urllib.parse import quote, urljoin, urlsplit

from docling.backend.xml.mediawiki import _tokens as tokens
from docling.backend.xml.mediawiki._parser import Node, Syntax, parse_wikitext

_LIST_CONTAINER = {"*": "ul", "#": "ol", ";": "dl", ":": "dl"}
_LIST_ITEM = {"*": "li", "#": "li", ";": "dt", ":": "dd"}
_BLOCK_TAGS = frozenset(
    {
        "blockquote",
        "dd",
        "div",
        "dl",
        "dt",
        "figure",
        "figcaption",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "ol",
        "p",
        "pre",
        "source",
        "syntaxhighlight",
        "table",
        "ul",
    }
)
_INLINE_TAGS = frozenset(
    {
        "abbr",
        "b",
        "big",
        "cite",
        "code",
        "del",
        "em",
        "i",
        "ins",
        "kbd",
        "mark",
        "q",
        "s",
        "samp",
        "small",
        "span",
        "strike",
        "strong",
        "sub",
        "sup",
        "tt",
        "u",
        "var",
    }
)
_TABLE_TAGS = frozenset({"caption", "thead", "tbody", "tfoot", "tr", "td", "th"})
_HIDDEN_TAGS = frozenset(
    {"categorytree", "includeonly", "references", "script", "style", "templatedata"}
)
_LINK_SCHEMES = frozenset({"http", "https", "ftp", "ftps", "mailto", "tel"})
_MAGIC_WORDS = (
    "__NOTOC__",
    "__FORCETOC__",
    "__TOC__",
    "__NOEDITSECTION__",
    "__INDEX__",
    "__NOINDEX__",
    "__NEWSECTIONLINK__",
    "__NONEWSECTIONLINK__",
)


@dataclass(frozen=True)
class _Tag:
    name: str
    attrs: dict[str, str]
    body: Sequence[Node]
    marker: str


@dataclass
class _ListFrame:
    container: str
    item: str | None = None


def _split(nodes: Sequence[Node], separator: type[tokens.Token]) -> list[list[Node]]:
    parts: list[list[Node]] = [[]]
    for node in nodes:
        if isinstance(node, separator):
            parts.append([])
        else:
            parts[-1].append(node)
    return parts


def _plain(nodes: Sequence[Node]) -> str:
    return "".join(
        str(node["text"])
        if isinstance(node, tokens.Text)
        else _entity(node)
        if isinstance(node, Syntax) and isinstance(node.start, tokens.HTMLEntityStart)
        else _plain(node.children)
        if isinstance(node, Syntax)
        else ""
        for node in nodes
    )


def _entity(node: Syntax) -> str:
    pieces = ["&"]
    for part in node.children:
        if isinstance(part, tokens.HTMLEntityNumeric):
            pieces.append("#")
        elif isinstance(part, tokens.HTMLEntityHex):
            pieces.append(part["char"])
        elif isinstance(part, tokens.Text):
            pieces.append(part["text"])
    pieces.append(";")
    return unescape("".join(pieces))


def _tag(node: Syntax) -> _Tag:
    head, *body = _split(node.children, tokens.TagCloseOpen)
    name_tokens, *attributes = _split(head, tokens.TagAttrStart)
    attrs: dict[str, str] = {}
    for attribute in attributes:
        name, *value = _split(attribute, tokens.TagAttrEquals)
        if value:
            attrs[_plain(name).lower()] = _plain(value[0])
    return _Tag(
        name=_plain(name_tokens).strip().lower(),
        attrs=attrs,
        body=_split(body[0], tokens.TagOpenClose)[0] if body else (),
        marker=node.start.get("wiki_markup") or "",
    )


def _list_marker(node: Node) -> str:
    if isinstance(node, Syntax) and isinstance(node.start, tokens.TagOpenOpen):
        marker = node.start.get("wiki_markup")
        if marker in _LIST_CONTAINER:
            return marker
    return ""


def _close_list(frame: _ListFrame) -> str:
    return (
        f"</{frame.item}>" if frame.item is not None else ""
    ) + f"</{frame.container}>"


def _list_line(line: list[Node]) -> tuple[str, list[Node]]:
    depth = 0
    markers: list[str] = []
    for node in line:
        marker = _list_marker(node)
        if not marker:
            break
        markers.append(marker)
        depth += 1
    return "".join(markers), line[depth:]


def _template_name(node: Syntax) -> str:
    name: list[Node] = []
    for child in node.children:
        if isinstance(child, tokens.TemplateParamSeparator):
            break
        name.append(child)
    return " ".join(_plain(name).replace("_", " ").lower().split())


def _is_infobox(name: str) -> bool:
    return (
        name.startswith("infobox ")
        or name == "infobox"
        or name
        in {
            "taxobox",
            "automatic taxobox",
            "speciesbox",
            "subspeciesbox",
        }
    )


def _is_block(node: Syntax) -> bool:
    if isinstance(node.start, tokens.TemplateOpen):
        return _is_infobox(_template_name(node))
    if isinstance(node.start, tokens.HeadingStart):
        return True
    if isinstance(node.start, tokens.TagOpenOpen):
        return not _list_marker(node) and _tag(node).name in _BLOCK_TAGS
    return False


def _lines(nodes: Sequence[Node]) -> Iterator[list[Node] | Syntax]:
    line: list[Node] = []
    for node in nodes:
        if isinstance(node, Syntax) and _is_block(node):
            if line:
                yield line
                line = []
            yield node
        elif isinstance(node, tokens.Text) and "\n" in node["text"]:
            first, *rest = node["text"].split("\n")
            if first:
                line.append(tokens.Text(text=first))
            for text in rest:
                yield line
                line = [tokens.Text(text=text)] if text else []
        else:
            line.append(node)
    if line:
        yield line


def _link(href: str, label: str, *, allow_relative: bool) -> str:
    try:
        scheme = urlsplit(href).scheme
    except ValueError:
        return label
    if scheme not in _LINK_SCHEMES and (scheme or not allow_relative):
        return label
    return f'<a href="{escape(href, quote=True)}">{label}</a>'


class _Renderer:
    def __init__(self, wiki_base: str | None) -> None:
        self.wiki_base = wiki_base

    def blocks(self, nodes: Sequence[Node]) -> str:
        output: list[str] = []
        paragraph: list[str] = []
        preformatted: list[str] = []
        lists: list[_ListFrame] = []

        def flush_paragraph() -> None:
            if paragraph:
                output.append("<p>" + "\n".join(paragraph).strip() + "</p>")
                paragraph.clear()

        def flush_preformatted() -> None:
            if preformatted:
                output.append("<pre>" + "\n".join(preformatted) + "</pre>")
                preformatted.clear()

        def flush_lists() -> None:
            while lists:
                output.append(_close_list(lists.pop()))

        for line in _lines(nodes):
            if isinstance(line, Syntax):
                flush_paragraph()
                flush_preformatted()
                flush_lists()
                output.append(self.inline(line))
                continue

            markers, content = _list_line(line)
            if markers:
                flush_paragraph()
                flush_preformatted()
                if markers.endswith(";"):
                    term, *definitions = self._definition(content)
                    self._add_list_item(output, lists, markers, term)
                    for definition in definitions:
                        self._add_list_item(
                            output, lists, markers[:-1] + ":", definition
                        )
                else:
                    self._add_list_item(output, lists, markers, self.inlines(content))
                continue

            flush_lists()
            rendered = self.inlines(content)
            if (
                content
                and isinstance(content[0], tokens.Text)
                and content[0]["text"].startswith(" ")
            ):
                flush_paragraph()
                preformatted.append(rendered.removeprefix(" "))
            else:
                flush_preformatted()
                if rendered.strip():
                    paragraph.append(rendered)
                else:
                    flush_paragraph()

        flush_lists()
        flush_preformatted()
        flush_paragraph()
        return "".join(output)

    def _definition(self, nodes: list[Node]) -> list[str]:
        parts: list[list[Node]] = [[]]
        for node in nodes:
            if _list_marker(node) == ":":
                parts.append([])
            else:
                parts[-1].append(node)
        return [self.inlines(part) for part in parts]

    @staticmethod
    def _add_list_item(
        output: list[str], lists: list[_ListFrame], markers: str, content: str
    ) -> None:
        containers = [_LIST_CONTAINER[marker] for marker in markers]
        common = 0
        for frame, container in zip(lists, containers):
            if frame.container != container:
                break
            common += 1
        while len(lists) > common:
            output.append(_close_list(lists.pop()))
        for depth in range(common, len(containers)):
            if lists and lists[-1].item is None:
                lists[-1].item = _LIST_ITEM[markers[depth - 1]]
                output.append(f"<{lists[-1].item}>")
            container = containers[depth]
            output.append(f"<{container}>")
            lists.append(_ListFrame(container))
        frame = lists[-1]
        if frame.item is not None:
            output.append(f"</{frame.item}>")
        frame.item = _LIST_ITEM[markers[-1]]
        output.append(f"<{frame.item}>{content.strip()}")

    def inlines(self, nodes: Sequence[Node]) -> str:
        return "".join(self.inline(node) for node in nodes)

    def inline(self, node: Node) -> str:
        if isinstance(node, tokens.Text):
            value = node["text"]
            for word in _MAGIC_WORDS:
                value = value.replace(word, "")
            return escape(value, quote=False)
        if not isinstance(node, Syntax):
            return ""
        if isinstance(node.start, tokens.HTMLEntityStart):
            return escape(_entity(node), quote=False)
        if isinstance(node.start, tokens.HeadingStart):
            level = node.start["level"]
            return f"<h{level}>{self.inlines(node.children).strip()}</h{level}>"
        if isinstance(node.start, tokens.WikilinkOpen):
            return self._wiki_link(node.children)
        if isinstance(node.start, tokens.ExternalLinkOpen):
            target, *label = _split(node.children, tokens.ExternalLinkSeparator)
            href = _plain(target).strip()
            visible = self.inlines(label[0]).strip() if label else escape(href)
            return _link(href, visible, allow_relative=href.startswith("//"))
        if isinstance(node.start, tokens.TemplateOpen):
            return self._template(node)
        if isinstance(node.start, tokens.ArgumentOpen):
            _, *default = _split(node.children, tokens.ArgumentSeparator)
            return self.inlines(default[0]) if default else ""
        if isinstance(node.start, tokens.TagOpenOpen):
            return self._render_tag(_tag(node))
        # Comments and unexpanded templates have no visible content.
        return ""

    def _template(self, node: Syntax) -> str:
        name = _template_name(node)
        supported = {
            "lang",
            "transl",
            "transliteration",
            "in lang",
            "small",
            "nowrap",
            "nobr",
            "center",
            "ubl",
            "unbulleted list",
            "plainlist",
            "hlist",
            "convert",
            "cvt",
            "birth date",
            "birth date and age",
            "death date",
            "death date and age",
            "start date",
            "end date",
            "sortname",
            "citation",
        }
        if not (
            _is_infobox(name)
            or name.startswith(("lang-", "cite "))
            or name in supported
        ):
            return ""
        parts = _split(node.children, tokens.TemplateParamSeparator)
        positional: list[str] = []
        named: list[tuple[str, str]] = []
        for part in parts[1:]:
            keys = _split(part, tokens.TemplateParamEquals)
            if len(keys) > 1:
                key = _plain(keys[0]).strip()
                value = self.inlines(part[len(keys[0]) + 1 :]).strip()
                if key.isdecimal():
                    positional.append(value)
                else:
                    named.append((key, value))
            else:
                positional.append(self.inlines(part).strip())
        if _is_infobox(name):
            rows = []
            for key, value in named:
                normalized = key.lower().strip().replace(" ", "_")
                if not value or normalized.endswith(("_style", "_class")):
                    continue
                if normalized in {"image", "image_size", "image_upright", "width"}:
                    continue
                label = escape(key.strip().replace("_", " "), quote=False)
                rows.append(f"<tr><th>{label}</th><td>{value}</td></tr>")
            return "<table>" + "".join(rows) + "</table>" if rows else ""
        if name in {"lang", "transl", "transliteration", "in lang"}:
            return positional[1] if len(positional) > 1 else ""
        if name.startswith("lang-") or name in {"small", "nowrap", "nobr", "center"}:
            return positional[0] if positional else ""
        if name in {"ubl", "unbulleted list", "plainlist", "hlist"}:
            return "; ".join(value for value in positional if value)
        if name in {"convert", "cvt"}:
            return " ".join(value for value in positional[:2] if value)
        if name in {
            "birth date",
            "birth date and age",
            "death date",
            "death date and age",
            "start date",
            "end date",
        }:
            return (
                "-".join(positional[:3])
                if len(positional) >= 3
                else " ".join(positional)
            )
        if name in {"sortname"}:
            return " ".join(positional[:2])
        if name.startswith("cite ") or name == "citation":
            fields = {key.lower().strip(): value for key, value in named}
            return ". ".join(
                fields[key]
                for key in (
                    "author",
                    "last",
                    "first",
                    "title",
                    "work",
                    "publisher",
                    "date",
                    "year",
                )
                if fields.get(key)
            )
        return ""

    def _wiki_link(self, nodes: list[Node]) -> str:
        target, *label = _split(nodes, tokens.WikilinkSeparator)
        title = _plain(target).strip()
        namespace, colon, local_title = title.partition(":")
        if colon and namespace.lower() == "category":
            return ""
        if colon and namespace.lower() in {"file", "image"}:
            caption = self.inlines(label[0]).rsplit("|", 1)[-1].strip() if label else ""
            return (
                caption
                if caption
                and caption not in {"thumb", "thumbnail", "frame", "frameless"}
                else escape(local_title)
            )
        title = title.removeprefix(":")
        visible = self.inlines(label[0]) if label and label[0] else escape(title)
        page, separator, fragment = title.partition("#")
        href = quote(page.replace(" ", "_"), safe="/")
        if separator:
            href = f"{href}#{quote(fragment.replace(' ', '_'))}"
        if self.wiki_base and page:
            href = urljoin(self.wiki_base, href)
        return _link(href, visible, allow_relative=True) if href else visible

    def _render_tag(self, tag: _Tag) -> str:
        name = tag.name
        if name in _HIDDEN_TAGS:
            return ""
        if name in {"hr", "br", "wbr"}:
            return f"<{name}>"
        if name in {"pre", "source", "syntaxhighlight"}:
            return f"<pre>{escape(_plain(tag.body), quote=False)}</pre>"
        if name in {"math", "chem", "ce"}:
            return f"<code>{escape(_plain(tag.body), quote=False)}</code>"
        if name == "table":
            return self._table(tag.body)
        content = self.inlines(tag.body)
        if name == "a":
            return _link(tag.attrs.get("href", ""), content, allow_relative=True)
        if name == "ref":
            return f"<sup>{content}</sup>" if content else ""
        if name in _BLOCK_TAGS or name in _INLINE_TAGS or name in _TABLE_TAGS:
            attrs = ""
            if name in {"td", "th"}:
                attrs = "".join(
                    f' {key}="{escape(tag.attrs[key], quote=True)}"'
                    for key in ("colspan", "rowspan")
                    if key in tag.attrs
                )
            return f"<{name}{attrs}>{content}</{name}>"
        return content

    def _table(self, nodes: Sequence[Node]) -> str:
        output: list[str] = []
        cells: list[str] = []
        for node in nodes:
            if not isinstance(node, Syntax) or not isinstance(
                node.start, tokens.TagOpenOpen
            ):
                continue
            tag = _tag(node)
            if tag.name in {"td", "th"}:
                content = self.inlines(tag.body)
                # The tokenizer represents the |+ caption as an implicit cell.
                if tag.marker == "|" and content.startswith("+"):
                    output.append(f"<caption>{content[1:].strip()}</caption>")
                else:
                    cells.append(self._render_tag(tag))
                continue
            if cells:
                output.append("<tr>" + "".join(cells) + "</tr>")
                cells.clear()
            output.append(self._render_tag(tag))
        if cells:
            output.append("<tr>" + "".join(cells) + "</tr>")
        return "<table>" + "".join(output) + "</table>"


def render_wikitext(source: str, wiki_base: str | None = None) -> str:
    """Render wiki structure and text without fetching resources."""
    return _Renderer(wiki_base).blocks(parse_wikitext(source))
