# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Render parsed wikitext to the HTML structures consumed by Docling."""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from html import escape, unescape
from typing import Literal
from urllib.parse import quote, urljoin, urlsplit

from docling.backend.xml.mediawiki import _tokens as tokens
from docling.backend.xml.mediawiki._emphasis import normalize_emphasis
from docling.backend.xml.mediawiki._parser import Node, Syntax, parse_wikitext
from docling.exceptions import DocumentLoadError

_LIST_CONTAINER = {"*": "ul", "#": "ol", ";": "dl", ":": "dl"}
_LIST_ITEM = {"*": "li", "#": "li", ";": "dt", ":": "dd"}
_EMPHASIS_ALIASES = {"strong": "b", "em": "i"}
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


@dataclass
class InlineReferenceBudget:
    limit: int
    used: int = 0

    def charge(self, size: int) -> None:
        if size > self.limit - self.used:
            raise DocumentLoadError(
                f"MediaWiki inline references exceed max_inline_reference_bytes ({self.limit})."
            )
        self.used += size


def _split(nodes: Sequence[Node], separator: type[tokens.Token]) -> list[list[Node]]:
    parts: list[list[Node]] = [[]]
    for node in nodes:
        if isinstance(node, separator):
            parts.append([])
        else:
            parts[-1].append(node)
    return parts


def _file_options(nodes: Sequence[Node]) -> list[list[Node]]:
    options: list[list[Node]] = [[]]
    for node in nodes:
        if isinstance(node, tokens.Text):
            first, *rest = node["text"].split("|")
            if first:
                options[-1].append(tokens.Text(text=first))
            options.extend([tokens.Text(text=value)] if value else [] for value in rest)
        else:
            options[-1].append(node)
    return options


def _is_file_option(value: str) -> bool:
    return (
        value
        in {
            "thumb",
            "thumbnail",
            "frame",
            "frameless",
            "border",
            "left",
            "right",
            "center",
            "none",
            "baseline",
            "sub",
            "super",
            "top",
            "text-top",
            "middle",
            "bottom",
            "text-bottom",
            "upright",
        }
        or value.startswith(("alt=", "link=", "class=", "lang=", "page=", "upright="))
        or re.fullmatch(r"(?:\d+(?:x\d+)?|x\d+)px", value) is not None
    )


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
        return _is_infobox(_template_name(node)) or _template_name(node) in {
            "reflist",
            "references",
        }
    if isinstance(node.start, tokens.HeadingStart):
        return True
    if isinstance(node.start, tokens.WikilinkOpen):
        target = next(
            (part for part in node.children if isinstance(part, tokens.Text)), None
        )
        return target is not None and target["text"].strip().lower().startswith(
            ("file:", "image:")
        )
    if isinstance(node.start, tokens.TagOpenOpen):
        return not _list_marker(node) and _tag(node).name in (
            _BLOCK_TAGS | {"references"}
        )
    return False


def _lines(nodes: Sequence[Node]) -> Iterator[list[Node] | Syntax]:
    line: list[Node] = []
    for node in nodes:
        if (
            isinstance(node, Syntax)
            and _is_block(node)
            and not (
                isinstance(node.start, tokens.WikilinkOpen)
                and line
                and _list_marker(line[0])
            )
        ):
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
    def __init__(
        self,
        wiki_base: str | None,
        reference_mode: Literal["omit", "inline", "list"],
        reference_budget: InlineReferenceBudget,
    ) -> None:
        self.wiki_base = wiki_base
        self.reference_mode = reference_mode
        self._reference_budget = reference_budget
        self._inline_references: dict[tuple[str, str], tuple[str, int]] = {}
        self._reference_nodes: dict[int, tuple[str, str]] = {}
        self._reference_bodies: dict[tuple[str, str], Sequence[Node]] = {}
        self._reference_order: dict[str, list[tuple[str, str]]] = {}
        self._reference_seen: set[tuple[str, str]] = set()
        self._reference_numbers: dict[tuple[str, str], int] = {}
        self._emitted_groups: set[str] = set()

    def collect_references(
        self, nodes: Sequence[Node], definitions: bool = False
    ) -> None:
        for node in nodes:
            if not isinstance(node, Syntax):
                continue
            if isinstance(node.start, tokens.TagOpenOpen):
                tag = _tag(node)
                if tag.name == "ref":
                    group = tag.attrs.get("group", "")
                    name = tag.attrs.get("name") or f"anonymous:{id(node)}"
                    key = (group, name)
                    if tag.body and key not in self._reference_bodies:
                        self._reference_bodies[key] = tag.body
                    if not definitions:
                        self._reference_nodes[id(node)] = key
                        order = self._reference_order.setdefault(group, [])
                        if key not in self._reference_seen:
                            order.append(key)
                            self._reference_seen.add(key)
                    continue
                if tag.name == "references":
                    self.collect_references(tag.body, definitions=True)
                    continue
            self.collect_references(node.children, definitions)

    def _reference_list(self, group: str = "") -> str:
        if self.reference_mode != "list" or group in self._emitted_groups:
            return ""
        keys = [
            key
            for key in self._reference_order.get(group, [])
            if key in self._reference_bodies
        ]
        if not keys:
            return ""
        self._emitted_groups.add(group)
        items = [
            f"<li>{self.inlines(self._reference_bodies[key])}</li>" for key in keys
        ]
        return "<ol>" + "".join(items) + "</ol>"

    def _reference(self, node: Syntax, tag: _Tag) -> str:
        if self.reference_mode == "omit":
            return ""
        key = self._reference_nodes.get(id(node))
        if key is None:
            return ""
        body = self._reference_bodies.get(key)
        if body is None:
            return ""
        if self.reference_mode == "inline":
            if key not in self._inline_references:
                content = self.inlines(body)
                rendered = f"<sup>{content}</sup>" if content else ""
                self._inline_references[key] = rendered, len(rendered.encode("utf-8"))
            rendered, size = self._inline_references[key]
            self._reference_budget.charge(size)
            return rendered
        number = self._reference_numbers[key]
        prefix = f"{escape(key[0])} " if key[0] else ""
        return f"<sup>[{prefix}{number}]</sup>"

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
        if self.reference_mode == "list":
            for group in self._reference_order:
                references = self._reference_list(group)
                if references:
                    output.extend(["<h2>References</h2>", references])
        # Retain a heading only if its section or one of its subsections has
        # visible content. Delay emission so chains of empty headings vanish.
        result: list[str] = []
        pending: list[tuple[int, str]] = []
        pending_markup: list[str] = []
        for part in output:
            heading = re.fullmatch(r"<h([1-6])>.*</h\1>", part, flags=re.DOTALL)
            if heading:
                pending_markup.clear()
                level = int(heading.group(1))
                while pending and pending[-1][0] >= level:
                    pending.pop()
                pending.append((level, part))
            else:
                if pending and not re.sub(r"<[^>]*>", "", part).strip():
                    pending_markup.append(part)
                    continue
                result.extend(text for _, text in pending)
                pending.clear()
                result.extend(pending_markup)
                pending_markup.clear()
                result.append(part)
        return "".join(result)

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
            tag = _tag(node)
            if tag.name == "ref":
                return self._reference(node, tag)
            if tag.name == "references":
                return self._reference_list(tag.attrs.get("group", ""))
            return self._render_tag(tag)
        # Comments and unexpanded templates have no visible content.
        return ""

    def _template(self, node: Syntax) -> str:
        name = _template_name(node)
        if name in {"reflist", "references"}:
            parts = _split(node.children, tokens.TemplateParamSeparator)
            group = ""
            for part in parts[1:]:
                fields = _split(part, tokens.TemplateParamEquals)
                if len(fields) > 1 and _plain(fields[0]).strip().lower() == "group":
                    group = _plain(fields[1]).strip()
            return self._reference_list(group)
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
            caption_nodes: Sequence[Node] = ()
            for option in _file_options(label[0]) if label else ():
                if option and (
                    not all(isinstance(part, tokens.Text) for part in option)
                    or not _is_file_option(_plain(option).strip().lower())
                ):
                    caption_nodes = option
            caption = self.inlines(caption_nodes).strip()
            image = f'<img alt="{escape(local_title.strip(), quote=True)}">'
            return (
                f"<figure>{image}<figcaption>{caption}</figcaption></figure>"
                if caption
                else f"<figure>{image}</figure>"
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

    def _render_tag(self, tag: _Tag, content: str | None = None) -> str:
        name = _EMPHASIS_ALIASES.get(tag.name, tag.name)
        if name in _HIDDEN_TAGS:
            return ""
        if name in {"hr", "br", "wbr"}:
            return f"<{name}>"
        if name in {"pre", "source", "syntaxhighlight"}:
            body = escape(_plain(tag.body), quote=False)
            language = tag.attrs.get("lang", "").strip()
            if language and name != "pre":
                return (
                    f'<pre class="language-{escape(language, quote=True)}">{body}</pre>'
                )
            return f"<pre>{body}</pre>"
        if name == "math":
            return (
                f'<span data-math-tex="">{escape(_plain(tag.body), quote=False)}</span>'
            )
        if name in {"chem", "ce"}:
            return f"<code>{escape(_plain(tag.body), quote=False)}</code>"
        if name == "table":
            return self._table(tag.body)
        if content is None:
            content = self.inlines(tag.body)
        if name == "a":
            return _link(tag.attrs.get("href", ""), content, allow_relative=True)
        if name == "ref":
            return f"<sup>{content}</sup>" if content else ""
        if name == "span":
            return content
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
                    cells.append(self._render_tag(tag, content))
                continue
            if cells:
                output.append("<tr>" + "".join(cells) + "</tr>")
                cells.clear()
            output.append(self._render_tag(tag))
        if cells:
            output.append("<tr>" + "".join(cells) + "</tr>")
        return "<table>" + "".join(output) + "</table>"


def render_wikitext(
    source: str,
    wiki_base: str | None = None,
    reference_mode: Literal["omit", "inline", "list"] = "omit",
    *,
    preserve_bold: bool = True,
    preserve_italic: bool = True,
    reference_budget: InlineReferenceBudget | None = None,
) -> str:
    """Render wiki structure and text without fetching resources."""
    nodes = parse_wikitext(source)
    if reference_budget is None:
        reference_budget = InlineReferenceBudget(16 * 1024 * 1024)
    renderer = _Renderer(wiki_base, reference_mode, reference_budget)
    if reference_mode != "omit":
        renderer.collect_references(nodes)
        for keys in renderer._reference_order.values():
            valid = [key for key in keys if key in renderer._reference_bodies]
            renderer._reference_numbers.update(
                (key, number) for number, key in enumerate(valid, 1)
            )
    return normalize_emphasis(
        renderer.blocks(nodes),
        preserve_bold=preserve_bold,
        preserve_italic=preserve_italic,
    )
