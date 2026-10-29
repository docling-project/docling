# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read iCalendar (RFC 5545) content into a tree of components."""

import re
from dataclasses import dataclass, field

_FOLD_BYTES = re.compile(rb"\r\n[ \t]|\n[ \t]|\r[ \t]")
_FOLD_TEXT = re.compile(r"\r\n[ \t]|\n[ \t]|\r[ \t]")
_LINE_BREAK = re.compile(r"\r\n|\r|\n")
_NAME = re.compile(r"[A-Za-z0-9_-]+")
_PARAM_VALUE = re.compile(r'[^";:,]*')
_CARET = re.compile(r"\^([n^'])")
_CARET_CHARS = {"n": "\n", "^": "^", "'": '"'}

# Properties whose value type is TEXT, so backslash escapes apply.
_TEXT_PROPERTIES = frozenset(
    {
        "ACTION",
        "CATEGORIES",
        "CLASS",
        "COMMENT",
        "CONTACT",
        "DESCRIPTION",
        "LOCATION",
        "METHOD",
        "NAME",
        "RELATED-TO",
        "RESOURCES",
        "STATUS",
        "SUMMARY",
        "TRANSP",
        "TZID",
        "TZNAME",
        "UID",
        "X-WR-CALDESC",
        "X-WR-CALNAME",
        "X-WR-TIMEZONE",
    }
)
# Properties that hold a comma-separated list of values.
_LIST_PROPERTIES = frozenset({"CATEGORIES", "EXDATE", "FREEBUSY", "RDATE", "RESOURCES"})

# Components that hold calendar entries; one of them never nests in another.
_ENTRY_COMPONENTS = frozenset(
    {"VAVAILABILITY", "VEVENT", "VFREEBUSY", "VJOURNAL", "VTIMEZONE", "VTODO"}
)
_RFC9073_COMPONENTS = frozenset({"PARTICIPANT", "VLOCATION", "VRESOURCE"})
_CHILDREN: dict[str, frozenset[str]] = {
    "VCALENDAR": _ENTRY_COMPONENTS,
    "VEVENT": frozenset({"VALARM"}) | _RFC9073_COMPONENTS,
    "VTODO": frozenset({"VALARM"}) | _RFC9073_COMPONENTS,
    "VJOURNAL": _RFC9073_COMPONENTS,
    "VFREEBUSY": _RFC9073_COMPONENTS,
    "VTIMEZONE": frozenset({"DAYLIGHT", "STANDARD"}),
    "VAVAILABILITY": frozenset({"AVAILABLE"}),
    "PARTICIPANT": frozenset({"VLOCATION", "VRESOURCE"}),
    "VALARM": frozenset(),
    "VLOCATION": frozenset(),
    "VRESOURCE": frozenset(),
    "STANDARD": frozenset(),
    "DAYLIGHT": frozenset(),
    "AVAILABLE": frozenset(),
}
_KNOWN_COMPONENTS = frozenset(_CHILDREN) | {"VCALENDAR"}


@dataclass
class Property:
    """A content line: its name, its parameters and its values.

    TEXT values are already unescaped, and list properties hold one entry per
    item.

    Attributes:
        name: The property name in upper case, for example ``DTSTART``.
        params: The parameters by upper-case name. A parameter with several
            values keeps them joined by commas.
        values: The values. Most properties have one; list properties such
            as CATEGORIES or EXDATE have one per item.
    """

    name: str
    params: dict[str, str] = field(default_factory=dict)
    values: list[str] = field(default_factory=list)

    @property
    def value(self) -> str:
        """The first value, or an empty string if there is none."""
        return self.values[0] if self.values else ""

    def param(self, name: str) -> str | None:
        """Return a parameter value.

        Args:
            name: The parameter name in upper case, for example ``TZID``.

        Returns:
            The parameter value, or ``None`` if the property does not have it.
        """
        return self.params.get(name)


@dataclass
class Component:
    """A BEGIN/END block with its properties and nested components.

    Attributes:
        name: The component name in upper case, for example ``VEVENT``.
        properties: The properties in file order.
        components: The nested components in file order, for example the
            VALARM blocks of an event.
    """

    name: str
    properties: list[Property] = field(default_factory=list)
    components: list["Component"] = field(default_factory=list)

    def get(self, name: str) -> Property | None:
        """Return the first property with a name.

        Args:
            name: The property name in upper case.

        Returns:
            The first matching property, or ``None`` if there is none.
        """
        return next((p for p in self.properties if p.name == name), None)

    def get_all(self, name: str) -> list[Property]:
        """Return every property with a name, in file order.

        Args:
            name: The property name in upper case.

        Returns:
            The matching properties; empty if there are none.
        """
        return [p for p in self.properties if p.name == name]

    def text(self, name: str) -> str:
        """Return the first value of a property.

        Args:
            name: The property name in upper case.

        Returns:
            The value, or an empty string if the property is missing.
        """
        prop = self.get(name)
        return prop.value if prop is not None else ""


def unfold_bytes(raw: bytes) -> bytes:
    """Join folded lines before decoding, since a fold may split a UTF-8 character.

    Args:
        raw: The file content in an ASCII-compatible encoding.

    Returns:
        The content with every line break that starts a fold removed.
    """
    return _FOLD_BYTES.sub(b"", raw)


def unfold_text(text: str) -> str:
    """Join folded lines of text that was decoded before unfolding.

    Args:
        text: The decoded file content.

    Returns:
        The text with every line break that starts a fold removed.
    """
    return _FOLD_TEXT.sub("", text)


def _can_contain(parent: str, child: str) -> bool:
    if child == "VCALENDAR":
        return False
    allowed = _CHILDREN.get(parent)
    if allowed is None:
        return child not in _ENTRY_COMPONENTS
    return child in allowed or child not in _KNOWN_COMPONENTS


def parse_components(text: str) -> list[Component]:
    """Build the component tree from unfolded text.

    A missing END is recovered from the containment rules: a BEGIN that the
    open component cannot hold closes it first, so one bad block does not
    swallow the entries that follow it. A stray END is ignored.

    Args:
        text: The unfolded file content.

    Returns:
        The top-level components in file order, usually one VCALENDAR.
    """
    roots: list[Component] = []
    stack: list[Component] = []
    # str.splitlines() would also break on U+2028 and similar inside a value.
    for line in _LINE_BREAK.split(text):
        prop = _parse_line(line)
        if prop is None:
            continue
        if prop.name in ("BEGIN", "END"):
            component_name = prop.value.strip().upper()
            if not component_name:
                continue
            if prop.name == "BEGIN":
                while stack and not _can_contain(stack[-1].name, component_name):
                    stack.pop()
                component = Component(name=component_name)
                (stack[-1].components if stack else roots).append(component)
                stack.append(component)
            elif any(c.name == component_name for c in stack):
                while stack.pop().name != component_name:
                    pass
            continue
        if stack:
            stack[-1].properties.append(prop)
    return roots


def _parse_line(line: str) -> Property | None:
    match = _NAME.match(line)
    if match is None:
        return None
    name = match.group().upper()
    pos = match.end()
    params: dict[str, str] = {}
    while pos < len(line) and line[pos] == ";":
        param_match = _NAME.match(line, pos + 1)
        if param_match is None:
            return None
        param_name = param_match.group().upper()
        pos = param_match.end()
        if pos >= len(line) or line[pos] != "=":
            # A vCalendar 1.0 parameter without a value, such as ;QUOTED-PRINTABLE.
            params[param_name] = ""
            continue
        values: list[str] = []
        pos += 1
        while True:
            if pos < len(line) and line[pos] == '"':
                end = line.find('"', pos + 1)
                if end < 0:
                    return None
                values.append(line[pos + 1 : end])
                pos = end + 1
            else:
                value_match = _PARAM_VALUE.match(line, pos)
                end = value_match.end() if value_match else pos
                values.append(line[pos:end])
                pos = end
            if pos < len(line) and line[pos] == ",":
                pos += 1
                continue
            break
        params[param_name] = ",".join(
            _CARET.sub(lambda m: _CARET_CHARS[m.group(1)], v) for v in values
        )
    if pos >= len(line) or line[pos] != ":":
        return None
    raw_value = line[pos + 1 :]
    is_text = name in _TEXT_PROPERTIES or params.get("VALUE", "").upper() == "TEXT"
    if name in _LIST_PROPERTIES:
        values = _split_list(raw_value, unescape=is_text)
    else:
        values = [_unescape(raw_value) if is_text else raw_value]
    return Property(name=name, params=params, values=values)


def _unescape(value: str) -> str:
    return _split_list(value, unescape=True, split=False)[0]


def _split_list(value: str, unescape: bool, split: bool = True) -> list[str]:
    """Split on commas that are not escaped, unescaping TEXT on the way.

    An escape the grammar does not define, such as ``\\:``, is kept as it is.
    """
    items: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(value):
        char = value[i]
        if unescape and char == "\\" and i + 1 < len(value):
            escaped = value[i + 1]
            if escaped in "nN":
                current.append("\n")
            elif escaped in "\\;,":
                current.append(escaped)
            else:
                current.append(char + escaped)
            i += 2
            continue
        if split and char == ",":
            items.append("".join(current))
            current = []
        else:
            current.append(char)
        i += 1
    items.append("".join(current))
    return [item.strip() for item in items] if split else items
