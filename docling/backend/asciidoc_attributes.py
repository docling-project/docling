# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Document attributes for the AsciiDoc backend.

Attribute entries set and unset attributes, and attribute references are
replaced by their values.
"""

import re
from typing import Final

# An attribute entry ":name: value" sets a document attribute; ":name!:" and
# ":!name:" unset it. A reference "{name}" is replaced by the value of the
# attribute, "\{name}" is kept as the literal "{name}", and a reference to an
# attribute that is not set stays as written (Asciidoctor's default
# attribute-missing behavior).
_ATTRIBUTE_ENTRY_RE: Final = re.compile(
    r"^:(?P<unset_before>!)?(?P<name>\w[\w.-]*?)(?P<unset_after>!)?:(?:\s+(?P<value>.*))?$"
)
_ATTRIBUTE_REFERENCE_RE: Final = re.compile(r"(?P<escape>\\)?\{(?P<name>\w[\w-]*)\}")
# Attributes that Asciidoctor predefines for characters that are awkward to type
# or would otherwise be read as markup.
INTRINSIC_ATTRIBUTES: Final = {
    "empty": "",
    "blank": "",
    "sp": " ",
    "nbsp": "\u00a0",
    "zwsp": "\u200b",
    "wj": "\u2060",
    "apos": "'",
    "quot": '"',
    "lsquo": "\u2018",
    "rsquo": "\u2019",
    "ldquo": "\u201c",
    "rdquo": "\u201d",
    "deg": "\u00b0",
    "plus": "+",
    "brvbar": "\u00a6",
    "vbar": "|",
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "startsb": "[",
    "endsb": "]",
    "caret": "^",
    "asterisk": "*",
    "tilde": "~",
    "backslash": "\\",
    "backtick": "`",
    "two-colons": "::",
    "two-semicolons": ";;",
    "cpp": "C++",
    "pp": "++",
}

# Asciidoctor replaces attribute references after inline formatting, so marks
# that come from an attribute value ("{asterisk}", "{backtick}") are literal
# text. Paragraph text is formatted after the references are replaced, so the
# marks in a value are swapped for private-use stand-ins until then.
_PROTECTED_MARKS: Final = {"*": "\ue000", "_": "\ue001", "`": "\ue002"}
_PROTECT_MARKS: Final = str.maketrans(_PROTECTED_MARKS)
RESTORE_MARKS: Final = str.maketrans({v: k for k, v in _PROTECTED_MARKS.items()})


def apply_attribute_entry(line: str, attributes: dict[str, str]) -> bool:
    """Set or unset an attribute if `line` is an attribute entry."""
    match = _ATTRIBUTE_ENTRY_RE.match(line)
    if match is None:
        return False
    name = match.group("name").lower()
    if match.group("unset_before") or match.group("unset_after"):
        attributes.pop(name, None)
    else:
        value = (match.group("value") or "").strip()
        attributes[name] = substitute_attributes(value, attributes)
    return True


def substitute_attributes(
    text: str, attributes: dict[str, str], protect_marks: bool = False
) -> str:
    """Replace the attribute references in `text` by their values.

    With `protect_marks`, inline formatting marks in the values are
    replaced by stand-ins that the backend turns back into the
    marks after formatting, so they stay literal text.
    """
    if "{" not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if match.group("escape"):
            return "{" + name + "}"
        value = attributes.get(name.lower())
        if value is None:
            return match.group(0)
        return value.translate(_PROTECT_MARKS) if protect_marks else value

    return _ATTRIBUTE_REFERENCE_RE.sub(replace, text)
