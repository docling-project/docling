# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# Adapted from mwparserfromhell v0.7.2 (MIT).
# SPDX-FileCopyrightText: 2012-2025 Ben Kurtovic

"""Character, tag, and link rules for MediaWiki wikitext."""

import html.entities

from docling.backend.xml.mediawiki import _tokens as tokens

URI_SCHEMES = {
    # [wikimedia/mediawiki.git]/includes/DefaultSettings.php @ 5c660de5d0
    "bitcoin": False,
    "ftp": True,
    "ftps": True,
    "geo": False,
    "git": True,
    "gopher": True,
    "http": True,
    "https": True,
    "irc": True,
    "ircs": True,
    "magnet": False,
    "mailto": False,
    "mms": True,
    "news": False,
    "nntp": True,
    "redis": True,
    "sftp": True,
    "sip": False,
    "sips": False,
    "sms": False,
    "ssh": True,
    "svn": True,
    "tel": False,
    "telnet": True,
    "urn": False,
    "worldwind": True,
    "xmpp": False,
}

PARSER_BLACKLIST = [
    # https://www.mediawiki.org/wiki/Parser_extension_tags @ 2020-12-21
    "categorytree",
    "ce",
    "chem",
    "gallery",
    "graph",
    "hiero",
    "imagemap",
    "inputbox",
    "math",
    "nowiki",
    "pre",
    "score",
    "section",
    "source",
    "syntaxhighlight",
    "templatedata",
    "timeline",
]

# [wikimedia/mediawiki.git]/includes/parser/Sanitizer.php @ 95e17ee645
SINGLE_ONLY = ["br", "wbr", "hr", "meta", "link", "img"]
SINGLE = [*SINGLE_ONLY, "li", "dt", "dd", "th", "td", "tr"]

MARKUP_TO_HTML = {
    "#": "li",
    "*": "li",
    ";": "dt",
    ":": "dd",
}


def get_html_tag(markup: str) -> str:
    """Return the HTML tag associated with the given wiki-markup."""
    return MARKUP_TO_HTML[markup]


def normalize_tag_name(name: str) -> str:
    """Normalize a tag name for matching opening and closing tags."""
    return name.rstrip().lower()


def is_parsable(tag: str) -> bool:
    """Return if the given *tag*'s contents should be passed to the parser."""
    return tag.lower() not in PARSER_BLACKLIST


def is_single(tag: str) -> bool:
    """Return whether or not the given *tag* can exist without a close tag."""
    return tag.lower() in SINGLE


def is_single_only(tag: str) -> bool:
    """Return whether or not the given *tag* must exist without a close tag."""
    return tag.lower() in SINGLE_ONLY


def is_scheme(scheme: str, slashes: bool = True) -> bool:
    """Return whether *scheme* is valid for external links."""
    scheme = scheme.lower()
    if slashes:
        return scheme in URI_SCHEMES
    return scheme in URI_SCHEMES and not URI_SCHEMES[scheme]


def parse_entity(body: str, numeric: bool) -> list[tokens.Token] | None:
    """Return the tokens for a valid, semicolon-terminated entity body."""
    result = [tokens.HTMLEntityStart()]
    if numeric:
        result.append(tokens.HTMLEntityNumeric())
        if body.startswith(("x", "X")):
            result.append(tokens.HTMLEntityHex(char=body[0]))
            body = body[1:]
            alphabet, base, max_digits = "0123456789abcdefABCDEF", 16, 6
        else:
            alphabet, base, max_digits = "0123456789", 10, 7
        significant = body.lstrip("0")
        if (
            not significant
            or len(significant) > max_digits
            or any(char not in alphabet for char in significant)
            or int(significant, base) > 0x10FFFF
        ):
            return None
        body = significant
    elif body not in html.entities.entitydefs:
        return None
    result.extend((tokens.Text(text=body), tokens.HTMLEntityEnd()))
    return result
