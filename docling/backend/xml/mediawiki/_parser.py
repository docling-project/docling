# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Build a read-only syntax tree from MediaWiki tokens."""

from __future__ import annotations

from dataclasses import dataclass, field

from docling.backend.xml.mediawiki import _tokens as tokens
from docling.backend.xml.mediawiki._tokenizer import Tokenizer
from docling.backend.xml.mediawiki._tokenizer_state import ParserError

_END_TOKENS = {
    tokens.TemplateOpen: (tokens.TemplateClose,),
    tokens.ArgumentOpen: (tokens.ArgumentClose,),
    tokens.WikilinkOpen: (tokens.WikilinkClose,),
    tokens.ExternalLinkOpen: (tokens.ExternalLinkClose,),
    tokens.HTMLEntityStart: (tokens.HTMLEntityEnd,),
    tokens.HeadingStart: (tokens.HeadingEnd,),
    tokens.CommentStart: (tokens.CommentEnd,),
    tokens.TagOpenOpen: (tokens.TagCloseSelfclose, tokens.TagCloseClose),
}


@dataclass
class Syntax:
    start: tokens.Token
    children: list[Node] = field(default_factory=list)


Node = Syntax | tokens.Token


def parse_wikitext(source: str) -> list[Node]:
    """Group balanced syntax while retaining separators and token fields."""
    root: list[Node] = []
    scopes: list[Syntax] = []
    for token in Tokenizer().tokenize(source):
        if scopes and isinstance(token, _END_TOKENS[type(scopes[-1].start)]):
            scopes.pop()
            continue
        siblings = scopes[-1].children if scopes else root
        if type(token) in _END_TOKENS:
            syntax = Syntax(token)
            siblings.append(syntax)
            scopes.append(syntax)
        else:
            siblings.append(token)
    if scopes:
        raise ParserError("Unclosed wikitext syntax after tokenization.")
    return root
