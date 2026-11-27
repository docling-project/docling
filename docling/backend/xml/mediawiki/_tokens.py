# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# Adapted from mwparserfromhell v0.7.2 (MIT).
# SPDX-FileCopyrightText: 2012-2025 Ben Kurtovic

"""Token types emitted by the wikitext tokenizer."""

from typing import Any


class Token(dict[str, Any]):
    """One syntactic token with fields specific to its type."""


def _token_type(name: str) -> type[Token]:
    return type(name, (Token,), {"__module__": __name__})


Text = _token_type("Text")
TemplateOpen = _token_type("TemplateOpen")  # {{
TemplateParamSeparator = _token_type("TemplateParamSeparator")  # |
TemplateParamEquals = _token_type("TemplateParamEquals")  # =
TemplateClose = _token_type("TemplateClose")  # }}
ArgumentOpen = _token_type("ArgumentOpen")  # {{{
ArgumentSeparator = _token_type("ArgumentSeparator")  # |
ArgumentClose = _token_type("ArgumentClose")  # }}}
WikilinkOpen = _token_type("WikilinkOpen")  # [[
WikilinkSeparator = _token_type("WikilinkSeparator")  # |
WikilinkClose = _token_type("WikilinkClose")  # ]]
ExternalLinkOpen = _token_type("ExternalLinkOpen")  # [
ExternalLinkSeparator = _token_type("ExternalLinkSeparator")  #
ExternalLinkClose = _token_type("ExternalLinkClose")  # ]
HTMLEntityStart = _token_type("HTMLEntityStart")  # &
HTMLEntityNumeric = _token_type("HTMLEntityNumeric")  # #
HTMLEntityHex = _token_type("HTMLEntityHex")  # x
HTMLEntityEnd = _token_type("HTMLEntityEnd")  # ;
HeadingStart = _token_type("HeadingStart")  # =...
HeadingEnd = _token_type("HeadingEnd")  # =...
CommentStart = _token_type("CommentStart")  # <!--
CommentEnd = _token_type("CommentEnd")  # -->
TagOpenOpen = _token_type("TagOpenOpen")  # <
TagAttrStart = _token_type("TagAttrStart")
TagAttrEquals = _token_type("TagAttrEquals")  # =
TagAttrQuote = _token_type("TagAttrQuote")  # ", '
TagCloseOpen = _token_type("TagCloseOpen")  # >
TagCloseSelfclose = _token_type("TagCloseSelfclose")  # />
TagOpenClose = _token_type("TagOpenClose")  # </
TagCloseClose = _token_type("TagCloseClose")  # >
