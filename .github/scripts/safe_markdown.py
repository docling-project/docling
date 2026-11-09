# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Render untrusted text for the comments that the bot posts."""

from __future__ import annotations

import re

MAX_MODEL_TEXT_CHARS = 600
# Written into untrusted text to stop GitHub from linking or notifying.
ZERO_WIDTH_SPACE = "\u200b"


def clean_text(text: str, limit: int = MAX_MODEL_TEXT_CHARS) -> str:
    """Reduce untrusted text to one line of printable characters.

    This does not escape Markdown. The publish step validates a stored result
    again, so this must give the same text when it runs twice. Escape with
    `markdown_text` or `markdown_code` where the text goes into a comment.
    """
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def markdown_text(text: str) -> str:
    """Render untrusted text as literal text in a Markdown line or table cell.

    The model answer can be steered by text in the pull request, and the bot
    signs the comment. So the text must not add links, images, HTML,
    headings, lists, code blocks, table cells, or mentions. CommonMark
    renders a backslash before any ASCII punctuation character as that
    character, so escaping all of them covers every Markdown construct.
    """
    text = clean_text(text, limit=len(text))
    # GitHub links a bare URL and notifies a mention by itself, also after an
    # escape, so break them with a zero-width space.
    text = re.sub(
        r"(?i)\b(https?|ftps?|mailto|www)(?=[:.])", rf"\1{ZERO_WIDTH_SPACE}", text
    )
    text = re.sub(r"@(?=[\w-])", f"@{ZERO_WIDTH_SPACE}", text)
    return re.sub(r"([!-/:-@\[-`{-~])", r"\\\1", text)


def markdown_code(text: str, *, in_table: bool = False) -> str:
    """Render untrusted text as one inline code span.

    Markdown has no syntax inside a code span. The fence is longer than any
    run of backticks in the text, so the text cannot close it.
    """
    text = clean_text(text, limit=len(text))
    if in_table:
        # GitHub splits a table cell at a pipe, also inside a code span.
        text = text.replace("|", "\\|")
    fence = "`" * (max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"
