# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from enum import Enum


class ApiAnswerField(str, Enum):
    """A field of an OpenAI-compatible chat response that can hold the answer."""

    CONTENT = "content"
    # `reasoning` or `reasoning_content` on the wire.
    REASONING = "reasoning"
    # The raw `arguments` string of the first tool call that has arguments.
    TOOL_CALL_ARGUMENTS = "tool_call_arguments"


ANSWER_FIELDS_DESCRIPTION = (
    "Response fields to read the answer from, in order. The first field with "
    "text is the answer. The OpenAI contract puts the answer in `content`; add "
    "other fields only for servers that put it elsewhere."
)
