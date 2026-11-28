# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# Adapted from mwparserfromhell v0.7.2 (MIT).
# SPDX-FileCopyrightText: 2012-2025 Ben Kurtovic

"""Recognize MediaWiki markup using context-sensitive tokenization."""

import math
import re
from typing import NoReturn, cast

from docling.backend.xml.mediawiki import _contexts as contexts, _tokens as tokens
from docling.backend.xml.mediawiki._syntax import (
    get_html_tag,
    is_parsable,
    is_scheme,
    is_single,
    is_single_only,
    normalize_tag_name,
    parse_entity,
)
from docling.backend.xml.mediawiki._tokenizer_state import (
    END,
    MARKERS,
    START,
    URI_SCHEME_CHARS,
    BadRoute,
    ClosingPositions,
    ParserError,
    Sentinel,
    TagOpenData,
    TokenizerWorkLimitError,
)


class Tokenizer:
    """Creates a list of tokens from a string of wikicode."""

    MAX_DEPTH = 100
    MAX_LIST_DEPTH = 32
    MAX_WORK_PER_CHAR = 64
    regex = re.compile(r"([{}\[\]<>|=&'#*;:/\\\"\-!\n])", flags=re.IGNORECASE)
    tag_splitter = re.compile(r"([\s\"\'\\]+)")

    def __init__(self):
        self._text: list[str] = []
        self._head: int = 0
        self._stacks = []
        self._global = 0
        self._depth = 0
        self._bad_routes = set()
        self._closers = ClosingPositions()
        self.work = self._work_limit = 0

    @property
    def _stack(self):
        return self._stacks[-1][0]

    @property
    def _context(self):
        return self._stacks[-1][1]

    @_context.setter
    def _context(self, value):
        self._stacks[-1][1] = value

    @property
    def _textbuffer(self):
        return self._stacks[-1][2]

    @_textbuffer.setter
    def _textbuffer(self, value):
        self._stacks[-1][2] = value

    @property
    def _stack_ident(self):
        """Identify the starting position and context for route memoization."""
        return self._stacks[-1][3]

    def _push(self, context=0):
        new_ident = (self._head, context)
        if new_ident in self._bad_routes:
            raise BadRoute(context)

        self._stacks.append([[], context, [], new_ident])
        self._depth += 1

    def _push_textbuffer(self):
        if self._textbuffer:
            self._stack.append(tokens.Text(text="".join(self._textbuffer)))
            self._textbuffer = []

    def _pop(self, keep_context=False):
        """Pop the token stack and optionally propagate its context."""
        self._push_textbuffer()
        self._depth -= 1
        if keep_context:
            context = self._context
            stack = self._stacks.pop()[0]
            self._context = context
            return stack
        return self._stacks.pop()[0]

    def _can_recurse(self):
        return self._depth < self.MAX_DEPTH

    def _memoize_bad_route(self):
        """Memoize an invalid route by its starting position and context."""
        self._bad_routes.add(self._stack_ident)

    def _fail_route(self) -> NoReturn:
        """Discard the current stack and mark its route invalid."""
        context = self._context
        self._memoize_bad_route()
        self._pop()
        raise BadRoute(context)

    def _emit(self, token):
        """Write a token to the end of the current token stack."""
        self._push_textbuffer()
        self._stack.append(token)

    def _emit_first(self, token):
        """Write a token to the beginning of the current token stack."""
        self._push_textbuffer()
        self._stack.insert(0, token)

    def _emit_text(self, text):
        """Write text to the current textbuffer."""
        self._textbuffer.append(text)

    def _emit_all(self, tokenlist):
        """Write a series of tokens to the current stack at once."""
        if tokenlist and isinstance(tokenlist[0], tokens.Text):
            self._emit_text(tokenlist.pop(0)["text"])
        self._push_textbuffer()
        self._stack.extend(tokenlist)

    def _emit_text_then_stack(self, text):
        """Pop the current stack, write *text*, and then write the stack."""
        stack = self._pop()
        self._emit_text(text)
        if stack:
            self._emit_all(stack)
        self._head -= 1

    def _read(self, delta: int = 0) -> str | Sentinel:
        """Read a relative input segment and charge it to the work limit."""
        index = self._head + delta
        segment = self._text[index] if 0 <= index < len(self._text) else None
        self.work += len(segment) if segment is not None else 1
        if self.work > self._work_limit:
            raise TokenizerWorkLimitError(
                "MediaWiki tokenizer exceeded its work limit."
            )
        if segment is not None:
            return segment
        return START if index < 0 else END

    def _parse_template(self, has_content):
        """Parse a template at the head of the wikicode string."""
        reset = self._head
        context = contexts.TEMPLATE_NAME
        if has_content:
            context |= contexts.HAS_TEMPLATE
        try:
            template = self._parse(context)
        except BadRoute:
            self._head = reset
            raise
        self._emit_first(tokens.TemplateOpen())
        self._emit_all(template)
        self._emit(tokens.TemplateClose())

    def _parse_argument(self):
        """Parse an argument at the head of the wikicode string."""
        reset = self._head
        try:
            argument = self._parse(contexts.ARGUMENT_NAME)
        except BadRoute:
            self._head = reset
            raise
        self._emit_first(tokens.ArgumentOpen())
        self._emit_all(argument)
        self._emit(tokens.ArgumentClose())

    def _parse_template_or_argument(self):
        """Parse a template or argument at the head of the wikicode string."""
        if self._head >= self._closers.brace:
            self._emit_text("{")
            return
        self._head += 2
        braces = 2
        while self._read() == "{":
            self._head += 1
            braces += 1
        has_content = False
        self._push()

        while braces:
            if braces == 1:
                return self._emit_text_then_stack("{")
            if braces == 2:
                try:
                    self._parse_template(has_content)
                except BadRoute:
                    return self._emit_text_then_stack("{{")
                break
            try:
                self._parse_argument()
                braces -= 3
            except BadRoute:
                try:
                    self._parse_template(has_content)
                    braces -= 2
                except BadRoute:
                    return self._emit_text_then_stack("{" * braces)
            if braces:
                has_content = True
                self._head += 1

        self._emit_all(self._pop())
        if self._context & contexts.FAIL_NEXT:
            self._context ^= contexts.FAIL_NEXT

    def _handle_template_param(self):
        """Handle a template parameter at the head of the string."""
        if self._context & contexts.TEMPLATE_NAME:
            if not self._context & (contexts.HAS_TEXT | contexts.HAS_TEMPLATE):
                self._fail_route()
            self._context ^= contexts.TEMPLATE_NAME
        elif self._context & contexts.TEMPLATE_PARAM_VALUE:
            self._context ^= contexts.TEMPLATE_PARAM_VALUE
        else:
            self._emit_all(self._pop())
        self._context |= contexts.TEMPLATE_PARAM_KEY
        self._emit(tokens.TemplateParamSeparator())
        self._push(self._context)

    def _handle_template_param_value(self):
        """Handle a template parameter's value at the head of the string."""
        self._emit_all(self._pop())
        self._context ^= contexts.TEMPLATE_PARAM_KEY
        self._context |= contexts.TEMPLATE_PARAM_VALUE
        self._emit(tokens.TemplateParamEquals())

    def _handle_template_end(self):
        """Handle the end of a template at the head of the string."""
        if self._context & contexts.TEMPLATE_NAME:
            if not self._context & (contexts.HAS_TEXT | contexts.HAS_TEMPLATE):
                self._fail_route()
        elif self._context & contexts.TEMPLATE_PARAM_KEY:
            self._emit_all(self._pop())
        self._head += 1
        return self._pop()

    def _handle_argument_separator(self):
        """Handle the separator between an argument's name and default."""
        self._context ^= contexts.ARGUMENT_NAME
        self._context |= contexts.ARGUMENT_DEFAULT
        self._emit(tokens.ArgumentSeparator())

    def _handle_argument_end(self):
        """Handle the end of an argument at the head of the string."""
        self._head += 2
        return self._pop()

    def _parse_wikilink(self):
        """Parse an internal wikilink at the head of the wikicode string."""
        reset = self._head + 1
        self._head += 2
        try:
            # If the wikilink looks like an external link, parse it as such:
            link, _extra = self._really_parse_external_link(True)
        except BadRoute:
            self._head = reset + 1
            try:
                # Otherwise, actually parse it as a wikilink:
                if self._closers.bracket < self._head:
                    raise BadRoute()
                wikilink = self._parse(contexts.WIKILINK_TITLE)
            except BadRoute:
                self._head = reset
                self._emit_text("[[")
            else:
                self._emit(tokens.WikilinkOpen())
                self._emit_all(wikilink)
                self._emit(tokens.WikilinkClose())
        else:
            if self._context & contexts.EXT_LINK_TITLE:
                # In this exceptional case, an external link that looks like a
                # wikilink inside of an external link is parsed as text:
                self._head = reset
                self._emit_text("[[")
                return
            self._emit_text("[")
            self._emit(tokens.ExternalLinkOpen(brackets=True))
            self._emit_all(link)
            self._emit(tokens.ExternalLinkClose())

    def _handle_wikilink_separator(self):
        """Handle the separator between a wikilink's title and its text."""
        self._context ^= contexts.WIKILINK_TITLE
        self._context |= contexts.WIKILINK_TEXT
        self._emit(tokens.WikilinkSeparator())

    def _handle_wikilink_end(self):
        """Handle the end of a wikilink at the head of the string."""
        self._head += 1
        return self._pop()

    def _parse_bracketed_uri_scheme(self):
        """Parse the URI scheme of a bracket-enclosed external link."""
        self._push(contexts.EXT_LINK_URI)
        if self._read() == self._read(1) == "/":
            self._emit_text("//")
            self._head += 2
        else:

            def all_valid(this: str):
                return all(char in URI_SCHEME_CHARS for char in this)

            scheme = ""
            while (this := self._read()) is not END and all_valid(this):
                scheme += this
                self._emit_text(self._read())
                self._head += 1
            if self._read() != ":":
                self._fail_route()
            self._emit_text(":")
            self._head += 1
            slashes = self._read() == self._read(1) == "/"
            if slashes:
                self._emit_text("//")
                self._head += 2
            if not is_scheme(scheme, slashes):
                self._fail_route()

    def _parse_free_uri_scheme(self):
        """Parse the URI scheme of a free (no brackets) external link."""
        scheme = []
        try:
            # We have to backtrack through the textbuffer looking for our
            # scheme since it was just parsed as text:
            for chunk in reversed(self._textbuffer):
                for char in reversed(chunk):
                    # Stop at the first non-word character
                    if re.fullmatch(r"\W", char):
                        raise StopIteration()
                    if char not in URI_SCHEME_CHARS:
                        raise BadRoute()
                    scheme.append(char)
        except StopIteration:
            pass
        scheme = "".join(reversed(scheme))
        slashes = self._read() == self._read(1) == "/"
        if not is_scheme(scheme, slashes):
            raise BadRoute()
        self._push(self._context | contexts.EXT_LINK_URI)
        self._emit_text(scheme)
        self._emit_text(":")
        if slashes:
            self._emit_text("//")
            self._head += 2

    def _handle_free_link_text(self, punct, tail, this):
        """Handle text in a free ext link, including trailing punctuation."""
        if "(" in this and ")" in punct:
            punct = punct[:-1]  # ')' is not longer valid punctuation
        if this.endswith(punct):
            for i in range(len(this) - 1, 0, -1):
                if this[i - 1] not in punct:
                    break
            else:
                i = 0
            stripped = this[:i]
            if stripped and tail:
                self._emit_text(tail)
                tail = ""
            tail += this[i:]
            this = stripped
        elif tail:
            self._emit_text(tail)
            tail = ""
        self._emit_text(this)
        return punct, tail

    def _is_uri_end(self, this, nxt):
        """Return whether the current head is the end of a URI."""
        # Built from _parse()'s end sentinels:
        after, ctx = self._read(2), self._context
        return (
            this in (END, "\n", "[", "]", "<", ">", '"')
            or " " in this
            or this == nxt == "'"
            or (this == "|" and ctx & contexts.TEMPLATE)
            or (this == "=" and ctx & (contexts.TEMPLATE_PARAM_KEY | contexts.HEADING))
            or (this == nxt == "}" and ctx & contexts.TEMPLATE)
            or (this == nxt == after == "}" and ctx & contexts.ARGUMENT)
        )

    def _really_parse_external_link(self, brackets):
        """Really parse an external link."""
        if brackets:
            if self._closers.bracket < self._head:
                raise BadRoute()
            self._parse_bracketed_uri_scheme()
            invalid = ("\n", " ", "]")
            punct = ()
        else:
            self._parse_free_uri_scheme()
            invalid = ("\n", " ", "[", "]")
            punct = tuple(",;\\.:!?)")
        if (this := self._read()) is END or this[0] in invalid:
            self._fail_route()
        tail = ""
        while True:
            this, nxt = self._read(), self._read(1)
            if this == "&":
                if tail:
                    self._emit_text(tail)
                    tail = ""
                self._parse_entity()
            elif this == "<" and nxt == "!" and self._read(2) == self._read(3) == "-":
                if tail:
                    self._emit_text(tail)
                    tail = ""
                self._parse_comment()
            elif this == nxt == "{" and self._can_recurse():
                if tail:
                    self._emit_text(tail)
                    tail = ""
                self._parse_template_or_argument()
            elif brackets:
                if this is END or this == "\n":
                    self._fail_route()
                if this == "]":
                    return self._pop(), None
                if self._is_uri_end(this, nxt):
                    if " " in this:
                        before, after = this.split(" ", 1)
                        self._emit_text(before)
                        self._emit(tokens.ExternalLinkSeparator())
                        if after:
                            self._emit_text(after)
                        self._head += 1
                    else:
                        separator = tokens.ExternalLinkSeparator()
                        separator["suppress_space"] = True
                        self._emit(separator)
                    self._context ^= contexts.EXT_LINK_URI
                    self._context |= contexts.EXT_LINK_TITLE
                    return self._parse(push=False), None
                self._emit_text(this)
            else:
                if self._is_uri_end(this, nxt):
                    if this is not END and " " in this:
                        before, after = this.split(" ", 1)
                        punct, tail = self._handle_free_link_text(punct, tail, before)
                        tail += " " + after
                    else:
                        self._head -= 1
                    return self._pop(), tail
                punct, tail = self._handle_free_link_text(punct, tail, this)
            self._head += 1

    def _remove_uri_scheme_from_textbuffer(self, scheme):
        """Remove the URI scheme of a new external link from the textbuffer."""
        length = len(scheme)
        while length:
            if length < len(self._textbuffer[-1]):
                self._textbuffer[-1] = self._textbuffer[-1][:-length]
                break
            length -= len(self._textbuffer[-1])
            self._textbuffer.pop()

    def _parse_external_link(self, brackets):
        """Parse an external link at the head of the wikicode string."""
        if self._context & contexts.NO_EXT_LINKS or not self._can_recurse():
            if not brackets and self._context & contexts.DL_TERM:
                self._handle_dl_term()
            else:
                self._emit_text(self._read())
            return

        reset = self._head
        self._head += 1
        try:
            link, extra = self._really_parse_external_link(brackets)
        except BadRoute:
            self._head = reset
            if not brackets and self._context & contexts.DL_TERM:
                self._handle_dl_term()
            else:
                self._emit_text(self._read())
        else:
            if not brackets:
                scheme = link[0]["text"].split(":", 1)[0]
                self._remove_uri_scheme_from_textbuffer(scheme)
            self._emit(tokens.ExternalLinkOpen(brackets=brackets))
            self._emit_all(link)
            self._emit(tokens.ExternalLinkClose())
            if extra:
                self._emit_text(extra)

    def _parse_heading(self):
        """Parse a section heading at the head of the wikicode string."""
        self._global |= contexts.GL_HEADING
        reset = self._head
        self._head += 1
        best = 1
        while self._read() == "=":
            best += 1
            self._head += 1
        context = contexts.HEADING_LEVEL_1 << min(best - 1, 5)

        try:
            title, level = self._parse(context)
        except BadRoute:
            self._head = reset + best - 1
            self._emit_text("=" * best)
        else:
            self._emit(tokens.HeadingStart(level=level))
            if level < best:
                self._emit_text("=" * (best - level))
            self._emit_all(title)
            self._emit(tokens.HeadingEnd())
        finally:
            self._global ^= contexts.GL_HEADING

    def _handle_heading_end(self):
        """Handle the end of a section heading at the head of the string."""
        reset = self._head
        self._head += 1
        best = 1
        while self._read() == "=":
            best += 1
            self._head += 1
        current = int(math.log(self._context / contexts.HEADING_LEVEL_1, 2)) + 1
        level = min(current, min(best, 6))

        try:  # Try to check for a heading closure after this one
            after, after_level = self._parse(self._context)
        except BadRoute:
            if level < best:
                self._emit_text("=" * (best - level))
            self._head = reset + best - 1
            return self._pop(), level
        else:  # Found another closure
            self._emit_text("=" * best)
            self._emit_all(after)
            return self._pop(), after_level

    def _parse_entity(self):
        numeric = self._read(1) == "#"
        end = 3 if numeric else 2
        body = self._read(end - 1)
        if isinstance(body, str) and self._read(end) == ";":
            if entity := parse_entity(body, numeric):
                self._emit_all(entity)
                self._head += end
                return
        self._emit_text("&")

    def _parse_comment(self):
        """Parse an HTML comment at the head of the wikicode string."""
        self._head += 4
        reset = self._head - 1
        if self._closers.comment < self._head:
            self._head = reset
            self._emit_text("<!--")
            return
        self._push()
        while True:
            this = self._read()
            if this == END:
                self._pop()
                self._head = reset
                self._emit_text("<!--")
                return
            if this == self._read(1) == "-" and self._read(2) == ">":
                self._emit_first(tokens.CommentStart())
                self._emit(tokens.CommentEnd())
                self._emit_all(self._pop())
                self._head += 2
                if self._context & contexts.FAIL_NEXT:
                    # _verify_safe() sets this flag while parsing a template
                    # or link when it encounters what might be a comment -- we
                    # must unset it to let _verify_safe() know it was correct:
                    self._context ^= contexts.FAIL_NEXT
                return
            self._emit_text(this)
            self._head += 1

    def _push_tag_buffer(self, data):
        """Write a pending tag attribute from *data* to the stack."""
        if data.context & data.CX_QUOTED:
            self._emit_first(tokens.TagAttrQuote(char=data.quoter))
            self._emit_all(self._pop())
        buf = data.padding_buffer
        self._emit_first(
            tokens.TagAttrStart(
                pad_first=buf["first"],
                pad_before_eq=buf["before_eq"],
                pad_after_eq=buf["after_eq"],
            )
        )
        self._emit_all(self._pop())
        for key in data.padding_buffer:
            data.padding_buffer[key] = ""

    def _handle_tag_space(self, data, text):
        """Handle whitespace (*text*) inside of an HTML open tag."""
        ctx = data.context
        end_of_value = ctx & data.CX_ATTR_VALUE and not ctx & (
            data.CX_QUOTED | data.CX_NOTE_QUOTE
        )
        if end_of_value or (ctx & data.CX_QUOTED and ctx & data.CX_NOTE_SPACE):
            self._push_tag_buffer(data)
            data.context = data.CX_ATTR_READY
        elif ctx & data.CX_NOTE_SPACE:
            data.context = data.CX_ATTR_READY
        elif ctx & data.CX_ATTR_NAME:
            data.context |= data.CX_NOTE_EQUALS
            data.padding_buffer["before_eq"] += text
        if ctx & data.CX_QUOTED and not ctx & data.CX_NOTE_SPACE:
            self._emit_text(text)
        elif data.context & data.CX_ATTR_READY:
            data.padding_buffer["first"] += text
        elif data.context & data.CX_ATTR_VALUE:
            data.padding_buffer["after_eq"] += text

    def _handle_tag_text(self, text):
        """Handle regular *text* inside of an HTML open tag."""
        nxt = self._read(1)
        if not self._can_recurse() or text not in MARKERS:
            self._emit_text(text)
        elif text == nxt == "{":
            self._parse_template_or_argument()
        elif text == nxt == "[":
            self._parse_wikilink()
        elif text == "<":
            self._parse_tag()
        else:
            self._emit_text(text)

    def _handle_tag_data(self, data, text):
        """Handle all sorts of *text* data inside of an HTML open tag."""
        for chunk in self.tag_splitter.split(text):
            if not chunk:
                continue
            if data.context & data.CX_NAME:
                if chunk in MARKERS or chunk.isspace():
                    self._fail_route()  # Tags must start with text, not spaces
                data.context = data.CX_NOTE_SPACE
            elif chunk.isspace():
                self._handle_tag_space(data, chunk)
                continue
            elif data.context & data.CX_NOTE_SPACE:
                if data.context & data.CX_QUOTED:
                    data.context = data.CX_ATTR_VALUE
                    self._memoize_bad_route()
                    self._pop()
                    self._head = data.reset - 1  # Will be auto-incremented
                    return  # Break early
                self._fail_route()
            elif data.context & data.CX_ATTR_READY:
                data.context = data.CX_ATTR_NAME
                self._push(contexts.TAG_ATTR)
            elif data.context & data.CX_ATTR_NAME:
                if chunk == "=":
                    data.context = data.CX_ATTR_VALUE | data.CX_NOTE_QUOTE
                    self._emit(tokens.TagAttrEquals())
                    continue
                if data.context & data.CX_NOTE_EQUALS:
                    self._push_tag_buffer(data)
                    data.context = data.CX_ATTR_NAME
                    self._push(contexts.TAG_ATTR)
            else:  # data.context & data.CX_ATTR_VALUE assured
                escaped = self._read(-1) == "\\" and self._read(-2) != "\\"
                if data.context & data.CX_NOTE_QUOTE:
                    data.context ^= data.CX_NOTE_QUOTE
                    if chunk in "'\"" and not escaped:
                        data.context |= data.CX_QUOTED
                        data.quoter = chunk
                        data.reset = self._head
                        try:
                            self._push(self._context)
                        except BadRoute:
                            # Already failed to parse this as a quoted string
                            data.context = data.CX_ATTR_VALUE
                            self._head -= 1
                            return
                        continue
                elif data.context & data.CX_QUOTED:
                    if chunk == data.quoter and not escaped:
                        data.context |= data.CX_NOTE_SPACE
                        continue
            self._handle_tag_text(chunk)

    def _handle_tag_close_open(self, data, token):
        """Handle the closing of a open tag (``<foo>``)."""
        if data.context & (data.CX_ATTR_NAME | data.CX_ATTR_VALUE):
            self._push_tag_buffer(data)
        self._emit(token(padding=data.padding_buffer["first"]))
        self._head += 1

    def _handle_tag_open_close(self):
        """Handle the opening of a closing tag (``</foo>``)."""
        self._emit(tokens.TagOpenClose())
        self._push(contexts.TAG_CLOSE)
        self._head += 1

    def _handle_tag_close_close(self):
        """Handle the ending of a closing tag (``</foo>``)."""
        closing = self._pop()
        if len(closing) != 1 or (
            not isinstance(closing[0], tokens.Text)
            or normalize_tag_name(closing[0]["text"])
            != normalize_tag_name(self._stack[1]["text"])
        ):
            self._fail_route()
        self._emit_all(closing)
        self._emit(tokens.TagCloseClose())
        return self._pop()

    def _handle_blacklisted_tag(self):
        """Handle the body of an HTML tag that is parser-blacklisted."""
        while True:
            this, nxt = self._read(), self._read(1)
            if this is END:
                self._fail_route()
            elif this == "<" and nxt == "/":
                self._head += 3
                if self._read() != ">" or (
                    normalize_tag_name(cast(str, self._read(-1)))
                    != normalize_tag_name(self._stack[1]["text"])
                ):
                    self._head -= 1
                    self._emit_text("</")
                    continue
                self._emit(tokens.TagOpenClose())
                self._emit_text(self._read(-1))
                self._emit(tokens.TagCloseClose())
                return self._pop()
            elif this == "&":
                self._parse_entity()
            else:
                self._emit_text(this)
            self._head += 1

    def _handle_single_only_tag_end(self):
        """Handle the end of an implicitly closing single-only HTML tag."""
        padding = self._stack.pop().get("padding")
        self._emit(tokens.TagCloseSelfclose(padding=padding, implicit=True))
        self._head -= 1  # Offset displacement done by _handle_tag_close_open
        return self._pop()

    def _handle_single_tag_end(self):
        """Handle the stream end when inside a single-supporting HTML tag."""
        stack = self._stack
        # We need to find the index of the TagCloseOpen token corresponding to
        # the TagOpenOpen token located at index 0:
        depth = 1
        for index, token in enumerate(stack[2:], 2):
            if isinstance(token, tokens.TagOpenOpen):
                depth += 1
            elif isinstance(token, tokens.TagCloseOpen):
                depth -= 1
                if depth == 0:
                    break
            elif isinstance(token, tokens.TagCloseSelfclose):
                depth -= 1
                if depth == 0:  # pragma: no cover (untestable/exceptional)
                    raise ParserError(
                        "_handle_single_tag_end() got an unexpected TagCloseSelfclose"
                    )
        else:  # pragma: no cover (untestable/exceptional case)
            raise ParserError("_handle_single_tag_end() missed a TagCloseOpen")
        padding = stack[index].get("padding")
        stack[index] = tokens.TagCloseSelfclose(padding=padding, implicit=True)
        return self._pop()

    def _really_parse_tag(self):
        """Actually parse an HTML tag, starting with the open (``<foo>``)."""
        if self._head > self._closers.tag_end:
            raise BadRoute()
        data = TagOpenData()
        self._push(contexts.TAG_OPEN)
        self._emit(tokens.TagOpenOpen())
        while True:
            this, nxt = self._read(), self._read(1)
            can_exit = (
                not data.context & (data.CX_QUOTED | data.CX_NAME)
                or data.context & data.CX_NOTE_SPACE
            )
            if this is END:
                if self._context & contexts.TAG_ATTR:
                    if data.context & data.CX_QUOTED:
                        # Unclosed attribute quote: reset, don't die
                        data.context = data.CX_ATTR_VALUE
                        self._memoize_bad_route()
                        self._pop()
                        self._head = data.reset
                        continue
                    self._pop()
                self._fail_route()
            elif this == ">" and can_exit:
                self._handle_tag_close_open(data, tokens.TagCloseOpen)
                self._context = contexts.TAG_BODY
                name = self._stack[1]["text"]
                if is_single_only(name):
                    return self._handle_single_only_tag_end()
                # A differently named closer cannot end this tag's body.
                last_close = self._closers.tags.get(normalize_tag_name(name), -1)
                if not is_single(name) and last_close < self._head:
                    self._fail_route()
                if is_parsable(name):
                    return self._parse(push=False)
                return self._handle_blacklisted_tag()
            elif this == "/" and nxt == ">" and can_exit:
                self._handle_tag_close_open(data, tokens.TagCloseSelfclose)
                return self._pop()
            else:
                self._handle_tag_data(data, this)
            self._head += 1

    def _handle_invalid_tag_start(self):
        """Handle the (possible) start of an implicitly closing single tag."""
        reset = self._head + 1
        self._head += 2
        try:
            this = self._read()
            assert this is not END
            if not is_single_only(self.tag_splitter.split(this)[0]):
                raise BadRoute()
            tag = self._really_parse_tag()
        except BadRoute:
            self._head = reset
            self._emit_text("</")
        else:
            tag[0]["invalid"] = True  # Set flag of TagOpenOpen
            self._emit_all(tag)

    def _parse_tag(self):
        """Parse an HTML tag at the head of the wikicode string."""
        reset = self._head
        self._head += 1
        try:
            tag = self._really_parse_tag()
        except BadRoute:
            self._head = reset
            self._emit_text("<")
        else:
            self._emit_all(tag)

    def _emit_style_tag(self, tag, markup, body):
        """Write the body of a tag and the tokens that should surround it."""
        self._emit(tokens.TagOpenOpen(wiki_markup=markup))
        self._emit_text(tag)
        self._emit(tokens.TagCloseOpen())
        self._emit_all(body)
        self._emit(tokens.TagOpenClose())
        self._emit_text(tag)
        self._emit(tokens.TagCloseClose())

    def _parse_italics(self):
        """Parse wiki-style italics."""
        reset = self._head
        try:
            stack = self._parse(contexts.STYLE_ITALICS)
        except BadRoute as route:
            self._head = reset
            if route.context & contexts.STYLE_PASS_AGAIN:
                new_ctx = contexts.STYLE_ITALICS | contexts.STYLE_SECOND_PASS
                try:
                    stack = self._parse(new_ctx)
                except BadRoute:
                    self._head = reset
                    self._emit_text("''")
                    return
            else:
                self._emit_text("''")
                return
        self._emit_style_tag("i", "''", stack)

    def _parse_bold(self):
        """Parse wiki-style bold."""
        reset = self._head
        try:
            stack = self._parse(contexts.STYLE_BOLD)
        except BadRoute:
            self._head = reset
            if self._context & contexts.STYLE_SECOND_PASS:
                self._emit_text("'")
                return True
            if self._context & contexts.STYLE_ITALICS:
                self._context |= contexts.STYLE_PASS_AGAIN
                self._emit_text("'''")
            else:
                self._emit_text("'")
                self._parse_italics()
        else:
            self._emit_style_tag("b", "'''", stack)
        return False

    def _parse_italics_and_bold(self):
        """Parse wiki-style italics and bold together (i.e., five ticks)."""
        reset = self._head
        try:
            stack = self._parse(contexts.STYLE_BOLD)
        except BadRoute:
            self._head = reset
            try:
                stack = self._parse(contexts.STYLE_ITALICS)
            except BadRoute:
                self._head = reset
                self._emit_text("'''''")
            else:
                reset = self._head
                try:
                    stack2 = self._parse(contexts.STYLE_BOLD)
                except BadRoute:
                    self._head = reset
                    self._emit_text("'''")
                    self._emit_style_tag("i", "''", stack)
                else:
                    self._push()
                    self._emit_style_tag("i", "''", stack)
                    self._emit_all(stack2)
                    self._emit_style_tag("b", "'''", self._pop())
        else:
            reset = self._head
            try:
                stack2 = self._parse(contexts.STYLE_ITALICS)
            except BadRoute:
                self._head = reset
                self._emit_text("''")
                self._emit_style_tag("b", "'''", stack)
            else:
                self._push()
                self._emit_style_tag("b", "'''", stack)
                self._emit_all(stack2)
                self._emit_style_tag("i", "''", self._pop())

    def _parse_style(self):
        """Parse wiki-style formatting (``''``/``'''`` for italics/bold)."""
        self._head += 2
        ticks = 2
        while self._read() == "'":
            self._head += 1
            ticks += 1
        italics = self._context & contexts.STYLE_ITALICS
        bold = self._context & contexts.STYLE_BOLD

        if ticks > 5:
            self._emit_text("'" * (ticks - 5))
            ticks = 5
        elif ticks == 4:
            self._emit_text("'")
            ticks = 3

        # A failed bold parse inside italics leaves three apostrophes. Treat
        # two as the italic close and the last as literal text outside it.
        if italics and ticks == 3 and self._context & contexts.STYLE_SECOND_PASS:
            self._head -= 1
            return self._pop()
        if (italics and ticks in (2, 5)) or (bold and ticks in (3, 5)):
            if ticks == 5:
                self._head -= 3 if italics else 2
            return self._pop()
        if not self._can_recurse():
            if ticks == 3:
                if self._context & contexts.STYLE_SECOND_PASS:
                    self._emit_text("'")
                    return self._pop()
                if self._context & contexts.STYLE_ITALICS:
                    self._context |= contexts.STYLE_PASS_AGAIN
            self._emit_text("'" * ticks)
        elif ticks == 2:
            self._parse_italics()
        elif ticks == 3:
            if self._parse_bold():
                return self._pop()
        else:  # ticks == 5
            self._parse_italics_and_bold()
        self._head -= 1

    def _handle_list_marker(self):
        """Handle a list marker at the head (``#``, ``*``, ``;``, ``:``)."""
        markup = self._read()
        assert markup is not END
        if markup == ";":
            self._context |= contexts.DL_TERM
        self._emit(tokens.TagOpenOpen(wiki_markup=markup))
        self._emit_text(get_html_tag(markup))
        self._emit(tokens.TagCloseSelfclose())

    def _handle_list(self):
        """Handle a wiki-style list (``#``, ``*``, ``;``, ``:``)."""
        markers = ("#", "*", ";", ":")
        for _ in range(self.MAX_LIST_DEPTH):
            self._handle_list_marker()
            if self._read(1) not in markers:
                return
            self._head += 1
        while (marker := self._read()) in markers:
            self._emit_text(marker)
            self._head += 1
        self._head -= 1

    def _handle_hr(self):
        """Handle a wiki-style horizontal rule (``----``) in the string."""
        length = 4
        self._head += 3
        while self._read(1) == "-":
            length += 1
            self._head += 1
        self._emit(tokens.TagOpenOpen(wiki_markup="-" * length))
        self._emit_text("hr")
        self._emit(tokens.TagCloseSelfclose())

    def _handle_dl_term(self):
        """Handle the term in a description list (``foo`` in ``;foo:bar``)."""
        self._context ^= contexts.DL_TERM
        if self._read() == ":":
            self._handle_list_marker()
        else:
            self._emit_text("\n")

    def _emit_table_tag(
        self,
        open_open_markup,
        tag,
        style,
        padding,
        close_open_markup,
        contents,
        open_close_markup,
    ):
        """Emit a table tag."""
        self._emit(tokens.TagOpenOpen(wiki_markup=open_open_markup))
        self._emit_text(tag)
        if style:
            self._emit_all(style)
        if close_open_markup:
            self._emit(
                tokens.TagCloseOpen(wiki_markup=close_open_markup, padding=padding)
            )
        else:
            self._emit(tokens.TagCloseOpen(padding=padding))
        if contents:
            self._emit_all(contents)
        self._emit(tokens.TagOpenClose(wiki_markup=open_close_markup))
        self._emit_text(tag)
        self._emit(tokens.TagCloseClose())

    def _handle_table_style(self, end_token: str):
        """Handle style attributes for a table until ``end_token``."""
        data = TagOpenData()
        data.context = TagOpenData.CX_ATTR_READY
        while True:
            this = self._read()
            can_exit = (
                not data.context & data.CX_QUOTED or data.context & data.CX_NOTE_SPACE
            )
            if this == end_token and can_exit:
                assert this is not END
                if data.context & (data.CX_ATTR_NAME | data.CX_ATTR_VALUE):
                    self._push_tag_buffer(data)
                if this.isspace():
                    data.padding_buffer["first"] += this
                return data.padding_buffer["first"]
            if this is END or this == end_token:
                if self._context & contexts.TAG_ATTR:
                    if data.context & data.CX_QUOTED:
                        # Unclosed attribute quote: reset, don't die
                        data.context = data.CX_ATTR_VALUE
                        self._memoize_bad_route()
                        self._pop()
                        self._head = data.reset
                        continue
                    self._pop()
                self._fail_route()
            else:
                self._handle_tag_data(data, this)
            self._head += 1

    def _parse_table(self):
        """Parse a wikicode table by starting with the first line."""
        reset = self._head
        self._head += 2
        try:
            self._push(contexts.TABLE_OPEN)
            padding = self._handle_table_style("\n")
        except BadRoute:
            self._head = reset
            self._emit_text("{")
            return
        style = self._pop()

        self._head += 1
        restore_point = self._stack_ident
        try:
            table = self._parse(contexts.TABLE_OPEN)
        except BadRoute:
            while self._stack_ident != restore_point:
                self._memoize_bad_route()
                self._pop()
            self._head = reset
            self._emit_text("{")
            return

        self._emit_table_tag("{|", "table", style, padding, None, table, "|}")
        # Offset displacement done by _parse():
        self._head -= 1

    def _handle_table_row(self):
        """Parse as style until end of the line, then continue."""
        self._head += 2
        if not self._can_recurse():
            self._emit_text("|-")
            self._head -= 1
            return

        self._push(contexts.TABLE_OPEN | contexts.TABLE_ROW_OPEN)
        padding = self._handle_table_style("\n")
        style = self._pop()

        # Don't parse the style separator:
        self._head += 1
        row = self._parse(contexts.TABLE_OPEN | contexts.TABLE_ROW_OPEN)

        self._emit_table_tag("|-", "tr", style, padding, None, row, "")
        # Offset displacement done by parse():
        self._head -= 1

    def _handle_table_cell(self, markup, tag, line_context):
        """Parse as normal syntax unless we hit a style marker, then parse
        style as HTML attributes and the remainder as normal syntax."""
        old_context = self._context
        padding, style = "", None
        self._head += len(markup)
        reset = self._head
        if not self._can_recurse():
            self._emit_text(markup)
            self._head -= 1
            return

        cell = self._parse(
            contexts.TABLE_OPEN
            | contexts.TABLE_CELL_OPEN
            | line_context
            | contexts.TABLE_CELL_STYLE
        )
        cell_context = self._context
        self._context = old_context
        reset_for_style = cell_context & contexts.TABLE_CELL_STYLE
        if reset_for_style:
            self._head = reset
            self._push(contexts.TABLE_OPEN | contexts.TABLE_CELL_OPEN | line_context)
            padding = self._handle_table_style("|")
            style = self._pop()
            # Don't parse the style separator:
            self._head += 1
            cell = self._parse(
                contexts.TABLE_OPEN | contexts.TABLE_CELL_OPEN | line_context
            )
            cell_context = self._context
            self._context = old_context

        close_open_markup = "|" if reset_for_style else None
        self._emit_table_tag(markup, tag, style, padding, close_open_markup, cell, "")
        # Keep header/cell line contexts:
        self._context |= cell_context & (
            contexts.TABLE_TH_LINE | contexts.TABLE_TD_LINE
        )
        # Offset displacement done by parse():
        self._head -= 1

    def _handle_table_cell_end(self, reset_for_style=False):
        """Returns the current context, with the TABLE_CELL_STYLE flag set if
        it is necessary to reset and parse style attributes."""
        if reset_for_style:
            self._context |= contexts.TABLE_CELL_STYLE
        else:
            self._context &= ~contexts.TABLE_CELL_STYLE
        return self._pop(keep_context=True)

    def _handle_table_row_end(self):
        """Return the stack in order to handle the table row end."""
        return self._pop()

    def _handle_table_end(self):
        """Return the stack in order to handle the table end."""
        self._head += 2
        return self._pop()

    def _handle_end(self):
        """Handle the end of the stream of wikitext."""
        if self._context & contexts.FAIL:
            if self._context & contexts.TAG_BODY:
                if is_single(self._stack[1]["text"]):
                    return self._handle_single_tag_end()
            if self._context & contexts.TABLE_CELL_OPEN:
                self._pop()
            if self._context & contexts.DOUBLE:
                self._pop()
            self._fail_route()
        return self._pop()

    def _verify_safe(self, this):
        """Make sure we are not trying to write an invalid character."""
        context = self._context
        if context & contexts.FAIL_NEXT:
            return False
        if context & contexts.WIKILINK_TITLE:
            if this in ("]", "{"):
                self._context |= contexts.FAIL_NEXT
            elif this in ("\n", "[", "}", ">"):
                return False
            elif this == "<":
                if self._read(1) == "!":
                    self._context |= contexts.FAIL_NEXT
                else:
                    return False
            return True
        if context & contexts.EXT_LINK_TITLE:
            return this != "\n"
        if context & contexts.TEMPLATE_NAME:
            if this == "{":
                self._context |= contexts.HAS_TEMPLATE | contexts.FAIL_NEXT
                return True
            if this == "}" or (this == "<" and self._read(1) == "!"):
                self._context |= contexts.FAIL_NEXT
                return True
            if this in ("[", "]", "<", ">"):
                return False
            if this == "|":
                return True
            if context & contexts.HAS_TEXT:
                if context & contexts.FAIL_ON_TEXT:
                    if this is END or not this.isspace():
                        return False
                elif this == "\n":
                    self._context |= contexts.FAIL_ON_TEXT
            elif this is END or not this.isspace():
                self._context |= contexts.HAS_TEXT
            return True
        if context & contexts.TAG_CLOSE:
            return this != "<"
        if context & contexts.FAIL_ON_EQUALS:
            if this == "=":
                return False
        elif context & contexts.FAIL_ON_LBRACE:
            if this == "{" or (self._read(-1) == self._read(-2) == "{"):
                if context & contexts.TEMPLATE:
                    self._context |= contexts.FAIL_ON_EQUALS
                else:
                    self._context |= contexts.FAIL_NEXT
                return True
            self._context ^= contexts.FAIL_ON_LBRACE
        elif context & contexts.FAIL_ON_RBRACE:
            if this == "}":
                self._context |= contexts.FAIL_NEXT
                return True
            self._context ^= contexts.FAIL_ON_RBRACE
        elif this == "{":
            self._context |= contexts.FAIL_ON_LBRACE
        elif this == "}":
            self._context |= contexts.FAIL_ON_RBRACE
        return True

    def _parse(self, context=0, push=True):  # noqa: C901
        """Parse the wikicode string, using *context* for when to stop."""
        if push:
            self._push(context)
        while True:
            this = self._read()
            if self._context & contexts.UNSAFE:
                if not self._verify_safe(this):
                    if self._context & contexts.DOUBLE:
                        self._pop()
                    self._fail_route()
            if this not in MARKERS:
                self._emit_text(this)
                self._head += 1
                continue
            if this is END:
                return self._handle_end()
            nxt = self._read(1)
            if this == nxt == "{":
                if self._can_recurse():
                    self._parse_template_or_argument()
                else:
                    self._emit_text("{")
            elif this == "|" and self._context & contexts.TEMPLATE:
                self._handle_template_param()
            elif this == "=" and self._context & contexts.TEMPLATE_PARAM_KEY:
                if (
                    not self._global & contexts.GL_HEADING
                    and self._read(-1) in ("\n", START)
                    and nxt == "="
                ):
                    self._parse_heading()
                else:
                    self._handle_template_param_value()
            elif this == nxt == "}" and self._context & contexts.TEMPLATE:
                return self._handle_template_end()
            elif this == "|" and self._context & contexts.ARGUMENT_NAME:
                self._handle_argument_separator()
            elif this == nxt == "}" and self._context & contexts.ARGUMENT:
                if self._read(2) == "}":
                    return self._handle_argument_end()
                self._emit_text("}")
            elif this == nxt == "[" and self._can_recurse():
                # TODO: Only do this if not in a file context:
                # if self._context & contexts.WIKILINK_TEXT:
                #     self._fail_route()
                if not self._context & contexts.NO_WIKILINKS:
                    self._parse_wikilink()
                else:
                    self._emit_text("[")
            elif this == "|" and self._context & contexts.WIKILINK_TITLE:
                self._handle_wikilink_separator()
            elif this == nxt == "]" and self._context & contexts.WIKILINK:
                return self._handle_wikilink_end()
            elif this == "[":
                self._parse_external_link(True)
            elif this == ":" and self._read(-1) not in MARKERS:
                self._parse_external_link(False)
            elif this == "]" and self._context & contexts.EXT_LINK_TITLE:
                return self._pop()
            elif (
                this == "="
                and not self._global & contexts.GL_HEADING
                and not self._context & contexts.TEMPLATE
            ):
                if self._read(-1) in ("\n", START):
                    self._parse_heading()
                else:
                    self._emit_text("=")
            elif this == "=" and self._context & contexts.HEADING:
                return self._handle_heading_end()
            elif this == "\n" and self._context & (contexts.HEADING | contexts.STYLE):
                # Wiki quote formatting is line-scoped. An unmatched opener must
                # not consume headings or paragraphs on subsequent lines.
                self._fail_route()
            elif this == "&":
                self._parse_entity()
            elif this == "<" and nxt == "!":
                if self._read(2) == self._read(3) == "-":
                    self._parse_comment()
                else:
                    self._emit_text(this)
            elif this == "<" and nxt == "/" and self._read(2) is not END:
                if self._context & contexts.TAG_BODY:
                    self._handle_tag_open_close()
                else:
                    self._handle_invalid_tag_start()
            elif this == "<" and not self._context & contexts.TAG_CLOSE:
                if self._can_recurse():
                    self._parse_tag()
                else:
                    self._emit_text("<")
            elif this == ">" and self._context & contexts.TAG_CLOSE:
                return self._handle_tag_close_close()
            elif this == nxt == "'":
                result = self._parse_style()
                if result is not None:
                    return result
            elif self._read(-1) in ("\n", START) and this in ("#", "*", ";", ":"):
                self._handle_list()
            elif self._read(-1) in ("\n", START) and (
                this == nxt == self._read(2) == self._read(3) == "-"
            ):
                self._handle_hr()
            elif this in ("\n", ":") and self._context & contexts.DL_TERM:
                self._handle_dl_term()
                if this == "\n":
                    # Kill potential table contexts
                    self._context &= ~contexts.TABLE_CELL_LINE_CONTEXTS
            # Start of table parsing
            elif (
                this == "{"
                and nxt == "|"
                and (
                    self._read(-1) in ("\n", START)
                    or (
                        self._read(-2) in ("\n", START)
                        and cast(str, self._read(-1)).isspace()
                    )
                )
            ):
                if self._can_recurse() and self._head < self._closers.table:
                    self._parse_table()
                else:
                    self._emit_text("{")
            elif self._context & contexts.TABLE_OPEN:
                if this == nxt == "|" and self._context & contexts.TABLE_TD_LINE:
                    if self._context & contexts.TABLE_CELL_OPEN:
                        return self._handle_table_cell_end()
                    self._handle_table_cell("||", "td", contexts.TABLE_TD_LINE)
                elif this == nxt == "|" and self._context & contexts.TABLE_TH_LINE:
                    if self._context & contexts.TABLE_CELL_OPEN:
                        return self._handle_table_cell_end()
                    self._handle_table_cell("||", "th", contexts.TABLE_TH_LINE)
                elif this == nxt == "!" and self._context & contexts.TABLE_TH_LINE:
                    if self._context & contexts.TABLE_CELL_OPEN:
                        return self._handle_table_cell_end()
                    self._handle_table_cell("!!", "th", contexts.TABLE_TH_LINE)
                elif this == "|" and self._context & contexts.TABLE_CELL_STYLE:
                    return self._handle_table_cell_end(reset_for_style=True)
                # on newline, clear out cell line contexts
                elif this == "\n" and self._context & contexts.TABLE_CELL_LINE_CONTEXTS:
                    self._context &= ~contexts.TABLE_CELL_LINE_CONTEXTS
                    self._emit_text(this)
                elif self._read(-1) in ("\n", START) or (
                    self._read(-2) in ("\n", START)
                    and cast(str, self._read(-1)).isspace()
                ):
                    if this == "|" and nxt == "}":
                        if self._context & contexts.TABLE_CELL_OPEN:
                            return self._handle_table_cell_end()
                        if self._context & contexts.TABLE_ROW_OPEN:
                            return self._handle_table_row_end()
                        return self._handle_table_end()
                    if this == "|" and nxt == "-":
                        if self._context & contexts.TABLE_CELL_OPEN:
                            return self._handle_table_cell_end()
                        if self._context & contexts.TABLE_ROW_OPEN:
                            return self._handle_table_row_end()
                        self._handle_table_row()
                    elif this == "|":
                        if self._context & contexts.TABLE_CELL_OPEN:
                            return self._handle_table_cell_end()
                        self._handle_table_cell("|", "td", contexts.TABLE_TD_LINE)
                    elif this == "!":
                        if self._context & contexts.TABLE_CELL_OPEN:
                            return self._handle_table_cell_end()
                        self._handle_table_cell("!", "th", contexts.TABLE_TH_LINE)
                    else:
                        self._emit_text(this)
                else:
                    self._emit_text(this)

            else:
                self._emit_text(this)
            self._head += 1

    def tokenize(self, text: str) -> list[tokens.Token]:
        split = self.regex.split(text)
        self._text = [segment for segment in split if segment]
        self._head = self._global = self._depth = 0
        self._bad_routes = set()
        self._closers = ClosingPositions.from_segments(self._text)
        self.work = 0
        self._work_limit = self.MAX_WORK_PER_CHAR * len(text) + 1024

        try:
            result = self._parse()
        except BadRoute as exc:  # pragma: no cover (untestable/exceptional case)
            raise ParserError("Python tokenizer exited with BadRoute") from exc
        if self._stacks:  # pragma: no cover (untestable/exceptional case)
            err = "Python tokenizer exited with non-empty token stack"
            raise ParserError(err)
        return result
