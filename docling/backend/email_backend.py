# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from datetime import datetime
from email.header import decode_header, make_header
from email.message import EmailMessage, Message
from email.utils import format_datetime, formataddr
from io import BytesIO
from pathlib import Path

from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
)

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.calendar.parser import Component, parse_components, unfold_text
from docling.backend.calendar.render import add_calendar_components
from docling.backend.calendar_backend import holds_calendar, read_calendar_text
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.backend_options import EmailBackendOptions, HTMLBackendOptions
from docling.datamodel.base_models import FormatToExtensions, InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

# mail-parser is only installed by the `format-email` extra, but
# DocumentConverter imports every backend eagerly. Importing it at module load
# would therefore break `import docling` on installs that omit the extra (the
# slim packages in particular). Guard the import like the opendocument and xbrl
# backends do, and surface the failure only when an email is actually parsed.
# See https://github.com/docling-project/docling/issues/3613.
_MAILPARSER_AVAILABLE: bool = False
_MAILPARSER_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - import-time guard
    import mailparser

    _MAILPARSER_AVAILABLE = True
except ImportError as e:  # pragma: no cover - import-time guard
    _MAILPARSER_IMPORT_ERROR = e

# python-oxmsg (MIT, from the python-docx/python-pptx author, built on olefile)
# reads Outlook .msg files. We assemble a standard email.message.EmailMessage
# from the parsed message, which mailparser then parses through the same path as
# .eml input. It ships in the `format-email` extra, so guard the import for the
# same slim-install reason as mailparser above.
_OXMSG_AVAILABLE: bool = False
_OXMSG_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - import-time guard
    from oxmsg import Message as OxMsgMessage

    _OXMSG_AVAILABLE = True
except ImportError as e:  # pragma: no cover - import-time guard
    _OXMSG_IMPORT_ERROR = e

_log = logging.getLogger(__name__)

# RFC 5322 3.2.3 specials, minus "." which parses back unquoted and is common
# in real names. A rendered display name holding any of these is quoted.
_ADDRESS_SPECIALS = re.compile(r'[][()<>@,:;"\\]')
_ADDRESS_ESCAPES = re.compile(r'[\\"]')

# OLE2 / Compound File Binary signature that prefixes every Outlook .msg file.
_MSG_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# MAPI PidTagRecipientType (0x0C15) values on each .msg recipient row.
_RECIPIENT_TYPE_PROP_ID = 0x0C15
_RECIPIENT_TYPE_CC = 2
_RECIPIENT_TYPE_BCC = 3

_INSTALL_HINT = (
    "The 'mail-parser' package is required to process email files. "
    "Install it with `pip install 'docling-slim[format-email]'`."
)

_MSG_INSTALL_HINT = (
    "The 'python-oxmsg' package is required to process Outlook .msg files. "
    "Install it with `pip install 'docling-slim[format-email]'`."
)

# Gmail sends the invite.ics copy of an invitation as application/ics.
_CALENDAR_CONTENT_TYPES = frozenset({"text/calendar", "application/ics"})
_CALENDAR_ENTRIES = frozenset({"VEVENT", "VTODO", "VJOURNAL", "VFREEBUSY"})
# The calendar properties that are shown when a calendar has no entry.
_CALENDAR_LABELS = ("X-WR-CALNAME", "NAME", "X-WR-CALDESC", "DESCRIPTION")
# The delimiter mailparser puts between the parts of its whole-message body.
_MAILPARSER_BODY_DELIMITER = "\n--- mail_boundary ---\n"


@dataclass(frozen=True)
class _BodyParagraph:
    text: str
    content_layer: ContentLayer


def _is_calendar_part(part: Message) -> bool:
    if part.get_content_type() in _CALENDAR_CONTENT_TYPES:
        return True
    filename = part.get_filename()
    return (
        filename is not None
        and Path(filename).suffix.lower().removeprefix(".")
        in FormatToExtensions[InputFormat.CALENDAR]
    )


def _is_new_entry(entry: Component, seen: set[tuple[str, str, str]]) -> bool:
    uid = entry.text("UID").strip()
    if not uid:
        return True
    key = (uid, entry.text("RECURRENCE-ID").strip(), entry.text("SEQUENCE").strip())
    if key in seen:
        return False
    seen.add(key)
    return True


def _new_components(
    components: list[Component], seen: set[tuple[str, str, str]]
) -> list[Component]:
    """Return the components of one calendar part that add something to show.

    A calendar is kept when it has an entry that was not seen before. A
    calendar without entries is kept when it has a name or a description, as
    the .ics backend shows them.
    """
    kept: list[Component] = []
    for component in components:
        if component.name == "VCALENDAR":
            entries = [c for c in component.components if c.name in _CALENDAR_ENTRIES]
            if not entries:
                if any(component.text(name).strip() for name in _CALENDAR_LABELS):
                    kept.append(component)
                continue
            children = [
                child
                for child in component.components
                if child.name not in _CALENDAR_ENTRIES or _is_new_entry(child, seen)
            ]
            if any(child.name in _CALENDAR_ENTRIES for child in children):
                kept.append(replace(component, components=children))
        elif component.name in _CALENDAR_ENTRIES and _is_new_entry(component, seen):
            kept.append(component)
    return kept


def _merge_calendar_parts(
    calendar_parts: list[list[Component]],
) -> tuple[list[list[Component]], list[Component]]:
    """Join the calendar parts and keep the first copy of each entry.

    Gmail sends an invitation twice, as an inline part and as an invite.ics
    attachment. Two entries are the same when their UID, RECURRENCE-ID and
    SEQUENCE are the same.

    Returns:
        The parts that add something to show, and the components to show.
    """
    seen: set[tuple[str, str, str]] = set()
    shown_parts: list[list[Component]] = []
    merged: list[Component] = []
    for components in calendar_parts:
        kept = _new_components(components, seen)
        if kept:
            shown_parts.append(components)
            merged.extend(kept)
    return shown_parts, merged


class EmailDocumentBackend(DeclarativeDocumentBackend):
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: EmailBackendOptions | None = None,
    ):
        # Raised before super().__init__() so a missing optional dependency
        # gives an actionable message rather than a NameError when mailparser
        # is dereferenced below.
        if not _MAILPARSER_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _MAILPARSER_IMPORT_ERROR
        if options is None:
            options = EmailBackendOptions()
        super().__init__(in_doc, path_or_stream, options)

        self.options: EmailBackendOptions = options
        self.valid = False
        self.is_msg = False
        self.mail: mailparser.MailParser | None = None

        try:
            raw = self._read_bytes()
            self.is_msg = raw.startswith(_MSG_MAGIC)
            if self.is_msg:
                raw = self._msg_to_rfc822_bytes(raw)
            self.mail = mailparser.parse_from_bytes(raw)

            self.valid = self.mail is not None
        except ImportError:
            raise
        except Exception as exc:
            raise DocumentLoadError(
                f"Could not initialize email backend for file with hash {self.document_hash}."
            ) from exc

    def _read_bytes(self) -> bytes:
        if isinstance(self.path_or_stream, BytesIO):
            return self.path_or_stream.getvalue()
        if isinstance(self.path_or_stream, Path):
            return self.path_or_stream.read_bytes()
        raise TypeError(f"Unsupported input type: {type(self.path_or_stream)}")

    @staticmethod
    def _decode_rfc2047(value: str) -> str:
        """Decode any RFC 2047 encoded-words in a header string.

        Args:
            value: Raw header string, possibly containing RFC 2047 encoded-words.

        Returns:
            Decoded string, or *value* unchanged if decoding raises.
        """
        try:
            return str(make_header(decode_header(value)))
        except Exception:
            return value

    @staticmethod
    def _header_safe(value: str) -> str:
        """Collapse a header value to a single line of single-spaced text.

        `.msg` header assembly needs single-line values, and a decoded RFC 2047
        encoded-word must not stand up a forged header in the rendered document.
        `str.split()` covers every character `str.splitlines()` breaks on, and
        collapsing runs also unfolds a folded header to one space.

        Args:
            value: Header string to sanitize.

        Returns:
            Single-line, single-spaced string.
        """
        return " ".join(value.split())

    @staticmethod
    def _msg_to_rfc822_bytes(data: bytes) -> bytes:
        """Project an Outlook ``.msg`` (OLE2/CFB) onto RFC 822 bytes.

        python-oxmsg reads the MAPI message; we assemble a standard
        ``email.message.EmailMessage`` from it so the ``.msg`` path shares the
        exact body, HTML, address, and attachment handling used for ``.eml``
        input.
        """
        if not _OXMSG_AVAILABLE:
            raise ImportError(_MSG_INSTALL_HINT) from _OXMSG_IMPORT_ERROR

        message = OxMsgMessage.load(data)
        email_message = EmailMessage()

        if message.subject:
            email_message["Subject"] = EmailDocumentBackend._header_safe(
                message.subject
            )
        if message.sender:
            email_message["From"] = EmailDocumentBackend._header_safe(message.sender)

        # Preserve the To/Cc/Bcc split from the recipient rows so downstream
        # rendering (which shows "To" and "Cc") matches the .eml behavior.
        grouped: dict[str, list[str]] = {}
        for recipient in message.recipients:
            formatted = formataddr(
                (recipient.name or "", recipient.email_address or "")
            )
            if not formatted:
                continue
            rtype = recipient.properties.int_prop_value(_RECIPIENT_TYPE_PROP_ID)
            if rtype == _RECIPIENT_TYPE_CC:
                header = "Cc"
            elif rtype == _RECIPIENT_TYPE_BCC:
                header = "Bcc"
            else:
                header = "To"
            grouped.setdefault(header, []).append(formatted)
        for header in ("To", "Cc", "Bcc"):
            if grouped.get(header):
                email_message[header] = EmailDocumentBackend._header_safe(
                    ", ".join(grouped[header])
                )

        if isinstance(message.sent_date, datetime):
            email_message["Date"] = format_datetime(message.sent_date)

        plain_body = message.body
        html_body = message.html_body
        if plain_body:
            email_message.set_content(plain_body)
            if html_body:
                email_message.add_alternative(html_body, subtype="html")
        elif html_body:
            email_message.set_content(html_body, subtype="html")
        else:
            email_message.set_content("")

        for attachment in message.attachments:
            mime_type = attachment.mime_type or "application/octet-stream"
            maintype, _, subtype = mime_type.partition("/")
            # Only filename and MIME type are needed; pass an empty payload to avoid
            # copying attachment content into the RFC 822 byte stream.
            email_message.add_attachment(
                b"",
                maintype=maintype or "application",
                subtype=subtype or "octet-stream",
                filename=attachment.file_name,
            )

        return email_message.as_bytes()

    def is_valid(self) -> bool:
        return self.valid

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.EMAIL}

    @staticmethod
    def _quote_display_name(name: str) -> str:
        """Quote a display name that would otherwise not parse back as itself.

        ``formataddr`` and ``headerregistry.Address`` apply the same rule but
        also RFC 2047 encode a non-ASCII name, which this rendering keeps as
        readable text.
        """
        if _ADDRESS_SPECIALS.search(name):
            return '"' + _ADDRESS_ESCAPES.sub(r"\\\g<0>", name) + '"'
        return name

    def _format_addresses(
        self, addresses: list[tuple[str, str]] | None, fallback: str
    ) -> str:
        if not addresses:
            return fallback

        formatted = []
        for name, email in addresses:
            name = self._header_safe(self._decode_rfc2047(name))
            if name:
                formatted.append(f"{self._quote_display_name(name)} <{email}>")
            else:
                formatted.append(email)

        return ", ".join(formatted)

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split a body into paragraphs, normalising CRLF and lone CR first."""
        text = re.sub(r"\r\n|\r", "\n", text)
        return [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n+", text.strip())
            if paragraph.strip()
        ]

    def _convert_html_part(self, html: str) -> DoclingDocument:
        # The part is already decoded, but its markup can still declare the
        # original charset in a <meta> tag. The UTF-8 byte order mark makes the
        # HTML parser use UTF-8 and ignore that declaration.
        html_stream = BytesIO(html.encode("utf-8-sig"))
        in_doc = InputDocument(
            path_or_stream=html_stream,
            format=InputFormat.HTML,
            filename="email-body.html",
            backend=HTMLDocumentBackend,
        )
        html_stream.seek(0)
        backend = HTMLDocumentBackend(
            in_doc=in_doc,
            path_or_stream=html_stream,
            options=HTMLBackendOptions(add_title=False, infer_furniture=False),
        )
        return backend.convert()

    def _get_calendar_parts(self) -> list[list[Component]]:
        """Parse each calendar part of the message, inline or attached.

        The part is read from its raw bytes, because a folded line can split a
        UTF-8 character.
        """
        assert self.mail is not None

        calendar_parts: list[list[Component]] = []
        for part in self.mail.message.walk():
            if part.is_multipart() or not _is_calendar_part(part):
                continue
            try:
                text = read_calendar_text(
                    part.get_payload(decode=True), part.get_content_charset()
                )
            except DocumentLoadError as exc:
                _log.warning("Skipped a calendar part that cannot be decoded: %s", exc)
                continue
            components = parse_components(text)
            if holds_calendar(components):
                calendar_parts.append(components)
        return calendar_parts

    def _get_fallback_body(self, shown_parts: list[list[Component]]) -> str:
        assert self.mail is not None

        if not shown_parts:
            return self.mail.body
        # mailparser also puts a calendar part in its body. Drop a part only
        # when it parses to a calendar part that is added below, so that text
        # is never lost.
        other_parts = [
            text
            for text in self.mail.text_not_managed
            if parse_components(unfold_text(text)) not in shown_parts
        ]
        return _MAILPARSER_BODY_DELIMITER.join(
            self.mail.text_plain + self.mail.text_html + other_parts
        )

    def _get_body_paragraphs(
        self, shown_parts: list[list[Component]]
    ) -> list[_BodyParagraph]:
        assert self.mail is not None

        # A part being present does not mean it holds a paragraph. A
        # multipart/alternative message can carry a blank or whitespace-only
        # text/plain part beside a real text/html one, which many senders
        # generate automatically. Only return once a part has actually
        # produced text, or the fallbacks below never run and the message
        # renders with no body at all.
        if self.mail.text_plain:
            paragraphs: list[_BodyParagraph] = []
            for part in self.mail.text_plain:
                paragraphs.extend(
                    _BodyParagraph(text=paragraph, content_layer=ContentLayer.BODY)
                    for paragraph in self._split_paragraphs(part)
                )
            if paragraphs:
                return paragraphs

        if self.mail.text_html:
            paragraphs = []
            for part in self.mail.text_html:
                html_doc = self._convert_html_part(part)
                # Export each layer separately so furniture remains available
                # without becoming part of the email's default body export.
                for content_layer in (ContentLayer.BODY, ContentLayer.FURNITURE):
                    # The outer email export escapes these text items.
                    markdown = html_doc.export_to_markdown(
                        included_content_layers={content_layer},
                        escape_html=False,
                        escape_underscores=False,
                    )
                    paragraphs.extend(
                        _BodyParagraph(text=paragraph, content_layer=content_layer)
                        for paragraph in self._split_paragraphs(markdown)
                    )
            if paragraphs:
                return paragraphs

        return [
            _BodyParagraph(text=paragraph, content_layer=ContentLayer.BODY)
            for paragraph in self._split_paragraphs(
                self._get_fallback_body(shown_parts)
            )
        ]

    def _get_date_text(self) -> str:
        assert self.mail is not None

        # mailparser returns a datetime or None here, never the raw header.
        mail_date = self.mail.date
        return mail_date.isoformat() if isinstance(mail_date, datetime) else ""

    def _get_attachment_labels(self) -> list[str]:
        """Return one display label per attachment (name, optional content type).

        Only attachment metadata is surfaced; the encoded payload is never
        included, matching how ``.eml`` attachment content is excluded.

        Filenames are header-derived, so they get the same single-line treatment.
        """
        assert self.mail is not None

        labels: list[str] = []
        for index, attachment in enumerate(self.mail.attachments or []):
            filename = self._header_safe(attachment.get("filename") or "")
            if not filename:
                filename = f"attachment-{index + 1}"
            content_type = self._header_safe(attachment.get("mail_content_type") or "")
            labels.append(f"{filename} ({content_type})" if content_type else filename)
        return labels

    def convert(self) -> DoclingDocument:
        if not self.is_valid() or self.mail is None:
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend failed to init."
            )

        # A .msg is projected onto RFC 822 for conversion, so the origin uses
        # the message/rfc822 mimetype for both inputs (DocumentOrigin only
        # accepts registered MIME types); the .msg filename fallback preserves
        # the distinction when no filename is available.
        origin = DocumentOrigin(
            filename=self.file.name or ("file.msg" if self.is_msg else "file.eml"),
            mimetype="message/rfc822",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        subject = (
            self._header_safe(self.mail.subject)
            if isinstance(self.mail.subject, str)
            else ""
        )
        from_text = self._format_addresses(self.mail.from_, fallback="")
        to_text = self._format_addresses(self.mail.to, fallback="")
        cc_text = self._format_addresses(self.mail.cc, fallback="")
        date_text = self._get_date_text()
        calendar_parts = (
            self._get_calendar_parts()
            if self.options.parse_calendar_invitations
            else []
        )
        shown_parts, calendars = _merge_calendar_parts(calendar_parts)
        body_paragraphs = self._get_body_paragraphs(shown_parts)

        if subject:
            doc.add_title(text=subject)
        if from_text:
            doc.add_text(label=DocItemLabel.TEXT, text=f"From: {from_text}")
        if to_text:
            doc.add_text(label=DocItemLabel.TEXT, text=f"To: {to_text}")
        if cc_text:
            doc.add_text(label=DocItemLabel.TEXT, text=f"Cc: {cc_text}")
        if date_text:
            doc.add_text(label=DocItemLabel.TEXT, text=f"Date: {date_text}")
        for body_paragraph in body_paragraphs:
            doc.add_text(
                label=DocItemLabel.TEXT,
                text=body_paragraph.text,
                content_layer=body_paragraph.content_layer,
            )

        if calendars:
            doc.add_heading(text="Calendar", level=2)
            add_calendar_components(doc, calendars, base_level=3, name_as_title=False)

        if self.options.list_attachments:
            attachment_labels = self._get_attachment_labels()
            if attachment_labels:
                doc.add_heading(text="Attachments", level=2)
                attachments_group = doc.add_list_group(name="attachments")
                for label in attachment_labels:
                    doc.add_list_item(text=label, parent=attachments_group)

        return doc
