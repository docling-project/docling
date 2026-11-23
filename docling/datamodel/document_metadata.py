# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Structured document-level metadata model and attachment helpers."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import BaseMeta
from pydantic import BaseModel, Field

DOCUMENT_METADATA_META_KEY = "docling__document_metadata"


def normalize_metadata_value(value: Any) -> Any:
    """Convert parsed metadata values into JSON-serializable primitives."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): normalize_metadata_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_metadata_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalize_string_list(value: Any) -> list[str]:
    """Normalize a scalar or sequence metadata value into a deduplicated list."""
    if value is None:
        return []
    raw_items: list[Any]
    if isinstance(value, str):
        raw_items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    items: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        if isinstance(raw, dict):
            candidate = raw.get("name") or raw.get("author") or raw.get("value")
            text = str(candidate).strip() if candidate is not None else ""
        else:
            text = str(raw).strip()
        if text and text not in seen:
            seen.add(text)
            items.append(text)
    return items


class DocumentMetadata(BaseModel):
    """Normalized document-level metadata extracted by declarative backends."""

    title: str | None = Field(
        default=None, description="Document title from file metadata."
    )
    authors: list[str] = Field(
        default_factory=list, description="Document authors or creators."
    )
    date: str | None = Field(
        default=None,
        description="Creation or publication date string (ISO-8601 when available).",
    )
    language: str | None = Field(
        default=None, description="Primary language tag (e.g. 'en', 'zh-CN')."
    )
    description: str | None = Field(
        default=None, description="Document summary, abstract, or subject."
    )
    keywords: list[str] = Field(
        default_factory=list, description="Subject keywords or tags."
    )
    publisher: str | None = Field(
        default=None, description="Publisher or organization name."
    )
    identifier: str | None = Field(
        default=None,
        description="Canonical identifier such as DOI, ISBN, URI, or UUID.",
    )
    custom: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional format-specific metadata fields.",
    )

    def is_empty(self) -> bool:
        """Return True when no metadata field has a non-empty value."""
        return not any(
            [
                self.title,
                self.authors,
                self.date,
                self.language,
                self.description,
                self.keywords,
                self.publisher,
                self.identifier,
                self.custom,
            ]
        )

    def to_meta_dict(self) -> dict[str, Any]:
        """Serialize non-empty fields into a compact dictionary."""
        return self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)

    def attach_to_document(self, doc: DoclingDocument) -> None:
        """Attach non-empty metadata to ``doc.body.meta`` without clobbering."""
        if self.is_empty():
            return
        existing: dict[str, Any] = {}
        if doc.body.meta is not None:
            existing = doc.body.meta.model_dump(exclude_none=True)
        existing[DOCUMENT_METADATA_META_KEY] = self.to_meta_dict()
        doc.body.meta = BaseMeta.model_validate(existing)

    @classmethod
    def from_mapping(cls, meta: dict[Any, Any]) -> DocumentMetadata:
        """Build ``DocumentMetadata`` from a key-value mapping (e.g. YAML frontmatter)."""
        known_keys = frozenset(
            {
                "title",
                "author",
                "authors",
                "date",
                "created",
                "published",
                "lang",
                "language",
                "description",
                "summary",
                "abstract",
                "keywords",
                "tags",
                "publisher",
                "doi",
                "identifier",
                "isbn",
                "url",
            }
        )
        lowered = {str(k).strip().lower(): (k, v) for k, v in meta.items()}

        def _first_str(*keys: str) -> str | None:
            for key in keys:
                if key in lowered:
                    val = normalize_metadata_value(lowered[key][1])
                    if val is not None:
                        text = str(val).strip()
                        if text:
                            return text
            return None

        authors: list[str] = []
        for key in ("authors", "author"):
            if key in lowered:
                authors = normalize_string_list(lowered[key][1])
                if authors:
                    break

        keywords: list[str] = []
        for key in ("keywords", "tags"):
            if key in lowered:
                keywords = normalize_string_list(lowered[key][1])
                if keywords:
                    break

        custom = {
            str(orig_key): normalize_metadata_value(val)
            for low_key, (orig_key, val) in lowered.items()
            if low_key not in known_keys and val is not None
        }

        return cls(
            title=_first_str("title"),
            authors=authors,
            date=_first_str("date", "published", "created"),
            language=_first_str("language", "lang"),
            description=_first_str("description", "summary", "abstract"),
            keywords=keywords,
            publisher=_first_str("publisher"),
            identifier=_first_str("identifier", "doi", "isbn", "url"),
            custom=custom,
        )
