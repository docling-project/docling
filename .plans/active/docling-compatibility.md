# DoclingDocument Version Compatibility Strategy

> Generated against library versions **v2.36.0 – v2.85.0**,
> schema versions **1.4.0 → 1.10.0** (50 releases).

---

## Part 1 — Change Taxonomy: Last 50 Releases

### Overview

In 50 library releases the schema `version` field embedded in every
`DoclingDocument` JSON was only bumped **6 times** (1.4 → 1.5 → 1.6 → 1.7 →
1.8 → 1.9 → 1.10), yet there were **36 individual schema-touching commits**.
**30 of those 36 shipped additive model changes without any schema version bump**
and without any downgrade-projection companion.

### Category legend

| Symbol | Category | Pydantic behaviour on old client |
|---|---|---|
| 🟡 | New enum value added | Hard crash — enum membership check fails |
| 🔵 | New optional field / model added | Hard crash — `extra="forbid"` on `NodeItem` |
| 🟣 | New DocItem subtype in union | Hard crash — discriminator lookup fails |
| 🔴 | Enum value removed / renamed | Silent wrong result (aliases usually kept) |
| 🟢 | New required field | Hard crash — missing field error |
| 🟠 | New stricter validation rule | Old documents rejected by new client |

### Count summary (50-release vs 30-release window)

| Category | 50-release | Old client crash? |
|---|---:| ---|
| 🟡 New enum value | **14** | Yes — hard crash |
| 🔵 New optional field / model | **13** | Yes — hard crash (`extra="forbid"`) |
| 🟣 New DocItem subtype in union | **4** | Yes — discriminator fail |
| 🔴 Enum value removed / renamed | **3** | Partial — silent wrong result |
| 🟢 New required field | **0** | — |
| 🟠 New stricter validation rule | **2** | Old data fails on new client |
| **Total schema-touching commits** | **36** | |
| — with schema version bump | **6** | |
| — **without** schema version bump | **30** | |

> **Zero required-field additions in all 50 releases.** Every breaking change
> is either a new optional field or a new enum value — confirming that the
> entire problem space is solvable with two targeted changes (see Part 2).

---

### Schema version discipline — full history

| Schema bump | Library version | Downgrade projector? | Changes covered |
|---|---|---|---|
| 1.0.0 → 1.1.0 | pre-v2.36 | ✅ Yes | `transform_to_content_layer` migrates old `PAGE_HEADER/PAGE_FOOTER` — the **only ever-written projector** |
| 1.1.0 → 1.2.0 | pre-v2.36 | ❌ No | Inline groups, revamped Markdown export |
| 1.2.0 → 1.3.0 | pre-v2.36 | ❌ No | Serializers, text formatting |
| 1.3.0 → 1.4.0 | pre-v2.36 | ❌ No | Table annotations |
| 1.4.0 → 1.5.0 | v2.38.2 | ❌ No | Remodelled lists, `ListGroup`, deprecated `UnorderedList/OrderedList` |
| 1.5.0 → 1.6.0 | v2.45.0 | ❌ No | Rich table cells (`RichTableCell`, `AnyTableCell` union) |
| 1.6.0 → 1.7.0 | v2.47.0 | ❌ No | Fillable `TableCell.fillable` field |
| 1.7.0 → 1.8.0 | v2.49.0 | ❌ No | Metadata model hierarchy (`BasePrediction`, `BaseMeta`, `meta` field on `NodeItem`) |
| 1.8.0 → 1.9.0 | v2.57.0 | ❌ No | `FineRef`, `DocItem.comments` |
| 1.9.0 → 1.10.0 | v2.69.0 | ❌ No | Field data model, 7 new `DocItemLabel` values, new top-level lists |

---

### Commit-level classification

#### v2.36 – v2.56 (new in 50-release window)

| Library ver. | Schema ver. | PR | Category | What changed | Client impact |
|---|---|---|---|---|---|
| **v2.38.2** | 1.4 → **1.5** | [#339](https://github.com/docling-project/docling-core/pull/339) | 🟣 New subtype, 🔴 Enum renamed | `UnorderedList`/`OrderedList` replaced by `ListGroup` in `groups` union. `GroupLabel.ORDERED_LIST` deprecated. | **Hard crash**: old union does not know `ListGroup` discriminator. |
| **v2.39.0** | 1.5 (no bump) | [#345](https://github.com/docling-project/docling-core/pull/345) | 🟡 New enum values | Added `ContentLayer.INVISIBLE = "invisible"` and `ContentLayer.NOTES = "notes"`. | **Hard crash**: Pydantic enum validation fails for `content_layer="invisible"` or `"notes"`. |
| **v2.44.2** | 1.5 (no bump) | [#349](https://github.com/docling-project/docling-core/pull/349) | 🟠 New validation rule | Added `_validate_rules()`: enforces `ListGroup` children are `ListItem`s, non-root groups are non-empty. | Old data with mixed-child `ListGroup` now **rejected** by new client. |
| **v2.45.0** | 1.5 → **1.6** | [#368](https://github.com/docling-project/docling-core/pull/368) | 🟣 New subtype, 🔵 New field | Added `RichTableCell` with `ref` field. `TableData.table_cells` changed to `list[AnyTableCell]`. | **Hard crash**: extra `ref` field rejected by `extra="forbid"` on old `TableCell` parser. |
| **v2.46.0** | 1.6 (no bump) | [#378](https://github.com/docling-project/docling-core/pull/378) | 🔵 New field (internal) | Added `DoclingDocument.filter(page_nrs)` API, internal `_DocIndex` helper. No new serialised fields. | Low risk — internal only. |
| **v2.47.0** | 1.6 → **1.7** | [#384](https://github.com/docling-project/docling-core/pull/384) | 🔵 New optional field | Added `TableCell.fillable: bool = False`. | **Hard crash** if old model is strict and receives `fillable=True`. |
| **v2.49.0** | 1.7 → **1.8** | [#408](https://github.com/docling-project/docling-core/pull/408) | 🔵 New optional field | Introduced `_ExtraAllowingModel`, `BasePrediction`, `BaseMeta`, `meta: Optional[BaseMeta]` on `NodeItem`/`FloatingItem`. | **Hard crash** on `meta` field — `NodeItem` has `extra="forbid"`. |
| **v2.50.0** | 1.8 (no bump) | [#413](https://github.com/docling-project/docling-core/pull/413) | 🟡 New enum value | Added `CodeLanguageLabel.JSON = "JSON"`. | **Hard crash** for old clients receiving a `CodeItem` with `code_language="JSON"`. |
| **v2.57.0** | 1.8 → **1.9** | [#465](https://github.com/docling-project/docling-core/pull/465) | 🔵 New field, 🟣 New subtype | Added `DocItem.comments: list[FineRef]`. New `FineRef` subclass with `range` field. | **Hard crash** on `comments` field with strict model. |
| **v2.60.2** | 1.9 (no bump) | [#484](https://github.com/docling-project/docling-core/pull/484) | 🟡 New enum values | Added several `CodeLanguageLabel` values (Linguist alignment). | **Hard crash** for old clients receiving `CodeItem`s with new language labels. |
| **v2.61.0** | 1.9 (no bump) | [#426](https://github.com/docling-project/docling-core/pull/426) | 🔵 New field, 🟣 New subtype | Added `DocItem.source: list[SourceType]`. New `BaseSource`/`TrackSource` models with `kind` discriminator. No version bump — no signal to client. | **Hard crash** on `source` field with strict model. |
| **v2.62.0** | 1.9 (no bump) | [#502](https://github.com/docling-project/docling-core/pull/502) | 🔵 New optional field | Added `BitMapResource.image: Optional[ImageRef]` and `BitMapResource.mode: ImageRefMode`. Deprecated `uri`. | **Hard crash** on `image`/`mode` fields with strict model. |
| **v2.63.0** | 1.9 (no bump) | [#507](https://github.com/docling-project/docling-core/pull/507) | 🔵 New field, 🔴 Type renamed | Added `PdfShape` and `SegmentedPdfPage.shapes`. Deprecated `PdfLine`/`lines`. | **Hard crash** on `shapes` field with strict model. |
| **v2.64.0** | 1.9 (no bump) | [#515](https://github.com/docling-project/docling-core/pull/515) | 🔵 New optional field | Added `PdfPage.widgets: list[PdfWidget]`, `PdfPage.hyperlinks: list[PdfWidget]`. | **Hard crash** on `widgets`/`hyperlinks` fields with strict model. |
| **v2.65.0** | 1.9 (no bump) | [#516](https://github.com/docling-project/docling-core/pull/516) | 🔴 Type corrected | Fixed `hyperlinks` element type from `PdfWidget` → `PdfHyperlink`. | Clients at exactly v2.64 see a type mismatch. Low real-world impact. |
| **v2.69.0** | 1.9 → **1.10** | [#519](https://github.com/docling-project/docling-core/pull/519) | 🟣 New subtypes, 🟡 New enum values | Added `FieldRegionItem`, `FieldHeadingItem`, `FieldItem`, `FieldValueItem` in `texts` union. Added `field_regions`, `field_items` top-level lists. Added 7 `DocItemLabel` values. | **Hard crash**: discriminator fails; unknown top-level keys. |
| **v2.70.1** | 1.10 (no bump) | [#529](https://github.com/docling-project/docling-core/pull/529) | 🔴 Enum values removed, 🟡 New values | Synced `PictureClassificationLabel` with v2.0 ML model: removed 11 primary values, added 14 new ones. Old values kept as deprecated aliases. | **Hard crash** for old clients receiving new label values. |
| **v2.70.2** | 1.10 (no bump) | [#573](https://github.com/docling-project/docling-core/pull/573) | 🔵 New optional field | Added `CodeMetaField` and `FloatingMeta.code: Optional[CodeMetaField]`. | **Hard crash** on `code` meta field with strict model. |
| **v2.70.2** | 1.10 (no bump) | [#561](https://github.com/docling-project/docling-core/pull/561) | 🟡 New enum value | Added `DocItemLabel.HANDWRITTEN_TEXT = "handwritten_text"`. | **Hard crash** when old client receives a node with label `"handwritten_text"`. |
| **v2.70.2** | 1.10 (no bump) | [#565](https://github.com/docling-project/docling-core/pull/565) | 🟠 New validation rule | Added `_validate_unique_refs`: raises on duplicate `self_ref`. | Old data with duplicate refs now **rejected** by new client. |
| **v2.72.0** | 1.10 (no bump) | [#579](https://github.com/docling-project/docling-core/pull/579) | 🟡 New enum values | Added `CodeLanguageLabel.DOCLANG`, `LATEX`, `TIKZ`. | **Hard crash** for old clients receiving `CodeItem`s with these labels. |
| **v2.76.0** | 1.10 (no bump) | [#611](https://github.com/docling-project/docling-core/pull/611) | 🔵 New optional fields, 🟡 New enum | Added `BaseMeta.language`, `BaseMeta.entities`, `LanguageMetaField`, `EntitiesMetaField`, `HumanLanguageLabel` (~180 values). | **Hard crash** on `language`/`entities` fields with strict model. |
| **v2.77.1** | 1.10 (no bump) | [#622](https://github.com/docling-project/docling-core/pull/622) | 🔵 New optional field, 🟡 New enum | Added `Orientation` enum (`ROT_0/90/180/270`), `TableData.orientation: Orientation`. | **Hard crash** on `orientation` field + new enum values with strict model. |
| **v2.78.1** | 1.10 (no bump) | [#617](https://github.com/docling-project/docling-core/pull/617) | 🔵 New optional fields, 🟡 New enum | Added `BaseMeta.keywords`, `BaseMeta.topics`, `KeywordsMetaField`, `TopicsMetaField`, `MetaFieldName.KEYWORDS/TOPICS`. | **Hard crash** on `keywords`/`topics` fields with strict model. |
| **v2.83.1** | 1.10 (no bump) | [#654](https://github.com/docling-project/docling-core/pull/654) | 🟡 New enum value | Added `PictureClassificationLabel.OTHER_CHART = "other_chart"`. Deprecated `CHART`. | **Hard crash** when old client receives a picture with label `"other_chart"`. |

---

## Part 2 — Recommended Architecture

### Problem framing

`DoclingDocument` has **two independent versions**:

| Version | Location | Current value | What it governs |
|---|---|---|---|
| Library version | `pyproject.toml` | `2.86.0` | Python package release cadence |
| Schema version | `CURRENT_VERSION` in `document.py` | `1.10.0` | Data schema embedded in every serialised document |

The existing [`check_version_is_compatible`](docling_core/types/doc/document.py) validator raises a hard
`ValueError` when `doc.minor > sdk.minor`, and `NodeItem` carries
`model_config = ConfigDict(extra="forbid")`, so any unknown field also crashes
immediately. Both are correctness guards, but they make clients brittle against
additive server changes.

---

### Layer 1 — Make the client a Tolerant Reader (highest leverage)

The single most impactful change. Requires only a docling-core upgrade on the
client; no server-side negotiation needed.

#### 1a. Enum coercion via `BeforeValidator`

Replace bare enum validation on fields like `content_layer` with a coercing
wrapper that falls back gracefully and emits a warning instead of crashing:

```python
# In document.py (or a shared compat module)
def _coerce_content_layer(v: object) -> object:
    """Accept any unknown enum value; log a warning and fall back to BODY."""
    if isinstance(v, str) and v not in ContentLayer._value2member_map_:
        import logging
        logging.getLogger(__name__).warning(
            "Unknown ContentLayer value %r — treating as BODY", v
        )
        return ContentLayer.BODY
    return v

# On NodeItem:
content_layer: Annotated[
    ContentLayer, BeforeValidator(_coerce_content_layer)
] = ContentLayer.BODY
```

This applies equally to `DocItemLabel`, `CodeLanguageLabel`,
`PictureClassificationLabel`, and any other enum field on a serialised model.

#### 1b. Switch `extra="forbid"` → `extra="ignore"` on `NodeItem`

```python
class NodeItem(BaseModel):
    model_config = ConfigDict(extra="ignore")   # was "forbid"
```

Unknown fields from a newer server are silently dropped instead of crashing.
This alone handles every "new optional field" case observed in the 50-release
analysis.

> **Trade-off:** `extra="ignore"` means typos in field names during
> construction are also silently dropped. Mitigate by keeping
> `extra="forbid"` in unit tests (inject a strict config via a test fixture)
> or by running strict validation only at construction time, not at
> wire-deserialization time.

#### 1c. Soften the version validator

Downgrade from a hard error to a warning for same-major, higher-minor
documents:

```python
@field_validator("version")
@classmethod
def check_version_is_compatible(cls, v: str) -> str:
    sdk_match = re.match(VERSION_PATTERN, CURRENT_VERSION)
    doc_match = re.match(VERSION_PATTERN, v)
    if doc_match is None or sdk_match is None:
        raise ValueError(f"Cannot parse version {v!r}")
    sdk_major = int(sdk_match["major"])
    doc_major, doc_minor = int(doc_match["major"]), int(doc_match["minor"])
    sdk_minor = int(sdk_match["minor"])
    if doc_major != sdk_major:
        raise ValueError(
            f"Doc major version {doc_major} != SDK major {sdk_major}: incompatible."
        )
    if doc_minor > sdk_minor:
        warnings.warn(
            f"Doc schema {v} is newer than SDK schema {CURRENT_VERSION}. "
            "Unknown fields and enum values will be ignored.",
            UserWarning,
            stacklevel=2,
        )
    return v   # keep the original version — do not overwrite with SDK version
```

Preserving the original version string (rather than overwriting with
`CURRENT_VERSION`) lets downstream code inspect what schema version was
actually received.

---

### Layer 2 — Server-side downgrade projectors

Every schema minor bump must be accompanied by a registered projector that
can reduce a newer document to one parseable by the previous minor version.
Ship a `docling_core/compat.py` module:

```python
# docling_core/compat.py
from packaging.version import Version

_projectors: dict[tuple[int, int], callable] = {}

def register_projector(from_minor: int, to_minor: int):
    """Decorator: register a function that downgrades from_minor → to_minor."""
    def decorator(fn):
        _projectors[(from_minor, to_minor)] = fn
        return fn
    return decorator

def project_to(doc: "DoclingDocument", target_version: str) -> "DoclingDocument":
    """Return a copy of doc projected to be parseable by target_version SDK."""
    target = Version(target_version)
    current = Version(doc.version)
    data = doc.model_dump(mode="python")
    for minor in range(int(current.minor), int(target.minor), -1):
        fn = _projectors.get((minor, minor - 1))
        if fn:
            data = fn(data)
    from docling_core.types.doc.document import DoclingDocument
    return DoclingDocument.model_validate(data)

# Example — ships alongside the 1.11.0 schema bump:
@register_projector(from_minor=11, to_minor=10)
def _project_1_11_to_1_10(data: dict) -> dict:
    """Map SOCIAL content layer to BODY for clients at schema 1.10."""
    for item in data.get("texts", []):
        if item.get("content_layer") == "social":
            item["content_layer"] = "body"
    data["version"] = "1.10.0"
    return data
```

The server reads an `Accept-Schema-Version` request header and runs the
projection chain before returning JSON:

```python
# Server-side (conceptual)
doc: DoclingDocument = converter.convert(pdf)
client_version = request.headers.get("Accept-Schema-Version", CURRENT_VERSION)
if Version(client_version) < Version(doc.version):
    doc = project_to(doc, target_version=client_version)
return doc.model_dump_json()
```

> **Rule:** Every schema minor bump (1.10 → 1.11) **must** be accompanied by
> a `@register_projector` function committed in the same PR. Enforce this with
> a unit test that asserts `len(_projectors) == CURRENT_MINOR - 1`.

---

### Layer 3 — Schema version advertisement (low cost, optional)

Expose the server's schema version in a capabilities or health endpoint so
clients can check proactively:

```json
GET /capabilities
{
  "docling_core_version": "2.86.0",
  "document_schema_version": "1.10.0",
  "min_compatible_schema_version": "1.0.0"
}
```

The client reads this on startup and logs a warning (or refuses to start)
if the server's schema version is ahead of the client's `CURRENT_VERSION`.

---


### Change classification and version-bump rules

Formalise in `CONTRIBUTING.md`:

| Change | Schema version bump | Required companion work |
|---|---|---|
| New enum value | Minor bump | Downgrade projector mapping new value to nearest old equivalent |
| New optional field on existing model | Minor bump | Downgrade projector stripping the field |
| New required field | Minor bump | Downgrade projector supplying a safe default |
| New `DocItem` subtype (new discriminator label) | Minor bump | Projector converting to nearest existing type or stripping item |
| Field removed or renamed | **Major bump** | Migration guide; old clients must upgrade |
| Semantic change (same name, different meaning) | **Major bump** | Migration guide |

---

### Summary

| Mechanism | Handles | Cost | Client upgrade needed? |
|---|---|---|---|
| Tolerant Reader (`extra="ignore"` + enum coercion) | New fields, new enum values | Low — one-time model change | Yes |
| Softer version validator (warn instead of raise on higher minor) | Version mismatch detection | Minimal | Yes |
| Server downgrade projector | All additive changes | Medium — per-bump projector + server endpoint | No — client sends version header |
| Capabilities endpoint | Proactive mismatch detection | Low — server-side only | No — opt-in |

> **Bottom line:** The two-mechanism Tolerant Reader fix (`extra="ignore"` +
> `BeforeValidator` on enum fields) is sufficient for **100% of the observed
> breaking-change space** across all 50 releases analysed. No required-field
> addition has ever appeared. The downgrade-projector layer additionally
> handles future new `DocItem` subtypes (4 occurrences in 50 releases) where
> the client truly cannot process a feature at all.


### Conclusion

The following sections describe 3 possible layers (non-exclusive) for compatibility.
Since we want to keep the benefits of strict validation and keep the compatibility logic in the server system, we will proceed with the implementation of the **Layer 2** as remediation strategy.

As part of the solution, we will need to:
- automate the downgrade projectors as much as possible
- add validation scripts in the CI/CD
- coordinate the downgrade projectors and `DoclingDocument` versioning with `Doclang` 
