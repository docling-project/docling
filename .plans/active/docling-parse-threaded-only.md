# Threaded-only Docling Parse backend

**Status:** implementation in progress

## Objective

Make `DoclingThreadedPdfParser` the only docling-parse parser used by Docling.
Remove all `DoclingPdfParser`, pypdfium2, and `pypdfium2_lock` usage from
`docling/backend/docling_parse_backend.py`.

Preserve compatibility for users who select the former non-threaded backend:

- selecting `PdfBackend.DOCLING_PARSE` emits `DeprecationWarning`;
- configuring `backend=DoclingParseDocumentBackend` emits
  `DeprecationWarning`;
- both paths execute through `ThreadedDoclingParseDocumentBackend` and
  `DoclingThreadedPdfParser`;
- deprecated `dlparse_v1`, `dlparse_v2`, and `dlparse_v4` selections also map
  directly to the threaded backend.

## Non-goals

- Do not remove `PyPdfiumDocumentBackend`.
- Do not remove the `format-pdf-pypdfium2` extra.
- Do not remove pdfium outline extraction used by `PyPdfiumDocumentBackend`.
- Do not preserve random page access for the deprecated docling-parse backend.
  The compatibility shim follows the threaded backend contract and exposes
  pages through `iter_pages()`.

## Starting state

`docling_parse_backend.py` contains two implementations:

1. `DoclingParseDocumentBackend` combines `DoclingPdfParser` with pypdfium2.
   Docling-parse supplies page content while pdfium supplies page handles,
   dimensions, rendering, and lifecycle management.
2. `ThreadedDoclingParseDocumentBackend` uses
   `DoclingThreadedPdfParser` for parsing, rendering, and dimensions. Its only
   remaining pdfium operation is the page-count probe used to clip an explicit
   page range.

The threaded backend also constructs a temporary `DoclingPdfParser` document
when reading an outline. In docling-parse 7.19, the threaded parser exposes
document annotations, including the table of contents, so this second parser
is no longer required.

PR [docling-parse#350](https://github.com/docling-project/docling-parse/pull/350)
adds direct inclusive `page_range` support to the threaded loader. This removes
the need for a separate page-count probe or for expanding a range into a list.
PR #344, which provides page-aware outline destinations through threaded
annotations, has been merged.

## Target architecture

```text
new backend selection
    |
    v
ThreadedDoclingParseDocumentBackend
    |
    v
DoclingThreadedPdfParser

deprecated enum selection ------------------+
                                            |
deprecated backend class construction ------+--> warning --> threaded backend
                                            |
dlparse_v1/v2/v4 selection -----------------+
```

The backend module retains one functional document backend and one functional
page backend:

- `ThreadedDoclingParseDocumentBackend`
- `ThreadedDoclingParsePageBackend`

`DoclingParseDocumentBackend` remains temporarily as a deprecated compatibility
shim. It contains no parsing implementation and delegates entirely to
`ThreadedDoclingParseDocumentBackend`.

`DoclingParsePageBackend` is removed. Threaded page results are represented
only by `ThreadedDoclingParsePageBackend`.

## Compatibility behavior

| User input | Warning | Effective backend |
|---|---|---|
| `ThreadedDoclingParseDocumentBackend` | none | threaded |
| `PdfBackend.THREADED_DOCLING_PARSE` | none | threaded |
| `DoclingParseDocumentBackend` | `DeprecationWarning` | threaded |
| `PdfBackend.DOCLING_PARSE` | `DeprecationWarning` | threaded |
| `PdfBackend.DLPARSE_V1` | `DeprecationWarning` | threaded |
| `PdfBackend.DLPARSE_V2` | `DeprecationWarning` | threaded |
| `PdfBackend.DLPARSE_V4` | `DeprecationWarning` | threaded |
| `DoclingParseV2DocumentBackend` | one deprecation warning | threaded |
| `DoclingParseV4DocumentBackend` | one deprecation warning | threaded |

Warnings must name `ThreadedDoclingParseDocumentBackend` as the replacement
and use a stack level that points to the caller.

The V2/V4 compatibility classes should inherit from the threaded backend
directly, rather than through `DoclingParseDocumentBackend`, to avoid emitting
two warnings.

## Implementation plan

### 1. Pin docling-parse PR #350

Update `pyproject.toml`:

```toml
format-pdf-docling = [
  'docling-parse>=7.19.1,<8.0.0',
]
```

Pin uv resolution to commit `ee9fa4a9f57741653f36fcaa99dc8d258df5d971`,
the current head of PR #350. Replace this temporary source pin with the first
released docling-parse version containing PR #350 before publishing Docling.

Remove pypdfium2 from this extra. Keep it exclusively in
`format-pdf-pypdfium2`. The combined `format-pdf` extra continues to install
both implementations.

Regenerate `uv.lock` and verify that all docling-parse resolution entries use
the new lower bound.

Update `packages/docling-slim/README.md` to state that
`format-pdf-docling` is independent of pypdfium2.

### 2. Forward page ranges to the threaded loader

Pass `in_doc.limits.page_range` directly as the keyword-only `page_range`
argument to `DoclingThreadedPdfParser.load()`. PR #350 validates the inclusive,
1-indexed range and clips its end against the document page count internally.
Remove `_resolve_threaded_page_numbers()` and every external page-count probe.

Keep the explicit `BytesIO.seek(0)` immediately before
`DoclingThreadedPdfParser.load()`. The page-count API reads from the beginning,
but the backend should not depend on its final stream-position behavior.

Update load-error comments so they refer only to docling-parse exceptions and
the page-count sentinel.

### 3. Read outlines from the threaded parser

Use:

```python
annotations = self.parser.get_annotations(self.doc_key)
```

Read `annotations.table_of_contents` when annotations are available. Do not
open a second parser or rewind the input stream for outline extraction.

Refactor `docling/utils/pdf_outline.py` so its docling-parse outline helper
accepts `PdfTableOfContents | None` directly rather than a `PdfDocument`.
Use the native `iterate()` method and preserve the destination page and
top-left-origin vertical coordinate added by PR #344.

Update the outline unit tests to construct table-of-contents nodes directly.
Update integration tests to obtain outlines through
`ThreadedDoclingParseDocumentBackend`; they must not instantiate
`DoclingPdfParser`.

### 4. Remove the hybrid backend

Delete the implementation of:

- `DoclingParsePageBackend`;
- the current `DoclingParseDocumentBackend`;
- `_pdoc`, `_ppage`, `dp_doc`, and their cleanup code;
- synchronous page parsing and rendering;
- the pdfium/docling-parse page-count comparison.

Remove the now-unused imports:

```python
import logging
import pypdfium2 as pdfium
from pypdfium2 import PdfPage
from docling_parse.pdf_parser import DoclingPdfParser, PdfDocument
from docling.backend.managed_pdfium_backend import (
    ManagedPdfiumDocumentBackend,
    ManagedPdfiumPageBackend,
)
from docling.utils.locks import pypdfium2_lock
```

Also remove `_log` if it has no remaining use.

Define `DoclingParseDocumentBackend` after the threaded backend as a thin
compatibility subclass:

```python
class DoclingParseDocumentBackend(ThreadedDoclingParseDocumentBackend):
    """Deprecated alias for ThreadedDoclingParseDocumentBackend."""

    def __init__(...):
        warnings.warn(
            "DoclingParseDocumentBackend is deprecated; use "
            "ThreadedDoclingParseDocumentBackend instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(...)
```

The shim must inherit `supports_random_page_access = False` and must not
override `load_page()` or contain parser logic.

### 5. Normalize deprecated enum selections

Update `PdfBackend` documentation in
`docling/datamodel/pipeline_options.py`:

- mark `DOCLING_PARSE` as deprecated;
- describe `THREADED_DOCLING_PARSE` as the sole docling-parse backend;
- state that all `DLPARSE_*` values map to the threaded value.

Update `normalize_pdf_backend()` so these values map directly to
`THREADED_DOCLING_PARSE`:

```python
{
    PdfBackend.DOCLING_PARSE: PdfBackend.THREADED_DOCLING_PARSE,
    PdfBackend.DLPARSE_V1: PdfBackend.THREADED_DOCLING_PARSE,
    PdfBackend.DLPARSE_V2: PdfBackend.THREADED_DOCLING_PARSE,
    PdfBackend.DLPARSE_V4: PdfBackend.THREADED_DOCLING_PARSE,
}
```

Emit exactly one `DeprecationWarning` for each deprecated selection.

Because CLI routing normalizes before selecting backend options,
`--pdf-backend docling_parse` should resolve to:

- `ThreadedDoclingParseDocumentBackend`;
- `ThreadedDoclingParseBackendOptions`;
- the requested parser thread count and native-memory release interval.

Remove the dedicated non-threaded branch from `docling/cli/main.py`.

### 6. Update legacy classes and callers

Change `DoclingParseV2DocumentBackend` and
`DoclingParseV4DocumentBackend` to deprecated wrappers around
`ThreadedDoclingParseDocumentBackend`. Each direct construction emits one
warning and then follows the threaded lifecycle.

Migrate internal examples, tests, and annotations from
`DoclingParseDocumentBackend` to `ThreadedDoclingParseDocumentBackend`.
In particular:

- update `docs/examples/batch_convert.py`;
- update backend selection tests;
- remove tests specific to synchronous `load_page()` behavior;
- migrate crop, rendering, text-cell, password, OCR, heading, and conversion
  tests to `iter_pages()`;
- remove fake pdfium documents and monkeypatches from docling-parse tests;
- update comments that identify `DoclingParsePageBackend` as an example.

Keep focused tests for the deprecated class itself rather than using it
throughout the behavioral suite.

### 7. Update user-facing documentation

Update `docs/reference/cli.md`:

- retain `docling_parse` as an accepted deprecated value;
- identify `threaded_docling_parse` as its replacement;
- document that deprecated values emit a warning and use the threaded parser.

Update relevant package-extra documentation to distinguish:

- `format-pdf-docling`: native docling-parse parsing and rendering;
- `format-pdf-pypdfium2`: explicit pdfium backend;
- `format-pdf`: installs both backend choices.

## Test plan

### Page-range forwarding

Add focused tests covering bounded, open-ended, and default page ranges. Assert
that the exact range reaches `DoclingThreadedPdfParser.load()` and that stream
position is zero when loading begins. Range validation and clipping remain
covered in docling-parse PR #350.

### Deprecation and mapping

Verify:

- direct `DoclingParseDocumentBackend` construction emits one
  `DeprecationWarning`;
- its parser is a `DoclingThreadedPdfParser`;
- its page access contract is streaming;
- `PdfBackend.DOCLING_PARSE` normalizes to
  `THREADED_DOCLING_PARSE`;
- every `DLPARSE_*` value normalizes to the threaded value;
- CLI `--pdf-backend docling_parse` chooses the threaded backend and threaded
  options;
- V2/V4 wrappers emit exactly one warning.

### Outline extraction

Verify:

- no annotations returns an empty outline;
- annotations without a table of contents return an empty outline;
- nested table-of-contents entries preserve document order and depth;
- the threaded backend calls `get_annotations(self.doc_key)`;
- outline access does not construct another parser or consume page results.

### Dependency isolation

Add an import/operation test with `pypdfium2` unavailable that:

- imports `docling.backend.docling_parse_backend`;
- constructs the threaded backend;
- resolves an explicit page range;
- parses and renders a page;
- unloads successfully.

This is the principal regression test proving that the docling-parse extra no
longer relies on pdfium.

## Verification

Run focused tests first:

```bash
uv run pytest \
  tests/test_backend_docling_parse.py \
  tests/test_backend_docling_parse_legacy.py \
  tests/test_pdf_outline.py \
  tests/test_pdf_password.py \
  tests/test_cli.py \
  tests/test_options.py
```

Run affected pipeline and OCR tests:

```bash
uv run pytest \
  tests/test_threaded_pipeline.py \
  tests/test_native_pdf_pipeline.py \
  tests/test_heading_hierarchy_pdf.py \
  tests/test_heading_hierarchy_bookmarks.py \
  tests/test_ocr_rects_invisible_text.py \
  tests/test_e2e_ocr_conversion.py
```

Check that the old parser and pdfium are absent from the backend:

```bash
rg 'DoclingPdfParser|pypdfium2|pdfium|ManagedPdfium' \
  docling/backend/docling_parse_backend.py
```

The command must return no matches.

Check that production code no longer uses the synchronous parser:

```bash
rg 'DoclingPdfParser' docling
```

The command must return no matches.

Finally run:

```bash
make validate
make check
```

If validation modifies files, review the changes and rerun it until clean.

## Acceptance criteria

- `DoclingThreadedPdfParser` is the only docling-parse parser used in
  production code.
- `docling_parse_backend.py` has no pdfium import, handle, operation, or lock.
- Explicit page ranges are forwarded directly to the PR #350 threaded loader.
- Threaded annotations provide PDF outlines without a second document load.
- The docling-parse package extra does not install pypdfium2.
- Current threaded selections produce no deprecation warning.
- Every former docling-parse selection emits one warning and runs through the
  threaded backend.
- The compatibility shim does not retain synchronous/random-access behavior.
- Targeted tests, `make validate`, and `make check` pass.

## Risks and mitigations

### Random-access callers

Direct users of `DoclingParseDocumentBackend.load_page()` will move to a
streaming contract. The warning must state the replacement clearly, and the
migration must be called out in release notes.

### Warning duplication

Enum normalization and compatibility subclasses are separate entry points.
CLI selection must normalize directly to the threaded class, while direct
class construction warns in the shim. V2/V4 wrappers must bypass the shim.

### Stream state

Rewind `BytesIO` immediately before threaded loading so document-key hashing
always covers the full input.

### Optional dependency regression

Removing pypdfium2 from package metadata can reveal hidden imports not visible
in the main backend. The dependency-isolation test must exercise actual
page-range resolution and rendering, not only module import.

## Follow-up removal

In a later breaking release:

- remove `DoclingParseDocumentBackend`;
- remove `PdfBackend.DOCLING_PARSE`;
- remove `DLPARSE_V1`, `DLPARSE_V2`, and `DLPARSE_V4`;
- remove the V2/V4 compatibility modules;
- remove the corresponding CLI choices and warning tests.
