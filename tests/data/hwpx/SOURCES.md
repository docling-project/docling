# HWPX test fixture sources

These sample documents exercise the native HWPX backend end-to-end (ZIP +
OWPML XML parsing into a `DoclingDocument`). Each covers a distinct v1 element
class so the backend's structural mapping is tested against real documents.

## Provenance & license

Except for the one synthetic fixture noted below, every file is copied verbatim
from the **rhwp** parser project's sample corpus:

- Repository: <https://github.com/edwardkim/rhwp>
- Commit pinned: `bc38ff55a7e8acb65aebebe237dca0542480d381`
- License: **MIT** (Copyright (c) 2025-2026 Edward Kim). The MIT license permits
  redistribution and modification, including bundling these sample documents in
  a downstream test suite.

## Inventory (by element class)

| File | Upstream path | Elements exercised |
|------|---------------|--------------------|
| `para-001.hwpx` | `samples/hwpx/para-001.hwpx` | paragraphs, inline runs, mixed Korean/Hanja text |
| `table-text.hwpx` | `samples/hwpx/table-text.hwpx` | a 3×8 table with two `colSpan="4"` merged header cells and text cells |
| `footnote-01.hwpx` | `samples/hwpx/footnote-01.hwpx` | footnotes, plus numbered/bulleted outline paragraphs (`<hh:heading type="BULLET"/NUMBER">`) → list items |
| `eq-002.hwpx` | `samples/hwpx/eq-002.hwpx` | `<hp:equation>` with a Hancom equation script → formula items |
| `test-image.hwpx` | `samples/test-image.hwpx` | an embedded raster picture (`BinData/image1.bmp`) referenced from `<hp:pic>` |
| `headings-synth.hwpx` | *synthetic* — see below | outline headings (`<hh:heading type="OUTLINE"/>`) at levels 1–3 |

## Synthetic fixture: `headings-synth.hwpx`

No document in the upstream corpus applies a genuine outline **heading**
(`<hh:heading type="OUTLINE"/>`) to body text — the outline styles that appear
in the corpus are configured as bulleted/numbered lists (see `footnote-01.hwpx`,
whose "개요"/Outline paragraphs carry `type="BULLET"`/`"NUMBER"`, not
`"OUTLINE"`). To cover the heading path honestly, this fixture is
**synthesised**, not taken from the corpus — the same approach the upstream
corpus itself uses for the coverage gaps it cannot fill from real documents:

- **Base**: `samples/hwpx/blank_hwpx.hwpx` from the same rhwp corpus (MIT, commit
  `bc38ff55`) — a minimal, valid single-section HWPX whose default style table
  already defines the outline paragraph shapes `개요 1`/`개요 2`/`개요 3` as
  `<hh:heading type="OUTLINE" level="0"/1"/2"/>`.
- **Edit**: four paragraphs were appended to `Contents/section0.xml` — three
  referencing the outline `paraPr` shapes (levels 0–2, i.e. heading levels 1–3)
  and one ordinary body paragraph — each carrying a short run of Korean text.
  The container was re-zipped with `mimetype` stored first (uncompressed), as
  the HWPX/OPC packaging requires.

The edit only adds heading paragraphs; it is the smallest change that makes a
valid HWPX emit outline headings. If a corpus document that genuinely uses
outline headings is found later, this synthetic fixture can be replaced.
