When Docling reads a PDF, the layout model tells it *this block is a section header* — but not how
deep that header sits in the document. Every heading therefore arrives at level 1, so a report
whose real structure runs `PART I` → `1. Definitions` → `1.1 Interpretation` → `(a)` comes out as
a flat stack of `#` headings. Everything downstream that leans on the hierarchy loses out too:
Markdown and DocTags exports, hierarchical chunking, and any table of contents you build yourself.

Docling can put those levels back. The heading-hierarchy stage runs right after reading order and
rewrites `SectionHeaderItem.level` from three signals it reads out of the document itself — no
extra model to run, nothing to download. It is **disabled by default**, because a wrong level is
worse than a missing one for pipelines already tuned around flat headings.

## Enable it

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions()
pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
# The font-style signal reads the parsed PDF cells, which are dropped unless you keep them:
pipeline_options.generate_parsed_pages = True

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)
doc = converter.convert("report.pdf").document
print(doc.export_to_markdown())
```

On a paper with numbered sections, that turns a flat run of headings into (abridged):

```text
L1  TableFormer: Table Structure Understanding with Transformers.
  L2  Abstract
L1  1. Introduction
L1  4. The TableFormer model
  L2  4.1. Model architecture.
L1  5. Experimental Results
  L2  5.1. Implementation Details
```

Levels show up wherever the hierarchy is expressed: `#`/`##`/`###` in Markdown, the
`section_header_level_N` tags in DocTags, and the heading path that the hierarchical chunker
attaches to every chunk. A runnable version of the above lives in
[the heading levels example](../examples/heading_levels.py).

## Where the levels come from

Three signals are consulted, and the first one that has something to say about a given heading
wins. Precedence is applied per heading, not per document, so a half-finished outline or a
document that numbers only its top sections still gets the best available answer everywhere else.

| Order | Signal | Option | Reads |
| ----- | ------ | ------ | ----- |
| 1 | Bookmarks | `use_bookmarks` | the PDF outline / table of contents |
| 2 | Numbering | `use_numbering` | the marker at the start of the heading text |
| 3 | Visual style | `use_style` | font size, weight, slant and letter case of the heading |

### Bookmarks

If the PDF carries an outline, that outline *is* the author's own declared hierarchy, so it is
treated as authoritative. Docling matches each bookmark to a detected heading by title and page,
comparing them with and without their numbering prefix — a bookmark reading "Definitions" still
finds the heading "1.1 Definitions", and a truncated bookmark still finds its full heading.
`bookmark_match_threshold` sets how close the titles have to be before Docling believes the match.

Two things are worth knowing about this pass:

- Layout models sometimes classify a heading as a list item. When such an item matches a bookmark
  confidently, it is **promoted** to a section header in place, keeping its text and position.
  This is the only structural change the stage ever makes.
- Bookmarks that match nothing are simply dropped. A stale or partial outline can only add
  correct levels, never take away the ones numbering and style would have found.

!!! note "Which backends supply bookmarks"

    The pypdfium2 backend returns the richest outline: title, depth, target page and vertical
    position. The docling-parse backends read their own native table of contents, which carries
    titles and hierarchy but no page numbers — matching then falls back to titles alone, with a
    stricter similarity threshold to compensate. Backends with no embedded outline, image inputs
    among them, report nothing and the stage moves on to numbering.

### Numbering

For everything the outline does not cover, the leading marker of the heading text is the most
reliable signal — on legal and regulatory documents far more reliable than styling, which tends to
be uniform throughout. Docling recognizes keyword markers (`PART`, `TITLE`, `BOOK`, `CHAPTER`,
`ARTICLE`, `SECTION`, `CLAUSE`, `SCHEDULE`, `ANNEX`, `APPENDIX`, `RULE`, `§`), Roman and Arabic
numerals, dotted decimals and parenthesized letters, and ranks them in this default order:

```text
part  →  chapter  →  article  →  roman_u  →  arabic  →  alpha_u  →  alpha_l  →  roman_l
PART I   CHAPTER 1   ARTICLE 1     I.          1.          A.          (a)         (i)
```

Dotted decimals share the `arabic` rank and sort by their depth, so `1.1` lands one level below
`1.` and `1.1.1` one below that. If your documents follow a different convention, reorder the
scheme names with `numbering_schemes` (highest level first).

A lone `I.` is genuinely ambiguous — Roman one, or the ninth letter? Docling resolves it from the
rest of the document: if unambiguous Roman markers (`II.`, `III.`) appear elsewhere it reads as
Roman, if unambiguous letters (`B.`, `F.`) appear it reads as alpha, and with no evidence either
way `I`/`i` default to Roman, which is the common legal case.

### Visual style

Headings with neither a bookmark match nor a recognizable marker fall back to how they look on the
page. They are ranked by font size first and then — when `use_font_style` is on — by weight, slant
and letter case, so headings that share a size are still separated: bold above regular, upright
above italic, all caps above mixed case.

Font size deserves a word of explanation. Docling measures a heading's size as the median height
of the text cells under it, which is the height of the glyphs actually on that line rather than
the declared point size. "Securing and protecting" therefore measures a couple of points taller
than "Contents" in the very same font, purely because of the descenders. Treating every distinct
height as its own level would invent levels out of that noise, so sizes within
`style_size_tolerance` (5% by default) are merged into one. That merging is also what gives weight
and slant something to do: with every heading alone in its own size bucket, a tie-breaker never
runs.

Weight and slant are read from the embedded PDF font names — `Helvetica-Bold`,
`NKDKGK+HelveticaNeueLTPro-Bd`, `Times-Italic`. Nothing standardizes how style is encoded in that
string, only foundry convention, so the parser is deliberately conservative: it matches style
words as whole tokens and honors abbreviations only when they form a complete part of the name.
`Avenir-Book` is a regular weight while the family `Bookman` is not, and the `LT` in
`HelveticaNeueLTPro` is a foundry tag rather than "light". A name it cannot read leaves the
heading at regular weight, and the ranking quietly degrades to font size alone.

!!! note "Scanned pages and OCR"

    OCR produces no font metadata, so weight and slant are unavailable on scanned documents and
    the style signal ranks by size only. Bookmarks and numbering are unaffected.

## Levels are relative to the document

Docling does not assign absolute depths. It collects the distinct signals actually present and
compresses them into contiguous levels, so a document that starts at `1.` starts at level 1
instead of being pushed down by a `PART` that never appears:

| Headings | Levels |
| -------- | ------ |
| `I. Introduction`, `1. Background`, `2. Motivation`, `II. Methods` | 1, 2, 2, 1 |
| `1. A`, `1.1 B`, `1.1.1 C` | 1, 2, 3 |
| `PART I`, `1.`, `1.1`, `(a)`, `(b)`, `(i)`, `(ii)` | 1, 2, 3, 4, 4, 5, 5 |

The third row is also where the ambiguity rule shows up: it is the `(ii)` that makes `(i)` Roman.
Drop it and the lone `(i)` reads as the letter *i*, landing at the same level as `(a)`.

The same compression applies to style: if every heading in a document is bold, weight adds no
levels at all. Levels deeper than `max_level` (6 by default) are clamped.

## What the stage changes

It rewrites `SectionHeaderItem.level`, and — only through a confident bookmark match — promotes a
mis-classified list item to a section header. It never adds, removes or reorders anything else,
and a heading for which no signal applies keeps the level it already had. Enable the stage on a
document where nothing is recognizable and the output is exactly what you had before.

## Options

All of these live on `HeadingHierarchyOptions`, set as
`PdfPipelineOptions.heading_hierarchy_options`.

| Option | Default | Description |
| ------ | ------- | ----------- |
| `enabled` | `False` | Master switch for the stage. |
| `use_bookmarks` | `True` | Use the PDF outline as the authoritative signal. |
| `use_numbering` | `True` | Use the leading numbering marker of the heading text. |
| `use_style` | `True` | Fall back to the heading's visual style. Requires `generate_parsed_pages=True`. |
| `use_font_style` | `True` | Refine the style fallback with font weight, slant and all-caps detection. Ignored when `use_style` is off. |
| `style_size_tolerance` | `0.05` | Relative difference below which two font sizes count as one. Higher merges more sizes into a single level. |
| `numbering_schemes` | `None` | Override the scheme precedence, highest level first. |
| `max_level` | `6` | Deepest level assigned; anything deeper is clamped. |
| `bookmark_match_threshold` | `0.8` | Minimum title similarity (0–1) for a bookmark to claim a heading. Higher is stricter. |

Signals can be switched off individually, which is worth doing when you know what your corpus
looks like. Legal filings with immaculate numbering and erratic typography do better with
`use_style=False`; a design report with no numbering at all leans entirely on style.

```python
pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(
    enabled=True,
    use_style=False,  # bookmarks and numbering only
    max_level=4,
)
```

!!! warning "Keep the parsed pages for the style signal"

    The style fallback reads the parsed PDF cells, and the pipeline discards those as soon as a
    page is finished unless `generate_parsed_pages=True`. Without them, style inference is skipped
    silently — no error, just fewer levels. Bookmarks and numbering do not need this option.

## Through the API server

[docling-serve](./api_server/index.md) exposes the same feature as `do_pdf_heading_hierarchy`,
with the fine-tuning under `pdf_heading_hierarchy_options`. The nested `enabled` flag is set for
you from `do_pdf_heading_hierarchy`, so you only send the options you actually want to change:

```json
{
  "do_pdf_heading_hierarchy": true,
  "pdf_heading_hierarchy_options": {
    "use_bookmarks": false,
    "max_level": 4
  }
}
```

## Applying it to a document you already have

The inference itself does not need a pipeline. `HeadingHierarchyModel.assign_heading_levels()`
works on a plain `DoclingDocument`, which is handy for re-levelling a document you converted
earlier, or for trying out scheme orders without re-running layout:

```python
from docling.datamodel.pipeline_options import HeadingHierarchyOptions
from docling.models.stages.heading_hierarchy.heading_hierarchy_model import (
    HeadingHierarchyModel,
)

model = HeadingHierarchyModel(options=HeadingHierarchyOptions(use_style=False))
model.assign_heading_levels(doc)  # modifies doc in place
```

With no parsed pages and no outline to hand, only numbering can apply, so `use_style=False` simply
says so explicitly. Pass `parsed_pages=` and `outline=` if you have them.
