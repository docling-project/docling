Docling can parse various documents formats into a unified representation (Docling
Document), which it can export to different formats too — check out
[Architecture](../concepts/architecture.md) for more details.

Below you can find a listing of all supported input and output formats.

## Supported input formats

| Format | Description |
|--------|-------------|
| PDF | |
| DOCX, XLSX, PPTX | Default formats in MS Office 2007+, based on Office Open XML |
| DOC, XLS, PPT | Legacy binary Office formats (97–2004); requires LibreOffice |
| RTF | Rich Text Format; requires LibreOffice |
| ODT, ODS, ODP | OpenDocument Format for text documents, spreadsheets, and presentations |
| EPUB | Electronic Publication format for e-books |
| Pages | Apple Pages documents (`.pages`), both container generations (Pages 5+ and iWork '09); requires the `format-iwork` extra |
| Numbers | Apple Numbers spreadsheets (`.numbers`), both container generations (Numbers 3+ and iWork '09); requires the `format-iwork` extra |
| Keynote | Apple Keynote presentations (`.key`), every container generation (Keynote 6+, the flattened package Keynote 2018+ writes, and iWork '09); requires the `format-iwork` extra |
| Markdown | |
| MediaWiki wikitext | `.wiki`, `.wikitext`, and gzip-compressed copies |
| AsciiDoc | Human-readable, plain-text markup language for structured technical content |
| LaTeX | Scientific document preparation system |
| HTML, XHTML | |
| MHTML, MHT | MIME HTML archives |
| CSV | |
| PNG, JPEG, TIFF, BMP, WEBP | Image formats |
| WAV, MP3, M4A, AAC, OGG, FLAC | Audio formats (requires `audio` or `asr` extra — see [Processing audio and video](processing_audio_media.md)) |
| MP4, AVI, MOV, MKV, WEBM | Video formats — audio track is extracted and transcribed (requires `audio` or `asr` extra, and `ffmpeg`) |
| WebVTT | Web Video Text Tracks format for displaying timed text |
| BoxNote | Box Notes collaborative note format |
| Email | MIME (`.eml`) and Outlook (`.msg`) email messages; attachment names can optionally be listed via `EmailBackendOptions` |
| AFP | IBM Advanced Function Presentation / MO:DCA |
| iCalendar | Calendar files |

For Markdown, CSV, and AsciiDoc, a byte-order mark (BOM) selects the text encoding
unless the backend's `encoding` option is set. If the bytes are not valid in the
BOM-detected encoding, conversion fails with an error that names the encoding
and retains the decoder error as its cause.

In HTML, a `<span>` with a non-empty `data-math-tex` attribute produces a formula
item from the attribute value. The span can contain a visible preview.

In Markdown prose, inline HTML `<br>` tags become hard line breaks in paragraphs,
headings, and list items. Code spans, code blocks, and escaped tags stay literal.
The document representation keeps heading line breaks; Markdown export normalizes
them to spaces in plain headings. Headings with formatted inline content follow
the existing Markdown hard-break serialization. Inline HTML inside Markdown table
cells follows separate handling.

PowerPoint equations stored as Office Math (OMML) in text shapes are converted to
LaTeX formula items. Equations mixed with text or list content retain their order.

Video and audio on PPTX and ODP slides become tracks, with their poster frame or
icon as the cover. The `video` or `audio` of the track gives the path of the media
file inside the package, or the link of a linked file. Set `keep_media=True` in the
pipeline options to hold the media file in the track, and `do_media_conversion=True`
to also transcribe it (`--keep-media` and `--convert-media` on the CLI)
(see [Video and audio in slides](processing_audio_media.md#video-and-audio-in-slides)).

Email conversion prefers non-empty plain-text bodies. When it uses an HTML body,
text in `<footer>` elements is retained in the furniture content layer. Default
Markdown exports omit this layer; include `ContentLayer.FURNITURE` in
`included_content_layers` to export it.

Email conversion also reads the calendar parts of a message, such as meeting
invitations, cancellations and attached `.ics` files. Their events are added under
a "Calendar" heading after the body, with the same fields as iCalendar input. An
event that the message carries both inline and as an attachment is added once. An
event description that repeats the body text, as Outlook writes it, is left out. Set
`EmailBackendOptions.parse_calendar_invitations` to `False` to leave them out.

For LaTeX input, front matter such as authors, affiliations, institutes, email
addresses, and dates is stored as text items. Inline formulas remain formula
children. The `metadata`, `authors`, and `affiliations` groups record source roles.
`\thanks` and `\altaffiliation` use footnote items in these groups. Retained
source commands use the furniture layer and do not appear in default Markdown.

Schema-specific support:

| Format | Description |
|--------|-------------|
| DocLang XML | XML format following [DocLang](https://doclang.ai); supported extensions: `.dclg`, `.dclg.xml`, and generic `.xml` with a `<doclang>` root element |
| DocLang archive | Zipped DocLang bundle including page images; supported extension: `.dclx` |
| USPTO XML | XML format followed by [USPTO](https://www.uspto.gov/patents) patents |
| JATS XML | XML format followed by [JATS](https://jats.nlm.nih.gov/) articles |
| XBRL XML | XML format for business and financial reporting following [XBRL](https://www.xbrl.org/) standard; also inline XBRL (iXBRL) reports in `.htm`, `.html` and `.xhtml` files; needs the `xbrl` extra (iXBRL reports convert as HTML without it) |
| MediaWiki XML | MediaWiki page exports (`.xml`, `.xml.gz`) |
| EBCDIC | Mainframe fixed-width data files; needs the COBOL record layout passed through `EbcdicBackendOptions`; supported extensions: `.ebc`, `.ebcdic` |
| Docling JSON | JSON-serialized [Docling Document](../concepts/docling_document.md); images referencing local files are ignored unless enabled, see [Docling JSON input](./advanced_options.md#docling-json-input) |

### MediaWiki

Use `InputFormat.MEDIAWIKI` in the SDK or `--from mediawiki` in the CLI. Each
MediaWiki XML export or wikitext file produces one Docling document. Both can use
gzip compression. For a multi-page XML export, Docling imports the latest
revision of each page in export order. The SDK also accepts a generic ZIP as a
batch and converts each supported member to its own document.

The parser runs locally and preserves headings, emphasis, links, lists, tables,
formulas, code languages, and file captions. Pictures retain their captions
without loading the image. The parser extracts named fields from common infobox
and taxobox templates. It approximates selected templates for dates, lists,
units, language text, and citations. It retains text from presentation templates
such as `{{Small|…}}` and `{{Mono|…}}`. It leaves out other templates and all
Lua modules. The `format-html` extra enables MediaWiki conversion in Docling Slim.

The parser uses these size limits by default. You can set them with
`MediaWikiBackendOptions`.

| Option | Default | Limit |
|--------|---------|-------|
| `max_total_bytes` | 512 MiB | Input size after any decompression |
| `max_file_bytes` | 16 MiB | Standalone wikitext size after any decompression |
| `max_page_chars` | 4,194,304 | Characters in a page revision or standalone wikitext file |
| `max_inline_reference_bytes` | 16 MiB | Total rendered inline citation bytes per input, including repeated citations |

The XML reader limits each retained metadata field to 8,192 characters and
discards other XML content as it reads. The wikitext parser keeps list markers
beyond level 32 as text. If parsing exceeds the work limit, Docling logs a
warning and keeps that page as plain text.

By default, MediaWiki conversion omits `<ref>` citations and drops empty
sections. Regular superscripts remain. Set `preserve_bold=False` or
`preserve_italic=False` to remove that style and keep its text. This example
keeps italic formatting, removes bold formatting, and lists citations.

```python
from docling.datamodel.backend_options import MediaWikiBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, MediaWikiFormatOption

converter = DocumentConverter(format_options={
    InputFormat.MEDIAWIKI: MediaWikiFormatOption(
        backend_options=MediaWikiBackendOptions(
            reference_mode="list", preserve_bold=False
        )
    )
})
result = converter.convert("article.wiki")
```

`reference_mode="inline"` places citation text in superscript at each use.
`reference_mode="list"` uses numbered superscripts and places the referenced
entries at `<references />` or `{{reflist}}`, or appends a References section
when there is no matching list. Named references are reused, including forward
definitions. Unresolved names are omitted. Groups have separate numbering.
`reference_mode="omit"` is the default. This is an approximation of MediaWiki
citation rendering and does not execute templates or Lua.

## Supported output formats

| Format | Description |
|--------|-------------|
| HTML | Both image embedding and referencing are supported |
| Markdown | |
| JSON | Lossless serialization of Docling Document |
| DocLang XML | XML serialization following [DocLang](https://doclang.ai); CLI output format: `doclang` |
| Text | Plain text, i.e. without Markdown markers |
| [Doctags](https://arxiv.org/pdf/2503.11576) | Markup format for efficiently representing the full content and layout characteristics of a document |
| WebVTT | Web Video Text Tracks format for displaying timed text |
| DocLang archive | Zipped DocLang bundle including page images; CLI output format: `dclx` |
| Chunks (JSONL) | Chunked document output for RAG pipelines; configurable via `--chunks-type`, `--chunks-max-tokens`, `--chunks-tokenizer` |
| LaTeX | Standalone `.tex` document; images are emitted as placeholders |
