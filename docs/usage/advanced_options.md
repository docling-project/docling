## Model prefetching and offline usage

By default, models are downloaded automatically upon first usage. If you would prefer
to explicitly prefetch them for offline use (e.g. in air-gapped environments) you can do
that as follows:

**Step 1: Prefetch the models**

Use the `docling-tools models download` utility:

```sh
$ docling-tools models download
Downloading layout model...
Downloading tableformer model...
Downloading picture classifier model...
Downloading code formula model...
Downloading rapidocr torch ch models...
Downloading rapidocr onnxruntime ch models...
Models downloaded into $HOME/.cache/docling/models.
```

To prefetch EasyOCR recognition models for specific languages, repeat
`--easyocr-lang` with the same values used by `EasyOcrOptions.lang` -- EasyOCR's own codes, or
BCP-47 tags behind the `iso:` prefix:

```sh
$ docling-tools models download easyocr --easyocr-lang iso:zh-Hans --easyocr-lang ja
```

To prefetch an OpenAI-format Whisper checkpoint for native ASR, select its model
name. The file is stored directly in the output directory, so the directory can
be copied to an artifact store and used with `--artifacts-path`:

```sh
docling-tools models download whisper_tiny --output-dir /models
docling --asr-model whisper_tiny_native --artifacts-path /models audio.wav
```

The native Distil-Whisper presets work the same way. Their OpenAI-format
checkpoints come from Hugging Face and are stored in the output directory in the
Hugging Face cache layout:

```sh
docling-tools models download whisper_distil_small_en --output-dir /models
docling --asr-model whisper_distil_small_en_native --artifacts-path /models audio.wav
```

The WhisperS2T presets use their `--asr-model` names. Each one downloads its
CTranslate2 model in `float16` (for CUDA) and `float32` (for CPU), plus the
small word-alignment model that word timestamps use if the preset needs one,
into the output directory in the Hugging Face cache layout:

```sh
docling-tools models download whisper_small_en_s2t --output-dir /models
docling --asr-model whisper_small_en_s2t --artifacts-path /models audio.wav
```

`docling-tools models download --all` does not include ASR models. Add `--asr`
to prefetch all native Whisper model sizes, English-only variants, and
Distil-Whisper presets:

```sh
docling-tools models download --all --asr --output-dir /models
```

This requires the `asr` extra and downloads OpenAI-format checkpoints. It does
not prefetch Transformers-format, MLX, or WhisperS2T models. Prefetch a
WhisperS2T preset by its name, as shown above.

Alternatively, models can be programmatically downloaded using `docling.utils.model_downloader.download_models()`.

Also, you can use `download-hf-repo` parameter to download arbitrary models from HuggingFace by specifying repo id:

```sh
$ docling-tools models download-hf-repo ds4sd/SmolDocling-256M-preview
Downloading ds4sd/SmolDocling-256M-preview model from HuggingFace...
```

**Step 2: Use the prefetched models**

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

artifacts_path = "/local/path/to/models"

pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

Or using the CLI:

```sh
docling --artifacts-path="/local/path/to/models" FILE
```

Or using the `DOCLING_ARTIFACTS_PATH` environment variable:

```sh
export DOCLING_ARTIFACTS_PATH="/local/path/to/models"
python my_docling_script.py
```

## Using remote services

The main purpose of Docling is to run local models which are not sharing any user data with remote services.
Anyhow, there are valid use cases for processing part of the pipeline using remote services, for example invoking OCR engines from cloud vendors or the usage of hosted LLMs.

In Docling we decided to allow such models, but we require the user to explicitly opt in to communicating with external services.

```py
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions(enable_remote_services=True)
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

When the value `enable_remote_services=True` is not set, the system will raise an exception `OperationNotAllowed()`.

_Note: This option is only related to the system sending user data to remote services. Control of pulling data (e.g. model weights) follows the logic described in [Model prefetching and offline usage](#model-prefetching-and-offline-usage)._

### List of remote model services

The options in this list require the explicit `enable_remote_services=True` when processing the documents.

- `PictureDescriptionApiOptions`: Using vision models via API calls.
- `KserveV2OcrOptions`: OCR on a KServe v2 inference server (e.g. Triton).
- `ApiKserveV2ObjectDetectionEngineOptions`: Object-detection layout models served by a KServe v2 inference server, set as the layout `engine_options`.
- `ApiKserveV2ImageClassificationEngineOptions`: Picture classification served by a KServe v2 inference server, set as the classifier `engine_options`.
- `ApiVlmEngineOptions`: VLM stages (VLM conversion, code/formula enrichment, picture description) calling an OpenAI-compatible API, set as the stage `engine_options`.
- `ApiVlmOptions`: VLM pipeline models calling an OpenAI-compatible API.


## Adjust pipeline features

The example file [custom_convert.py](../examples/custom_convert.py) contains multiple ways
one can adjust the conversion pipeline and features.

### Extract native PDF form fields

Set `PdfPipelineOptions(extract_form_fields=True)` (CLI: `--extract-form-fields`)
to convert native PDF widgets (AcroForm) into format-neutral `FieldItem` objects
with fillable `FieldValueItem` children. Each field key comes from printed text
associated with its widget by page geometry. The key can be text inside or
beside the widget's cell, or a paragraph that contains the widget. Checkbox and
radio option text becomes an option key. A shared prompt introduces several
options and stays as ordinary text. Additional context becomes a field hint.
A widget inside a detected table goes into its table cell instead (a rich table
cell holding the field). Its key comes only from text printed in that same
cell; the table's row and column headers already carry the association.
Text used as a field key leaves the body text; a section header or formula
used as a key stays inside the key, as a field heading or a formula. PDF field
names are not used as keys.

Fields keep the native order of their widgets. Page geometry places them
among the text printed around them: the reading-order model orders the text
only, never the widgets. Fields outside a detected form region go in field
regions of their own between the page's other elements, one region per run
of consecutive fields.

Here, **key** means text that names one or more field values, and **hint** means
extra context. **Layout label** means an element class (`DocItemLabel`).
**Caption** means a document caption associated with a picture, table, code
block, or form; it does not mean a field key or hint.

The docling-parse backend supplies the widgets. Scanned or flattened forms,
backends without page widgets and full-page OCR produce no field items; table
cells need `do_table_structure=True`. Raw widget metadata is not stored in the
`DoclingDocument`; keep parsed pages with `generate_parsed_pages=True` when
that PDF-specific data is needed. Field items are not supported by the DocTags
export.

### Image resolution and scale

Page coordinates use 72 points per inch. For image inputs, embedded DPI metadata
determines the physical page size; missing DPI, `(1, 1)` DPI and `(0, 0)` DPI are treated as 72 DPI.
Rendering at scale `n` produces `n` pixels per document point.

### Control PDF table extraction options

You can control if table structure recognition should map the recognized structure back to PDF cells (default) or use text cells from the structure prediction itself.
This can improve output quality if you find that multiple columns in extracted tables are erroneously merged into one.


```python
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options = PdfPipelineOptions(do_table_structure=True)
pipeline_options.table_structure_options.do_cell_matching = False  # uses text cells predicted from table structure model

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

Since docling 1.16.0: You can control which TableFormer mode you want to use. Choose between `TableFormerMode.FAST` (faster but less accurate) and `TableFormerMode.ACCURATE` (default) to receive better quality with difficult table structures.

```python
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

pipeline_options = PdfPipelineOptions(do_table_structure=True)
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # use more accurate TableFormer model

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```


### Use visible PDF rules for reading order

For PDFs whose columns or horizontal bands are separated by visible rules, the
rule-based reading-order stage can use those rules as additional structural
signals. This is enabled by default. Disable it when needed:

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions(use_reading_order_separators=False)
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

The option uses visible vector geometry exposed by the PDF backend. Separator
geometry affects ordering only and is not added to the resulting document.

The same option is available from the CLI. Use `--no-reading-order-separators`
to disable it. `--output-file` selects an exact destination when converting one
input to one output format:

```bash
uv run docling convert --from pdf --to dclx \
  --output-file ./Elsevier-with-separators.dclx \
  ./Elsevier.pdf

uv run docling convert --from pdf --to dclx \
  --no-reading-order-separators \
  --output-file ./Elsevier-without-separators.dclx \
  ./Elsevier.pdf
```


### Control PDF parser threads

The PDF parser and the model pipeline use separate thread settings.
`PdfPipelineOptions.accelerator_options.num_threads` controls CPU threads for
model inference, while `ThreadedDoclingParseBackendOptions.parser_threads`
controls the docling-parse backend.

For the standard PDF pipeline, configure parser threads on `PdfFormatOption`:

```python
from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions()

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend_options=ThreadedDoclingParseBackendOptions(parser_threads=8),
        )
    }
)
```

The equivalent CLI option is independent of `--num-threads`:

```sh
docling --pipeline native --from pdf --parser-threads 8 FILE
```


### Extract the native content of a PDF

`NativePdfPipeline` uses docling-parse alone: one text item per native text cell
and one picture per embedded bitmap, without layout, OCR or table models.

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import NativePdfPipelineOptions
from docling.document_converter import DocumentConverter, NativePdfFormatOption

pipeline_options = NativePdfPipelineOptions()
pipeline_options.generate_page_images = True
pipeline_options.images_scale = 2.0

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: NativePdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

Set `generate_page_images=False` to skip rendering. `parser_threads` configures
docling-parse independently of model-inference `accelerator_options.num_threads`.

```sh
docling --pipeline native --from pdf FILE
docling --pipeline native --from pdf --parser-threads 8 FILE
```


### Recover PDF heading levels

The layout model marks section headers but not how deep they sit, so by default every heading in a
PDF comes out at level 1. Docling can infer the levels from the PDF bookmarks, from outline
numbering and from the heading's font styling:

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions()
pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

See [PDF heading levels](./heading_levels.md) for the signals, their precedence and all options.

### PowerPoint and Excel charts

A chart on a PowerPoint slide or an Excel sheet becomes a picture classified by
its kind, with the data it plots in the picture's `meta.tabular_chart` and its
title as the caption. The first column of the data holds the categories, under the title of
the category axis, and each series gets a column. Dates on the category axis
are written as ISO 8601 dates, such as `2012-01-01`.

A combination chart, such as columns with a line on a secondary axis, gets one
classification prediction per chart type. The back-most plot comes first, so
it stays the main prediction, and the series of every plot are in the data:

```python
from docling.document_converter import DocumentConverter

deck = DocumentConverter().convert("deck.pptx").document
for picture in deck.pictures:
    if picture.meta is not None and picture.meta.classification is not None:
        kinds = [p.class_name for p in picture.meta.classification.predictions]
        print(kinds)  # for example ['bar_chart', 'line_chart']
```

The picture itself is empty unless you set `render_chart_images=True` in
`MsPowerpointBackendOptions` or `MsExcelBackendOptions`. That draws each chart
with LibreOffice.

### Apple iWork options

Pages (`.pages`), Numbers (`.numbers`) and Keynote (`.key`) share their
options, since they share their container.

In a Pages document, headers, footers and footnotes go into the `furniture`
content layer and comments into `notes`. In a Keynote presentation, each slide
becomes a chapter group holding what is on it, and the presenter notes and
comments of that slide go into `notes` under it. In a Numbers spreadsheet, each
sheet becomes a page and a sheet group, and the sticky notes on it go into
`notes`. Either way those layers stay out of the reading order by default; to
include them in an export, pass the extra layers explicitly (this applies to any
`DoclingDocument`, not just these):

```python
from docling_core.types.doc import ContentLayer
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Pages: headers, footers and footnotes are furniture, comments are notes.
report = converter.convert("report.pages").document
print(report.export_to_markdown(
    included_content_layers={
        ContentLayer.BODY,
        ContentLayer.FURNITURE,
        ContentLayer.NOTES,
    }
))

# Keynote: the presenter notes and comments of each slide are notes.
deck = converter.convert("deck.key").document
print(deck.export_to_markdown(
    included_content_layers={ContentLayer.BODY, ContentLayer.NOTES}
))

# Numbers: the sticky notes on each sheet are notes.
budget = converter.convert("budget.numbers").document
print(budget.export_to_markdown(
    included_content_layers={ContentLayer.BODY, ContentLayer.NOTES}
))
```

A chart on a Keynote slide or a Numbers sheet becomes a picture classified by
its kind, with the data it plots in the picture's `meta.tabular_chart` and its
title as the caption, which is the shape the PowerPoint backend gives a chart. Neither app keeps
a picture of a chart, so the picture itself is empty unless you opt in to
`render_chart_images`. That rebuilds each chart from its data as an Office chart
and draws it with LibreOffice, so it needs a LibreOffice installation. The image
has the chart's kind, data and title but not its colours or fonts, and a mixed,
two-axis, bubble or interactive chart gets none:

```python
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, IWorkKeynoteFormatOption

converter = DocumentConverter(
    format_options={
        InputFormat.IWORK_KEYNOTE: IWorkKeynoteFormatOption(
            backend_options=IWorkBackendOptions(render_chart_images=True)
        )
    }
)
deck = converter.convert("deck.key").document
for picture in deck.pictures:
    if picture.meta is not None and picture.meta.tabular_chart is not None:
        print(picture.caption_text(deck), picture.meta.tabular_chart.chart_data)
```

The same option draws the charts of a Numbers spreadsheet, passed through
`IWorkNumbersFormatOption` instead. Charts are read from Keynote 6 and later, and
from Numbers documents of either generation; a chart in an iWork '09
presentation is not read.

A Numbers table cell that is filled with an image becomes a `RichTableCell`. Its
group holds the cell's text and the picture, as for a picture in a Word table
cell. The image of a cell is read only from documents that Numbers saved since
2017.

`sheet_names` converts only the sheets it names, and `page_range` narrows the
selection further, since each sheet is a page:

```python
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, IWorkNumbersFormatOption

doc_converter = DocumentConverter(
    format_options={
        InputFormat.IWORK_NUMBERS: IWorkNumbersFormatOption(
            backend_options=IWorkBackendOptions(sheet_names=["Summary", "Q1"])
        )
    }
)
```

The container is untrusted input, so size limits apply. They can be tuned with
`IWorkBackendOptions`, which all three formats take:

```python
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    IWorkKeynoteFormatOption,
    IWorkNumbersFormatOption,
    IWorkPagesFormatOption,
)

limits = IWorkBackendOptions(max_total_bytes=50 * 1024 * 1024)
doc_converter = DocumentConverter(
    format_options={
        InputFormat.IWORK_PAGES: IWorkPagesFormatOption(backend_options=limits),
        InputFormat.IWORK_NUMBERS: IWorkNumbersFormatOption(backend_options=limits),
        InputFormat.IWORK_KEYNOTE: IWorkKeynoteFormatOption(backend_options=limits),
    }
)
```

### Word embedded documents

A Word file can embed an Excel workbook, a PowerPoint presentation or another
Word document as an OLE object. By default, Docling keeps only the preview
picture of the object. Set `process_embedded_objects` to also convert the
embedded document:

```python
from docling.datamodel.backend_options import MsWordBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, WordFormatOption

converter = DocumentConverter(
    format_options={
        InputFormat.DOCX: WordFormatOption(
            backend_options=MsWordBackendOptions(process_embedded_objects=True)
        )
    }
)
doc = converter.convert("report.docx").document
for group in doc.groups:
    if group.name.startswith("embedded: "):
        print(group.name)  # e.g. "embedded: Microsoft_Excel_Worksheet.xlsx"
```

Each object becomes a group at the position of the object. The group holds the
preview picture, followed by the content of the embedded document, for example
the sheets and tables of a workbook. An embedded Word document is converted with
the same options, so its own embedded objects are converted too. An embedded
workbook or presentation is converted with the default options of its backend.
Linked objects, and objects of other types such as a PDF or a text file, keep
only their preview picture. The `format-xlsx` and `format-pptx` extras are
necessary to convert embedded workbooks and presentations.

An embedded document can come from an untrusted source, so limits apply. An
object that goes above a limit keeps only its preview picture, and a warning is
logged:

- Each embedded package must pass the same archive limits as the Word file
  (`max_total_bytes`, `max_file_bytes` and `max_member_count`), before it is
  opened.
- `max_embedded_objects` (default 32) and `max_embedded_total_bytes` (default
  512 MiB of decompressed data) apply to all embedded documents together,
  including those inside an embedded Word document.
- `max_embedding_depth` (default 3) is the deepest nesting level that is
  converted. With `1`, only the objects of the Word file itself are converted.
- A package that several objects share is converted once.

```python
from docling.datamodel.backend_options import MsWordBackendOptions

options = MsWordBackendOptions(
    process_embedded_objects=True,
    max_embedded_objects=10,
    max_embedding_depth=1,
)
```


### AsciiDoc comments

AsciiDoc comments carry editorial notes, review remarks and blocks that were
disabled on purpose, so the backend keeps them in the `notes` content layer
instead of dropping them. A run of `//` lines is one note and a `////` block is
another; the line breaks inside each one are kept, because a commented-out block
is markup rather than prose. Exports leave `notes` out by default, so comments
reach an export only when you ask for the layer:

```python
from docling_core.types.doc import ContentLayer
from docling.document_converter import DocumentConverter

doc = DocumentConverter().convert("manual.adoc").document
print(doc.export_to_markdown())  # body only, comments left out
print(doc.export_to_markdown(
    included_content_layers={ContentLayer.BODY, ContentLayer.NOTES}
))
```

A `//` line inside a `----` listing or a `....` literal block is code, not a
comment, and stays in the block. To drop comments entirely rather than keep
them in `notes`, turn `capture_comments` off:

```python
from docling.datamodel.backend_options import AsciiDocBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import AsciiDocFormatOption, DocumentConverter

converter = DocumentConverter(
    format_options={
        InputFormat.ASCIIDOC: AsciiDocFormatOption(
            backend_options=AsciiDocBackendOptions(capture_comments=False)
        )
    }
)
```

### Docling JSON input

A `DoclingDocument` JSON file can be converted again, e.g. to re-export it to
another format. Image references in that JSON which point at local files (bare
paths, relative paths or `file:` URIs, for pictures, tables and page images
alike) are ignored by default and a warning is logged: the images are dropped
from the loaded document, together with their size and resolution. Embedded
`data:` images and `http(s)` URLs are kept.

This also applies to a document saved with `ImageRefMode.REFERENCED`, whose
images are separate files. To load those images again from a JSON file you
trust, enable local fetching on the backend options:

```python
from docling.datamodel.backend_options import DeclarativeBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, DoclingJSONFormatOption

converter = DocumentConverter(
    format_options={
        InputFormat.JSON_DOCLING: DoclingJSONFormatOption(
            backend_options=DeclarativeBackendOptions(enable_local_fetch=True)
        )
    }
)
doc = converter.convert("saved_document.json")
```

Relative image paths are resolved against the current working directory. The
`docling` CLI has no option for this and always ignores local image references
in JSON input; save the document with `ImageRefMode.EMBEDDED` if it has to go
through the CLI again with its images.

### Fetch HTML images from remote hosts

The HTML backend only downloads images referenced by a page when you opt in with
`fetch_images=True` and `enable_remote_fetch=True` (the CLI equivalent is
`--html-image-fetch remote`). Downloads connect only to public, globally
routable addresses: every address of a host is checked, every redirect is
checked again before it is followed (up to `max_redirects`), and downloads stop
at `max_remote_image_bytes`. An image larger than `max_image_pixels` (width x
height, default 89,478,485) is skipped with a warning, and the rest of the
document converts; the Markdown backend has the same option. When `render_page=True`, the browser requests
remote resources through the same download path, and navigating the page away
from the source document is refused.

`headers` adds HTTP headers, such as credentials, to these downloads. They are
sent only to the origin of the source document and dropped on redirects to
other origins. To send them to other hosts, such as a CDN, list the allowed
origins in `headers_allowed_origins` (this replaces the default, so include the
source origin too if it needs the headers). For a local file or a stream
without a remote `source_uri`, headers are only sent when
`headers_allowed_origins` is set.

```python
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption

html_options = HTMLBackendOptions(
    fetch_images=True,
    enable_remote_fetch=True,
    headers={"Authorization": "Bearer TOKEN"},
    headers_allowed_origins=["https://example.com", "https://cdn.example.com"],
)
converter = DocumentConverter(
    format_options={
        InputFormat.HTML: HTMLFormatOption(backend_options=html_options)
    }
)
result = converter.convert("https://example.com/page.html")
```

On the CLI, pass `--html-image-headers` with a JSON object and repeat
`--html-image-headers-origin` for each allowed origin.

When a proxy is configured through the `HTTP_PROXY` / `HTTPS_PROXY`
environment variables, downloads go through the proxy and Docling does not check
the destination addresses; the proxy is then responsible for restricting which
destinations it connects to.

### XBRL taxonomy references

The XBRL backend reads the schemas and linkbases that an instance document
references. It needs `enable_local_fetch=True`, `enable_remote_fetch=True`, or
both, and applies them to every reference:

- Local paths and `file:` URLs are read only with `enable_local_fetch=True`. The
  instance document and the `taxonomy` folder (including its taxonomy packages)
  are always read.
- `http(s)` URLs are downloaded only with `enable_remote_fetch=True`. Without
  it, they resolve from the Arelle web cache only. Downloads connect only to
  public, globally routable addresses, and every redirect is checked again.
  With a proxy, the destination address is checked before the request goes to
  the proxy.
- Other URL schemes are refused.

A refused reference stops the conversion with an error that lists it.

## Impose limits on the document size

You can limit the file size and number of pages which should be allowed to process per document:

```python
from pathlib import Path
from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869"
converter = DocumentConverter()
result = converter.convert(source, max_num_pages=100, max_file_size=20971520)
```

## Convert ZIP archives

Docling expands generic ZIP archives into separate documents. Use
`convert_all()` to convert every supported archive member:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
results = list(converter.convert_all(["documents.zip"]))
```

`convert()` returns one result. It raises `ConversionError` when its source
contains multiple documents. Use `convert_all()` for a ZIP archive with more
than one member. By default, `convert_all()` raises on the first failed member.
Pass `raises_on_error=False` to receive failure results and continue with later
members. The configured file-size limit applies to each member and to the total
size of successfully read members.
For image inputs, the page count is checked against `max_num_pages` before any
page is decoded, and each page (for example each frame of a multi-page TIFF) is
decoded only when it is processed. The `max_image_pixels` backend option limits
the pixels (width x height) of one page; the default, 89,478,485, is the default
of Pillow's `Image.MAX_IMAGE_PIXELS`. A larger first page rejects the document,
and a larger later page fails that page:

```python
from docling.datamodel.backend_options import PdfBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, ImageFormatOption

converter = DocumentConverter(
    format_options={
        InputFormat.IMAGE: ImageFormatOption(
            backend_options=PdfBackendOptions(max_image_pixels=50_000_000)
        )
    }
)
```

### Limits on the decompressed size of DOCX, XLSX and EPUB files

DOCX, XLSX and EPUB files are ZIP archives, so a small file can decompress to a
much larger size. Docling checks the decompressed sizes that the archive declares
before it parses the file, and stops each read once the decompressed bytes go
above the limit. A file that exceeds a limit fails to convert with an
`ArchiveLimitError`, which is a `DocumentLoadError`.

| Option | `MsWordBackendOptions` / `MsExcelBackendOptions` | `EpubBackendOptions` |
| --- | --- | --- |
| `max_file_bytes` | 128 MiB per member | 10 MB per content document |
| `max_total_bytes` | 512 MiB for all members | 100 MB for all content documents |
| `max_member_count` | 10,000 members | 1,000 content documents |

The DOCX and XLSX limits apply to every member of the package, because
python-docx and openpyxl load the package members into memory. The EPUB limits
apply to the container, the package document and the content documents in the
spine, and also to the members extracted for image processing. Raise the limits
for large trusted files:

```python
from docling.datamodel.backend_options import MsExcelBackendOptions, MsWordBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    ExcelFormatOption,
    WordFormatOption,
)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.DOCX: WordFormatOption(
            backend_options=MsWordBackendOptions(max_file_bytes=512 * 1024 * 1024)
        ),
        InputFormat.XLSX: ExcelFormatOption(
            backend_options=MsExcelBackendOptions(max_total_bytes=2 * 1024**3)
        ),
    }
)
```

## Convert from binary PDF streams

You can convert PDFs from a binary stream instead of from the filesystem as follows:

```python
from io import BytesIO
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter

buf = BytesIO(your_binary_stream)
source = DocumentStream(name="my_doc.pdf", stream=buf)
converter = DocumentConverter()
result = converter.convert(source)
```

## Track conversion progress

To see progress without writing any code, turn on the built-in printer:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter(show_progress=True)
result = converter.convert("report.pdf")
```

It prints to stderr one line per document, a progress bar for its pages and
one for each enrichment step (picture classification or description, chart
extraction, code and formulas):

```text
[1] Converting report.pdf
  pages: 100%|██████████| 9/9 [00:49<00:00,  6.23s/page]
  DocumentPictureClassifier: 100%|██████████| 6/6 [00:00<00:00, 12.90item/s]
Finished report.pdf: success
```

On a terminal the bars fill up in docling orange while the document converts.
For plain bars, pass `ProgressPrinter(color=False)` as the callback or set the
`NO_COLOR` environment variable, which also works for the CLI. In a log file
or a pipe each bar is written once as a count, such as `pages 9/9`. For a batch,
give the printer the number of documents to get `[3/12]` instead of `[3]`:

```python
from docling.utils.progress import ProgressPrinter

printer = ProgressPrinter(total_documents=len(sources))
for result in DocumentConverter(progress_callback=printer).convert_all(sources):
    ...
```

With `settings.perf.doc_batch_concurrency > 1`, each document keeps its own
bar and the labels carry the document index, for example `[2] pages 9/9`.

On the command line the same output is on by default when stderr is a
terminal. `--no-progress` or `--quiet` turns it off, and `--progress` turns it
on for pipes and log files. While the bars are shown, the CLI prints its log
lines above them. If your own program logs to the terminal while
`show_progress=True` draws bars, wrap the conversion in
`tqdm.contrib.logging.logging_redirect_tqdm()` for the same effect.

### Your own progress callback

Pass a `progress_callback` to drive your own progress bar or forward the
events elsewhere. Given to `convert`, `convert_all` or `convert_string`, it
receives the events of that call only, so one converter can serve several
callers, such as the tasks of a service. Given to the `DocumentConverter`
constructor, it receives the events of every call. The callback receives one
event object at a time (see `docling.datamodel.progress`):

| Event | When |
| --- | --- |
| `DocumentStartedProgress` | An input document is picked up, including inputs that are then skipped. |
| `PhaseStartedProgress` | The document enters the `initialize` (pipeline set-up, slow only while the models of a new pipeline load), `build`, `assemble` or `enrich` phase. Every format reports them. |
| `PageCompletedProgress` | A page went through all page-level models, table structure included. PDF and image pipelines only. |
| `EnrichmentProgress` | Item counts of one enrichment step, reported per batch. `step` is a stable `EnrichmentStep` (`code_formula`, `picture_classification`, `picture_description`, `chart_extraction`, or `other` for enrichment models docling does not know); `label` is a readable name for display. |
| `DocumentCompletedProgress` | The document finished, with its `ConversionStatus`. Always the last event of a document, also when it failed. |

Every event carries `document_index`, the position of the document in the
`convert_all` call, and `document_name`. The events are Pydantic models, so
they are easy to forward, for example as JSON to a web client:

```python
from docling.datamodel.progress import ConversionProgressEvent
from docling.document_converter import DocumentConverter


def forward(event: ConversionProgressEvent) -> None:
    print(event.model_dump_json())  # or push it to a queue or websocket


converter = DocumentConverter()
result = converter.convert("report.pdf", progress_callback=forward)
```

```text
{"kind":"document_started","document_index":1,"document_name":"report.pdf"}
{"kind":"phase_started","document_index":1,"document_name":"report.pdf","phase":"initialize"}
...
{"kind":"page_completed","document_index":1,"document_name":"report.pdf","page_no":2,"success":true,"completed_pages":1,"total_pages":9}
...
```

Things to know:

- Pages can finish out of order. Use `completed_pages`, not `page_no`, for
  the progress value. `total_pages` counts the pages selected by `page_range`.
- Enrichment runs after the last page, on the whole document, so the page bar
  reaching 100% does not mean the document is done. Watch the
  `EnrichmentProgress` events, or wait for `DocumentCompletedProgress`.
- The VLM pipeline enriches the pictures of each page before it reports the
  page, so its page count already covers that work and it sends no
  `EnrichmentProgress`.
- Failed pages, and pages cut by `document_timeout`, are reported with
  `success=False`, so `completed_pages` always reaches `total_pages`.
- Exceptions raised by the callback are logged and ignored.
- For a single document all events come from the thread that called
  `convert`. With `settings.perf.doc_batch_concurrency > 1` several documents
  report at once from worker threads, so the callback must be thread-safe.

## Limit resource usage

You can limit the CPU threads used by Docling by setting the environment variable `OMP_NUM_THREADS` accordingly. The default setting is using 4 CPU threads.

Peak memory on large or image-heavy PDFs is driven by how many decoded pages are held at once.
`queue_max_size` bounds the pages buffered between pipeline stages, and `max_concurrent_results`
bounds the pages the threaded docling-parse backend decodes ahead of the pipeline:

```python
from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions(queue_max_size=1)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend_options=ThreadedDoclingParseBackendOptions(max_concurrent_results=1),
        )
    }
)
```
