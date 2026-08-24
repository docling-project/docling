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
Downloading rapidocr torch chinese models...
Downloading rapidocr torch english models...
Downloading rapidocr onnxruntime chinese models...
Downloading rapidocr onnxruntime english models...
Models downloaded into $HOME/.cache/docling/models.
```

To prefetch EasyOCR recognition models for specific languages, repeat
`--easyocr-lang` with the same language codes used by `EasyOcrOptions.lang`:

```sh
$ docling-tools models download easyocr --easyocr-lang ch_sim --easyocr-lang ja
```

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

In Docling we decided to allow such models, but we require the user to explicitly opt-in in communicating with external services.

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


## Adjust pipeline features

The example file [custom_convert.py](../examples/custom_convert.py) contains multiple ways
one can adjust the conversion pipeline and features.

### Image resolution and scale

Page coordinates use 72 points per inch. For image inputs, embedded DPI metadata
determines the physical page size; missing DPI and `(1, 1)` DPI are treated as 72 DPI.
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
pipeline_options.generate_parsed_pages = True  # required by the font-style signal

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

See [PDF heading levels](./heading_levels.md) for the signals, their precedence and all options.

### Convert Apple Pages documents

Apple Pages (`.pages`) documents convert like any other format, and both
container generations are read (requires the `format-iwork` extra):

```python
from docling.document_converter import DocumentConverter

doc = DocumentConverter().convert("report.pages").document
print(doc.export_to_markdown())
```

Pages changed its container completely in 2013, so Docling reads whichever one
the document uses:

- **Pages 5 and later (2013 onwards)** store the document as `Index/*.iwa`,
  Snappy-framed protobuf archives. Docling walks that object graph directly.
- **iWork '09 and earlier** stored a plain `index.xml`, which is parsed instead.
  Template placeholder text (`sf:ghost-text`) is skipped, so an untouched
  template yields no spurious content.

Titles and headings are recovered from the paragraph styles Pages applies
("Title", "Heading 1", "Subheading"), which are named identically in both
generations.

!!! note "Not yet extracted"

    Character formatting, lists, text boxes, headers, footers, footnotes
    and comments are not included — only the main body and its tables are
    read — and password-protected documents cannot be read. Table cells
    holding anything other than text are left empty.

The container is untrusted input, so member count, total size, per-member size
and decompressed output are all bounded. Those limits can be tuned with
`IWorkBackendOptions`:

```python
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, IWorkPagesFormatOption

doc_converter = DocumentConverter(
    format_options={
        InputFormat.IWORK_PAGES: IWorkPagesFormatOption(
            backend_options=IWorkBackendOptions(max_total_bytes=50 * 1024 * 1024)
        )
    }
)
```

## Impose limits on the document size

You can limit the file size and number of pages which should be allowed to process per document:

```python
from pathlib import Path
from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869"
converter = DocumentConverter()
result = converter.convert(source, max_num_pages=100, max_file_size=20971520)
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

## Limit resource usage

You can limit the CPU threads used by Docling by setting the environment variable `OMP_NUM_THREADS` accordingly. The default setting is using 4 CPU threads.
