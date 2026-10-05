# Docling for RAG (integrations)

For retrieval-augmented generation you usually want two things: convert the
document, then split it into retrieval-ready chunks with metadata (headings,
page numbers) preserved. You can do this directly with the SDK's
`HybridChunker` (see [python-sdk.md](python-sdk.md#chunking-for-rag)), but for
most frameworks there are **ready-made loaders** that wrap convert + chunk so
you don't write glue code.

## Framework loaders

| Framework | Package | Component |
|---|---|---|
| LangChain | `langchain-docling` | `DoclingLoader` |
| LlamaIndex | `llama-index-readers-docling` (+ `llama-index-node-parser-docling`) | `DoclingReader` + Docling node parser |
| Haystack | `docling-haystack` | Docling converter component |

### LangChain

```bash
pip install langchain-docling
```

```python
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

loader = DoclingLoader(
    file_path=["report.pdf", "https://example.com/paper.pdf"],
    export_type=ExportType.DOC_CHUNKS,   # chunked; or ExportType.MARKDOWN
)
docs = loader.load()                     # list[langchain_core.documents.Document]
# each chunk keeps metadata: source, headings, page numbers
```

### LlamaIndex

```bash
pip install llama-index-readers-docling llama-index-node-parser-docling
```

```python
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser

reader = DoclingReader()
nodes = DoclingNodeParser().get_nodes_from_documents(
    reader.load_data(file_path="report.pdf")
)
```

### Haystack

```bash
pip install docling-haystack
```

Use the Docling converter component in a Haystack indexing pipeline; it emits
chunked documents ready for a writer/embedder.

## When to skip the loaders

If you already convert with `DocumentConverter` (to control the pipeline) and
just need chunks, call the `HybridChunker` yourself — it exposes headings and
source page per chunk and lets you pick the tokenizer to match your embedding
model. See [python-sdk.md](python-sdk.md#chunking-for-rag).

For remote/at-scale conversion feeding a RAG index, the Service Client's
`chunk()` returns retrieval-ready pieces without any local models — see
[service-client.md](service-client.md).

> Package names/APIs for the third-party loaders can change; check each
> framework's Docling integration docs for the current import paths.
