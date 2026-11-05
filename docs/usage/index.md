## Basic usage

### Python

In Docling, working with documents is as simple as:

1. converting your source file to a Docling document
2. using that Docling document for your workflow

For example, the snippet below shows conversion with export to Markdown:

```python
from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869"  # file path or URL
converter = DocumentConverter()
doc = converter.convert(source).document

print(doc.export_to_markdown())  # output: "### Docling Technical Report[...]"
```

Docling supports a wide array of [file formats](./supported_formats.md) and, as outlined in the
[architecture](../concepts/architecture.md) guide, provides a versatile document model along with a full suite of
supported operations.

#### Releasing input resources

After conversion, call `result.input.release()` when you have finished any work
that needs the input backend. This unloads the backend, closes input streams it
owns, and drops the backend reference. The converted document and input metadata
remain available, so you can still export or otherwise use the result:

```python
result = converter.convert(source)
result.input.release()
print(result.document.export_to_markdown())
```

Calling `release()` more than once is safe, including when the pipeline has
already unloaded the backend or the input was rejected before a backend opened.
Do not call it while conversion or another operation is using the input backend.
If cleanup raises an exception, the backend reference is retained so you can
retry `release()`.

### CLI

You can additionally use Docling directly from your terminal, for instance:

```console
docling https://arxiv.org/pdf/2206.01062
```

The CLI provides various options, such as 🥚[GraniteDocling](https://huggingface.co/ibm-granite/granite-docling-258M) (incl. MLX acceleration) & other VLMs:
```bash
docling --pipeline vlm --vlm-model granite_docling https://arxiv.org/pdf/2206.01062
```

For all available options, run `docling --help` or check the [CLI reference](../reference/cli.md).

## What's next

Check out the Usage subpages (navigation menu on the left) as well as our [featured examples](../examples/index.md) for
additional usage workflows, including conversion customization, RAG, framework integrations, chunking, serialization,
enrichments, and much more!
