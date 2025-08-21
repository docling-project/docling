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
[Architecture](../concepts/architecture.md) guide, provides a versatile document model along with a full suite of
supported operations.

### CLI

You can additionally use Docling directly from your terminal, for instance:

```console
docling https://arxiv.org/pdf/2206.01062
```

The CLI provides various options, such as 🥚[SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview) (incl. MLX acceleration) & other VLMs:
```bash
docling --pipeline vlm --vlm-model smoldocling https://arxiv.org/pdf/2206.01062
```

For all available options, run `docling --help` or check the [CLI reference](../reference/cli.md).

## What's next

Check out the Usage subpages (navigation menu on the left) as well as our [featured examples](../examples/index.md) for
additional usage workflows, including conversion customization, RAG, framework integrations, chunking, serialization,
enrichments, and much more!
