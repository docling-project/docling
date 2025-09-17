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
