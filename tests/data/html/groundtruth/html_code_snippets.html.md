# Code snippets

The Pythagorean theorem can be written as an equation relating the lengths of the sides *a* , *b* and the hypotenuse *c* .

To use Docling, simply install `docling` from your package manager, e.g. pip: `pip install docling`

To convert individual documents with python, use `convert()` , for example:

```
from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869"
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())
```

The program will output: `## Docling Technical Report[...]`

Prefetch the models:

- Use the `docling-tools models download` utility:
- Alternatively, models can be programmatically downloaded using `docling.utils.model_downloader.download_models()` .
- Also, you can use download-hf-repo parameter to download arbitrary models from HuggingFace by specifying repo id: `$ docling-tools models download-hf-repo ds4sd/SmolDocling-256M-preview Downloading ds4sd/SmolDocling-256M-preview model from HuggingFace...`