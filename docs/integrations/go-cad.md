Parse CAD drawings (DWG/DXF) with Docling through [go-cad], a pure-Go DWG/DXF
parser. Install the [docling-go-cad] integration package and `.dwg`, `.dxf`,
and `.dxfb` become first-class inputs: each detected sheet frame becomes a
document section with the sheet title as heading, the rendered image as a
picture item, and the drawing's text entities as text items — flowing through
the same `DocumentConverter` pipeline as PDF and Office documents.

```bash
pip install "docling-go-cad[docling]"
go install github.com/unitedrhino/go-cad/cmd/caddocling@latest
```

```python
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

from docling_go_cad import CadFormatOption, register_docling

register_docling()

converter = DocumentConverter(
    format_options={InputFormat.CAD: CadFormatOption()}
)
result = converter.convert("fire_protection.dwg")
print(result.document.export_to_markdown())
```

For results without the converter routing, the convenience API
`docling_go_cad.convert_cad("drawing.dwg")` returns a `DoclingDocument`
directly. The integration reuses go-cad's smart sheet splitting (title-block
frame detection, per-sheet rendering with automatic text-legibility retry),
and works with DWG R9–R2018 and ASCII/binary DXF.

- 💻 [go-cad][go-cad]
- 📦 [docling-go-cad integration package][docling-go-cad]

[go-cad]: https://github.com/unitedrhino/go-cad
[docling-go-cad]: https://github.com/unitedrhino/go-cad/tree/main/integrations/docling-go-cad
