---
  title: Docling API
  emoji: 📄
  colorFrom: blue
  colorTo: indigo
  sdk: docker
  app_port: 7860
  pinned: false
  license: mit
  short_description: Convert PDF, DOCX, HTML and more to Markdown / JSON via REST
  ---

  # Docling API 🚀

  Free, permanent REST API for converting documents (PDF, DOCX, PPTX, HTML, images…)
  to Markdown, plain text, or JSON — powered by
  [Docling](https://github.com/docling-project/docling).

  ## Endpoints

  | Method | Path | Description |
  |--------|------|-------------|
  | `GET` | `/` | Redirects to Swagger UI (interactive docs) |
  | `GET` | `/health` | Returns `{"status":"ok"}` |
  | `POST` | `/convert` | Upload a file → get back converted content |
  | `POST` | `/convert-url` | Pass a public URL → get back converted content |

  ### Request parameters (multipart/form-data)

  | Field | Required | Values | Default |
  |-------|----------|--------|---------|
  | `file` | ✅ for /convert | any supported file | — |
  | `url` | ✅ for /convert-url | public URL string | — |
  | `format` | ❌ | `markdown` · `text` · `json` | `markdown` |

  ## Usage examples

  ```bash
  # Convert a local PDF to Markdown
  curl -X POST "https://<your-space-url>.hf.space/convert" \
    -F "file=@resume.pdf" \
    -F "format=markdown"

  # Convert from a public URL to plain text
  curl -X POST "https://<your-space-url>.hf.space/convert-url" \
    -F "url=https://arxiv.org/pdf/2206.01062" \
    -F "format=text"

  # Get structured JSON output
  curl -X POST "https://<your-space-url>.hf.space/convert" \
    -F "file=@report.pdf" \
    -F "format=json"
  ```

  ## Supported input formats

  PDF, DOCX, PPTX, XLSX, HTML, Markdown, AsciiDoc, CSV, images (PNG / JPEG / TIFF / BMP / WebP), and more.

  ## Local development

  ```bash
  pip install fastapi uvicorn python-multipart docling
  uvicorn "hf-space.app:app" --reload --port 7860
  # Then open http://localhost:7860/docs
  ```
  