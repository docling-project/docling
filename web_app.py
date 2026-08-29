"""Web UI: upload a PDF, convert it with OCR (RU + CH), download the Markdown.

FastAPI app served by uvicorn. A single page lets you pick a file, converts it
with the Docling PDF pipeline (EasyOCR restricted to ru + ch_sim), and returns
the resulting Markdown as a downloadable .md file.

Run inside the container (entrypoint):  python /root/web_app.py
"""

from __future__ import annotations

import logging
import tempfile
from io import BytesIO
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DocumentStream
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("web_app")

OCR_LANGUAGES = ["ru", "en"]

_pipeline_options = PdfPipelineOptions()
_pipeline_options.do_ocr = True
_pipeline_options.do_table_structure = False
_pipeline_options.ocr_options = EasyOcrOptions(lang=OCR_LANGUAGES, use_gpu=False)

_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options),
    }
)

app = FastAPI(title="Docling OCR Converter", version="1.0")

_UPLOAD_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Docling OCR Converter</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 16px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 640px; margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.4rem; }
  form { display: grid; gap: 1rem; margin-top: 1.5rem; }
  .row { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; }
  input[type=file] { flex: 1 1 auto; }
  button { padding: .6rem 1.2rem; font: inherit; cursor: pointer; }
  p.hint { color: #888; font-size: .9rem; }
  code { background: rgba(127,127,127,.15); padding: .1em .3em; border-radius: .3em; }
</style>
</head>
<body>
  <h1>Docling OCR Converter</h1>
  <p class="hint">Upload a PDF. It is converted with EasyOCR
    (<code>ru</code> + <code>ch_sim</code> + <code>en</code>) and you get a Markdown file back.</p>
  <form action="/convert" method="post" enctype="multipart/form-data">
    <div class="row">
      <input type="file" name="file" accept="application/pdf,.pdf" required>
      <button type="submit">Convert &amp; Download</button>
    </div>
  </form>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _UPLOAD_PAGE


@app.post("/convert")
async def convert(file: UploadFile = File(...)) -> Response:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    _log.info("converting %s (%d bytes)", file.filename, len(data))
    stream = DocumentStream(name=file.filename, stream=BytesIO(data))
    try:
        result = _converter.convert(stream)
    except Exception as exc:  # noqa: BLE001
        _log.exception("conversion failed")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {exc}") from exc

    md = result.document.export_to_markdown()
    out_name = Path(file.filename).stem + ".md"
    _log.info("done -> %s", out_name)

    from urllib.parse import quote
    quoted_name = quote(out_name)

    return Response(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{quoted_name}"; '
                f"filename*=UTF-8''{quoted_name}"
            )
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# if __name__ == "__main__":
uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")