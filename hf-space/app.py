"""
  Docling HTTP API — wraps docling.DocumentConverter with a minimal FastAPI interface.

  Endpoints
  ---------
  GET  /            → redirect to /docs (Swagger UI)
  GET  /health      → {"status":"ok"}
  POST /convert     → upload a file, return markdown / text / json
  POST /convert-url → pass a public URL, return markdown / text / json
  """

  import tempfile
  from pathlib import Path

  from fastapi import FastAPI, File, Form, HTTPException, UploadFile
  from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse

  app = FastAPI(
      title="Docling API",
      description=(
          "Convert PDF, DOCX, PPTX, HTML, images, and more to Markdown, plain text, or JSON "
          "using the [Docling](https://github.com/docling-project/docling) library."
      ),
      version="1.0.0",
  )

  _converter = None


  def get_converter():
      global _converter
      if _converter is None:
          from docling.document_converter import DocumentConverter
          _converter = DocumentConverter()
      return _converter


  @app.on_event("startup")
  async def _warmup():
      """Pre-load ML models at startup so the first request is instant."""
      get_converter()


  @app.get("/", include_in_schema=False)
  def root():
      return RedirectResponse(url="/docs")


  @app.get("/health", summary="Health check")
  def health():
      return {"status": "ok"}


  @app.post("/convert", summary="Convert an uploaded document")
  async def convert_file(
      file: UploadFile = File(..., description="Document to convert (PDF, DOCX, PPTX, HTML, CSV…)"),
      format: str = Form("markdown", description="markdown | text | json"),
  ):
      _validate_format(format)
      suffix = Path(file.filename or "upload").suffix or ".pdf"
      with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
          tmp.write(await file.read())
          tmp_path = Path(tmp.name)
      try:
          return _run_conversion(str(tmp_path), format)
      finally:
          tmp_path.unlink(missing_ok=True)


  @app.post("/convert-url", summary="Convert a document from a public URL")
  async def convert_url(
      url: str = Form(..., description="Publicly accessible URL of the document"),
      format: str = Form("markdown", description="markdown | text | json"),
  ):
      _validate_format(format)
      return _run_conversion(url, format)


  ALLOWED_FORMATS = {"markdown", "text", "json"}


  def _validate_format(fmt: str) -> None:
      if fmt not in ALLOWED_FORMATS:
          raise HTTPException(400, f"Invalid format '{fmt}'. Allowed: {sorted(ALLOWED_FORMATS)}")


  def _run_conversion(source: str, fmt: str):
      try:
          result = get_converter().convert(source)
          doc = result.document
      except Exception as exc:
          raise HTTPException(422, f"Conversion failed: {exc}") from exc

      if fmt == "markdown":
          return PlainTextResponse(doc.export_to_markdown(), media_type="text/markdown; charset=utf-8")
      if fmt == "text":
          return PlainTextResponse(doc.export_to_text(), media_type="text/plain; charset=utf-8")
      return JSONResponse(doc.export_to_dict())
  