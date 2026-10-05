# Pipeline options

Pipeline options allow to customize the execution of the models during the conversion pipeline.
This includes options for the OCR engines, the table model as well as enrichment options which
can be enabled with `do_xyz = True`.


This is an automatic generated API reference of the all the pipeline options available in Docling.

## LaTeX TikZ Rendering

Docling's LaTeX backend can optionally render `tikzpicture` environments into
images using the Tectonic engine.

### Backend options

`LatexBackendOptions` supports the following TikZ-related options:

- `tikz_engine`
  Set to `"tectonic"` to enable optional TikZ rendering.
- `tikz_engine_timeout`
  Sets the timeout, in seconds, for rendering a single TikZ diagram.
- `tikz_engine_allow_shell_escape`
  Defaults to `False`. Enable this only for trusted input that needs
  `\write18`: it passes `-Z shell-escape` to Tectonic and skips the source
  pre-check described below.

These are Python options; the `docling` CLI does not expose TikZ rendering.

```python
from docling.datamodel.backend_options import LatexBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, LatexFormatOption

converter = DocumentConverter(
    format_options={
        InputFormat.LATEX: LatexFormatOption(
            backend_options=LatexBackendOptions(tikz_engine="tectonic")
        )
    }
)
```

### Rendering without shell escape

Each diagram is compiled in its own temporary directory, together with the
`\input`/`\include` files and `\includegraphics` assets it references
relative to the document's directory. Without shell escape:

- Tectonic runs with `--untrusted --only-cached`: shell escape is off and
  missing bundle files are not downloaded.
- Before compiling, Docling does a best-effort check of the diagram, its
  preamble and every staged file: rendering is skipped when `\input`,
  `\include`, `\includegraphics`, `\InputIfFileExists` or `\graphicspath`
  name an absolute, `~`, drive-letter or `..` path, or when the source uses
  `\openin`, `\openout`, `\XeTeXpicfile` or `\XeTeXpdffile`.

!!! note
    The pre-check only reads the source text and does not cover every way TeX
    can access files. When rendering untrusted LaTeX with Tectonic, run the
    conversion in an isolated environment, for example a container without
    host mounts or network access.

### Fallback behavior

- When Tectonic compilation succeeds, the TikZ diagram is rasterized and stored
  as an image.
- When the pre-check skips a diagram, or compilation fails, times out,
  produces no PDF, or rasterization fails, Docling preserves the original TikZ
  source as fallback code metadata instead of dropping the figure.


::: docling.datamodel.pipeline_options
    handler: python
    options:
        show_if_no_docstring: true
        show_submodules: true
        docstring_section_style: list
        filters: ["!^_"]
        heading_level: 2
        inherited_members: true
        merge_init_into_class: true
        separate_signature: true
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        show_source: false
        show_symbol_type_heading: true
        show_symbol_type_toc: true
        signature_crossrefs: true
        summary: true

<!-- ::: docling.document_converter.DocumentConverter
    handler: python
    options:
        show_if_no_docstring: true
        show_submodules: true -->
        
