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
- `allow_tikz_engine_download`
  Allows the LaTeX backend to download the Tectonic binary when TikZ rendering
  is enabled and no local binary is available.
- `tikz_engine_timeout`
  Sets the timeout, in seconds, for rendering a single TikZ diagram.
- `tikz_engine_allow_shell_escape`
  Defaults to `False`. Enable this only when required by the input document,
  since shell escape is less safe for untrusted LaTeX sources.

### CLI flags

The CLI exposes the same behavior with these flags:

- `--tikz-engine` / `-T`
- `--no-tikz-engine-download`
- `--tikz-engine-timeout`
- `--tikz-shell-escape`

### Fallback behavior

- When Tectonic compilation succeeds, the TikZ diagram is rasterized and stored
  as an image.
- When compilation fails, times out, produces no PDF, or rasterization fails,
  Docling preserves the original TikZ source as fallback code metadata instead
  of dropping the figure.


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
        
