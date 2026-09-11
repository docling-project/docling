# Synthetic AFP fixture

`sources/synthetic.afp` is a hand-built MO:DCA stream containing two pages and
PTOCA Transparent Data controls. It contains no third-party document content and
is distributed under the repository license.

The matching Markdown and Docling JSON exports live in `groundtruth/` and are
verified by `tests/test_backend_afp.py` through `DocumentConverter`.
