# Scripts

Maintenance scripts for the Docling repository. Run them from the repository
root with `uv run python <script>` unless a script says otherwise.

Scripts that GitHub workflows call (triage, release, CI helpers) live in
[`.github/scripts/`](../.github/scripts/), not here.

## Layout

| Folder | Purpose |
| --- | --- |
| [`tools/`](tools/) | Maintainer tools to get an overview of GitHub issues and PRs. |
| [`benchmarking/`](benchmarking/) | Performance and memory benchmarks. |
| [`checks/`](checks/) | Repository checks. `make check`, pre-commit, and CI run them. |
| `*.py` (this folder) | Other scripts: docs rendering and model validation. |

## `tools/`

See [`tools/README.md`](tools/README.md) for the details.

- `sync_prs.py`: rank the open PRs (ready to merge, CI green but needs
  approval, merge conflict, other). It can also ask authors to resolve merge
  conflicts, and it can approve the waiting CI runs of PRs that have the
  label `ai:ci-safe`.

## `benchmarking/`

- `iterate_pdf_pages.py`: load the PDFs of a directory with the threaded
  docling-parse backend and extract the text cells and page images of every
  page. Memory mode records the metrics for `plot_memory_metrics.py`.
- `plot_memory_metrics.py`: plot the memory metrics from
  `iterate_pdf_pages.py`.
- `xlsx_merged_cells.py`: measure XLSX conversion time and memory for
  workbooks with many merged cells.

## `checks/`

- `check_max_lines.py`: fail when a source file is longer than the limit.
- `check_tach_module_coverage.py`: fail when a `docling` module is not
  covered by a Tach module in `tach.toml`.

## Other scripts

- `render_notebooks.py`: render `docs/examples` notebooks to Markdown for
  MkDocs (`make docs`).
- `render_cli_reference.py`: generate `docs/reference/cli.md` from the Typer
  apps (`make docs`).
- `validate_asr_models.py`: run real ASR conversions on one audio fragment,
  outside normal pytest runs.
