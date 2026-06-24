"""Perform memory profile with memray to compare TableCell and FastTableCell. Usage examples:
1. `python3 perfs/simplexlsx_profiling.py --cell "TableCell" -o "./perfs/data/profiles/profiling_table_cell"`
2. `python3 perfs/simplexlsx_profiling.py --cell "FastTableCell" -o "./perfs/data/profiles/profiling_fast_table_cell"`

Requires:
- memray >= 1.19.3
"""

import argparse
import gc
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import memray
import yaml

from docling.backend.msexcel_backend import MsExcelDocumentBackend
from docling.datamodel.backend_options import MsExcelBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.document_converter import DocumentConverter, ExcelFormatOption

_log = logging.getLogger(__name__)


### HELPERS


def collect_xlsx_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a valid directory")

    files = list(folder.iterdir())

    # check for invalid extensions
    invalid_files = [f for f in files if f.is_file() and f.suffix.lower() != ".xlsx"]
    if invalid_files:
        raise ValueError(
            "Non-xlsx files found:\n" + "\n".join(str(f) for f in invalid_files)
        )

    # keep only xlsx files (sorted for determinism)
    xlsx_files = sorted(f for f in files if f.is_file() and f.suffix.lower() == ".xlsx")

    return xlsx_files


def _write_report(report_file: Path, payload: dict[str, Any]) -> None:
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f)
    except (OSError, yaml.YAMLError):
        _log.exception("Failed to write YAML report")


def delete_file(path: Path):
    try:
        path.unlink()
    except OSError:
        _log.warning(f"Failed to remove file with path {path!s}")


### CORE


def run_benchmark(
    filepath: Path,
    backend_options: MsExcelBackendOptions,
    outdir: Path,
    report: dict[str, Any] | None,
    overwrite: bool,
) -> None:
    outfile = outdir / f"memray-simple_xlsx_test_{filepath.stem}.bin"
    outfile_flamegraph = (
        outdir / f"memray-flamegraph-simple_xlsx_test_{filepath.stem}.html"
    )
    outfile_markdown = outdir / f"memray-flamegraph-simple_xlsx_test_{filepath.stem}.md"
    # handle overwriting, you can create a helper overwrite
    if outfile.is_file():
        _log.info(f"skipped, output already exists {filepath.name} -> {outfile.name}")
        if not overwrite:
            return

        # Delete previous run files
        delete_file(outfile)
        delete_file(outfile_flamegraph)
        delete_file(outfile_markdown)

    run_start_time = time.monotonic()
    _log.info(
        f"Processing {filepath.name} -> {outfile.name}; start time {run_start_time}"
    )

    # trace_python_allocators (bool), requires Python 3.13.3
    with memray.Tracker(outfile, trace_python_allocators=True):
        # options = MsExcelBackendOptions(table_cell_type="TableCell")
        format_options = {
            InputFormat.XLSX: ExcelFormatOption(backend_options=backend_options)
        }
        converter = DocumentConverter(
            allowed_formats=[InputFormat.XLSX], format_options=format_options
        )

        conversion_result = converter.convert(filepath)
        doc = conversion_result.document

    run_elapsed_time = time.monotonic() - run_start_time

    _log.info("Writing excel data to disc as markdown")
    md_text = doc.export_to_markdown()
    with open(outfile.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(md_text)

    _log.info(f"Processing took {run_elapsed_time:.2f}s")
    if report is not None:
        _log.info(f"Updating {outfile.name} report")
        run_report = {}
        report[outfile.name] = run_report
        run_report["elapsed_time"] = run_elapsed_time
    # Convert the file to html
    subprocess.run(
        ["python3", "-m", "memray", "flamegraph", "--leaks", "--temporal", outfile]
    )
    _log.info("\n")


### ARGUMENT PARSING


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input-dir",
        default=Path("./perfs/data/xlsx_benchmark"),
        type=Path,
        help="Local directory containing xlsx benchmark files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=Path("./perfs/data/profiles"),
        type=Path,
        help="Output directory path",
    )
    parser.add_argument(
        "--overwrite", default=True, type=bool, help="Overwrite previous generated data"
    )
    parser.add_argument(
        "--cell",
        choices=("TableCell", "FastTableCell"),
        default="TableCell",
        type=str,
        help=(
            "Use TableCell or FastTableCell object, TableCell is a pydantic object, FastTable cell is a dataclass"
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args()
    print(args)
    input_dir = args.input_dir
    outdir = args.output_dir  # Path("./perfs/data/profiles")
    outdir.mkdir(parents=True, exist_ok=True)
    overwrite = args.overwrite

    report = {}
    report_file = outdir / "report.yaml"
    if report_file.is_file() and not overwrite:
        raise Exception(
            "Previous report already exists, please delete it to avoid inconsistent data"
        )
    if report_file is not None:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text("", encoding="utf-8")
        _log.info("Collecting report to %s", report_file.name)

    files = collect_xlsx_files(input_dir)

    # Select the type of backend option to use
    if args.cell == "TableCell":
        options = MsExcelBackendOptions(table_cell_type="TableCell")
    elif args.cell == "FastTableCell":
        options = MsExcelBackendOptions(table_cell_type="FastTableCell")
    else:
        raise ValueError(f"Unrecognized type of TableCell {args.cell}")

    ### Perform benchmark run
    processed_files = []
    skipped_files = []
    for file_path in files:
        if file_path.is_file():
            # -> this profiling shows that the the problem is in self._find_tables_in_sheet(doc, sheet, page_no)
            run_benchmark(file_path, options, outdir, report, overwrite)
            gc.collect()
            processed_files.append(file_path.name)
        else:
            skipped_files.append(file_path.name)
            _log.info(f"file {file_path.name} not found. Skipping to next file")

    ### add metrics to report
    report["processed_files"] = processed_files
    if len(skipped_files) > 0:
        report["skipped_files"] = skipped_files
    report["warning"] = (
        "Timing information are not accurate since we only perform one run per excel sheet and there is no experiment isolation."
    )
    sorted_report = dict(sorted(report.items()))
    _write_report(report_file, sorted_report)


if __name__ == "__main__":
    main()
