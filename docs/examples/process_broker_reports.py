# %% [markdown]
# Process a batch of broker research reports (Chinese PDFs) into a
# machine-readable financial summary.
#
# What this example does
# - Converts every PDF in an input directory with Docling's standard pipeline
#   (picture + table structure enabled).
# - Exports one Markdown file per report with referenced figures.
# - Extracts key financial metrics (revenue, net profit, ROE, NPL ratio, EPS,
#   valuation multiples, ...) from the report tables into a compact JSON summary.
# - Prints a per-file audit (pages / tables / pictures / extracted metrics) and
#   optionally renders a comparison chart when matplotlib is installed.
#
# Why this matters
# - Broker reports mix dense tables, figures and footnotes. Pulling the
#   headline numbers into a structured summary is the backbone of research
#   report knowledge bases and downstream BI / Q&A systems.
#
# How to run
# - From the repo root: `python docs/examples/process_broker_reports.py`.
# - Input: pass a directory via `--input` (default: the bundled sample report).
# - Outputs are written to `scratch/broker_reports/`.
#
# Key options
# - `ImageRefMode.REFERENCED` keeps the Markdown small and images on disk.
# - `PdfPipelineOptions` enable the picture/table structures used by the summary.

# %%

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

from docling_core.types.doc import DocItemLabel, ImageRefMode

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

# Row labels commonly found in the "key financials" table of a research report,
# mapped to stable, database-friendly metric keys.
METRIC_ALIASES = {
    "营业收入": "revenue",
    "营收": "revenue",
    "归母净利润": "net_profit",
    "净利润": "net_profit",
    "净利润增速": "net_profit_growth",
    "加权平均roe": "roe",
    "roe": "roe",
    "不良贷款率": "npl_ratio",
    "拨备覆盖率": "provision_coverage",
    "eps": "eps",
    "pe": "pe",
    "pb": "pb",
}

_UNIT_PAREN = re.compile(r"[（(].*?[)）]")


def normalize_label(label: str) -> str:
    """Normalize a table row label for alias matching.

    Strips whitespace and parenthetical units, e.g.
    "加权平均ROE（%）" -> "加权平均roe".
    """
    return _UNIT_PAREN.sub("", label).replace(" ", "").lower()


def report_title(doc) -> str | None:
    """First title or section header of the document, if any."""
    for item in doc.texts:
        if item.label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER):
            text = item.text.strip()
            if text:
                return text
    return None


def _column_label(col) -> str:
    """Render a table column header (possibly a tuple) as a plain string."""
    if isinstance(col, tuple):
        parts = [str(c) for c in col if pd.notna(c) and str(c).strip()]
        return " ".join(parts)
    return str(col).strip()


def extract_metrics(doc) -> dict:
    """Walk detected tables and collect rows that match known financial metrics.

    Returns ``{metric_key: {"periods": [...], "values": [...]}}`` using the
    table column headers as periods (e.g. 2024A / 2025E / 2026E).
    """
    metrics: dict = {}
    for table in doc.tables:
        df = table.export_to_dataframe(doc=doc)
        if df.empty:
            continue
        periods = [_column_label(c) for c in df.columns[1:]]
        for _, row in df.iterrows():
            key = METRIC_ALIASES.get(normalize_label(str(row.iloc[0])))
            if key is None:
                continue
            values = [
                str(v).strip()
                for v in row.iloc[1:]
                if pd.notna(v) and str(v).strip() not in ("", "-", "--", "nan", "None")
            ]
            if values:
                metrics[key] = {"periods": periods[: len(values)], "values": values}
    return metrics


def render_chart(metrics: dict, out_path: Path) -> Path | None:
    """Bar chart of revenue vs net profit across periods (matplotlib optional)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    series = [(key, metrics[key]) for key in ("revenue", "net_profit") if key in metrics]
    if not series:
        return None
    periods = series[0][1]["periods"]
    if not periods:
        return None

    x = range(len(periods))
    _, ax = plt.subplots(figsize=(6, 3.5))
    for key, data in series:
        values = []
        for value in data["values"]:
            try:
                values.append(float(value.replace(",", "")))
            except ValueError:
                values.append(float("nan"))
        ax.plot(x, values, marker="o", label=key)
    ax.set_xticks(list(x))
    ax.set_xticklabels(periods)
    ax.set_ylabel("RMB 100M")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=150)
    plt.close(ax.figure)
    return out_path


def build_converter() -> DocumentConverter:
    """Converter with picture and table structure enabled for report QA."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_structure = True
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def process_report(converter: DocumentConverter, pdf_path: Path, out_dir: Path) -> dict:
    """Convert one report and export Markdown, metrics JSON and an optional chart."""
    result = converter.convert(pdf_path)
    doc = result.document
    metrics = extract_metrics(doc)
    summary = {
        "file": pdf_path.name,
        "title": report_title(doc) or pdf_path.stem,
        "pages": len(doc.pages),
        "tables": len(doc.tables),
        "pictures": len(doc.pictures),
        "metrics": metrics,
    }

    if doc.tables or doc.pictures:
        out_dir.mkdir(parents=True, exist_ok=True)
        doc.save_as_markdown(out_dir / f"{pdf_path.stem}.md", image_mode=ImageRefMode.REFERENCED)
        if metrics:
            (out_dir / f"{pdf_path.stem}-metrics.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            chart_path = render_chart(metrics, out_dir / f"{pdf_path.stem}-metrics.png")
            if chart_path is not None:
                _log.info("saved chart %s", chart_path.name)
    return summary


def print_summary(report: dict) -> None:
    """Human-readable audit line for one report."""
    print(
        f"  {report['file']}: {report['pages']} pages, "
        f"{report['tables']} tables, {report['pictures']} pictures"
    )
    if report["metrics"]:
        for key, data in report["metrics"].items():
            pairs = ", ".join(
                f"{period}={value}" for period, value in zip(data["periods"], data["values"])
            )
            print(f"    {key}: {pairs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-process broker research reports.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/examples/data"),
        help="Directory containing report PDFs.",
    )
    args = parser.parse_args()

    out_dir = Path("scratch/broker_reports")
    converter = build_converter()
    reports = []
    for pdf_path in sorted(args.input.glob("*.pdf")):
        reports.append(process_report(converter, pdf_path, out_dir))
        _log.info("converted %s", pdf_path.name)

    print(f"\nProcessed {len(reports)} report(s) -> {out_dir}")
    for report in reports:
        print_summary(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
