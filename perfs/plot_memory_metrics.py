"""Plot process memory metrics collected by iterate_pdf_pages.py."""

import argparse
import json
from pathlib import Path
from typing import Any

MEMORY_KEYS = ("rss_mb", "uss_mb", "pss_mb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        type=Path,
        nargs="?",
        default=Path("memory-metrics.jsonl"),
        help=(
            "JSONL metrics file or JSON summary report produced by "
            "iterate_pdf_pages.py."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("memory-metrics.png"),
        help="Output plot path.",
    )
    return parser.parse_args()


def _resolve_metrics_files(input_file: Path) -> list[tuple[str, Path]]:
    if input_file.suffix.lower() == ".jsonl":
        return [(input_file.stem, input_file)]

    try:
        report = json.loads(input_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{input_file} is not valid JSONL or a JSON summary report."
        ) from exc

    runs = report.get("runs")
    if not isinstance(runs, list):
        raise SystemExit(
            f"{input_file} does not look like an iterate_pdf_pages.py summary report."
        )

    metrics_files: list[tuple[str, Path]] = []
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            continue
        metrics_file = run.get("metrics_file")
        if not isinstance(metrics_file, str):
            continue
        thread_count = run.get("threads")
        label = (
            f"threads={thread_count}"
            if isinstance(thread_count, int)
            else f"run {index}"
        )
        metrics_path = Path(metrics_file)
        if not metrics_path.is_absolute():
            metrics_path = input_file.parent / metrics_path
        metrics_files.append((label, metrics_path))

    if not metrics_files:
        raise SystemExit(f"No metrics files found in summary report: {input_file}")
    return metrics_files


def _empty_series() -> dict[str, list[float]]:
    return {key: [] for key in MEMORY_KEYS}


def _append_memory_values(
    target: dict[str, list[float]],
    memory_payload: dict[str, Any] | None,
) -> None:
    for key in MEMORY_KEYS:
        value = None if memory_payload is None else memory_payload.get(key)
        if isinstance(value, int | float):
            target[key].append(float(value))
        else:
            target[key].append(float("nan"))


def load_points(
    metrics_file: Path,
) -> tuple[list[int], dict[str, list[float]], dict[str, list[float]], list[int]]:
    total_pages: list[int] = []
    loaded_metrics = _empty_series()
    after_metrics = _empty_series()
    doc_boundaries: list[int] = []
    previous_pdf: str | None = None

    with metrics_file.open(encoding="utf-8") as fp:
        for line in fp:
            payload = json.loads(line)
            event = payload.get("event")
            total_page_no = payload.get("total_page_no")
            if not isinstance(total_page_no, int):
                continue

            if event == "loaded":
                memory_loaded = payload.get("memory_loaded_mb")
                if isinstance(memory_loaded, dict):
                    pdf_name = payload.get("pdf")
                    if (
                        isinstance(pdf_name, str)
                        and previous_pdf is not None
                        and pdf_name != previous_pdf
                    ):
                        doc_boundaries.append(total_page_no)
                    if isinstance(pdf_name, str):
                        previous_pdf = pdf_name
                    total_pages.append(total_page_no)
                    _append_memory_values(loaded_metrics, memory_loaded)
            elif event == "after_unload":
                memory_after = payload.get("memory_after_mb")
                if isinstance(memory_after, dict):
                    _append_memory_values(after_metrics, memory_after)

    return total_pages, loaded_metrics, after_metrics, doc_boundaries


def main() -> None:
    args = parse_args()
    if not args.input_file.is_file():
        raise SystemExit(f"Input file does not exist: {args.input_file}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Plotting requires matplotlib. Install it in the environment first."
        ) from exc

    loaded_runs = []
    for label, metrics_file in _resolve_metrics_files(args.input_file):
        if not metrics_file.is_file():
            raise SystemExit(f"Metrics file does not exist: {metrics_file}")
        total_pages, loaded_metrics, after_metrics, doc_boundaries = load_points(
            metrics_file
        )
        if total_pages:
            loaded_runs.append(
                (label, total_pages, loaded_metrics, after_metrics, doc_boundaries)
            )

    if not loaded_runs:
        raise SystemExit(f"No plotable data found in {args.input_file}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for (
        label,
        total_pages,
        loaded_metrics,
        _after_metrics,
        _doc_boundaries,
    ) in loaded_runs:
        point_count = min(
            [len(total_pages)] + [len(values) for values in loaded_metrics.values()]
        )
        x_values = total_pages[:point_count]

        for key in MEMORY_KEYS:
            series = loaded_metrics[key][:point_count]
            if all(value != value for value in series):
                continue
            metric_label = key.replace("_mb", "").upper()
            ax.plot(
                x_values,
                series,
                label=f"{label} {metric_label}",
                linewidth=1.5,
            )

    ax.set_xlabel("Total page number")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title("Memory at page loaded")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
