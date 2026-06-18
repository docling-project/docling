"""Plot process memory metrics collected by iterate_pdf_pages.py."""

import argparse
import json
from pathlib import Path
from typing import Any

MEMORY_KEYS = ("rss_mb", "uss_mb", "pss_mb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics_file",
        type=Path,
        nargs="?",
        default=Path("memory-metrics.jsonl"),
        help="JSONL metrics file produced by iterate_pdf_pages.py.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("memory-metrics.png"),
        help="Output plot path.",
    )
    return parser.parse_args()


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
    if not args.metrics_file.is_file():
        raise SystemExit(f"Metrics file does not exist: {args.metrics_file}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Plotting requires matplotlib. Install it in the environment first."
        ) from exc

    total_pages, loaded_metrics, after_metrics, doc_boundaries = load_points(
        args.metrics_file
    )
    if not total_pages:
        raise SystemExit(f"No plotable data found in {args.metrics_file}")

    point_count = min(
        [len(total_pages)]
        + [len(values) for values in loaded_metrics.values()]
        + [len(values) for values in after_metrics.values()]
    )
    x_values = total_pages[:point_count]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    plot_specs = [
        (axes[0], loaded_metrics, "Memory at page loaded"),
        (axes[1], after_metrics, "Memory after page unload"),
    ]

    for ax, metrics, title in plot_specs:
        for key in MEMORY_KEYS:
            series = metrics[key][:point_count]
            if all(value != value for value in series):
                continue
            ax.plot(
                x_values,
                series,
                label=key.replace("_mb", "").upper(),
                linewidth=1.5,
            )
        for boundary in doc_boundaries:
            ax.axvline(
                boundary,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )
        ax.set_ylabel("Memory (MiB)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[1].set_xlabel("Total page number")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
