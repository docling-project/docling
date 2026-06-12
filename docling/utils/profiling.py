import logging
import time
import unicodedata
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, List

import numpy as np
from pydantic import BaseModel

from docling.datamodel.settings import settings

if TYPE_CHECKING:
    from docling.datamodel.document import ConversionResult

_log = logging.getLogger(__name__)


class ProfilingScope(str, Enum):
    PAGE = "page"
    DOCUMENT = "document"


class ProfilingItem(BaseModel):
    scope: ProfilingScope
    count: int = 0
    times: List[float] = []
    start_timestamps: List[datetime] = []

    def avg(self) -> float:
        return np.average(self.times)  # type: ignore

    def std(self) -> float:
        return np.std(self.times)  # type: ignore

    def mean(self) -> float:
        return np.mean(self.times)  # type: ignore

    def percentile(self, perc: float) -> float:
        return np.percentile(self.times, perc)  # type: ignore

    def wall_clock(self) -> float:
        """Document-level (wall-clock) time covered by this stage.

        Computed as the union of the recorded ``[start, start + elapsed]``
        intervals. For sequential stages the intervals don't overlap so this
        equals ``sum(times)``; for concurrent stages (e.g. the per-page layout
        threads) overlapping intervals collapse into the true elapsed span, so
        ``total(s)`` (a sum inflated by concurrency) is converted into the real
        wall-clock contribution. Falls back to ``sum(times)`` when start
        timestamps are unavailable/mismatched.

        Note: ``datetime.timestamp()`` on naive timestamps applies the same
        local-time offset to every interval of a stage, so it cancels out in
        the relative-overlap math.
        """
        if not self.times:
            return 0.0
        if len(self.start_timestamps) != len(self.times):
            return float(np.sum(self.times))
        intervals = sorted(
            (ts.timestamp(), ts.timestamp() + d)
            for ts, d in zip(self.start_timestamps, self.times)
        )
        union = 0.0
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start > cur_end:
                union += cur_end - cur_start
                cur_start, cur_end = start, end
            else:
                cur_end = max(cur_end, end)
        return union + (cur_end - cur_start)


class TimeRecorder:
    def __init__(
        self,
        conv_res: "ConversionResult",
        key: str,
        scope: ProfilingScope = ProfilingScope.PAGE,
    ):
        if settings.debug.profile_pipeline_timings:
            if key not in conv_res.timings.keys():
                conv_res.timings[key] = ProfilingItem(scope=scope)
            self.conv_res = conv_res
            self.key = key

    def __enter__(self):
        if settings.debug.profile_pipeline_timings:
            self.start = time.monotonic()
            self.conv_res.timings[self.key].start_timestamps.append(datetime.utcnow())
        return self

    def __exit__(self, *args):
        if settings.debug.profile_pipeline_timings:
            elapsed = time.monotonic() - self.start
            self.conv_res.timings[self.key].times.append(elapsed)
            self.conv_res.timings[self.key].count += 1


# Declared parent/child hierarchy of pipeline stages, in display order. Used to
# render a non-overlapping flow tree where each parent's children (plus a
# derived "(기타)" residual) sum to the parent's wall-clock — unlike the flat
# table where nested stages (e.g. layout ⊃ vlm_call) double-count. Keys absent
# from timings are skipped; keys present but not listed here surface under
# "(트리 외)" so nothing is silently dropped. Update this when stages change.
_FLOW_TREE = [
    ("pipeline_total", None),
    ("doc_build", "pipeline_total"),
    ("page_init", "doc_build"),
    ("page_parse", "doc_build"),
    ("ocr", "doc_build"),  # OCR engine stage (easyocr/paddle/tesseract/...)
    ("layout", "doc_build"),  # standard docling layout path
    ("table_structure", "doc_build"),  # TableFormer / VLM table structure
    ("vlm", "doc_build"),  # VLM pipeline path
    ("dotsocr_layout_wallclock", "doc_build"),  # GENOS/dotsocr layout path
    ("dotsocr_vlm_call", "dotsocr_layout_wallclock"),
    ("dotsocr_postprocess", "dotsocr_layout_wallclock"),
    ("dotsocr_parse", "dotsocr_layout_wallclock"),
    ("dotsocr_table_build", "dotsocr_layout_wallclock"),
    ("page_assemble", "doc_build"),
    ("doc_assemble", "pipeline_total"),
    ("doc_assemble_page_images", "doc_assemble"),
    ("doc_assemble_element_images", "doc_assemble"),
    ("reading_order", "doc_assemble"),
    ("reading_order_bypass", "doc_assemble"),
    ("doc_enrich", "pipeline_total"),
]


def _render_flow_tree(wall_by_key: dict, pipeline_wall: float) -> List[str]:
    """Render stage wall-clock times as a non-overlapping hierarchy.

    Children plus a derived ``(기타)`` residual sum to their parent, so reading
    one level gives a clean breakdown without the flat table's nested
    double-counting. Stages in ``wall_by_key`` but not in ``_FLOW_TREE`` are
    listed under ``(트리 외)``.
    """
    parent_of = {k: p for k, p in _FLOW_TREE}

    # On the dotsocr path, ``layout`` (per-page sum) duplicates
    # ``dotsocr_layout_wallclock``; drop it so the stage isn't counted twice.
    ignore = set()
    if "dotsocr_layout_wallclock" in wall_by_key and "layout" in wall_by_key:
        ignore.add("layout")

    present = [k for k, _ in _FLOW_TREE if k in wall_by_key and k not in ignore]
    present_set = set(present)

    children: dict = {}
    for key in present:
        children.setdefault(parent_of[key], []).append(key)

    def fmt(label: str, wall: float, indent: str, connector: str) -> str:
        pct = (100.0 * wall / pipeline_wall) if pipeline_wall > 0 else 0.0
        head = f"{indent}{connector}{label} "
        # East-Asian wide/fullwidth chars occupy 2 terminal columns; count
        # display width so the dot leader aligns the value column.
        width = sum(
            2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in head
        )
        dots = "." * max(2, 44 - width)
        return f"{head}{dots} {wall:>7.2f}s {pct:>5.1f}%"

    lines: List[str] = []

    def walk(key: str, indent: str, is_last: bool) -> None:
        wall = wall_by_key[key]
        is_root = parent_of[key] is None
        connector = "" if is_root else ("└─ " if is_last else "├─ ")
        lines.append(fmt(key, wall, indent, connector))

        child_indent = indent if is_root else indent + ("   " if is_last else "│  ")
        kids = children.get(key, [])
        residual = wall - sum(wall_by_key[c] for c in kids)
        show_residual = bool(kids) and residual > 0.05 and (
            pipeline_wall <= 0 or residual > 0.001 * pipeline_wall
        )
        total_kids = len(kids) + (1 if show_residual else 0)
        for i, child in enumerate(kids):
            walk(child, child_indent, i == total_kids - 1)
        if show_residual:
            lines.append(fmt("(기타)", residual, child_indent, "└─ "))

    roots = [k for k, p in _FLOW_TREE if p is None and k in present_set]
    for i, root in enumerate(roots):
        walk(root, "", i == len(roots) - 1)

    extras = [k for k in wall_by_key if k not in present_set and k not in ignore]
    if extras:
        lines.append("(트리 외)")
        for key in sorted(extras, key=lambda k: wall_by_key[k], reverse=True):
            lines.append(fmt(key, wall_by_key[key], "  ", ""))

    return lines


def log_profiling_summary(
    conv_res: "ConversionResult", logger: logging.Logger = _log
) -> None:
    """Log a per-stage timing breakdown for one converted document.

    No-op unless ``settings.debug.profile_pipeline_timings`` is enabled (set
    ``DOCLING_DEBUG__PROFILE_PIPELINE_TIMINGS=true``). Stages are ordered by
    total elapsed time so the dominant bottleneck appears first.
    """
    if not settings.debug.profile_pipeline_timings:
        return

    timings = getattr(conv_res, "timings", None)
    if not timings:
        return

    rows = []
    for key, item in timings.items():
        if not item.times:
            continue
        total = float(np.sum(item.times))
        wall = item.wall_clock()
        rows.append((key, item, total, wall))

    if not rows:
        return

    # Sort by wall-clock (document-level) so the real bottleneck appears first.
    rows.sort(key=lambda r: r[3], reverse=True)

    # Denominator for %pipe: prefer pipeline_total's wall-clock, else the
    # largest document-scoped wall-clock.
    pipeline_wall = next(
        (w for k, _i, _t, w in rows if k == "pipeline_total"), 0.0
    )
    if pipeline_wall <= 0:
        pipeline_wall = max(
            (w for _k, i, _t, w in rows if i.scope == ProfilingScope.DOCUMENT),
            default=0.0,
        )

    file_name = getattr(getattr(conv_res, "input", None), "file", "")
    header = f"[profiling] timing summary for {file_name}"
    lines = [
        header,
        "# wall(s)=동시 page 구간 합집합(실제 경과시간 추정), %pipe=pipeline_total 대비. "
        "중첩 스테이지는 합산 시 중복됨.",
        f"{'stage':<28} {'scope':<9} {'count':>6} {'total(s)':>10} "
        f"{'wall(s)':>10} {'%pipe':>6} {'avg(s)':>9} {'p95(s)':>9}",
    ]
    for key, item, total, wall in rows:
        scope = item.scope.value if hasattr(item.scope, "value") else str(item.scope)
        p95 = item.percentile(95) if len(item.times) > 1 else item.times[0]
        pct = f"{100.0 * wall / pipeline_wall:>5.1f}%" if pipeline_wall > 0 else "    -"
        lines.append(
            f"{key:<28} {scope:<9} {item.count:>6} {total:>10.3f} "
            f"{wall:>10.3f} {pct:>6} {item.avg():>9.3f} {p95:>9.3f}"
        )

    # Non-overlapping flow tree (children + residual sum to their parent).
    wall_by_key = {key: wall for key, _item, _total, wall in rows}
    lines.append("")
    lines.append("[profiling] flow (wall-clock, %pipe)")
    lines.extend(_render_flow_tree(wall_by_key, pipeline_wall))

    logger.info("\n".join(lines))
