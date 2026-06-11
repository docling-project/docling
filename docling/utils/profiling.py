import logging
import time
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
        rows.append((key, item, total))

    if not rows:
        return

    rows.sort(key=lambda r: r[2], reverse=True)

    file_name = getattr(getattr(conv_res, "input", None), "file", "")
    header = f"[profiling] timing summary for {file_name}"
    lines = [
        header,
        f"{'stage':<28} {'scope':<9} {'count':>6} {'total(s)':>10} "
        f"{'avg(s)':>9} {'p95(s)':>9}",
    ]
    for key, item, total in rows:
        scope = item.scope.value if hasattr(item.scope, "value") else str(item.scope)
        p95 = item.percentile(95) if len(item.times) > 1 else item.times[0]
        lines.append(
            f"{key:<28} {scope:<9} {item.count:>6} {total:>10.3f} "
            f"{item.avg():>9.3f} {p95:>9.3f}"
        )
    logger.info("\n".join(lines))
