import time
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, List

import numpy as np
from pydantic import BaseModel

from docling.datamodel.settings import settings

if TYPE_CHECKING:
    from docling.datamodel.document import ConversionResult


class ProfilingScope(str, Enum):
    PAGE = "page"
    DOCUMENT = "document"


class ProfilingItem(BaseModel):
    scope: ProfilingScope
    count: int = 0
    times: List[float] = []
    start_timestamps: List[datetime] = []

    def total(self) -> float:
        return np.sum(self.times)  # type: ignore

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
            if key not in conv_res.timings:
                conv_res.timings[key] = ProfilingItem(scope=scope)
            self.conv_res = conv_res
            self.key = key

    def __enter__(self):
        if settings.debug.profile_pipeline_timings:
            self.start = time.monotonic()
            self.conv_res.timings[self.key].start_timestamps.append(
                datetime.now(timezone.utc)
            )
        return self

    def __exit__(self, *args):
        if settings.debug.profile_pipeline_timings:
            elapsed = time.monotonic() - self.start
            self.conv_res.timings[self.key].times.append(elapsed)
            self.conv_res.timings[self.key].count += 1


class TimeIntervalRecorder:
    r"""Accumulates several disjoint time intervals into a single timing sample"""

    def __init__(
        self,
        conv_res: "ConversionResult",
        key: str,
        scope: ProfilingScope = ProfilingScope.PAGE,
    ):
        self._enabled = settings.debug.profile_pipeline_timings
        if self._enabled:
            if key not in conv_res.timings:
                conv_res.timings[key] = ProfilingItem(scope=scope)
            self.conv_res = conv_res
            self.key = key
            self._total: float = 0.0
            self._running_since: float | None = None
            self._start_ts: datetime | None = None
            self._closed = False

    def resume(self) -> None:
        r"""Start (or restart) the clock for the next interval."""
        if not self._enabled:
            return
        if self._closed:
            raise RuntimeError("resume() called on a closed TimeIntervalRecorder")
        if self._running_since is not None:
            raise RuntimeError("resume() called while the recorder is already running")
        if self._start_ts is None:
            self._start_ts = datetime.now(timezone.utc)
        self._running_since = time.monotonic()

    def pause(self) -> None:
        r"""Stop the clock and bank the elapsed span into the running total."""
        if not self._enabled:
            return
        if self._closed:
            raise RuntimeError("pause() called on a closed TimeIntervalRecorder")
        if self._running_since is None:
            raise RuntimeError("pause() called without a matching resume()")
        self._total += time.monotonic() - self._running_since
        self._running_since = None

    def add(self, secs: float) -> None:
        r"""Bank a precomputed, non-negative duration into the running total."""
        if not self._enabled:
            return
        if self._closed:
            raise RuntimeError("add() called on a closed TimeIntervalRecorder")
        self._total += secs

    def close(self) -> None:
        r"""Flush the accumulated total as one sample and invalidate the recorder."""
        if not self._enabled:
            return
        if self._closed:
            raise RuntimeError(
                "close() called on an already closed TimeIntervalRecorder"
            )
        if self._running_since is not None:
            raise RuntimeError("close() called while the recorder is still running")
        item = self.conv_res.timings[self.key]
        item.start_timestamps.append(self._start_ts or datetime.now(timezone.utc))
        item.times.append(self._total)
        item.count += 1
        self._closed = True
