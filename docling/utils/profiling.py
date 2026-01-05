import time
from datetime import datetime
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
