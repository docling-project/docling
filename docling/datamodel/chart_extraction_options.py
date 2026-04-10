from enum import Enum
from typing import Literal

from pydantic import BaseModel


class ChartExtractionModelKind(str, Enum):
    GRANITE_VISION = "granite-vision"
    GRANITE_VISION_V4 = "granite-vision-v4"


class ChartExtractionModelOptions(BaseModel):
    kind: Literal["chart_extraction"] = "chart_extraction"

    model: ChartExtractionModelKind = ChartExtractionModelKind.GRANITE_VISION_V4

    chart2csv: bool = True  # prompt <chart2csv>      Chart to CSV with table with headers and numeric values
    chart2code: bool = (
        False  # prompt <chart2code>  Chart to Python code that recreates the chart
    )
    chart2summary: bool = False  # prompt <chart2summary>  Chart to summary with natural-language description of the chart
