"""chart_detection.py — PictureItem 이 차트류인지 판정하는 공용 모듈.

docling `DocumentPictureClassifier`(do_picture_classification=True 일 때 동작)가
각 PictureItem.annotations 에 `PictureClassificationData` 를 채운다. 그 top-1 클래스
이름으로 차트 여부를 판정한다.

`DocumentFigureClassifier` 는 16개 클래스를 내며, 그중 차트류는 아래 4종이다:
bar_chart / flow_chart / line_chart / pie_chart.
(그 외: bar_code, chemistry_*, icon, logo, map, other, qr_code, remote_sensing,
 screenshot, signature, stamp)
"""
from __future__ import annotations

from typing import Any

# DocumentFigureClassifier 가 내는 차트류 클래스. 모두 '_chart' 로 끝나므로
# 판정은 suffix 로 하되, 근거·가독성을 위해 명시 목록도 남긴다.
CHART_PICTURE_CLASSES = ("bar_chart", "flow_chart", "line_chart", "pie_chart")


def is_chart(item: Any) -> bool:
    """PictureItem 이 차트류로 분류됐는지 판정.

    PictureClassificationData.predicted_classes[0].class_name 이 '*_chart' 이면 차트로 본다.
    분류 annotation 이 없으면(do_picture_classification=False 등) False.
    """
    for ann in getattr(item, "annotations", []) or []:
        classes = getattr(ann, "predicted_classes", None)
        if classes and str(getattr(classes[0], "class_name", "")).endswith("_chart"):
            return True
    return False
