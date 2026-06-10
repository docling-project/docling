"""
TEDS-S (Tree-Edit-Distance based Similarity, structure-only) 지표.

표 인식 평가의 표준 지표 TEDS 의 구조 전용 변형. 두 표 HTML 을 노드 트리
(tag + colspan + rowspan, 셀 텍스트는 무시) 로 변환한 뒤 APTED 트리 편집거리를
계산해 다음 유사도를 반환한다.

    score = 1 - edit_distance / max(n_nodes_pred, n_nodes_gt)   in [0, 1]

셀 텍스트를 보지 않으므로 OCR 텍스트 인식 오류와 무관하게 "병합셀(span) 구조"만
평가한다. (참고: PubTabNet TEDS, structure_only=True)
"""
from __future__ import annotations

from typing import Optional

from bs4 import BeautifulSoup, Tag
from apted import APTED, Config
from apted.helpers import Tree


# 표 구조에 의미가 있는 태그만 노드로 사용한다.
_RELEVANT_TAGS = {"table", "thead", "tbody", "tfoot", "tr", "td", "th"}


class _TableNode(Tree):
    """APTED 노드. 표 구조 식별자(tag, colspan, rowspan)만 보유."""

    def __init__(self, tag: str, colspan: int, rowspan: int, *children: "_TableNode"):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.children = list(children)

    # APTED 가 자식 순회에 사용
    def __iter__(self):
        return iter(self.children)


class _TableConfig(Config):
    """구조 전용 비용 정책: (tag, colspan, rowspan) 일치 시 rename 0, 아니면 1."""

    def rename(self, node1: _TableNode, node2: _TableNode) -> int:
        if (
            node1.tag == node2.tag
            and node1.colspan == node2.colspan
            and node1.rowspan == node2.rowspan
        ):
            return 0
        return 1

    def children(self, node: _TableNode):
        return node.children


def _int_attr(tag: Tag, name: str) -> int:
    val = tag.get(name, "1")
    if isinstance(val, str) and val.isdigit():
        return int(val)
    return 1


def _build_tree(tag: Tag) -> _TableNode:
    children = []
    for child in tag.find_all(recursive=False):
        if isinstance(child, Tag) and child.name in _RELEVANT_TAGS:
            children.append(_build_tree(child))
    colspan = _int_attr(tag, "colspan") if tag.name in ("td", "th") else 1
    rowspan = _int_attr(tag, "rowspan") if tag.name in ("td", "th") else 1
    return _TableNode(tag.name, colspan, rowspan, *children)


def _count_nodes(node: _TableNode) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


def _parse_table(html: str) -> Optional[_TableNode]:
    if not html or not isinstance(html, str):
        return None
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not isinstance(table, Tag):
        return None
    return _build_tree(table)


def teds_s(pred_html: str, gt_html: str) -> float:
    """두 표 HTML 의 구조 전용 TEDS(TEDS-S) 점수 [0, 1].

    한쪽이라도 <table> 파싱에 실패하면 0.0 (구조 불일치로 간주).
    """
    pred_tree = _parse_table(pred_html)
    gt_tree = _parse_table(gt_html)
    if pred_tree is None or gt_tree is None:
        return 0.0

    n_pred = _count_nodes(pred_tree)
    n_gt = _count_nodes(gt_tree)
    denom = max(n_pred, n_gt)
    if denom == 0:
        return 1.0

    distance = APTED(pred_tree, gt_tree, _TableConfig()).compute_edit_distance()
    return 1.0 - distance / denom
