from dataclasses import dataclass, field
from typing import Optional, Union

from .enums import NumberingLevel, StyleAttributes


class UnkownNumberingLevel(Exception):
    def __init__(self, level_name: NumberingLevel):
        super().__init__(
            f"Level kind must be one of {NumberingLevel.__members__.values()}, not '{level_name}'."
        )


@dataclass
class HierarchicalHeader:
    index: Optional[int] = None
    level_toc: Optional[int] = None
    level_fontsize: Optional[int] = None
    style_attrs: list[StyleAttributes] = field(default_factory=list)
    level_latin: list[int] = field(default_factory=list)
    level_alpha: list[int] = field(default_factory=list)
    level_numerical: list[int] = field(default_factory=list)
    parent: Optional["HierarchicalHeader"] = None
    children: list["HierarchicalHeader"] = field(default_factory=list)
    doc_ref: Optional[str] = None
    text: Optional[str] = None

    def __post_init__(self):
        self._doc_ref_to_parent_doc_ref: dict[Union[str, None], Union[str, None]] = None

    def any_level(self) -> bool:
        return bool(self.level_alpha or self.level_alpha or self.level_numerical)

    def last_level_of_kind(
        self, kind: NumberingLevel
    ) -> tuple[list[int], Union["HierarchicalHeader", None]]:
        if kind not in NumberingLevel.__members__.values():
            raise UnkownNumberingLevel(kind)
        if self.parent:
            if last := getattr(self.parent, kind.value):
                return last, self.parent
            return self.parent.last_level_of_kind(kind)
        return [], None

    def string_repr(self, prefix: str = "") -> str:
        out_text = ""
        if self.text:
            out_text += prefix + self.text + "\n"
        for child in self.children:
            out_text += child.string_repr(prefix + "  ")
        return out_text

    def __str__(self) -> str:
        return self.string_repr()

    def _build_doc_ref_to_parent_doc_ref(
        self,
    ) -> dict[Union[str, None], Union[str, None]]:
        self._doc_ref_to_parent_doc_ref = {}
        for child in self.children:
            self._doc_ref_to_parent_doc_ref.update(
                child._build_doc_ref_to_parent_doc_ref()
            )
        if self.parent is not None and self.parent.doc_ref is not None:
            self._doc_ref_to_parent_doc_ref[self.doc_ref] = self.parent.doc_ref
        else:
            self._doc_ref_to_parent_doc_ref[self.doc_ref] = None
        return self._doc_ref_to_parent_doc_ref

    def get_parent_cid_of(self, doc_ref: str) -> Union[str, None]:
        if self._doc_ref_to_parent_doc_ref is None:
            self._build_doc_ref_to_parent_doc_ref()
        return self._doc_ref_to_parent_doc_ref[doc_ref]
