# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""
List Item Marker Processor for Docling Documents

This module provides a rule-based model to identify list item markers and
merge marker-only TextItems with their content to create proper ListItems.
"""

import logging
import re
from typing import Optional, Union

from docling_core.types.doc.document import (
    DoclingDocument,
    ListItem,
    NodeItem,
    ProvenanceItem,
    RefItem,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel

_log = logging.getLogger(__name__)


class ListItemMarkerProcessor:
    """
    A rule-based processor for identifying and processing list item markers.

    This class can:
    1. Identify various list item markers (bullets, numbers, letters)
    2. Detect marker-only TextItems followed by content TextItems
    3. Merge them into proper ListItems
    4. Group consecutive ListItems into appropriate list containers
    """

    def __init__(self, infer_enumerated: bool = True):
        """Initialize the processor with marker patterns."""
        self._infer_enumerated = infer_enumerated
        # Bullet markers (unordered lists)
        self._bullet_patterns = [
            r"[\u2022\u2023\u25E6\u2043\u204C\u204D\u2219\u25AA\u25AB\u25CF\u25CB]",  # Various bullet symbols
            r"[-*+•·‣⁃]",  # noqa: RUF001 - intentional Unicode bullet glyphs
            r"[►▶▸‣➤➢]",  # Arrow-like bullets
            r"[✓✔✗✘]",  # Checkmark bullets
        ]

        # Numbered markers (ordered lists). Compound/hierarchical markers are listed
        # first: they are the more specific patterns, and matching is first-wins.
        self._numbered_patterns = [
            r"\d+(?:\.\d+)+\.?",  # 1.1  1.2.3  1.1. (dotted decimal outline)
            r"\d+\.?[a-zA-Z]\.",  # 9a. 3.a.
            r"\d+\.?[a-zA-Z]\)",  # 9a) 3.a)
            r"\(\d+\.?[a-zA-Z]\)",  # (9a) (3.a)
            r"\d+\.",  # 1. 2. 3.
            r"\d+\)",  # 1) 2) 3)
            r"\(\d+\)",  # (1) (2) (3)
            r"\[\d+\]",  # [1] [2] [3]
            r"[ivxlcdm]+\.",  # i. ii. iii. (Roman numerals lowercase)
            r"[IVXLCDM]+\.",  # I. II. III. (Roman numerals uppercase)
            r"[a-z]\.",  # a. b. c.
            r"[A-Z]\.",  # A. B. C.
            r"[a-z]\)",  # a) b) c)
            r"[A-Z]\)",  # A) B) C)
        ]

        # Compile all patterns
        self._compiled_bullet_patterns = [
            re.compile(f"^{pattern}$") for pattern in self._bullet_patterns
        ]
        self._compiled_numbered_patterns = [
            re.compile(f"^{pattern}$") for pattern in self._numbered_patterns
        ]

        # re.DOTALL so the content group spans the whole item: page backends emit
        # newlines inside a single item's text, and without it the text would be
        # silently cut at the first one.
        self._compiled_bullet_item_patterns = [
            re.compile(f"^({pattern})" + r"\s(.+)", re.DOTALL)
            for pattern in self._bullet_patterns
        ]
        self._compiled_numbered_item_patterns = [
            re.compile(f"^({pattern})" + r"\s(.+)", re.DOTALL)
            for pattern in self._numbered_patterns
        ]

    def _is_bullet_marker(self, text: str) -> bool:
        """Check if text is a bullet marker."""
        text = text.strip()
        return any(pattern.match(text) for pattern in self._compiled_bullet_patterns)

    def _is_numbered_marker(self, text: str) -> bool:
        """Check if text is a numbered marker."""
        text = text.strip()
        return any(pattern.match(text) for pattern in self._compiled_numbered_patterns)

    def _is_bullet_item(self, text: str) -> bool:
        return any(
            pattern.match(text) for pattern in self._compiled_bullet_item_patterns
        )

    def _is_numbered_item(self, text: str) -> bool:
        return any(
            pattern.match(text) for pattern in self._compiled_numbered_item_patterns
        )

    @classmethod
    def _create_list_item(
        cls,
        self_ref,
        marker: str,
        text: str,
        orig: str,
        prov: list[ProvenanceItem],
        enumerated: Optional[bool] = None,
    ) -> ListItem:
        item = ListItem(
            self_ref=self_ref,
            marker=marker,
            text=text,
            orig=orig,
            prov=prov,
        )
        if enumerated is not None:
            item.enumerated = enumerated
        return item

    def _find_match(
        self, text: str, patterns: list[re.Pattern]
    ) -> Optional[re.Match[str]]:
        for pattern in patterns:
            mtch = pattern.match(text)
            if mtch:
                return mtch
        return None

    def _find_marker_content_pairs(self, doc: DoclingDocument):
        """
        Find pairs of marker-only TextItems and their content TextItems.

        Returns:
            List of (marker_item, content_item) tuples. content_item can be None
            if the marker item already contains content.
        """
        self._matched_items: dict[
            int, tuple[RefItem, bool, bool]
        ] = {}  # index to (self_ref, is_pure_marker, is_enumerated)
        self._other: dict[int, RefItem] = {}  # index to self_ref

        for i, (item, _) in enumerate(doc.iterate_items(with_groups=False)):
            if not isinstance(item, TextItem):
                continue

            if self._is_bullet_marker(item.orig):
                self._matched_items[i] = (item.get_ref(), True, False)
            elif self._is_numbered_marker(item.orig):
                self._matched_items[i] = (item.get_ref(), True, True)
            elif self._is_bullet_item(item.orig):
                self._matched_items[i] = (item.get_ref(), False, False)
            elif self._is_numbered_item(item.orig):
                self._matched_items[i] = (item.get_ref(), False, True)
            else:
                self._other[i] = item.get_ref()

    def _group_consecutive_list_items(self, doc: DoclingDocument) -> DoclingDocument:
        """
        Might need to group list-items, not sure yet how...
        """
        return doc

    def process_list_item(self, item: ListItem) -> ListItem:
        """Process a ListItem to extract and update marker and text from bullet/numbered patterns.

        This method applies compiled regex patterns to match bullet point or numbered list
        formatting in the original text, then updates the ListItem's marker and text fields
        accordingly.

        Args:
            item (ListItem): The list item to process, containing original text that may
                           have bullet or numbered list formatting.

        Returns:
            ListItem: The same ListItem instance with updated marker and text fields
                     if a pattern match was found, otherwise unchanged.

        Note:
            The method modifies the input item in place when a pattern matches.
            If the item is not actually a ListItem type, a warning is logged.
        """

        is_enumerated: bool = False
        if mtch := self._find_match(
            text=item.orig, patterns=self._compiled_bullet_item_patterns
        ):
            is_enumerated = False
        elif mtch := self._find_match(
            text=item.orig, patterns=self._compiled_numbered_item_patterns
        ):
            is_enumerated = True

        if mtch:
            if isinstance(item, ListItem):  # update item in place
                item.marker = mtch[1]
                item.text = mtch[2]
                if self._infer_enumerated:
                    item.enumerated = is_enumerated
            else:
                _log.warning(
                    f"matching text for bullet_item_patterns that is not ListItem: {item.label}"
                )
        return item

    def process_text_item(self, item: TextItem) -> Union[TextItem, ListItem]:
        """Process a TextItem to detect and convert bullet/numbered list formatting.

        This method examines TextItem instances to determine if they contain bullet point
        or numbered list formatting. If detected and appropriate, it either updates an
        existing ListItem or converts the TextItem into a new ListItem.

        Args:
            item (TextItem): The text item to process, which may contain bullet or
                             numbered list formatting in its original text.

        Returns:
            Union[TextItem, ListItem]:
                - If item is already a ListItem: returns the updated ListItem
                - If item is a TextItem with list formatting (and not a section heading
                  or footnote): returns a new ListItem with extracted marker and text
                - Otherwise: returns the original TextItem unchanged

        Note:
            Section headings and footnotes are excluded from conversion to preserve
            their semantic meaning. A warning is logged if pattern matching occurs
            on unexpected item types.
        """
        is_enumerated: bool = False
        if mtch := self._find_match(
            text=item.orig, patterns=self._compiled_bullet_item_patterns
        ):
            is_enumerated = False
        elif mtch := self._find_match(
            text=item.orig, patterns=self._compiled_numbered_item_patterns
        ):
            is_enumerated = True

        if mtch:
            marker = mtch[1]
            text = mtch[2]

            if isinstance(item, ListItem):  # update item in place
                item.marker = marker
                item.text = text
                if self._infer_enumerated:
                    item.enumerated = is_enumerated

                return item
            elif isinstance(item, TextItem) and (
                item.label not in [DocItemLabel.SECTION_HEADER, DocItemLabel.FOOTNOTE]
            ):
                # Create new ListItem
                return self._create_list_item(
                    self_ref=item.get_ref().cref,
                    marker=marker,
                    text=text,
                    orig=item.orig,
                    prov=item.prov,
                    enumerated=is_enumerated if self._infer_enumerated else None,
                )
            else:
                _log.warning(
                    f"matching text for bullet_item_patterns that is not ListItem: {item.label}"
                )
        return item

    def update_list_items_in_place(
        self, doc: DoclingDocument, allow_textitem: bool = False
    ) -> DoclingDocument:
        for item, level in doc.iterate_items():
            if isinstance(item, ListItem):
                item = self.process_list_item(item)
            elif allow_textitem and isinstance(item, TextItem):
                item = self.process_text_item(item)

        return doc

    def merge_markers_and_text_items_into_list_items(
        self, doc: DoclingDocument
    ) -> DoclingDocument:

        # Find all marker-content pairs: this function will identify text-items
        # with a marker fused into the text
        self._find_marker_content_pairs(doc)

        # Accumulate items for post-loop deletion to avoid reference validity issues
        to_delete: list[NodeItem] = []

        # If you find a sole marker-item followed by a text, there are
        # good chances we need to merge them into a list-item. This
        # function is only necessary as long as the layout-model does not
        # recognize list-items properly
        for ind, (self_ref, is_marker, is_enumerated) in self._matched_items.items():
            if is_marker:
                marker_item = self_ref.resolve(doc=doc)

                if ind + 1 in self._other:
                    next_item = self._other[ind + 1].resolve(doc=doc)

                    if (isinstance(next_item, TextItem)) and (
                        next_item.label in [DocItemLabel.TEXT, DocItemLabel.LIST_ITEM]
                    ):
                        marker_text: str = marker_item.text
                        content_text: str = next_item.text
                        prov = marker_item.prov
                        prov.extend(next_item.prov)

                        list_item = self._create_list_item(
                            self_ref="#",
                            marker=marker_text,
                            text=content_text,
                            orig=f"{marker_text} {content_text}",
                            prov=prov,
                            enumerated=(
                                is_enumerated
                                if self._infer_enumerated
                                else (
                                    marker_item.enumerated
                                    if isinstance(marker_item, ListItem)
                                    else None
                                )
                            ),
                        )

                        # Insert the new ListItem
                        doc.insert_item_before_sibling(
                            new_item=list_item, sibling=marker_item
                        )

                        # Accumulate items to delete
                        to_delete.extend([marker_item, next_item])

        doc.delete_items(node_items=to_delete)

        return doc

    def process_document(
        self,
        doc: DoclingDocument,
        allow_textitem: bool = False,
        merge_items: bool = False,
    ) -> DoclingDocument:
        """
        Process the entire document to identify and convert list markers.

        Args:
            doc: The DoclingDocument to process

        Returns:
            The processed document (modified in-place)
        """
        doc = self.update_list_items_in_place(doc, allow_textitem=allow_textitem)

        if merge_items:
            doc = self.merge_markers_and_text_items_into_list_items(doc)

        # Group consecutive list items
        doc = self._group_consecutive_list_items(doc)

        return doc
