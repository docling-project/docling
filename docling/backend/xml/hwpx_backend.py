import logging
import re
import zipfile
from collections import defaultdict
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union
from xml.etree.ElementTree import Element

from lxml import etree
from PIL import Image, UnidentifiedImageError

try:
    from wand.image import Image as WandImage
    from wand.exceptions import WandException
    WAND_AVAILABLE = True
except ImportError:
    WAND_AVAILABLE = False

from docling_core.types.doc import (
    BoundingBox, DocItemLabel, DoclingDocument, DocumentOrigin,
    GroupLabel, ImageRef, ImageRefMode, NodeItem, ProvenanceItem,
    Size, TableCell, TableData,
)
from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


class HwpxDocumentBackend(DeclarativeDocumentBackend):
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[Path, BytesIO],
        save_images: bool = True,
        include_wmf: bool = False,
    ) -> None:
        """HWPX 파일(zip 아카이브)을 로드하는 구버전 XML 파싱 백엔드."""
        super().__init__(in_doc, path_or_stream)

        # include_wmf가 True이면 save_images도 자동으로 True
        self.save_images = save_images or include_wmf
        self.include_wmf = include_wmf

        self.zip = None
        self.valid = False

        # 계층 추적
        self.parents: dict[int, Optional[NodeItem]] = {}
        self.max_levels = 10
        for i in range(-1, self.max_levels):
            self.parents[i] = None
        self.current_section_group = None
        self.current_list_group = None
        self.current_list_item = None
        self._seen_section_texts: set[str] = set()
        self.list_stack: List[tuple[NodeItem, int]] = []
        self.current_indent: int = 0
        self._next_as_header = False

        try:
            if isinstance(path_or_stream, BytesIO):
                self.zip = zipfile.ZipFile(path_or_stream)
            elif isinstance(path_or_stream, Path):
                self.zip = zipfile.ZipFile(str(path_or_stream))
            if "Contents/section0.xml" in self.zip.namelist():
                self.valid = True
        except Exception as e:
            self.valid = False
            raise RuntimeError(f"Failed to open HWPX document: {e}")

    def _extract_text(self, elem: etree._Element) -> str:
        """hp:t 요소에서 tab, fwSpace를 공백으로 치환하면서 텍스트를 추출."""
        parts: List[str] = []
        if elem.text:
            parts.append(elem.text)
        for inline in elem:
            tag = etree.QName(inline).localname
            if tag in ("tab", "fwSpace", "linesegarray"):
                parts.append(" ")
            if inline.tail:
                parts.append(inline.tail)
        return "".join(parts).strip()

    def is_valid(self) -> bool:
        return self.valid

    @classmethod
    def supported_formats(cls) -> set:
        return {InputFormat.XML_HWPX}

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    def unload(self) -> None:
        if self.zip:
            self.zip.close()
            self.zip = None

    def _is_toc_numbered_entry(self, t_elem: etree._Element) -> bool:
        """
        숫자+점 패턴이긴 하지만,
        TOC 항목인지(탭 다음 페이지 번호가 붙어 있는지) 검사.
        예) <hp:t>3. 제목<hp:tab .../>9</hp:t>
        """
        tabs = t_elem.findall("hp:tab", namespaces=t_elem.nsmap)
        if not tabs:
            return False
        for tab in tabs:
            tail = (tab.tail or "").lstrip()
            if re.match(r"^\d+", tail):
                return True
        return False

    def _handle_list_symbol(self, txt: str, doc: DoclingDocument):
        SYMBOL_LEVEL = {
            '●': 0,
            'o': 1,
            '-': 2,
            '*': 2,
        }
        if not txt:
            return False
        sym = txt[0]
        if sym not in SYMBOL_LEVEL:
            return False

        level = SYMBOL_LEVEL[sym]

        while self.list_stack and self.list_stack[-1][1] >= level:
            self.list_stack.pop()

        parent_group = (
            self.list_stack[-1][0]
            if self.list_stack
            else self.current_section_group
        )
        new_group = doc.add_group(
            label=GroupLabel.LIST,
            name="ul",
            parent=parent_group
        )
        self.list_stack.append((new_group, level))

        doc.add_text(
            label=DocItemLabel.PARAGRAPH,
            text=txt,
            parent=new_group,
            prov=ProvenanceItem(
                page_no=1,
                bbox=BoundingBox(l=0, t=0, r=1, b=1),
                charspan=(0, len(txt))
            )
        )
        return True

    def _extract_page_size(self) -> tuple[float, float]:
        """section0.xml의 hp:pagePr에서 페이지 크기 추출."""
        try:
            section_xml = self.zip.read("Contents/section0.xml")
            section_root = etree.fromstring(section_xml)
            page_pr = section_root.find(".//hp:pagePr", namespaces=section_root.nsmap)
            if page_pr is not None:
                width_str = page_pr.get("width", "59528")
                height_str = page_pr.get("height", "84188")
                hwp_to_points = 0.0178 * 2.83465
                width = float(width_str) * hwp_to_points
                height = float(height_str) * hwp_to_points
                return width, height
            else:
                return 595.0, 842.0
        except Exception:
            return 595.0, 842.0

    def _get_image_ref(self, pic_elem: etree._Element) -> Optional[ImageRef]:
        """binaryItemIDRef를 읽어 이미지 바이트를 가져오고 ImageRef로 래핑."""
        img_node = pic_elem.find("hc:img", namespaces=pic_elem.nsmap)
        if img_node is None:
            return None
        bin_id = img_node.get("binaryItemIDRef")
        if not bin_id:
            return None

        img_bytes = self._read_image_bytes(bin_id)
        if not img_bytes:
            return None

        try:
            pil_img = Image.open(BytesIO(img_bytes))
        except (UnidentifiedImageError, OSError) as e:
            logging.debug(f"PIL failed to open image: {e}")
            return None

        try:
            return ImageRef.from_pil(image=pil_img, dpi=72)
        except (UnidentifiedImageError, OSError) as e:
            logging.debug(f"Failed to create ImageRef: {e}")
            return None

    def convert(self) -> DoclingDocument:
        """HWPX 파일을 DoclingDocument 구조로 파싱."""
        if not self.is_valid():
            raise RuntimeError("Invalid or unsupported HWPX document")

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/zip",
            binary_hash=self.document_hash
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        page_width, page_height = self._extract_page_size()
        doc.pages[1] = doc.add_page(page_no=1, size=Size(width=page_width, height=page_height))

        root_group = doc.add_group(parent=None, label=GroupLabel.SECTION, name="root")
        self.parents[0] = root_group
        self.current_section_group = root_group

        section_index = 0
        while True:
            section_path = f"Contents/section{section_index}.xml"
            if section_path not in self.zip.namelist():
                break
            section_xml = self.zip.read(section_path)
            section_root = etree.fromstring(section_xml)
            for elem in section_root:
                if not isinstance(elem, etree._Element):
                    continue
                tag_name = etree.QName(elem).localname
                if tag_name == "p":
                    self._process_paragraph(elem, doc)
            section_index += 1

        self._end_list()
        return doc

    def _process_paragraph(self, p_elem: etree._Element, doc: DoclingDocument) -> None:
        # (0) secPr만 있고 텍스트가 없는 메타데이터 문단은 스킵
        has_secPr = p_elem.find(".//hp:secPr", namespaces=p_elem.nsmap) is not None
        has_text = p_elem.find(".//hp:run/hp:t", namespaces=p_elem.nsmap) is not None
        if has_secPr and not has_text:
            return

        header_found = False
        header_level = None
        header_text = None

        parents = [etree.QName(x).localname for x in p_elem.iterancestors()]
        runs = p_elem.findall("./hp:run", namespaces=p_elem.nsmap)

        valid_runs: list[etree._Element] = []
        run_texts: dict[int, str] = {}
        for run in runs:
            t_tag = run.find(".//hp:t", namespaces=run.nsmap)
            if t_tag is None:
                continue
            parts = [
                self._extract_text(t0)
                for t0 in run.findall(".//hp:t", namespaces=run.nsmap)
            ]
            full = " ".join(parts).strip()
            valid_runs.append(run)
            run_texts[len(valid_runs) - 1] = full

        any_header_added = False
        header_runs: set[int] = set()

        for idx, run in enumerate(valid_runs):
            header_text = None
            header_level = None
            norm_text = None

            for child in run:
                tag = etree.QName(child).localname

                if tag == "tbl" and not self._is_toc_numbered_entry(child):
                    rc = child.get("rowCnt")
                    rows = int(rc) if rc is not None else len(child.findall("hp:tr", namespaces=child.nsmap))
                    cc = child.get("colCnt")
                    cols = int(cc) if cc is not None else len(
                        child.find("hp:tr", namespaces=child.nsmap)
                             .findall("hp:tc", namespaces=child.nsmap)
                    )
                    if (rows, cols) in [(1, 1), (1, 2), (1, 3)]:
                        parts = [
                            self._extract_text(t0)
                            for t0 in child.findall(".//hp:t", namespaces=child.nsmap)
                        ]
                        txt = " ".join(parts).strip()
                        norm = "".join(txt.split())
                        if txt and len(txt) <= 200 and norm not in self._seen_section_texts:
                            header_text = txt
                            header_level = 1
                            norm_text = norm
                            break

                elif tag == "rect":
                    draw_txt = child.find(".//hp:drawText", namespaces=child.nsmap)
                    if draw_txt is None:
                        break
                    parts = [
                        self._extract_text(t0)
                        for t0 in draw_txt.findall(".//hp:t", namespaces=draw_txt.nsmap)
                    ]
                    full_txt = "".join(parts).strip()
                    norm = "".join(full_txt.split())
                    if not full_txt:
                        continue
                    if len(full_txt) <= 200 and norm not in self._seen_section_texts:
                        header_text = full_txt
                        header_level = 1
                        norm_text = norm
                        p_elem.set("_was_rect_header", "true")
                        break

            if header_text is not None:
                self._seen_section_texts.add(norm_text)
                self._end_list()
                self._add_header(doc, header_level, header_text)
                self.current_section_group = self.parents[header_level]
                any_header_added = True
                header_runs.add(idx)

        if any_header_added:
            for idx, text in run_texts.items():
                if idx not in header_runs and text:
                    doc.add_text(
                        label=DocItemLabel.PARAGRAPH,
                        text=text,
                        parent=self.current_section_group,
                        prov=ProvenanceItem(
                            page_no=1,
                            bbox=BoundingBox(l=0, t=0, r=1, b=1),
                            charspan=(0, len(text))
                        )
                    )
            return

        for anc in p_elem.iterancestors():
            if etree.QName(anc).localname == "drawText":
                return

        full_para = " ".join(
            self._extract_text(t)
            for run in p_elem.findall("hp:run", namespaces=p_elem.nsmap)
            for t in run.findall("hp:t", namespaces=p_elem.nsmap)
        )

        toc_candidate = False
        for tab_elem in p_elem.findall(".//hp:tab", namespaces=p_elem.nsmap):
            if re.search(r"\d+\s*$", full_para):
                toc_candidate = True
                break

        for anc in p_elem.iterancestors():
            if etree.QName(anc).localname == "drawText":
                return

        if not toc_candidate and re.match(
            r'^(?:\d+\.\s+|[①②③④⑤⑥⑦⑧⑨⑩]+\.\s*)', full_para.strip()
        ):
            header_found = True
            header_level = 1
            header_text = full_para

        if header_found:
            self._seen_section_texts.add("".join(header_text.split()))
            self._end_list()
            self._add_header(doc, header_level, header_text)
            self.current_section_group = self.parents[header_level]
            return

        if "tc" in parents:
            runs = p_elem.findall("hp:run", namespaces=p_elem.nsmap)
            inlines = []
            for ri, run in enumerate(runs):
                for inline in run:
                    inlines.append((ri, inline))

            nested_idx = next(
                (i for i, (_, elem) in enumerate(inlines)
                 if etree.QName(elem).localname == "tbl"),
                None
            )

            if nested_idx is not None:
                parent_node = self.current_list_item or self.current_section_group

                for i, (ri, elem) in enumerate(inlines[:nested_idx]):
                    tag = etree.QName(elem).localname
                    if tag == "t":
                        txt = self._extract_text(elem).strip()
                        if not txt and not self._is_toc_numbered_entry(elem):
                            continue
                        norm = "".join(full_para.split())
                        final_text = full_para
                        if re.match(r'^(?:\d+|[①②③④⑤⑥⑦⑧⑨⑩]+)\.\s+', final_text):
                            self._seen_section_texts.add(norm)
                            self._end_list()
                            level = 1
                            self._add_header(doc, level, final_text)
                            self.current_section_group = self.parents[level]
                            continue
                        if txt.startswith("<참고"):
                            doc.add_text(
                                label=DocItemLabel.PARAGRAPH,
                                text=txt,
                                parent=self.current_section_group,
                                prov=ProvenanceItem(
                                    page_no=1,
                                    bbox=BoundingBox(l=0, t=0, r=1, b=1),
                                    charspan=(0, len(txt))
                                )
                            )
                        if self._handle_list_symbol(full_para, doc):
                            return
                        else:
                            self._end_list()
                            doc.add_text(
                                label=DocItemLabel.PARAGRAPH,
                                text=txt,
                                parent=parent_node,
                                prov=ProvenanceItem(
                                    page_no=1,
                                    bbox=BoundingBox(l=0, t=0, r=1, b=1),
                                    charspan=(0, len(txt))
                                )
                            )
                    elif tag == "pic":
                        self._process_picture(elem, doc)
                    elif tag == "rect":
                        self._process_rect(elem, doc)
                    elif tag == "equation":
                        self._process_equation(elem, doc)

                _, tbl_elem = inlines[nested_idx]
                self._process_table(tbl_elem, doc)

                for j, (ri, elem) in enumerate(inlines[nested_idx + 1:], start=nested_idx + 1):
                    tag = etree.QName(elem).localname
                    if tag == "t":
                        txt = self._extract_text(elem).strip()
                        if txt:
                            doc.add_text(
                                label=DocItemLabel.PARAGRAPH,
                                text=txt,
                                parent=parent_node,
                                prov=ProvenanceItem(
                                    page_no=1,
                                    bbox=BoundingBox(l=0, t=0, r=1, b=1),
                                    charspan=(0, len(txt))
                                )
                            )
                    elif tag == "pic":
                        self._process_picture(elem, doc)
                    elif tag == "rect":
                        self._process_rect(elem, doc)
                    elif tag == "equation":
                        self._process_equation(elem, doc)

                if self.current_list_group and self.current_list_item is None:
                    self._end_list()

                return

        parent_node = self.current_list_item or self.current_section_group
        text_buffer = ""
        runs = p_elem.findall(".//hp:run", namespaces=p_elem.nsmap)

        children = []
        for run in runs:
            children.extend(list(run))

        seen = set()
        i = 0
        while i < len(children):
            child = children[i]
            cid = id(child)
            i += 1

            if cid in seen:
                continue
            seen.add(cid)

            tag = etree.QName(child).localname
            if tag == "t":
                text_buffer += (child.text or "")
                for inline in child:
                    if etree.QName(inline).localname in ("tab", "fwSpace", "lineBreak"):
                        text_buffer += " "
                    if inline.tail:
                        text_buffer += inline.tail

            if tag == "tbl":
                if text_buffer.strip():
                    doc.add_text(
                        label=DocItemLabel.PARAGRAPH,
                        text=text_buffer.rstrip(),
                        parent=parent_node,
                        prov=ProvenanceItem(
                            page_no=1,
                            bbox=BoundingBox(l=0, t=0, r=1, b=1),
                            charspan=(0, len(text_buffer.rstrip()))
                        )
                    )
                    text_buffer = ""
                self._process_table(child, doc)
                for desc in child.iter():
                    seen.add(id(desc))
                continue

            elif tag == "rect":
                if text_buffer.strip():
                    doc.add_text(
                        label=DocItemLabel.PARAGRAPH,
                        text=text_buffer.rstrip(),
                        parent=parent_node,
                        prov=ProvenanceItem(
                            page_no=1,
                            bbox=BoundingBox(l=0, t=0, r=1, b=1),
                            charspan=(0, len(text_buffer.rstrip()))
                        )
                    )
                    text_buffer = ""
                self._process_rect(child, doc)
                if child.tail:
                    text_buffer += child.tail

            elif tag == "pic":
                if text_buffer.strip():
                    doc.add_text(
                        label=DocItemLabel.PARAGRAPH,
                        text=text_buffer.rstrip(),
                        parent=parent_node,
                        prov=ProvenanceItem(
                            page_no=1,
                            bbox=BoundingBox(l=0, t=0, r=1, b=1),
                            charspan=(0, len(text_buffer.rstrip()))
                        )
                    )
                    text_buffer = ""
                self._process_picture(child, doc)
                if child.tail:
                    text_buffer += child.tail

            elif tag == "equation":
                self._process_equation(child, doc)
                if child.tail:
                    text_buffer += child.tail

        final_text = text_buffer.rstrip()
        full_text = final_text

        if full_text.startswith("<참고"):
            doc.add_text(
                label=DocItemLabel.PARAGRAPH,
                text=full_text,
                parent=self.current_section_group,
                prov=ProvenanceItem(
                    page_no=1,
                    bbox=BoundingBox(l=0, t=0, r=1, b=1),
                    charspan=(0, len(full_text))
                )
            )
            return

        if self._handle_list_symbol(full_text, doc):
            return

        if final_text:
            norm = "".join(final_text.split())
            if re.match(r'^(?:\d+|[①②③④⑤⑥⑦⑧⑨⑩]+)\.\s+', final_text):
                self._seen_section_texts.add(norm)
                self._end_list()
                level = 1
                self._add_header(doc, level, final_text)
                self.current_section_group = self.parents[level]
                return
            doc.add_text(
                label=DocItemLabel.PARAGRAPH,
                text=final_text,
                parent=self.current_section_group,
                prov=ProvenanceItem(
                    page_no=1,
                    bbox=BoundingBox(l=0, t=0, r=1, b=1),
                    charspan=(0, len(final_text))
                )
            )

    def _process_table(self, tbl_elem: etree._Element, doc: DoclingDocument) -> None:
        """<hp:tbl> 요소를 파싱하여 TableData로 변환."""
        toc = False
        for t in tbl_elem.findall(".//hp:t", namespaces=tbl_elem.nsmap):
            if self._is_toc_numbered_entry(t):
                for p in tbl_elem.findall(".//hp:p", namespaces=tbl_elem.nsmap):
                    parts = []
                    for run in p.findall("hp:run", namespaces=p.nsmap):
                        t0 = run.find("hp:t", namespaces=run.nsmap)
                        if t0 is None:
                            continue
                        parts.append(self._extract_text(t0))
                    full = " ".join(parts).strip()
                    if full:
                        doc.add_text(
                            label=DocItemLabel.PARAGRAPH,
                            text=full,
                            parent=self.current_section_group,
                            prov=ProvenanceItem(
                                page_no=1,
                                bbox=BoundingBox(l=0, t=0, r=1, b=1),
                                charspan=(0, len(full))
                            )
                        )
                return

        try:
            num_rows = int(tbl_elem.get("rowCnt", "0"))
            num_cols = int(tbl_elem.get("colCnt", "0"))
        except ValueError:
            trs = tbl_elem.findall("hp:tr", namespaces=tbl_elem.nsmap)
            num_rows = len(trs)
            num_cols = len(trs[0].findall("hp:tc", namespaces=tbl_elem.nsmap)) if trs else 0

        parent = self.current_list_item or self.current_section_group

        if (num_rows, num_cols) == (1, 1):
            parts = [
                self._extract_text(t0)
                for t0 in tbl_elem.findall(".//hp:t", namespaces=tbl_elem.nsmap)
            ]
            txt = " ".join(parts).strip()
            has_pic = bool(tbl_elem.findall(".//hp:pic", namespaces=tbl_elem.nsmap))
            nested_tbl = len(tbl_elem.findall(".//hp:tbl", namespaces=tbl_elem.nsmap)) > 1

            if txt and has_pic and (len(txt) <= 50) and not nested_tbl:
                self._process_paragraph(tbl_elem, doc)
                return
            else:
                level = 1 if num_rows == 1 else 2
                norm = "".join(txt.split())
                if txt and (len(txt) <= 200) and norm != "공백":
                    self._seen_section_texts.add(norm)
                    self._end_list()
                    self._add_header(doc, level, txt)
                    self.current_section_group = self.parents[level]
                    return

        if (num_rows, num_cols) in [(1, 2), (1, 3)]:
            parts = [
                self._extract_text(t0)
                for t0 in tbl_elem.findall(".//hp:t", namespaces=tbl_elem.nsmap)
            ]
            txt = "".join(parts).strip()
            norm = "".join(txt.split())
            if txt and (len(txt) <= 200):
                self._seen_section_texts.add(norm)
                self._end_list()
                level = 1
                self._add_header(doc, level, txt)
                self.current_section_group = self.parents[level]
                return

        data = TableData(num_rows=num_rows, num_cols=num_cols)
        occupied = [[False] * num_cols for _ in range(num_rows)]

        cell_items = defaultdict(list)
        caption_map = {}
        skip_caption = set()
        skip_rows = set()
        rows = tbl_elem.findall("hp:tr", namespaces=tbl_elem.nsmap)
        has_top_title = False

        for r_idx, tr in enumerate(rows):
            tcs = tr.findall("hp:tc", namespaces=tbl_elem.nsmap)
            num_tcs_curr_row = len(tcs)

            for tc in tr.findall("hp:tc", namespaces=tbl_elem.nsmap):
                addr = tc.find("hp:cellAddr", namespaces=tc.nsmap)
                span = tc.find("hp:cellSpan", namespaces=tc.nsmap)
                if addr is None or span is None:
                    continue

                r = int(addr.get("rowAddr"))
                c = int(addr.get("colAddr"))
                rs = int(span.get("rowSpan"))
                cs = int(span.get("colSpan"))

                if occupied[r][c]:
                    continue
                for rr in range(r, r + rs):
                    for cc in range(c, c + cs):
                        occupied[rr][cc] = True

                if num_tcs_curr_row == 1 and r_idx + 1 < len(rows):
                    next_row_tcs = rows[r_idx + 1].findall("hp:tc", namespaces=tbl_elem.nsmap)
                    if len(next_row_tcs) >= 2:
                        next_has_pic = any(
                            tc2.findall(".//hp:pic", namespaces=tbl_elem.nsmap)
                            for tc2 in next_row_tcs
                        )
                        if next_has_pic:
                            cap_text = "".join(
                                self._extract_text(t0)
                                for t0 in tc.findall(".//hp:t", namespaces=tc.nsmap)
                            ).strip()
                            norm_cap = re.sub(r"\s+", "", cap_text)
                            if cap_text and norm_cap not in self._seen_section_texts:
                                self._seen_section_texts.add(norm_cap)
                                for tc2 in next_row_tcs:
                                    addr2 = tc2.find("hp:cellAddr", namespaces=tc2.nsmap)
                                    if addr2 is None:
                                        continue
                                    r2 = int(addr2.get("rowAddr"))
                                    c2 = int(addr2.get("colAddr"))
                                    cell_items[(r2, c2)].append(('caption', cap_text))
                            continue

                nested_in_this = bool(tc.findall(".//hp:tbl", namespaces=tc.nsmap))

                if (r, c) in skip_caption:
                    continue

                next_nested = False
                next_pic = False
                if r_idx + rs < len(rows):
                    for tc2 in rows[r_idx + rs].findall("hp:tc", namespaces=tbl_elem.nsmap):
                        addr2 = tc2.find("hp:cellAddr", namespaces=tc2.nsmap)
                        if addr2 is None:
                            continue
                        col2 = int(addr2.get("colAddr"))
                        if col2 == c:
                            if tc2.findall(".//hp:tbl", namespaces=tc2.nsmap):
                                next_nested = True
                            if tc2.findall(".//hp:pic", namespaces=tc2.nsmap):
                                next_pic = True

                if not nested_in_this and (next_nested or next_pic):
                    if 0 <= r_idx - 1 < len(rows):
                        prev_row = rows[r_idx - 1]
                        tc1_list = prev_row.findall("hp:tc", namespaces=tbl_elem.nsmap)
                        cell_texts = [
                            "".join(tc.itertext()).strip()
                            for tc in tc1_list
                        ]
                        if cell_texts and len(set(cell_texts)) == 1:
                            toptitle = cell_texts[0]
                            if not re.match(r"^\s*(?:(?:주|자료)\s*[:：]|\*)", toptitle):
                                norm_toptitle = re.sub(r"\s+", "", toptitle)
                                if norm_toptitle not in self._seen_section_texts:
                                    cell_items[(r - 1, c)].append(('top_caption', toptitle))
                                    skip_caption.add((r - 1, c))
                                    skip_rows.add(r - 1)
                                    has_top_title = True

                    title = "".join(
                        self._extract_text(t) for t in tc.findall(".//hp:t", namespaces=tc.nsmap)
                    ).strip()
                    cell_items[(r, c)].append(('caption', title))
                    continue

                if nested_in_this and not toc:
                    for p in tc.findall("./hp:subList/hp:p", namespaces=tc.nsmap):
                        tbl = p.find(".//hp:tbl", namespaces=p.nsmap)
                        if tbl is not None:
                            cell_items[(r, c)].append(('table', tbl))
                        else:
                            cell_items[(r, c)].append(('paragraph', p))
                    continue

                pics = tc.findall(".//hp:pic", namespaces=tc.nsmap)
                if pics:
                    for p in tc.findall("./hp:subList/hp:p", namespaces=tc.nsmap):
                        t_elem = p.find(".//hp:t", namespaces=p.nsmap)
                        pic_elem = p.find(".//hp:pic", namespaces=p.nsmap)
                        if t_elem is not None and self._extract_text(t_elem).strip():
                            cell_items[(r, c)].append(('paragraph', p))
                        if pic_elem is not None:
                            img = self._get_image_ref(pic_elem)
                            cap_node = caption_map.get((r, c))
                            cell_items[(r, c)].append(('picture', (img, cap_node)))
                    continue

                texts = [
                    "".join(self._extract_text(t) for t in p.findall(".//hp:t", namespaces=tc.nsmap)).strip()
                    for p in tc.findall(".//hp:p", namespaces=tc.nsmap)
                ]
                txt = " ".join(filter(None, texts)).strip()

                if re.match(r"^\s*(?:(?:주|자료)\s*[:：]|\*)", txt):
                    prev_row_tcs = (
                        rows[r_idx - 1].findall("hp:tc", namespaces=tbl_elem.nsmap)
                        if (r_idx - 1) >= 0 else []
                    )
                    if num_tcs_curr_row == 1 and len(prev_row_tcs) >= 2:
                        prev_has_pic = any(
                            p_tc.findall(".//hp:pic", namespaces=tbl_elem.nsmap)
                            for p_tc in prev_row_tcs
                        )
                        if prev_has_pic:
                            addr = tc.find("hp:cellAddr", namespaces=tc.nsmap)
                            span = tc.find("hp:cellSpan", namespaces=tc.nsmap)
                            r_cur = int(addr.get("rowAddr"))
                            c_cur = int(addr.get("colAddr"))
                            cs = int(span.get("colSpan"))
                            if cs > 1:
                                for offset in range(1, 2):
                                    target_col = c_cur + offset
                                    cell_items.setdefault((r_cur, target_col), []).append(('comment', txt))
                            cell_items.setdefault((r_cur, c_cur), []).append(('comment', txt))
                            continue
                    cell_items.setdefault((r, c), []).append(('comment', txt))
                    continue

                parts = []
                for p in tc.findall(".//hp:p", namespaces=tc.nsmap):
                    for t in p.findall(".//hp:t", namespaces=p.nsmap):
                        parts.append(self._extract_text(t))
                cell_text = "\n".join(parts).strip()

                if len(cell_text) > 200:
                    for sub_p in tc.findall(".//hp:p", namespaces=tc.nsmap):
                        cell_items[(r, c)].append(('paragraph', sub_p))
                    continue

                data.table_cells.append(
                    TableCell(
                        text=cell_text,
                        row_span=rs,
                        col_span=cs,
                        start_row_offset_idx=r,
                        end_row_offset_idx=r + rs,
                        start_col_offset_idx=c,
                        end_col_offset_idx=c + cs,
                        column_header=(r == 0),
                        row_header=False,
                    )
                )

        has_table = any(
            typ == 'table'
            for items in cell_items.values()
            for typ, _ in items
        )
        has_picture = any(
            typ == 'picture'
            for (row_idx, col_idx), items in cell_items.items()
            if col_idx == c
            for typ, _ in items
        )
        has_comment = any(
            typ == 'comment'
            for items in cell_items.values()
            for typ, _ in items
        )

        if (not has_table and has_comment and not has_picture
                and not nested_in_this and not toc):
            is_empty = not any(cell.text for cell in data.table_cells)
            if not is_empty:
                copied_cells = deepcopy(data.table_cells)
                temp_data = TableData(num_rows=data.num_rows, num_cols=data.num_cols)
                temp_data.table_cells = copied_cells
                doc.add_table(
                    data=temp_data,
                    parent=parent,
                    prov=ProvenanceItem(
                        page_no=1,
                        bbox=BoundingBox(l=0, t=0, r=1, b=1),
                        charspan=(0, 0)
                    )
                )
                data.table_cells.clear()

                for items in cell_items.values():
                    for typ, txt in items:
                        if typ == 'comment':
                            doc.add_text(
                                label=DocItemLabel.CAPTION,
                                text=txt,
                                parent=parent,
                                prov=ProvenanceItem(
                                    page_no=1,
                                    bbox=BoundingBox(l=0, t=0, r=1, b=1),
                                    charspan=(0, len(txt))
                                )
                            )

                for (r, c), items in list(cell_items.items()):
                    new_items = [(typ, payload) for (typ, payload) in items if typ != 'comment']
                    if new_items:
                        cell_items[(r, c)] = new_items
                    else:
                        del cell_items[(r, c)]

        sorted_coords = sorted(cell_items.keys(), key=lambda x: (x[1], x[0]))
        for r, c in sorted_coords:
            for typ, payload in cell_items[(r, c)]:
                if typ == 'top_caption':
                    norm_payload = re.sub(r"\s+", "", payload)
                    if norm_payload in self._seen_section_texts:
                        continue
                    doc.add_text(
                        label=DocItemLabel.PARAGRAPH,
                        text=payload,
                        parent=self.current_section_group,
                        prov=ProvenanceItem(
                            page_no=1,
                            bbox=BoundingBox(l=0, t=0, r=1, b=1),
                            charspan=(0, len(payload))
                        )
                    )
                elif typ == 'caption':
                    parent = self.current_section_group
                    norm = "".join(payload.split())
                    if re.match(r'^(?:\d+\.\s+|[①②③④⑤⑥⑦⑧⑨⑩]+\.\s*)', payload):
                        self._seen_section_texts.add(norm)
                        self._end_list()
                        level = 1
                        self._add_header(doc, level, payload)
                        self.current_section_group = self.parents[level]
                        continue
                    doc.add_text(
                        label=DocItemLabel.PARAGRAPH,
                        text=payload,
                        parent=parent,
                        prov=ProvenanceItem(
                            page_no=1,
                            bbox=BoundingBox(l=0, t=0, r=1, b=1),
                            charspan=(0, len(payload))
                        )
                    )
                elif typ == 'paragraph':
                    self._process_paragraph(payload, doc)
                elif typ == 'table':
                    self._process_table(payload, doc)
                elif typ == 'picture':
                    if not self.save_images:
                        continue
                    img, cap = payload
                    if img is None:
                        continue
                    doc.add_picture(
                        parent=parent,
                        image=img,
                        caption=cap,
                        prov=ProvenanceItem(
                            page_no=1,
                            bbox=BoundingBox(l=0, t=0, r=1, b=1),
                            charspan=(0, 0)
                        )
                    )
                elif typ == 'comment':
                    doc.add_text(
                        label=DocItemLabel.CAPTION,
                        text=payload,
                        parent=parent,
                        prov=ProvenanceItem(
                            page_no=1,
                            bbox=BoundingBox(l=0, t=0, r=1, b=1),
                            charspan=(0, len(payload))
                        )
                    )

        is_empty_tbl = True
        for i in data.table_cells:
            if i.text:
                is_empty_tbl = False
                break
        if is_empty_tbl or has_top_title:
            return

        parent = self.current_section_group
        doc.add_table(
            data=data,
            parent=parent,
            prov=ProvenanceItem(
                page_no=1,
                bbox=BoundingBox(l=0, t=0, r=1, b=1),
                charspan=(0, 0)
            )
        )

    def _process_rect(self, rect_elem: etree._Element, doc: DoclingDocument) -> None:
        """최상위 <hp:rect> 요소(텍스트 박스) 처리."""
        draw_text_elem = rect_elem.find(".//hp:drawText", namespaces=rect_elem.nsmap)
        if draw_text_elem is None:
            return

        text_parts = [
            t.text for t in draw_text_elem.findall(".//hp:t", namespaces=draw_text_elem.nsmap)
            if t.text
        ]
        full_text = "".join(text_parts).strip()
        norm_text = "".join(full_text.split())

        if not full_text:
            return

        if len(full_text) <= 100:
            if not hasattr(self, "_seen_section_texts"):
                self._seen_section_texts = set()
            self._seen_section_texts.add(norm_text)
            self._end_list()
            self._add_header(doc, 1, full_text)
            self.current_section_group = self.parents[1]
            return
        else:
            for p in draw_text_elem.findall(".//hp:p", namespaces=draw_text_elem.nsmap):
                self._process_paragraph(p, doc)

    def _convert_wmf_to_png(self, wmf_bytes: bytes) -> Optional[bytes]:
        """Wand를 사용해 WMF → PNG 변환 (가능한 경우)."""
        if not WAND_AVAILABLE:
            return None
        try:
            import sys
            original_stderr = sys.stderr
            with open(os.devnull, 'w') as devnull:
                sys.stderr = devnull
                try:
                    with WandImage(blob=wmf_bytes) as wand_img:
                        wand_img.format = 'png'
                        png_bytes = wand_img.make_blob()
                    sys.stderr = original_stderr
                    return png_bytes
                except Exception as wand_e:
                    sys.stderr = original_stderr
                    raise wand_e
        except Exception as e:
            logging.warning(f"WMF 변환 실패 (변환이 생략됨): {e}")
            return None

    def _read_image_bytes(self, bin_id: str) -> Optional[bytes]:
        """BinData에서 이미지 바이트 읽기 (필요시 WMF 변환)."""
        if self.include_wmf:
            extensions = (".bmp", ".png", ".jpg", ".jpeg", ".wmf", ".tif")
        else:
            extensions = (".bmp", ".png", ".jpg", ".jpeg", ".tif")

        for ext in extensions:
            try:
                data = self.zip.read(f"BinData/{bin_id}{ext}")
            except KeyError:
                continue
            if ext == ".wmf":
                converted = self._convert_wmf_to_png(data)
                return converted if converted else None
            return data
        return None

    def _process_picture(
        self,
        pic_elem: etree._Element,
        doc: DoclingDocument,
        caption: Optional[str] = None,
    ) -> None:
        """<hp:pic> 요소를 처리하여 이미지 노드 추가."""
        if not self.save_images:
            return

        parent_node = self.current_list_item or self.current_section_group

        img_ref = pic_elem.find("hc:img", namespaces=pic_elem.nsmap)
        if img_ref is None:
            return
        bin_id = img_ref.get("binaryItemIDRef")
        if not bin_id:
            return

        image_bytes = self._read_image_bytes(bin_id)
        if not image_bytes:
            return

        try:
            pil_image = Image.open(BytesIO(image_bytes))
        except (UnidentifiedImageError, OSError) as e:
            logging.debug(f"PIL failed to open image: {e}")
            return

        try:
            img_ref_obj = ImageRef.from_pil(image=pil_image, dpi=72)
        except (UnidentifiedImageError, OSError) as e:
            logging.debug(f"Failed to create ImageRef: {e}")
            return

        doc.add_picture(
            parent=parent_node,
            image=img_ref_obj,
            caption=caption,
            prov=ProvenanceItem(
                page_no=1,
                bbox=BoundingBox(l=0, t=0, r=1, b=1),
                charspan=(0, 0),
            ),
        )

    def _process_equation(self, eq_elem: etree._Element, doc: DoclingDocument) -> None:
        """<hp:equation> 요소를 수식 텍스트로 추가."""
        parent_node = self.current_list_item or self.current_section_group or None
        formula_text = "".join(eq_elem.itertext()).strip()
        doc.add_text(
            label=DocItemLabel.FORMULA,
            text=formula_text,
            parent=parent_node,
            prov=ProvenanceItem(
                page_no=1,
                bbox=BoundingBox(l=0, t=0, r=1, b=1),
                charspan=(0, len(formula_text))
            )
        )

    def _add_header(self, doc: DoclingDocument, level: int, text: str) -> None:
        """지정된 레벨의 헤딩 노드를 추가."""
        curr_level = level

        for lvl in range(0, curr_level):
            if self.parents.get(lvl) is None:
                self.parents[lvl] = doc.add_group(
                    parent=self.parents[lvl - 1] if lvl - 1 >= 0 else None,
                    label=GroupLabel.SECTION,
                    name=f"header-{lvl}"
                )

        for lvl in range(curr_level, self.max_levels):
            self.parents[lvl] = None

        parent_node = self.parents[curr_level - 1] if curr_level - 1 >= 0 else None
        self.parents[curr_level] = doc.add_heading(
            parent=parent_node,
            text=text,
            level=curr_level,
            prov=ProvenanceItem(
                page_no=1,
                bbox=BoundingBox(l=0, t=0, r=1, b=1),
                charspan=(0, len(text))
            )
        )

    def _end_list(self) -> None:
        """현재 리스트 그룹 종료."""
        self.current_list_group = None
        self.current_list_item = None


import os
