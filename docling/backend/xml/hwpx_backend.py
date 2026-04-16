import logging

import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Union, List
from xml.etree.ElementTree import Element
from lxml import etree
from PIL import Image, UnidentifiedImageError

try:
    from wand.image import Image as WandImage
    from wand.exceptions import WandException
    WAND_AVAILABLE = True
except ImportError:
    WAND_AVAILABLE = False
from docling_core.types.doc import DocItemLabel, DoclingDocument, DocumentOrigin, GroupLabel, ImageRef, TableCell, TableData, NodeItem, ProvenanceItem, BoundingBox, Size
from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling_core.types.doc import ImageRefMode
import re
import logging
from collections import defaultdict
from copy import deepcopy


class HwpxDocumentBackend(DeclarativeDocumentBackend):
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[Path, BytesIO], save_images: bool = True, include_wmf: bool = False) -> None:
        """Initialize the HWPX backend by loading the .hwpx file (zip archive)."""
        super().__init__(in_doc, path_or_stream)

        # include_wmf가 True이면 save_images도 자동으로 True
        self.save_images = save_images or include_wmf
        self.include_wmf = include_wmf

        # print(f'save_images: {self.save_images}, include_wmf: {self.include_wmf}')
        self.zip = None
        self.valid = False
        # Hierarchy tracking for section (heading) groups and list items
        self.parents: dict[int, Optional[NodeItem]] = {}
        self.max_levels = 10
        for i in range(-1, self.max_levels):
            self.parents[i] = None
        self.current_section_group = None
        self.current_list_group = None
        self.current_list_item = None
        self._seen_section_texts: set[str] = set()
        # 리스트 중첩 관리를 위한 스택 [(NodeItem, indent_level), …]
        self.list_stack: List[tuple[NodeItem, int]] = []
        self.current_indent: int = 0
        self._next_as_header = False # '참고'다음 문장을 SECTION-HEADER로
        # Open the HWPX zip file
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
        """hp:t 요소에서 tab, fwSpace를 공백으로 치환하면서 텍스트를 뽑아냄"""
        parts: List[str] = []
        if elem.text:
            parts.append(elem.text)
        for inline in elem:
            tag = etree.QName(inline).localname
            if tag in ("tab", "fwSpace","linesegarray"):
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
        ◇ TOC 항목인지(탭 뒤에 페이지 번호가 붙어 있는지) 검사.
        ex)
        <hp:t>3. 제목<hp:tab .../>9</hp:t>
        """
        # (1) t_elem 바로 아래에 hp:tab 이 있는지
        tabs = t_elem.findall("hp:tab", namespaces=t_elem.nsmap)
        if not tabs:
            return False

        # (2) 각 tab 의 tail 이 숫자로 시작하는지
        for tab in tabs:
            tail = (tab.tail or "").lstrip()
            if re.match(r"^\d+", tail):
                return True

        return False

    def _handle_list_symbol(self, txt: str, doc: DoclingDocument):

        # 1) 심볼 → 레벨 매핑
        SYMBOL_LEVEL = {
            '□': 0,
            'o': 1,
            '-': 2,
            '*': 2,
        }
        # 텍스트에서 맨 앞 심볼 추출
        if not txt:
            return False
        sym = txt[0]
        if sym not in SYMBOL_LEVEL:
            return False

        level = SYMBOL_LEVEL[sym]

        # 2) 스택에서 현재 레벨보다 크거나 같은 항목 팝
        while self.list_stack and self.list_stack[-1][1] >= level:
            self.list_stack.pop()

        # 3) 새 리스트 그룹 생성 (부모는 스택 최상단 그룹 또는 section)
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
        # 스택에 (그룹, 레벨) 푸시
        self.list_stack.append((new_group, level))

        # 4) 리스트 아이템 추가
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
        """Extract page size from section0.xml hp:pagePr element."""
        try:
            # Read section0.xml to get page properties
            section_xml = self.zip.read("Contents/section0.xml")
            section_root = etree.fromstring(section_xml)

            # Find hp:pagePr element
            page_pr = section_root.find(".//hp:pagePr", namespaces=section_root.nsmap)
            if page_pr is not None:
                # Get width and height attributes
                width_str = page_pr.get("width", "59528")  # Default HWPX width
                height_str = page_pr.get("height", "84188")  # Default HWPX height

                # Convert HWPUNIT to points (1 HWPUNIT ≈ 0.0178 mm ≈ 0.0506 points)
                hwp_to_points = 0.0178 * 2.83465  # HWPUNIT to points conversion

                width = float(width_str) * hwp_to_points
                height = float(height_str) * hwp_to_points

                return width, height
            else:
                # Fallback to A4 size if hp:pagePr not found
                return 595.0, 842.0

        except Exception as e:
            # Fallback to A4 size on any error
            # logging.warning(f"Failed to extract page size from HWPX: {e}")
            return 595.0, 842.0

    def _get_image_ref(self, pic_elem: etree._Element) -> Optional[ImageRef]:
        """ hc:img 태그에서 binaryItemIDRef 읽기 """
        img_ref = pic_elem.find("hc:img", namespaces=pic_elem.nsmap)
        if img_ref is None:
            return None
        bin_id = img_ref.get("binaryItemIDRef")
        if not bin_id:
            return None

        # Zip에서 .bmp 포함 모든 확장자로 시도
        for ext in (".bmp", ".png", ".jpg", ".jpeg"):
            try:
                img_bytes = self.zip.read(f"BinData/{bin_id}{ext}")
            except KeyError:
                continue
            # 읽었다면 PIL로 열고
            try:
                pil_img = Image.open(BytesIO(img_bytes))
            except (UnidentifiedImageError, OSError):
                return None
            # ImageRef.from_pil로 포장해 리턴
            return ImageRef.from_pil(image=pil_img, dpi=72)
        return None
    def _get_image_ref(self, pic_elem: etree._Element) -> Optional[ImageRef]:
        """Extract binaryItemIDRef, read bytes, and wrap as ImageRef."""
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
        """Parses the HWPX file into a DoclingDocument structure."""
        if not self.is_valid():
            raise RuntimeError("Invalid or unsupported HWPX document")
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/zip",
            binary_hash=self.document_hash
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        # Extract page size from section0.xml
        page_width, page_height = self._extract_page_size()
        # Add page for prov values using actual page size from XML
        doc.pages[1] = doc.add_page(page_no=1, size=Size(width=page_width, height=page_height))

        # Create a root group as the top-level parent for all content
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
                # elif tag_name == "tbl":
                #     self._process_table(elem, doc)
                # elif tag_name == "rect":
                #     self._process_rect(elem, doc)
                # elif tag_name == "pic":
                #     self._process_picture(elem, doc)
                # elif tag_name == "equation":
                #     self._process_equation(elem, doc)
            section_index += 1
        # Close any open list group at end of document
        self._end_list()
        return doc

    def _process_paragraph(self, p_elem: etree._Element, doc: DoclingDocument) -> None:
        # ── (0) secPr 전용 문단: hp:secPr은 있지만 hp:t(text)는 전혀 없으면 "메타데이터" 이므로 스킵
        has_secPr = p_elem.find(".//hp:secPr", namespaces=p_elem.nsmap) is not None
        has_text = p_elem.find(".//hp:run/hp:t", namespaces=p_elem.nsmap) is not None
        if has_secPr and not has_text:
            return
        header_found = False
        header_level = None
        header_text  = None
        # ── (0.5) 테이블 셀 내부면 A-타입 숫자·점 헤더 감지를 건너뛴다 ──
        parents = [etree.QName(x).localname for x in p_elem.iterancestors()]
        # 1) 바로 아래 hp:run들만 골라낸다
        runs = p_elem.findall("./hp:run", namespaces=p_elem.nsmap)

        # 유효한 런(valid_runs)과, 런별 <hp:t> 합친 텍스트(run_texts)를 미리 저장
        valid_runs: list[etree._Element] = []
        run_texts: dict[int, str] = {}
        for run in runs:
            t_tag = run.find(".//hp:t", namespaces=run.nsmap)
            if t_tag is None:
                continue
            # 이 런 안의 모든 <hp:t>를 합친 문자열
            parts = [ self._extract_text(t0)
                      for t0 in run.findall(".//hp:t", namespaces=run.nsmap) ]
            full = " ".join(parts).strip()
            valid_runs.append(run)
            run_texts[len(valid_runs)-1] = full

        any_header_added = False
        header_runs: set[int] = set()

        # 2) valid_runs 순회하며 "런별로 헤더를 감지 → add_header"
        for idx, run in enumerate(valid_runs):
            header_text  = None
            header_level = None
            norm_text    = None

            for child in run:
                tag = etree.QName(child).localname

                # 2.a) 작은 표(tbl) 헤더 감지
                if tag == "tbl" and not self._is_toc_numbered_entry(child):
                    rc = child.get("rowCnt")
                    rows = int(rc) if rc is not None else len(child.findall("hp:tr", namespaces=child.nsmap))
                    cc = child.get("colCnt")
                    cols = int(cc) if cc is not None else len(
                        child.find("hp:tr", namespaces=child.nsmap)
                             .findall("hp:tc", namespaces=child.nsmap)
                    )
                    if (rows, cols) in [(1,1), (1,2), (1,3)]:
                        parts = [ self._extract_text(t0)
                                  for t0 in child.findall(".//hp:t", namespaces=child.nsmap) ]
                        txt  = " ".join(parts).strip()
                        norm = "".join(txt.split())
                        if txt and len(txt) <= 200 and norm not in self._seen_section_texts:
                            header_text  = txt
                            header_level = 1
                            norm_text    = norm
                            break

                # 2.b) 도형(rect) 안 텍스트 헤더 감지
                elif tag == "rect":
                    draw_txt = child.find(".//hp:drawText", namespaces=child.nsmap)
                    if draw_txt is None:
                        break
                    parts = [ self._extract_text(t0)
                              for t0 in draw_txt.findall(".//hp:t", namespaces=draw_txt.nsmap) ]
                    full_txt = "".join(parts).strip()
                    norm     = "".join(full_txt.split())
                    if not full_txt:
                        continue
                    if len(full_txt) <= 200 and norm not in self._seen_section_texts:
                        header_text  = full_txt
                        header_level = 1
                        norm_text    = norm
                        # 내부 drawText가 p_elem로 재진입되는 것을 막기 위한 표시
                        p_elem.set("_was_rect_header", "true")
                        break

            if header_text is not None:
                self._seen_section_texts.add(norm_text)
                self._end_list()
                self._add_header(doc, header_level, header_text)
                self.current_section_group = self.parents[header_level]
                any_header_added = True
                header_runs.add(idx)
                # 더 남은 런도 계속 확인

        # 3) 한 번이라도 헤더를 추가했다면 헤더가 아닌 런들의 텍스트를 본문으로 추가하고 리턴
        if any_header_added:
            for idx, text in run_texts.items():
                # 헤더로 쓰이지 않은 런(text)만 add_text
                if idx not in header_runs and text:
                    doc.add_text(
                        label=DocItemLabel.PARAGRAPH,
                        text=text,
                        parent=self.current_section_group,
                        prov=ProvenanceItem(
                            page_no=1,
                            bbox=BoundingBox(l=0, t=0, r=1, b=1),
                            charspan=(0, len(txt))
                        )
                    )
            return

        for anc in p_elem.iterancestors():
            if etree.QName(anc).localname == "drawText":
                return

        full_para = " ".join(
            self._extract_text(t)
            for run in p_elem.findall("hp:run", namespaces=p_elem.nsmap)
            for t   in run.findall("hp:t", namespaces=p_elem.nsmap)
        )

        # ── "문단 어딘가에 탭+숫자"면 TOC로 간주 ──
        toc_candidate = False
        for tab_elem in p_elem.findall(".//hp:tab", namespaces=p_elem.nsmap):
            if re.search(r"\d+\s*$", full_para):
                toc_candidate = True
                break

        # if not toc_candidate and re.match(r'^(?:\d+\.\s+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s*)', full_para.strip()):
        #     norm = "".join(full_para.split())
        #     if norm not in self._seen_section_texts:
        #         self._seen_section_texts.add(norm)
        #         self._end_list()
        #         self._add_header(doc, 1, full_para)
        #         self.current_section_group = self.parents[1]
        #         return


        # 문단이 <hp:rect> 내부에서 이미 헤더 처리되었으면 여기서 더 처리하지 않음
        for anc in p_elem.iterancestors():
            if etree.QName(anc).localname == "drawText":
                return


        # 정규표현식 헤더:
        #    - 숫자+점 뒤에 반드시 공백: r'^\d+\.\s+'
        #    - 로마자+점 뒤에는 공백 없어도 O: r'^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s*'
        if not toc_candidate and re.match(r'^(?:\d+\.\s+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s*)', full_para.strip()):
            header_found  = True
            header_level  = 1
            header_text   = full_para

        if header_found:
            self._seen_section_texts.add("".join(header_text.split()))
            self._end_list()
            self._add_header(doc, header_level, header_text)
            self.current_section_group = self.parents[header_level]
            return

        # ── 2) 셀 내부(tc)이면서 중첩 테이블이 run 안에 있는 경우 ──
        if "tc" in parents:
            runs = p_elem.findall("hp:run", namespaces=p_elem.nsmap)

            # 2.1) 모든 run 안의 inline 요소(flatten) 수집
            inlines = []
            for ri, run in enumerate(runs):
                for inline in run:
                    inlines.append((ri, inline))
            # 2.2) 첫 번째 중첩 <hp:tbl> 위치 찾기
            nested_idx = next(
                (i for i, (_, elem) in enumerate(inlines)
                if etree.QName(elem).localname == "tbl"),
                None
            )

            if nested_idx is not None:
                parent_node = self.current_list_item or self.current_section_group
                # ── 2.3) pre-content: nested 테이블 이전의 inlines 처리 ──
                for i, (ri, elem) in enumerate(inlines[:nested_idx]):
                    tag = etree.QName(elem).localname
                    if tag == "t":
                        txt = self._extract_text(elem).strip()
                        if not txt and not self._is_toc_numbered_entry(elem):
                            continue
                        norm = "".join(final_text.split())
                        if re.match(r'^(?:\d+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)\.\s+', final_text):
                            self._seen_section_texts.add(norm)
                            self._end_list()
                            level = 1
                            self._add_header(doc, level, final_text)
                            self.current_section_group = self.parents[level]
                            continue
                        if txt.startswith("<참고"):
                            # 이전 섹션 헤더를 parent로 사용
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
                        if self._handle_list_symbol(full_text, doc):
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

                # ── 2.4) nested table 처리 ──
                _, tbl_elem = inlines[nested_idx]
                self._process_table(tbl_elem, doc)

                # ── 2.5) post-content: nested 이후의 inlines 처리 ──
                for j, (ri, elem) in enumerate(inlines[nested_idx+1:], start=nested_idx+1):
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

                # ── 2.6) 열려 있는 리스트 닫기 ──
                if self.current_list_group and self.current_list_item is None:
                    self._end_list()

                return

        # ── 4) 기본 본문 누적 ──
        parent_node = self.current_list_item or self.current_section_group
        text_buffer = ""
        # p_elem: <hp:p> element
        runs = p_elem.findall(".//hp:run", namespaces=p_elem.nsmap)

        # 모든 child 요소를 flat list로 모아두기
        children = []
        for run in runs:
            children.extend(list(run))
        seen = set()
        i = 0
        while i < len(children):
            child = children[i]
            cid = id(child)
            i += 1

            # 이미 처리한 child면 건너뛰기
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
                # 테이블 내부의 모든 요소 ID를 seen에 추가하여 스킵
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
        # ── 3) 리스트 감지 (셀 내부(tc) 아닌 경우) ──
        full_text = final_text
        if full_text.startswith("<참고"):
            # 이전 섹션 헤더를 parent로 사용
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
            if re.match(r'^(?:\d+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)\.\s+', final_text):
                self._seen_section_texts.add(norm)
                self._end_list()
                level = 1
                self._add_header(doc, level, final_text)
                self.current_section_group = self.parents[level]
                return
            doc.add_text(
                label=DocItemLabel.PARAGRAPH,
                text=final_text,
                # parent=parent_node
                parent=self.current_section_group,
                prov=ProvenanceItem(
                    page_no=1,
                    bbox=BoundingBox(l=0, t=0, r=1, b=1),
                    charspan=(0, len(final_text))
                )
            )


    def _process_table(self, tbl_elem: etree._Element, doc: DoclingDocument) -> None:
        """ Process a <hp:tbl> element and extract its content into a TableData object."""
        # 0) TOC 감지
        toc = False
        for t in tbl_elem.findall(".//hp:t", namespaces=tbl_elem.nsmap):
            if self._is_toc_numbered_entry(t):
                # → TOC 테이블이라면, 각 <hp:p> 에서 run 안의 텍스트를 합쳐서 plain paragraph로 추가
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

        # parent = self.current_list_item or self.current_section_group or None

        # 1) 크기 파싱
        try:
            num_rows = int(tbl_elem.get("rowCnt","0"))
            num_cols = int(tbl_elem.get("colCnt","0"))
        except ValueError:
            trs = tbl_elem.findall("hp:tr", namespaces=tbl_elem.nsmap)
            num_rows = len(trs)
            num_cols = len(trs[0].findall("hp:tc", namespaces=tbl_elem.nsmap)) if trs else 0
        parent = self.current_list_item or self.current_section_group

        # ── 특수 1×1 케이스: txt+pic 섞여있으면 헤더가 아니라 일반 테이블 분기 타기 ──
        if (num_rows, num_cols) == (1, 1):
            # (a) 셀 안의 텍스트 추출
            parts    = [ self._extract_text(t0)
                         for t0 in tbl_elem.findall(".//hp:t", namespaces=tbl_elem.nsmap) ] # .//없애지말기
            txt      = " ".join(parts).strip()
            # (b) 이미지 존재 여부
            has_pic  = bool(tbl_elem.findall(".//hp:pic", namespaces=tbl_elem.nsmap))
            # (c) 중첩 tbl(헤더로 처리되지 않게 하기 위해) 없음 확인
            nested_tbl = len(tbl_elem.findall(".//hp:tbl", namespaces=tbl_elem.nsmap)) > 1

            # 텍스트+이미지 둘 다 있고, 50자 이하, 중첩 tbl 없으면
            if txt and has_pic and (len(txt) <= 50) and not nested_tbl:
                parent = self.current_section_group
                self._process_paragraph(tbl_elem, doc)
                return
            else:
                # 그 외의 경우엔 기존 헤더 분기 그대로 수행
                # ── 1a) 작거나 단순한 표를 헤더로 간주 ──
                level = 1  if num_rows == 1 else 2
                norm = "".join(txt.split())
                if (txt
                    and (len(txt) <= 200)
                    and norm != "여백"
                ):
                    self._seen_section_texts.add(norm)
                    self._end_list()
                    self._add_header(doc, level, txt)
                    self.current_section_group = self.parents[level]
                    return
        # ── 1a) 작거나 단순한 표를 헤더로 간주 ──

        if (num_rows, num_cols) in [(1,2), (1,3)]:
            # 표 내부의 모든 텍스트 조합
            parts = [
                self._extract_text(t0)
                for t0 in tbl_elem.findall(".//hp:t", namespaces=tbl_elem.nsmap)
            ]
            txt = "".join(parts).strip()
            norm = "".join(txt.split())
            has_pic = any(
                etree.QName(e).localname in "pic"
                for e in tbl_elem.iter()
            )

            if txt and (len(txt) <= 200):
                self._seen_section_texts.add(norm)
                self._end_list()
                level = 1
                self._add_header(doc, level, txt)
                self.current_section_group = self.parents[level]
                return

        data = TableData(num_rows=num_rows, num_cols=num_cols)
        occupied = [[False]*num_cols for _ in range(num_rows)]

        # 2) 위치별 정보 수집
        cell_items = defaultdict(list)   # (r,c) -> [ ('caption', str), ('paragraph', Element), ('table', Element), … ]
        caption_map   = {}                      # (r,c) -> TextItem
        skip_caption  = set()                   # nested 처리된 셀 좌표를 담을 집합
        # toptitle이 먼저 table cell로 저장되는 것을 삭제하기 위한 집합
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

                # 중복 방지
                if occupied[r][c]:
                    continue
                for rr in range(r, r+rs):
                    for cc in range(c, c+cs):
                        occupied[rr][cc] = True

                # ── 다음 행에 tc가 2개 이상이고, 그중 하나 이상에 그림(pic)이 있으면,
                    # 현재 행(tc=1)의 caption을 다음 행의 각 열 위로 복제해서 붙이기 ──
                if num_tcs_curr_row == 1 and r_idx + 1 < len(rows):
                    next_row_tcs = rows[r_idx + 1].findall("hp:tc", namespaces=tbl_elem.nsmap)

                    # (A) 다음 행에 tc가 2개 이상인지
                    if len(next_row_tcs) >= 2:
                        # (B) 다음 행의 tc들 중 적어도 하나에 <hp:pic> 태그가 있는지 확인
                        next_has_pic = any(
                            tc2.findall(".//hp:pic", namespaces=tbl_elem.nsmap)
                            for tc2 in next_row_tcs
                        )

                        if next_has_pic:
                            # 현재 행(tc) 안에서 모든 <hp:t>를 뽑아서 caption 문자열을 만든다
                            cap_text = "".join(
                                self._extract_text(t0)
                                for t0 in tc.findall(".//hp:t", namespaces=tc.nsmap)
                            ).strip()
                            norm_cap = re.sub(r"\s+", "", cap_text)

                            # 이미 같은 caption을 처리한 적이 없으면
                            if cap_text and norm_cap not in self._seen_section_texts:
                                self._seen_section_texts.add(norm_cap)

                                # "다음 행의 각 tc" 위치에 동일한 caption을 붙여준다
                                for tc2 in next_row_tcs:
                                    addr2 = tc2.find("hp:cellAddr", namespaces=tc2.nsmap)
                                    if addr2 is None:
                                        continue
                                    r2 = int(addr2.get("rowAddr"))
                                    c2 = int(addr2.get("colAddr"))
                                    cell_items[(r2, c2)].append(('caption', cap_text))

                            # 한 번 복제해 주었으므로, 이 tc는 본문 처리하지 않고 다음 tc로 넘어간다
                            continue
                nested_in_this = bool(tc.findall(".//hp:tbl", namespaces=tc.nsmap))
                # 2) 캡션 감지 전에, 이미 processed_cells에 있다면 스킵
                if (r,c) in skip_caption:
                    continue

               # 1) 바로 아래 행(r_idx+1)이 존재하는지, 같은 열(c)을 가진 tc2만 찾기
                next_nested = False
                next_pic    = False
                if r_idx + rs < len(rows):
                   # 다음 행에서 "지금 tc와 같은 열(colAddr==c)"인 <hp:tc>만 검색
                   for tc2 in rows[r_idx + rs].findall("hp:tc", namespaces=tbl_elem.nsmap):
                       addr2 = tc2.find("hp:cellAddr", namespaces=tc2.nsmap)
                       if addr2 is None:
                           continue
                       col2 = int(addr2.get("colAddr"))
                       # 만약 같은 열(col2 == c)이면 nested/pic 검사
                       if col2 == c:
                           if tc2.findall(".//hp:tbl", namespaces=tc2.nsmap):
                               next_nested = True
                           if tc2.findall(".//hp:pic", namespaces=tc2.nsmap):
                               next_pic = True

                if not nested_in_this and (next_nested or next_pic) :
                    if 0 <= r_idx - 1 < len(rows):
                        prev_row = rows[r_idx -1]
                        tc1_list = prev_row.findall("hp:tc", namespaces=tbl_elem.nsmap)
                        cell_texts = [
                            "".join(tc.itertext()).strip()
                            for tc in tc1_list
                        ]

                        if cell_texts and len(set(cell_texts)) == 1:
                            toptitle = cell_texts[0]
                            # toptitle이 comment 패턴(* 또는 주: / 자료: )이면 toptitle 처리하지 않는다
                            if not re.match(r"^\s*(?:(?:주|자료)\s*[:：]|\*)", toptitle):
                                norm_toptitle = re.sub(r"\s+", "", toptitle)
                                if norm_toptitle not in self._seen_section_texts:
                                    cell_items[(r-1,c)].append(('top_caption', toptitle))
                                    skip_caption.add((r-1,c))
                                    skip_rows.add(r-1)
                                    has_top_title = True

                    title = "".join(self._extract_text(t) for t in tc.findall(".//hp:t", namespaces=tc.nsmap)).strip()
                    # ─────────────── colSpan ≥ 2일 때 "같은 행에 붙여넣기" ───────────────
                    # cs = int(span.get("colSpan"))
                    # if cs > 1:
                    #     # 현재 셀(r, c)에 title이 있으면, 오른쪽으로 colSpan만큼 복제
                    #     for offset in range(1, cs):
                    #         target_col = c + offset
                    #         cell_items[(r, target_col)].append(('caption', title))
                    #         print(f"[DBG] colspan-caption at ({r},{target_col}): {title}")
                    #     # 복제 후에도, 원본 셀에는 한 번만 추가
                    cell_items[(r,c)].append(('caption', title))
                    continue

                # 2-b) nested table 감지
                if nested_in_this and not toc:
                    # 중첩 테이블 앞뒤 p까지 셀 내부 순서대로 처리
                    for p in tc.findall("./hp:subList/hp:p", namespaces=tc.nsmap):
                        tbl = p.find(".//hp:tbl", namespaces=p.nsmap)
                        if tbl is not None:
                            cell_items[(r,c)].append(('table', tbl))
                        else:
                            cell_items[(r,c)].append(('paragraph', p))
                    continue

                pics = tc.findall(".//hp:pic", namespaces=tc.nsmap)
                if pics:
                #     cap_node = caption_map.get((r,c))

                    # (tc 안의 <hp:p> 전부를 가져올 때는, hp:subList/hp:p 경로를 사용)
                    for p in tc.findall("./hp:subList/hp:p", namespaces=tc.nsmap):
                        # (1) 이 <hp:p> 안에 <hp:t>가 있어서 실제 문자열이 있는지 확인
                        t_elem = p.find(".//hp:t", namespaces=p.nsmap)
                        # (2) 이 <hp:p> 안에 <hp:pic>이 있는지 확인
                        pic_elem = p.find(".//hp:pic", namespaces=p.nsmap)

                        if t_elem is not None and self._extract_text(t_elem).strip():
                            # 텍스트 내용이 있는 경우: paragraph로 처리
                            cell_items[(r, c)].append(('paragraph', p))

                        if pic_elem is not None:
                            # 그림 감지 → ImageRef 뽑아서 picture로 처리
                            img = self._get_image_ref(pic_elem)
                            cap_node = caption_map.get((r, c))
                            cell_items[(r, c)].append(('picture', (img, cap_node)))

                        # 해당 셀(tc)의 다른 처리를 건너뛰기 위해 continue
                    continue

                # ── 2-d) comment 감지 ('주:' or '자료:' or '*') ───────────────────────────────────────
                texts = [
                    "".join(self._extract_text(t) for t in p.findall(".//hp:t", namespaces=tc.nsmap)).strip()
                    for p in tc.findall(".//hp:p", namespaces=tc.nsmap)
                ]
                txt = " ".join(filter(None, texts)).strip()
                # "주:", "자료:", 또는 "*" 로 시작하는 패턴 (양쪽 공백 허용)
                if re.match(r"^\s*(?:(?:주|자료)\s*[:：]|\*)", txt):
                    # 1) 현재 행에 tc가 딱 1개인지 (num_tcs_curr_row == 1) 확인
                    # 2) 바로 위(prev) 행에 셀이 2개 이상인 경우만 복제 로직 시도
                    prev_row_tcs = (
                        rows[r_idx - 1].findall("hp:tc", namespaces=tbl_elem.nsmap)
                        if (r_idx - 1) >= 0 else []
                    )

                    if num_tcs_curr_row == 1 and len(prev_row_tcs) >= 2:
                        # 3) prev_row 중 하나라도 <hp:pic> 태그가 있는지 확인하여 순수테이블과 구분
                        prev_has_pic = any(
                            p_tc.findall(".//hp:pic", namespaces=tbl_elem.nsmap)
                            for p_tc in prev_row_tcs
                        )

                        if prev_has_pic:
                            # (A) 이 comment가 들어 있는 "현재 셀(tc)"의 주소 정보
                            addr   = tc.find("hp:cellAddr",  namespaces=tc.nsmap)
                            span   = tc.find("hp:cellSpan",  namespaces=tc.nsmap)
                            r_cur  = int(addr.get("rowAddr"))
                            c_cur  = int(addr.get("colAddr"))
                            cs     = int(span.get("colSpan"))

                            # (B) colSpan ≥ 2라면, 같은 행(r_cur)의 오른쪽 셀들에도 복제
                            if cs > 1:
                                for offset in range(1,2): # 한 번씩만 복제
                                    target_col = c_cur + offset
                                    cell_items.setdefault((r_cur, target_col), []).append(('comment', txt))

                            # (C) 원본 셀(r_cur, c_cur)에도 한 번만 comment 저장
                            cell_items.setdefault((r_cur, c_cur), []).append(('comment', txt))
                            continue

                    # 4) 위 조건이 하나라도 만족되지 않으면, "현재 셀(r,c)"에만 comment 저장
                    cell_items.setdefault((r, c), []).append(('comment', txt))
                    continue

                # 2-e) 순수 데이터 셀 → TableData 로만
                parts = []
                for p in tc.findall(".//hp:p", namespaces=tc.nsmap):
                    for t in p.findall(".//hp:t", namespaces=p.nsmap):
                        parts.append(self._extract_text(t))
                cell_text = "\n".join(parts).strip()
                if len(cell_text) > 200:
                    for sub_p in tc.findall(".//hp:p", namespaces=tc.nsmap):
                        cell_items[(r,c)].append(('paragraph', sub_p))
                    continue

                data.table_cells.append(
                    TableCell(
                        text=cell_text,
                        row_span=rs,
                        col_span=cs,
                        start_row_offset_idx=r,
                        end_row_offset_idx=r+rs,
                        start_col_offset_idx=c,
                        end_col_offset_idx=c+cs,
                        column_header=(r==0),
                        row_header=False
                    )
                )
        # 한 table 안에 주석이 포함되어 있는 경우에 사용
        # 'table' 없이 'comment'만 있는 경우 판별
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
        ) # 해당 열에 picture 없으면 has picture = False처리
        has_comment = any(
            typ == 'comment'
            for items in cell_items.values()
            for typ, _ in items
        )

        # comment만 있고 nested 테이블이 없는 경우 (순수 테이블 맨 아래 행에 주석이 있는 경우)
        if (not has_table
            and has_comment
            and not has_picture
            and not nested_in_this
            and not toc
        ):
            # 실제 테이블 추가
            is_empty = not any(cell.text for cell in data.table_cells)
            if not is_empty:

                copied_cells = deepcopy(data.table_cells)
                temp_data = TableData(num_rows=data.num_rows, num_cols=data.num_cols)
                temp_data.table_cells = copied_cells
                doc.add_table(data=temp_data, parent=parent,
                             prov=ProvenanceItem(
                                 page_no=1,
                                 bbox=BoundingBox(l=0, t=0, r=1, b=1),
                                 charspan=(0, 0)
                             ))
                data.table_cells.clear()

                # comment 출력
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


             # return 대신 초기화
             # 3) comment만 제거
                for (r, c), items in list(cell_items.items()):
                    # 'comment' 타입이 아닌 것만 필터링
                    new_items = [(typ, payload) for (typ, payload) in items if typ != 'comment']
                    if new_items:
                        cell_items[(r, c)] = new_items
                    else:
                        # 만약 해당 위치에 남은 항목이 없으면 키 자체를 삭제
                        del cell_items[(r, c)]


        # 4) 마지막에 한 번에 "순수 테이블"만 TableItem 으로 출력
        sorted_coords = sorted(cell_items.keys(), key=lambda x: (x[1], x[0]))  # (c,r) 순서로 정렬
        for r, c in sorted_coords:
            for typ, payload in cell_items[(r,c)]:
                if typ == 'top_caption':
                    # 제목 있으면 이게 parent
                    norm_payload = re.sub(r"\s+", "",payload)
                    if norm_payload in self._seen_section_texts: # section header 로 처리된거면 제외
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
                    if re.match(r'^(?:\d+\.\s+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s*)', payload):
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
                    doc.add_picture(parent=parent,
                                    image=img,
                                    caption=cap,
                                    prov=ProvenanceItem(
                                        page_no=1,
                                        bbox=BoundingBox(l=0, t=0, r=1, b=1),
                                        charspan=(0, 0)
                                    ))
                elif typ == 'comment':
                    doc.add_text(label=DocItemLabel.CAPTION,
                                text=payload,
                                parent=parent,
                                prov=ProvenanceItem(
                                    page_no=1,
                                    bbox=BoundingBox(l=0, t=0, r=1, b=1),
                                    charspan=(0, len(payload))
                                ))

        # 빈 테이블은 추가하지 않음
        # 모든 셀에 텍스트가 없으면 빈 테이블로 간주
        is_empty_tbl = True
        for i in data.table_cells:
            if i.text:
                is_empty_tbl = False
                break
        if is_empty_tbl or has_top_title:
            return

        parent = self.current_section_group
        # 마지막에 순수 테이블 한 번만 추가
        doc.add_table(data=data, parent=parent,
                     prov=ProvenanceItem(
                         page_no=1,
                         bbox=BoundingBox(l=0, t=0, r=1, b=1),
                         charspan=(0, 0)
                     ))

    def _process_rect(self, rect_elem: etree._Element, doc: DoclingDocument) -> None:
        """Process a top-level <hp:rect> element (text box)."""
        draw_text_elem = rect_elem.find(".//hp:drawText", namespaces=rect_elem.nsmap)
        if draw_text_elem is None:
            return

        text_parts = [t.text for t in draw_text_elem.findall(".//hp:t", namespaces=draw_text_elem.nsmap) if t.text]
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
            #     100자 초과 시: drawText 내의 <hp:p> 단락들을 _process_paragraph로 넘기기
            for p in draw_text_elem.findall(".//hp:p", namespaces=draw_text_elem.nsmap):
                # p 안의 모든 run/t/pic/equation은 _process_paragraph에서 다시 분기 처리
                self._process_paragraph(p, doc)

    def _convert_wmf_to_png(self, wmf_bytes: bytes) -> Optional[bytes]:
        """Convert WMF bytes to PNG using Wand, if available."""
        if not WAND_AVAILABLE:
            return None
        try:
            import os
            import sys

            # ImageMagick의 stderr 경고 메시지를 임시로 억제
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
            logging.warning(f"WMF 변환 실패 (변환은 시도됨): {e}")
            return None

    def _read_image_bytes(self, bin_id: str) -> Optional[bytes]:
        """Read image bytes from BinData, converting WMF if necessary."""
        # include_wmf 옵션에 따라 확장자 리스트 결정
        if self.include_wmf:
            extensions = (".bmp", ".png", ".jpg", ".jpeg", ".wmf", ".tif")
        else:
            extensions = (".bmp", ".png", ".jpg", ".jpeg", ".tif")  # wmf 제외

        for ext in extensions:
            try:
                data = self.zip.read(f"BinData/{bin_id}{ext}")
            except KeyError:
                continue
            # If WMF, try converting to PNG (include_wmf=True인 경우에만 도달)
            if ext == ".wmf":
                converted = self._convert_wmf_to_png(data)
                if converted:
                    return converted
                else:
                    return None
            return data
        return None

    def _process_picture(
        self,
        pic_elem: etree._Element,
        doc: DoclingDocument,
        caption: Optional[str] = None,
    ) -> None:
        """Process a picture <hp:pic> element and add an image node."""
        if not self.save_images:
            return

        parent_node = self.current_list_item or self.current_section_group

        # 1) Extract binaryItemIDRef
        img_ref = pic_elem.find("hc:img", namespaces=pic_elem.nsmap)
        if img_ref is None:
            return
        bin_id = img_ref.get("binaryItemIDRef")
        if not bin_id:
            return

        # 2) Read (and convert) image bytes
        image_bytes = self._read_image_bytes(bin_id)
        if not image_bytes:
            return

        # 3) Open with PIL
        try:
            pil_image = Image.open(BytesIO(image_bytes))
        except (UnidentifiedImageError, OSError) as e:
            logging.debug(f"PIL failed to open image: {e}")
            return

        # 4) Create ImageRef
        try:
            img_ref_obj = ImageRef.from_pil(image=pil_image, dpi=72)
        except (UnidentifiedImageError, OSError) as e:
            logging.debug(f"Failed to create ImageRef: {e}")
            return

        # 5) Add picture node
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
        """Process an equation <hp:equation> element by adding its text content."""
        parent_node = self.current_list_item or self.current_section_group or None
        formula_text = "".join(eq_elem.itertext()).strip()
        doc.add_text(label=DocItemLabel.FORMULA, text=formula_text, parent=parent_node,
                    prov=ProvenanceItem(
                        page_no=1,
                        bbox=BoundingBox(l=0, t=0, r=1, b=1),
                        charspan=(0, len(formula_text))
                    ))

    def _add_header(self, doc: DoclingDocument, level: int, text: str) -> None:
        """Add a heading node of the given level with the specified text."""
        curr_level = level

        # 상위 그룹들 채워 넣기 - 헤더가 뒤에서 나오는 경우 해결 위해 추가됨
        for lvl in range(0, curr_level):
            if self.parents.get(lvl) is None:
                self.parents[lvl] = doc.add_group(
                    parent=self.parents[lvl - 1] if lvl - 1 >= 0 else None,
                    label=GroupLabel.SECTION,
                    name=f"header-{lvl}"
                )
        # -------------
        # Add intermediate groups for skipped levels
        for lvl in range(0, curr_level):
            if self.parents.get(lvl) is None and lvl < curr_level:
                self.parents[lvl] = doc.add_group(parent=(self.parents[lvl-1] if lvl-1 >= 0 else None), label=GroupLabel.SECTION, name=f"header-{lvl}")
        # Clear deeper levels when moving to a higher-level heading
        for lvl in range(curr_level, self.max_levels):
            self.parents[lvl] = None
        parent_node = self.parents[curr_level-1] if curr_level - 1 >= 0 else None
        self.parents[curr_level] = doc.add_heading(parent=parent_node, text=text, level=curr_level,
                                                  prov=ProvenanceItem(
                                                      page_no=1,
                                                      bbox=BoundingBox(l=0, t=0, r=1, b=1),
                                                      charspan=(0, len(text))
                                                  ))

    def _end_list(self) -> None:
        """End the current list grouping (if any)."""
        self.current_list_group = None
        self.current_list_item = None
