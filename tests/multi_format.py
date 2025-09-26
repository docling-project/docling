import json
import logging
from pathlib import Path

import yaml

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.backend.genos_msword_backend import GenosMsWordDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

_log = logging.getLogger(__name__)
from docling.document_converter import DocumentConverter

def main():
    input_paths = [
        # Path("README.md"),
        # Path("/workspaces/삼증리서치_리포트반출_일부/hd한국조선해양/original/2025042414165512K_01.docx"),
        # Path("/workspaces/삼증리서치_리포트반출_일부/비상장기업/original/2025042510145245K_01.docx"),
        # BOK HWPX
        # Path("/workspaces/test_data/25.09 주요 상품 라인업(상품지원담당)(사내한).pptx"),
        Path("/workspaces/doc_parser/sample_files/sample.html"),
        # Path("/workspaces/test_data/259_그룹.pptx"),
        # Path("/workspaces/hwpx/★(통화정책국)의결문(안) 및 참고자료(1810)_의결문제외.hwp"),
        # Path("/workspaces/test_data/25.09 주요 상품 라인업(상품지원담당)(사내한).pdf"),
        # Path("/workspaces/hwpx/외환국제금융동향(2018.4.12)_최종(송부본).hwpx"),
        # Path("/workspaces/hwpx/★(통화정책국)의결문(안) 및 참고자료(1810)_의결문제외.hwpx"),
        # Path("/workspaces/hwpx/(통화정책국)통화정책 여건점검(1810)_송부.hwpx"),
        # Path("/workspaces/hwpx/(통화정책국)의결문(안) 및 참고자료(1804)_송부용.hwpx"),
        # Path("/workspaces/hwpx/(1810) 통화정책방향 여건점검 메모_F.hwpx"),
        # Path("/workspaces/hwpx/2021.01.29_주간 글로벌 펀드자금 유출입 동향(20210121-20210127)_210129_주간 글로벌 펀드자금 유출입 동향(210121~210127)_F.hwpx"),
        # Path("/workspaces/hwpx/2016.11.21_주중 금융시장 동향_주중_금융시장동향(20161121)_MI.hwpx"),
        # Path("/workspaces/hwpx/2017.09.01_Market View(81호_170901) 미 증시 거품설 열흘 붉은 꽃은 없다_제81호_MV_미 증시 거품설, 열흘 붉은 꽃은 없다_170901.hwpx"),
        # Path("/workspaces/hwpx/2022.11.07_위원협의회 국제금융 외환시장 동향 및 전망(2022.10.31~11.4)_f.hwpx"),
        # Path("/workspaces/hwpx/2023.05.03_MI-NET_2023.05.02_위원협의회 국제금융 외환시장 동향 및 전망(2023.04.25~05.01)_f.hwpx"),
        # Path("/workspaces/hwpx/201810_금융시장동향_FF.hwpx"),
        # Path("/workspaces/hwpx/(1810) 통화정책방향 여건점검 메모_F.hwpx"),
    ]

    ## for defaults use:
    # doc_converter = DocumentConverter()

    ## to customize use:

    doc_converter = (
        DocumentConverter(  # all of the below is optional, has internal defaults.
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.ASCIIDOC,
                InputFormat.CSV,
                InputFormat.MD,
                InputFormat.HWP,
                InputFormat.XML_HWPX
            ],  # whitelist formats, non-matching files are ignored.
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline, backend=PyPdfiumDocumentBackend
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline, backend=GenosMsWordDocumentBackend  # , backend=MsWordDocumentBackend 
                ),
            },
        )
    )

    conv_results = doc_converter.convert_all(input_paths)

    for res in conv_results:
        out_path = Path("/workspaces/doc_parser/scratch")
        print(
            f"Document {res.input.file.name} converted."
            f"\nSaved markdown output to: {out_path!s}"
        )
        _log.debug(res.document._export_to_indented_text(max_text_len=16))

        # Markdown (변경 없음)
        with (out_path / f"{res.input.file.stem}.md").open("w", encoding="utf-8") as fp:
            fp.write(res.document.export_to_markdown())

        # JSON: ensure_ascii=False 로 한글 출력
        with (out_path / f"{res.input.file.stem}.json").open("w", encoding="utf-8") as fp:
            fp.write(
                json.dumps(
                    res.document.export_to_dict(),
                    ensure_ascii=False,
                    indent=2,
                )
            )

        # YAML: allow_unicode=True 로 한글 출력
        with (out_path / f"{res.input.file.stem}.yaml").open("w", encoding="utf-8") as fp:
            fp.write(
                yaml.safe_dump(
                    res.document.export_to_dict(),
                    allow_unicode=True,
                    sort_keys=False,
                    indent=2,
                )
            )
if __name__ == "__main__":
    main()
