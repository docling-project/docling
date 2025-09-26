# mspowerpoint backend

작성자: 서영 윤
상태: 초안
카테고리: PRD
최종 편집 일시: 2025년 9월 22일 오후 2:00

## 🎯 개요

기술 구현, 아키텍처 결정, 주요 구성 요소에 대한 전반적인 설명.

## 🔧 기술 상세정보

API, 데이터 모델, 알고리즘, 시스템 상호 작용을 포함한 구현에 대한 자세한 설명.

## ⚡성과 고려 사항

성능에 미치는 영향, 확장성 문제, 최적화 전략 분석.

## 🧪 시험 전략

안정성을 보장하기 위한 단위 테스트, API 통합 테스트 및 테스트 접근 방식에 대한 개요.

## 📋 종속성 & 요구 사항

기술적 종속성, 시스템 요구 사항 및 필요한 외부 서비스 리스트.

- MSO_SHAPE_TYPE: 파워포인트 도형의 “종류”를 나타내는 열거형. 예) AUTO_SHAPE, TEXT_BOX, PICTURE, TABLE, CHART, GROUP, LINE, PLACEHOLDER 등.
- PP_PLACEHOLDER: 슬라이드의 “플레이스홀더 타입”을 나타내는 열거형. 예) TITLE, CENTER_TITLE, SUBTITLE, BODY, TABLE, CHART, PICTURE, SLIDE_NUMBER, DATE, FOOTER 등.

### DocItemLabel

- title
    - PP_PLACEHOLDER.CENER_TITLE
    - PP_PLACEHOLDER.TITLE
- SECTION_HEADER
    - PP_PLACEHOLDER.SUBTITLE
- shape.has_chart, shape.chart print 내용
    
    ```bash
    <pptx.chart.chart.Chart object at 0x7fb5968ebf10> shape.chart
    True shape.has_chart
    True shape.has_chart
    ```
    

### 처리 분류

- 객체 API로 처리하는 부분:
    
    *python-pptx*는 PowerPoint(.pptx) 파일을 만들고, 읽고, 업데이트하기 위한 Python 라이브러리입니다.
    
    - 파일/슬라이드/도형 순회: Presentation(...), pptx_obj.slides, for shape in slide.shapes
    - 그룹 도형: `shape.shape_type == MSO_SHAPE_TYPE.GROUP, shape.shapes`
    - 위치/크기(bbox): `shape.left/top/width/height → generate_prov(...)`
    - 텍스트: `shape.has_text_frame, shape.text_frame.paragraphs, paragraph.level, shape.text`
    - 플레이스홀더: `shape.is_placeholder, shape.placeholder_format.type(TITLE/SUBTITLE)`
    - 이미지: `shape.shape_type == PICTURE, shape.image, image.blob, image.dpi`
    - 표(기본): `shape.has_table, shape.table, table.rows, row.cells, cell.text`
    - 차트: `shape.has_chart, shape.chart, chart.series, series.points/values, plots[0].categories`
    - 노트: slide.has_notes_slide, slide.notes_slide.notes_text_frame.text
- OOXML로 직접 처리하는 부분
    - 리스트 항목 판정(글머리/번호 확인):
    - paragraph._element.find(".//a:buChar", ...), paragraph._element.find(".//a:buAutoNum", ...)
    - 표 병합(span) 추출:
    - shape._element.xpath(".//a:tbl/a:tr[{row_idx+1}]/a:tc[{col_idx+1}]")
    - rowSpan, gridSpan 속성 읽어서 row_span, col_span 계산
    - 네임스페이스 사용: self.namespaces = {"a": ..., "c": ..., "p": ...}

요약

- 기본은 python-pptx 객체 API로 처리하고, 객체 API로 접근이 어려운 “리스트 글머리/번호 확인”과 “표 병합(span)”만 OOXML(XPath)로 보완합니다.

## Placeholder types

There are 18 types of placeholder.

Title, Center Title, Subtitle, BodyThese placeholders typically appear on a conventional “word chart” containing text only, often organized as a title and a series of bullet points. All of these placeholders can accept text only.ContentThis multi-purpose placeholder is the most commonly used for the body of a slide. When unpopulated, it displays 6 buttons to allow insertion of a table, a chart, SmartArt, a picture, clip art, or a media clip.Picture, Clip ArtThese both allow insertion of an image. The insert button on a clip art placeholder brings up the clip art gallery rather than an image file chooser, but otherwise these behave the same.Chart, Table, Smart ArtThese three allow the respective type of rich graphical content to be inserted.Media ClipAllows a video or sound recording to be inserted.Date, Footer, Slide NumberThese three appear on most slide masters and slide layouts, but do not behave as most users would expect. These also commonly appear on the Notes Master and Handout Master.HeaderOnly valid on the Notes Master and Handout Master.Vertical Body, Vertical Object, Vertical TitleUsed with vertically oriented languages such as Japanese.

https://python-pptx.readthedocs.io/en/latest/