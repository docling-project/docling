import base64
import json
import webbrowser
import tempfile
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time
from urllib.parse import parse_qs, urlparse
import html
import re
import socketserver

def clean_json_line(line):
    """제어 문자와 HTML 엔터티를 처리하여 깨지지 않게 하는 함수"""
    # 제어 문자 제거 (예: \n, \r, \t 등)
    line = re.sub(r'[\x00-\x1F\x7F]', '', line)
    # HTML 엔터티 디코딩 (예: &lt; -> <, &gt; -> > 등)
    line = line.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return line


def load_json_data(file_path):
    """파일에서 JSON 데이터를 읽어오는 함수"""
    with open(file_path, 'r', encoding='utf-8') as file:
        # 각 라인을 읽고, 하나의 리스트로 합침
        lines = file.readlines()
        data = []

        for line in lines:
            # 제어 문자 및 HTML 엔터티 정리
            cleaned_line = clean_json_line(line.strip())
            try:
                line_data = json.loads(cleaned_line)  # 정리된 라인을 JSON 파싱
                data.extend(line_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

        return data


class BOKReportWebViewer:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.data = []
        self.total_pages = 0
        self.load_data()

    def load_data(self):
        """JSON 파일 로드 (개선된 버전)"""
        self.data = load_json_data(self.json_file_path)

        if not self.data:
            print("데이터를 로드할 수 없습니다.")
            return False

        # 페이지 정보 추출
        pages = set()
        for item in self.data:
            if isinstance(item, dict) and 'page' in item:
                pages.add(item['page'])

        self.total_pages = max(pages) if pages else 1
        print(f"데이터 로드 완료: {len(self.data)}개 항목, {self.total_pages}페이지")

        # 데이터 구조 분석
        if self.data:
            first_item = self.data[0]
            print(f"첫 번째 항목 구조: {list(first_item.keys()) if isinstance(first_item, dict) else type(first_item)}")

        return True

    def get_page_data(self, page_num):
        """특정 페이지의 데이터 추출"""
        return [item for item in self.data if isinstance(item, dict) and item.get('page') == page_num]

    def format_text_for_display(self, item):
        """텍스트 항목을 표시용으로 포맷팅"""
        if not isinstance(item, dict):
            return str(item)

        text = item.get('value', item.get('text', str(item)))
        font_info = item.get('font', {})

        # HTML 이스케이프
        text = html.escape(str(text))

        # 스타일 적용
        style = []

        if font_info.get('bold'):
            style.append('font-weight: bold')

        font_size = font_info.get('size', 12)
        style.append(f'font-size: {font_size}pt')

        color = font_info.get('color', '#000000')
        if color and color != '#000000':
            style.append(f'color: {color}')

        font_name = font_info.get('name', '맑은 고딕')
        style.append(f"font-family: '{font_name}', sans-serif")

        style_str = '; '.join(style)
        return f'<span style="{style_str}">{text}</span>'

    def format_table_for_display(self, item):
        """테이블 항목을 표시용으로 포맷팅"""
        if not isinstance(item, dict):
            return f'<div class="table-container"><p>{html.escape(str(item))}</p></div>'

        table_html = item.get('value', item.get('html', ''))
        title = item.get('title', '')

        result = '<div class="table-container">'
        if title:
            result += f'<h4 class="table-title">{html.escape(title)}</h4>'

        if table_html:
            # 테이블 스타일 개선
            styled_table = table_html.replace('<table', '<table class="data-table"')
            result += styled_table
        else:
            result += '<p>[테이블 내용 없음]</p>'

        result += '</div>'
        return result

    def generate_page_html(self, page_num, view_mode='formatted'):
        """특정 페이지의 HTML 생성"""
        page_data = self.get_page_data(page_num)

        # 페이지 데이터가 없는 경우 전체 데이터에서 일부 표시
        if not page_data and self.data:
            items_per_page = 20
            start_idx = (page_num - 1) * items_per_page
            end_idx = start_idx + items_per_page
            page_data = self.data[start_idx:end_idx]

            if not page_data:
                return '<div class="page-content"><p>이 페이지에는 데이터가 없습니다.</p></div>'

        if view_mode == 'raw':
            return f'<pre>{html.escape(json.dumps(page_data, ensure_ascii=False, indent=2))}</pre>'

        content = f'<div class="page-content">'
        current_paragraph = []

        for item in page_data:
            if not isinstance(item, dict):
                content += f'<p>{html.escape(str(item))}</p>'
                continue

            item_type = item.get('item', item.get('type', 'text'))

            if item_type == 'text' or not item_type:
                if view_mode == 'html':
                    current_paragraph.append(html.escape(item.get('value', item.get('text', str(item)))))
                else:
                    current_paragraph.append(self.format_text_for_display(item))

            elif item_type == 'para' or item_type == 'paragraph':
                if current_paragraph:
                    # 단락 정렬 처리
                    align = item.get('align', '')
                    align_class = f' class="text-{align}"' if align else ''

                    # 들여쓰기 처리
                    indent = item.get('indent', 0)
                    style = f' style="margin-left: {max(0, indent * 20)}px;"' if indent else ''

                    content += f'<p{align_class}{style}>{"".join(current_paragraph)}</p>'
                    current_paragraph = []
                else:
                    content += '<br>'

            elif item_type == 'image':
                if current_paragraph:
                    content += f'<p>{"".join(current_paragraph)}</p>'
                    current_paragraph = []
                img_path = item.get('value', '')
                if img_path and os.path.exists(img_path):
                    ext = os.path.splitext(img_path)[1].lower()
                    mime_map = {'.bmp': 'image/bmp', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif'}
                    mime = mime_map.get(ext, 'image/png')
                    with open(img_path, 'rb') as f:
                        img_b64 = base64.b64encode(f.read()).decode('utf-8')
                    content += f'<div class="image-container"><img src="data:{mime};base64,{img_b64}" style="max-width:100%;" alt="이미지"></div>'
                else:
                    content += f'<div class="image-container"><p>[이미지 없음: {html.escape(img_path)}]</p></div>'

            elif item_type == 'table':
                if current_paragraph:
                    content += f'<p>{"".join(current_paragraph)}</p>'
                    current_paragraph = []
                content += self.format_table_for_display(item)

            elif item_type == 'break':
                if current_paragraph:
                    content += f'<p>{"".join(current_paragraph)}</p>'
                    current_paragraph = []
                content += '<div class="page-break"></div>'
            else:
                # 알 수 없는 타입의 경우
                text_content = item.get('value', item.get('text', str(item)))
                current_paragraph.append(html.escape(str(text_content)))

        # 남은 내용 처리
        if current_paragraph:
            content += f'<p>{"".join(current_paragraph)}</p>'

        content += '</div>'
        return content

    def generate_full_html(self, page_num=1, view_mode='formatted'):
        """완전한 HTML 페이지 생성"""
        # 페이지 번호 유효성 검사
        if self.total_pages == 0:
            # 페이지 정보가 없는 경우 전체 데이터를 페이지로 나누기
            items_per_page = 20
            self.total_pages = max(1, (len(self.data) + items_per_page - 1) // items_per_page)

        page_num = max(1, min(page_num, self.total_pages))

        page_content = self.generate_page_html(page_num, view_mode)

        # 페이지 네비게이션 생성
        nav_html = '<div class="navigation">'
        nav_html += f'<button onclick="changePage({page_num-1})" {"disabled" if page_num <= 1 else ""}>← 이전</button>'
        nav_html += f'<span class="page-info">페이지 {page_num} / {self.total_pages}</span>'
        nav_html += f'<button onclick="changePage({page_num+1})" {"disabled" if page_num >= self.total_pages else ""}>다음 →</button>'
        nav_html += '</div>'

        # 보기 모드 선택
        mode_html = f'''
        <div class="view-mode">
            <label>보기 모드:</label>
            <button onclick="changeMode('formatted')" class="mode-btn{' active' if view_mode == 'formatted' else ''}">서식</button>
            <button onclick="changeMode('html')" class="mode-btn{' active' if view_mode == 'html' else ''}">HTML</button>
            <button onclick="changeMode('raw')" class="mode-btn{' active' if view_mode == 'raw' else ''}">원본</button>
        </div>
        '''

        html_template = f'''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>자유소프트 HWP/HWPX 파서 뷰어 - 페이지 {page_num}</title>
    <style>
        body {{
            font-family: '맑은 고딕', 'Malgun Gothic', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}

        .controls {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }}

        .navigation button {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}

        .navigation button:hover:not(:disabled) {{
            background: #45a049;
        }}

        .navigation button:disabled {{
            background: #cccccc;
            cursor: not-allowed;
        }}

        .page-info {{
            font-weight: bold;
            margin: 0 15px;
            color: #333;
        }}

        .view-mode {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .mode-btn {{
            background: #f0f0f0;
            border: 1px solid #ddd;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}

        .mode-btn:hover {{
            background: #e0e0e0;
        }}

        .mode-btn.active {{
            background: #2196F3;
            color: white;
            border-color: #2196F3;
        }}

        .content {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-height: 600px;
        }}

        .page-content {{
            line-height: 1.8;
        }}

        .page-content p {{
            margin: 15px 0;
        }}

        .text-center {{ text-align: center; }}
        .text-right {{ text-align: right; }}
        .text-left {{ text-align: left; }}

        .table-container {{
            margin: 20px 0;
            overflow-x: auto;
        }}

        .table-title {{
            margin: 10px 0;
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 5px;
        }}

        .data-table, table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}

        .data-table th, .data-table td,
        table th, table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}

        .data-table th, table th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #333;
        }}

        .data-table tr:nth-child(even),
        table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}

        .page-break {{
            border-top: 3px solid #ccc;
            margin: 30px 0;
            position: relative;
        }}

        .page-break::after {{
            content: "페이지 구분";
            background: white;
            padding: 0 10px;
            position: absolute;
            top: -10px;
            left: 50%;
            transform: translateX(-50%);
            color: #666;
            font-size: 12px;
        }}

        pre {{
            background: #f4f4f4;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.4;
        }}

        @media (max-width: 768px) {{
            .controls {{
                flex-direction: column;
                gap: 15px;
            }}

            .navigation, .view-mode {{
                width: 100%;
                justify-content: center;
            }}

            body {{
                padding: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏦 자유소프트 HWP/HWPX 파서 뷰어</h1>
        <p>데이터 분석 뷰어</p>
    </div>

    <div class="controls">
        {nav_html}
        {mode_html}
    </div>

    <div class="content">
        {page_content}
    </div>

    <script>
        function changePage(page) {{
            if (page >= 1 && page <= {self.total_pages}) {{
                const currentMode = getCurrentMode();
                window.location.href = `?page=${{page}}&mode=${{currentMode}}`;
            }}
        }}

        function changeMode(mode) {{
            const currentPage = getCurrentPage();
            window.location.href = `?page=${{currentPage}}&mode=${{mode}}`;
        }}

        function getCurrentPage() {{
            const urlParams = new URLSearchParams(window.location.search);
            return urlParams.get('page') || {page_num};
        }}

        function getCurrentMode() {{
            const urlParams = new URLSearchParams(window.location.search);
            return urlParams.get('mode') || 'formatted';
        }}

        // 키보드 네비게이션
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'ArrowLeft') {{
                changePage({page_num} - 1);
            }} else if (e.key === 'ArrowRight') {{
                changePage({page_num} + 1);
            }}
        }});
    </script>
</body>
</html>
        '''

        return html_template


class BOKReportHandler(SimpleHTTPRequestHandler):
    def __init__(self, viewer, *args, **kwargs):
        self.viewer = viewer
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """GET 요청 처리"""
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        # 페이지와 모드 파라미터 추출
        page_num = int(query_params.get('page', [1])[0])
        view_mode = query_params.get('mode', ['formatted'])[0]

        # HTML 생성
        html_content = self.viewer.generate_full_html(page_num, view_mode)

        # 응답 전송
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))

    def log_message(self, format, *args):
        """로그 메시지 출력 비활성화"""
        pass


def create_web_server(json_file_path, port=8000):
    """웹 서버 생성 및 실행"""
    if not os.path.exists(json_file_path):
        print(f"파일을 찾을 수 없습니다: {json_file_path}")
        return None

    viewer = BOKReportWebViewer(json_file_path)

    if not viewer.data:
        print("데이터를 로드할 수 없어 서버를 시작할 수 없습니다.")
        return None

    # 사용 가능한 포트 찾기
    for attempt_port in range(port, port + 10):
        try:
            # 핸들러 클래스 생성
            def handler_factory(*args, **kwargs):
                return BOKReportHandler(viewer, *args, **kwargs)

            # 서버 생성
            with socketserver.TCPServer(("", attempt_port), handler_factory) as httpd:
                print(f"웹 서버 시작: http://localhost:{attempt_port}")

                # 브라우저에서 열기
                try:
                    webbrowser.open(f'http://localhost:{attempt_port}')
                    print("브라우저에서 열었습니다.")
                except Exception as e:
                    print(f"브라우저 열기 실패: {e}")
                    print(f"수동으로 다음 주소를 열어주세요: http://localhost:{attempt_port}")

                print(f"\n사용법:")
                print("- 좌우 화살표 키로 페이지 이동")
                print("- 보기 모드 버튼으로 표시 형식 변경")
                print("- Ctrl+C로 서버 종료")

                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\n서버를 종료합니다.")

                return httpd

        except OSError as e:
            if "Address already in use" in str(e):
                continue
            else:
                print(f"서버 시작 실패: {e}")
                return None

    print(f"포트 {port}-{port+9} 모두 사용 중입니다.")
    return None


def main():
    print("=== 자유소프트 HWP/HWPX 파서 뷰어 ===")

    # 파일 경로 입력받기
    json_file = input("JSON 파일 경로를 입력하세요 (또는 엔터키로 현재 경로의 'output.json' 사용): ").strip()

    if not json_file:
        json_file = "output.json"

    if not os.path.exists(json_file):
        print(f"파일을 찾을 수 없습니다: {json_file}")
        print("현재 디렉토리의 파일들:")
        for f in os.listdir('.'):
            if f.endswith(('.txt', '.json')):
                print(f"  - {f}")
        return

    # 웹 서버 시작
    create_web_server(json_file)


if __name__ == "__main__":
    main()
