import json
import re
import html
import webbrowser
import os
import socketserver
from collections import defaultdict
from http.server import SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


class VectorsWebViewer:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.chunks = []
        self.pages_content = {}  # page_no -> [chunk, ...]
        self.total_pages = 0
        self.n_chunk_of_doc = 0
        self.load_data()

    def load_data(self):
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)

            if not self.chunks:
                print("❌ 벡터 데이터가 비어있습니다.")
                return False

            # i_page 기준으로 페이지별 분류
            temp = defaultdict(list)
            for chunk in self.chunks:
                page_no = chunk.get('i_page', 1)
                temp[page_no].append(chunk)

            # i_chunk_on_doc 순서로 정렬
            for p in temp:
                temp[p].sort(key=lambda c: c.get('i_chunk_on_doc', 0))

            self.pages_content = dict(temp)
            self.total_pages = max(self.pages_content.keys()) if self.pages_content else 0
            self.n_chunk_of_doc = self.chunks[0].get('n_chunk_of_doc', len(self.chunks)) if self.chunks else 0

            print(f"✅ 데이터 로드 완료: {len(self.chunks)}개 청크, {self.total_pages}페이지")
            return True
        except Exception as e:
            print(f"❌ 로드 실패: {e}")
            return False

    def render_markdown_table(self, text: str) -> str:
        """마크다운 테이블을 HTML 테이블로 변환"""
        lines = text.strip().splitlines()
        if len(lines) < 2:
            return html.escape(text).replace('\n', '<br>')

        # 마크다운 테이블 여부 확인 (구분선 행 필요)
        has_table = any(re.match(r'^\s*\|?[-:| ]+\|?\s*$', l) for l in lines)
        if not has_table:
            return html.escape(text).replace('\n', '<br>')

        html_out = '<div class="table-container"><table>'
        is_header_done = False
        for line in lines:
            # 구분선 행 스킵
            if re.match(r'^\s*\|?[-:| ]+\|?\s*$', line):
                is_header_done = True
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            tag = 'th' if not is_header_done else 'td'
            html_out += '<tr>' + ''.join(f'<{tag}>{html.escape(c)}</{tag}>' for c in cells) + '</tr>'
        html_out += '</table></div>'
        return html_out

    def render_chunk(self, chunk: dict, view_mode: str) -> str:
        chunk_idx = chunk.get('i_chunk_on_doc', 0)
        i_page = chunk.get('i_page', '-')
        e_page = chunk.get('e_page', '-')
        n_char = chunk.get('n_char', 0)
        n_word = chunk.get('n_word', 0)
        n_line = chunk.get('n_line', 0)
        media_files_raw = chunk.get('media_files', '[]') or '[]'
        page_range = f"p.{i_page}" if i_page == e_page else f"p.{i_page}~{e_page}"

        if view_mode == 'raw':
            raw_json = json.dumps(chunk, indent=2, ensure_ascii=False)
            return f'''
            <div class="chunk-card">
                <div class="chunk-header">
                    <span class="chunk-badge">#{chunk_idx + 1}</span>
                    <span class="page-badge">{page_range}</span>
                </div>
                <pre class="raw-json">{html.escape(raw_json)}</pre>
            </div>'''

        # 텍스트 렌더링
        text = chunk.get('text', '')
        text_html = self.render_markdown_table(text)

        # 미디어 파일 렌더링
        media_html = ''
        try:
            media_files = json.loads(media_files_raw)
            if media_files:
                items_html = ''.join(
                    f'<span class="media-tag">🖼️ {html.escape(m.get("name", ""))}</span>'
                    for m in media_files
                )
                media_html = f'<div class="media-files">{items_html}</div>'
        except Exception:
            pass

        return f'''
        <div class="chunk-card">
            <div class="chunk-header">
                <span class="chunk-badge">#{chunk_idx + 1} / {self.n_chunk_of_doc}</span>
                <span class="page-badge">{page_range}</span>
                <span class="meta-info">{n_char}자 · {n_word}단어 · {n_line}줄</span>
            </div>
            <div class="chunk-text">{text_html}</div>
            {media_html}
        </div>'''

    def generate_full_html(self, page_num: int, view_mode: str = 'formatted') -> str:
        page_num = max(1, min(page_num, self.total_pages))
        chunks_on_page = self.pages_content.get(page_num, [])
        content_body = ''.join(self.render_chunk(c, view_mode) for c in chunks_on_page)

        n_page = self.chunks[0].get('n_page', self.total_pages) if self.chunks else self.total_pages

        return f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Vectors Viewer - Page {page_num}</title>
    <style>
        body {{
            font-family: 'Pretendard', '맑은 고딕', sans-serif;
            line-height: 1.6;
            max-width: 960px;
            margin: 0 auto;
            padding: 20px;
            background: #f4f7f9;
        }}
        .controls {{
            background: white;
            padding: 15px 20px;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        }}
        .controls button {{
            padding: 8px 18px;
            cursor: pointer;
            background: #1a73e8;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 14px;
        }}
        .controls button:disabled {{ background: #ccc; cursor: not-allowed; }}
        .page-info {{ font-weight: bold; margin: 0 15px; color: #333; }}
        .mode-btn {{
            background: #eee;
            color: #333;
            margin-left: 5px;
            padding: 6px 14px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }}
        .mode-btn.active {{ background: #333; color: white; }}

        .chunk-card {{
            background: white;
            border-radius: 10px;
            padding: 20px 25px;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border-left: 4px solid #1a73e8;
        }}
        .chunk-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }}
        .chunk-badge {{
            background: #1a73e8;
            color: white;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .page-badge {{
            background: #e8f0fe;
            color: #1a73e8;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.8em;
        }}
        .meta-info {{
            color: #999;
            font-size: 0.8em;
        }}
        .chunk-text {{
            color: #3c4043;
            font-size: 0.95em;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .media-files {{
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .media-tag {{
            background: #fef3c7;
            color: #92400e;
            padding: 2px 8px;
            border-radius: 8px;
            font-size: 0.75em;
        }}
        .table-container {{
            overflow-x: auto;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88em;
        }}
        td, th {{
            border: 1px solid #dfe1e5;
            padding: 8px 10px;
            text-align: left;
        }}
        th {{ background: #f8f9fa; font-weight: bold; }}
        tr:nth-child(even) td {{ background: #fafafa; }}
        .raw-json {{
            background: #f8f9fa;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 12px;
            line-height: 1.4;
        }}
        .empty {{ text-align: center; color: #999; padding: 60px 0; }}
    </style>
</head>
<body>
    <div class="controls">
        <div>
            <button onclick="location.href='?page={page_num-1}&mode={view_mode}'" {'disabled' if page_num <= 1 else ''}>← 이전</button>
            <span class="page-info">페이지 {page_num} / {n_page} &nbsp;|&nbsp; 청크 {len(chunks_on_page)}개</span>
            <button onclick="location.href='?page={page_num+1}&mode={view_mode}'" {'disabled' if page_num >= self.total_pages else ''}>다음 →</button>
        </div>
        <div>
            <button class="mode-btn {'active' if view_mode == 'formatted' else ''}" onclick="location.href='?page={page_num}&mode=formatted'">서식</button>
            <button class="mode-btn {'active' if view_mode == 'raw' else ''}" onclick="location.href='?page={page_num}&mode=raw'">JSON</button>
        </div>
    </div>
    <div>
        {content_body if content_body else '<div class="empty">이 페이지에는 청크가 없습니다.</div>'}
    </div>
    <script>
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'ArrowLeft' && {page_num} > 1) location.href = '?page={page_num-1}&mode={view_mode}';
            if (e.key === 'ArrowRight' && {page_num} < {self.total_pages}) location.href = '?page={page_num+1}&mode={view_mode}';
        }});
    </script>
</body>
</html>'''


class VectorsHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        page = int(params.get('page', [1])[0])
        mode = params.get('mode', ['formatted'])[0]

        html_out = self.server.viewer.generate_full_html(page, mode)
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_out.encode('utf-8'))

    def log_message(self, format, *args):
        pass


def run_server(file_path, port=8000):
    viewer = VectorsWebViewer(file_path)
    if not viewer.pages_content:
        print("❌ 표시할 데이터가 없습니다.")
        return

    class ReusableServer(socketserver.TCPServer):
        allow_reuse_address = True

    for p in range(port, port + 10):
        try:
            server = ReusableServer(("", p), VectorsHandler)
            server.viewer = viewer
            print(f"🚀 Vectors 뷰어 서버 시작: http://localhost:{p}")
            webbrowser.open(f"http://localhost:{p}")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\n서버 종료")
            return
        except OSError:
            continue
    print(f"포트 {port}~{port+9} 모두 사용 중입니다.")


if __name__ == "__main__":
    path = input("vectors.json 경로 (또는 엔터키로 현재 경로의 'vectors.json' 사용): ").strip() or "vectors.json"
    if not os.path.exists(path):
        print(f"파일을 찾을 수 없습니다: {path}")
    else:
        run_server(path)
