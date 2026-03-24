import json
import webbrowser
import os
import re
import html
import socketserver
from http.server import SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

class DoclingReportWebViewer:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.data = []
        self.total_pages = 0
        self.load_data()

    def load_data(self):
        """Docling JSON 데이터 로드 및 중첩 구조 해제"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Docling 특유의 문자열화된 JSON 필드들 복구
            for item in raw_data:
                if isinstance(item.get('chunk_bboxes'), str):
                    try:
                        item['chunk_bboxes'] = json.loads(item['chunk_bboxes'])
                    except: item['chunk_bboxes'] = []
                if isinstance(item.get('media_files'), str):
                    try:
                        item['media_files'] = json.loads(item['media_files'])
                    except: item['media_files'] = []
            
            self.data = raw_data
            
            # 페이지 정보 추출 (Docling은 i_page 필드 사용)
            pages = set(item.get('i_page', 1) for item in self.data)
            self.original_pages = sorted(list(pages)) if pages else [1]
            self.total_pages = len(self.original_pages)
            self.page_mapping = {i+1: page for i, page in enumerate(self.original_pages)}
            
            print(f"✅ 데이터 로드 완료: {len(self.data)}개 항목, {self.total_pages}페이지")
            return True
        except Exception as e:
            print(f"❌ 로드 실패: {e}")
            return False

    def markdown_to_html_table(self, md_text):
        """마크다운 표를 HTML로 변환"""
        lines = [l.strip() for l in md_text.strip().split('\n') if '|' in l]
        if not lines: return f"<pre>{html.escape(md_text)}</pre>"
        
        table_html = '<div class="table-container"><table>'
        for i, line in enumerate(lines):
            if '---|' in line or ':---|' in line: continue
            cells = [c.strip() for c in line.split('|') if c.strip()]
            tag = "th" if i == 0 else "td"
            table_html += "<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>"
        table_html += "</table></div>"
        return table_html

    def render_item(self, item, view_mode):
        """Docling 아이템의 타입에 따른 렌더링"""
        text = item.get('text', '')
        bboxes = item.get('chunk_bboxes', [])
        # 가장 첫 번째 좌표의 타입을 기준으로 삼음
        item_type = bboxes[0].get('type', 'paragraph') if bboxes else 'paragraph'
        media = item.get('media_files', [])

        if view_mode == 'raw':
            return f'<pre>{html.escape(json.dumps(item, indent=2, ensure_ascii=False))}</pre>'

        res = ""
        # 1. 이미지 처리 (타입이 picture거나 media_files가 있을 때)
        if media:
            for m in media:
                img_name = m.get('name', '')
                res += f'''<div class="media-box">
                    <img src="{img_name}" onerror="this.src='https://via.placeholder.com/400x200?text=Image+Not+Found'">
                    <p class="caption">🖼️ {img_name}</p>
                </div>'''

        # --- 핵심 수정 부분: 백슬래시 에러 방지를 위해 미리 이스케이프 및 변환 ---
        display_text = html.escape(text).replace("\n", "<br>")

        # 2. 텍스트 타입별 처리
        if item_type == 'table':
            res += self.markdown_to_html_table(text)
        elif item_type == 'title':
            res += f'<h1 class="doc-title">{html.escape(text)}</h1>'
        elif item_type == 'section_header':
            res += f'<h2 class="doc-header">{html.escape(text)}</h2>'
        elif item_type == 'list_item':
            # f-string 밖에서 처리한 display_text 사용
            res += f'<div class="list-item">• {display_text}</div>'
        else:
            # f-string 밖에서 처리한 display_text 사용
            res += f'<p class="para">{display_text}</p>'
        
        return res

    def generate_full_html(self, display_page_num, view_mode='formatted'):
        actual_page = self.page_mapping.get(display_page_num, 1)
        page_items = [item for item in self.data if item.get('i_page') == actual_page]
        
        content_body = "".join([self.render_item(it, view_mode) for it in page_items])
        
        # --- HTML 템플릿 (기존 스타일 유지) ---
        return f'''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Docling 분석 뷰어 - Page {display_page_num}</title>
    <style>
        body {{ font-family: 'Malgun Gothic', sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f0f2f5; }}
        .header {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px; }}
        .controls {{ background: white; padding: 15px; border-radius: 10px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .content {{ background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); min-height: 600px; }}
        
        /* 타입별 스타일 */
        .doc-title {{ color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; margin-top: 0; }}
        .doc-header {{ color: #333; border-left: 6px solid #1a73e8; padding-left: 15px; margin-top: 30px; }}
        .para {{ margin: 15px 0; color: #444; }}
        .list-item {{ margin-left: 20px; color: #555; position: relative; padding: 5px 0; }}
        
        .table-container {{ overflow-x: auto; margin: 25px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f8f9fa; }}
        
        .media-box {{ text-align: center; margin: 30px 0; padding: 20px; background: #fafafa; border: 1px dashed #bbb; border-radius: 8px; }}
        .media-box img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .caption {{ font-size: 13px; color: #777; margin-top: 10px; }}

        /* 버튼 스타일 */
        button {{ padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; background: #1a73e8; color: white; }}
        button:disabled {{ background: #ccc; }}
        .mode-btn {{ background: #eee; color: #333; margin-left: 5px; }}
        .mode-btn.active {{ background: #333; color: white; }}
    </style>
</head>
<body>
    <div class="header"><h1>📑 Docling Document Viewer</h1><p>AI 기반 문서 구조 분석</p></div>
    <div class="controls">
        <div>
            <button onclick="move({display_page_num-1})" {"disabled" if display_page_num<=1 else ""}>이전</button>
            <span style="font-weight:bold; margin:0 15px;">Page {display_page_num} / {self.total_pages} (원본: {actual_page})</span>
            <button onclick="move({display_page_num+1})" {"disabled" if display_page_num>=self.total_pages else ""}>다음</button>
        </div>
        <div>
            <button class="mode-btn {'active' if view_mode=='formatted' else ''}" onclick="setMode('formatted')">서식</button>
            <button class="mode-btn {'active' if view_mode=='raw' else ''}" onclick="setMode('raw')">JSON</button>
        </div>
    </div>
    <div class="content">{content_body}</div>
    <script>
        function move(p) {{ window.location.href = `?page=${{p}}&mode=${{getMode()}}`; }}
        function setMode(m) {{ window.location.href = `?page=${{getPage()}}&mode=${{m}}`; }}
        function getPage() {{ return new URLSearchParams(window.location.search).get('page') || 1; }}
        function getMode() {{ return new URLSearchParams(window.location.search).get('mode') || 'formatted'; }}
        document.onkeydown = (e) => {{
            if(e.key === 'ArrowLeft') move({display_page_num-1});
            if(e.key === 'ArrowRight') move({display_page_num+1});
        }};
    </script>
</body>
</html>
'''

class DoclingHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        # 이미지 파일 서빙 (로컬 파일)
        if url.path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            return super().do_GET()
            
        params = parse_qs(url.query)
        page = int(params.get('page', [1])[0])
        mode = params.get('mode', ['formatted'])[0]
        
        html_out = self.server.viewer.generate_full_html(page, mode)
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_out.encode('utf-8'))

def run_server(file_path):
    viewer = DoclingReportWebViewer(file_path)
    if not viewer.data: return

    class ReusableServer(socketserver.TCPServer):
        allow_reuse_address = True

    server = ReusableServer(("", 8000), DoclingHandler)
    server.viewer = viewer
    print(f"🚀 뷰어 서버 시작: http://localhost:8000")
    webbrowser.open("http://localhost:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료")

if __name__ == "__main__":
    path = input("JSON 경로 (엔터: result.json): ").strip() or "result.json"
    run_server(path)