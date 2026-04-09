import json
import webbrowser
import os
import re
import html
import socketserver
from http.server import SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from collections import defaultdict

class DoclingDocumentWebViewer:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.pages_content = {}  # 🚀 에러 방지를 위해 이름을 'pages_content'로 고정
        self.total_pages = 0
        self.load_data()

    def load_data(self):
        """Docling JSON(texts, tables, pictures)을 읽어 페이지별로 분류합니다."""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            
            # 1. 데이터 카테고리별 로드
            ref_map = {
                "text": doc.get("texts", []),
                "table": doc.get("tables", []),
                "picture": doc.get("pictures", [])
            }

            temp_pages = defaultdict(list)

            # 2. 모든 아이템 순회하며 prov.page_no 추출
            for it_type, items in ref_map.items():
                for item in items:
                    # prov 정보 확인
                    prov = item.get("prov", [])
                    page_no = prov[0].get("page_no", 1) if prov else 1
                    
                    # 뷰어 렌더링용 데이터 구조화
                    temp_pages[page_no].append({
                        "type": it_type,
                        "data": item,  # Raw 모드용 전체 원본 데이터
                        "text": item.get("text", ""),
                        "label": item.get("label", "paragraph"),
                        "self_ref": item.get("self_ref", "")
                    })

            # 3. 각 페이지 내에서 self_ref 인덱스 순서대로 정렬
            for p_no in temp_pages:
                temp_pages[p_no].sort(key=lambda x: int(x["self_ref"].split("/")[-1]) if "/" in x["self_ref"] else 0)
            
            self.pages_content = dict(temp_pages)
            self.total_pages = max(self.pages_content.keys()) if self.pages_content else 0
            
            if self.total_pages > 0:
                print(f"✅ 데이터 로드 완료: {sum(len(v) for v in self.pages_content.values())}개 항목, {self.total_pages}페이지")
            return True
        except Exception as e:
            print(f"❌ 로드 실패: {e}")
            return False

    def render_item(self, item, view_mode):
        """서식 모드와 JSON(Raw) 모드를 구분하여 렌더링"""
        # 🚀 [복구]: 원본 JSON 보기 기능
        if view_mode == 'raw':
            raw_json = json.dumps(item["data"], indent=2, ensure_ascii=False)
            return f'<pre style="background:#f8f9fa; padding:15px; border:1px solid #ddd; overflow-x:auto; font-size:12px;">{html.escape(raw_json)}</pre>'

        res = ""
        it_type = item["type"]
        label = item["label"]
        text = item["text"]

        # 1. 테이블 처리
        if it_type == "table":
            res += f'<div class="item-label" style="font-size:0.7em; color:#999;">📊 {item["self_ref"]}</div>'
            res += self.render_docling_table(item["data"])
        
        # 🚀 2. 이미지(Picture) 처리 추가
        elif it_type == "picture":
            # Docling 구조에서 이미지 URI와 캡션 추출
            img_data = item["data"].get("image", {})
            img_uri = img_data.get("uri", "")
            
            # 캡션 텍스트 추출 (있을 경우)
            caption_dict = item["data"].get("caption", {})
            caption_text = caption_dict.get("text", "") if caption_dict else ""
            
            res += f'<div class="item-label" style="font-size:0.7em; color:#999;">🖼️ {item["self_ref"]}</div>'
            if img_uri:
                # 💡 주의: 브라우저에서 접근 가능한 상대 경로인지 확인 필요
                res += f'''
                <div class="image-container">
                    <img src="{img_uri}" alt="{html.escape(caption_text)}">
                    {f'<p class="image-caption">{html.escape(caption_text)}</p>' if caption_text else ''}
                </div>
                '''
            else:
                res += '<p class="para" style="color:red;">[이미지 데이터는 있으나 경로(URI)가 없습니다]</p>'

        # 3. 텍스트 처리
        else:
            clean_text = html.escape(text).replace("\n", "<br>")
            if label == "title":
                res += f'<h1 class="doc-title">{clean_text}</h1>'
            elif label == "section_header":
                res += f'<h2 class="doc-header">{clean_text} <span class="badge">{item["self_ref"]}</span></h2>'
            else:
                res += f'<p class="para">{clean_text}</p>'
        return res

    def render_docling_table(self, table_data):
        """Docling Grid 데이터를 HTML Table로 변환 (중복 렌더링 방지 로직 추가)"""
        grid = table_data.get("data", {}).get("grid", [])
        if not grid: return "<p>[표 데이터 없음]</p>"
        
        html_out = '<div class="table-container"><table>'
        for r_idx, row in enumerate(grid):
            html_out += "<tr>"
            for c_idx, cell in enumerate(row):
                # 🚀 핵심 수정 포인트:
                # 현재 좌표(r_idx, c_idx)가 셀의 시작 좌표(Top-Left)인 경우에만 <td>를 만듭니다.
                # 병합된 '그림자' 셀들은 시작 좌표가 현재 좌표보다 작을 것이므로 여기서 걸러집니다.
                if cell.get("start_row_offset_idx") == r_idx and \
                   cell.get("start_col_offset_idx") == c_idx:
                    
                    rs = cell.get("row_span", 1)
                    cs = cell.get("col_span", 1)
                    tag = "th" if cell.get("column_header") else "td"
                    
                    # 🚀 텍스트 내의 \n을 <br>로 바꿔서 줄바꿈 유지
                    txt = html.escape(cell.get("text", "")).replace("\n", "<br>")
                    
                    # 병합 속성 적용
                    html_out += f'<{tag} rowspan="{rs}" colspan="{cs}">{txt}</{tag}>'
            html_out += "</tr>"
        html_out += "</table></div>"
        return html_out

    def generate_full_html(self, page_num, view_mode='formatted'):
        items = self.pages_content.get(page_num, [])
        content_body = "".join([self.render_item(it, view_mode) for it in items])
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Docling Viewer - Page {page_num}</title>
            <style>
                body {{ font-family: 'Pretendard', sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; background: #f4f7f9; }}
                .controls {{ background: white; padding: 15px; border-radius: 10px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .content {{ background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); min-height: 700px; }}
                .doc-title {{ color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }}
                .doc-header {{ color: #202124; border-left: 5px solid #1a73e8; padding-left: 15px; margin-top: 30px; }}
                .para {{ margin: 15px 0; color: #3c4043; }}
                .badge {{ font-size: 0.6em; background: #eee; padding: 2px 5px; vertical-align: middle; }}
                .table-container {{ overflow-x: auto; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                td, th {{ border: 1px solid #dfe1e5; padding: 10px; font-size: 0.9em; }}
                th {{ background: #f8f9fa; font-weight: bold; }}
                button {{ padding: 8px 18px; cursor: pointer; background: #1a73e8; color: white; border: none; border-radius: 6px; }}
                button:disabled {{ background: #ccc; }}
                .mode-btn {{ background: #eee; color: #333; margin-left: 5px; }}
                .mode-btn.active {{ background: #333; color: white; }}
                
                /* 🚀 이미지 관련 스타일 추가 */
                .image-container {{
                    text-align: center;
                    margin: 30px 0;
                    padding: 10px;
                    background: #fdfdfd;
                    border: 1px solid #f0f0f0;
                    border-radius: 8px;
                }}
                .image-container img {{
                    max-width: 100%; /* 화면보다 크면 자동으로 줄여줌 (필수!) */
                    height: auto;
                    border-radius: 4px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                }}
                .image-caption {{
                    font-size: 0.85em;
                    color: #777;
                    margin-top: 10px;
                    font-style: italic;
                }}
                .item-label {{
                    font-family: monospace;
                    background: #f0f0f0;
                    padding: 2px 6px;
                    border-radius: 4px;
                    margin-bottom: 8px;
                    display: inline-block;
                }}
            </style>
        </head>
        <body>
            <div class="controls">
                <div>
                    <button onclick="location.href='?page={page_num-1}&mode={view_mode}'" {'disabled' if page_num<=1 else ''}>이전</button>
                    <span style="font-weight:bold; margin:0 15px;">Page {page_num} / {self.total_pages}</span>
                    <button onclick="location.href='?page={page_num+1}&mode={view_mode}'" {'disabled' if page_num>=self.total_pages else ''}>다음</button>
                </div>
                <div>
                    <button class="mode-btn {'active' if view_mode=='formatted' else ''}" onclick="location.href='?page={page_num}&mode=formatted'">서식</button>
                    <button class="mode-btn {'active' if view_mode=='raw' else ''}" onclick="location.href='?page={page_num}&mode=raw'">JSON</button>
                </div>
            </div>
            <div class="content">{content_body if content_body else "<p>데이터가 없습니다.</p>"}</div>
        </body>
        </html>
        '''

class DoclingHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
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
    # 🚀 클래스 명칭을 뷰어 내부와 일치시킴
    viewer = DoclingDocumentWebViewer(file_path)
    
    # 🚀 pages_content 존재 여부 체크
    if not viewer.pages_content: 
        print("❌ 표시할 페이지 데이터가 없습니다. (JSON 구조나 load_data 로직 확인 필요)")
        return

    class ReusableServer(socketserver.TCPServer):
        allow_reuse_address = True

    server = ReusableServer(("", 8000), DoclingHandler)
    server.viewer = viewer
    print(f"🚀 Docling 통합 뷰어 서버 시작: http://localhost:8000")
    webbrowser.open("http://localhost:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료")

if __name__ == "__main__":
    path = input("Docling JSON 경로 (또는 엔터키로 현재 경로의 'docling.json' 사용): ").strip() or "result_docling.json"
    run_server(path)