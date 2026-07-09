#!/usr/bin/env python3
"""main.py(/health, /parser, /chunker) 게이트웨이 테스트 스크립트.

배포된 코드서빙(단일 서빙)의 게이트웨이를 통해 파싱·청킹 엔드포인트를 호출한다.
게이트웨이 URL 패턴은 health curl 과 동일하다:

    {base}/api/gateway/code_serving/{serving_id}/{route}
    헤더: Authorization: Bearer <auth_key>

흐름(E2E):
    1) POST /parser  {file_path, params:{}}
         → docling 포맷:    data.document = DoclingDocument JSON
         → 비-docling 포맷:  data.elements = parse-format(csv/xlsx/txt/md/ppt/이미지/오디오 등)
    2) POST /chunker {file_path, params:{document, chunk_size}} → data = 청크 리스트
       (params.document 에 docling 문서 또는 parse-format(data) 어느 쪽이든 넣으면 chunker 가
        형태를 스스로 판별해 청킹한다.)

전제:
  - 단일 서빙(예: 139)이 /parser 와 /chunker 를 모두 노출한다.
  - docling 포맷(pdf/docx/hwp/hwpx/html)은 파싱 서빙 config 가 output.format: "docling" 이어야
    data.document 가 생긴다. 그 외 포맷은 자동으로 parse-format(data.elements)으로 반환된다.
  - /parser 의 file_path 는 *서빙 컨테이너 내부의 로컬 경로*다(MinIO 키 아님).
    게이트웨이로 파싱을 테스트하려면 서버가 접근 가능한 경로여야 한다.

requests 등 외부 의존 없이 표준 라이브러리(urllib)만 사용한다.

실행 예:
    # 1) 헬스체크
    python serving_gateway_test.py --mode health

    # 2) 파싱 → 청킹 E2E (서버 접근 가능 경로 필요)
    python serving_gateway_test.py --mode e2e \
        --file-path /data/documents/report.pdf \
        --out /tmp/chunks.json

    # 3) 파싱만 (docling JSON 저장)
    python serving_gateway_test.py --mode parser \
        --file-path /data/documents/report.pdf --out-doc /tmp/doc.json

    # 4) 청킹만 (저장해둔 docling JSON 사용)
    python serving_gateway_test.py --mode chunker --doc-json /tmp/doc.json

    # 5) 업로드 파싱 (클라이언트 로컬 파일 → multipart /parser_upload)
    python serving_gateway_test.py --mode parser_upload \
        --upload-file ./report.pdf --out-doc /tmp/doc.json

    # 6) 업로드 → 청킹 E2E (--upload-file 지정 시 e2e 가 업로드 파싱을 사용)
    python serving_gateway_test.py --mode e2e \
        --upload-file ./report.pdf --out /tmp/chunks.json

    # 7) 차트 description 검증 (kwargs 오버라이드). chart_detection=1=auto(docling 자동판별)/0=all
    python serving_gateway_test.py --mode parser_upload --upload-file ./chart.pdf \
        --param img_desc=1 --param chart_desc=1 --param chart_detection=1 \
        --param doc_summary=1 --out-doc /tmp/doc.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
import uuid
from pathlib import Path

# health curl 에서 가져온 기본값. 필요 시 CLI 인자로 덮어쓴다.
DEFAULT_BASE_URL = "https://genos.genon.ai"
DEFAULT_SERVING_ID = "139"
DEFAULT_AUTH_KEY = "b8c0b48f7b4d410699ed1aa8f2c0da8a"


def _url(args, route: str) -> str:
    return f"{args.base_url.rstrip('/')}/api/gateway/code_serving/{args.serving_id}/{route}"


def _request(args, method: str, route: str, payload: dict | None = None):
    """게이트웨이로 요청하고 envelope(code/data)를 검사해 반환한다.

    - GET (예: health): envelope 가 아닐 수 있으므로 body 를 그대로 반환.
    - POST (parser/chunker): {"code":0,"data":...} 를 검사해 data 를 반환.
    """
    url = _url(args, route)
    data_bytes = None
    headers = {
        "Authorization": f"Bearer {args.auth_key}",
        "Content-Type": "application/json",
    }
    if payload is not None:
        data_bytes = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"[{route}] HTTP {e.code} 오류: {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"[{route}] 연결 실패: {e.reason}")

    return _check_envelope(route, body)


def _check_envelope(route: str, body):
    """전처리기 응답 envelope({"code":0,"data":...})를 검사해 data 를 반환한다.

    health 처럼 envelope 가 아닌 응답은 그대로 반환하고, code!=0 이면 SystemExit.
    """
    if not isinstance(body, dict) or "code" not in body:
        return body
    if body.get("code") != 0:
        err = body.get("errMsg") or body.get("error_msg") or body
        raise SystemExit(f"[{route}] 요청 실패 (code={body.get('code')}): {err}")
    return body.get("data")


def _encode_multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """multipart/form-data 본문을 stdlib 만으로 인코딩한다.

    fields: name -> str
    files:  name -> (filename, bytes, content_type)
    반환: (body, boundary)
    """
    boundary = "----doc-parser-" + uuid.uuid4().hex
    buf = bytearray()
    for name, value in fields.items():
        buf += f"--{boundary}\r\n".encode()
        buf += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        buf += str(value).encode("utf-8") + b"\r\n"
    for name, (filename, content, ctype) in files.items():
        buf += f"--{boundary}\r\n".encode()
        buf += f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        buf += f"Content-Type: {ctype}\r\n\r\n".encode()
        buf += content + b"\r\n"
    buf += f"--{boundary}--\r\n".encode()
    return bytes(buf), boundary


def _request_multipart(args, route: str, fields: dict, files: dict):
    """multipart/form-data 로 게이트웨이에 POST 하고 envelope 의 data 를 반환한다."""
    url = _url(args, route)
    body, boundary = _encode_multipart(fields, files)
    headers = {
        "Authorization": f"Bearer {args.auth_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body_json = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"[{route}] HTTP {e.code} 오류: {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"[{route}] 연결 실패: {e.reason}")

    return _check_envelope(route, body_json)


def _json_output_path(raw_path: str, default_filename: str) -> Path:
    path = Path(raw_path).expanduser()
    if raw_path.endswith(("/", "\\")) or (path.exists() and path.is_dir()):
        path.mkdir(parents=True, exist_ok=True)
        return path / default_filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def do_health(args) -> int:
    body = _request(args, "GET", "health")
    print(f"[health] {json.dumps(body, ensure_ascii=False)}")
    return 0


def _handle_parser_data(args, data) -> dict:
    """파싱 응답(data)을 검증·출력하고(옵션 저장) chunker 로 forward 할 payload 를 반환한다.

    docling(`data.document`) 과 parse-format(`data.elements`; 비-docling 포맷) 모두 지원한다.
    chunker 는 payload 형태를 스스로 판별하므로 두 경우 모두 그대로 넘기면 된다.
    /parser 와 /parser_upload 가 공유한다.
    """
    data = data or {}
    document = data.get("document")
    if document:
        # docling 경로
        pages = (data.get("usage") or {}).get("pages")
        print(f"[parser] docling 문서 수신 — pages={pages}, top-level keys={list(document.keys())[:8]}")
        if args.out_doc:
            out_doc_path = _json_output_path(args.out_doc, "docling.json")
            with open(out_doc_path, "w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            print(f"[parser] docling JSON 저장 → {out_doc_path}")
        return document

    if isinstance(data.get("elements"), list):
        # parse-format(비-docling) 경로 — data 전체(elements 포함)를 forward.
        n_elems = len(data["elements"])
        pages = (data.get("usage") or {}).get("pages")
        print(f"[parser] parse-format 수신 — elements={n_elems}, pages={pages} (비-docling → 공통 청킹)")
        if args.out_doc:
            out_doc_path = _json_output_path(args.out_doc, "parse.json")
            with open(out_doc_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[parser] parse-format JSON 저장 → {out_doc_path}")
        return data

    raise SystemExit(
        "파싱 응답에 data.document 도 data.elements 도 없습니다. 파싱 서빙 응답을 확인하세요."
    )


def _extra_params(args) -> dict:
    """--param KEY=VALUE (반복) 를 kwargs dict 로 변환. 값은 int 로 우선 해석.

    예: --param img_desc=1 --param chart_desc=1 --param chart_detection=1 --param doc_summary=1
    """
    params: dict = {}
    for raw in getattr(args, "param", None) or []:
        if "=" not in raw:
            raise SystemExit(f"--param 은 KEY=VALUE 형식이어야 합니다: {raw!r}")
        key, _, value = raw.partition("=")
        key = key.strip()
        value = value.strip()
        try:
            params[key] = int(value)
        except ValueError:
            params[key] = value
    return params


def do_parser(args) -> dict:
    """파싱 서빙 호출(서버 로컬 경로) → DoclingDocument JSON(data.document) 반환."""
    if not args.file_path:
        raise SystemExit("--file-path 가 필요합니다(서버가 접근 가능한 경로).")
    data = _request(
        args, "POST", "parser",
        payload={"file_path": args.file_path, "params": _extra_params(args)},
    )
    return _handle_parser_data(args, data)


def do_parser_upload(args) -> dict:
    """업로드 파싱 호출(클라이언트 로컬 파일 → multipart /parser_upload) → 문서 반환."""
    if not args.upload_file:
        raise SystemExit("--upload-file 가 필요합니다(이 스크립트를 실행하는 로컬 파일 경로).")
    path = Path(args.upload_file).expanduser()
    if not path.is_file():
        raise SystemExit(f"업로드할 파일이 없습니다: {path}")
    print(f"[parser_upload] 업로드 → {path.name} ({path.stat().st_size} bytes)")
    data = _request_multipart(
        args, "parser_upload",
        fields={"params": json.dumps(_extra_params(args))},
        files={"file": (path.name, path.read_bytes(), "application/octet-stream")},
    )
    return _handle_parser_data(args, data)


def do_chunker(args, document: dict | None = None) -> list:
    """청킹 서빙 호출 → 청크(GenOSVectorMeta) 리스트 반환."""
    if document is None:
        if not args.doc_json:
            raise SystemExit("chunker 모드에는 --doc-json <docling JSON 파일> 이 필요합니다.")
        with open(args.doc_json, "r", encoding="utf-8") as f:
            document = json.load(f)
        # parser 응답을 통째로 저장한 경우({"document":...}) 도 허용.
        if isinstance(document, dict) and "document" in document and "schema_name" not in document:
            document = document["document"]

    params: dict = {"document": document}
    if args.chunk_size is not None:
        params["chunk_size"] = args.chunk_size
    data = _request(
        args, "POST", "chunker",
        payload={"file_path": "", "params": params},
    )
    chunks = data or []
    print(f"[chunker] 청크 {len(chunks)}개 생성")
    for chunk in chunks:
        text = str(chunk.get("text", "")).replace("\n", " ")
        print(f"  - [{chunk.get('i_chunk_on_doc')}] page={chunk.get('i_page')} {text[:80]}")
    if args.out:
        out_path = _json_output_path(args.out, "chunks.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"[chunker] 청크 {len(chunks)}개 저장 → {out_path}")
    return chunks


def do_e2e(args) -> int:
    # --upload-file 가 있으면 업로드 파싱으로, 없으면 서버 로컬 경로 파싱으로 진행.
    document = do_parser_upload(args) if args.upload_file else do_parser(args)
    do_chunker(args, document)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="main.py /health·/parser·/chunker 게이트웨이 테스트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["health", "parser", "parser_upload", "chunker", "e2e"],
                   default="e2e", help="실행 모드")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="게이트웨이 base URL")
    p.add_argument("--serving-id", default=DEFAULT_SERVING_ID, help="코드서빙 id")
    p.add_argument("--auth-key", default=DEFAULT_AUTH_KEY, help="Authorization: Bearer <key>")
    p.add_argument("--file-path", default="",
                   help="파싱할 문서 경로(서버 기준). parser/e2e 에 필요")
    p.add_argument("--upload-file", default="",
                   help="업로드할 로컬 파일 경로(스크립트 실행 호스트 기준). parser_upload 에 필요. "
                        "e2e 에 지정하면 업로드 파싱을 사용")
    p.add_argument("--chunk-size", type=int, default=0,
                   help="청크 최대 크기(0=토큰/문자 기반 분할 안 함). 청킹 config 기본값을 덮어씀")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="파싱 kwargs 오버라이드(반복 가능). "
                        "예: --param img_desc=1 --param chart_desc=1 --param chart_detection=1 --param doc_summary=1")
    p.add_argument("--doc-json", default=None, help="chunker 모드: 입력 docling JSON 파일 경로")
    p.add_argument("--out", default=None, help="청크 결과 JSON 저장 경로 또는 디렉터리(옵션)")
    p.add_argument("--out-doc", default=None, help="parser 모드: docling JSON 저장 경로 또는 디렉터리(옵션)")
    p.add_argument("--timeout", type=float, default=3600.0, help="요청 타임아웃(초)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "health":
        return do_health(args)
    if args.mode == "parser":
        do_parser(args)
        return 0
    if args.mode == "parser_upload":
        do_parser_upload(args)
        return 0
    if args.mode == "chunker":
        do_chunker(args)
        return 0
    return do_e2e(args)


if __name__ == "__main__":
    sys.exit(main())
