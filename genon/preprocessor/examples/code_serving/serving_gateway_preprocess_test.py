#!/usr/bin/env python3
"""main.py(/preprocess*) 게이트웨이 테스트 스크립트.

배포된 코드서빙(단일 서빙)의 게이트웨이를 통해 적재/첨부/변환용 전처리 엔드포인트를 호출한다.
게이트웨이 URL 패턴은 serving_gateway_test.py(파싱/청킹)와 동일하다:

    {base}/api/gateway/code_serving/{serving_id}/{route}
    헤더: Authorization: Bearer <auth_key>

대상 엔드포인트(route):
    - preprocess_attachment   : 첨부용
    - preprocess_intelligent  : 적재용(지능형)
    - preprocess_convert      : 변환용
    - preprocess              : intelligent 의 하위호환 별칭(bare 경로)

게이트웨이({base}/api/gateway/code_serving/{id}/{route})는 route 를 단일 세그먼트로만
포워딩하므로 평탄 경로(preprocess_*)를 사용한다(중첩 경로 /preprocess/xxx 는 호출 불가).

요청/응답:
    - 요청 본문: {"file_path": "<서버 기준 경로>", "params": {...}}
    - 응답 envelope: {"code":0,"errMsg":"success","data":[ {...청크(GenOSVectorMeta)...}, ... ]}
      세 엔드포인트 모두 data 는 청크 리스트(list[GenOSVectorMeta])다.

전제:
  - 단일 서빙(예: 139)이 /preprocess* 엔드포인트를 노출한다.
  - file_path 는 *서빙 컨테이너 내부의 로컬 경로*다(MinIO 키 아님).
    게이트웨이로 전처리를 테스트하려면 서버가 접근 가능한 경로여야 한다.

requests 등 외부 의존 없이 표준 라이브러리(urllib)만 사용한다.

실행 예:
    # 1) 헬스체크
    python serving_gateway_preprocess_test.py --mode health

    # 2) 첨부용 전처리
    python serving_gateway_preprocess_test.py --mode attachment \
        --file-path /app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf \
        --out /tmp/attachment.json

    # 3) 적재용(지능형) 전처리 (+ 추가 파라미터 예시)
    python serving_gateway_preprocess_test.py --mode intelligent \
        --file-path /data/documents/report.pdf --ocr-mode auto --chunk-size 4096

    # 4) 변환용 전처리
    python serving_gateway_preprocess_test.py --mode convert \
        --file-path /data/documents/report.docx

    # 5) 세 엔드포인트 순차 호출
    python serving_gateway_preprocess_test.py --mode all \
        --file-path /app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf

    # 6) 임의 파라미터 전달(반복 가능, 값은 JSON 파싱 시도)
    python serving_gateway_preprocess_test.py --mode intelligent \
        --file-path /data/report.pdf --param save_images=true --param chunk_overlap=120
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

# serving_gateway_test.py 와 동일한 기본값. 필요 시 CLI 인자로 덮어쓴다.
DEFAULT_BASE_URL = "https://genos.genon.ai"
DEFAULT_SERVING_ID = "139"
DEFAULT_AUTH_KEY = "b8c0b48f7b4d410699ed1aa8f2c0da8a"

# --mode → 게이트웨이 route 매핑. health 는 별도 처리.
# 게이트웨이는 route 를 단일 세그먼트로만 포워딩하므로 평탄 경로(preprocess_*)를 쓴다.
ROUTE_BY_MODE = {
    "attachment": "preprocess_attachment",
    "intelligent": "preprocess_intelligent",
    "convert": "preprocess_convert",
    "preprocess": "preprocess",  # intelligent 별칭(bare)
}


def _url(args, route: str) -> str:
    return f"{args.base_url.rstrip('/')}/api/gateway/code_serving/{args.serving_id}/{route}"


def _request(args, method: str, route: str, payload: dict | None = None):
    """게이트웨이로 요청하고 envelope(code/data)를 검사해 반환한다.

    - GET (예: health): envelope 가 아닐 수 있으므로 body 를 그대로 반환.
    - POST (preprocess*): {"code":0,"data":...} 를 검사해 data 를 반환.
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

    # health 처럼 envelope 가 아닌 응답은 그대로 반환.
    if not isinstance(body, dict) or "code" not in body:
        return body

    # 전처리기 응답 envelope: {"code": 0, "data": ..., "errMsg": ...}
    if body.get("code") != 0:
        err = body.get("errMsg") or body.get("error_msg") or body
        raise SystemExit(f"[{route}] 요청 실패 (code={body.get('code')}): {err}")
    return body.get("data")


def _json_output_path(raw_path: str, default_filename: str) -> Path:
    path = Path(raw_path).expanduser()
    if raw_path.endswith(("/", "\\")) or (path.exists() and path.is_dir()):
        path.mkdir(parents=True, exist_ok=True)
        return path / default_filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _parse_param(token: str):
    """--param key=value 토큰을 (key, value) 로 파싱한다. value 는 JSON 파싱 시도 후 실패 시 문자열."""
    if "=" not in token:
        raise SystemExit(f"--param 은 key=value 형식이어야 합니다: {token!r}")
    key, _, raw = token.partition("=")
    key = key.strip()
    if not key:
        raise SystemExit(f"--param 의 key 가 비어 있습니다: {token!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def _build_params(args) -> dict:
    """CLI 인자로부터 params 객체를 조립한다(빈 값은 제외)."""
    params: dict = {}
    if args.chunk_size is not None:
        params["chunk_size"] = args.chunk_size
    if args.ocr_mode:
        params["ocr_mode"] = args.ocr_mode
    for token in args.param or []:
        key, value = _parse_param(token)
        params[key] = value
    return params


def do_health(args) -> int:
    body = _request(args, "GET", "health")
    print(f"[health] {json.dumps(body, ensure_ascii=False)}")
    return 0


def do_preprocess(args, mode: str) -> list:
    """전처리 엔드포인트 호출 → 청크(GenOSVectorMeta) 리스트 반환."""
    if not args.file_path:
        raise SystemExit("--file-path 가 필요합니다(서버가 접근 가능한 경로).")
    route = ROUTE_BY_MODE[mode]
    params = _build_params(args)
    data = _request(
        args, "POST", route,
        payload={"file_path": args.file_path, "params": params},
    )
    chunks = data or []
    if not isinstance(chunks, list):
        # 혹시 dict 등 다른 형태로 오면 그대로 출력하고 빈 리스트 반환.
        print(f"[{mode}] data 가 리스트가 아닙니다: {json.dumps(chunks, ensure_ascii=False)[:200]}")
        return chunks

    first = chunks[0] if chunks else {}
    n_page = first.get("n_page")
    print(f"[{mode}] 청크 {len(chunks)}개 생성 (route=/{route}, pages={n_page})")
    for chunk in chunks[:5]:
        text = str(chunk.get("text", "")).replace("\n", " ")
        print(f"  - [{chunk.get('i_chunk_on_doc')}] page={chunk.get('i_page')} {text[:80]}")
    if len(chunks) > 5:
        print(f"  ... (+{len(chunks) - 5}개)")

    if args.out:
        out_path = _json_output_path(args.out, f"{mode}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"[{mode}] 청크 {len(chunks)}개 저장 → {out_path}")
    return chunks


def do_all(args) -> int:
    for mode in ("attachment", "intelligent", "convert"):
        do_preprocess(args, mode)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="main.py /preprocess* 게이트웨이 테스트(첨부/적재/변환)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",
                   choices=["health", "attachment", "intelligent", "convert", "preprocess", "all"],
                   default="all", help="실행 모드(엔드포인트). all=attachment·intelligent·convert 순차")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="게이트웨이 base URL")
    p.add_argument("--serving-id", default=DEFAULT_SERVING_ID, help="코드서빙 id")
    p.add_argument("--auth-key", default=DEFAULT_AUTH_KEY, help="Authorization: Bearer <key>")
    p.add_argument("--file-path", default="",
                   help="전처리할 문서 경로(서버 기준). health 외 모든 모드에 필요")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="청크 최대 크기(미지정 시 서빙 config 기본값 사용)")
    p.add_argument("--ocr-mode", default="",
                   help='OCR 모드: "auto"|"force"|"disable"(미지정 시 서빙 config 기본값 사용)')
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="추가 params 항목(반복 가능). 값은 JSON 파싱 시도 후 실패 시 문자열")
    p.add_argument("--out", default=None, help="결과 청크 JSON 저장 경로 또는 디렉터리(옵션)")
    p.add_argument("--timeout", type=float, default=3600.0, help="요청 타임아웃(초)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "health":
        return do_health(args)
    if args.mode == "all":
        return do_all(args)
    do_preprocess(args, args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
