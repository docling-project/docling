"""서버 없이(in-process) 파싱(docling) → 청킹 테스트 러너.

shkim_labs/test.py 와 동일한 in-process 패턴(mock_request + await doc_processor(...))을 따른다.
facade 의 DocumentProcessor 를 직접 import 해 호출하므로 uvicorn/게이트웨이 불필요.

사용:
    # 풀 E2E (PDF/문서 → 파싱(docling) → 청킹).  파싱은 layout/OCR 모델서빙 필요.
    python parse_chunk_test.py <input.pdf|dir> <output_dir> [--chunk-size N]

    # docling JSON 입력 → 청킹만 (모델서버 불필요)
    python parse_chunk_test.py <doc.json> <output_dir> [--chunk-size N]
      - doc.json 은 parser(output.format=docling) 의 data.document, 또는 그 응답 전체({"document": ...}),
        또는 DoclingDocument.model_dump(mode="json") 결과 어느 쪽이든 허용.

    # 비-docling 포맷(csv/xlsx/txt/md/ppt/pptx/이미지/오디오 등) → 파싱(parse-format)→공통 청킹
    python parse_chunk_test.py <input.csv|dir> <output_dir> [--chunk-size N]
      - parser 가 docling 을 못 만드는 포맷은 {"elements":[...]} parse-format 을 반환하고,
        chunker 가 이를 legacy(attachment) 와 동일하게 공통 청킹한다.
"""
import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

from fastapi import Request

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
PREPROCESSOR_SRC = PROJECT_ROOT / "genon" / "preprocessor" / "src"
for path in (PREPROCESSOR_SRC, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)  # doc_parser 루트 / preprocessor src 참조

# in-process 테스트라 코드서빙 단일 마운트 제약과 무관 → 두 facade 동시 import 가능.
from genon.preprocessor.facade.parser_processor import DocumentProcessor as ParserProcessor
from genon.preprocessor.facade.chunking_processor import DocumentProcessor as ChunkerProcessor

mock_request = Request(scope={"type": "http"})

# 파싱 경로(docling) 확장자 + docling JSON 입력
PARSE_EXTENSIONS = {".pdf", ".docx", ".hwp", ".hwpx", ".html", ".htm"}
# parser 가 docling 을 못 만드는 포맷 → parse-format({"elements":[...]}) → 공통 청킹
NONDOCLING_EXTENSIONS = {
    ".csv", ".xlsx", ".txt", ".md", ".ppt", ".pptx", ".doc",
    ".jpg", ".jpeg", ".png", ".wav", ".mp3", ".m4a",
}
SUPPORTED_EXTENSIONS = PARSE_EXTENSIONS | NONDOCLING_EXTENSIONS | {".json"}

# 지연 인스턴스화 (파싱이 필요할 때만 ParserProcessor 생성)
_parser: ParserProcessor | None = None
_chunker: ChunkerProcessor | None = None


def get_parser() -> ParserProcessor:
    global _parser
    if _parser is None:
        _parser = ParserProcessor()
        _parser._output_format = "docling"  # config 편집 없이 docling 출력 강제
    return _parser


def get_chunker() -> ChunkerProcessor:
    global _chunker
    if _chunker is None:
        _chunker = ChunkerProcessor()
    return _chunker


def load_docling_json(path: Path) -> dict:
    """docling JSON 입력을 dict 로 로드. {"document": ...} 래핑/비래핑 모두 허용."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "document" in obj and isinstance(obj["document"], dict):
        return obj["document"]
    return obj


async def parse_document(file_path: Path, kwargs: dict) -> dict:
    """파싱 실행 → parser 응답 dict 전체.

    docling 경로면 {"document": {...}, ...}, 비-docling 경로면 {"elements": [...], ...}.
    """
    return await get_parser()(mock_request, str(file_path), **kwargs)


async def chunk_payload(file_path: Path, payload: dict, chunk_size: int, chunk_mode: str,
                        cache_kwargs: dict | None = None) -> list[dict]:
    """청킹 실행 → GenOSVectorMeta dict 리스트.

    payload 는 docling({"document":...}) 또는 parse-format({"elements":...}) 어느 쪽이든 허용.
    chunker 가 형태를 스스로 판별한다(file_path 확장자 무관).
    #329 스코프(cache_kwargs)는 청킹엔 LLM 이 없어 no-op 이지만 API 일관성 위해 전달(내부에서 pop).
    """
    vectors = await get_chunker()(
        mock_request, str(file_path), document=payload, chunk_size=chunk_size, chunk_mode=chunk_mode,
        **(cache_kwargs or {})
    )
    return [v.model_dump() if hasattr(v, "model_dump") else v for v in vectors]


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  saved: {path}")


def collect_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"지원하지 않는 형식: {input_path.suffix} (지원: {', '.join(sorted(SUPPORTED_EXTENSIONS))})")
            raise SystemExit(1)
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print(f"처리 가능한 파일이 없습니다: {input_path}")
            raise SystemExit(1)
        return files
    print(f"입력 경로를 찾을 수 없습니다: {input_path}")
    raise SystemExit(1)


async def process_one(file_path: Path, out_base: Path, chunk_size: int, chunk_mode: str,
                      cache_kwargs: dict | None = None) -> None:
    """파일 1개: (파싱→)청킹 수행 후 결과 저장."""
    cache_kwargs = cache_kwargs or {}
    is_json = file_path.suffix.lower() == ".json"

    if is_json:
        # .json 입력은 docling JSON(청킹만) 으로 해석.
        payload = load_docling_json(file_path)
        print("  [parse] skip (docling JSON 입력)")
    else:
        # #329 스코프를 parse kwargs 에 병합(캐시는 parse 단계 LLM 호출에서 동작).
        kwargs = {"org_filename": file_path.name, "log_level": 5, **cache_kwargs}
        payload = await parse_document(file_path, kwargs)
        if isinstance(payload, dict) and isinstance(payload.get("document"), dict):
            kind = "docling"
            save_json(out_base.with_suffix(".docling.json"), payload["document"])
        else:
            kind = "parse"
            n_elems = len(payload.get("elements", []) or []) if isinstance(payload, dict) else 0
            save_json(out_base.with_suffix(".parse.json"), payload)
            print(f"  [parse] parse-format ({n_elems} elements) → 공통 청킹")

    vectors = await chunk_payload(file_path, payload, chunk_size, chunk_mode, cache_kwargs)
    save_json(out_base.with_suffix(".chunks.json"), vectors)
    print(f"  [chunk] {len(vectors)} chunks")


def parse_args():
    ap = argparse.ArgumentParser(description="in-process 파싱→청킹 테스트(docling + parse-format 공통)")
    ap.add_argument(
        "input_path",
        help="입력 파일/디렉터리 (docling: PDF/DOCX/HWP/HWPX/HTML 또는 docling .json | "
             "parse-format: CSV/XLSX/TXT/MD/PPT/PPTX/이미지/오디오)",
    )
    ap.add_argument("output_dir", help="결과 저장 디렉터리")
    ap.add_argument("--chunk-size", type=int, default=0, help="청크 최대 크기 (0=토큰/문자 분할 안 함, 0 초과 시 최소 1024)")
    ap.add_argument(
        "--chunk-mode",
        choices=["split_only", "resize_all"],
        default="split_only",
        help="split_only=chunk_size 초과 청크만 분할(기본) | resize_all=모든 청크를 chunk_size 에 맞게 병합/분할",
    )
    # ── #329: LLM 캐시 / error_policy / deadline (opt-in) ──────────────────────
    # 캐시는 parse 단계(LLM 호출: OCR VLM/TOC/이미지·표 desc/메타데이터)에서 동작한다.
    # chunk 단계는 LLM 호출이 없어 스코프만 전달(no-op). 두 단계 모두 같은 스코프를 준다.
    ap.add_argument(
        "--llm_cache",
        action="store_true",
        help="LLM 호출 입출력을 파일 캐시(재실행 시 재사용). workflow_id + interim_root(or env) 필요",
    )
    ap.add_argument(
        "--interim_root",
        default=None,
        help="캐시 루트(<interim_root>/<workflow_id>/<run_id>/llm_cache/). 미지정 시 env INTERIM_ROOT",
    )
    ap.add_argument("--workflow_id", default=None, help="캐시 스코프 workflow_id (재실행 간 동일해야 재사용)")
    ap.add_argument("--run_id", default=None, help="캐시 스코프 run_id (미지정 시 'default')")
    ap.add_argument(
        "--error_policy",
        choices=["lenient", "strict"],
        default=None,
        help="enrichment 실패 처리: lenient(기본, soft-fail) | strict(실패 시 예외 전파)",
    )
    ap.add_argument("--request_deadline", type=float, default=None, help="요청 전체 deadline(초). 초과 시 timeout")
    return ap.parse_args()


def build_cache_kwargs(args) -> dict:
    """#329 파라미터를 kwargs dict 로 (지정된 것만). parse/chunk 양쪽에 동일하게 전달."""
    kw: dict = {}
    if getattr(args, "llm_cache", False):
        kw["llm_cache"] = 1
    if getattr(args, "interim_root", None):
        kw["interim_root"] = args.interim_root
    if getattr(args, "workflow_id", None):
        kw["workflow_id"] = args.workflow_id
    if getattr(args, "run_id", None):
        kw["run_id"] = args.run_id
    if getattr(args, "error_policy", None):
        kw["error_policy"] = args.error_policy
    if getattr(args, "request_deadline", None) is not None:
        kw["request_deadline"] = args.request_deadline
    return kw


def main():
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    files = collect_files(input_path)
    is_dir = input_path.is_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_kwargs = build_cache_kwargs(args)

    for idx, file_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {file_path}")
        # 출력 베이스 경로(.docling.json / .chunks.json 접미사가 붙음)
        if is_dir:
            out_base = output_dir / file_path.relative_to(input_path)
        else:
            out_base = output_dir / file_path.name
        asyncio.run(process_one(file_path, out_base, args.chunk_size, args.chunk_mode, cache_kwargs))


if __name__ == "__main__":
    main()
