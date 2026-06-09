"""prompt_files.py — enrichment prompt 를 별도 파일에서 읽기 위한 공용 유틸.

YAML config 안에 inline 으로 박혀 있던 prompt 본문을 별도 `.md` 파일로 분리하기 위해,
`*_file` 키가 가리키는 경로를 안전하게 해석(resolve)하고 읽어들이는 헬퍼를 제공한다.

경로 해석 규칙:
  - 절대경로: 운영자가 명시적으로 지정한 것으로 보고 그대로 사용 (containment 검사 생략).
    기존 external parser 파일(metadata/custom_fields)의 절대경로 허용 동작과 일치.
  - 상대경로: base_dir(보통 config yaml 이 위치한 디렉토리) 기준으로 해석하되,
    base_dir 를 벗어나는 경로(`../` 등)는 거부한다.
"""
from __future__ import annotations

from pathlib import Path


def resolve_prompt_path(file_ref: str, base_dir: Path) -> Path:
    """prompt 파일 경로 문자열을 안전한 절대경로로 해석한다.

    Args:
        file_ref: YAML 에 적힌 경로 문자열 (상대 또는 절대).
        base_dir: 상대경로 해석의 기준 디렉토리.

    Returns:
        해석된 절대 Path.

    Raises:
        ValueError: 상대경로가 base_dir 범위를 벗어난 경우.
    """
    ref_path = Path(file_ref)
    if ref_path.is_absolute():
        # 절대경로는 운영자의 명시적 opt-in 으로 간주하고 그대로 사용.
        return ref_path.resolve()

    base = base_dir.resolve()
    resolved = (base / ref_path).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(
            f"prompt 파일 경로가 허용 범위를 벗어났습니다: {resolved} (base: {base})"
        ) from exc
    return resolved


def read_prompt_file(file_ref: str, base_dir: Path) -> str:
    """prompt 파일을 해석·로드하여 문자열로 반환한다.

    Args:
        file_ref: YAML 에 적힌 경로 문자열.
        base_dir: 상대경로 해석 기준 디렉토리.

    Returns:
        파일 내용 (앞뒤 공백 strip).

    Raises:
        ValueError: 경로가 허용 범위를 벗어난 경우.
        FileNotFoundError: 파일이 존재하지 않는 경우.
    """
    path = resolve_prompt_path(file_ref, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"prompt 파일이 없습니다: {path}")
    return path.read_text(encoding="utf-8").strip()
