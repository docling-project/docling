import json
import re


def parse(llm_output: str, *, output_fields: list[str], **kwargs) -> dict:
    """작성자(authors) 추출 파서.

    LLM 출력에서 output_fields 키만 추출해 반환한다.
    마크다운 코드블록 래핑 및 embedded JSON을 처리한다.
    파싱 실패 시 각 필드를 null로 반환한다.
    """
    raw = _extract_json_str(llm_output)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {k: data.get(k) for k in output_fields}
    except (json.JSONDecodeError, TypeError):
        pass
    return {k: None for k in output_fields}


def _extract_json_str(text: str) -> str:
    # 마크다운 코드블록 우선
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", text):
        return block.strip()
    # raw JSON 스캔
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                _, end = decoder.raw_decode(text, i)
                return text[i:end]
            except json.JSONDecodeError:
                continue
    return text
