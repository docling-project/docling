"""가드레일(민감정보 분류/마스킹, #315) 공용 모듈.

facade 의 processor 들이 공유한다. 사용 예:

    from genon.preprocessor.facade import guardrail as gr

    if gr.call_enabled(kwargs):                       # 요청 guardrail_call(0/1)
        infos = gr.classify_document(gr.doc_text(document), url, wid, key, timeout)
    text, cats = gr.apply_to_text(text, infos, masking)   # 청크에 라벨+마스킹

역할 분담:
- config   : GuardrailConfig(yaml `guardrail:` 섹션) + call_enabled(요청 플래그 0/1)
- workflow_client : classify_document (분류 워크플로우 호출, fail-open)
- matcher  : apply_to_text / find_spans (청크 quote 매칭 → 라벨+마스킹)
- source_text : doc_text / elements_text / docs_text (분류에 보낼 문서 텍스트 생성)
"""
from .config import GuardrailConfig, call_enabled
from .workflow_client import classify_document, classify_with_config
from .matcher import apply_to_text, find_spans
from .source_text import doc_text, elements_text, docs_text

__all__ = [
    "GuardrailConfig",
    "call_enabled",
    "classify_document",
    "classify_with_config",
    "apply_to_text",
    "find_spans",
    "doc_text",
    "elements_text",
    "docs_text",
]
