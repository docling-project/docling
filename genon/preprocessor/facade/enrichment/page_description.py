"""page_description.py — 페이지 단위 image description 공용 모듈.

facade 의 "페이지 자체"를 렌더링해 VLM 으로 설명하는 로직을 적재용/첨부용이 공유한다.
(기존 PictureItem 단위 image_description enricher 와는 별개; 이미지형 슬라이드/PDF 대응)

- `PageDescriptionOptions`  : config(formats.ppt.page_description) → 옵션 dataclass
- `collect_page_texts`      : DoclingDocument → {page_no: native text}
- `describe_pages`          : 각 페이지 렌더 이미지 + native text 를 VLM 에 보내 설명 반환

프롬프트는 `{{page_text}}` 변수를 지원한다(PromptTemplate). 페이지 native text 를 프롬프트에
직접 반영해 요청한다.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from docling.utils.api_image_request import api_image_request
from docling.utils.llm_cache import in_current_context

from genon.preprocessor.facade.enrichment.prompt_files import read_prompt_file
from genon.preprocessor.facade.enrichment.prompt_template import PromptTemplate

_log = logging.getLogger(__name__)

# 페이지 텍스트를 반영하는 기본 프롬프트 템플릿({{page_text}} 변수 포함).
DEFAULT_PAGE_DESCRIPTION_PROMPT = (
    "다음은 문서(프레젠테이션) 한 페이지를 렌더링한 이미지입니다.\n"
    "아래 '페이지에서 추출된 텍스트'와 이미지를 함께 참고하여 이 페이지의 핵심 내용을 "
    "한국어로 자세히 설명하세요.\n\n"
    "[페이지에서 추출된 텍스트]\n"
    "{{page_text}}\n\n"
    "작성 지침:\n"
    "1) 제목/소제목, 표, 그래프, 다이어그램, 그림에 담긴 의미를 모두 반영\n"
    "2) 추출 텍스트와 이미지가 상충하면 이미지를 우선하되, 추출 텍스트로 맥락을 보완\n"
    "3) 추측은 최소화하고 이미지에서 확인 가능한 사실 중심으로 작성\n"
    "4) 표/그래프는 수치와 관계를 가능한 한 구체적으로 서술\n"
    "5) 검색/질의응답에 활용될 수 있도록 페이지의 주제와 요지를 명확히 포함"
)

# page_text 가 비었을 때 프롬프트에 넣을 placeholder
_EMPTY_PAGE_TEXT = "(페이지에서 추출된 텍스트 없음)"


def _to_float(value: Any, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


def _to_int(value: Any, default: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


@dataclass(frozen=True)
class PageDescriptionOptions:
    """formats.ppt.page_description 옵션."""
    enabled: bool = False
    url: str = ""
    api_key: str = ""
    model: str = "model"
    timeout: float = 360.0
    concurrency: int = 16
    images_scale: float = 2.0
    prompt_template: str = DEFAULT_PAGE_DESCRIPTION_PROMPT
    template_mode: str = "lenient"
    # 속도 옵션(첨부용). 0/빈값이면 미적용(적재용 기본값 = 무영향).
    max_tokens: int = 0           # VLM 출력 토큰 상한(생성 시간↓). 0=상한 없음
    max_image_side: int = 0       # 전송 전 이미지 최대 변(px)으로 다운스케일(payload↓). 0=원본
    params: dict = field(default_factory=dict)  # 임의 VLM 파라미터 passthrough

    @classmethod
    def from_config(cls, cfg: Optional[dict], config_dir: Optional[Path]) -> "PageDescriptionOptions":
        cfg = cfg if isinstance(cfg, dict) else {}

        # 프롬프트 우선순위: prompt_template_file > prompt(inline) > 내장 기본값
        prompt = DEFAULT_PAGE_DESCRIPTION_PROMPT
        prompt_file = cfg.get("prompt_template_file") or cfg.get("prompt_file")
        if isinstance(prompt_file, str) and prompt_file.strip():
            base = config_dir if config_dir is not None else Path.cwd()
            try:
                loaded = read_prompt_file(prompt_file.strip(), base)
                if loaded.strip():
                    prompt = loaded
            except Exception as exc:
                _log.warning(
                    f"[page_description] prompt_template_file 읽기 실패({prompt_file!r}): {exc}. 기본 프롬프트 사용."
                )
        else:
            inline = cfg.get("prompt") or cfg.get("prompt_template")
            if isinstance(inline, str) and inline.strip():
                prompt = inline.strip()

        enabled = cfg.get("enable")
        if enabled is None:
            enabled = cfg.get("enabled")

        _tmpl_cfg = cfg.get("template")
        mode = (_tmpl_cfg.get("mode") if isinstance(_tmpl_cfg, dict) else None) \
            or cfg.get("template_mode") or "lenient"

        params = cfg.get("params")
        params = dict(params) if isinstance(params, dict) else {}

        def _nonneg_int(v: Any) -> int:
            try:
                iv = int(v)
            except (TypeError, ValueError):
                return 0
            return iv if iv > 0 else 0

        return cls(
            enabled=bool(enabled) if enabled is not None else False,
            url=str(cfg.get("url") or cfg.get("api_url") or "").strip(),
            api_key=str(cfg.get("api_key") or "").strip(),
            model=str(cfg.get("model") or "model").strip() or "model",
            timeout=_to_float(cfg.get("timeout"), 360.0),
            concurrency=_to_int(cfg.get("concurrency"), 16),
            images_scale=_to_float(cfg.get("images_scale"), 2.0),
            prompt_template=prompt,
            template_mode=str(mode).strip().lower(),
            max_tokens=_nonneg_int(cfg.get("max_tokens")),
            max_image_side=_nonneg_int(cfg.get("max_image_side")),
            params=params,
        )


def _maybe_downscale(image: "Image.Image", max_side: int) -> "Image.Image":
    """이미지의 최대 변이 max_side 를 넘으면 aspect 비율 유지하며 축소한다(payload↓)."""
    if not max_side or max_side <= 0:
        return image
    try:
        w, h = image.size
    except Exception:
        return image
    if max(w, h) <= max_side:
        return image
    resized = image.copy()
    resized.thumbnail((max_side, max_side))  # aspect 보존, in-place
    return resized


def collect_page_texts(document: Any) -> "dict[int, str]":
    """DoclingDocument 의 아이템을 prov.page_no 로 그룹핑해 페이지별 native text 를 만든다.

    반환: {page_no(1-based): joined text}
    """
    parts: "dict[int, list[str]]" = {}
    for item, _ in document.iterate_items():
        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        prov = getattr(item, "prov", None) or []
        page_no = prov[0].page_no if prov and getattr(prov[0], "page_no", None) else 1
        parts.setdefault(page_no, []).append(text)
    return {pno: "\n".join(v).strip() for pno, v in parts.items()}


def describe_pages(
    document: Any,
    options: PageDescriptionOptions,
    page_texts: Optional[dict] = None,
) -> "dict[int, str]":
    """각 페이지를 렌더 이미지로 VLM 에 보내 설명을 반환한다.

    generate_page_images=True 로 파싱되어 page.image.pil_image 가 있어야 한다.
    page_texts 가 주어지면 프롬프트의 `{{page_text}}` 변수에 반영해 요청한다.
    반환: {page_no(1-based): description text}. 비활성/URL 없음/페이지 이미지 없음 → 빈 dict.
    """
    if not options.enabled:
        return {}
    if not options.url:
        _log.warning("[page_description] enable=true 이지만 url 이 비어 있어 건너뜁니다.")
        return {}

    pages = getattr(document, "pages", None) or {}
    page_nos = sorted(pages.keys())
    if not page_nos:
        return {}

    page_texts = page_texts or {}
    tpl = PromptTemplate(
        options.prompt_template,
        mode=options.template_mode or "lenient",
        allowed_names={"page_text"},
    )

    req_headers = {}
    if options.api_key:
        req_headers["Authorization"] = f"Bearer {options.api_key}"
    # 모델 + 임의 passthrough 파라미터 + (있으면) 출력 토큰 상한.
    req_params = dict(options.params or {})
    if options.model and "model" not in req_params:
        req_params["model"] = options.model
    if options.max_tokens and options.max_tokens > 0 and "max_tokens" not in req_params:
        req_params["max_tokens"] = options.max_tokens

    results: "dict[int, str]" = {}
    lock = threading.Lock()

    def _describe(page_no: int) -> None:
        page = pages.get(page_no)
        if page is None or getattr(page, "image", None) is None:
            return
        image = page.image.pil_image
        if image is None:
            return
        image = _maybe_downscale(image, options.max_image_side)
        page_text = (page_texts.get(page_no) or "").strip() or _EMPTY_PAGE_TEXT
        prompt = tpl.render(page_text=page_text)
        try:
            output = api_image_request(
                image=image,
                prompt=prompt,
                url=options.url,
                timeout=options.timeout,
                headers=req_headers,
                **req_params,
            )
        except Exception as exc:
            _log.warning(f"[page_description] page={page_no} 설명 실패: {exc}")
            return
        text = str(output or "").strip()
        if text:
            with lock:
                results[page_no] = text

    with ThreadPoolExecutor(max_workers=max(1, int(options.concurrency or 1))) as executor:
        # #329: 워커 스레드에도 llm_cache 컨텍스트 전파
        list(executor.map(in_current_context(_describe), page_nos))
    return results
