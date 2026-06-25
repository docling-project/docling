import json
import logging
import os
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional

_log = logging.getLogger(__name__)

# thinking(추론) 토글 공유 헬퍼. enrichment 패키지의 단일 소스를 재사용한다.
# 일부 로컬 테스트 환경에서는 facade import 가 깨질 수 있어(no-op 폴백) 레거시 경로를 보호한다.
try:
    from genon.preprocessor.facade.enrichment.thinking import (
        resolve_thinking_kwargs,
        strip_reasoning,
    )
except Exception:  # pragma: no cover - import 환경에 따른 폴백
    def resolve_thinking_kwargs(mode, dialect="standard"):
        return None

    def strip_reasoning(message):
        if isinstance(message, dict):
            return (message.get("content") or "").strip()
        return (message or "").strip()


class LLMApiError(Exception):
    """OpenAI-compatible provider call error with raw response payload."""

    def __init__(
        self,
        raw_error_message: str,
        *,
        status_code: Optional[int] = None,
        category: Optional[str] = None,
        prompt_type: Optional[str] = None,
    ) -> None:
        self.raw_error_message = raw_error_message
        self.status_code = status_code
        self.category = category
        self.prompt_type = prompt_type
        super().__init__(raw_error_message)

class PromptManager:
    """프롬프트 및 API 설정 관리 클래스"""

    def __init__(self, prompts_file: Optional[str] = None, custom_prompts: Optional[Dict[str, Any]] = None, custom_api_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Args:
            prompts_file: 프롬프트 JSON 파일 경로 (기본값: prompts.json)
            custom_prompts: 사용자 정의 프롬프트 딕셔너리
            custom_api_configs: 카테고리별 사용자 정의 API 설정 딕셔너리
                예: {
                    "toc_extraction": {"provider": "openai", "api_key": "...", "model": "gpt-4"},
                    "metadata_extraction": {"provider": "openrouter", "api_key": "...", "model": "..."}
                }
        """
        if prompts_file is None:
            self.prompts_file = Path(__file__).parent / "prompts.json"
        else:
            self.prompts_file = Path(prompts_file)

        self._config = self._load_config()
        self.custom_prompts = custom_prompts or {}
        self.custom_api_configs = custom_api_configs or {}

    def _load_config(self) -> Dict[str, Any]:
        """설정 JSON 파일 로드"""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            _log.error(f"설정 파일을 찾을 수 없습니다: {self.prompts_file}")
            return {}
        except json.JSONDecodeError as e:
            _log.error(f"설정 파일 JSON 파싱 오류: {e}")
            return {}

    def get_api_config(self, provider: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """API 설정 반환 (카테고리별 사용자 정의 우선)"""
        # 카테고리별 사용자 정의 API 설정 확인
        if category and category in self.custom_api_configs:
            custom_config = self.custom_api_configs[category]
            if "provider" in custom_config and custom_config["provider"] == provider:
                config = custom_config.copy()

                # 환경변수에서 API 키 우선 조회
                if config.get("api_key") is None:
                    env_key_name = f"{provider.upper()}_API_KEY"
                    env_api_key = os.getenv(env_key_name)
                    if env_api_key:
                        config["api_key"] = env_api_key
                        _log.info(f"환경변수에서 {provider} API 키 로드됨 (카테고리: {category})")

                return config

        # 기본 설정 사용
        try:
            api_config = self._config["api_config"][provider].copy()

            # 환경변수에서 API 키 우선 조회
            env_key_name = f"{provider.upper()}_API_KEY"
            env_api_key = os.getenv(env_key_name)
            if env_api_key:
                api_config["api_key"] = env_api_key
                _log.info(f"환경변수에서 {provider} API 키 로드됨")

            return api_config
        except KeyError:
            _log.error(f"API 설정을 찾을 수 없습니다: {provider}")
            return None

    def get_prompt_config(self, category: str, prompt_type: str) -> Optional[Dict[str, Any]]:
        """특정 프롬프트 설정 반환 (사용자 정의 프롬프트 및 카테고리별 모델 설정 우선)"""
        # 먼저 사용자 정의 프롬프트에서 찾기
        if category in self.custom_prompts and prompt_type in self.custom_prompts[category]:
            custom_config = self.custom_prompts[category][prompt_type]
            # 기본 설정과 병합
            try:
                default_config = self._config[category][prompt_type].copy()
                default_config.update(custom_config)

                # 카테고리별 사용자 정의 모델 설정 적용
                if category in self.custom_api_configs:
                    api_config = self.custom_api_configs[category]
                    if "model" in api_config:
                        default_config["model"] = api_config["model"]
                    if "provider" in api_config:
                        default_config["api_provider"] = api_config["provider"]
                    if "temperature" in api_config:
                        default_config["temperature"] = api_config["temperature"]
                    if "top_p" in api_config:
                        default_config["top_p"] = api_config["top_p"]
                    if "seed" in api_config:
                        default_config["seed"] = api_config["seed"]
                    if "max_tokens" in api_config:
                        default_config["max_tokens"] = api_config["max_tokens"]
                    if "repetition_penalty" in api_config:
                        default_config["repetition_penalty"] = api_config["repetition_penalty"]
                    if "precheck_enabled" in api_config:
                        default_config["precheck_enabled"] = api_config["precheck_enabled"]
                    if "max_context_tokens" in api_config:
                        default_config["max_context_tokens"] = api_config["max_context_tokens"]
                    if "completion_reserved_tokens" in api_config:
                        default_config["completion_reserved_tokens"] = api_config["completion_reserved_tokens"]
                    if "thinking" in api_config:
                        default_config["thinking"] = api_config["thinking"]
                    if "thinking_dialect" in api_config:
                        default_config["thinking_dialect"] = api_config["thinking_dialect"]

                return default_config
            except KeyError:
                # 기본 설정이 없으면 사용자 정의만 사용
                return custom_config

        # 기본 프롬프트에서 찾기
        try:
            config = self._config[category][prompt_type].copy()

            # 카테고리별 사용자 정의 모델 설정 적용
            if category in self.custom_api_configs:
                api_config = self.custom_api_configs[category]
                if "model" in api_config:
                    config["model"] = api_config["model"]
                if "provider" in api_config:
                    config["api_provider"] = api_config["provider"]
                if "temperature" in api_config:
                    config["temperature"] = api_config["temperature"]
                if "top_p" in api_config:
                    config["top_p"] = api_config["top_p"]
                if "seed" in api_config:
                    config["seed"] = api_config["seed"]
                if "max_tokens" in api_config:
                    config["max_tokens"] = api_config["max_tokens"]
                if "repetition_penalty" in api_config:
                    config["repetition_penalty"] = api_config["repetition_penalty"]
                if "precheck_enabled" in api_config:
                    config["precheck_enabled"] = api_config["precheck_enabled"]
                if "max_context_tokens" in api_config:
                    config["max_context_tokens"] = api_config["max_context_tokens"]
                if "completion_reserved_tokens" in api_config:
                    config["completion_reserved_tokens"] = api_config["completion_reserved_tokens"]
                if "thinking" in api_config:
                    config["thinking"] = api_config["thinking"]
                if "thinking_dialect" in api_config:
                    config["thinking_dialect"] = api_config["thinking_dialect"]

            return config
        except KeyError:
            _log.error(f"프롬프트를 찾을 수 없습니다: {category}.{prompt_type}")
            return None

    def get_system_prompt(self, category: str, prompt_type: str, custom_system: Optional[str] = None) -> Optional[str]:
        """시스템 프롬프트 반환 (사용자 정의 우선)"""
        if custom_system:
            return custom_system
        config = self.get_prompt_config(category, prompt_type)
        return config.get("system") if config else None

    def get_user_prompt_template(self, category: str, prompt_type: str, custom_user: Optional[str] = None) -> Optional[str]:
        """사용자 프롬프트 템플릿 반환 (사용자 정의 우선)"""
        if custom_user:
            return custom_user
        config = self.get_prompt_config(category, prompt_type)
        return config.get("user") if config else None

    @staticmethod
    def _normalize_raw_text_placeholder(template: str, **kwargs: Any) -> str:
        """`{{var}}`를 `{var}`의 별칭으로 허용한다(주입되는 kwargs 키 한정).

        - 실제로 주입되는 키(`raw_text`, `prior_toc` 등)에 대해서만 `{{key}}`→`{key}` 로 정규화한다.
        - 이미 단일 `{key}` 가 포함된 템플릿은 건드리지 않는다.
        - JSON 예시 표현용 `{{ ... }}`(주입 키가 아닌 것)는 그대로 보존한다.
        """
        for key in kwargs:
            token2 = "{{" + key + "}}"
            if token2 not in template:
                continue
            token1 = "{" + key + "}"
            # 이미 단일 형태가 있으면 그대로 둔다(중복 치환 방지).
            if token1 in template.replace(token2, ""):
                continue
            template = template.replace(token2, token1)
        return template

    def format_user_prompt(self, category: str, prompt_type: str, custom_user: Optional[str] = None, **kwargs) -> Optional[str]:
        """사용자 프롬프트 템플릿에 변수를 삽입하여 완성된 프롬프트 반환"""
        template = self.get_user_prompt_template(category, prompt_type, custom_user)
        if template is None:
            return None
        template = self._normalize_raw_text_placeholder(template, **kwargs)

        try:
            return template.format(**kwargs)
        except KeyError as e:
            _log.error(f"프롬프트 템플릿 변수 오류: {e}")
            return None

    def get_messages(self, category: str, prompt_type: str, custom_system: Optional[str] = None, custom_user: Optional[str] = None, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """API 요청용 메시지 형식으로 반환"""
        system_prompt = self.get_system_prompt(category, prompt_type, custom_system)
        user_prompt = self.format_user_prompt(category, prompt_type, custom_user, **kwargs)

        if system_prompt is None or user_prompt is None:
            return None

        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

    def get_model_config(self, category: str, prompt_type: str) -> Dict[str, Any]:
        """모델 설정 반환 (카테고리별 사용자 정의 우선)"""
        config = self.get_prompt_config(category, prompt_type)
        if config is None:
            return {
                "model": "google/gemma-3-27b-it:free",
                "seed": 3,
                "api_provider": "openrouter"
            }

        model_config = {
            "model": config.get("model", "google/gemma-3-27b-it:free"),
            "seed": config.get("seed", 3),
            "api_provider": config.get("api_provider", "openrouter")
        }

        # 선택적 파라미터들 추가
        if "temperature" in config:
            model_config["temperature"] = config["temperature"]
        if "top_p" in config:
            model_config["top_p"] = config["top_p"]
        if "max_tokens" in config:
            model_config["max_tokens"] = config["max_tokens"]
        if "repetition_penalty" in config:
            model_config["repetition_penalty"] = config["repetition_penalty"]
        if "precheck_enabled" in config:
            model_config["precheck_enabled"] = config["precheck_enabled"]
        if "max_context_tokens" in config:
            model_config["max_context_tokens"] = config["max_context_tokens"]
        if "completion_reserved_tokens" in config:
            model_config["completion_reserved_tokens"] = config["completion_reserved_tokens"]
        if "thinking" in config:
            model_config["thinking"] = config["thinking"]
        if "thinking_dialect" in config:
            model_config["thinking_dialect"] = config["thinking_dialect"]

        return model_config

    @staticmethod
    def _parse_bool(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "no", "n", "off"}:
                return False
        return None

    @staticmethod
    def _parse_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        if value is None or value == "":
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _flatten_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)

    def _render_messages_for_token_count(self, messages: List[Dict[str, Any]]) -> str:
        rendered = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = self._flatten_message_content(msg.get("content"))
            rendered.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
        rendered.append("<|im_start|>assistant\n")
        return "\n".join(rendered)

    @staticmethod
    def _rough_token_count(text: str) -> int:
        # Measured against Qwen2.5-7B-Instruct: digits/punct tokenize at ~1.0/char,
        # Korean at ~0.72-0.93/char (0.95 is conservative), English at ~0.16-0.22/char.
        digits = len(re.findall(r"[0-9]", text))
        punct  = len(re.findall(r"[^\w\s]", text))
        kor    = len(re.findall(r"[\uAC00-\uD7AF]", text))
        cjk    = len(re.findall(r"[\u4E00-\u9FFF\u3040-\u30FF]", text))
        en     = len(re.findall(r"[a-zA-Z]", text))
        other  = max(len(text) - digits - punct - kor - cjk - en, 0)
        return int(digits * 1.0 + punct * 0.8 + kor * 0.95 + cjk * 0.95 + en * 0.25 + other * 0.3)

    def estimate_tokens(self, text: str) -> int:
        """프롬프트 토큰 수 추정값을 반환한다.

        precheck(`_run_prompt_precheck`)와 동일한 추정기(`_rough_token_count`)를 사용하므로,
        분할 로직이 precheck와 같은 척도로 토큰 예산을 산정할 수 있다.
        """
        return self._rough_token_count(text or "")

    def _run_prompt_precheck(
        self,
        messages: List[Dict[str, Any]],
        model_config: Dict[str, Any],
        category: str,
        prompt_type: str,
        raise_on_error: bool,
    ) -> bool:
        precheck_enabled = self._parse_bool(model_config.get("precheck_enabled"))
        max_context_tokens = self._parse_int(model_config.get("max_context_tokens"))
        if precheck_enabled is False:
            return True
        if max_context_tokens is None or max_context_tokens <= 0:
            return True

        completion_reserved_tokens = self._parse_int(
            model_config.get("completion_reserved_tokens"),
            default=self._parse_int(model_config.get("max_tokens"), default=0) or 0,
        ) or 0
        if completion_reserved_tokens < 0:
            completion_reserved_tokens = 0

        allowed_prompt_tokens = max_context_tokens - completion_reserved_tokens
        if allowed_prompt_tokens < 0:
            allowed_prompt_tokens = 0

        rendered = self._render_messages_for_token_count(messages)
        estimated_prompt_tokens = self._rough_token_count(rendered)

        if estimated_prompt_tokens > allowed_prompt_tokens:
            raw_error_message = json.dumps(
                {
                    "object": "error",
                    "message": (
                        f"프롬프트 입력 토큰 ({estimated_prompt_tokens}) 초과 하였습니다. "
                        f"({max_context_tokens} - reserved {completion_reserved_tokens})."
                    ),
                    "type": "BadRequestError",
                    "param": "prompt",
                    "code": 400,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if raise_on_error:
                raise LLMApiError(
                    raw_error_message,
                    status_code=400,
                    category=category,
                    prompt_type=prompt_type,
                )
            _log.error(
                "Prompt token precheck blocked request (%s.%s): %s",
                category,
                prompt_type,
                raw_error_message,
            )
            return False
        return True

    def call_ai_model(
        self,
        category: str,
        prompt_type: str,
        custom_system: Optional[str] = None,
        custom_user: Optional[str] = None,
        raise_on_error: bool = False,
        **kwargs,
    ) -> Optional[str]:
        """AI 모델을 직접 requests.post로 호출하여 결과 반환"""
        # 메시지 구성
        messages = self.get_messages(category, prompt_type, custom_system, custom_user, **kwargs)
        if messages is None:
            return None

        # 모델 설정 가져오기
        model_config = self.get_model_config(category, prompt_type)
        api_provider = model_config["api_provider"]

        # 카테고리별 API 설정 가져오기
        api_config = self.get_api_config(api_provider, category)
        if api_config is None:
            return None

        try:
            # 요청 헤더 구성
            headers = {"Content-Type": "application/json"}
            if api_config.get("api_key"):
                headers["Authorization"] = f"Bearer {api_config['api_key']}"

            # 요청 본문 구성
            payload = {
                "model": model_config["model"],
                "messages": messages
            }

            # 선택적 파라미터 추가
            if "seed" in model_config:
                payload["seed"] = model_config["seed"]
            if "temperature" in model_config:
                payload["temperature"] = model_config["temperature"]
            if "top_p" in model_config:
                payload["top_p"] = model_config["top_p"]
            if "max_tokens" in model_config:
                payload["max_tokens"] = model_config["max_tokens"]
            if "repetition_penalty" in model_config:
                payload["repetition_penalty"] = model_config["repetition_penalty"]
            # thinking(추론) 모드: 모델 chat template 에 맞는 chat_template_kwargs 주입.
            # 미설정(None)이면 아무것도 보내지 않아 기존 동작을 보존한다.
            ctk = resolve_thinking_kwargs(
                model_config.get("thinking"),
                model_config.get("thinking_dialect", "standard"),
            )
            if ctk:
                payload["chat_template_kwargs"] = ctk

            if not self._run_prompt_precheck(
                messages=messages,
                model_config=model_config,
                category=category,
                prompt_type=prompt_type,
                raise_on_error=raise_on_error,
            ):
                return None

            # API URL 구성
            base_url = api_config.get("api_base_url", api_config.get("base_url"))
            if base_url.endswith("/chat/completions"):
                api_url = base_url
            else:
                api_url = f"{base_url}/chat/completions"

            _log.debug(f"API 요청 URL ({category}): {api_url}")
            _log.debug(f"API 요청 페이로드 ({category}): {json.dumps(payload, ensure_ascii=False, indent=2)}")

            # requests.post로 API 호출
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=3600
            )

            # 응답 상태 확인
            response.raise_for_status()

            # 응답 파싱
            response_data = response.json()

            # 응답에서 텍스트 추출 (thinking 출력 분리/제거 후 본문만 반환)
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return strip_reasoning(response_data["choices"][0]["message"])
            else:
                _log.error(f"예상하지 못한 응답 형식 ({category}): {response_data}")
                return None

        except requests.exceptions.RequestException as e:
            _log.error(f"API 요청 실패 ({category}.{prompt_type}): {e}")
            status_code = None
            raw_error_message = None
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                _log.error(f"응답 상태: {e.response.status_code}")
                _log.error(f"응답 내용: {e.response.text}")
                raw_error_message = e.response.text
            if raise_on_error:
                if not raw_error_message:
                    raw_error_message = str(e)
                raise LLMApiError(
                    raw_error_message,
                    status_code=status_code,
                    category=category,
                    prompt_type=prompt_type,
                ) from e
            return None
        except json.JSONDecodeError as e:
            _log.error(f"JSON 응답 파싱 실패 ({category}.{prompt_type}): {e}")
            return None
        except LLMApiError:
            if raise_on_error:
                raise
            return None
        except Exception as e:
            _log.error(f"AI 모델 호출 중 예상치 못한 오류 ({category}.{prompt_type}): {e}")
            return None

    def reload_config(self):
        """설정 파일 다시 로드"""
        self._config = self._load_config()
        _log.info("프롬프트 설정이 다시 로드되었습니다.")

    def build_windowed_messages(
        self,
        category: str,
        prompt_type: str,
        raw_text: str,
        *,
        custom_system: Optional[str] = None,
        custom_user: Optional[str] = None,
        overlap_tokens: int = 256,
        context_length: Optional[int] = None,  # 없으면 모델 설정 또는 Splitter의 자동 추정 사용
    ) -> Optional[List[List[Dict[str, Any]]]]:
        """
        SlidingWindowSplitter를 그대로 사용해:
        1) 사용자 프롬프트 템플릿 → Splitter의 BASE_PROMPT로 주입
        2) raw_text를 분할하여 완성된 user 프롬프트들을 생성
        3) 각 윈도우에 대해 (system, user) messages 배열 생성
        """
        # 1) 시스템/유저 템플릿 취득
        system_prompt = self.get_system_prompt(category, prompt_type, custom_system)
        user_template = self.get_user_prompt_template(category, prompt_type, custom_user)
        if system_prompt is None or user_template is None:
            return None
        user_template = self._normalize_raw_text_placeholder(user_template, raw_text=raw_text)

        # {raw_text} 플레이스홀더가 없다면 안전하게 붙여준다.
        if "{raw_text}" not in user_template:
            user_template = user_template.rstrip() + "\n\n## 실제 작업할 입력\n{raw_text}"

        # 2) 모델/파라미터
        model_cfg = self.get_model_config(category, prompt_type)
        model_name = model_cfg.get("model", "")
        max_tokens = int(model_cfg.get("max_tokens", 512))
        # context_length 우선순위: 인자 > 모델설정(context_length 키가 있다면) > None(=Splitters auto/8192)
        max_model_len = int(context_length or model_cfg.get("context_length", 0) or 0) or None

        # 3) Splitter 생성 (그대로 사용)
        splitter = SlidingWindowSplitter(
            model_name=model_name,
            base_prompt=user_template,
            max_model_len=None,  # None이면 내부에서 HF config로 추정하거나 8192 사용
            max_output_tokens=max_tokens if max_tokens < 5000 else 5000,        # BASE_PROMPT 토큰 + max_tokens로 reserve 자동계산
            overlap_tokens=overlap_tokens,
        )

        # 4) 프롬프트들 생성 (완성된 user 프롬프트 문자열 리스트)
        user_prompts: List[str] = splitter.build_prompts(raw_text)

        # 5) 각 윈도우에 대해 messages 구성
        windowed_messages: List[List[Dict[str, Any]]] = []
        for up in user_prompts:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": up},
            ]
            windowed_messages.append(msgs)
        return windowed_messages

    def call_ai_model_windowed(
        self,
        category: str,
        prompt_type: str,
        raw_text: str,
        *,
        custom_system: Optional[str] = None,
        custom_user: Optional[str] = None,
        overlap_tokens: int = 256,
        context_length: Optional[int] = None,
        **kwargs,
    ) -> Optional[str]:
        """
        SlidingWindowSplitter로 분할된 각 윈도우에 대해 /chat/completions를 순차 호출 후 결과를 합쳐 반환.
        기존 call_ai_model과 동일한 API 경로/헤더/파라미터를 재사용.
        """
        # 1) 윈도우별 messages 생성
        windowed = self.build_windowed_messages(
            category=category,
            prompt_type=prompt_type,
            raw_text=raw_text,
            custom_system=custom_system,
            custom_user=custom_user,
            overlap_tokens=overlap_tokens,
            context_length=context_length,
        )
        if not windowed:
            return None

        # 2) 모델/프로바이더 설정
        model_config = self.get_model_config(category, prompt_type)
        api_provider = model_config["api_provider"]
        api_config = self.get_api_config(api_provider, category)
        if api_config is None:
            return None

        headers = {"Content-Type": "application/json"}
        if api_config.get("api_key"):
            headers["Authorization"] = f"Bearer {api_config['api_key']}"

        base_url = api_config.get("api_base_url", api_config.get("base_url"))
        api_url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"

        pieces: List[str] = []
        for idx, messages in enumerate(windowed, 1):
            payload = {"model": model_config["model"], "messages": messages}
            # 기존 선택 파라미터 반영
            for k in ("seed", "temperature", "top_p", "max_tokens"):
                if k in model_config:
                    payload[k] = model_config[k]

            _log.debug(f"[{category}] window {idx}/{len(windowed)} payload: {json.dumps(payload, ensure_ascii=False)[:800]}...")
            resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            if "choices" in data and data["choices"]:
                pieces.append(data["choices"][0]["message"]["content"])
            else:
                _log.error(f"[{category}] Unexpected response for window {idx}: {data}")
                pieces.append("")

        return pieces


from typing import Any, Callable, Dict, List, Optional, Tuple
import re

# Optional deps
try:
    import tiktoken  # OpenAI tokenizer
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

try:
    from transformers import AutoTokenizer, AutoConfig
    _HAS_HF = True
except Exception:
    _HAS_HF = False


class SlidingWindowSplitter:
    def __init__(
        self,
        model_name: str,
        base_prompt: str,
        max_model_len: Optional[int] = None,
        reserve: Optional[int] = None,   # ← 직접 주면 우선
        overlap_tokens: int = 256,
        max_output_tokens: int = 512,    # ← 생성 토큰 길이
    ) -> None:
        self.model_name = model_name
        self.base_prompt = base_prompt
        self.max_model_len = max_model_len or self._infer_max_len(model_name) or 8192
        self.overlap_tokens = overlap_tokens
        self.max_output_tokens = max_output_tokens
        self.tokenizer = self._load_tokenizer(model_name)

        # BASE_PROMPT에서 {raw_text}를 제거한 본문만 토크나이즈해서 길이 계산
        base_without_placeholder = self.base_prompt.replace("{raw_text}", "")
        base_prompt_tokens = self._count_tokens(base_without_placeholder)

        # reserve 자동 계산: BASE_PROMPT 토큰 + max_output_tokens
        self.reserve = reserve if reserve is not None else (base_prompt_tokens + self.max_output_tokens)

        if self.max_model_len <= self.reserve:
            raise ValueError(f"reserve must be smaller than max_model_len. Given reserve={self.reserve}, max_model_len={self.max_model_len}")

    def _load_tokenizer(self, model_name: str):
        name = model_name.lower()
        if name.startswith(("gpt-3", "gpt-4")) and _HAS_TIKTOKEN:
            return tiktoken.encoding_for_model(model_name)
        if _HAS_HF:
            try:
                return AutoTokenizer.from_pretrained(model_name, use_fast=True)
            except Exception:
                return None
        return None

    def _infer_max_len(self, model_name: str) -> Optional[int]:
        if _HAS_HF:
            try:
                cfg = AutoConfig.from_pretrained(model_name)
                if getattr(cfg, "max_position_embeddings", None):
                    return int(cfg.max_position_embeddings)
            except Exception:
                pass
            try:
                tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                val = getattr(tok, "model_max_length", None)
                if isinstance(val, int) and val < 10_000_000:
                    return val
            except Exception:
                pass
        return None

    def _count_tokens(self, text: str) -> int:
        # OpenAI(tiktoken)
        if _HAS_TIKTOKEN and hasattr(self.tokenizer, "encode") and self.model_name.lower().startswith(("gpt-3", "gpt-4")):
            return len(self.tokenizer.encode(text))
        # HF(Transformers)
        if _HAS_HF and hasattr(self.tokenizer, "encode"):
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Rough fallback
        cjk = len(re.findall(r"[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]", text))
        asc = len(re.findall(r"[\x00-\x7F]", text))
        other = max(len(text) - (cjk + asc), 0)
        return int(cjk * 0.95 + asc * 0.25 + other * 0.5)

    def _largest_slice(self, text: str, start: int, budget: int) -> int:
        """Binary search + 확대 탐색으로 budget 이하 최대 구간 찾기"""
        n = len(text)
        if start >= n:
            return start
        # 1글자도 넘치면 최소 1자 이동(이례적 안전장치)
        if self._count_tokens(text[start:start+1]) > budget:
            return min(start + 1, n)

        lo, hi = start, min(start + 8000, n)
        best = start
        # 지수적 확대
        while hi < n:
            seg = text[start:hi]
            if self._count_tokens(seg) > budget:
                break
            best = hi
            step = min((hi - start) * 2, 32768)
            if hi - start >= 32768:
                break
            hi = min(start + step, n)
        # 아직도 예산 이내면 hi 반환
        if self._count_tokens(text[start:hi]) <= budget:
            return hi
        # 이진 탐색
        lo, hi = best, hi
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid <= start:
                lo = mid + 1
                continue
            if self._count_tokens(text[start:mid]) <= budget:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return max(best, start + 1)

    def _overlap_start(self, text: str, prev_start: int, prev_end: int) -> int:
        """마지막 구간 뒤쪽을 기준으로 overlap_tokens 만큼 토큰 중첩이 되도록 시작점 되감기"""
        if prev_start >= prev_end:
            return prev_start
        lo, hi = prev_start, prev_end
        best = prev_start
        while lo <= hi:
            mid = (lo + hi) // 2
            t = self._count_tokens(text[mid:prev_end])
            if t > self.overlap_tokens:
                lo = mid + 1
            else:
                best = mid
                hi = mid - 1
        return best

    def split(self, text: str) -> List[str]:
        # BASE_PROMPT + max_output_tokens 를 이미 reserve로 차감
        budget = self.max_model_len - self.reserve
        if budget <= 0:
            raise ValueError(f"Computed budget <= 0. Increase max_model_len or reduce max_output_tokens/base prompt. Given budget={budget}, reserve={self.reserve}, max_model_len={self.max_model_len}")
        s = 0
        n = len(text)
        windows = []
        while s < n:
            e = self._largest_slice(text, s, budget)
            windows.append(text[s:e])
            if e >= n:
                break
            s = self._overlap_start(text, s, e)
        return windows

    def build_prompts(self, raw_text: str) -> List[str]:
        """
        분할된 각 청크를 BASE_PROMPT에 주입한 완성 프롬프트 반환.
        reserve = (BASE_PROMPT 토큰; {raw_text} 제외) + max_output_tokens 으로 자동 계산됨.
        """
        windows = self.split(raw_text)
        return [self.base_prompt.format(raw_text=chunk) for chunk in windows]
