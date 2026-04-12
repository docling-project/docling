"""Transformers-based VLM inference engine."""

import importlib.metadata
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import torch
from packaging import version
from PIL.Image import Image
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedModel,
    ProcessorMixin,
    StoppingCriteriaList,
    StopStringCriteria,
)

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import (
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.inference_engines.vlm._utils import (
    extract_generation_stoppers,
    preprocess_image_batch,
    resolve_model_artifacts_path,
)
from docling.models.inference_engines.vlm.base import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
)
from docling.models.utils.generation_utils import (
    GenerationStopper,
    HFStoppingCriteriaWrapper,
)
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin
from docling.utils.accelerator_utils import decide_device

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)
_FALCON_OCR_DEFAULT_PROMPT = "Extract the text content from this image."
_FALCON_OCR_CATEGORY_BY_PROMPT_SUBSTRING = (
    ("formula", "formula"),
    ("table", "table"),
    ("caption", "caption"),
    ("footnote", "footnote"),
    ("list-item", "list-item"),
    ("page-footer", "page-footer"),
    ("page-header", "page-header"),
    ("section-header", "section-header"),
    ("title", "title"),
)


def _value_mentions_falcon_ocr(value: Any) -> bool:
    return isinstance(value, str) and (
        "falcon-ocr" in value.lower() or "falcon_ocr" in value.lower()
    )


def _config_mentions_falcon_ocr(config_obj: Any) -> bool:
    model_type = getattr(config_obj, "model_type", None)
    if _value_mentions_falcon_ocr(model_type):
        return True

    architectures = getattr(config_obj, "architectures", None)
    if isinstance(architectures, list | tuple) and any(
        _value_mentions_falcon_ocr(architecture) for architecture in architectures
    ):
        return True

    auto_map = getattr(config_obj, "auto_map", None)
    return isinstance(auto_map, dict) and any(
        _value_mentions_falcon_ocr(mapped_value) for mapped_value in auto_map.values()
    )


def _unwrap_compiled_vlm_model(
    vlm_model: Optional[PreTrainedModel],
) -> Optional[PreTrainedModel]:
    return getattr(vlm_model, "_orig_mod", vlm_model)


def _supports_falcon_ocr_native_generate(
    vlm_model: Optional[PreTrainedModel],
) -> bool:
    base_model = _unwrap_compiled_vlm_model(vlm_model)
    return base_model is not None and (
        callable(getattr(base_model, "_generate_batch", None))
        or callable(getattr(base_model, "generate", None))
    )


def _normalize_falcon_ocr_prompt(prompt: str) -> str:
    normalized_prompt = prompt.strip() or _FALCON_OCR_DEFAULT_PROMPT
    if "<|image|>" not in normalized_prompt:
        normalized_prompt = f"<|image|>{normalized_prompt}"
    if "<|OCR_PLAIN|>" not in normalized_prompt:
        normalized_prompt = f"{normalized_prompt.rstrip()}\n<|OCR_PLAIN|>"
    return normalized_prompt


def _falcon_ocr_category_from_prompt(prompt: str) -> str:
    normalized_prompt = prompt.lower()
    for prompt_substring, category in _FALCON_OCR_CATEGORY_BY_PROMPT_SUBSTRING:
        if prompt_substring in normalized_prompt:
            return category
    return "plain"


class TransformersVlmEngine(BaseVlmEngine, HuggingFaceModelDownloadMixin):
    """HuggingFace Transformers engine for VLM inference.

    This engine uses the transformers library to run vision-language models
    locally on CPU, CUDA, or XPU devices.
    """

    def __init__(
        self,
        options: TransformersVlmEngineOptions,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Union[Path, str]],
        model_config: Optional["EngineModelConfig"] = None,
    ):
        """Initialize the Transformers engine.

        Args:
            options: Transformers-specific runtime options
            accelerator_options: Hardware accelerator configuration
            artifacts_path: Path to cached model artifacts
            model_config: Model configuration (repo_id, revision, extra_config)
        """
        super().__init__(options, model_config=model_config)
        self.options: TransformersVlmEngineOptions = options
        self.accelerator_options = accelerator_options
        self.artifacts_path = artifacts_path

        # These will be set during initialization
        self.device: Optional[str] = None
        self.processor: Optional[ProcessorMixin] = None
        self.vlm_model: Optional[PreTrainedModel] = None
        self.generation_config: Optional[GenerationConfig] = None

        # Initialize immediately if model_config is provided
        if self.model_config is not None:
            self.initialize()

    def initialize(self) -> None:
        """Initialize the Transformers model and processor."""
        if self._initialized:
            return

        _log.info("Initializing Transformers VLM inference engine...")

        # Determine device
        supported_devices = [
            AcceleratorDevice.CPU,
            AcceleratorDevice.CUDA,
            AcceleratorDevice.XPU,
        ]
        self.device = decide_device(
            self.options.device or self.accelerator_options.device,
            supported_devices=supported_devices,
        )
        _log.info(f"Using device: {self.device}")

        # Load model if model_config is provided
        if self.model_config is not None and self.model_config.repo_id is not None:
            repo_id = self.model_config.repo_id
            revision = self.model_config.revision or "main"

            # Get model_type from extra_config
            model_type = self.model_config.extra_config.get(
                "transformers_model_type",
                TransformersModelType.AUTOMODEL,
            )

            _log.info(
                f"Loading model {repo_id} (revision: {revision}, "
                f"model_type: {model_type.value})"
            )
            self._load_model_for_repo(repo_id, revision=revision, model_type=model_type)

        self._initialized = True

    def _load_model_for_repo(
        self,
        repo_id: str,
        revision: str = "main",
        model_type: TransformersModelType = TransformersModelType.AUTOMODEL,
    ) -> None:
        """Load model and processor for a specific repository.

        Args:
            repo_id: HuggingFace repository ID
            revision: Model revision
            model_type: Type of model architecture
        """
        # Check for Phi-4 compatibility
        transformers_version = importlib.metadata.version("transformers")
        if (
            repo_id == "microsoft/Phi-4-multimodal-instruct"
            and transformers_version >= "4.52.0"
        ):
            raise NotImplementedError(
                f"Phi 4 only works with transformers<4.52.0 but you have {transformers_version=}. "
                f"Please downgrade by running: pip install -U 'transformers<4.52.0'"
            )

        # Download or locate model artifacts using shared utility
        def download_wrapper(repo_id: str, revision: str) -> Path:
            return self.download_models(repo_id, revision=revision)

        artifacts_path = resolve_model_artifacts_path(
            repo_id=repo_id,
            revision=revision,
            artifacts_path=self.artifacts_path,
            download_fn=download_wrapper,
        )

        # Setup quantization if needed
        quantization_config: Optional[BitsAndBytesConfig] = None
        if self.options.quantized:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.options.load_in_8bit,
                llm_int8_threshold=self.options.llm_int8_threshold,
            )

        # Select model class
        model_cls: type[
            Union[
                AutoModel,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
            ]
        ] = AutoModel
        if model_type == TransformersModelType.AUTOMODEL_CAUSALLM:
            model_cls = AutoModelForCausalLM  # type: ignore[assignment]
        elif model_type == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT:
            model_cls = AutoModelForImageTextToText  # type: ignore[assignment]

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            artifacts_path,
            trust_remote_code=self.options.trust_remote_code,
            revision=revision,
        )
        tokenizer = self._get_tokenizer()
        if tokenizer is not None and hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

        # Resolve torch_dtype: options override > extra_config > None
        torch_dtype = self.options.torch_dtype
        if torch_dtype is None and self.model_config is not None:
            torch_dtype = self.model_config.extra_config.get("torch_dtype")

        # Load model
        self.vlm_model = model_cls.from_pretrained(
            artifacts_path,
            device_map=self.device,
            dtype=torch_dtype,
            attn_implementation=self._get_attn_implementation(),
            trust_remote_code=self.options.trust_remote_code,
            revision=revision,
            quantization_config=quantization_config,
        )

        self.vlm_model.eval()

        # Optionally compile model for better performance (model must be in eval mode first)
        # Works for Python < 3.14 with any torch 2.x
        # Works for Python >= 3.14 with torch >= 2.10
        if self.options.compile_model:
            if sys.version_info < (3, 14):
                self.vlm_model = torch.compile(self.vlm_model)  # type: ignore[assignment]
            elif version.parse(torch.__version__) >= version.parse("2.10"):
                self.vlm_model = torch.compile(self.vlm_model)  # type: ignore[assignment]
            else:
                _log.warning(
                    "Model compilation requested but not available "
                    "(requires Python < 3.14 or torch >= 2.10 for Python 3.14+)"
                )

        # Load generation config
        self.generation_config = self._load_generation_config(
            artifacts_path=artifacts_path,
            revision=revision,
        )

        _log.info(f"Loaded model {repo_id} (revision: {revision})")

    def _load_generation_config(
        self,
        *,
        artifacts_path: Union[Path, str],
        revision: str,
    ) -> GenerationConfig:
        try:
            return GenerationConfig.from_pretrained(artifacts_path, revision=revision)
        except OSError as exc:
            if "generation_config.json" not in str(exc):
                raise
            if self.vlm_model is None:
                raise
            _log.warning(
                "Model %s does not provide generation_config.json; deriving generation config from model config instead.",
                self.model_config.repo_id
                if self.model_config is not None
                else artifacts_path,
            )
            return GenerationConfig.from_model_config(self.vlm_model.config)

    def _get_tokenizer(self) -> Any:
        """Resolve the tokenizer from the processor.

        Why: transformers v5 may return a tokenizer-like object directly from
        AutoProcessor.from_pretrained for pure-tokenizer processors (e.g.
        AUTOMODEL_CAUSALLM OCR models), whereas v4 and wrapper processors
        expose the tokenizer via a ``.tokenizer`` attribute.
        """
        if self.processor is None:
            return None
        return getattr(self.processor, "tokenizer", None) or self.processor

    def _get_attn_implementation(self) -> str:
        """Resolve the attention backend for model loading.

        Model-specific overrides take precedence over the engine defaults.
        """
        if self.model_config is not None:
            explicit_attn = self.model_config.extra_config.get("attn_implementation")
            if explicit_attn is None:
                explicit_attn = self.model_config.extra_config.get(
                    "_attn_implementation"
                )
            if explicit_attn is not None:
                return explicit_attn
            if _value_mentions_falcon_ocr(self.model_config.repo_id):
                return "eager"

        if (
            self.device is not None
            and self.device.startswith("cuda")
            and self.accelerator_options.cuda_use_flash_attention2
        ):
            return "flash_attention_2"

        return "sdpa"

    def _uses_falcon_ocr_native_generate(self) -> bool:
        return _value_mentions_falcon_ocr(
            self.model_config.repo_id if self.model_config is not None else None
        ) or _config_mentions_falcon_ocr(
            getattr(_unwrap_compiled_vlm_model(self.vlm_model), "config", None)
        )

    def _get_falcon_ocr_generation_kwargs(
        self,
        first_input: VlmEngineInput,
    ) -> dict[str, Any]:
        extra_generation_config = dict(first_input.extra_generation_config or {})
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": first_input.max_new_tokens,
            "temperature": first_input.temperature,
            "top_k": extra_generation_config.get("top_k"),
            "min_dimension": extra_generation_config.get("min_dimension", 64),
            "max_dimension": extra_generation_config.get("max_dimension", 1024),
        }
        if "seed" in extra_generation_config:
            generation_kwargs["seed"] = extra_generation_config["seed"]
        return generation_kwargs

    def _predict_batch_with_falcon_ocr_native_generate(
        self, input_batch: List[VlmEngineInput]
    ) -> List[VlmEngineOutput]:
        vlm_model = _unwrap_compiled_vlm_model(self.vlm_model)
        if vlm_model is None:
            raise RuntimeError("Falcon-OCR model is not loaded.")

        ensure_device_buffers = getattr(vlm_model, "_ensure_device_buffers", None)
        if callable(ensure_device_buffers):
            ensure_device_buffers()

        first_input = input_batch[0]
        generation_kwargs = self._get_falcon_ocr_generation_kwargs(first_input)
        metadata = {"falcon_ocr_native_generate": True}

        generate_batch = getattr(vlm_model, "_generate_batch", None)
        if callable(generate_batch):
            image_prompt_pairs = [
                (input_data.image, _normalize_falcon_ocr_prompt(input_data.prompt))
                for input_data in input_batch
            ]
            generated_texts = generate_batch(
                image_prompt_pairs,
                **generation_kwargs,
            )
        else:
            public_generate = getattr(vlm_model, "generate", None)
            if not callable(public_generate):
                raise RuntimeError(
                    "Falcon-OCR model exposes no compatible generate method."
                )

            generate_kwargs = {
                "category": [
                    _falcon_ocr_category_from_prompt(input_data.prompt)
                    for input_data in input_batch
                ],
                **generation_kwargs,
                "compile": False,
            }
            try:
                generated_texts = public_generate(
                    [input_data.image for input_data in input_batch],
                    **generate_kwargs,
                )
            except TypeError as exc:
                if "compile" not in str(exc):
                    raise
                generate_kwargs.pop("compile", None)
                generated_texts = public_generate(
                    [input_data.image for input_data in input_batch],
                    **generate_kwargs,
                )

            metadata["falcon_ocr_public_generate"] = True

        return [
            VlmEngineOutput(text=text, metadata=dict(metadata))
            for text in generated_texts
        ]

    def predict_batch(self, input_batch: List[VlmEngineInput]) -> List[VlmEngineOutput]:
        """Run inference on a batch of inputs efficiently.

        This method processes multiple images in a single forward pass,
        which is much more efficient than processing them sequentially.

        Args:
            input_batch: List of inputs to process

        Returns:
            List of outputs, one per input
        """
        if not self._initialized:
            self.initialize()

        if not input_batch:
            return []

        # Model should already be loaded via initialize()
        if self.vlm_model is None or self.processor is None:
            raise RuntimeError(
                "Model not loaded. Ensure EngineModelConfig was provided during initialization."
            )

        if (
            self._uses_falcon_ocr_native_generate()
            and _supports_falcon_ocr_native_generate(self.vlm_model)
        ):
            return self._predict_batch_with_falcon_ocr_native_generate(input_batch)

        # Get prompt style from first input's extra config
        first_input = input_batch[0]
        prompt_style = first_input.extra_generation_config.get(
            "transformers_prompt_style",
            TransformersPromptStyle.CHAT,
        )

        # Prepare images using shared utility
        images = preprocess_image_batch([inp.image for inp in input_batch])

        # Prepare prompts
        prompts = []
        for input_data in input_batch:
            # Format prompt
            if prompt_style == TransformersPromptStyle.CHAT:
                # Use structured message format with image placeholder (like legacy implementation)
                # This is required for vision models like Granite Vision to properly tokenize
                # both image features and text tokens
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": input_data.prompt},
                        ],
                    }
                ]
                from typing import cast

                formatted_prompt = self.processor.apply_chat_template(  # type: ignore[union-attr]
                    cast(Any, messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            elif prompt_style == TransformersPromptStyle.RAW:
                formatted_prompt = input_data.prompt
            else:  # NONE
                formatted_prompt = None

            prompts.append(formatted_prompt)

        # Process batch
        if prompt_style == TransformersPromptStyle.NONE:
            inputs = self.processor(  # type: ignore[misc]
                images,
                return_tensors="pt",
                padding=True,
                **first_input.extra_generation_config.get("extra_processor_kwargs", {}),
            )
        else:
            inputs = self.processor(  # type: ignore[misc]
                text=[prompt for prompt in prompts if prompt is not None],
                images=images,
                return_tensors="pt",
                padding=True,
                **first_input.extra_generation_config.get("extra_processor_kwargs", {}),
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Setup stopping criteria (use first input's config)
        stopping_criteria_list = StoppingCriteriaList()

        tokenizer = self._get_tokenizer()

        if first_input.stop_strings:
            stopping_criteria_list.append(
                StopStringCriteria(
                    stop_strings=first_input.stop_strings,
                    tokenizer=tokenizer,
                )
            )

        # Add custom stopping criteria using shared utility
        custom_stoppers = extract_generation_stoppers(
            first_input.extra_generation_config
        )
        for stopper in custom_stoppers:
            wrapped_criteria = HFStoppingCriteriaWrapper(
                tokenizer,
                stopper,
            )
            stopping_criteria_list.append(wrapped_criteria)

        # Also handle any HF StoppingCriteria directly passed
        custom_criteria = first_input.extra_generation_config.get(
            "custom_stopping_criteria", []
        )
        for criteria in custom_criteria:
            # Skip GenerationStopper instances (already handled above)
            if not isinstance(criteria, GenerationStopper) and not (
                isinstance(criteria, type) and issubclass(criteria, GenerationStopper)
            ):
                stopping_criteria_list.append(criteria)

        # Filter decoder-specific keys
        decoder_keys = {
            "skip_special_tokens",
            "clean_up_tokenization_spaces",
            "spaces_between_special_tokens",
        }
        generation_config = {
            k: v
            for k, v in first_input.extra_generation_config.items()
            if k not in decoder_keys
            and k
            not in {
                "transformers_model_type",
                "transformers_prompt_style",
                "extra_processor_kwargs",
                "custom_stopping_criteria",
                "revision",
            }
        }
        decoder_config = {
            k: v
            for k, v in first_input.extra_generation_config.items()
            if k in decoder_keys
        }

        # Generate
        gen_kwargs = {
            **inputs,
            "max_new_tokens": first_input.max_new_tokens,
            "use_cache": self.options.use_kv_cache,
            "generation_config": self.generation_config,
            **generation_config,
        }

        if first_input.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = first_input.temperature
        else:
            gen_kwargs["do_sample"] = False

        if stopping_criteria_list:
            gen_kwargs["stopping_criteria"] = stopping_criteria_list

        start_time = time.time()
        with torch.inference_mode():
            generated_ids = self.vlm_model.generate(**gen_kwargs)  # type: ignore[union-attr,operator]
        generation_time = time.time() - start_time

        # Decode
        input_len = inputs["input_ids"].shape[1]
        trimmed_sequences = generated_ids[:, input_len:]

        decode_fn = getattr(self.processor, "batch_decode", None)
        if decode_fn is None and tokenizer is not None:
            decode_fn = getattr(tokenizer, "batch_decode", None)
        if decode_fn is None:
            raise RuntimeError(
                "Neither processor.batch_decode nor tokenizer.batch_decode is available."
            )

        decoded_texts = decode_fn(trimmed_sequences, **decoder_config)

        # Remove padding
        pad_token = getattr(tokenizer, "pad_token", None)
        if pad_token:
            decoded_texts = [text.rstrip(pad_token) for text in decoded_texts]

        # Create outputs
        outputs = []
        for i, text in enumerate(decoded_texts):
            outputs.append(
                VlmEngineOutput(
                    text=text,
                    stop_reason="unspecified",
                    metadata={
                        "generation_time": generation_time / len(input_batch),
                        "num_tokens": int(generated_ids[i].shape[0])
                        if i < generated_ids.shape[0]
                        else None,
                        "batch_size": len(input_batch),
                    },
                )
            )

        _log.info(
            f"Batch processed {len(input_batch)} images in {generation_time:.2f}s "
            f"({generation_time / len(input_batch):.2f}s per image)"
        )

        return outputs

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.vlm_model is not None:
            del self.vlm_model
            self.vlm_model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        # Clear CUDA cache if using GPU
        if self.device and self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        _log.info("Transformers runtime cleaned up")
