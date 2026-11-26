# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Transformers-based VLM inference engine."""

import importlib
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
    StoppingCriteriaList,
    StopStringCriteria,
)

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import VlmStopReason
from docling.datamodel.pipeline_options_vlm_model import (
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.inference_engines.vlm._utils import (
    check_min_engine_version,
    extract_generation_stoppers,
    generation_stopper_stop_reason,
    preprocess_image_batch,
    resolve_model_artifacts_path,
)
from docling.models.inference_engines.vlm.base import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
    VlmEngineType,
)
from docling.models.utils.generation_utils import (
    GenerationStopper,
    build_generation_config,
)
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin
from docling.models.utils.hf_stopping_criteria import HFStoppingCriteriaWrapper
from docling.utils.accelerator_utils import decide_device
from docling.utils.granite_vision_utils import (
    GRANITE_VISION_4_REPO_ID,
    granite_vision_4_needs_remote_code,
)
from docling.utils.vlm_utils import strip_stop_strings, strip_trailing_token

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)

_DOTS_REPO_IDS = {"rednote-hilab/dots.ocr", "rednote-hilab/dots.mocr"}
_DOTS_FLASH_ATTN_REQUIRED_REPO_IDS = {"rednote-hilab/dots.mocr"}
_FALCON_OCR_REPO_ID = "tiiuae/Falcon-OCR"
# Falcon-OCR does not support SDPA yet; it must load with eager attention
# (see #3278). The ``_attn_implementation="eager"`` kwarg reaches
# ``FalconOCRConfig.__init__`` through the config kwargs, which forwards
# them to ``PreTrainedConfig``; this entry is the whole attention fix.
_EAGER_ATTN_REQUIRED_REPO_IDS = {
    "nvidia/NVIDIA-Nemotron-Parse-2.0",
    _FALCON_OCR_REPO_ID,
}


def _coerce_transformers_model_type(value: Any) -> TransformersModelType:
    if isinstance(value, TransformersModelType):
        return value
    return TransformersModelType(value)


def _ensure_dots_flash_attn_import() -> None:
    try:
        importlib.import_module("flash_attn")
    except ImportError as exc:
        raise ImportError(
            "rednote-hilab/dots.mocr requires flash-attn with the Transformers "
            "engine. Install flash-attn in the transformers-v4 environment "
            "before using this model."
        ) from exc


def _stopper_stop_reasons(
    wrappers: list[HFStoppingCriteriaWrapper], row_lengths: list[int]
) -> list[str]:
    """The stop reason of each batch row, as declared by the stopper that ended it.

    A stopper ended a row when it fired at the final length of that row.
    Transformers keeps calling the stoppers for rows that already finished, so a
    stopper that fired later did not end the row. If several stoppers ended a
    row at the same step, a cut output wins over a valid end. A row that no
    stopper ended stays "unspecified".
    """
    stop_reasons: list[str] = []
    for row, row_length in enumerate(row_lengths):
        reasons = {
            generation_stopper_stop_reason(wrapper.stopper)
            for wrapper in wrappers
            if wrapper.fired_at.get(row) == row_length
        }
        if VlmStopReason.STOP_SEQUENCE in reasons:
            stop_reasons.append(VlmStopReason.STOP_SEQUENCE.value)
        elif reasons:
            stop_reasons.append(VlmStopReason.END_OF_SEQUENCE.value)
        else:
            stop_reasons.append(VlmStopReason.UNSPECIFIED.value)
    return stop_reasons


def _falcon_ocr_generate(
    model: Any,
    images: list[Image],
    *,
    category: str = "plain",
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    top_k: int | None = None,
    min_dimension: int = 64,
    max_dimension: int = 1024,
    compile_model: bool = False,
    seed: int | None = None,
) -> list[str]:
    """Run Falcon-OCR's native ``generate`` on a batch of images.

    The remote-code model (pinned revision
    ``42ec56b72a23984ac059e7c8a6d397a8529423fe``) defines ``generate(images,
    *, category, max_new_tokens, temperature, top_k, min_dimension,
    max_dimension, compile, seed) -> list[str]`` and builds the prompt
    itself from the category. Defaults match the remote model's own
    defaults. ``category`` is one value for the whole batch; it comes from
    the model preset, not from the input prompts.
    """
    generate_kwargs: dict[str, Any] = {
        "category": category,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "min_dimension": min_dimension,
        "max_dimension": max_dimension,
        "compile": compile_model,
    }
    if seed is not None:
        generate_kwargs["seed"] = seed
    return model.generate(images, **generate_kwargs)


def _falcon_ocr_outputs(texts: list[str], expected_count: int) -> list[VlmEngineOutput]:
    """Build one output per input from Falcon-OCR's plain texts.

    The native ``generate`` returns only ``list[str]`` — no token counts
    and no end-of-sequence flag — so the stop reason is unknown, and
    ``stop_strings`` from ``VlmEngineInput`` are not applied on this path.
    """
    if len(texts) != expected_count:
        raise RuntimeError(
            f"Falcon-OCR returned {len(texts)} texts for {expected_count} images."
        )
    return [
        VlmEngineOutput(
            text=text,
            stop_reason=VlmStopReason.UNSPECIFIED.value,
            metadata={"falcon_ocr": "native_generate"},
        )
        for text in texts
    ]


class TransformersVlmEngine(BaseVlmEngine, HuggingFaceModelDownloadMixin):
    """HuggingFace Transformers engine for VLM inference.

    This engine uses the transformers library to run vision-language models
    locally on CPU, CUDA, or XPU devices.
    """

    def __init__(
        self,
        options: TransformersVlmEngineOptions,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Union[Path, str] | None,
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
        self.device: str | None = None
        self.processor: Any | None = None
        self.vlm_model: PreTrainedModel | None = None
        self.generation_config: GenerationConfig | None = None
        # ``config.model_type`` of the loaded model (for example "falcon_ocr").
        # ``_load_model_for_repo`` reads it once, before ``torch.compile``.
        self.model_type: str | None = None
        self.strip_stop_strings = (
            bool(
                model_config.extra_config.get("transformers_strip_stop_strings", False)
            )
            if model_config is not None
            else False
        )

        # Initialize immediately if model_config is provided
        if self.model_config is not None:
            self.initialize()

    def initialize(self) -> None:
        """Initialize the Transformers model and processor."""
        if self._initialized:
            return

        _log.info("Initializing Transformers VLM inference engine...")

        check_min_engine_version(
            VlmEngineType.TRANSFORMERS,
            self.model_config.min_engine_version if self.model_config else None,
        )

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
            model_type = _coerce_transformers_model_type(model_type)

            if (
                self.model_config.extra_config.get("requires_trust_remote_code")
                and not self.options.trust_remote_code
            ):
                raise ValueError(
                    f"{repo_id} (revision {revision}) runs code from its model "
                    "repository. Review that code, then opt in with "
                    "trust_remote_code=True on the VLM engine options, or with "
                    "--trust-remote-code in the docling CLI."
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
        transformers_version = importlib.metadata.version("transformers")
        parsed_transformers_version = version.parse(transformers_version)

        # Check for Phi-4 compatibility
        if (
            repo_id == "microsoft/Phi-4-multimodal-instruct"
            and parsed_transformers_version >= version.parse("4.52.0")
        ):
            raise NotImplementedError(
                f"Phi 4 only works with transformers<4.52.0 but you have {transformers_version=}. "
                f"Please downgrade by running: pip install -U 'transformers<4.52.0'"
            )
        is_dots_model = repo_id in _DOTS_REPO_IDS
        if is_dots_model and parsed_transformers_version.major >= 5:
            raise NotImplementedError(
                f"{repo_id} is supported by the Transformers engine only with "
                f"transformers<5, but you have {transformers_version=}. "
                "Use a transformers-v4 environment, or use the vLLM engine."
            )
        if repo_id in _DOTS_FLASH_ATTN_REQUIRED_REPO_IDS:
            _ensure_dots_flash_attn_import()

        trust_remote_code = self.options.trust_remote_code
        is_granite_vision_4 = repo_id == GRANITE_VISION_4_REPO_ID
        if is_granite_vision_4 and not granite_vision_4_needs_remote_code(
            transformers_version
        ):
            trust_remote_code = False

        # Download or locate model artifacts using shared utility
        def download_wrapper(repo_id: str, revision: str) -> Path:
            return self.download_models(
                repo_id,
                revision=revision,
                trust_remote_code=self.options.trust_remote_code,
            )

        artifacts_path = resolve_model_artifacts_path(
            repo_id=repo_id,
            revision=revision,
            artifacts_path=self.artifacts_path,
            download_fn=download_wrapper,
        )

        # Setup quantization if needed
        quantization_config: BitsAndBytesConfig | None = None
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

        self.processor = AutoProcessor.from_pretrained(
            artifacts_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        tokenizer = self._get_tokenizer()
        if tokenizer is not None and hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

        # Resolve torch_dtype: options override > model_config field > extra_config > None
        torch_dtype = self.options.torch_dtype
        if torch_dtype is None and self.model_config is not None:
            torch_dtype = (
                self.model_config.torch_dtype
                or self.model_config.extra_config.get("torch_dtype")
            )

        # Load model
        attn_implementation: Optional[str] = (
            "flash_attention_2"
            if self.device.startswith("cuda")  # type: ignore[union-attr]
            and self.accelerator_options.cuda_use_flash_attention2
            else "sdpa"
        )
        if is_dots_model:
            attn_implementation = "sdpa"
        elif is_granite_vision_4 and attn_implementation == "sdpa":
            # The native granite4_vision Q-Former rejects an explicit sdpa
            # request before transformers 5.13; the transformers default
            # selects sdpa where the model supports it.
            attn_implementation = None

        if repo_id in _EAGER_ATTN_REQUIRED_REPO_IDS:
            attn_implementation = "eager"

        dtype_arg_name = (
            "dtype" if parsed_transformers_version.major >= 5 else "torch_dtype"
        )

        _log.info(f"Loading model {repo_id} into memory (device: {self.device})...")
        load_start_time = time.monotonic()
        self.vlm_model = model_cls.from_pretrained(
            artifacts_path,
            device_map=self.device,
            **{dtype_arg_name: torch_dtype},
            _attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            revision=revision,
            quantization_config=quantization_config,
        )

        self.vlm_model.eval()

        # The loaded model declares its own identity: Falcon-OCR's config
        # reports ``model_type`` "falcon_ocr", which selects the native
        # generate path in ``predict_batch``. Read it before
        # ``torch.compile`` wraps the model; a local copy or a fine-tune
        # keeps the same ``model_type`` and takes the same path.
        self.model_type = self.vlm_model.config.model_type

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

        _log.info(
            f"Loaded model {repo_id} (revision: {revision}) in "
            f"{time.monotonic() - load_start_time:.2f} sec."
        )

    def _load_generation_config(
        self,
        *,
        artifacts_path: Union[Path, str],
        revision: str,
    ) -> GenerationConfig:
        """Load the generation config, deriving it when the repo ships none.

        Some models (for example Falcon-OCR) ship no
        ``generation_config.json``; without the fallback the engine fails
        before the first page is read. ``artifacts_path`` is always a local
        directory at this point, so a missing file is checked directly.
        """
        if (Path(artifacts_path) / "generation_config.json").exists():
            return GenerationConfig.from_pretrained(artifacts_path, revision=revision)
        if self.vlm_model is None:
            raise RuntimeError(
                "Model not loaded. Ensure EngineModelConfig was provided during initialization."
            )
        _log.warning(
            "Model %s does not ship generation_config.json; deriving the "
            "generation config from the model config instead.",
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

    def _predict_falcon_batch(
        self, input_batch: List[VlmEngineInput]
    ) -> List[VlmEngineOutput]:
        """Run Falcon-OCR through its native ``generate`` API.

        The remote-code model ships no usable chat template, so the generic
        chat-template processor flow cannot run it. The OCR category comes
        from the model preset (``falcon_ocr_category``, default ``"plain"``);
        it is data for this path, not a switch that selects which code runs.
        The native ``generate`` returns only ``list[str]`` — no token counts
        and no end-of-sequence flag — so ``stop_strings`` from
        ``VlmEngineInput`` are not applied on this path and the stop reason
        stays unspecified.
        """
        if not input_batch:
            return []
        if self.vlm_model is None:
            raise RuntimeError(
                "Model not loaded. Ensure EngineModelConfig was provided during initialization."
            )

        first_input = input_batch[0]
        extra_generation_config = dict(first_input.extra_generation_config or {})
        extra_config = (
            self.model_config.extra_config if self.model_config is not None else {}
        )
        category = str(extra_config.get("falcon_ocr_category", "plain"))

        _log.info(
            "Running Falcon-OCR native generate on %s image(s) on %s (max_new_tokens=%s)...",
            len(input_batch),
            self.device,
            first_input.max_new_tokens,
        )
        start_time = time.time()
        with torch.inference_mode():
            generated_texts = _falcon_ocr_generate(
                self.vlm_model,
                [input_data.image for input_data in input_batch],
                category=category,
                max_new_tokens=first_input.max_new_tokens,
                temperature=first_input.temperature,
                top_k=extra_generation_config.get("top_k"),
                min_dimension=extra_generation_config.get("min_dimension", 64),
                max_dimension=extra_generation_config.get("max_dimension", 1024),
                compile_model=self.options.compile_model,
                seed=extra_generation_config.get("seed"),
            )
        generation_time = time.time() - start_time
        _log.info(
            "Falcon-OCR generated %s text(s) for %s image(s) in %.2f sec.",
            len(generated_texts),
            len(input_batch),
            generation_time,
        )

        return _falcon_ocr_outputs(generated_texts, len(input_batch))

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
        if self.vlm_model is None:
            raise RuntimeError(
                "Model not loaded. Ensure EngineModelConfig was provided during initialization."
            )

        # Falcon-OCR ships no usable chat template and its remote-code model
        # exposes a native OCR generation API; the generic chat-template
        # processor flow cannot run it. The loaded model declares its own
        # identity via ``config.model_type``, read once at load time.
        if self.model_type == "falcon_ocr":
            return self._predict_falcon_batch(input_batch)

        if self.processor is None:
            raise RuntimeError(
                "Processor not loaded. Ensure EngineModelConfig was provided during initialization."
            )

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
        stopper_wrappers: list[HFStoppingCriteriaWrapper] = []
        for stopper in custom_stoppers:
            wrapped_criteria = HFStoppingCriteriaWrapper(
                tokenizer,
                stopper,
            )
            stopper_wrappers.append(wrapped_criteria)
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
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        merged_generation_config = build_generation_config(
            self.generation_config,
            overrides=generation_config,
            max_new_tokens=first_input.max_new_tokens,
            use_cache=self.options.use_kv_cache,
            do_sample=first_input.temperature > 0,
            temperature=(
                first_input.temperature if first_input.temperature > 0 else None
            ),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=eos_token_id,
        )
        gen_kwargs = {
            **inputs,
            "generation_config": merged_generation_config,
        }

        if stopping_criteria_list:
            gen_kwargs["stopping_criteria"] = stopping_criteria_list

        _log.info(
            "Running Transformers inference on %s image(s) on %s (max_new_tokens=%s)...",
            len(input_batch),
            self.device,
            first_input.max_new_tokens,
        )
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
            decoded_texts = strip_trailing_token(decoded_texts, pad_token)

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            generated_token_counts = [
                int(trimmed_sequences.shape[1])
            ] * trimmed_sequences.shape[0]
        else:
            generated_token_counts = (
                (trimmed_sequences != pad_token_id).sum(dim=1).tolist()
            )
        eos_token_ids = merged_generation_config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = eos_token_id
        ended_with_eos = [False] * len(input_batch)
        if eos_token_ids is not None:
            # Finished batch items may have padding after EOS, including EOS as pad.
            ended_with_eos = (
                torch.isin(
                    trimmed_sequences,
                    torch.tensor(eos_token_ids, device=trimmed_sequences.device),
                )
                .any(dim=1)
                .tolist()
            )
        total_generated_tokens = sum(generated_token_counts)

        _log.info(
            "Transformers generated %s tokens for %s image(s) in %.2f sec. (%.2f tok/s)",
            total_generated_tokens,
            len(input_batch),
            generation_time,
            total_generated_tokens / generation_time if generation_time > 0 else 0.0,
        )

        if self.strip_stop_strings and first_input.stop_strings:
            decoded_texts = strip_stop_strings(decoded_texts, first_input.stop_strings)

        stop_reasons = _stopper_stop_reasons(
            stopper_wrappers,
            [input_len + count for count in generated_token_counts],
        )

        # Create outputs
        outputs = []
        for i, text in enumerate(decoded_texts):
            stop_reason = stop_reasons[i] if i < len(stop_reasons) else "unspecified"
            if (
                stop_reason == "unspecified"
                and not ended_with_eos[i]
                and merged_generation_config.max_new_tokens is not None
                and generated_token_counts[i] >= merged_generation_config.max_new_tokens
            ):
                stop_reason = "length"
            outputs.append(
                VlmEngineOutput(
                    text=text,
                    stop_reason=stop_reason,
                    metadata={
                        "generation_time": generation_time / len(input_batch),
                        "num_tokens": generated_token_counts[i]
                        if i < len(generated_token_counts)
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
