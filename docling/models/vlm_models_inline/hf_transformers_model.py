import importlib.metadata
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
)
from docling.datamodel.base_models import Page, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import (
    InlineVlmOptions,
    TransformersModelType,
)
from docling.models.base_model import BasePageModel
from docling.models.utils.hf_model_download import (
    HuggingFaceModelDownloadMixin,
)
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class HuggingFaceTransformersVlmModel(BasePageModel, HuggingFaceModelDownloadMixin):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        vlm_options: InlineVlmOptions,
    ):
        self.enabled = enabled

        self.vlm_options = vlm_options

        if self.enabled:
            import torch
            from transformers import (
                AutoModel,
                AutoModelForCausalLM,
                AutoModelForVision2Seq,
                AutoProcessor,
                BitsAndBytesConfig,
                GenerationConfig,
            )

            transformers_version = importlib.metadata.version("transformers")
            if (
                self.vlm_options.repo_id == "microsoft/Phi-4-multimodal-instruct"
                and transformers_version >= "4.52.0"
            ):
                raise NotImplementedError(
                    f"Phi 4 only works with transformers<4.52.0 but you have {transformers_version=}. Please downgrage running pip install -U 'transformers<4.52.0'."
                )

            self.device = decide_device(
                accelerator_options.device,
                supported_devices=vlm_options.supported_devices,
            )
            _log.debug(f"Available device for VLM: {self.device}")

            self.use_cache = vlm_options.use_kv_cache
            self.max_new_tokens = vlm_options.max_new_tokens
            self.temperature = vlm_options.temperature

            repo_cache_folder = vlm_options.repo_id.replace("/", "--")

            if artifacts_path is None:
                artifacts_path = self.download_models(self.vlm_options.repo_id)
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            self.param_quantization_config: Optional[BitsAndBytesConfig] = None
            if vlm_options.quantized:
                self.param_quantization_config = BitsAndBytesConfig(
                    load_in_8bit=vlm_options.load_in_8bit,
                    llm_int8_threshold=vlm_options.llm_int8_threshold,
                )

            model_cls: Any = AutoModel
            if (
                self.vlm_options.transformers_model_type
                == TransformersModelType.AUTOMODEL_CAUSALLM
            ):
                model_cls = AutoModelForCausalLM
            elif (
                self.vlm_options.transformers_model_type
                == TransformersModelType.AUTOMODEL_VISION2SEQ
            ):
                model_cls = AutoModelForVision2Seq

            self.processor = AutoProcessor.from_pretrained(
                artifacts_path,
                trust_remote_code=vlm_options.trust_remote_code,
            )
            self.vlm_model = model_cls.from_pretrained(
                artifacts_path,
                device_map=self.device,
                _attn_implementation=(
                    "flash_attention_2"
                    if self.device.startswith("cuda")
                    and accelerator_options.cuda_use_flash_attention2
                    else "eager"
                ),
                trust_remote_code=vlm_options.trust_remote_code,
            )

            # Load generation config
            self.generation_config = GenerationConfig.from_pretrained(artifacts_path)

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "vlm"):
                    assert page.size is not None

                    hi_res_image = page.get_image(scale=self.vlm_options.scale)

                    # Define prompt structure
                    prompt = self.formulate_prompt()

                    inputs = self.processor(
                        text=prompt, images=[hi_res_image], return_tensors="pt"
                    ).to(self.device)

                    start_time = time.time()
                    # Call model to generate:
                    generated_ids = self.vlm_model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=self.use_cache,
                        temperature=self.temperature,
                        generation_config=self.generation_config,
                        **self.vlm_options.extra_generation_config,
                    )

                    generation_time = time.time() - start_time
                    generated_texts = self.processor.batch_decode(
                        generated_ids[:, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=False,
                    )[0]

                    num_tokens = len(generated_ids[0])
                    _log.debug(
                        f"Generated {num_tokens} tokens in time {generation_time:.2f} seconds."
                    )
                    page.predictions.vlm_response = VlmPrediction(
                        text=generated_texts,
                        generation_time=generation_time,
                    )

                yield page

    def formulate_prompt(self) -> str:
        """Formulate a prompt for the VLM."""

        if self.vlm_options.repo_id == "microsoft/Phi-4-multimodal-instruct":
            _log.debug("Using specialized prompt for Phi-4")
            # more info here: https://huggingface.co/microsoft/Phi-4-multimodal-instruct#loading-the-model-locally

            user_prompt = "<|user|>"
            assistant_prompt = "<|assistant|>"
            prompt_suffix = "<|end|>"

            prompt = f"{user_prompt}<|image_1|>{self.vlm_options.prompt}{prompt_suffix}{assistant_prompt}"
            _log.debug(f"prompt for {self.vlm_options.repo_id}: {prompt}")

            return prompt

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is a page from a document.",
                    },
                    {"type": "image"},
                    {"type": "text", "text": self.vlm_options.prompt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=False
        )
        return prompt
