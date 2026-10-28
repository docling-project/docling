# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Model-specific speech decoding through documented Transformers interfaces."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from docling.datamodel.pipeline_options_asr_model import InlineAsrTransformersOptions

if TYPE_CHECKING:
    import torch


def _expand_granite_queries(
    _module: "torch.nn.Module", args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Match Q-Former queries to the projector's batch of audio windows."""
    queries = kwargs["query_embeds"]
    window_batch = kwargs["encoder_hidden_states"].shape[0]
    if queries.shape[0] == 1 and window_batch > 1:
        kwargs["query_embeds"] = queries.expand(window_batch, -1, -1)
        return args, kwargs
    return None


class TransformersAsrDecoder:
    """Decode models that need a processor or custom model method."""

    def __init__(
        self,
        options: InlineAsrTransformersOptions,
        device: str,
        checkpoint: str,
        local: bool,
    ) -> None:
        # These imports belong to the optional Transformers ASR feature.
        from transformers import (
            AutoModelForCTC,
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
        )

        self.options = options
        self.device = device
        load_kwargs: dict[str, Any] = {
            "local_files_only": local,
        }
        if not local:
            load_kwargs["revision"] = options.revision

        model_kwargs = dict(load_kwargs)
        if options.torch_dtype is not None:
            model_kwargs["dtype"] = options.torch_dtype
        # Granite's BF16 normalization can abort the MPS process. Set the
        # loading dtype explicitly; catching an exception cannot recover it.
        if device == "mps" and options.model_family in {"granite", "granite_ctc"}:
            model_kwargs["dtype"] = "float32"

        self.processor = AutoProcessor.from_pretrained(checkpoint, **load_kwargs)
        if options.model_family == "granite_ctc":
            self.model = AutoModelForCTC.from_pretrained(checkpoint, **model_kwargs)
        else:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                checkpoint, **model_kwargs
            )
        self.model.to(device).eval()
        if options.model_family == "granite":
            # Transformers 5.16.1 passes a single query batch for all audio
            # windows. Its Q-Former can return only one window, so the projector
            # then fails to reshape the output. Scope this correction to the
            # loaded instance; already expanded queries are left intact.
            self.model.model.projector.qformer.register_forward_pre_hook(
                _expand_granite_queries, with_kwargs=True
            )

    def transcribe(self, path: Path) -> str:
        import torch
        from transformers.pipelines.audio_utils import ffmpeg_read

        waveform = ffmpeg_read(path.read_bytes(), sampling_rate=16000).copy()
        if waveform.size == 0:
            return ""
        with torch.inference_mode():
            if self.options.model_family == "granite_ctc":
                inputs = self.processor(
                    [waveform],
                    sampling_rate=16000,
                    device=self.device,
                    return_tensors="pt",
                ).to(self.device)
                output = self.model.generate(**inputs)
                return self.processor.batch_decode(output, skip_special_tokens=True)[
                    0
                ].strip()
            # Granite's decoder is a language model: it needs an audio prompt,
            # and generate() includes the input tokens in the returned sequence.
            prompt = self.options.prompt or (
                "translate the speech into English."
                if self.options.task == "translate"
                else "transcribe the speech with proper punctuation and capitalization."
            )
            formatted = self.processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"<|audio|>{prompt}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                formatted,
                torch.from_numpy(waveform).unsqueeze(0),
                device=self.device,
                return_tensors="pt",
            ).to(self.device)
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.options.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            generated = output[:, inputs["input_ids"].shape[-1] :]
            return self.processor.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()
