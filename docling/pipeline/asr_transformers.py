# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Model-specific speech decoding through documented Transformers interfaces."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from docling.datamodel.pipeline_options_asr_model import InlineAsrTransformersOptions
from docling.pipeline.asr_conversation import _ConversationItem

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

_SAMPLE_RATE: Final = 16000
_SEQ2SEQ_GRANITE_FAMILIES: Final = {"granite", "granite_plus"}


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
        if device == "mps" and options.model_family in {
            *_SEQ2SEQ_GRANITE_FAMILIES,
            "granite_ctc",
        }:
            model_kwargs["dtype"] = "float32"

        self.processor = AutoProcessor.from_pretrained(checkpoint, **load_kwargs)
        if options.model_family == "granite_ctc":
            self.model = AutoModelForCTC.from_pretrained(checkpoint, **model_kwargs)
        else:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                checkpoint, **model_kwargs
            )
        self.model.to(device).eval()
        if options.model_family in _SEQ2SEQ_GRANITE_FAMILIES:
            # Transformers 5.16.1 passes a single query batch for all audio
            # windows. Its Q-Former can return only one window, so the projector
            # then fails to reshape the output. Scope this correction to the
            # loaded instance; already expanded queries are left intact.
            self.model.model.projector.qformer.register_forward_pre_hook(
                _expand_granite_queries, with_kwargs=True
            )

    def transcribe(self, path: Path) -> list[_ConversationItem]:
        """Decode the file in windows, because the models return no timestamps.

        Windows also stop the repetition loop that Granite shows on long audio.
        """
        from transformers.pipelines.audio_utils import ffmpeg_read

        from docling.pipeline.asr_windows import split_on_pauses

        waveform = ffmpeg_read(path.read_bytes(), sampling_rate=_SAMPLE_RATE).copy()
        if self.options.model_family == "granite_plus":
            from docling.pipeline.asr_granite_plus import transcribe_plus

            return transcribe_plus(
                waveform,
                _SAMPLE_RATE,
                self.options.max_time_chunk,
                self.options.chunk_search_window,
                self.options.word_timestamps,
                self.options.speaker_attribution,
                self._decode_plus,
            )
        windows = split_on_pauses(
            waveform,
            _SAMPLE_RATE,
            self.options.max_time_chunk,
            self.options.chunk_search_window,
        )
        segments: list[_ConversationItem] = []
        for start, end in windows:
            first, last = (round(edge * _SAMPLE_RATE) for edge in (start, end))
            text = self._decode(waveform[first:last])
            if text:
                segments.append(
                    _ConversationItem(text=text, start_time=start, end_time=end)
                )
        return segments

    def _decode(self, waveform: "npt.NDArray[np.floating]") -> str:
        import torch

        with torch.inference_mode():
            if self.options.model_family == "granite_ctc":
                inputs = self.processor(
                    [waveform],
                    sampling_rate=_SAMPLE_RATE,
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
            return self._generate(
                waveform,
                [{"role": "user", "content": f"<|audio|>{prompt}"}],
                self.options.max_new_tokens,
            )

    def _decode_plus(self, waveform: "npt.NDArray[np.floating]", task: Any) -> str:
        import torch

        from docling.pipeline.asr_granite_plus import (
            PLUS_PROMPTS,
            PLUS_SYSTEM_PROMPT,
            PLUS_TOKENS_PER_SECOND,
        )

        prompt = self.options.prompt if task == "asr" else None
        seconds = waveform.size / _SAMPLE_RATE
        with torch.inference_mode():
            return self._generate(
                waveform,
                [
                    {"role": "system", "content": PLUS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"<|audio|>{prompt or PLUS_PROMPTS[task]}",
                    },
                ],
                max(
                    self.options.max_new_tokens,
                    int(PLUS_TOKENS_PER_SECOND[task] * seconds),
                ),
            )

    def _generate(
        self,
        waveform: "npt.NDArray[np.floating]",
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        import torch

        formatted = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            formatted,
            torch.from_numpy(waveform).unsqueeze(0),
            device=self.device,
            return_tensors="pt",
        ).to(self.device)
        # Do not add the `repetition_penalty` of the model card. It also applies to the
        # prompt tokens, so it removes the spaces and the tags, and it stops the model
        # from ending. Short windows and the token limit stop a loop.
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        generated = output[:, inputs["input_ids"].shape[-1] :]
        return self.processor.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()
