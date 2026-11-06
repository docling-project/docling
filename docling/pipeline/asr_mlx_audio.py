# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Speech decoding through the mlx-audio library on Apple Silicon.

Importing `mlx` or `mlx_audio` initializes Metal, which can abort a restricted or
headless process. This module is on the import path of every install, so it must
import them only inside the decoder.
"""

import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

from docling.datamodel.asr_presets import is_apple_silicon
from docling.datamodel.pipeline_options_asr_model import InlineAsrMlxAudioOptions
from docling.models.utils.hf_model_download import (
    is_pinned_revision,
    warn_on_unpinned_trust_remote_code,
)
from docling.models.utils.mlx_lock import MLX_GLOBAL_LOCK
from docling.pipeline.asr_conversation import _ConversationItem, _ConversationWord

_log = logging.getLogger(__name__)

MISSING_MLX_AUDIO_MESSAGE: Final[str] = (
    "mlx-audio is not installed. Install it with "
    "`pip install 'docling-slim[feat-asr-mlx-audio]'`."
)

# The Nemotron streaming decoder holds back the last tokens of the audio. Added
# silence lets it emit them.
_NEMOTRON_TAIL_PADDING_SECONDS: Final = 2.0
# The mlx-audio Granite default is 4096. This limit is far above the speech in one
# window, and it stops a repetition loop early.
_GRANITE_MAX_TOKENS_PER_WINDOW: Final = 1024
_ENGLISH: Final = "en"
_METAL_PROBE: Final = (
    "import mlx.core as mx; raise SystemExit(0 if mx.metal.is_available() else 1)"
)
_METAL_PROBE_TIMEOUT_SECONDS: Final = 60


def metal_is_usable() -> bool:
    """Tell if this machine can import `mlx` and use a Metal device.

    The check runs in a child process, because the import can abort a restricted or
    headless process, and a caller cannot catch that.
    """
    if not is_apple_silicon():
        return False
    try:
        probe = subprocess.run(
            [sys.executable, "-c", _METAL_PROBE],
            capture_output=True,
            timeout=_METAL_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _log.debug(f"The Metal check did not finish: {exc}")
        return False
    return probe.returncode == 0


def _nemotron_prompt_key(language: str, prompt_keys: list[str]) -> str:
    """Map a language to a prompt key of the checkpoint.

    The model silently uses its default language for an unknown key, so reject the
    key here. A checkpoint without prompt keys is English only.
    """
    if not prompt_keys:
        if language.lower().split("-")[0] == _ENGLISH:
            return language
        raise ValueError(
            f"This Nemotron checkpoint is English only, not language '{language}'."
        )
    if language in prompt_keys:
        return language
    prefix = f"{language.lower()}-"
    for key in prompt_keys:
        if key.lower().startswith(prefix):
            return key
    raise ValueError(
        f"This Nemotron checkpoint does not support language '{language}'. "
        f"Supported values: {', '.join(sorted(prompt_keys))}."
    )


class MlxAudioAsrDecoder:
    """Load one mlx-audio checkpoint and convert its output to timed segments."""

    def __init__(self, options: InlineAsrMlxAudioOptions, checkpoint: Path | None):
        """Load the model.

        Args:
            options: The backend options.
            checkpoint: A local checkpoint directory, or None to use the Hub.
        """
        if not is_apple_silicon():
            raise RuntimeError(
                "The mlx-audio ASR backend needs Apple Silicon. Select a different "
                "ASR model on this machine."
            )
        self.options = options
        self._check_remote_code(checkpoint)
        self._validate_options()

        try:
            import mlx.core as mx
            from mlx_audio.stt import load
            from mlx_audio.stt.utils import SAMPLE_RATE, load_audio
        except ImportError as exc:
            raise ImportError(MISSING_MLX_AUDIO_MESSAGE) from exc
        if not mx.metal.is_available():
            raise RuntimeError("The mlx-audio ASR backend needs a Metal device.")

        self._mx = mx
        self._load_audio = load_audio
        self._sample_rate: int = SAMPLE_RATE
        _log.info(f"loading mlx-audio model {options.repo_id}")
        self.model = (
            load(checkpoint)
            if checkpoint is not None
            else load(options.repo_id, revision=options.revision)
        )

    def _check_remote_code(self, checkpoint: Path | None) -> None:
        """Apply the remote-code policy to the Granite family.

        mlx-audio always loads the Granite tokenizer with `trust_remote_code=True`,
        and the caller cannot disable this.
        """
        options = self.options
        if options.model_family not in {"granite", "granite_plus"}:
            return
        if not options.trust_remote_code and not is_pinned_revision(options.revision):
            raise ValueError(
                f"mlx-audio runs the remote code of Granite checkpoint "
                f"'{options.repo_id}'. Set `revision` to a full commit SHA, or set "
                "`trust_remote_code=True` to accept this revision."
            )
        if checkpoint is None:
            warn_on_unpinned_trust_remote_code(
                options.repo_id, options.revision, trust_remote_code=True
            )

    def _validate_options(self) -> None:
        options = self.options
        family = options.model_family
        if options.task == "translate" and family not in {"whisper", "granite"}:
            raise ValueError(f"The {family} family does not support task='translate'.")
        if options.prompt is not None and family not in {
            "whisper",
            "granite",
            "granite_plus",
        }:
            raise ValueError(f"The {family} family does not support a prompt.")
        if (
            family in {"parakeet", "granite_ctc"}
            and options.language is not None
            and options.language.lower() != _ENGLISH
        ):
            raise ValueError(
                f"The {family} checkpoints are English only, not language "
                f"'{options.language}'."
            )

    def transcribe(self, path: Path) -> list[_ConversationItem]:
        family = self.options.model_family
        if family == "whisper":
            return self._transcribe_whisper(path)
        if family in {"granite", "granite_ctc"}:
            return self._transcribe_granite_windows(path)
        if family == "granite_plus":
            return self._transcribe_granite_plus(path)
        if family == "parakeet":
            with MLX_GLOBAL_LOCK:
                return self._sentence_segments(self.model.generate(str(path)))
        if family == "nemotron":
            return self._transcribe_nemotron(path)
        raise ValueError(f"mlx-audio backend has no support for family {family}")

    def _transcribe_whisper(self, path: Path) -> list[_ConversationItem]:
        options = self.options
        kwargs: dict[str, Any] = {
            "language": options.language,
            "task": options.task,
            "word_timestamps": options.word_timestamps,
            "temperature": options.temperature,
            # mlx-audio shows a progress bar for False, and is silent for None.
            "verbose": True if options.verbose else None,
        }
        if options.prompt is not None:
            kwargs["initial_prompt"] = options.prompt
        with MLX_GLOBAL_LOCK:
            result = self.model.generate(str(path), **kwargs)
        return [
            _ConversationItem(
                text=segment["text"].strip(),
                start_time=segment["start"],
                end_time=segment["end"],
                words=[
                    _ConversationWord(
                        text=word["word"],
                        start_time=word["start"],
                        end_time=word["end"],
                    )
                    for word in segment["words"]
                ]
                if options.word_timestamps and "words" in segment
                else None,
            )
            for segment in result.segments
        ]

    @staticmethod
    def _sentence_segments(
        result: Any, duration: float = math.inf
    ) -> list[_ConversationItem]:
        return [
            _ConversationItem(
                text=sentence.text.strip(),
                start_time=min(sentence.start, duration),
                end_time=min(sentence.end, duration),
            )
            for sentence in result.sentences
        ]

    def _transcribe_nemotron(self, path: Path) -> list[_ConversationItem]:
        mx = self._mx
        kwargs: dict[str, Any] = {}
        if self.options.language is not None:
            kwargs["language"] = _nemotron_prompt_key(
                self.options.language, list(self.model.prompt_dictionary)
            )
        silence = mx.zeros(
            int(_NEMOTRON_TAIL_PADDING_SECONDS * self._sample_rate), dtype=mx.float32
        )
        with MLX_GLOBAL_LOCK:
            speech = self._load_audio(str(path))
            result = self.model.generate(mx.concatenate([speech, silence]), **kwargs)
        # The added silence must not extend a timestamp past the real audio.
        return self._sentence_segments(result, speech.size / self._sample_rate)

    def _granite_kwargs(self) -> dict[str, Any]:
        options = self.options
        if options.model_family == "granite_ctc":
            return {}
        kwargs: dict[str, Any] = {
            "max_tokens": _GRANITE_MAX_TOKENS_PER_WINDOW,
            "temperature": options.temperature,
            "punctuate": True,
        }
        if options.prompt is not None:
            kwargs["prompt"] = options.prompt
        # In mlx-audio, Granite `language` is the translation target and
        # `source_language` is the spoken language. They exclude each other.
        if options.task == "translate":
            kwargs["language"] = _ENGLISH
        elif options.language is not None:
            kwargs["source_language"] = options.language
        return kwargs

    def _transcribe_granite_plus(self, path: Path) -> list[_ConversationItem]:
        # Both need numpy, which must not reach the import path of a slim install.
        import numpy as np

        from docling.pipeline.asr_granite_plus import transcribe_plus

        options = self.options
        with MLX_GLOBAL_LOCK:
            waveform = np.array(self._load_audio(str(path)))
        return transcribe_plus(
            waveform,
            self._sample_rate,
            options.max_time_chunk,
            options.chunk_search_window,
            options.word_timestamps,
            options.speaker_attribution,
            self._decode_plus,
        )

    def _decode_plus(self, waveform: Any, task: Any) -> str:
        from mlx_audio.stt.models.granite_speech.granite_speech import (
            StructuredTranscriptError,
        )

        from docling.pipeline.asr_granite_plus import PLUS_TOKENS_PER_SECOND

        kwargs: dict[str, Any] = {
            "task": task,
            "max_tokens": max(
                _GRANITE_MAX_TOKENS_PER_WINDOW,
                int(PLUS_TOKENS_PER_SECOND[task] * waveform.size / self._sample_rate),
            ),
            "temperature": self.options.temperature,
        }
        if task == "asr" and self.options.prompt is not None:
            kwargs["prompt"] = self.options.prompt
        try:
            with MLX_GLOBAL_LOCK:
                return self.model.generate(self._mx.array(waveform), **kwargs).text
        except StructuredTranscriptError as exc:
            # Malformed tags must not lose the text of the window.
            _log.debug(f"Granite Speech Plus gave malformed {task} output: {exc}")
            return exc.raw_text

    def _transcribe_granite_windows(self, path: Path) -> list[_ConversationItem]:
        """Decode the file in windows, because the models return no timestamps.

        Windows also stop the repetition loop that Granite shows on long audio.
        """
        # Both need numpy, which must not reach the import path of a slim install.
        import numpy as np

        from docling.pipeline.asr_windows import split_on_pauses

        mx = self._mx
        options = self.options
        with MLX_GLOBAL_LOCK:
            waveform = np.array(self._load_audio(str(path)))
        windows = split_on_pauses(
            waveform,
            self._sample_rate,
            options.max_time_chunk,
            options.chunk_search_window,
        )
        kwargs = self._granite_kwargs()
        segments: list[_ConversationItem] = []
        for start, end in windows:
            first, last = (round(edge * self._sample_rate) for edge in (start, end))
            # Take the lock for each window, so that long audio does not hold back
            # other MLX work for its whole duration.
            with MLX_GLOBAL_LOCK:
                text = self.model.generate(
                    mx.array(waveform[first:last]), **kwargs
                ).text.strip()
            if text:
                segments.append(
                    _ConversationItem(text=text, start_time=start, end_time=end)
                )
        return segments
