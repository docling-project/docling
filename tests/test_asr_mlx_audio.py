# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the mlx-audio ASR backend, the `mlx` alias, and pause-aware windows."""

import importlib.metadata
import itertools
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from docling.cli.main import _resolve_asr_options
from docling.datamodel import asr_presets
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.asr_model_specs import (
    GRANITE_SPEECH_4_1_2B_PRESET,
    GRANITE_SPEECH_5_0_2B_PRESET,
    PARAKEET_CTC_1_1B_MLX_AUDIO,
    PARAKEET_CTC_1_1B_PRESET,
    WHISPER_ASR_PRESETS,
    WHISPER_TINY_MLX,
    AsrModelType,
)
from docling.datamodel.pipeline_options_asr_model import (
    InferenceAsrFramework,
    InlineAsrMlxAudioOptions,
    InlineAsrMlxWhisperOptions,
    InlineAsrTransformersOptions,
)
from docling.pipeline import asr_mlx_audio
from docling.pipeline.asr_mlx_audio import MlxAudioAsrDecoder
from docling.pipeline.asr_transcriber import _AsrModelFactory, _MlxAudioModel
from docling.pipeline.asr_windows import split_on_pauses

pytestmark = pytest.mark.ml_asr

SAMPLE_RATE = 16000
SHA = "a" * 40
REPO_ROOT = Path(__file__).resolve().parent.parent
SPEECH_SAMPLE = REPO_ROOT / "tests/data/audio/sources/sample_10s.mp3"
RUNS_MLX = find_spec("mlx_audio") is not None and sys.platform == "darwin"


class _Model:
    """Stand-in that records the calls and returns one fixed output."""

    def __init__(self, result, prompt_dictionary=None):
        self.result = result
        self.prompt_dictionary = prompt_dictionary or {}
        self.calls = []

    def generate(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return self.result


class _Mx(ModuleType):
    float32 = "float32"
    array = staticmethod(np.asarray)
    concatenate = staticmethod(np.concatenate)
    metal = SimpleNamespace(is_available=lambda: True)

    @staticmethod
    def zeros(size, dtype=None):
        return np.zeros(size, dtype=np.float32)


@pytest.fixture
def fake_mlx_audio(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Install stand-in MLX modules and report Apple Silicon. Returns `load`."""
    load = Mock()
    mx = _Mx("mlx.core")
    mlx = ModuleType("mlx")
    mlx.core = mx
    stt = ModuleType("mlx_audio.stt")
    stt.load = load
    utils = ModuleType("mlx_audio.stt.utils")
    utils.SAMPLE_RATE = SAMPLE_RATE
    utils.load_audio = lambda path: np.ones(SAMPLE_RATE, dtype=np.float32)
    for name, module in {
        "mlx": mlx,
        "mlx.core": mx,
        "mlx_audio": ModuleType("mlx_audio"),
        "mlx_audio.stt": stt,
        "mlx_audio.stt.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(asr_mlx_audio, "is_apple_silicon", lambda: True)
    return load


def _options(family: str = "whisper", **kwargs) -> InlineAsrMlxAudioOptions:
    defaults = {"repo_id": "org/model", "revision": SHA, "model_family": family}
    return InlineAsrMlxAudioOptions(**{**defaults, **kwargs})


def _speech_with_pauses(
    pauses: list[tuple[float, float]], total_s: float
) -> np.ndarray:
    waveform = np.full(int(total_s * SAMPLE_RATE), 0.2, dtype=np.float32)
    for start, end in pauses:
        waveform[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = 0.0
    return waveform


# ---------------------------------------------------------------- framework values


def test_mlx_alias_keeps_the_mlx_whisper_backend_and_its_saved_value():
    preset = WHISPER_ASR_PRESETS["tiny"]
    alias = preset.to_options(InferenceAsrFramework.MLX)
    explicit = preset.to_options(InferenceAsrFramework.MLX_WHISPER)

    assert isinstance(alias, InlineAsrMlxWhisperOptions)
    assert isinstance(explicit, InlineAsrMlxWhisperOptions)
    assert alias.repo_id == explicit.repo_id == "mlx-community/whisper-tiny-mlx"
    assert alias.model_dump(mode="json")["inference_framework"] == "mlx"
    assert explicit.model_dump(mode="json")["inference_framework"] == "mlx_whisper"
    assert WHISPER_TINY_MLX.inference_framework == InferenceAsrFramework.MLX
    assert WHISPER_TINY_MLX.repo_id == alias.repo_id


def test_mlx_alias_on_a_model_without_mlx_whisper_names_the_supported_engine():
    with pytest.raises(ValueError, match="mlx_audio"):
        GRANITE_SPEECH_5_0_2B_PRESET.to_options(InferenceAsrFramework.MLX)


@pytest.mark.parametrize(
    "model",
    [model for model in AsrModelType if model.value.endswith("_mlx_audio")],
)
def test_mlx_audio_presets_resolve_and_pin_a_commit(model: AsrModelType):
    options = _resolve_asr_options(model)

    assert isinstance(options, InlineAsrMlxAudioOptions)
    assert len(options.revision) == 40


@pytest.mark.parametrize(
    "model",
    [model for model in AsrModelType if model.value.endswith("_mlx_whisper")],
)
def test_mlx_whisper_presets_resolve(model: AsrModelType):
    options = _resolve_asr_options(model)

    assert isinstance(options, InlineAsrMlxWhisperOptions)
    assert options.inference_framework == InferenceAsrFramework.MLX_WHISPER


@pytest.mark.parametrize("family", ["granite", "granite_ctc"])
def test_chunk_search_window_must_be_smaller_than_the_target(family):
    with pytest.raises(ValueError, match="chunk_search_window"):
        _options(family, max_time_chunk=10.0, chunk_search_window=10.0)
    with pytest.raises(ValueError, match="chunk_search_window"):
        InlineAsrTransformersOptions(
            repo_id="r", model_family=family, max_time_chunk=10.0
        )


@pytest.mark.parametrize("family", ["whisper", "parakeet", "nemotron"])
def test_low_max_time_chunk_stays_valid_for_a_family_without_windows(family):
    # These families do not use windows. Saved options with this value must still load.
    assert _options(family, max_time_chunk=5.0).max_time_chunk == 5.0
    transformers = InlineAsrTransformersOptions(
        repo_id="r", model_family=family, max_time_chunk=5.0
    )
    assert transformers.max_time_chunk == 5.0


def test_max_new_tokens_reaches_the_transformers_fallback():
    options = _options("granite").model_copy(
        update={
            "max_new_tokens": 2048,
            "transformers_fallback": InlineAsrTransformersOptions(
                repo_id="r", model_family="granite"
            ),
        }
    )

    fallback = options.transformers_fallback_options()

    assert fallback is not None
    assert fallback.max_new_tokens == 2048


# ---------------------------------------------------------------- windows


def test_cut_lands_in_the_longest_pause_within_the_search_range():
    waveform = _speech_with_pauses([(24.0, 24.3), (33.0, 34.0), (45.0, 47.0)], 60.0)

    windows = split_on_pauses(waveform, SAMPLE_RATE, 30.0, 10.0)

    # The pause at 45 s is the longest in the file, but it is out of range.
    assert windows[0][1] == pytest.approx(33.5, abs=0.05)


def test_no_pause_cuts_at_the_target():
    waveform = np.full(90 * SAMPLE_RATE, 0.2, dtype=np.float32)

    windows = split_on_pauses(waveform, SAMPLE_RATE, 30.0, 10.0)

    assert windows[0][1] == pytest.approx(30.0)


def test_pause_shorter_than_the_minimum_is_not_a_cut_point():
    waveform = _speech_with_pauses([(25.0, 25.1)], 80.0)

    windows = split_on_pauses(waveform, SAMPLE_RATE, 30.0, 10.0)

    assert windows[0][1] == pytest.approx(30.0)


def test_low_level_recording_still_finds_its_pauses():
    waveform = _speech_with_pauses([(33.0, 34.0)], 60.0) * 0.001

    windows = split_on_pauses(waveform, SAMPLE_RATE, 30.0, 10.0)

    assert windows[0][1] == pytest.approx(33.5, abs=0.05)


def test_windows_cover_the_recording_without_gaps_and_keep_the_size_limits():
    pauses = [(float(s), s + 0.5) for s in range(7, 600, 13)]
    waveform = _speech_with_pauses(pauses, 600.0)

    windows = split_on_pauses(waveform, SAMPLE_RATE, 30.0, 10.0)

    assert windows[0][0] == 0.0
    assert windows[-1][1] == pytest.approx(600.0)
    for (_, first_end), (second_start, _) in itertools.pairwise(windows):
        assert first_end == second_start
    for start, end in windows[:-1]:
        assert 20.0 <= end - start <= 40.0
    assert windows[-1][1] - windows[-1][0] <= 40.0


def test_pause_near_the_end_does_not_leave_a_fragment():
    waveform = _speech_with_pauses([(39.5, 40.0)], 40.2)

    windows = split_on_pauses(waveform, SAMPLE_RATE, 30.0, 10.0)

    assert windows[-1][1] - windows[-1][0] >= 10.0


def test_short_recording_has_one_window():
    waveform = np.full(40 * SAMPLE_RATE, 0.2, dtype=np.float32)

    assert split_on_pauses(waveform, SAMPLE_RATE, 30.0, 10.0) == [(0.0, 40.0)]


def test_empty_recording_has_no_windows():
    assert split_on_pauses(np.zeros(0), SAMPLE_RATE, 30.0, 10.0) == []


# ---------------------------------------------------------------- output conversion


def test_whisper_segments_keep_times_and_words(fake_mlx_audio: Mock):
    fake_mlx_audio.return_value = _Model(
        SimpleNamespace(
            segments=[
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": " hello ",
                    "words": [{"word": " hello", "start": 1.0, "end": 2.0}],
                }
            ]
        )
    )
    model = _MlxAudioModel(True, None, AcceleratorOptions(), _options(language="en"))

    items = model.transcribe(Path("a.wav"))

    assert [(i.text, i.start_time, i.end_time) for i in items] == [("hello", 1.0, 2.0)]
    assert items[0].words is not None
    assert items[0].words[0].text == " hello"
    assert fake_mlx_audio.return_value.calls[0][1]["language"] == "en"


def test_parakeet_sentences_become_items(fake_mlx_audio: Mock):
    sentence = SimpleNamespace(text=" one two ", start=0.5, end=1.5)
    fake_mlx_audio.return_value = _Model(SimpleNamespace(sentences=[sentence]))
    model = _MlxAudioModel(True, None, AcceleratorOptions(), _options("parakeet"))

    items = model.transcribe(Path("a.wav"))

    assert [(i.text, i.start_time, i.end_time) for i in items] == [
        ("one two", 0.5, 1.5)
    ]


def test_nemotron_adds_silence_so_the_decoder_emits_the_last_word(
    fake_mlx_audio: Mock,
):
    fake_mlx_audio.return_value = _Model(SimpleNamespace(sentences=[]))
    model = _MlxAudioModel(True, None, AcceleratorOptions(), _options("nemotron"))

    model.transcribe(Path("a.wav"))

    audio = fake_mlx_audio.return_value.calls[0][0]
    assert audio.size == 3 * SAMPLE_RATE
    assert audio[-1] == 0.0


def test_nemotron_timestamps_stay_inside_the_real_audio(fake_mlx_audio: Mock):
    # The stand-in recording lasts 1 s; the decoder also sees the added silence.
    sentence = SimpleNamespace(text="last words", start=0.5, end=2.5)
    fake_mlx_audio.return_value = _Model(SimpleNamespace(sentences=[sentence]))
    model = _MlxAudioModel(True, None, AcceleratorOptions(), _options("nemotron"))

    items = model.transcribe(Path("a.wav"))

    assert (items[0].start_time, items[0].end_time) == (0.5, 1.0)


@pytest.mark.parametrize(
    ("language", "keys", "expected"),
    [
        ("en-US", ["en-US", "auto"], "en-US"),
        ("en", ["en-US", "auto"], "en-US"),
        ("auto", ["en-US", "auto"], "auto"),
        ("en", [], "en"),
    ],
)
def test_nemotron_sends_the_prompt_key_of_the_language(
    fake_mlx_audio: Mock, language, keys, expected
):
    nemotron = _Model(
        SimpleNamespace(sentences=[]), prompt_dictionary=dict.fromkeys(keys)
    )
    fake_mlx_audio.return_value = nemotron
    model = _MlxAudioModel(
        True, None, AcceleratorOptions(), _options("nemotron", language=language)
    )

    model.transcribe(Path("a.wav"))

    assert nemotron.calls[0][1]["language"] == expected


@pytest.mark.parametrize("keys", [["en-US"], []])
def test_nemotron_rejects_a_language_that_it_would_silently_ignore(
    fake_mlx_audio: Mock, keys
):
    nemotron = _Model(
        SimpleNamespace(sentences=[]), prompt_dictionary=dict.fromkeys(keys)
    )
    fake_mlx_audio.return_value = nemotron
    model = _MlxAudioModel(
        True, None, AcceleratorOptions(), _options("nemotron", language="fr")
    )

    with pytest.raises(ValueError, match="fr"):
        model.transcribe(Path("a.wav"))

    assert nemotron.calls == []


def _long_recording(monkeypatch: pytest.MonkeyPatch, seconds: int) -> None:
    waveform = np.full(seconds * SAMPLE_RATE, 0.2, dtype=np.float32)
    monkeypatch.setitem(
        sys.modules["mlx_audio.stt.utils"].__dict__, "load_audio", lambda path: waveform
    )


def test_granite_windows_give_timed_items_and_skip_empty_text(
    fake_mlx_audio: Mock, monkeypatch: pytest.MonkeyPatch
):
    _long_recording(monkeypatch, 90)
    granite = _Model(SimpleNamespace(text=" spoken ", segments=[]))
    fake_mlx_audio.return_value = granite
    model = _MlxAudioModel(True, None, AcceleratorOptions(), _options("granite_ctc"))

    items = model.transcribe(Path("a.wav"))

    assert [(i.start_time, i.end_time) for i in items] == [
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 90.0),
    ]
    assert {i.text for i in items} == {"spoken"}
    assert sum(audio.size for audio, _ in granite.calls) == 90 * SAMPLE_RATE

    granite.result = SimpleNamespace(text="  ", segments=[])
    assert model.transcribe(Path("a.wav")) == []


def test_granite_source_language_is_not_sent_as_a_translation_target(
    fake_mlx_audio: Mock,
):
    # In mlx-audio, Granite `language` translates the speech into that language.
    granite = _Model(SimpleNamespace(text="x", segments=[]))
    fake_mlx_audio.return_value = granite
    model = _MlxAudioModel(
        True, None, AcceleratorOptions(), _options("granite", language="fr")
    )

    model.transcribe(Path("a.wav"))

    kwargs = granite.calls[0][1]
    assert kwargs["source_language"] == "fr"
    assert "language" not in kwargs


def test_granite_translate_task_targets_english(fake_mlx_audio: Mock):
    granite = _Model(SimpleNamespace(text="x", segments=[]))
    fake_mlx_audio.return_value = granite
    model = _MlxAudioModel(
        True,
        None,
        AcceleratorOptions(),
        _options("granite", language="fr", task="translate"),
    )

    model.transcribe(Path("a.wav"))

    kwargs = granite.calls[0][1]
    assert kwargs["language"] == "en"
    assert "source_language" not in kwargs


@pytest.mark.parametrize(
    "kwargs",
    [{"task": "translate"}, {"prompt": "guide"}, {"language": "fr"}],
)
@pytest.mark.parametrize("family", ["parakeet", "granite_ctc"])
def test_unsupported_options_fail_early(fake_mlx_audio: Mock, family, kwargs):
    with pytest.raises(ValueError):
        _MlxAudioModel(True, None, AcceleratorOptions(), _options(family, **kwargs))
    fake_mlx_audio.assert_not_called()


# ---------------------------------------------------------------- loading


def test_online_load_pins_the_revision(fake_mlx_audio: Mock):
    _MlxAudioModel(True, None, AcceleratorOptions(), _options())

    fake_mlx_audio.assert_called_once_with("org/model", revision=SHA)


def test_artifacts_path_loads_the_local_checkpoint(
    fake_mlx_audio: Mock, tmp_path: Path
):
    checkpoint = tmp_path / "org--model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")

    _MlxAudioModel(True, tmp_path, AcceleratorOptions(), _options())

    fake_mlx_audio.assert_called_once_with(checkpoint)


def test_missing_local_checkpoint_does_not_fall_back_to_the_hub(
    fake_mlx_audio: Mock, tmp_path: Path
):
    with pytest.raises(FileNotFoundError, match="org/model"):
        _MlxAudioModel(True, tmp_path, AcceleratorOptions(), _options())

    fake_mlx_audio.assert_not_called()


def test_unpinned_granite_needs_an_explicit_opt_in(fake_mlx_audio: Mock):
    unpinned = _options("granite", revision="main")

    with pytest.raises(ValueError, match="remote code"):
        _MlxAudioModel(True, None, AcceleratorOptions(), unpinned)
    fake_mlx_audio.assert_not_called()

    allowed = _options("granite", revision="main", trust_remote_code=True)
    _MlxAudioModel(True, None, AcceleratorOptions(), allowed)
    fake_mlx_audio.assert_called_once()


def test_unpinned_granite_follows_the_refuse_setting(
    fake_mlx_audio: Mock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DOCLING_SECURITY_REFUSE_UNPINNED_REMOTE_CODE", "1")
    allowed = _options("granite", revision="main", trust_remote_code=True)

    with pytest.raises(ValueError, match="SECURITY"):
        _MlxAudioModel(True, None, AcceleratorOptions(), allowed)


def test_unpinned_parakeet_does_not_need_an_opt_in(fake_mlx_audio: Mock):
    _MlxAudioModel(
        True, None, AcceleratorOptions(), _options("parakeet", revision="main")
    )

    fake_mlx_audio.assert_called_once()


def test_other_platforms_get_a_clear_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(asr_mlx_audio, "is_apple_silicon", lambda: False)

    with pytest.raises(RuntimeError, match="Apple Silicon"):
        MlxAudioAsrDecoder(_options(), None)


def test_missing_library_error_names_the_extra(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(asr_mlx_audio, "is_apple_silicon", lambda: True)
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)

    with pytest.raises(ImportError, match="feat-asr-mlx-audio"):
        MlxAudioAsrDecoder(_options(), None)


def test_factory_builds_the_backend_without_torch() -> None:
    # `decide_device()` imports torch, so the backend must not call it.
    code = """
import sys
sys.modules["torch"] = None
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.asr_model_specs import WHISPER_TINY_MLX_AUDIO
from docling.pipeline import asr_mlx_audio
from docling.pipeline.asr_transcriber import _AsrModelFactory

asr_mlx_audio.is_apple_silicon = lambda: True
core = ModuleType("mlx.core")
core.metal = SimpleNamespace(is_available=lambda: True)
stt = ModuleType("mlx_audio.stt")
stt.load = Mock()
utils = ModuleType("mlx_audio.stt.utils")
utils.SAMPLE_RATE = 16000
utils.load_audio = Mock()
sys.modules.update({
    "mlx": ModuleType("mlx"),
    "mlx.core": core,
    "mlx_audio": ModuleType("mlx_audio"),
    "mlx_audio.stt": stt,
    "mlx_audio.stt.utils": utils,
})
sys.modules["mlx"].core = core
_AsrModelFactory.create(WHISPER_TINY_MLX_AUDIO, None, AcceleratorOptions())
stt.load.assert_called_once()
print("BUILT")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=180,
    )
    assert "BUILT" in result.stdout, result.stderr


def test_explicit_preset_without_metal_is_an_error(
    fake_mlx_audio: Mock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(_Mx, "metal", SimpleNamespace(is_available=lambda: False))

    with pytest.raises(RuntimeError, match="Metal"):
        _AsrModelFactory.create(PARAKEET_CTC_1_1B_MLX_AUDIO, None, AcceleratorOptions())

    fake_mlx_audio.assert_not_called()


# ---------------------------------------------------------------- engine selection


def _machine(
    monkeypatch: pytest.MonkeyPatch, apple_silicon: bool, mlx_audio: str | None
) -> None:
    def installed_version(package: str) -> str:
        assert package == "mlx-audio"
        if mlx_audio is None:
            raise importlib.metadata.PackageNotFoundError(package)
        return mlx_audio

    monkeypatch.setattr(asr_presets, "is_apple_silicon", lambda: apple_silicon)
    monkeypatch.setattr(importlib.metadata, "version", installed_version)


@pytest.mark.parametrize(
    ("apple_silicon", "mlx_audio", "device", "expected"),
    [
        (True, "0.6.4", AcceleratorDevice.AUTO, InferenceAsrFramework.MLX_AUDIO),
        (True, "0.6.4", AcceleratorDevice.MPS, InferenceAsrFramework.MLX_AUDIO),
        # AcceleratorOptions also accepts plain strings.
        (True, "0.6.4", "auto", InferenceAsrFramework.MLX_AUDIO),
        (True, "0.6.4", "cuda:0", InferenceAsrFramework.TRANSFORMERS),
        # The user asked for a device that MLX cannot use.
        (True, "0.6.4", AcceleratorDevice.CPU, InferenceAsrFramework.TRANSFORMERS),
        # mlx-vlm can install a version that this backend cannot use.
        (True, "0.6.1", AcceleratorDevice.AUTO, InferenceAsrFramework.TRANSFORMERS),
        (True, None, AcceleratorDevice.AUTO, InferenceAsrFramework.TRANSFORMERS),
        (False, "0.6.4", AcceleratorDevice.AUTO, InferenceAsrFramework.TRANSFORMERS),
    ],
)
def test_preset_selects_the_engine_for_the_machine(
    monkeypatch: pytest.MonkeyPatch, apple_silicon, mlx_audio, device, expected
):
    _machine(monkeypatch, apple_silicon, mlx_audio)

    options = PARAKEET_CTC_1_1B_PRESET.to_auto_options(device)

    assert options.inference_framework == expected


def test_model_without_an_mlx_audio_export_stays_on_transformers(
    monkeypatch: pytest.MonkeyPatch,
):
    _machine(monkeypatch, True, "0.6.4")

    options = GRANITE_SPEECH_4_1_2B_PRESET.to_auto_options()

    assert options.inference_framework == InferenceAsrFramework.TRANSFORMERS


def test_cli_name_selects_by_machine_and_transformers_name_does_not(
    monkeypatch: pytest.MonkeyPatch,
):
    _machine(monkeypatch, True, "0.6.4")

    auto = _resolve_asr_options(AsrModelType.GRANITE_SPEECH_5_0_470M_TURBOCTC)
    on_cpu = _resolve_asr_options(
        AsrModelType.GRANITE_SPEECH_5_0_470M_TURBOCTC, AcceleratorDevice.CPU
    )
    forced = _resolve_asr_options(
        AsrModelType.GRANITE_SPEECH_5_0_470M_TURBOCTC_TRANSFORMERS
    )
    streaming = _resolve_asr_options(AsrModelType.NEMOTRON_3_5_ASR_STREAMING)

    assert auto.inference_framework == InferenceAsrFramework.MLX_AUDIO
    assert on_cpu.inference_framework == InferenceAsrFramework.TRANSFORMERS
    assert forced.inference_framework == InferenceAsrFramework.TRANSFORMERS
    assert streaming.inference_framework == InferenceAsrFramework.MLX_AUDIO


# ---------------------------------------------------------------- real inference


@pytest.mark.skipif(not RUNS_MLX, reason="needs mlx-audio on Apple Silicon")
@pytest.mark.parametrize(
    "model",
    [
        AsrModelType.WHISPER_TINY_MLX_AUDIO,
        AsrModelType.GRANITE_SPEECH_5_0_470M_TURBOCTC_MLX_AUDIO,
    ],
)
def test_real_transcription(model: AsrModelType):
    transcriber = _AsrModelFactory.create(
        _resolve_asr_options(model), None, AcceleratorOptions()
    )

    items = transcriber.transcribe(SPEECH_SAMPLE)

    text = " ".join(item.text for item in items).lower()
    assert "shakespeare" in text
    assert all(item.end_time >= item.start_time for item in items)
