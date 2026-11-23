# MLX Audio ASR backend

**Status:** Phase 0 and Phase 1 implemented on `feat/support-mlx-audio`, not committed.
Phases 2 to 4 are open.

## Implementation notes (Phases 0 and 1)

- **Exit gate result:** passed in a clean venv with only `docling-slim[feat-asr-mlx-audio]`, no
  lock file, and no torch. All six presets convert MP3 and M4A through `AsrPipeline`. BytesIO,
  language options, Granite translation, and `VideoPipeline` also work. WAV and MP3 work without
  FFmpeg, and M4A then gives an `ErrorItem`. `import docling` prints no warning.
- **Presets in Phase 1:** `whisper_tiny`, Granite 5.0 2B, Granite TurboCTC, both Nemotron models,
  and `parakeet_ctc_1_1b`. `granite_speech_5_0_2b_mlx_audio` still needs the long-audio gate.
- **Differences from the design below:**
  - The decoder is in `docling/pipeline/asr_mlx_audio.py`, with a thin `_MlxAudioModel` in
    `asr_transcriber.py`. This follows `asr_transformers.py` and limits growth of that file.
  - Granite source language goes to `source_language`. In `mlx-audio`, Granite `language` is the
    translation target. `task="translate"` sends `language="en"`.
  - Granite gets `max_tokens=1024` for each window.
  - Nemotron rejects an unknown language, because the model silently ignores it. Added silence
    is removed from the timestamps.
  - All `_LocalAudioTranscriber` backends now report a transcription failure as an `ErrorItem`.
    Before, `_MlxWhisperModel` and `_TransformersAsrModel` only set the status.
  - `custom_stopping_criteria` validates through a function that finds `StoppingCriteria` in
    `sys.modules`, so the module does not import `transformers`.
- **Still open on the import path:** `_detect_hardware_and_libraries()` imports torch when it is
  installed, during `asr_model_specs` import. It is safe on a torch-free install. Phase 4 replaces it.
- **`models-asr-inline` (maintainer request, for symmetry with `models-vlm-inline`):** a new
  aggregate of `format-audio`, `feat-asr-transformers`, and `feat-asr-mlx-audio`. Those three stay
  the same. `docling[asr-inline]` re-exports it and replaces the unreleased `docling[asr-mlx-audio]`.
  `all` now uses it. This changes D5: `all` also gains `feat-asr-transformers`. `format-audio` and
  `docling[asr]` stay narrow, because adding `mlx-audio` would break macOS 13.
- **Auto-selection moved forward from Phase 4 (maintainer request):** the preset names without a
  suffix now use `AsrModelPreset.to_auto_options()`. It selects `mlx-audio` on Apple Silicon if the
  installed version is at least 0.6.4 (`mlx-vlm` allows older versions), the model has an export,
  and the device is `auto` or `mps`. Otherwise it selects Transformers. It reads package metadata
  and does not import `mlx`. `to_options()` keeps its meaning. New `*_TRANSFORMERS` presets force
  the old path. The bare `WHISPER_*` presets stay on `mlx-whisper` or native Whisper, and the
  comparison for the `mlx` alias target stays in Phase 4.
- **Transformers Granite windows (maintainer request):** `TransformersAsrDecoder` also uses
  `split_on_pauses()`, so both engines give the same structure. `chunk_search_window` moved to the
  shared `_WindowedAsrOptions`. `max_new_tokens` is now 1024 on `InlineAsrTransformersOptions`.
  The old 256 limit applied to the whole recording: Granite 5.0 2B lost about 60% of a 3-minute
  clip with a `success` status.
- **`docling models` (maintainer request, same branch):** lists VLM and ASR models by family, with
  backends as columns. `docling/cli/model_table.py` derives ASR rows from `AsrModelType` and its
  resolved backend, so a new preset cannot be missing. `--vlm-model` and `--asr-model` now show
  `NAME` and point to this command.
- **Moved forward:** behavior tests are in `tests/test_asr_mlx_audio.py` and the slim guards are in
  `tests/test_asr_import_isolation.py`. Phase 3 still has the documentation and the macOS CI job.

## Objective

Run the ASR models that Docling supports on Apple Silicon through
[`mlx-audio`](https://github.com/Neywa-Labs/mlx-audio). Use the same integration
pattern as `mlx-vlm` for VLM models. The backend must work on a torch-free
`docling-slim` install, and it must not change the import path of any other install.

Today, MLX ASR means Whisper only (`mlx-whisper`). Granite Speech, Parakeet, and
Nemotron run through Transformers on Apple Silicon.

## Conclusion

Yes, the integration is possible. The tests ran the Whisper, Granite, Parakeet, and
Nemotron families through `mlx-audio` 0.6.4 on Docling's sample audio.

Six results set the design:

1. **`mlx-audio` cannot directly replace `mlx-whisper`.** It fails on the Whisper
   checkpoints that Docling uses now. Add a backend, and keep the meaning of `mlx`.
2. **The output shape depends on the model family.** A small adapter for each family
   is necessary.
3. **Windows remove the Granite repetition loop.** Pause-aware windows are possible
   with no new dependency. They do not guarantee a sentence boundary (see D3).
4. **Granite loads its tokenizer with `trust_remote_code=True`, and the caller cannot
   change this.** MLX presets must pin a commit SHA.
5. **`mlx-audio` works without PyTorch, but parts of the current Docling ASR code do
   not.** `decide_device()` and the preset selector need torch. The new backend must
   avoid them (see the docling-slim section).
6. **A torch-free install that has `transformers` shows an existing eager import.**
   `import docling` then prints a PyTorch warning. Phase 0 corrects this.

## Confirmed decisions

| Question | Decision |
|---|---|
| Backend names | Three framework values: `mlx_audio`, `mlx_whisper`, and `mlx`. For now, `mlx` points to `mlx-whisper`, so backward compatibility stays intact |
| Granite timing | One item for each window of about 30 s, with a ±10 s search so that a cut does not fall in mid-sentence |
| Public preset name | `*_mlx_audio` |
| docling-slim | Must get particular care. Apply the lessons of PR 5072 and PR 5090 |

## docling-slim requirements

This section has priority over the rest of the plan.

### Lessons from the two pull requests

| Pull request | Fault | Rule for this work |
|---|---|---|
| [#5072](https://github.com/docling-project/docling/pull/5072) | `asr_transformers.py` imported torch at module level. `document_converter.py` reaches it through the ASR pipeline, so all torch-free installs failed on `from docling.document_converter import DocumentConverter` | An optional dependency must not reach the import path. Import it inside the function that needs it, and use `TYPE_CHECKING` for type hints. The `test-slim-import` CI job checks this on Linux with no extras |
| [#5090](https://github.com/docling-project/docling/pull/5090) | Importing ASR presets imported `mlx_whisper`, which initializes Metal. This can abort the process in restricted or headless environments, even for `docling --version` | Detect with `importlib.util.find_spec()`. Only the model constructor can import an MLX library. A `try/except ImportError` does not help, because the abort is native |

### Rules

- **S1: No optional import on the import path.** `asr_transcriber.py`, `asr_pipeline.py`,
  `video_pipeline.py`, `asr_presets.py`, `asr_model_specs.py`, and
  `pipeline_options_asr_model.py` must not import these modules at module level: `mlx`,
  `mlx_audio`, `mlx_whisper`, `torch`, `transformers`, `numpy`, or `scipy`.
- **S2: Only the model constructor initializes MLX.** Presets, `to_options()`, the factory
  import, and CLI startup must not load `mlx.core`. Do not copy the `import mlx_vlm` in
  `AutoInlineVlmEngine._select_engine()`. That call runs later, but it still initializes Metal.
- **S3: The backend does not need torch.** Do not call `decide_device()`, because it starts
  with `import torch`. This corrects the first version of this plan. `MlxVlmEngine` also does
  not call it. Check the platform with `platform.system()` and `platform.machine()`, then
  check `mx.metal.is_available()` inside the constructor.
- **S4: The extra stays torch-free.** See D5. Do not depend on `format-audio`.
- **S5: A missing dependency gives a clear error.** The `ImportError` names
  `docling-slim[feat-asr-mlx-audio]`. On other platforms, the error says that the backend
  needs Apple Silicon.
- **S6: FFmpeg is not always necessary.** `mlx-audio` decodes WAV, MP3, and FLAC with
  `miniaudio`. It uses FFmpeg only for formats such as M4A and Ogg. The backend must not
  apply the FFmpeg check of `_MlxWhisperModel.run()` to all files. Report a decode failure as an
  `ErrorItem`.
- **S7: Tests must not break the rule.** Do not use a module-level
  `pytest.importorskip("mlx_audio")`, because collection then initializes Metal. This corrects
  the first version of this plan. Guard with `find_spec` and import inside the test.

### Measured torch-free results

Environment: a clean venv with Docling from this repository (no extras) and `mlx-audio==0.6.4`.
`find_spec("torch")` returned `None`.

| Check | Result |
|---|---|
| Import `DocumentConverter`, `asr_model_specs`, `asr_pipeline`, `asr_transcriber` | Works. `mlx`, `mlx_audio`, `mlx_whisper`, and `torch` are not loaded |
| `mlx_audio.stt` with `whisper-tiny-asr-fp16`, Granite TurboCTC, Granite 5.0 2B | Load and generate work without torch |
| `WHISPER_TINY` auto-selection | Selects `InlineAsrNativeWhisperOptions`, because `_detect_hardware_and_libraries()` finds MPS through torch. Native Whisper needs torch, so the CLI default cannot work here |
| `decide_device("auto", ...)` | **Fails:** `ModuleNotFoundError: No module named 'torch'` |
| `import docling` | **Prints a warning:** `[transformers] PyTorch was not found...` |

Not tested without torch: Parakeet and Nemotron (their `mlx-audio` modules contain no torch
import, but this is a source reading), and a full `AsrPipeline` conversion, because the backend
does not exist yet. The Phase 1 gate covers both.

### The existing `transformers` import

`pipeline_options_vlm_model.py:12` runs `from transformers import StoppingCriteria` in a
`try/except ImportError`. It is safe when `transformers` is absent. When `transformers` is
present:

- Without torch: each `import docling` prints the warning above.
- With torch: `transformers` loads torch through `generation/stopping_criteria.py`. This is
  the import cost that PR 5090 removed for MLX.

The new extra makes the first case common. Phase 0 moves the import behind `TYPE_CHECKING` and
gives the field a runtime type that does not need `transformers`. Check the Pydantic validation
of `custom_stopping_criteria` with a test.

### Dependency resolution

| Requirements | macOS 13 arm64 | macOS 14 arm64 |
|---|---|---|
| `mlx-vlm>=0.7,<1` and `mlx-whisper` (today's `all`) | Already fails | Resolves |
| The same, with `mlx-audio>=0.6.4,<0.7` | Already fails | Resolves |
| `mlx-audio>=0.6.4,<0.7` and `transformers>=5.16,<6` (new extra) | Fails: no `mlx` 0.32.4 wheel | Resolves, **with no torch** |

Result: adding the extra to `all` is not a new failure. Adding `mlx-audio` to `format-audio`
would be a new failure, because `mlx-whisper` alone still resolves on macOS 13. Document the
macOS 14 minimum.

## Current design

### mlx-vlm pattern

| Concern | Implementation |
|---|---|
| Dependency | `mlx-vlm` in `models-vlm-inline`, with a macOS arm64 marker |
| Import | Lazy import in `MlxVlmEngine.initialize()`, with a clear `ImportError` |
| Engine type | `VlmEngineType.MLX`, `MlxVlmEngineOptions`, `create_vlm_engine()` |
| Model mapping | `VlmModelSpec.engine_overrides` and `supported_engines` |
| Auto-selection | `AutoInlineVlmEngine`: macOS, MPS, explicit MLX export, library present, and sufficient version. Otherwise Transformers |
| Version gate | `EngineModelConfig.min_engine_version`, `check_min_engine_version()` |
| Checkpoint | Downloads a pinned revision, or uses `artifacts_path/<org--repo>` with a clear error |
| Thread safety | `_MLX_GLOBAL_LOCK` around all generation |

### ASR today

| Concern | Implementation | Gap |
|---|---|---|
| Dependency | `mlx-whisper` in `format-audio` | Whisper only; brings torch |
| Import | Lazy import in `_MlxWhisperModel.__init__` | None |
| Engine type | `InferenceAsrFramework.MLX`, `InlineAsrMlxWhisperOptions` | `MLX` means `mlx-whisper` |
| Model mapping | `AsrModelSpec` and `AsrModelPreset` already copy the VLM pattern | Only Whisper declares an MLX override; `to_options()` always creates Whisper options for `MLX` |
| Auto-selection | `_get_whisper_*_model()` uses `find_spec` (PR 5090) | Whisper only; MPS detection needs torch |
| Version gate | None | Package floor only |
| Checkpoint | `_resolve_asr_artifacts()` requires `config.json` | No revision pin on `InlineAsrMlxWhisperOptions` |
| Thread safety | No lock | Can overlap with MLX VLM work |

The factory is `_AsrModelFactory.create()` in `docling/pipeline/asr_transcriber.py`.
`AsrPipeline` and `VideoPipeline` both use it, so a new backend reaches both pipelines.

## mlx-audio facts

Source: `Neywa-Labs/mlx-audio` at `main`, version 0.6.4, read on 2026-11-04.

- **Project:** PyPI name `mlx-audio`, MIT license, Python 3.10 or later.
- **Repository identity:** `Blaizzy/mlx-audio` redirects to GitHub repository ID 895253710,
  which is `Neywa-Labs/mlx-audio`. It is the same project and not a fork. PyPI metadata still
  shows the old URL.
- **Existing dependency:** `mlx-vlm` 0.7.11 requires `mlx-audio>=0.6.2`. `uv.lock` already has
  `mlx-audio` 0.6.1 through `mlx-vlm` 0.7.6.
- **Exact MLX pin:** 0.6.1 to 0.6.4 all require `mlx==0.32.4`. `mlx-vlm` requires `mlx>=0.32.4`.
  A new MLX release can block resolution until `mlx-audio` follows.
- **Transformers:** a core dependency, `>=5.14.0`. No PyTorch dependency.
- **Audio decoding:** `miniaudio` for WAV, MP3, and FLAC. FFmpeg for M4A, AAC, and Ogg.
- **API:** `from mlx_audio.stt import load`, then `model.generate(audio, **kwargs)`.
  `load(path_or_repo, revision=...)` accepts a local `Path`. `generate()` also accepts an
  `mx.array` waveform.
- **Side effect:** importing `mlx_audio` sets `TRANSFORMERS_NO_ADVISORY_WARNINGS=1` for the whole
  process.

### Output by model family

| Family | `generate()` returns | Timestamps | Language argument |
|---|---|---|---|
| Whisper | `STTOutput`, `segments` with `start`, `end`, `text`, `words` | Segment and word | `language` (ISO code) |
| Parakeet | `AlignedResult`, `sentences` with `start`, `end`, `text`, `tokens` | Sentence and token | None |
| Nemotron | `AlignedResult` | Sentence and token | `language` is a prompt key (`en-US`, `auto`) |
| Granite 4.x, 5.0 2B | `STTOutput`, `segments=[]` for plain ASR | None | `source_language` (verify per version) |
| Granite TurboCTC | `STTOutput`, `segments=[]` | None | English only |

`_process_conversation()` reads only `start_time`, `end_time`, `text`, and `speaker`. It does not
store `words`, and `video_pipeline.py` and `asr_pipeline.py` do not read them. Do not write
token-to-word merge code in this work. Nemotron tokens can be single characters, so this merge is
not trivial.

## Test results

Run on one arm64 Mac (macOS 26.6, 64 GB RAM, Python 3.12), in isolated environments:
`mlx-audio` 0.6.4, `mlx-whisper` 0.4.3, `mlx` 0.32.4.

Audio: `tests/data/audio/sources/sample_10s.mp3`, `sample_10s_audio-m4a.m4a`, and a 5-minute WAV
that repeats the MP3 30 times. The spoken text ends with "...in the public domain".

### One call for the whole file

| Checkpoint | Docling preset | 10 s MP3 and M4A | 5-minute file | Time for 5 min | Peak MLX memory |
|---|---|---|---|---|---|
| `mlx-community/whisper-tiny-mlx` with `mlx-whisper` (baseline) | `whisper_tiny_mlx` | Correct words for this model size | 40 segments, ends 299.76 s | 1.8 s | Not measured |
| `mlx-community/whisper-tiny-mlx` with `mlx-audio` | `whisper_tiny_mlx` | **Fails:** `Processor not found` | Not run | - | - |
| `mlx-community/whisper-tiny-asr-fp16` | New | Same text as baseline | 40 segments, ends 299.76 s | 3.0 s | 0.5 GB |
| `ibm-granite/granite-speech-5.0-470m-turboctc` | `granite_speech_5_0_470m_turboctc` | Good; M4A loses "domain" | 3029 characters, garbled joins, no timestamps | 0.4 s | 2.1 GB |
| `ibm-granite/granite-speech-5.0-2b` | `granite_speech_5_0_2b` | Correct | **Repetition loop:** 18582 characters; "librivox" 315 times, expected 60 | 56.7 s | 17.2 GB |
| `mlx-community/parakeet-ctc-1.1b` | `parakeet_ctc_1_1b` | Correct, no punctuation | 3421 characters, one "sentence" | 3.1 s | 9.1 GB |
| `mlx-community/nemotron-3.5-asr-streaming-0.6b` | `nemotron_3_5_asr_streaming` | Cuts the last word ("public do"); 2 s of added silence corrects it | 65 sentences | 17.1 s | 2.1 GB |
| `nvidia/nemotron-speech-streaming-en-0.6b` | `nemotron_speech_streaming_en_0_6b` | Correct | 88 sentences | 8.1 s | 2.5 GB |
| `mlx-community/parakeet-tdt-0.6b-v3` | None | **Unreliable:** drops a sentence on MP3, empty on M4A | 1018 characters; empty with chunking | 2.6 s | 7.5 GB |

### Windows on the 5-minute file

Each model has 10 windows. "Fixed" cuts each 30 s. "Pause" uses the D3 search.

| Model | Mode | Window length | Time | Peak MLX memory | Characters | "librivox" (expected 60) |
|---|---|---|---|---|---|---|
| Granite TurboCTC | No windows | - | 0.4 s | 2.1 GB | 3029 | Not counted |
| Granite TurboCTC | Fixed | 29.8-30.0 s | 0.5 s | 1.5 GB | 3500 | 57 |
| Granite TurboCTC | Pause | 24.8-35.2 s | 0.5 s | 1.6 GB | 3483 | 56 |
| Granite 5.0 2B | No windows | - | 56.7 s | 17.2 GB | 18582 | 315 |
| Granite 5.0 2B | Fixed | 29.8-30.0 s | 11.1 s | 9.0 GB | 2359 | 40 |
| Granite 5.0 2B | Pause | 24.8-35.2 s | 11.1 s | 9.6 GB | 2363 | 40 |

What this shows:

- Windows stop the Granite 5.0 2B loop and reduce time by about 5 times and memory by about half.
- Pause mode cut at the real pauses in this clip, and no window used the fallback.
- **Granite 5.0 2B returns about one third less text than expected in both modes.** The cause
  is not known. The LLM decoder can collapse repeated sentences, and this audio repeats one clip.
  This can be a test artifact or a real text loss. The long-audio gate must answer this with a
  real recording.

### Limits of these tests

- Only one Whisper size ran. The other sizes need their own test.
- Not run: `granite_speech_5_0_470m_turboctc_nc`, `granite_speech_4_1_2b`,
  `granite_speech_4_1_2b_plus`, and `granite_4_0_1b_speech`.
- The long file repeats one clip. It has regular pauses, so it is an easy case for pause search,
  and it does not show sentence quality on natural speech.
- My count of clean sentence endings was not reliable, so this plan does not report it.
- Peak memory is the MLX counter. Memory from a previous model in the same process can affect
  it. Measure again in separate processes before you publish numbers.
- Transformers on MPS was not compared. Speed and accuracy gains are not proven yet.
- I did not find the cause of the Parakeet TDT result. The M4A input has a low peak level (0.24).
  This is a possible factor, not a confirmed cause.

## Design decisions

### D1: Three framework values

```python
class InferenceAsrFramework(str, Enum):
    MLX = "mlx"                  # Alias. Now resolves to MLX_WHISPER.
    MLX_WHISPER = "mlx_whisper"
    MLX_AUDIO = "mlx_audio"
    TRANSFORMERS = "transformers"
    WHISPER = "whisper"
    WHISPER_S2T = "whisper_s2t"
```

Resolve the alias in one place, in `asr_presets.py`:

```python
_MLX_ALIAS_TARGET: Final = InferenceAsrFramework.MLX_WHISPER


def _resolve_framework(framework: InferenceAsrFramework) -> InferenceAsrFramework:
    return _MLX_ALIAS_TARGET if framework == InferenceAsrFramework.MLX else framework
```

| Value | Options class | Transcriber | Checkpoints |
|---|---|---|---|
| `mlx` | `InlineAsrMlxWhisperOptions` | `_MlxWhisperModel` | `whisper-*-mlx` (no change) |
| `mlx_whisper` | `InlineAsrMlxWhisperOptions` | `_MlxWhisperModel` | `whisper-*-mlx` |
| `mlx_audio` | `InlineAsrMlxAudioOptions` | `_MlxAudioModel` | `*-asr-fp16`, Granite, Parakeet, Nemotron |

Compatibility requirements:

- `InlineAsrMlxWhisperOptions` keeps the default `inference_framework=MLX`. Narrow the field to
  `Literal[MLX, MLX_WHISPER]`, so that it cannot hold `MLX_AUDIO`.
- `WHISPER_*`, `WHISPER_*_MLX`, and the serialized value `"mlx"` stay the same.
- `to_options(MLX)` still returns `mlx-community/whisper-tiny-mlx`, as
  `test_whisper_preset_engine_exports` requires. Its `inference_framework` stays `MLX`.
- `AsrModelSpec.get_engine_config()` resolves the alias first. Presets declare `MLX_WHISPER` and
  `MLX_AUDIO`, not `MLX`.
- `to_options(MLX)` on a Granite preset raises `ValueError`, and the message names `mlx_audio`.
- The factory dispatches on the options class, so it needs no alias logic.

Reasons for a separate backend: current `whisper-*-mlx` repositories have only `config.json` and
`weights.npz`, and `mlx-audio` needs a `WhisperProcessor`. Users with `artifacts_path` have those
checkpoints.

### D2: Dispatch the adapter on `model_family`

Do not use `hasattr` to detect the output shape (see `AGENTS.md`). Reuse `AsrModelFamily`:

| Family | Conversion to `_ConversationItem` |
|---|---|
| `whisper` | One item for each segment; keep `words` |
| `parakeet`, `nemotron` | One item for each sentence |
| `granite`, `granite_ctc` | One item for each audio window (see D3) |

Empty text gives an empty list. The pipeline then reports `PARTIAL_SUCCESS`, as it does for other
backends. The adapter reads attributes only, so it does not import `mlx_audio` classes (S1).

### D3: Pause-aware windows for Granite

The windows have a target of `max_time_chunk` (30 s), and a cut can move by
`chunk_search_window` (new option, default 10 s). A window is therefore 20 to 40 s. The last
window can be shorter, and a file of 40 s or less has one window.

Add a pure function in a new module, `docling/pipeline/asr_windows.py`. Only `_MlxAudioModel`
imports this module, inside a method (S1):

```python
def split_on_pauses(
    waveform: "np.ndarray", sample_rate: int, target_s: float, search_s: float
) -> list[tuple[float, float]]: ...
```

Algorithm, as tested:

1. Compute RMS energy in frames of 20 ms.
2. Set the threshold from this file: `floor + 0.15 * (speech - floor)`, with the 10th and 90th
   energy percentiles. An absolute threshold fails on quiet recordings.
3. For each cut, search `start + target ± search` for the longest quiet run of at least 200 ms.
   Cut at its middle.
4. If no run qualifies, cut at `start + target` and log the fallback at debug level.
5. The next window starts at the cut, so there is no overlap and no text to de-duplicate.

What this does and does not do:

- **It does:** cut at a pause when one exists. This should reduce mid-word and mid-sentence cuts,
  and Phase 2 measures by how much. It adds no dependency and no checkpoint.
- **It does not:** guarantee a sentence boundary. The audio has no punctuation at this stage. A
  speaker can pause in mid-sentence, or can speak for 40 s with no pause, so the fallback is
  necessary.

If the long-audio gate shows too many bad cuts, the next option is `mlx_audio.vad` with
`silero_vad`. It adds a checkpoint download, so do not use it unless a measurement requires it.

Apply this to `granite` and `granite_ctc` in the `mlx_audio` backend only. The Transformers path
stays the same. Window times give `start_time` and `end_time`, so `ffprobe` is not necessary.

Do not pass `InlineAsrOptions.max_new_tokens` (default 256) unchanged to Granite. Set a limit for
this family and test that it does not cut the text of a 40 s window.

### D4: Add explicit presets first. Do not change defaults.

New `AsrModelType` values:

- `whisper_{tiny,small,medium,base,large,turbo}_mlx_whisper`
- `whisper_{tiny,small,medium,base,large,turbo}_mlx_audio`
- `granite_*_mlx_audio`, `parakeet_ctc_1_1b_mlx_audio`, and `nemotron_*_mlx_audio`

Existing presets keep their engine, so no user output changes in Phases 1 to 3. Auto-selection is
a separate decision in Phase 4.

### D5: A separate torch-free extra

```toml
# Apple Silicon only; installs nothing on other platforms. Needs macOS 14 or later.
# Keep this free of format-audio: its mlx-whisper and openai-whisper dependencies need torch.
feat-asr-mlx-audio = [
  'mlx-audio>=0.6.4,<0.7.0 ; python_version >= "3.10" and sys_platform == "darwin" and platform_machine == "arm64"',
  'transformers>=5.16.0,<6.0.0 ; sys_platform == "darwin" and platform_machine == "arm64"',
]
```

- The `>=5.16.0` floor avoids the MPS crash range that `models-vlm-inline` documents for
  Transformers 5.9 to 5.15, and it matches `feat-asr-transformers`.
- The `<0.7.0` cap controls an API that changes quickly.
- Do not use `mlx-audio[stt]`: its `zstandard` dependency is only for Phonon, which is out of scope.
- Add the extra to the `all` bundle (no new failure, see the table above). Do not add it to
  `standard` or `format-audio`.
- Add `asr-mlx-audio = ['docling-slim[feat-asr-mlx-audio]==<version>']` in
  `packages/docling/pyproject.toml`, in the same form as the other re-exports.

### D6: Share one MLX lock

Create `docling/models/utils/mlx_lock.py`, in the `foundation` layer. Use the lock in `MlxVlmEngine`,
the legacy `HuggingFaceMlxModel`, `_MlxWhisperModel`, and `_MlxAudioModel`. It holds only a
`threading.Lock`, so it is safe on the import path.

This also corrects a current fault. `mlx_engine.py` and `vlm_pipeline_models/mlx_model.py` each
define their own `_MLX_GLOBAL_LOCK`, so the "global" locks do not exclude each other. Hold the lock
for each window, not for the whole file, so that VLM work is not blocked for a long time.

### D7: Treat Granite as remote-code loading

`Model.post_load_hook` in `granite_speech.py` always calls
`AutoTokenizer.from_pretrained(..., trust_remote_code=True)`. Docling cannot disable this. Therefore:

- All `MLX_AUDIO` presets pin a full commit SHA.
- Call `warn_on_unpinned_trust_remote_code()` for `model_family == "granite"`. It logs a warning or,
  with `DOCLING_SECURITY_REFUSE_UNPINNED_REMOTE_CODE`, raises an error.
- For a custom unpinned Granite repository, require an explicit `trust_remote_code=True` on the
  options, as `MlxVlmEngineOptions` does.
- `parakeet`, `nemotron`, `granite_ctc`, and `whisper` do not set this flag.
- Allow only the families in `AsrModelFamily`. Other `mlx-audio` families use
  `trust_remote_code=True` and are out of scope.

### D8: Download through `load()` when online

`download_hf_model()` has no file filter. `nvidia/nemotron-speech-streaming-en-0.6b` has 5.4 GB of
files, including a `.nemo` file that `mlx-audio` does not need. The `mlx-audio` allow-list avoids
this file. Use:

- Online: `load(repo_id, revision=revision)`.
- Offline: `load(_resolve_asr_artifacts(artifacts_path, repo_id))`.

This is a deliberate difference from `MlxVlmEngine`. `docling-tools models download-hf-repo` already
writes `<output>/<org--repo>` and works for offline use, but it downloads all files.

## Model mapping

Resolve and pin full SHAs when you implement this. Confirm each checkpoint with a test.

| Docling model | MLX Audio checkpoint | Status |
|---|---|---|
| Whisper tiny | `mlx-community/whisper-tiny-asr-fp16` | Tested |
| Whisper base, small, medium | `mlx-community/whisper-{size}-asr-fp16` | Repository exists; not tested |
| Whisper large | `mlx-community/whisper-large-v3-asr-fp16` | Repository exists; not tested; confirm that the version matches the current preset |
| Whisper turbo | `mlx-community/whisper-large-v3-turbo-asr-fp16` | Repository exists; not tested |
| Granite Speech 5.0 2B | `ibm-granite/granite-speech-5.0-2b` | Tested; text-loss gate |
| Granite 5.0 470M TurboCTC | `ibm-granite/granite-speech-5.0-470m-turboctc` | Tested; quality gate |
| Granite 5.0 470M TurboCTC NC | `ibm-granite/granite-speech-5.0-470m-turboctc-nc` | Listed by upstream; not tested |
| Granite Speech 4.1 2B | `ibm-granite/granite-speech-4.1-2b` | Not tested |
| Granite Speech 4.1 2B Plus | `ibm-granite/granite-speech-4.1-2b-plus` | Listed by upstream; not tested; plain ASR only |
| Granite 4.0 1B Speech | `ibm-granite/granite-4.0-1b-speech` | Listed by upstream; not tested |
| Nemotron 3.5 Streaming 0.6B | `mlx-community/nemotron-3.5-asr-streaming-0.6b` | Tested; last-word cut, see Phase 1 |
| Nemotron Streaming EN 0.6B | `nvidia/nemotron-speech-streaming-en-0.6b` | Tested |
| Parakeet CTC 1.1B | `mlx-community/parakeet-ctc-1.1b` | Tested; 9.1 GB for 5 minutes; test windows |

Out of scope: Distil-Whisper and models that Docling does not support today (Parakeet TDT, Qwen,
Voxtral, VibeVoice, and others). Add them later, one family at a time.

## Implementation plan

### Phase 0: Prepare (no behavior change)

1. Remove the eager `transformers` import from `pipeline_options_vlm_model.py` (see the docling-slim
   section). Add a test for `custom_stopping_criteria` with a real `StoppingCriteria` and a
   `GenerationStopper`.
2. Move the MLX lock to `docling/models/utils/mlx_lock.py` (D6). Update both VLM modules. Add the lock
   to `_MlxWhisperModel`.
3. Extract the shared file handling from `_MlxWhisperModel.run()` into a small base class, for example
   `_LocalAudioTranscriber`. It holds the `BytesIO` temporary file, the error handling, and the cleanup.
   A class attribute controls the FFmpeg check (S6). `_TransformersAsrModel` now inherits from
   `_MlxWhisperModel` only to get `run()`, and the new backend must not repeat this.
4. Add the slim guards from Phase 3 here, so that they protect all later phases.
5. Keep the `asr_pipeline.py` re-exports.

### Phase 1: Framework split and backend

1. **Dependency:** add `feat-asr-mlx-audio`, the `all` entry, and the meta-package re-export (D5). Add
   `mlx_audio.**` to `tool.ty.analysis.allowed-unresolved-imports`. Update the lock with
   `uv lock --upgrade-package mlx-audio`. Review any `mlx-vlm` version change in the lock.
2. **Framework split:** add `MLX_WHISPER`, `MLX_AUDIO`, and the alias resolver (D1). Add the
   `*_mlx_whisper` presets.
3. **Options** in `pipeline_options_asr_model.py`: `InlineAsrMlxAudioOptions(InlineAsrOptions)`, with
   `revision`, `model_family`, `language`, `task`, `prompt`, `word_timestamps`, `chunk_search_window`,
   `trust_remote_code`, and `supported_devices=[MPS]`.
4. **Spec** in `asr_presets.py`: add a `MLX_AUDIO` branch to `to_options()`. Pass `model_family` and
   `revision`.
5. **Windows:** add `docling/pipeline/asr_windows.py` (D3). Add the module to `tach.toml` if necessary.
6. **Transcriber** in `asr_transcriber.py`: add `_MlxAudioModel`.
   - Lazy import `mlx_audio.stt.load`, with the S5 error.
   - Check the platform as in S3. Do not call `decide_device()`.
   - Load as in D8. Apply the checks in D7.
   - `transcribe()` builds family kwargs, calls `generate()` under the lock, and converts the output
     (D2, D3).
   - Add the backend to `_AsrModelFactory.create()`.
7. **Family kwargs:**
   - `whisper`: `language`, `task`, `word_timestamps`, `temperature`, thresholds.
   - `granite`: `source_language` or `language` by version, `prompt`, `punctuate=True`. Map
     `task="translate"` only if the checkpoint supports it. Otherwise raise a clear error.
   - `parakeet`, `granite_ctc`: if `language` is set to a value that the model does not support, raise a
     clear error.
   - `nemotron`: map ISO codes to prompt keys only if the tests need it. Otherwise forward only a valid
     prompt key.
8. **Nemotron last-word cut:** reproduce with `nemotron_3_5`. If silence padding corrects it, pad the
   waveform in the adapter. If upstream has a defect, report it upstream.
9. **Presets and CLI:**
   - Add overrides and the `*_mlx_audio` values to `AsrModelType` (D4).
   - Extend `_resolve_asr_options()` in `docling/cli/main.py`.
   - Extend the `--asr-language` condition near `docling/cli/main.py:2083`, and update the help text
     ("Source language for Transformers ASR").

**Exit gate:** on Apple Silicon, in a torch-free venv with `docling-slim[feat-asr-mlx-audio]` only:

- `find_spec("torch")` is `None`.
- Each new preset converts the MP3 and M4A samples through `AsrPipeline`, with status `SUCCESS` and the
  expected words.
- Each preset produces valid `TrackSource` values.
- `import docling` prints no PyTorch warning.

`VideoPipeline` works with one preset, in an environment that has what the video path needs. Record if
it can run without torch.

### Phase 2: Remaining presets

Add the untested Granite and Whisper rows from the mapping table. Add a preset only if it passes the
Phase 1 gate and the long-audio gate. Otherwise record the reason here and leave it out.

**Long-audio gate:** real recordings of at least 10 minutes, not a repeated clip. Use one recording with
natural pauses and one with little silence. Pass criteria:

- No repetition loop, and no large text loss against the Whisper turbo reference. This settles the Granite
  5.0 2B result.
- Inspect the text at each window edge. Record the rate of cuts in mid-word and in mid-sentence, and
  the fallback rate. Compare with fixed 30 s windows.
- Record peak memory, each model in a separate process.

If pause windows are not better than fixed windows on real speech, use the simpler code and report this to
the maintainer. Use `docs/examples/asr_pipeline_performance_comparison.py` where possible.

### Phase 3: Tests and documentation

**Slim guards** (Phase 0 adds these):

- Extend `tests/test_document_converter_import_without_convert_core.py`, or add a test with the same
  `_run_without` subprocess method. Block `torch`, `transformers`, `mlx`, `mlx.core`, `mlx_audio`, and
  `mlx_whisper`, then import `docling.document_converter`, `docling.pipeline.asr_pipeline`, and
  `docling.datamodel.asr_model_specs`. Build all presets with `to_options()`.
- Add the PR 5090 regression that does not exist yet: in a fresh subprocess, import
  `docling.datamodel.asr_model_specs` and `docling.pipeline.asr_transcriber`, with a stub `mlx_audio` and
  `mlx_whisper` on the path that fail if they are imported.
- Extend the `test-slim-import` job in `.github/workflows/checks.yml` to import
  `docling.datamodel.asr_model_specs` and `docling.pipeline.asr_pipeline`. The job already fails on a torch
  leak through `DocumentConverter`.

**Behavior tests** (`tests/test_asr_mlx_audio.py`, marker `ml_asr`). Follow `AGENTS.md`: test behavior, with
no trivial or self-validating tests.

- Framework: `mlx` and `mlx_whisper` give `mlx-whisper` options with the same repository, `mlx_audio` gives the
  new options, and `to_options(MLX)` on Granite raises an error that names `mlx_audio`.
- `split_on_pauses()`, with synthetic waveforms and no MLX: a cut lands in the longest pause in the search range,
  no pause gives the target cut, and a low-level recording still finds its pauses. All windows cover the file with no
  gap, and a short file has one window.
- Adapter: `STTOutput` and `AlignedResult` shapes give the correct items. Includes a Granite empty `segments` list,
  empty text, a zero-duration sentence, and correct time offsets across windows.
- Missing library: the `ImportError` names the extra. Use the `sys.modules` method of the existing tests.
- No torch: `_MlxAudioModel` construction does not call `decide_device()`, tested with `torch` blocked in a subprocess.
- Offline: `artifacts_path` loads a local path, and a missing checkpoint gives `FileNotFoundError` without Hub access.
- Remote code: an unpinned custom Granite repository fails without an explicit opt-in.
- Regression: `whisper_*_mlx` presets still create `InlineAsrMlxWhisperOptions` with `inference_framework="mlx"`.
- Selector tests patch `find_spec`, as PR 5090 did.
- Integration (Apple Silicon only, guard as in S7): `whisper_tiny_mlx_audio` and
  `granite_speech_5_0_470m_turboctc_mlx_audio` on `sample_10s.mp3`.
- Extend the parametrized test in `tests/test_asr_model_decoders.py` that resolves presets.

**CI:** Linux CI cannot run `mlx-audio`. Add a job for the `run_macos` input to `.github/workflows/checks.yml`: install
only `.[feat-asr-mlx-audio]` without a lock in a clean venv, assert that torch is absent, and run one small
conversion. `run-tests-macos` uses `uv sync --all-packages`, so it cannot detect a torch leak. Add the job to `check` and
`ALLOWED_SKIPS`, as PR 5072 did. Check that the macOS runner has enough memory and disk space.

**Documentation** (`AGENTS.md` requires the usage skills to stay in sync with the extras):

- `packages/docling-slim/README.md`: the extra, with a torch-free note, the Apple Silicon limit, and macOS 14 minimum.
- `docs/usage/processing_audio_media.md`: backend table with the three values, install, presets, window behavior, and
  limits. State that a window edge is a pause and not a guaranteed sentence boundary.
- `docs/usage/model_catalog.md`: ASR engine support.
- `docling/.agents/skills/docling/references/python-sdk.md` and `cli.md`.
- `docling/datamodel/pipeline_options_asr_model.py`: module docstring.
- `docs/examples/minimal_asr_pipeline.py` or a short new example.
- `docs/reference/cli.md`, if the file is not generated.

**Before finishing each phase:** `make validate`, the affected tests, and `tach` (it runs in CI).

### Phase 4: Default-selection decision (separate pull request)

Run the comparison only after Phases 1 to 3 are in use: `mlx-audio`, `mlx-whisper`, and Transformers on MPS, on the same
recordings. Record accuracy, time, and memory.

Then decide:

- **`mlx` target:** if `mlx-audio` is better or equal for Whisper, change `_MLX_ALIAS_TARGET` to `MLX_AUDIO`. The static
  `WHISPER_*_MLX` constants must then derive from `to_options(MLX)`. Announce a change of checkpoint for `artifacts_path`
  users. `mlx_whisper` stays as the explicit way to keep the old backend.
- **Auto-selection:** auto-select `mlx-audio` on Apple Silicon when it is installed and the preset declares an export. Use
  `find_spec` and a platform check, not torch (S2, S3). Also change `_detect_hardware_and_libraries()` so that a torch-free
  install does not select native Whisper.
- **`mlx-whisper` removal:** removing it also removes a dependency that needs torch and that has had no release since 2025-08-29.
- **`docling[asr]`:** decide if it includes `mlx-audio`.

## Risks

| Risk | Result | Mitigation |
|---|---|---|
| A later change adds an optional import to the ASR path | Slim installs fail or a process aborts | S1, S2, subprocess tests, and `test-slim-import` |
| Torch leak only on macOS | Linux CI cannot detect it | New macOS job with no lock |
| `mlx-audio` pins an exact `mlx` version | Resolution can block on a new MLX release | Already true for `mlx-vlm` users. Keep the extra optional. Add a regression check on lock updates |
| Upstream API changes quickly | Adapter breaks | `<0.7.0` cap; macOS integration test |
| New checkpoints change the output | Whisper text differs from `mlx-whisper` | Separate presets; `mlx` stays on `mlx-whisper` until Phase 4 |
| Granite 5.0 2B text loss | Missing text with a success status | Long-audio gate before this preset ships |
| Pause cut in mid-sentence | Item boundary is not at the end of a sentence | Documented limit; rate measured in Phase 2; VAD as the next option |
| High memory use | 9.6 GB seen on Granite 5.0 2B with windows | Measure in separate processes; document the requirement |
| Process-wide environment change at import | Hides Transformers advisory warnings | Lazy import; document it |
