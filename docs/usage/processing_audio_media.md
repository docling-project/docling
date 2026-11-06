# Processing audio and video

Docling converts audio and video files into a structured [`DoclingDocument`](../concepts/docling_document.md) — the same intermediate representation used for PDFs, DOCX files, and everything else. From there you can export to Markdown, JSON, HTML, or DocTags, and plug the result directly into RAG pipelines, summarizers, or search indexes.

Audio files run through the **ASR pipeline**: transcription only. Video files run through the dedicated **video pipeline**, which transcribes the audio track *and* samples representative frames, with optional speaker diarization — see [Processing video](#processing-video) below.

Under the hood, both pipelines transcribe with an ASR model that you can select. The default is [OpenAI Whisper](https://github.com/openai/whisper), and Docling also supports the Granite Speech, Parakeet, and Nemotron model families. The presets auto-select the best backend for your hardware, so the [basic example](#basic-usage) below needs no configuration. To change the model, force a particular backend, or opt into the faster experimental WhisperS2T backend, see [Choosing an ASR model and backend](#choosing-an-asr-model-and-backend).

## Supported formats

| Type | Formats |
|------|---------|
| Audio | WAV, MP3, M4A, AAC, OGG, FLAC |
| Video | MP4, AVI, MOV, MKV, WEBM |

For video files, Docling extracts the audio track automatically before transcription. You don't need to run FFmpeg manually.

!!! note "ffmpeg required"
    Most backends require the `ffmpeg` executable to be installed and available on your `PATH` to decode audio. This applies to common audio formats such as MP3, WAV, M4A, AAC, OGG, and FLAC, and to video files whose audio track and frames are extracted before transcription. The `mlx-audio` backend decodes WAV, MP3, and FLAC itself, and needs `ffmpeg` for the other formats. Install it with your system package manager — e.g. `brew install ffmpeg` on macOS, `apt-get install ffmpeg` on Debian-based Linux, or `winget install ffmpeg` on Windows.

## Installation

The ASR pipeline is an optional extra. Install it alongside the base package:

```bash
pip install "docling[asr]"
```

Or with `uv`:

```bash
uv add "docling[asr]"
```

This installs all inline ASR engines: Whisper, Transformers, and `mlx-audio`. On Apple Silicon, it needs macOS 14 or later, and the installation fails on macOS 13. Use `docling[audio]` on macOS 13. Select another extra for a smaller installation:

| Goal | Install | Notes |
|------|---------|-------|
| All inline ASR engines | `docling[asr]` or `docling-slim[models-asr-inline]` | Whisper, Transformers, and `mlx-audio`. Needs macOS 14 or later on Apple Silicon |
| Whisper only | `docling[audio]` or `docling-slim[format-audio]` | |
| Granite Speech, Parakeet, and Nemotron through Transformers | `docling-slim[feat-asr-transformers]` | Includes the Whisper backends |
| Apple Silicon without PyTorch | `docling-slim[feat-asr-mlx-audio]` | Installs nothing on other platforms. Needs macOS 14 or later |
| Video, including optional speaker diarization | `docling-slim[format-video]` | See [Processing video](#processing-video) |

`format-video` already includes everything `audio` provides (transcription), plus the frame-sampling and diarization dependencies (`resemblyzer`, `soundfile`, `scikit-learn`, `librosa`).

!!! note "`feat-asr-mlx-audio` needs a selected model"
    The auto-selecting `WHISPER_*` presets, which include the CLI default, need PyTorch or `mlx-whisper`. With only `feat-asr-mlx-audio`, select an `mlx-audio` preset, such as `--asr-model granite_speech_5_0_470m_turboctc`.

!!! note "WhisperS2T on Linux with CUDA"
    The optional WhisperS2T backend uses CTranslate2, which loads NVIDIA's cuBLAS shared library at runtime. On Linux, if WhisperS2T model loading fails because the library cannot be found, add it to your `LD_LIBRARY_PATH`. When cuBLAS is installed from a pip wheel (e.g. `nvidia-cublas-cu12`), the shared library lives under the `nvidia/cublas/lib` directory inside your environment's `site-packages`.

## Basic usage

```python
from pathlib import Path

from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline

pipeline_options = AsrPipelineOptions()
pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

converter = DocumentConverter(
    format_options={
        InputFormat.AUDIO: AudioFormatOption(
            pipeline_cls=AsrPipeline,
            pipeline_options=pipeline_options,
        )
    }
)

result = converter.convert(Path("recording.mp3"))
doc = result.document

# Export to Markdown
print(doc.export_to_markdown())
```

This example uses the plain `AudioFormatOption`/`AsrPipeline`, which only transcribes — it works for any file `docling` can extract audio from, video included. For video files you'll usually want the dedicated video pipeline instead, which also samples frames and can attribute speakers — see [Processing video](#processing-video).

### Exporting to different formats

`result.document` is a `DoclingDocument`. You can export it to any supported format:

```python
doc.export_to_markdown()   # Markdown
doc.export_to_dict()       # JSON-serializable dict
doc.export_to_html()       # HTML
doc.export_to_doctags()    # DocTags
```

See [Serialization](../concepts/serialization.md) for more on export options.

## Understanding the output

The ASR pipeline produces **paragraph-level** Markdown with timestamps per segment:

```
[time: 0.0-4.0]  Shakespeare on Scenery by Oscar Wilde

[time: 5.28-9.96]  This is a LibriVox recording. All LibriVox recordings are in the public domain.
```

The length of a segment depends on the model. Whisper gives short phrases. Granite gives a segment for each window of about 30 seconds, as [Long recordings](#long-recordings) explains.

This structured output is immediately suitable as input to a vector embedding model, a summarizer, or any other downstream stage.

## A practical use case: searchable meeting archives

A common problem in engineering teams: every all-hands, customer call, and design review gets recorded. The recordings accumulate on Google Drive or S3. Nobody watches them. Nobody can search them. Institutional knowledge is locked inside audio files.

Docling solves the ingestion step. Pair it with a vector store and you have a queryable knowledge base over your entire audio archive.

### Standalone transcription script

For a full working example, see the [example-docling-media](https://github.com/TejasQ/example-docling-media) repository, which processes a directory of audio/video files and writes each transcript to a Markdown file.

The core of that project is ~30 lines:

```python
from pathlib import Path

from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline


def main():
    audio_path = Path("videoplayback.mp3")

    pipeline_options = AsrPipelineOptions()
    pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    result = converter.convert(audio_path)
    md = result.document.export_to_markdown()
    Path("transcript.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
```

### Building a RAG pipeline with LangChain

Docling integrates with LangChain via `DoclingLoader`, which wraps `DocumentConverter` and handles chunking automatically. To build a retrieval pipeline over your audio archive:

```python
from langchain_docling import DoclingLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load and chunk all audio files in a directory
loader = DoclingLoader("recordings/")
docs = loader.load()

# Embed and index
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Query in natural language
results = retriever.invoke("What did we decide about the auth service in Q3?")
```

See the [LangChain integration guide](../integrations/langchain.md) for more details on `DoclingLoader` options.

## Processing video

Video files route through the dedicated **video pipeline** (`VideoPipeline`), which transcribes the audio track like the ASR pipeline above, and additionally:

- Samples representative frames from the video, embedded in the output `DoclingDocument` as picture items.
- Optionally assigns speaker labels to transcript segments via diarization.

Install the `format-video` extra (see [Installation](#installation)) to get frame sampling and diarization support alongside transcription.

```python
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VideoPipelineOptions
from docling.document_converter import DocumentConverter, VideoFormatOption
from docling.utils.video_frame_sampling import VideoFrameSamplingMode

pipeline_options = VideoPipelineOptions(
    frame_sampling_mode=VideoFrameSamplingMode.SCENE_CHANGE,
    scene_change_prominence=0.03,  # recommended for meetings
    enable_diarization=True,       # requires resemblyzer; see Installation
)

converter = DocumentConverter(
    format_options={
        InputFormat.VIDEO: VideoFormatOption(pipeline_options=pipeline_options)
    }
)

result = converter.convert(Path("meeting.mp4"))
doc = result.document

print(doc.export_to_markdown())
```

### Frame sampling modes

`VideoPipelineOptions.frame_sampling_mode` controls how representative frames are chosen:

| Mode | Option value | Behavior |
|------|--------------|----------|
| Fixed interval | `VideoFrameSamplingMode.FIXED_INTERVAL` (default) | One frame every `frame_interval_seconds` (default 10s). |
| Scene change | `VideoFrameSamplingMode.SCENE_CHANGE` | One frame per detected scene, picked for sharpness. Sensitivity auto-calibrates per video by default. |

Recommended configs by use case:

| Use case | Configuration |
|----------|---------------|
| Business meetings | `frame_sampling_mode=SCENE_CHANGE, scene_change_prominence=0.03` |
| Lecture recordings | `frame_sampling_mode=SCENE_CHANGE, cuts_per_minute=2.0` |
| General video | `frame_sampling_mode=FIXED_INTERVAL, frame_interval_seconds=10.0` |

`max_sampled_frames` caps the total number of frames sampled regardless of mode (default `200`, about 33 minutes at the default 10 s interval; set `None` for no limit). Set `generate_frame_images=False` to skip frame sampling entirely and transcribe only.

### Container handling and time limits

The video pipeline reads each file with the FFmpeg demuxer that matches its extension (`.mp4`/`.mov` as MP4/QuickTime, `.mkv`/`.webm` as Matroska, `.avi` as AVI) and only from the local file itself. A file whose content does not match its extension fails to open instead of being interpreted as another format.

FFmpeg calls share the document's `document_timeout` budget when it is set. Without it, each call is limited to 5 minutes or the video duration, whichever is longer. If a call runs out of time, the frames and transcript gathered so far are kept, the result status is `PARTIAL_SUCCESS`, and a `TIMEOUT` error is recorded in `ConversionResult.errors` (see `ConversionResult.has_timeout_errors()`).

### Speaker diarization

Set `enable_diarization=True` to attribute transcript segments to speakers. Diarization runs via [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) embedding clustering and auto-detects the number of speakers; it requires the extra dependencies bundled in `format-video` (`resemblyzer`, `soundfile`, `scikit-learn`, `librosa`). If those aren't installed, diarization is silently skipped and a warning is logged — transcription and frame sampling still proceed normally.

### From the command line

```bash
# fixed-interval sampling (default)
docling --to md video.mp4

# scene-change sampling, tuned for meetings
docling --to md --video-sampling-mode scene --video-prominence 0.03 video.mp4

# scene-change sampling with speaker diarization
docling --to md --video-sampling-mode scene --video-diarization video.mp4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video-sampling-mode` | `fixed` | `fixed` or `scene`. |
| `--video-frame-interval` | `10.0` | Seconds between frames in fixed-interval mode. |
| `--video-cuts-per-minute` | `0.0` (unset) | Target scene cuts per minute; overrides `--video-prominence` when set. |
| `--video-prominence` | `0.0` (auto) | Scene-change sensitivity threshold. `0` auto-calibrates to the video's motion. |
| `--video-diarization` | disabled | Enable speaker diarization. Requires `resemblyzer`. |
| `--document-timeout` | unset | Time budget in seconds for each video, shared by all FFmpeg calls. |

See the [CLI reference](../reference/cli.md) for the complete flag list.

## Choosing an ASR model and backend

Docling includes these local backends. The [Installation](#installation) table shows the extra for each one. A [remote server](#transcribing-with-a-remote-whisper-server) is also possible.

| Backend | Library | Hardware | Models | Notes |
|---------|---------|----------|--------|-------|
| **Native Whisper** | `openai-whisper` (PyTorch) | CPU, CUDA | Whisper | Default; broadest compatibility |
| **MLX Whisper** | `mlx-whisper` | Apple Silicon (MPS) | Whisper | Optimized for M-series Macs |
| **mlx-audio** | `mlx-audio` | Apple Silicon (MPS), macOS 14 or later | Whisper tiny, Granite Speech 5.0, Granite Speech 4.1 Plus, Parakeet, Nemotron | No PyTorch; preferred for these models on Apple Silicon |
| **Transformers** | `transformers` (PyTorch) | CPU, CUDA, MPS, XPU | Whisper, Granite Speech, Parakeet, Nemotron | All speech models on all platforms |
| **WhisperS2T** | `whisper-s2t-reborn` (CTranslate2) | CPU, CUDA | Whisper | Optional & experimental; batched decoding for high throughput |

### Automatic backend selection

A preset name without a backend suffix selects the backend for your machine. There is one rule for Whisper and one for the other speech models.

**Whisper.** The auto-selecting presets — `WHISPER_TINY`, `WHISPER_BASE`, `WHISPER_SMALL`, `WHISPER_MEDIUM`, `WHISPER_LARGE`, and `WHISPER_TURBO` — pick a backend for you based on the hardware they detect, in this priority order:

1. **MLX Whisper** — on Apple Silicon, when `mlx-whisper` is installed.
2. **Native Whisper** — on all other hardware.

WhisperS2T, `mlx-audio`, and Transformers are **never** auto-selected for these presets; you opt into them explicitly (see below).

This is why the [Basic usage](#basic-usage) example needs no hardware-specific code — `asr_model_specs.WHISPER_TURBO` runs on MLX on a Mac and on native Whisper on Linux and Windows. `WHISPER_TURBO` is a good default; to change the model size, swap in another auto-selecting preset:

```python
from docling.datamodel import asr_model_specs

pipeline_options.asr_options = asr_model_specs.WHISPER_LARGE
```

**Granite Speech, Parakeet, and Nemotron.** A preset from [Speech models](#speech-models) uses `mlx-audio` when all of these conditions are true:

- The machine is Apple Silicon.
- `mlx-audio` 0.6.4 or later is installed.
- The model has an `mlx-audio` export.
- The device is `auto` or `mps`.

In all other cases, the preset uses Transformers. For example, `--device cpu` selects Transformers on a Mac.

`docling[asr]` installs `mlx-audio` on Apple Silicon, and `mlx-vlm` can also install it. This changes the backend of these presets. The text can differ slightly between backends. Select a `_TRANSFORMERS` preset to keep one backend on all machines.

When a preset selects `mlx-audio`, Docling makes a last check before it loads the model. It uses Transformers if one of these conditions is true:

- The device in `AcceleratorOptions` is not `auto` or `mps`.
- The process cannot use a Metal device, for example on a headless runner, in a virtual machine, or in a restricted process. Docling logs a warning for this condition.

The fallback keeps the language, task, prompt, and window settings of the options. It needs a Transformers install. An explicit `_MLX_AUDIO` preset does not fall back, and it gives an error when it cannot run.

To see the models and the backend that each one selects on your machine, run `docling models --asr`. See [From the command line](#from-the-command-line).

### Forcing a specific backend

Explicit variants bypass hardware detection. Use them to pin a backend regardless of platform:

| Suffix | Backend | Available for |
|--------|---------|---------------|
| `_NATIVE` | Native Whisper | Whisper |
| `_MLX_WHISPER` | MLX Whisper | Whisper |
| `_MLX` | Alias that currently selects MLX Whisper | Whisper |
| `_S2T` | WhisperS2T | Whisper |
| `_TRANSFORMERS` | Transformers | Whisper, and the models with an `mlx-audio` export |
| `_MLX_AUDIO` | `mlx-audio` | `WHISPER_TINY`, and the models with an `mlx-audio` export |

For example:

```python
from docling.datamodel import asr_model_specs

# Native OpenAI Whisper (CPU / CUDA)
pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO_NATIVE

# MLX (Apple Silicon)
pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO_MLX

# Granite Speech through Transformers on every platform
pipeline_options.asr_options = asr_model_specs.GRANITE_SPEECH_5_0_2B_TRANSFORMERS

# Granite Speech through mlx-audio (Apple Silicon)
pipeline_options.asr_options = asr_model_specs.GRANITE_SPEECH_5_0_2B_MLX_AUDIO
```

The rest of the setup — the `DocumentConverter` from [Basic usage](#basic-usage) — is unchanged.

### Speech models

These presets run on Transformers or, when the model has an export, on `mlx-audio`. The SDK constants are in `docling.datamodel.asr_model_specs`; the CLI names are their lower-case names. [Automatic backend selection](#automatic-backend-selection) explains how a preset selects its backend, and the [Installation](#installation) table shows the extras.

| SDK constant | Hugging Face repository | `mlx-audio` export |
|---|---|---|
| `GRANITE_SPEECH_5_0_2B` | [`ibm-granite/granite-speech-5.0-2b`](https://huggingface.co/ibm-granite/granite-speech-5.0-2b) | Same repository |
| `GRANITE_SPEECH_5_0_470M_TURBOCTC` | [`ibm-granite/granite-speech-5.0-470m-turboctc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) | Same repository |
| `GRANITE_SPEECH_5_0_470M_TURBOCTC_NC` | [`ibm-granite/granite-speech-5.0-470m-turboctc-nc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc-nc) | None |
| `GRANITE_SPEECH_4_1_2B` | [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | None |
| `GRANITE_SPEECH_4_1_2B_PLUS` | [`ibm-granite/granite-speech-4.1-2b-plus`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) | Same repository |
| `GRANITE_4_0_1B_SPEECH` | [`ibm-granite/granite-4.0-1b-speech`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | None |
| `NEMOTRON_3_5_ASR_STREAMING` | [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | [`mlx-community/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b) |
| `NEMOTRON_SPEECH_STREAMING_EN_0_6B` | [`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) | Same repository |
| `PARAKEET_CTC_1_1B` | [`nvidia/parakeet-ctc-1.1b`](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | [`mlx-community/parakeet-ctc-1.1b`](https://huggingface.co/mlx-community/parakeet-ctc-1.1b) |

```python
from docling.datamodel import asr_model_specs

options = asr_model_specs.GRANITE_SPEECH_5_0_2B.model_copy(deep=True)
# Override the checkpoint revision when needed. A revision belongs to the repository
# of the selected backend.
# options.revision = "branch-tag-or-commit"
pipeline_options.asr_options = options
```

```bash
docling recording.wav --asr-model granite_speech_5_0_2b --to md
docling recording.wav --asr-model parakeet_ctc_1_1b --to md
```

The backend changes the output:

- Granite gives one segment for each window on both backends. See [Long recordings](#long-recordings).
- Parakeet and Nemotron give one segment for each sentence on `mlx-audio`, and one segment for the full audio on Transformers.
- The backends can also give slightly different text for the same model.

More notes:

- On Apple MPS with Transformers, Granite and TurboCTC load weights in FP32 to avoid a BF16 normalization failure. Allow more memory than for BF16 weights.
- On `mlx-audio`, the tokenizer of `GRANITE_SPEECH_5_0_2B` runs remote code from its repository. The preset pins a commit. If you select a revision that is not a full commit SHA, you must also set `trust_remote_code=True`.
- `GRANITE_SPEECH_4_1_2B_PLUS` gives phrase times and speaker labels by default. See [Timestamps and speakers with Granite Speech Plus](#timestamps-and-speakers-with-granite-speech-plus). If you switch both options off and use your own `prompt` to ask for speaker or timestamp tags, Docling does not map them to structured tracks. The tags stay in the text.
- Streaming model names identify the checkpoints; this pipeline performs file transcription.
- The `TURBOCTC_NC` preset uses CC-BY-NC-SA-4.0.

### Long recordings

Granite models return no timestamps, so Docling splits a recording into windows and decodes each window separately. This applies to the Granite presets on both backends. Each window gives one segment, and the time range of the segment is the range of its window.

A cut goes to the longest pause within the search range. With the defaults, windows are 20 to 40 seconds long, and a recording of up to 40 seconds is one window. A pause is not always a sentence boundary, so a cut can fall inside a sentence. If the range has no pause, the cut is at the target.

| Option | Default | Meaning |
|--------|---------|---------|
| `max_time_chunk` | `30.0` | Target window length in seconds |
| `chunk_search_window` | `10.0` | Seconds that a cut can move from the target to reach a pause. For the Granite families, it must be smaller than `max_time_chunk` |
| `max_new_tokens` | `1024` | Token limit for each window. Only the Transformers backend uses it, also when it is the fallback of an `mlx-audio` preset. A low value silently cuts text. `mlx-audio` ignores this option and uses at least `1024` tokens for each window |

```python
from docling.datamodel import asr_model_specs

pipeline_options.asr_options = asr_model_specs.GRANITE_SPEECH_5_0_2B.model_copy(
    update={"max_time_chunk": 20.0, "chunk_search_window": 5.0}
)
```

The backend affects time and memory. This table shows one 65-minute English recording (MP3) with the default options. The other conversions did not run at the same time:

| Model | Backend | Time | Peak memory |
|-------|---------|------|-------------|
| Granite Speech 5.0 470M TurboCTC | Transformers | 25 s | 3.5 GB |
| Granite Speech 5.0 470M TurboCTC | `mlx-audio` | 42 s | 2.6 GB |
| Granite Speech 5.0 2B | Transformers | 567 s | 14.6 GB |
| Granite Speech 5.0 2B | `mlx-audio` | 297 s | 6.2 GB |

Hardware and software: MacBook Pro with Apple M3 Max (16-core CPU), 64 GB of memory, macOS 26.6, `mlx-audio` 0.6.4, `mlx` 0.32.4, `transformers` 5.16.1, and PyTorch 2.14.1. Time is the full `docling convert` command, including model loading from a local cache. Your results can differ with other hardware and recordings.

### Timestamps and speakers with Granite Speech Plus

The other Granite presets give the time range of a full window. `GRANITE_SPEECH_4_1_2B_PLUS` gives one item for each phrase, with its real start and end times and a speaker label. A silence or a new speaker ends a phrase:

```bash
docling recording.mp3 --asr-model granite_speech_4_1_2b_plus --to vtt
```

```text
00:00:00.920 --> 00:00:02.680
<v Speaker 1>shakespeare on scenery

00:00:03.290 --> 00:00:04.390
<v Speaker 1>by oscar wilde
```

Both modes are on by default, and each one costs one decoding pass. Switch them off to get a faster conversion:

| Result | CLI | SDK option |
|--------|-----|------------|
| No speaker labels | `--no-asr-speakers` | `speaker_attribution=False` |
| No word timing: one item for each speaker turn, with estimated times | `--no-asr-timestamps` | `word_timestamps=False` |
| Plain text: one item for each window | `--no-asr-speakers --no-asr-timestamps` | Both |

```python
from docling.datamodel import asr_model_specs

pipeline_options.asr_options = asr_model_specs.GRANITE_SPEECH_4_1_2B_PLUS.model_copy(
    update={"speaker_attribution": False}
)
```

On the [hardware in the table above](#long-recordings), both modes on `mlx-audio` took about 31 minutes and 5.8 GB of memory for the same 65-minute recording. Word timing alone on Transformers took about 59 minutes and 13.4 GB. Other conversions ran during parts of these measurements, so use them only as a guide.

Know these limits:

- The model gives no punctuation or capitalization in most output, so a phrase is not a sentence.
- Speaker numbers are labels, not identities. The model numbers speakers by order of appearance in each call, so Docling decodes windows of about 4 minutes with 60 seconds of overlap, and uses the overlap to keep the numbers stable. A speaker who talks through a full overlap can make a later number wrong, and a new person can get the number of a known person.
- `DoclingDocument` keeps the time range and speaker of each phrase. It does not keep the time of each word.
- A silence of more than 10 seconds inside a window can make the later times of that window too early, because the model reports times modulo 10 seconds.
- `prompt` is available only when both modes are off. `task="translate"` is not available.
- The model card recommends `repetition_penalty=2.0`. Docling does not use it, because in tests it removed the timestamp tags and the spaces.

### Languages and translation

The `--asr-language` CLI flag applies to the Transformers and `mlx-audio` presets. In the SDK, set `language` and `task` on the options. Support depends on the backend and the model family:

| Family | `language` on Transformers | `language` on `mlx-audio` | `task="translate"` |
|--------|----------------------------|---------------------------|--------------------|
| Whisper | Source language | Source language | Both backends |
| Granite (2B, 4.x) | Ignored | Source language | Both backends; the target is English |
| Granite Speech Plus | Ignored | Ignored | An error on both backends |
| Granite CTC (TurboCTC) | Ignored | English only | Ignored on Transformers; an error on `mlx-audio` |
| Parakeet | Ignored | English only | Ignored on Transformers; an error on `mlx-audio` |
| Nemotron | Ignored | Must be a language of the checkpoint | Ignored on Transformers; an error on `mlx-audio` |

On `mlx-audio`, an unsupported language is an error. A custom `prompt` works with Granite on both backends and with Whisper on `mlx-audio`. On `mlx-audio`, the other families reject it.

### Local checkpoints

With `artifacts_path`, provide the local checkpoint or its parent directory. The loader does not download missing weights.

The checkpoint must match the selected backend. The `mlx-audio` exports of Parakeet and Nemotron 3.5 are in other repositories, and `WHISPER_TINY_MLX_AUDIO` also has its own. On Apple Silicon, select a `_TRANSFORMERS` preset to keep using a local Transformers checkpoint of these models.

### WhisperS2T: high-throughput transcription

WhisperS2T runs Whisper through [CTranslate2](https://github.com/OpenNMT/CTranslate2) with batched, VAD-segmented decoding. On CPU and CUDA it is typically the fastest backend and uses less VRAM than native Whisper at the larger model sizes, which makes it well suited to transcribing large batches of files. It is **experimental** and opt-in — select a `_S2T` preset:

```python
from docling.datamodel import asr_model_specs

pipeline_options.asr_options = asr_model_specs.WHISPER_LARGE_V3_S2T
```

Available `_S2T` presets:

| Preset | HuggingFace model | Multilingual? |
|--------|-------------------|---------------|
| `WHISPER_TINY_S2T` | `tiny` | yes |
| `WHISPER_TINY_EN_S2T` | `tiny.en` | English-only |
| `WHISPER_BASE_S2T` | `base` | yes |
| `WHISPER_BASE_EN_S2T` | `base.en` | English-only |
| `WHISPER_SMALL_S2T` | `small` | yes |
| `WHISPER_SMALL_EN_S2T` | `small.en` | English-only |
| `WHISPER_DISTIL_SMALL_EN_S2T` | `distil-small.en` | English-only |
| `WHISPER_MEDIUM_S2T` | `medium` | yes |
| `WHISPER_MEDIUM_EN_S2T` | `medium.en` | English-only |
| `WHISPER_DISTIL_MEDIUM_EN_S2T` | `distil-medium.en` | English-only |
| `WHISPER_LARGE_V3_S2T` | `large-v3` | yes |
| `WHISPER_DISTIL_LARGE_V3_S2T` | `distil-large-v3` | English-only |
| `WHISPER_DISTIL_LARGE_V3_5_S2T` | `distil-large-v3.5` | English-only |
| `WHISPER_LARGE_V3_TURBO_S2T` | `large-v3-turbo` | yes (no `translate`) |

The English-only presets reject a non-`en` language and the `translate` task; `large-v3-turbo` is multilingual but does not support `translate`. For multilingual transcription or speech translation, use a multilingual preset such as `WHISPER_LARGE_V3_S2T`.

To tune throughput and accuracy, construct the options directly instead of using a preset:

```python
from docling.datamodel.pipeline_options_asr_model import (
    InferenceAsrFramework,
    InlineAsrWhisperS2TOptions,
)

pipeline_options.asr_options = InlineAsrWhisperS2TOptions(
    repo_id="large-v3",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    torch_dtype="float16",  # float32 | float16 | bfloat16
    batch_size=8,           # higher = more throughput, more VRAM
    beam_size=1,            # 1 = greedy (fastest); higher may improve accuracy
)
```

!!! note "WhisperS2T is not available on Apple Silicon"
    The `whisper-s2t-reborn` dependency installs only on non-Apple-Silicon platforms, so `_S2T` presets can't be used on M-series Macs — use the native, MLX, or `mlx-audio` backends there. On Linux with CUDA, see the [cuBLAS note](#installation) above if model loading fails.

### Transcribing with a remote Whisper server

Instead of loading a model in the Docling process, the ASR pipeline can send the audio to an OpenAI-compatible
`/v1/audio/transcriptions` endpoint — a Whisper server that is deployed on its own (for example faster-whisper-server,
whisper.cpp's server or vLLM) or a hosted service. This keeps the model, and its GPU, out of the process that converts
documents, and lets several services share one transcription server.

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.datamodel.pipeline_options_asr_model import ApiAsrOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline

pipeline_options = AsrPipelineOptions(
    asr_options=ApiAsrOptions(
        url="http://whisper:8000/v1/audio/transcriptions",
        model="Systran/faster-whisper-small",
        language="en",
        initial_prompt="Columbus flight control loop: COL FLIGHT, COL SYSTEMS",
        headers={"Authorization": "Bearer TOKEN"},
    ),
    enable_remote_services=True,  # required: the audio leaves this process
)

converter = DocumentConverter(
    format_options={
        InputFormat.AUDIO: AudioFormatOption(
            pipeline_cls=AsrPipeline, pipeline_options=pipeline_options
        )
    }
)
result = converter.convert("recording.mp3")
```

The file is posted as `multipart/form-data` with `response_format=verbose_json`, and every timed segment of the answer
becomes a text item with a `TrackSource` (start and end time), like with the local backends. Words are added to each
segment when `word_timestamps=True` and the server supports `timestamp_granularities[]=word`. Settings that are specific to a
server go in `params`, which are sent as additional form fields.

Notes:

- `enable_remote_services=True` is required, as for remote picture description: without it the pipeline raises
  `OperationNotAllowed` and nothing is sent.
- FFmpeg is not needed for audio files, the server decodes them. (The video pipeline still uses it to extract the
  audio track.)
- The server must return timed segments (`verbose_json`). An answer without them fails the conversion with a message
  saying so. Speaker labels are not part of the OpenAI transcription API, so the items have no speaker.
- The request waits up to `timeout` seconds (600 by default), which long recordings may need to exceed.

### From the command line

`docling models --asr` lists the models by family, with one column for each backend. Two more columns show if the selected backend gives real time ranges (`Times`) and speaker labels (`Spk`). The output has a legend for its symbols, and `--format json` gives the same data for scripts:

```bash
docling models --asr
docling models --asr --family granite
docling models --asr --format json
```

The `docling` CLI selects any preset with `--asr-model` (values are the lower-case preset names). Audio inputs route to the ASR pipeline and video inputs to the video pipeline automatically based on file extension, so no extra flag is required to pick a pipeline — `--asr-model` controls the transcription backend for both:

```bash
# auto-selecting default
docling --to md --asr-model whisper_turbo recording.mp3

# force native Whisper
docling --to md --asr-model whisper_turbo_native recording.mp3

# WhisperS2T, distilled large-v3
docling --to md --asr-model whisper_distil_large_v3_s2t recording.mp3

# Granite Speech: mlx-audio on Apple Silicon, Transformers in other cases
docling --to md --asr-model granite_speech_5_0_470m_turboctc recording.mp3

# force Transformers for Granite Speech
docling --to md --asr-model granite_speech_5_0_470m_turboctc_transformers recording.mp3
```

See the [CLI reference](../reference/cli.md) for the complete list of options.

### Validate real model decoding

From a repository checkout, run the separate script against one short speech
fragment. It is outside pytest collection and never runs in the normal test suite.

```bash
# Select several models. Repeat --model for each preset.
uv run python scripts/validate_asr_models.py \\
  --audio tests/data/audio/sources/sample_10s.mp3 \\
  --model granite_speech_5_0_2b \\
  --model granite_speech_5_0_470m_turboctc \\
  --output /tmp/asr-validation

# List choices, or run every preset against the default ten-second fragment.
uv run python scripts/validate_asr_models.py --list-models
uv run python scripts/validate_asr_models.py --device auto
```

Each model runs in a separate process. The script checks conversion status and
requires a nonempty transcript. It saves `transcript.txt`, `transcript.md`,
`document.dclx`, `options.json`, and `run.log` in each model directory.
The shared `report.json` records the audio SHA-256, elapsed time, and failures.
It returns a nonzero exit code if any model fails, including missing dependencies
or unsupported hardware, and continues with the other models.

The default run selects every ASR preset, including native Whisper, MLX,
`mlx-audio`, and WhisperS2T variants. Install the required backends or select a subset suitable
for your machine. Models download weights unless `--artifacts-path` is set.
Use `--language` for the source language (default: English), `--revision` to
override Transformers and `mlx-audio` revisions, and `--timeout` to change the per-model limit
(default: 900 seconds, including download time).

Compare the saved transcripts to the spoken fragment to assess recognition
accuracy. Decoding success alone does not measure transcription accuracy.

## Limitations

| Limitation | Workaround |
|-----------|------------|
| No SRT subtitle output | WebVTT is supported via `doc.save_as_vtt(...)`. For SRT, use the `openai-whisper` CLI instead: `whisper audio.mp3 --output_format srt` |
| Audio-only ASR pipeline has no speaker diarization, except `GRANITE_SPEECH_4_1_2B_PLUS` | Diarization is available for video via `VideoPipelineOptions.enable_diarization` — see [Processing video](#processing-video). For other audio-only diarization, use [`pyannote-audio`](https://github.com/pyannote/pyannote-audio) as a pre- or post-processing step |
| No word-level timestamps | Not available in current export formats |
| Granite time ranges cover a full window of about 20 to 40 seconds, not a phrase | Use `GRANITE_SPEECH_4_1_2B_PLUS` or Whisper for finer timing, or reduce `max_time_chunk` |
| A Granite window can end inside a sentence, because a pause is not always a sentence boundary | Join adjacent segments before further processing |

For knowledge-retrieval use cases (RAG, search, summarization), paragraph-level Markdown is usually all you need. The limitations above matter primarily for subtitle generation workflows.

## See also

- [Minimal ASR pipeline example](../examples/minimal_asr_pipeline.py)
- [Video pipeline notebook](../examples/video_pipeline.ipynb)
- [Supported formats](supported_formats.md)
- [Chunking](../concepts/chunking.md)
- [LangChain integration](../integrations/langchain.md)
- [LlamaIndex integration](../integrations/llamaindex.md)
- [Full code repo example](https://github.com/TejasQ/example-docling-media)
