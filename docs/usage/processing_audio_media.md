# Processing audio and video

Docling converts audio and video files into a structured [`DoclingDocument`](../concepts/docling_document.md) — the same intermediate representation used for PDFs, DOCX files, and everything else. From there you can export to Markdown, JSON, HTML, or DocTags, and plug the result directly into RAG pipelines, summarizers, or search indexes.

Audio files run through the **ASR pipeline**: transcription only. Video files run through the dedicated **video pipeline**, which transcribes the audio track *and* samples representative frames, with optional speaker diarization — see [Processing video](#processing-video) below.

Under the hood, both pipelines transcribe with [OpenAI Whisper](https://github.com/openai/whisper). By default they auto-select the best backend for your hardware — `mlx-whisper` on Apple Silicon and native Whisper everywhere else — so the [basic example](#basic-usage) below needs no configuration. To change the model size, force a particular backend, or opt into the faster experimental WhisperS2T backend, see [Choosing an ASR model and backend](#choosing-an-asr-model-and-backend).

## Supported formats

| Type | Formats |
|------|---------|
| Audio | WAV, MP3, M4A, AAC, OGG, FLAC |
| Video | MP4, AVI, MOV, MKV, WEBM |

For video files, Docling extracts the audio track automatically before transcription. You don't need to run FFmpeg manually.

!!! note "ffmpeg required"
    Whisper audio decoding requires the `ffmpeg` executable to be installed and available on your `PATH`. This applies to common audio formats such as MP3, WAV, M4A, AAC, OGG, and FLAC, and to video files whose audio track and frames are extracted before transcription. Install it with your system package manager — e.g. `brew install ffmpeg` on macOS, `apt-get install ffmpeg` on Debian-based Linux, or `winget install ffmpeg` on Windows.

## Installation

The ASR pipeline is an optional extra. Install it alongside the base package:

```bash
pip install "docling[asr]"
```

Or with `uv`:

```bash
uv add "docling[asr]"
```

For video, including optional speaker diarization, install the `format-video` extra instead:

```bash
pip install "docling-slim[format-video]"
```

`format-video` already includes everything `asr` provides (transcription), plus the frame-sampling and diarization dependencies (`resemblyzer`, `soundfile`, `scikit-learn`, `librosa`). See [Processing video](#processing-video) for details.

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

`max_sampled_frames` caps the total number of frames sampled regardless of mode. Set `generate_frame_images=False` to skip frame sampling entirely and transcribe only.

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

See the [CLI reference](../reference/cli.md) for the complete flag list.

## Choosing an ASR model and backend

Docling ships three interchangeable ASR backends, all installed by the `asr` extra shown above:

| Backend | Library | Hardware | Notes |
|---------|---------|----------|-------|
| **Native Whisper** | `openai-whisper` (PyTorch) | CPU, CUDA | Default; broadest compatibility |
| **MLX Whisper** | `mlx-whisper` | Apple Silicon (MPS) | Optimized for M-series Macs |
| **WhisperS2T** | `whisper-s2t-reborn` (CTranslate2) | CPU, CUDA | Optional & experimental; batched decoding for high throughput |

### Automatic backend selection

The auto-selecting presets — `WHISPER_TINY`, `WHISPER_BASE`, `WHISPER_SMALL`, `WHISPER_MEDIUM`, `WHISPER_LARGE`, and `WHISPER_TURBO` — pick a backend for you based on the hardware they detect, in this priority order:

1. **MLX Whisper** — on Apple Silicon, when `mlx-whisper` is installed.
2. **Native Whisper** — on all other hardware.

WhisperS2T is **never** auto-selected; you opt into it explicitly (see below).

This is why the [Basic usage](#basic-usage) example needs no hardware-specific code — `asr_model_specs.WHISPER_TURBO` runs on MLX on a Mac and on native Whisper on Linux and Windows. `WHISPER_TURBO` is a good default; to change the model size, swap in another auto-selecting preset:

```python
from docling.datamodel import asr_model_specs

pipeline_options.asr_options = asr_model_specs.WHISPER_LARGE
```

### Forcing a specific backend

Each size also has explicit variants that bypass hardware detection, suffixed `_NATIVE`, `_MLX`, and `_S2T`. Use them to pin a backend regardless of platform:

```python
from docling.datamodel import asr_model_specs

# Native OpenAI Whisper (CPU / CUDA)
pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO_NATIVE

# MLX (Apple Silicon)
pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO_MLX
```

The rest of the setup — the `DocumentConverter` from [Basic usage](#basic-usage) — is unchanged.

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
    The `whisper-s2t-reborn` dependency installs only on non-Apple-Silicon platforms, so `_S2T` presets can't be used on M-series Macs — use the native or MLX backends there. On Linux with CUDA, see the [cuBLAS note](#installation) above if model loading fails.

### From the command line

The `docling` CLI selects any preset with `--asr-model` (values are the lower-case preset names). Audio inputs route to the ASR pipeline and video inputs to the video pipeline automatically based on file extension, so no extra flag is required to pick a pipeline — `--asr-model` controls the transcription backend for both:

```bash
# auto-selecting default
docling --to md --asr-model whisper_turbo recording.mp3

# force native Whisper
docling --to md --asr-model whisper_turbo_native recording.mp3

# WhisperS2T, distilled large-v3
docling --to md --asr-model whisper_distil_large_v3_s2t recording.mp3
```

See the [CLI reference](../reference/cli.md) for the complete list of `--asr-model` values.

## Limitations

| Limitation | Workaround |
|-----------|------------|
| No SRT subtitle output | WebVTT is supported via `doc.save_as_vtt(...)`. For SRT, use the `openai-whisper` CLI instead: `whisper audio.mp3 --output_format srt` |
| Audio-only ASR pipeline has no speaker diarization | Diarization is available for video via `VideoPipelineOptions.enable_diarization` — see [Processing video](#processing-video). For audio-only diarization, use [`pyannote-audio`](https://github.com/pyannote/pyannote-audio) as a pre- or post-processing step |
| No word-level timestamps | Not available in current export formats |

For knowledge-retrieval use cases (RAG, search, summarization), paragraph-level Markdown is usually all you need. The limitations above matter primarily for subtitle generation workflows.

## See also

- [Minimal ASR pipeline example](../examples/minimal_asr_pipeline.py)
- [Video pipeline notebook](../examples/video_pipeline.ipynb)
- [Supported formats](supported_formats.md)
- [Chunking](../concepts/chunking.md)
- [LangChain integration](../integrations/langchain.md)
- [LlamaIndex integration](../integrations/llamaindex.md)
- [Full code repo example](https://github.com/TejasQ/example-docling-media)
