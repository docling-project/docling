# Processing audio and video

Docling's ASR (Automatic Speech Recognition) pipeline lets you convert audio and video files into a structured [`DoclingDocument`](../concepts/docling_document.md) — the same intermediate representation used for PDFs, DOCX files, and everything else. From there you can export to Markdown, JSON, HTML, or DocTags, and plug the result directly into RAG pipelines, summarizers, or search indexes.

Under the hood, Docling uses [Whisper Turbo](https://github.com/openai/whisper) for transcription. On Apple Silicon it automatically selects `mlx-whisper` for optimized local inference; on all other hardware it falls back to native Whisper. You don't configure this — it just picks the right backend.

## Supported formats

| Type | Formats |
|------|---------|
| Audio | WAV, MP3, M4A, AAC, OGG, FLAC |
| Video | MP4, AVI, MOV |

For video files, Docling extracts the audio track automatically before transcription. You don't need to run FFmpeg manually.

!!! note "ffmpeg required"
    Some audio formats (M4A, AAC, OGG, FLAC) and all video formats require `ffmpeg` to be installed and available on your `PATH`. Install it with your system package manager — e.g. `brew install ffmpeg` on macOS or `apt-get install ffmpeg` on Debian-based Linux.

## Installation

The ASR pipeline is an optional extra. Install it alongside the base package:

```bash
pip install "docling[asr]"
```

Or with `uv`:

```bash
uv add "docling[asr]"
```

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

The same code works for video — pass an `.mp4`, `.mov`, or `.avi` path and Docling handles the rest.

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

## Customizing the ASR model

`asr_model_specs.WHISPER_TURBO` is the default and recommended starting point — it balances speed and accuracy for most use cases. To use a different model size, pass an alternative spec from `docling.datamodel.asr_model_specs`:

```python
from docling.datamodel import asr_model_specs

pipeline_options.asr_options = asr_model_specs.WHISPER_LARGE_V3
```

Available specs depend on your installed version. Check `dir(asr_model_specs)` for the full list.

## Limitations

| Limitation | Workaround |
|-----------|------------|
| No SRT/WebVTT subtitle output | Use `openai-whisper` CLI: `whisper audio.mp3 --output_format srt` |
| No speaker diarization | Use [`pyannote-audio`](https://github.com/pyannote/pyannote-audio) as a pre- or post-processing step |
| No word-level timestamps | Not available in current export formats |

For knowledge-retrieval use cases (RAG, search, summarization), paragraph-level Markdown is usually all you need. The limitations above matter primarily for subtitle generation workflows.

## See also

- [Minimal ASR pipeline example](../examples/minimal_asr_pipeline.py)
- [Supported formats](supported_formats.md)
- [Chunking](../concepts/chunking.md)
- [LangChain integration](../integrations/langchain.md)
- [LlamaIndex integration](../integrations/llamaindex.md)
- [Full code repo example](https://github.com/TejasQ/example-docling-media)
