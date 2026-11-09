# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the API-based ASR backend (an OpenAI-compatible transcription server).

The server is a stand-in running in a thread: no speech model is downloaded or run.
"""

import json
import threading
from email import message_from_bytes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from docling_core.types.doc import TrackSource

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions, VideoPipelineOptions
from docling.datamodel.pipeline_options_asr_model import ApiAsrOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.exceptions import OperationNotAllowed
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.pipeline.asr_transcriber import _ApiAsrModel, _AsrModelFactory

SAMPLE_AUDIO = Path("./tests/data/audio/sources/sample_10s.mp3")

VERBOSE_JSON = {
    "task": "transcribe",
    "language": "english",
    "duration": 9.0,
    "text": "Pump two is at one point eight bar. Go for the switch-over.",
    "segments": [
        {
            "id": 0,
            "start": 0.0,
            "end": 3.5,
            "text": " Pump two is at one point eight bar.",
        },
        {"id": 1, "start": 3.5, "end": 9.0, "text": " Go for the switch-over."},
    ],
}


class _FakeTranscriptionServer:
    """Answers POST /v1/audio/transcriptions and records what it was sent."""

    def __init__(self, status: int = 200, body: object = VERBOSE_JSON):
        outer = self
        self.status = status
        self.body = body
        self.requests: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                message = message_from_bytes(
                    b"Content-Type: "
                    + self.headers["Content-Type"].encode()
                    + b"\r\n\r\n"
                    + raw
                )
                fields: dict[str, list[str]] = {}
                upload = {}
                for part in message.get_payload():
                    name = part.get_param("name", header="content-disposition")
                    if part.get_filename():
                        upload = {
                            "filename": part.get_filename(),
                            "content_type": part.get_content_type(),
                            "size": len(part.get_payload(decode=True)),
                        }
                    else:
                        fields.setdefault(name, []).append(
                            part.get_payload(decode=True).decode()
                        )
                outer.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "fields": fields,
                        "file": upload,
                    }
                )
                payload = (
                    outer.body
                    if isinstance(outer.body, str)
                    else json.dumps(outer.body)
                ).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):  # keep the test output quiet
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = (
            f"http://127.0.0.1:{self._server.server_address[1]}/v1/audio/transcriptions"
        )
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def server():
    s = _FakeTranscriptionServer()
    yield s
    s.close()


def _converter(options: ApiAsrOptions) -> DocumentConverter:
    pipeline_options = AsrPipelineOptions(
        asr_options=options, enable_remote_services=True
    )
    return DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline, pipeline_options=pipeline_options
            )
        }
    )


class TestApiAsrOptions:
    def test_defaults(self):
        options = ApiAsrOptions()

        assert options.kind == "api_model_options"
        assert str(options.url) == "http://localhost:8000/v1/audio/transcriptions"
        assert options.model is None
        assert options.language is None
        assert options.initial_prompt is None
        assert options.word_timestamps is False
        assert options.headers == {}
        assert options.params == {}

    def test_accepted_by_the_audio_and_video_pipeline_options(self):
        options = ApiAsrOptions(model="whisper-1")

        assert AsrPipelineOptions(asr_options=options).asr_options is options
        assert VideoPipelineOptions(asr_options=options).asr_options is options

    def test_remote_services_must_be_enabled(self):
        with pytest.raises(OperationNotAllowed, match="enable_remote_services"):
            AsrPipeline(AsrPipelineOptions(asr_options=ApiAsrOptions()))

        pipeline = AsrPipeline(
            AsrPipelineOptions(asr_options=ApiAsrOptions(), enable_remote_services=True)
        )
        assert isinstance(pipeline._model, _ApiAsrModel)

    def test_factory_refuses_without_the_switch(self):
        with pytest.raises(OperationNotAllowed):
            _AsrModelFactory.create(
                asr_options=ApiAsrOptions(),
                artifacts_path=None,
                accelerator_options=AsrPipelineOptions().accelerator_options,
            )


class TestApiAsrConversion:
    def test_audio_file_becomes_timed_text_items(self, server):
        converter = _converter(ApiAsrOptions(url=server.url))

        result = converter.convert(SAMPLE_AUDIO)

        assert result.status == ConversionStatus.SUCCESS
        texts = result.document.texts
        assert [t.text for t in texts] == [
            "Pump two is at one point eight bar.",
            "Go for the switch-over.",
        ]
        tracks = [t.source[0] for t in texts]
        assert all(isinstance(track, TrackSource) for track in tracks)
        assert [(track.start_time, track.end_time) for track in tracks] == [
            (0.0, 3.5),
            (3.5, 9.0),
        ]

    def test_request_carries_the_audio_and_the_options(self, server):
        options = ApiAsrOptions(
            url=server.url,
            headers={"Authorization": "Bearer secret"},
            model="Systran/faster-whisper-small",
            language="en",
            initial_prompt="Columbus loop: COL FLIGHT, COL SYSTEMS",
            params={"temperature": 0.0, "vad_filter": True, "response_format": "text"},
        )

        _converter(options).convert(SAMPLE_AUDIO)

        (request,) = server.requests
        assert request["path"] == "/v1/audio/transcriptions"
        assert request["authorization"] == "Bearer secret"
        assert request["file"]["filename"] == "sample_10s.mp3"
        assert request["file"]["content_type"] == "audio/mp3"
        assert request["file"]["size"] == SAMPLE_AUDIO.stat().st_size
        assert request["fields"] == {
            "response_format": ["verbose_json"],
            "model": ["Systran/faster-whisper-small"],
            "language": ["en"],
            "prompt": ["Columbus loop: COL FLIGHT, COL SYSTEMS"],
            "temperature": ["0.0"],
            "vad_filter": ["true"],
        }

    def test_nothing_optional_is_sent_when_it_is_not_set(self, server):
        _converter(ApiAsrOptions(url=server.url)).convert(SAMPLE_AUDIO)

        assert server.requests[0]["fields"] == {"response_format": ["verbose_json"]}

    def test_word_timestamps_are_requested_and_given_to_their_segment(self, server):
        server.body = {
            **VERBOSE_JSON,
            "words": [
                {"word": "Pump", "start": 0.0, "end": 0.4},
                {"word": "two", "start": 0.4, "end": 0.7},
                {"word": "Go", "start": 3.6, "end": 3.9},
            ],
        }
        model = _ApiAsrModel(ApiAsrOptions(url=server.url, word_timestamps=True))

        conversation = model.transcribe(SAMPLE_AUDIO)

        assert server.requests[0]["fields"]["timestamp_granularities[]"] == [
            "segment",
            "word",
        ]
        assert [[w.text for w in item.words or []] for item in conversation] == [
            ["Pump", "two"],
            ["Go"],
        ]

    def test_words_inside_the_segments_are_taken_as_they_are(self, server):
        server.body = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": " Roger.",
                    "words": [{"word": "Roger", "start": 0.1, "end": 0.9}],
                }
            ]
        }
        model = _ApiAsrModel(ApiAsrOptions(url=server.url, word_timestamps=True))

        (item,) = model.transcribe(SAMPLE_AUDIO)

        assert item.text == "Roger."
        assert [(w.text, w.start_time, w.end_time) for w in item.words or []] == [
            ("Roger", 0.1, 0.9)
        ]

    def test_a_server_error_fails_the_conversion_with_its_reason(self):
        failing = _FakeTranscriptionServer(
            status=503, body={"detail": "model is loading"}
        )
        try:
            result = _converter(ApiAsrOptions(url=failing.url)).convert(
                SAMPLE_AUDIO, raises_on_error=False
            )
        finally:
            failing.close()

        assert result.status == ConversionStatus.FAILURE
        message = " ".join(error.error_message for error in result.errors)
        assert "HTTP 503" in message and "model is loading" in message

    def test_an_answer_without_segments_is_a_failure_that_says_what_is_needed(self):
        plain = _FakeTranscriptionServer(body={"text": "no timings"})
        try:
            result = _converter(ApiAsrOptions(url=plain.url)).convert(
                SAMPLE_AUDIO, raises_on_error=False
            )
        finally:
            plain.close()

        assert result.status == ConversionStatus.FAILURE
        assert "verbose_json" in result.errors[0].error_message

    def test_an_answer_that_is_not_json_is_a_failure(self):
        html = _FakeTranscriptionServer(body="<html>gateway</html>")
        try:
            result = _converter(ApiAsrOptions(url=html.url)).convert(
                SAMPLE_AUDIO, raises_on_error=False
            )
        finally:
            html.close()

        assert result.status == ConversionStatus.FAILURE
        assert "JSON" in result.errors[0].error_message
