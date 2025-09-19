# Assisted by watsonx Code Assistant

from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument
from pydantic import ValidationError

from docling.backend.webvtt_backend import (
    _WebVTTCueItalicSpan,
    _WebVTTCueTextSpan,
    _WebVTTCueTimings,
    _WebVTTCueVoiceSpan,
    _WebVTTFile,
    _WebVTTTimestamp,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def test_vtt_cue_commponents():
    """Test WebVTT components."""
    valid_timestamps = [
        "00:01:02.345",
        "12:34:56.789",
        "02:34.567",
        "00:00:00.000",
    ]
    valid_total_seconds = [
        1 * 60 + 2.345,
        12 * 3600 + 34 * 60 + 56.789,
        2 * 60 + 34.567,
        0.0,
    ]
    for idx, ts in enumerate(valid_timestamps):
        model = _WebVTTTimestamp(raw=ts)
        assert model.seconds == valid_total_seconds[idx]

    """Test invalid WebVTT timestamps."""
    invalid_timestamps = [
        "00:60:02.345",  # minutes > 59
        "00:01:60.345",  # seconds > 59
        "00:01:02.1000",  # milliseconds > 999
        "01:02:03",  # missing milliseconds
        "01:02",  # missing milliseconds
        ":01:02.345",  # extra : for missing hours
        "abc:01:02.345",  # invalid format
    ]
    for ts in invalid_timestamps:
        with pytest.raises(ValidationError):
            _WebVTTTimestamp(raw=ts)

    """Test the timestamp __str__ method."""
    model = _WebVTTTimestamp(raw="00:01:02.345")
    assert str(model) == "00:01:02.345"

    """Test valid cue timings."""
    start = _WebVTTTimestamp(raw="00:10.005")
    end = _WebVTTTimestamp(raw="00:14.007")
    cue_timings = _WebVTTCueTimings(start=start, end=end)
    assert cue_timings.start == start
    assert cue_timings.end == end
    assert str(cue_timings) == "00:10.005 --> 00:14.007"

    """Test invalid cue timings with end timestamp before start."""
    start = _WebVTTTimestamp(raw="00:10.700")
    end = _WebVTTTimestamp(raw="00:10.500")
    with pytest.raises(ValidationError) as excinfo:
        _WebVTTCueTimings(start=start, end=end)
    assert "End timestamp must be greater than start timestamp" in str(excinfo.value)

    """Test invalid cue timings with missing end."""
    start = _WebVTTTimestamp(raw="00:10.500")
    with pytest.raises(ValidationError) as excinfo:
        _WebVTTCueTimings(start=start)
    assert "Field required" in str(excinfo.value)

    """Test invalid cue timings with missing start."""
    end = _WebVTTTimestamp(raw="00:10.500")
    with pytest.raises(ValidationError) as excinfo:
        _WebVTTCueTimings(end=end)
    assert "Field required" in str(excinfo.value)

    """Test with valid text."""
    valid_text = "This is a valid cue text span."
    span = _WebVTTCueTextSpan(text=valid_text)
    assert span.text == valid_text
    assert str(span) == valid_text

    """Test with text containing newline characters."""
    invalid_text = "This cue text span\ncontains a newline."
    with pytest.raises(ValidationError):
        _WebVTTCueTextSpan(text=invalid_text)

    """Test with text containing ampersand."""
    invalid_text = "This cue text span contains &."
    with pytest.raises(ValidationError):
        _WebVTTCueTextSpan(text=invalid_text)

    """Test with text containing less-than sign."""
    invalid_text = "This cue text span contains <."
    with pytest.raises(ValidationError):
        _WebVTTCueTextSpan(text=invalid_text)

    """Test with empty text."""
    with pytest.raises(ValidationError):
        _WebVTTCueTextSpan(text="")

    """Test that annotation validation works correctly."""
    valid_annotation = "valid-annotation"
    invalid_annotation = "invalid\nannotation"
    with pytest.raises(ValidationError):
        _WebVTTCueVoiceSpan(annotation=invalid_annotation)
    assert _WebVTTCueVoiceSpan(annotation=valid_annotation)

    """Test that classes validation works correctly."""
    annotation = "speaker name"
    valid_classes = ["class1", "class2"]
    invalid_classes = ["class\nwith\nnewlines", ""]
    with pytest.raises(ValidationError):
        _WebVTTCueVoiceSpan(annotation=annotation, classes=invalid_classes)
    assert _WebVTTCueVoiceSpan(annotation=annotation, classes=valid_classes)

    """Test that components validation works correctly."""
    annotation = "speaker name"
    valid_components = [_WebVTTCueTextSpan(text="random text")]
    invalid_components = [123, "not a component"]
    with pytest.raises(ValidationError):
        _WebVTTCueVoiceSpan(annotation=annotation, components=invalid_components)
    assert _WebVTTCueVoiceSpan(annotation=annotation, components=valid_components)

    """Test valid cue voice spans."""
    cue_span = _WebVTTCueVoiceSpan(
        annotation="speaker",
        classes=["loud", "clear"],
        components=[_WebVTTCueTextSpan(text="random text")],
    )

    expected_str = "<v.loud.clear speaker>random text</v>"
    assert str(cue_span) == expected_str

    cue_span = _WebVTTCueVoiceSpan(
        annotation="speaker",
        components=[_WebVTTCueTextSpan(text="random text")],
    )
    expected_str = "<v speaker>random text</v>"
    assert str(cue_span) == expected_str


def test_webvtt_file():
    """Test WebVTT files."""
    with open("./tests/data/webvtt/webvtt_example_01.vtt", encoding="utf-8") as f:
        content = f.read()
        vtt = _WebVTTFile.parse(content)
    assert len(vtt) == 13
    block = vtt.cue_blocks[11]
    assert str(block.timings) == "00:32.500 --> 00:33.500"
    assert len(block.payload) == 1
    cue_span = block.payload[0]
    assert isinstance(cue_span, _WebVTTCueVoiceSpan)
    assert cue_span.annotation == "Neil deGrasse Tyson"
    assert not cue_span.classes
    assert len(cue_span.components) == 1
    comp = cue_span.components[0]
    assert isinstance(comp, _WebVTTCueItalicSpan)
    assert len(comp.components) == 1
    comp2 = comp.components[0]
    assert isinstance(comp2, _WebVTTCueTextSpan)
    assert comp2.text == "Laughs"

    with open("./tests/data/webvtt/webvtt_example_02.vtt", encoding="utf-8") as f:
        content = f.read()
        vtt = _WebVTTFile.parse(content)
    assert len(vtt) == 4
    reverse = "WEBVTT\n\n" + "\n\n".join([str(block) for block in vtt.cue_blocks])
    assert content == reverse

    with open("./tests/data/webvtt/webvtt_example_03.vtt", encoding="utf-8") as f:
        content = f.read()
        vtt = _WebVTTFile.parse(content)
    assert len(vtt) == 13
    for block in vtt:
        assert block.identifier
    block = vtt.cue_blocks[0]
    assert block.identifier == "62357a1d-d250-41d5-a1cf-6cc0eeceffcc/15-0"
    assert str(block.timings) == "00:00:04.963 --> 00:00:08.571"
    assert len(block.payload) == 1
    assert isinstance(block.payload[0], _WebVTTCueVoiceSpan)
    block = vtt.cue_blocks[2]
    assert isinstance(cue_span, _WebVTTCueVoiceSpan)
    assert block.identifier == "62357a1d-d250-41d5-a1cf-6cc0eeceffcc/16-0"
    assert str(block.timings) == "00:00:10.683 --> 00:00:11.563"
    assert len(block.payload) == 1
    assert isinstance(block.payload[0], _WebVTTCueTextSpan)
    assert block.payload[0].text == "Good."


def test_e2e_vtt_conversions():
    directory = Path("./tests/data/webvtt/")
    vtt_paths = sorted(directory.rglob("*.vtt"))
    converter = DocumentConverter(allowed_formats=[InputFormat.VTT])

    for vtt in vtt_paths:
        gt_path = vtt.parent.parent / "groundtruth" / "docling_v2" / vtt.name

        conv_result: ConversionResult = converter.convert(vtt)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
            "export to md"
        )

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE)
