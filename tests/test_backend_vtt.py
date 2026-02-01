import warnings
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument, GroupItem, TextItem

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult, _DocumentConversionInput
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def test_e2e_vtt_conversions():
    directory = Path("./tests/data/webvtt/")
    vtt_paths = sorted(directory.rglob("*.vtt"))
    converter = DocumentConverter(allowed_formats=[InputFormat.VTT])

    for vtt in vtt_paths:
        gt_path = vtt.parent.parent / "groundtruth" / "docling_v2" / vtt.name

        conv_result: ConversionResult = converter.convert(vtt)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown(escape_html=False)
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


def _create_vtt_stream(content: str) -> DocumentStream:
    stream = DocumentStream(name="test.vtt", stream=BytesIO(content.strip().encode()))
    dci = _DocumentConversionInput(path_or_stream_iterator=[])
    assert dci._guess_format(stream) == InputFormat.VTT

    return stream


def _process_vtt_doc(doc: DoclingDocument) -> str:
    text: str = ""
    for item in doc.texts:
        if (
            isinstance(item, TextItem)
            and item.source
            and item.source[0].kind == "track"
        ):
            parent = item.parent.resolve(doc)
            if parent and isinstance(parent, GroupItem):
                text += " "
            text += item.text

    return text.strip()


@pytest.fixture(scope="module")
def converter() -> DocumentConverter:
    return DocumentConverter()


def test_simple_two_cues_basic(converter):
    vtt = """
WEBVTT

00:00:00.000 --> 00:00:02.000
Hello world!

00:00:02.500 --> 00:00:04.000
Second cue.
"""
    stream = _create_vtt_stream(vtt)
    doc = converter.convert(stream).document

    expected = "Hello world! Second cue."
    assert _process_vtt_doc(doc) == expected


def test_cue_ids_present_are_ignored_in_output(converter):
    vtt = """
WEBVTT

1
00:00:00.000 --> 00:00:01.000
First with ID.

2
00:00:01.250 --> 00:00:02.000
Second with ID.
"""
    stream = _create_vtt_stream(vtt)
    doc = converter.convert(stream).document

    expected = "First with ID. Second with ID."
    assert _process_vtt_doc(doc) == expected


def test_multi_line_cue_text_preserved(converter):
    vtt = """
WEBVTT

00:00:00.000 --> 00:00:03.000
This is line one.
This is line two.

00:00:03.500 --> 00:00:05.000
Another cue line one.
Another cue line two.
"""
    stream = _create_vtt_stream(vtt)
    doc = converter.convert(stream).document

    expected = "This is line one. This is line two. Another cue line one. Another cue line two."
    assert _process_vtt_doc(doc) == expected


def test_styling_and_voice_tags_stripped(converter):
    vtt = """
WEBVTT

00:00:00.000 --> 00:00:02.000
<v Roger><b>Hello</b> <i>there</i><u>!</u></v>

00:00:02.200 --> 00:00:04.000
<c.red>Styled</c> and <v Ann>voiced</v> text.
"""
    stream = _create_vtt_stream(vtt)
    doc = converter.convert(stream).document

    # Expect tags removed but inner text retained, spacing preserved.
    # expected = "Hello there! Styled and voiced text."
    # TODO: temporary ground truth (issue docling-project/docling-core/#371)
    expected = "Hello   there ! Styled  and  voiced  text."
    assert _process_vtt_doc(doc) == expected


def test_blank_cue_contributes_no_text(converter):
    # First cue has text; second cue is intentionally blank (zero transcript lines).
    vtt = """
WEBVTT

00:00:00.000 --> 00:00:02.000
Visible text.

00:00:02.500 --> 00:00:04.000

"""
    stream = _create_vtt_stream(vtt)
    doc = converter.convert(stream).document

    expected = "Visible text."
    assert _process_vtt_doc(doc) == expected


def test_note_blocks_are_ignored(converter):
    vtt = """
WEBVTT


NOTE This is a file-level note
It can span multiple lines.


00:00:00.000 --> 00:00:02.000
First cue text.


NOTE Another note between cues


00:00:02.500 --> 00:00:04.000
Second cue text.
"""
    stream = _create_vtt_stream(vtt)
    doc = converter.convert(stream).document

    expected = "First cue text. Second cue text."
    assert _process_vtt_doc(doc) == expected


def test_region_block_ignored_but_region_reference_ok(converter):
    vtt = """
WEBVTT

REGION
id:top
width:40%
lines:3

00:00:00.000 --> 00:00:02.000 region:top line:90% position:50% size:35% align:start
Top region text.

00:00:02.500 --> 00:00:04.000
Normal region text.
"""
    stream = _create_vtt_stream(vtt)
    doc = converter.convert(stream).document

    expected = "Top region text. Normal region text."
    assert _process_vtt_doc(doc) == expected


def test_varied_timestamp_formats_and_settings_ignored(converter):
    # First cue uses MM:SS.mmm; second uses HH:MM:SS.mmm and includes settings.
    vtt = """
WEBVTT

00:01.000 --> 00:03.000
Under one minute format.

01:00:00.000 --> 01:00:02.000 line:0 position:10% align:end
Hour format with settings.
"""
    stream = _create_vtt_stream(vtt)
    doc = converter.convert(stream).document

    expected = "Under one minute format. Hour format with settings."
    assert _process_vtt_doc(doc) == expected


def test_cue_ids_plus_multiline_with_voice_and_style(converter):
    # Mix multiple concepts: cue IDs, multi-line text, voice tags, style tags.
    vtt = """
WEBVTT



intro
00:00:00.000 --> 00:00:02.000
<v Narrator><i>Welcome</i> to the show.</v>
<b>Enjoy</b> your time.



outro
00:00:02.500 --> 00:00:04.000
<v Host>Goodbye</v>, see you <u>soon</u>.
"""
    stream = _create_vtt_stream(vtt)
    doc = converter.convert(stream).document

    # expected = "Welcome to the show. Enjoy your time. Goodbye, see you soon."
    # TODO: temporary ground truth (issue docling-project/docling-core/#371)
    expected = "Welcome  to the show. Enjoy  your time. Goodbye , see you  soon ."
    assert _process_vtt_doc(doc) == expected


def test_style_blocks_and_note_between_styles_are_ignored(converter):
    vtt = """
WEBVTT

STYLE
::cue {
  background-image: linear-gradient(to bottom, dimgray, lightgray);
  color: papayawhip;
}
/* Style blocks cannot use blank lines nor "dash dash greater than" */

NOTE comment blocks can be used between style blocks.

STYLE
::cue(b) {
    color: peachpuff;
}

hello
00:00:00.000 --> 00:00:10.000
Hello <b>world</b>.
"""
    stream = _create_vtt_stream(vtt)
    with warnings.catch_warnings():
        # STYLE and NOTE blocks should be ignored without warnings
        warnings.simplefilter("error")
        doc = converter.convert(stream).document

    # expected = "Hello world."
    # TODO: temporary ground truth (issue docling-project/docling-core/#371)
    expected = "Hello  world ."
    assert _process_vtt_doc(doc) == expected
