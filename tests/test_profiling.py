"""Tests for pipeline timing instrumentation.

Timing is gated behind ``settings.debug.profile_pipeline_timings``, which is
off everywhere by default. These tests drive both states: the recorders must
be inert when profiling is off, and must enforce their interval state machine
when it is on.
"""

import pytest

from docling.datamodel.settings import settings
from docling.utils.profiling import (
    ProfilingItem,
    ProfilingScope,
    TimeIntervalRecorder,
    TimeRecorder,
)


class _StubConversionResult:
    """Minimal stand-in exposing the only attribute the recorders touch."""

    def __init__(self) -> None:
        self.timings: dict[str, ProfilingItem] = {}


@pytest.fixture
def profiling_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.debug, "profile_pipeline_timings", True)


@pytest.fixture
def profiling_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.debug, "profile_pipeline_timings", False)


def test_profiling_item_statistics():
    item = ProfilingItem(scope=ProfilingScope.PAGE, times=[1.0, 2.0, 3.0, 4.0])

    assert item.total() == pytest.approx(10.0)
    assert item.avg() == pytest.approx(2.5)
    assert item.mean() == pytest.approx(2.5)
    assert item.std() == pytest.approx(1.1180339887)
    assert item.percentile(50) == pytest.approx(2.5)
    assert item.percentile(0) == pytest.approx(1.0)
    assert item.percentile(100) == pytest.approx(4.0)


def test_time_recorder_records_one_sample_per_scope(profiling_enabled):
    conv_res = _StubConversionResult()

    with TimeRecorder(conv_res, "layout", scope=ProfilingScope.PAGE):
        pass

    item = conv_res.timings["layout"]
    assert item.scope == ProfilingScope.PAGE
    assert item.count == 1
    assert len(item.times) == 1
    assert len(item.start_timestamps) == 1
    assert item.times[0] >= 0.0


def test_time_recorder_accumulates_across_reuse_of_one_key(profiling_enabled):
    conv_res = _StubConversionResult()

    for _ in range(3):
        with TimeRecorder(conv_res, "ocr"):
            pass

    item = conv_res.timings["ocr"]
    assert item.count == 3
    assert len(item.times) == 3
    # The item is created once and reused, not replaced per invocation.
    assert len(item.start_timestamps) == 3


def test_time_recorder_defaults_to_page_scope(profiling_enabled):
    conv_res = _StubConversionResult()

    with TimeRecorder(conv_res, "assemble"):
        pass

    assert conv_res.timings["assemble"].scope == ProfilingScope.PAGE


def test_time_recorder_is_inert_when_profiling_is_disabled(profiling_disabled):
    conv_res = _StubConversionResult()

    with TimeRecorder(conv_res, "layout"):
        pass

    assert conv_res.timings == {}


def test_interval_recorder_banks_disjoint_intervals_into_one_sample(profiling_enabled):
    conv_res = _StubConversionResult()

    recorder = TimeIntervalRecorder(conv_res, "vlm", scope=ProfilingScope.DOCUMENT)
    recorder.resume()
    recorder.pause()
    recorder.resume()
    recorder.pause()
    recorder.close()

    item = conv_res.timings["vlm"]
    assert item.scope == ProfilingScope.DOCUMENT
    # Two intervals collapse into a single sample, unlike TimeRecorder.
    assert item.count == 1
    assert len(item.times) == 1
    assert len(item.start_timestamps) == 1


def test_interval_recorder_add_contributes_to_the_total(profiling_enabled):
    conv_res = _StubConversionResult()

    recorder = TimeIntervalRecorder(conv_res, "remote")
    recorder.add(1.5)
    recorder.add(2.5)
    recorder.close()

    assert conv_res.timings["remote"].times[0] == pytest.approx(4.0)


def test_interval_recorder_close_without_any_interval_records_zero(profiling_enabled):
    conv_res = _StubConversionResult()

    recorder = TimeIntervalRecorder(conv_res, "idle")
    recorder.close()

    assert conv_res.timings["idle"].times == [0.0]
    assert conv_res.timings["idle"].count == 1


def test_interval_recorder_rejects_double_resume(profiling_enabled):
    recorder = TimeIntervalRecorder(_StubConversionResult(), "vlm")
    recorder.resume()

    with pytest.raises(RuntimeError, match="already running"):
        recorder.resume()


def test_interval_recorder_rejects_pause_without_resume(profiling_enabled):
    recorder = TimeIntervalRecorder(_StubConversionResult(), "vlm")

    with pytest.raises(RuntimeError, match="without a matching resume"):
        recorder.pause()


def test_interval_recorder_rejects_close_while_running(profiling_enabled):
    recorder = TimeIntervalRecorder(_StubConversionResult(), "vlm")
    recorder.resume()

    with pytest.raises(RuntimeError, match="still running"):
        recorder.close()


@pytest.mark.parametrize("operation", ["resume", "pause", "add", "close"])
def test_interval_recorder_rejects_use_after_close(profiling_enabled, operation):
    recorder = TimeIntervalRecorder(_StubConversionResult(), "vlm")
    recorder.close()

    with pytest.raises(RuntimeError, match="closed"):
        if operation == "add":
            recorder.add(1.0)
        else:
            getattr(recorder, operation)()


def test_interval_recorder_is_inert_when_profiling_is_disabled(profiling_disabled):
    conv_res = _StubConversionResult()

    recorder = TimeIntervalRecorder(conv_res, "vlm")
    # Every call is a no-op, including sequences that would otherwise raise.
    recorder.pause()
    recorder.resume()
    recorder.resume()
    recorder.add(5.0)
    recorder.close()
    recorder.close()

    assert conv_res.timings == {}
