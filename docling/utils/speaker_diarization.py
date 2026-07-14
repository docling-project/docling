"""Speaker diarization using Resemblyzer embedding-based clustering.

Assigns speaker labels to transcript segments by:
1. Encoding sliding windows of audio into speaker embedding vectors
2. Estimating the optimal number of speakers via silhouette score
3. Clustering embeddings into speaker groups
4. Mapping each transcript segment to the dominant speaker in its time window
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

_MIN_SPEAKERS = 2
"""Minimum number of speakers to consider in clustering."""
_MAX_SPEAKERS = 8
"""Maximum number of speakers to consider in clustering."""
_WINDOW_STEP = 0.5  # seconds between embedding windows
"""Window step in seconds; 0.5 provides fine-grained speaker boundary detection without excessive computational overhead."""


@dataclass
class SpeakerSegment:
    """A time segment attributed to a single speaker."""

    start_time: float
    end_time: float
    speaker: str


@dataclass
class DiarizationResult:
    """Output of speaker diarization."""

    segments: list[SpeakerSegment] = field(default_factory=list)
    num_speakers: int = 0
    speaker_ids: list[str] = field(default_factory=list)


def _estimate_num_speakers(embeddings: np.ndarray) -> int:
    """Estimate optimal speaker count via silhouette score."""
    from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-untyped]
    from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]

    best_n, best_score = _MIN_SPEAKERS, -1.0
    for n in range(_MIN_SPEAKERS, min(_MAX_SPEAKERS + 1, len(embeddings))):
        labels = AgglomerativeClustering(n_clusters=n).fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        _log.debug("N=%d silhouette=%.4f", n, score)
        if score > best_score:
            best_score = score
            best_n = n
    _log.info("Estimated %d speakers (silhouette=%.4f)", best_n, best_score)
    return best_n


def diarize(
    wav_path: Path,
    num_speakers: int | None = None,
) -> DiarizationResult:
    """Run speaker diarization on a WAV file.

    Args:
        wav_path: Path to a 16kHz mono WAV file.
        num_speakers: Number of speakers. None = auto-detect.

    Returns:
        Per-segment speaker labels.
    """
    try:
        import librosa
        import soundfile as sf
        from resemblyzer import VoiceEncoder
        from resemblyzer.audio import (
            audio_norm_target_dBFS,
            normalize_volume,
            sampling_rate as _RESEMBLYZER_SR,
        )
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        _log.warning(
            "Speaker diarization requires resemblyzer, soundfile, "
            "scikit-learn, and librosa. Speaker diarization disabled. "
            "Install with: pip install 'docling[format-video]'"
        )
        return DiarizationResult()

    _log.info("Loading audio for diarization: %s", wav_path)

    raw, file_sr = sf.read(str(wav_path), dtype="float32")
    if raw.ndim > 1:  # collapse stereo to mono
        raw = raw.mean(axis=1)
    if file_sr != _RESEMBLYZER_SR:
        raw = librosa.resample(raw, orig_sr=file_sr, target_sr=_RESEMBLYZER_SR)
    # Normalize volume the way resemblyzer.preprocess_wav does, but WITHOUT its
    # silence trimming. Trimming compresses the timeline (e.g. 300s -> 253s),
    # so diarization timestamps no longer line up with the ASR transcript
    # timeline: segments past the trimmed length get no speaker and earlier ones
    # can be misattributed. Keeping the full waveform preserves 1:1 alignment.
    wav = normalize_volume(raw, audio_norm_target_dBFS, increase_only=True)
    if len(wav) == 0:
        _log.warning("Empty audio — skipping diarization")
        return DiarizationResult()

    encoder = VoiceEncoder(device="cpu")

    # Build per-window timestamps and embeddings
    sr = 16000
    window_samples = int(sr * 1.5)  # ~1.5s windows at 16kHz
    step_samples = int(_WINDOW_STEP * sr)

    timestamps: list[float] = []
    wav_splits: list[np.ndarray] = []

    i = 0
    while i + window_samples <= len(wav):
        timestamps.append(i / sr)
        wav_splits.append(wav[i : i + window_samples])
        i += step_samples

    if not wav_splits:
        _log.warning("Audio too short for diarization")
        return DiarizationResult()

    _log.info("Encoding %d audio windows", len(wav_splits))
    # ASR (Whisper) earlier in the pipeline raises torch's thread count to the
    # full core count. Encoding is thousands of tiny sequential forward passes,
    # where fanning each one across every core costs far more in thread dispatch
    # than it saves (observed ~12x slowdown). Cap threads while encoding, then
    # restore the previous setting.
    import torch

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, prev_threads))
    try:
        embeddings = np.array([encoder.embed_utterance(w) for w in wav_splits])
    finally:
        torch.set_num_threads(prev_threads)

    # Determine number of speakers
    n = num_speakers if num_speakers is not None else _estimate_num_speakers(embeddings)

    # Cluster embeddings
    labels = AgglomerativeClustering(n_clusters=n).fit_predict(embeddings)
    speaker_ids = [f"SPEAKER_{i:02d}" for i in range(n)]

    # Build continuous speaker segments by merging consecutive same-speaker windows
    segments: list[SpeakerSegment] = []
    if len(timestamps) > 0:
        cur_speaker = speaker_ids[labels[0]]
        cur_start = timestamps[0]
        cur_end = timestamps[0] + _WINDOW_STEP

        for ts, label in zip(timestamps[1:], labels[1:]):
            spk = speaker_ids[label]
            if spk == cur_speaker:
                cur_end = ts + _WINDOW_STEP
            else:
                segments.append(SpeakerSegment(cur_start, cur_end, cur_speaker))
                cur_speaker = spk
                cur_start = ts
                cur_end = ts + _WINDOW_STEP

        # Extend last segment to end of audio
        segments.append(SpeakerSegment(cur_start, len(wav) / sr, cur_speaker))

    return DiarizationResult(
        segments=segments,
        num_speakers=n,
        speaker_ids=speaker_ids,
    )


def assign_speakers(
    transcript_items: list,
    diarization: DiarizationResult,
) -> list:
    """Assign speaker labels to transcript ConversationItems.

    For each transcript segment, find the diarization segment with the
    maximum time overlap and assign its speaker label.

    Args:
        transcript_items: List of ConversationItem from ASR transcriber.
        diarization: Result from diarize().

    Returns:
        The same list with .speaker set on each item.
    """
    if not diarization.segments:
        return transcript_items

    for item in transcript_items:
        start = item.start_time or 0.0
        end = item.end_time or start

        best_speaker = None
        best_overlap = 0.0

        for seg in diarization.segments:
            overlap = max(0.0, min(end, seg.end_time) - max(start, seg.start_time))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg.speaker

        if best_speaker:
            item.speaker = best_speaker

    return transcript_items
