"""Align diarization speaker turns onto ASR words and segments.

Pure: given a Transcript (global word/segment timestamps) and a list of speaker
turns, assign each word to the turn it overlaps most, and each segment to the
majority speaker of its words. No model, no IO; unit-tested in isolation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

from parascribe.stitch import Transcript


@dataclass(frozen=True)
class SpeakerTurn:
    """A diarized speaker turn: ``speaker`` is an opaque label like 'SPEAKER_00'."""

    start: float
    end: float
    speaker: str


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _best_speaker(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    """The speaker whose turn overlaps [start, end] most (None if no overlap)."""
    best: str | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = _overlap(start, end, turn.start, turn.end)
        if overlap > best_overlap:
            best_overlap = overlap
            best = turn.speaker
    return best


def apply_speakers(transcript: Transcript, turns: list[SpeakerTurn]) -> Transcript:
    """Return a copy of ``transcript`` with speaker labels assigned.

    Each word takes the speaker of its max-overlap turn. Each segment takes the
    majority speaker of the words inside it; if it has no labelled words, it falls
    back to its own max-overlap turn. Ties go to the most common first-seen label.
    With no turns, the transcript is returned unchanged.
    """
    if not turns:
        return transcript

    # Sort by start so that on equal overlap the earliest turn wins the tie,
    # independent of the order the caller passed turns in.
    turns = sorted(turns, key=lambda t: t.start)

    new_words = [
        replace(word, speaker=_best_speaker(word.start, word.end, turns))
        for word in transcript.words
    ]

    # Group words to segments in one pass: both are sorted by start and segments
    # are non-overlapping, so a single advancing segment pointer suffices.
    segments = transcript.segments
    seg_labels: list[list[str]] = [[] for _ in segments]
    si = 0
    for word in new_words:
        midpoint = (word.start + word.end) / 2
        while si < len(segments) and midpoint > segments[si].end:
            si += 1
        if si >= len(segments):
            break
        if midpoint >= segments[si].start and word.speaker is not None:
            seg_labels[si].append(word.speaker)

    new_segments = []
    for segment, labels in zip(segments, seg_labels, strict=True):
        if labels:
            speaker: str | None = Counter(labels).most_common(1)[0][0]
        else:
            speaker = _best_speaker(segment.start, segment.end, turns)
        new_segments.append(replace(segment, speaker=speaker))

    return replace(transcript, segments=new_segments, words=new_words)
