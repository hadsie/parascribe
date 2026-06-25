"""Unit tests for diarization-to-ASR speaker alignment."""

from __future__ import annotations

from parascribe.align import SpeakerTurn, apply_speakers
from parascribe.stitch import Segment, Transcript, Word


def transcript(segments, words):
    return Transcript(text="", language="en", duration=10.0, segments=segments, words=words)


class TestApplySpeakers:
    def test_word_takes_max_overlap_turn(self):
        turns = [SpeakerTurn(0.0, 2.0, "SPEAKER_00"), SpeakerTurn(2.0, 4.0, "SPEAKER_01")]
        t = transcript(
            [Segment(0, 0.0, 4.0, "a b")],
            [Word("a", 0.2, 0.8), Word("b", 2.5, 3.0)],
        )
        out = apply_speakers(t, turns)
        assert out.words[0].speaker == "SPEAKER_00"
        assert out.words[1].speaker == "SPEAKER_01"

    def test_segment_speaker_is_majority_of_its_words(self):
        turns = [SpeakerTurn(0.0, 3.0, "SPEAKER_00"), SpeakerTurn(3.0, 6.0, "SPEAKER_01")]
        # 2 words in SPEAKER_00, 1 in SPEAKER_01 -> segment is SPEAKER_00
        t = transcript(
            [Segment(0, 0.0, 6.0, "a b c")],
            [Word("a", 0.1, 0.5), Word("b", 1.0, 1.5), Word("c", 4.0, 4.5)],
        )
        out = apply_speakers(t, turns)
        assert out.segments[0].speaker == "SPEAKER_00"

    def test_two_segments_get_distinct_speakers(self):
        turns = [SpeakerTurn(0.0, 2.0, "SPEAKER_00"), SpeakerTurn(2.0, 4.0, "SPEAKER_01")]
        t = transcript(
            [Segment(0, 0.0, 2.0, "hi"), Segment(1, 2.0, 4.0, "bye")],
            [Word("hi", 0.3, 0.9), Word("bye", 2.5, 3.1)],
        )
        out = apply_speakers(t, turns)
        assert [s.speaker for s in out.segments] == ["SPEAKER_00", "SPEAKER_01"]

    def test_word_in_gap_has_no_speaker_but_segment_falls_back(self):
        # Word midpoint lands in a diarization gap (no turn overlaps it).
        turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00"), SpeakerTurn(5.0, 6.0, "SPEAKER_01")]
        t = transcript(
            [Segment(0, 0.0, 1.2, "x")],
            [Word("x", 2.0, 2.4)],  # in the 1.0-5.0 gap
        )
        out = apply_speakers(t, turns)
        assert out.words[0].speaker is None
        # No labelled words -> segment falls back to its own max-overlap turn.
        assert out.segments[0].speaker == "SPEAKER_00"

    def test_single_speaker_labels_everything(self):
        turns = [SpeakerTurn(0.0, 10.0, "SPEAKER_00")]
        t = transcript(
            [Segment(0, 0.0, 2.0, "a")],
            [Word("a", 0.5, 1.0)],
        )
        out = apply_speakers(t, turns)
        assert out.segments[0].speaker == "SPEAKER_00"
        assert out.words[0].speaker == "SPEAKER_00"

    def test_equal_overlap_tie_breaks_to_earliest_turn(self):
        # Word [1,3] overlaps both turns equally (1.0s each); earliest (start 0) wins,
        # even when turns are passed out of chronological order.
        turns = [SpeakerTurn(2.0, 4.0, "SPEAKER_01"), SpeakerTurn(0.0, 2.0, "SPEAKER_00")]
        t = transcript([Segment(0, 0.0, 4.0, "x")], [Word("x", 1.0, 3.0)])
        out = apply_speakers(t, turns)
        assert out.words[0].speaker == "SPEAKER_00"

    def test_no_turns_returns_unchanged(self):
        t = transcript(
            [Segment(0, 0.0, 2.0, "a", speaker=None)],
            [Word("a", 0.5, 1.0)],
        )
        out = apply_speakers(t, [])
        assert out.segments[0].speaker is None
        assert out.words[0].speaker is None
