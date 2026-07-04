"""Unit tests for the crux: token offsetting + subword->word grouping + assembly.

The highest-priority tests in the project: they run against synthetic RawSegment
inputs with known offsets, no model.
"""

from __future__ import annotations

import pytest

from parascribe.asr import RawSegment
from parascribe.stitch import assemble, offset_segment


def raw(start, end, text, tokens, ts, logprobs=None):
    return RawSegment(
        start=start, end=end, text=text, tokens=tokens, timestamps=ts, logprobs=logprobs
    )


class TestOffsetSegment:
    def test_token_timestamps_are_offset_by_absolute_segment_start(self):
        # Local token times restart at 0; global must equal seg.start + local.
        seg, words = offset_segment(
            0, raw(4.0, 6.0, "Hello world", [" Hello", " world"], [0.0, 0.8])
        )
        assert words[0].start == 4.0  # 4.0 + 0.0
        assert words[1].start == 4.8  # 4.0 + 0.8

    def test_offset_correct_at_large_absolute_start(self):
        # Guards against precision / off-by-one drift far down the timeline.
        _, words = offset_segment(
            0, raw(3600.0, 3602.0, "word", [" word"], [0.16])
        )
        assert words[0].start == 3600.16

    def test_subwords_merge_into_one_word(self):
        _, words = offset_segment(
            0, raw(0.0, 2.0, "segment", [" seg", "ment"], [0.5, 0.7])
        )
        assert [w.word for w in words] == ["segment"]
        assert words[0].start == 0.5

    def test_trailing_punctuation_attaches_to_word(self):
        _, words = offset_segment(
            0, raw(0.0, 2.0, "The segment.", [" The", " seg", "ment", "."], [0.0, 0.5, 0.7, 0.9])
        )
        assert [w.word for w in words] == ["The", "segment."]

    def test_word_end_is_next_token_start(self):
        _, words = offset_segment(
            0, raw(0.0, 2.0, "a b", [" a", " b"], [0.0, 0.6])
        )
        assert words[0].end == 0.6  # start of next token

    def test_final_word_end_is_segment_end(self):
        _, words = offset_segment(
            0, raw(0.0, 2.0, "a b", [" a", " b"], [0.0, 0.6])
        )
        assert words[-1].end == 2.0  # no following token -> seg end

    def test_segment_start_end_passed_through_absolute(self):
        seg, _ = offset_segment(7, raw(8.162, 10.378, "x", [" x"], [0.0]))
        assert seg.start == 8.162
        assert seg.end == 10.378

    def test_segment_id_is_assigned(self):
        seg, _ = offset_segment(3, raw(0.0, 1.0, "x", [" x"], [0.0]))
        assert seg.id == 3

    def test_avg_logprob_is_mean_of_token_logprobs(self):
        seg, _ = offset_segment(
            0, raw(0.0, 1.0, "x", [" x", "y"], [0.0, 0.2], logprobs=[-0.1, -0.3])
        )
        assert seg.avg_logprob == -0.2

    def test_avg_logprob_none_when_absent(self):
        seg, _ = offset_segment(0, raw(0.0, 1.0, "x", [" x"], [0.0]))
        assert seg.avg_logprob is None

    def test_speaker_is_null_phase0(self):
        seg, _ = offset_segment(0, raw(0.0, 1.0, "x", [" x"], [0.0]))
        assert seg.speaker is None

    def test_empty_tokens_yield_no_words_but_keep_segment(self):
        seg, words = offset_segment(0, raw(0.0, 1.0, "x", [], []))
        assert words == []
        assert seg.text == "x"

    def test_padded_end_is_clamped_to_max_end(self):
        # VAD padding (chunk_overlap_s -> speech_pad_ms) can push a segment's
        # window past the file's end; the artifact must not become a timestamp.
        seg, words = offset_segment(
            0, raw(9.0, 10.6, "tail word", [" tail", " word"], [0.0, 1.5]),
            max_end=10.0,
        )
        assert seg.end == 10.0
        assert words[-1].end == 10.0
        assert words[-1].start == 10.0  # 9.0 + 1.5 clamped to the file's extent

    def test_negative_padded_start_is_clamped_to_zero(self):
        seg, words = offset_segment(
            0, raw(-0.2, 1.0, "head", [" head"], [0.0]), max_end=10.0
        )
        assert seg.start == 0.0
        assert words[0].start == 0.0

    def test_no_clamp_without_max_end(self):
        seg, _ = offset_segment(0, raw(9.0, 10.6, "x", [" x"], [0.0]))
        assert seg.end == 10.6

    def test_mismatched_token_timestamp_lengths_raise(self):
        with pytest.raises(ValueError, match="length mismatch"):
            offset_segment(0, raw(0.0, 1.0, "x", [" a", " b", " c"], [0.0, 0.5]))


class TestAssemble:
    @pytest.fixture
    def transcript(self):
        segs = [
            raw(0.0, 2.0, "The segment.", [" The", " seg", "ment", "."], [0.0, 0.5, 0.7, 0.9]),
            raw(4.0, 6.0, "Hello world", [" Hello", " world"], [0.0, 0.8]),
        ]
        return assemble(segs, language="en", duration=6.0)

    def test_text_is_concatenation_of_segment_texts(self, transcript):
        assert transcript.text == "The segment. Hello world"

    def test_segment_ids_are_sequential(self, transcript):
        assert [s.id for s in transcript.segments] == [0, 1]

    def test_first_segment_starts_near_zero(self, transcript):
        assert transcript.segments[0].start == pytest.approx(0.0, abs=0.05)

    def test_word_times_are_monotonic_across_segments(self, transcript):
        times = [(w.start, w.end) for w in transcript.words]
        flat = [t for pair in times for t in pair]
        assert flat == sorted(flat)

    def test_words_span_all_segments(self, transcript):
        assert [w.word for w in transcript.words] == ["The", "segment.", "Hello", "world"]

    def test_last_word_end_within_duration(self, transcript):
        assert transcript.words[-1].end <= transcript.duration

    def test_language_and_duration_recorded(self, transcript):
        assert transcript.language == "en"
        assert transcript.duration == 6.0

    def test_empty_input_yields_empty_transcript(self):
        t = assemble([], language=None, duration=0.0)
        assert t.text == ""
        assert t.segments == []
        assert t.words == []

    def test_token_count_sums_subwords_over_kept_segments(self, transcript):
        # 4 tokens in the first segment + 2 in the second.
        assert transcript.token_count == 6

    def test_token_count_excludes_dropped_empty_segments(self):
        segs = [
            raw(0.0, 1.0, "Hello.", [" Hello", "."], [0.0, 0.4]),
            raw(1.0, 1.3, "", [" noise"], [0.0]),  # dropped: no recognized speech
            raw(2.0, 3.0, "World.", [" World", "."], [0.0, 0.5]),
        ]
        t = assemble(segs, language="en", duration=3.0)
        assert t.token_count == 4  # only the two kept segments' tokens

    def test_assemble_clamps_segment_times_to_duration(self):
        t = assemble(
            [raw(9.0, 10.6, "tail", [" tail"], [0.0])], language=None, duration=10.0
        )
        assert t.segments[0].end == 10.0
        assert t.words[-1].end == 10.0

    def test_empty_text_segments_are_dropped_with_contiguous_ids(self):
        segs = [
            raw(0.0, 1.0, "Hello.", [" Hello", "."], [0.0, 0.4]),
            raw(1.0, 1.3, "", [], []),  # VAD region, no recognized speech
            raw(2.0, 3.0, "World.", [" World", "."], [0.0, 0.5]),
        ]
        t = assemble(segs, language="en", duration=3.0)
        assert [s.text for s in t.segments] == ["Hello.", "World."]
        assert [s.id for s in t.segments] == [0, 1]  # contiguous, no gap from the drop
        assert t.text == "Hello. World."
