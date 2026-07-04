"""The crux: token-timestamp offsetting + subword->word grouping + assembly.

onnx-asr already gives absolute segment ``start``/``end`` (it owns VAD chunking),
so there is no cross-chunk segment stitching to do. What remains, and what is
easy to get subtly wrong, is:

  * token ``timestamps`` are LOCAL to each segment (they restart at 0.0) -> add
    the segment's absolute ``start`` to every token timestamp;
  * ``tokens`` are SentencePiece subwords -> group into words on the leading-space
    marker to build the OpenAI ``words[]`` list with global timing.

This module is pure (no model, no IO) so it is unit-tested directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from parascribe.asr import RawSegment

# Timestamps are emitted in seconds at an 80 ms (TDT frame) granularity; 3 decimals
# (ms) is lossless for that and keeps the JSON tidy.
_PRECISION = 3


@dataclass(frozen=True)
class Word:
    word: str
    start: float
    end: float
    speaker: str | None = None


@dataclass(frozen=True)
class Segment:
    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    avg_logprob: float | None = None


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None
    duration: float
    segments: list[Segment] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    # Total ASR subword tokens over kept segments; source for the "token" usage unit.
    token_count: int = 0


def _group_words(tokens: list[str], global_ts: list[float], seg_end: float) -> list[Word]:
    """Merge subword tokens into words with global start/end times.

    A new word begins at a token starting with a space (the SentencePiece word
    marker); continuation subwords and trailing punctuation attach to the current
    word. ``word.start`` is the first sub-token's time; ``word.end`` is the next
    token's start time (= this token's end), or ``seg_end`` for the final token.
    """
    if not tokens:
        return []
    if len(tokens) != len(global_ts):
        # Parallel arrays from the backend; a mismatch is a pipeline bug, not a
        # valid edge case. Fail loudly rather than emit plausible-but-wrong times.
        raise ValueError(
            f"tokens/timestamps length mismatch: {len(tokens)} != {len(global_ts)}"
        )

    groups: list[list[int]] = []  # each: [first_idx, last_idx]
    for i, token in enumerate(tokens):
        if token.startswith(" ") or not groups:
            groups.append([i, i])
        else:
            groups[-1][1] = i

    words: list[Word] = []
    for first, last in groups:
        text = "".join(tokens[first : last + 1]).strip()
        if not text:
            continue
        start = global_ts[first]
        end = global_ts[last + 1] if last + 1 < len(global_ts) else seg_end
        words.append(
            Word(word=text, start=round(start, _PRECISION), end=round(end, _PRECISION))
        )
    return words


def _clamp(t: float, max_end: float | None) -> float:
    t = max(t, 0.0)
    return t if max_end is None else min(t, max_end)


def offset_segment(
    seg_id: int, raw: RawSegment, *, max_end: float | None = None
) -> tuple[Segment, list[Word]]:
    """Convert one VAD ``RawSegment`` into a global-timed ``Segment`` plus words.

    ``max_end`` bounds every emitted time to the audio's real extent: VAD
    boundary padding (``speech_pad_ms``, from ``chunk_overlap_s``) can push a
    segment's window before 0 or past the file's end, and those artifacts must
    not become plausible-looking timestamps.
    """
    start = _clamp(raw.start, max_end)
    end = _clamp(raw.end, max_end)
    global_ts = [_clamp(raw.start + t, max_end) for t in raw.timestamps]
    words = _group_words(raw.tokens, global_ts, end)
    avg_logprob = (
        round(sum(raw.logprobs) / len(raw.logprobs), 4) if raw.logprobs else None
    )
    segment = Segment(
        id=seg_id,
        start=round(start, _PRECISION),
        end=round(end, _PRECISION),
        text=raw.text.strip(),
        speaker=None,
        avg_logprob=avg_logprob,
    )
    return segment, words


def assemble(
    raw_segments: Iterable[RawSegment], *, language: str | None, duration: float
) -> Transcript:
    """Assemble VAD segments into a unified transcript with global timestamps.

    VAD regions that the ASR transcribed to nothing (a breath, a click) are
    dropped, and segment ids stay contiguous over the kept segments.
    """
    segments: list[Segment] = []
    words: list[Word] = []
    texts: list[str] = []
    token_count = 0
    next_id = 0
    for raw in raw_segments:
        if not raw.text.strip():
            continue
        segment, seg_words = offset_segment(next_id, raw, max_end=duration)
        next_id += 1
        segments.append(segment)
        words.extend(seg_words)
        texts.append(segment.text)
        token_count += len(raw.tokens)
    return Transcript(
        text=" ".join(texts),
        language=language,
        duration=round(duration, _PRECISION),
        segments=segments,
        words=words,
        token_count=token_count,
    )
