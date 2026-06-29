"""FastAPI app: lifespan model load, serialized inference, OpenAI transcription route."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response, StreamingResponse

from parascribe import __version__
from parascribe.align import apply_speakers
from parascribe.asr import RawSegment, Transcriber
from parascribe.auth import check_bearer
from parascribe.config import Settings
from parascribe.diarize import Diarizer
from parascribe.fetch import FetchError, FetchTooLargeError, fetch_to_file, looks_like_url
from parascribe.formats import (
    ALLOWED_FORMATS,
    STREAMABLE_FORMATS,
    delta_event,
    done_event,
    render,
    sse_event,
)
from parascribe.log import configure_logging, debug_enabled
from parascribe.media import DecodeError, contains_video, decode_to_pcm, duration_seconds
from parascribe.stitch import Transcript, assemble, offset_segment
from parascribe.usage import build_usage

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    Audio = npt.NDArray[np.float32]

logger = logging.getLogger(__name__)

_UPLOAD_CHUNK = 1024 * 1024


class QueueFullError(RuntimeError):
    """Raised when the inference admission queue is saturated (maps to 503)."""


class InferenceGate:
    """Serialize inference to one in-flight call and bound total admitted requests.

    The semaphore guarantees a single GPU forward pass at a time; the capacity
    counter bounds running + queued so an overload sheds load with 503 instead of
    growing an unbounded wait queue.
    """

    def __init__(self, capacity: int) -> None:
        self._sem = asyncio.Semaphore(1)
        self._capacity = max(1, capacity)
        self._admitted = 0
        self._guard = asyncio.Lock()

    async def acquire(self) -> None:
        """Admit this request (raising QueueFullError if saturated) then serialize."""
        async with self._guard:
            if self._admitted >= self._capacity:
                raise QueueFullError
            self._admitted += 1
        try:
            await self._sem.acquire()
        except BaseException:
            # Cancelled/interrupted while waiting: give the admitted slot back
            # before propagating so the capacity counter stays accurate.
            async with self._guard:
                self._admitted -= 1
            raise

    async def release(self) -> None:
        self._sem.release()
        async with self._guard:
            self._admitted -= 1

    async def __aenter__(self) -> InferenceGate:
        await self.acquire()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.release()


async def _stream_events(
    transcriber: Transcriber,
    gate: InferenceGate,
    audio: Audio,
    *,
    settings: Settings,
    language_hint: str | None,
    duration: float,
    response_format: str,
    include_words: bool,
    rid: str,
) -> AsyncIterator[str]:
    """Bridge the sync VAD-segment generator to SSE events as segments finalize.

    A worker thread iterates the (blocking) transcription generator and feeds an
    asyncio.Queue; we emit a delta per segment, then a terminal done event. The
    gate (acquired by the caller) is released when the stream finishes or the
    client disconnects (StreamingResponse closes this generator).
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[RawSegment | object] = asyncio.Queue()
    sentinel = object()
    log = {"rid": rid}
    infer_start = time.monotonic()

    def produce() -> None:
        try:
            for raw in transcriber.transcribe(audio, language=language_hint):
                loop.call_soon_threadsafe(queue.put_nowait, raw)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    segments = []
    words = []
    texts: list[str] = []
    token_count = 0
    try:
        worker = loop.run_in_executor(None, produce)
        seg_id = 0
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            assert isinstance(item, RawSegment)
            if not item.text.strip():
                continue  # drop VAD regions with no recognized speech (matches assemble)
            segment, seg_words = offset_segment(seg_id, item)
            seg_id += 1
            segments.append(segment)
            words.extend(seg_words)
            texts.append(segment.text)
            token_count += len(item.tokens)
            yield sse_event(delta_event(segment.text + " "))
        await worker  # surface any exception raised inside the producer thread
        transcript = Transcript(
            text=" ".join(texts), language=language_hint, duration=duration,
            segments=segments, words=words, token_count=token_count,
        )
        usage = build_usage(transcript, settings, diarized=False)  # streaming never diarizes
        yield sse_event(
            done_event(
                transcript, response_format=response_format,
                include_words=include_words, usage=usage,
            )
        )
        infer_ms = int((time.monotonic() - infer_start) * 1000)
        logger.info(
            "done(stream): dur=%.1fs infer=%dms segments=%d words=%d",
            duration, infer_ms, len(segments), len(words), extra=log,
        )
        if debug_enabled():
            logger.debug("transcript text=%r", transcript.text, extra=log)
    finally:
        await gate.release()


async def _save_upload(
    upload: UploadFile, dest: Path, max_bytes: int, *, fetch_enabled: bool
) -> str | None:
    """Stream the upload to ``dest`` (size-capped), or return a URL to fetch.

    When ``fetch_enabled`` and the entire upload is a single http(s) URL, nothing
    is written and the URL is returned for the caller to fetch. Otherwise the
    bytes are the audio.
    """
    first = await upload.read(_UPLOAD_CHUNK)
    # A URL fits in one read; only attempt detection when the first read reached
    # EOF (a larger upload spans multiple reads).
    if fetch_enabled and len(first) < _UPLOAD_CHUNK:
        url = looks_like_url(first)
        if url is not None:
            return url

    size = 0
    chunk = first
    with dest.open("wb") as handle:
        while chunk:
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(status_code=413, detail="Upload exceeds max_upload_mb.")
            handle.write(chunk)
            chunk = await upload.read(_UPLOAD_CHUNK)
    return None


def create_app(
    settings: Settings | None = None,
    transcriber: Transcriber | None = None,
    diarizer: Diarizer | None = None,
) -> FastAPI:
    """Build the app. Inject ``transcriber``/``diarizer`` in tests to skip model loads."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings)
        settings.work_dir.mkdir(parents=True, exist_ok=True)
        if settings.resolved_api_key() is None:
            logger.warning("No API key configured: authentication is DISABLED.")
        app.state.settings = settings
        app.state.transcriber = transcriber or Transcriber(settings)
        # Load the diarizer once when enabled; a load failure (missing deps/gated
        # model) fails startup loudly rather than silently disabling the feature.
        app.state.diarizer = diarizer or (
            Diarizer(settings) if settings.enable_diarization else None
        )
        app.state.gate = InferenceGate(settings.max_queue)
        logger.info(
            "ready: model=%s provider=%s diarization=%s max_queue=%d",
            settings.model_id, settings.execution_provider,
            app.state.diarizer is not None, settings.max_queue,
        )
        yield

    app = FastAPI(title="parascribe", version=__version__, lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        t: Transcriber = request.app.state.transcriber
        return JSONResponse(
            {
                "status": "ok",
                "model_id": settings.model_id,
                "device": t.device,
                "provider_active": t.provider_active,
            }
        )

    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        request: Request,
        # `file` is the audio upload. When enable_url_fetch is on, its content may
        # instead be an http(s) URL to fetch. See fetch.py.
        file: Annotated[UploadFile, File()],
        # Required for OpenAI compatibility; this server serves the one configured
        # model, so the value is accepted but not used for routing.
        model: Annotated[str, Form()],
        response_format: Annotated[str, Form()] = "json",
        language: Annotated[str | None, Form()] = None,
        stream: Annotated[bool, Form()] = False,
        temperature: Annotated[float | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        diarization: Annotated[bool, Form()] = False,
        num_speakers: Annotated[int | None, Form()] = None,
        # OpenAI sends `timestamp_granularities[]`; some clients send the bare key.
        # Accept both. A gateway may drop this param entirely, which is why
        # verbose_json emits segments regardless of whether it arrives.
        timestamp_granularities_bracket: Annotated[
            list[str] | None, Form(alias="timestamp_granularities[]")
        ] = None,
        timestamp_granularities: Annotated[list[str] | None, Form()] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> Response:
        st: Settings = request.app.state.settings
        transcriber: Transcriber = request.app.state.transcriber
        diarizer: Diarizer | None = request.app.state.diarizer
        gate: InferenceGate = request.app.state.gate

        check_bearer(st.resolved_api_key(), authorization)

        if response_format not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported response_format. Allowed: {list(ALLOWED_FORMATS)}",
            )
        if diarization and diarizer is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Diarization is not enabled on this server.",
            )
        if num_speakers is not None and num_speakers < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="num_speakers must be a positive integer.",
            )

        granularities = set(timestamp_granularities_bracket or []) | set(
            timestamp_granularities or []
        )
        include_words = "word" in granularities
        language_hint = language or st.default_language
        # Diarization needs the whole file (global clustering), so it can't stream.
        do_stream = stream and response_format in STREAMABLE_FORMATS and not diarization

        rid = uuid.uuid4().hex[:8]
        log = {"rid": rid}
        logger.info(
            "recv: format=%s stream=%s diarization=%s lang=%s words=%s",
            response_format, do_stream, diarization, language_hint or "-",
            include_words, extra=log,
        )
        if temperature is not None or prompt:
            logger.debug("ignoring unsupported params (temperature/prompt)", extra=log)
        if stream and not do_stream:
            reason = "diarization on" if diarization else f"format={response_format}"
            logger.warning("stream=true ignored (%s)", reason, extra=log)

        tmp_path = st.work_dir / f"upload-{rid}"
        # Decode fully into memory, then drop the upload immediately: streaming and
        # non-streaming alike work from the in-memory array, so the temp file never
        # outlives decode (keeps uploaded media off disk beyond the request).
        max_bytes = st.max_upload_mb * 1024 * 1024
        decode_start = time.monotonic()
        try:
            source_url = await _save_upload(
                file, tmp_path, max_bytes, fetch_enabled=st.enable_url_fetch
            )
            if source_url is not None:
                logger.debug("input is a remote URL; fetching", extra=log)
                await run_in_threadpool(
                    fetch_to_file, source_url, tmp_path,
                    max_bytes=max_bytes, timeout=st.url_fetch_timeout_s,
                    allowlist=st.url_fetch_allowlist,
                )
            if not st.enable_video and await run_in_threadpool(contains_video, tmp_path):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Video input is disabled (set enable_video=true).",
                )
            audio = await run_in_threadpool(decode_to_pcm, tmp_path)
        except FetchTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except (FetchError, DecodeError) as exc:
            logger.warning("input failed: %s", exc, extra=log)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        duration = duration_seconds(audio)
        decode_ms = int((time.monotonic() - decode_start) * 1000)
        logger.debug("decoded %.2fs audio in %dms", duration, decode_ms, extra=log)

        try:
            # Admit before responding so saturation returns 503 even for streaming
            # (where the 200 headers would otherwise already be sent).
            await gate.acquire()
        except QueueFullError:
            logger.warning("rejected: inference queue full", extra=log)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server busy: inference queue is full.",
            ) from None

        if do_stream:
            return StreamingResponse(
                _stream_events(
                    transcriber, gate, audio,
                    settings=st, language_hint=language_hint, duration=duration,
                    response_format=response_format, include_words=include_words, rid=rid,
                ),
                media_type="text/event-stream",
            )

        # ASR and (optionally) diarization run sequentially under the single-flight
        # gate -- one request's GPU work at a time.
        infer_start = time.monotonic()
        try:
            raw_segments = await run_in_threadpool(
                lambda: list(transcriber.transcribe(audio, language=language_hint))
            )
            turns = []
            if diarization and diarizer is not None:
                turns = await run_in_threadpool(
                    lambda: diarizer.diarize(audio, num_speakers=num_speakers, rid=rid)
                )
        finally:
            await gate.release()
        infer_ms = int((time.monotonic() - infer_start) * 1000)

        transcript = assemble(raw_segments, language=language_hint, duration=duration)
        if turns:
            transcript = apply_speakers(transcript, turns)
        logger.info(
            "done: dur=%.1fs infer=%dms segments=%d words=%d speakers=%d format=%s",
            duration, infer_ms, len(transcript.segments), len(transcript.words),
            len({s.speaker for s in transcript.segments if s.speaker}), response_format,
            extra=log,
        )
        if debug_enabled():
            logger.debug("transcript text=%r", transcript.text, extra=log)
        usage = build_usage(transcript, st, diarized=bool(turns))
        rendered = render(transcript, response_format, include_words=include_words, usage=usage)
        return Response(content=rendered.body, media_type=rendered.media_type)

    return app


app = create_app()
