"""API route tests. The model is faked; ffmpeg decode is real (tiny fixtures)."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from parascribe.align import SpeakerTurn
from parascribe.asr import RawSegment
from parascribe.config import Settings
from parascribe.main import (
    InferenceGate,
    QueueFullError,
    _start_stream_producer,
    _stream_events,
    create_app,
)
from parascribe.registry import ModelRegistry

_TOK = [" The", " second", " part", "."]
_TS = [0.0, 0.4, 0.8, 1.2]
CANNED = [
    RawSegment(0.0, 2.0, "The first part.", [" The", " first", " part", "."], _TS, [-0.1] * 4),
    RawSegment(4.0, 6.0, "The second part.", _TOK, _TS, None),
]


class FakeTranscriber:
    device = "cpu"
    provider_active = True

    def transcribe(self, audio, *, language=None):
        yield from CANNED


class FakeDiarizer:
    # CANNED segments sit at 0-2s and 4-6s; split the timeline between two speakers.
    def diarize(self, audio, *, num_speakers=None, rid="-"):
        return [SpeakerTurn(0.0, 3.0, "SPEAKER_00"), SpeakerTurn(3.0, 6.0, "SPEAKER_01")]


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    # 6s so the CANNED segment times (up to 6.0) fit inside the real duration;
    # emitted timestamps are clamped to the decoded audio's extent.
    out = tmp_path / "clip.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=300:duration=6:sample_rate=16000", str(out), "-y"],
        check=True,
    )
    return out


@pytest.fixture
def video(tmp_path: Path) -> Path:
    out = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=duration=1:size=128x96:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=1:sample_rate=16000",
         "-shortest", str(out), "-y"],
        check=True,
    )
    return out


def sse_events(text: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in text.splitlines()
            if line.startswith("data: ")]


def make_client(tmp_path: Path, *, diarizer=None, **overrides) -> TestClient:
    settings = Settings(
        execution_provider="cpu",
        work_dir=tmp_path / "work",
        api_key="secret",
        **overrides,
    )
    return TestClient(
        create_app(settings=settings, transcriber=FakeTranscriber(), diarizer=diarizer)
    )


def make_multi_client(tmp_path: Path, models, **overrides) -> TestClient:
    settings = Settings(
        execution_provider="cpu",
        work_dir=tmp_path / "work",
        api_key="secret",
        models=models,
        **overrides,
    )
    registry = ModelRegistry(settings, factory=lambda _model_id: FakeTranscriber())
    registry.preload()  # loads the default model (sets device/provider for /health)
    return TestClient(create_app(settings=settings, registry=registry))


def post(client, wav, *, key="secret", **data):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with wav.open("rb") as fh:
        return client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", fh, "audio/wav")},
            data={"model": "parascribe", **data},
            headers=headers,
        )


class TestAuth:
    def test_missing_key_is_401(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            assert post(client, wav, key=None).status_code == 401

    def test_wrong_key_is_401(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            assert post(client, wav, key="wrong").status_code == 401

    def test_correct_key_is_200(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            assert post(client, wav).status_code == 200


class TestErrorEnvelope:
    """Errors must use the OpenAI {"error": {...}} shape, not FastAPI's detail."""

    def test_401_uses_openai_envelope(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            r = post(client, wav, key="wrong")
            err = r.json()["error"]
            assert err["type"] == "invalid_request_error"
            assert err["code"] == "invalid_api_key"
            assert r.headers["www-authenticate"] == "Bearer"

    def test_bad_response_format_is_400_envelope(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            r = post(client, wav, response_format="nope")
            assert r.status_code == 400
            assert r.json()["error"]["type"] == "invalid_request_error"

    def test_missing_model_param_is_400_not_422(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            with wav.open("rb") as fh:
                r = client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("clip.wav", fh, "audio/wav")},
                    headers={"Authorization": "Bearer secret"},
                )
            assert r.status_code == 400
            assert r.json()["error"]["param"] == "model"

    def test_missing_file_param_is_400_not_422(self, tmp_path):
        with make_client(tmp_path) as client:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "parascribe"},
                headers={"Authorization": "Bearer secret"},
            )
            assert r.status_code == 400
            assert r.json()["error"]["param"] == "file"


class TestResponseFormats:
    def test_json_returns_text_and_usage(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            r = post(client, wav)
            body = r.json()
            assert body["text"] == "The first part. The second part."
            # 2 canned segments * 4 subword tokens, billed 1:1, no diarization.
            assert body["usage"]["output_tokens"] == 8
            # Audio input: 6s wav * 10 (OpenAI-parity default) -> input_tokens.
            assert body["usage"]["input_tokens"] == 60

    def test_verbose_json_has_segments_without_granularities(self, tmp_path, wav):
        # Invariant #1: segments + real times without timestamp_granularities[].
        with make_client(tmp_path) as client:
            r = post(client, wav, response_format="verbose_json")
            body = r.json()
            assert body["segments"][1]["start"] == 4.0
            assert "words" not in body

    def test_verbose_json_words_when_requested(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            r = post(
                client, wav,
                response_format="verbose_json",
                **{"timestamp_granularities[]": "word"},
            )
            words = r.json()["words"]
            # second segment's words are globally offset by 4.0
            assert any(w["word"] == "second" and w["start"] == 4.4 for w in words)

    def test_srt_content_type(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            r = post(client, wav, response_format="srt")
            assert r.text.startswith("1\n00:00:00,000 --> 00:00:02,000")

    def test_bad_format_is_400(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            assert post(client, wav, response_format="flac").status_code == 400


class TestErrors:
    def test_non_media_file_is_400(self, tmp_path):
        bogus = tmp_path / "x.bin"
        bogus.write_bytes(b"not media" * 50)
        with make_client(tmp_path) as client:
            assert post(client, bogus).status_code == 400

    def test_oversize_is_413(self, tmp_path, wav):
        with make_client(tmp_path, max_upload_mb=0) as client:
            assert post(client, wav).status_code == 413


class TestHealth:
    def test_health_reports_provider(self, tmp_path):
        with make_client(tmp_path) as client:
            body = client.get("/health").json()
            assert body["status"] == "ok"
            assert body["provider_active"] is True
            assert body["device"] == "cpu"

    def test_health_single_mode(self, tmp_path):
        with make_client(tmp_path) as client:
            assert client.get("/health").json()["mode"] == "single"

    def test_health_multi_mode_lists_models(self, tmp_path):
        with make_multi_client(tmp_path, models=["m1", "m2"]) as client:
            body = client.get("/health").json()
            assert body["mode"] == "multi"
            assert {"m1", "m2"} <= set(body["models"])
            assert body["loaded"]  # default model preloaded


class TestMultiModel:
    def test_allowed_model_is_200(self, tmp_path, wav):
        with make_multi_client(tmp_path, models=["m1", "m2"]) as client:
            assert post(client, wav, model="m1").status_code == 200

    def test_unknown_model_is_400(self, tmp_path, wav):
        with make_multi_client(tmp_path, models=["m1", "m2"]) as client:
            assert post(client, wav, model="nope").status_code == 400

    def test_single_mode_ignores_unknown_model(self, tmp_path, wav):
        # No allow-list => the model field is not used for routing; any value works.
        with make_client(tmp_path) as client:
            assert post(client, wav, model="whatever").status_code == 200


class TestListModels:
    AUTH = {"Authorization": "Bearer secret"}

    def test_lists_allow_list_in_openai_shape(self, tmp_path):
        with make_multi_client(tmp_path, models=["m1", "m2"]) as client:
            body = client.get("/v1/models", headers=self.AUTH).json()
            assert body["object"] == "list"
            ids = {m["id"] for m in body["data"]}
            assert {"m1", "m2"} <= ids
            assert all(m["object"] == "model" for m in body["data"])

    def test_single_mode_lists_the_one_model(self, tmp_path):
        with make_client(tmp_path) as client:
            body = client.get("/v1/models", headers=self.AUTH).json()
            ids = [m["id"] for m in body["data"]]
            assert ids == ["istupakov/parakeet-tdt-0.6b-v3-onnx"]

    def test_requires_auth(self, tmp_path):
        with make_client(tmp_path) as client:
            assert client.get("/v1/models").status_code == 401

    def test_advertised_model_is_accepted_by_transcription(self, tmp_path, wav):
        # What /v1/models lists must not 400 at the transcription route.
        with make_multi_client(tmp_path, models=["m1", "m2"]) as client:
            listed = client.get("/v1/models", headers=self.AUTH).json()["data"]
            for model in (m["id"] for m in listed):
                assert post(client, wav, model=model).status_code == 200


class TestStreaming:
    def test_stream_json_emits_deltas_then_done(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            r = post(client, wav, stream="true")
            assert r.headers["content-type"].startswith("text/event-stream")
            events = sse_events(r.text)
            deltas = [e for e in events if e["type"] == "transcript.text.delta"]
            done = [e for e in events if e["type"] == "transcript.text.done"]
            assert len(deltas) == 2
            assert len(done) == 1
            assert done[0]["text"] == "The first part. The second part."

    def test_stream_verbose_json_done_carries_segments(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            r = post(client, wav, stream="true", response_format="verbose_json")
            done = [e for e in sse_events(r.text) if e["type"] == "transcript.text.done"]
            assert done[0]["segments"][1]["start"] == 4.0

    def test_stream_done_carries_usage(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            r = post(client, wav, stream="true")
            done = [e for e in sse_events(r.text) if e["type"] == "transcript.text.done"]
            assert done[0]["usage"]["output_tokens"] == 8

    def test_stream_ignored_for_srt_falls_back(self, tmp_path, wav):
        # Decision #6: stream=true with a non-streamable format returns non-streamed.
        with make_client(tmp_path) as client:
            r = post(client, wav, stream="true", response_format="srt")
            assert not r.headers["content-type"].startswith("text/event-stream")
            assert r.text.startswith("1\n00:00:00,000 -->")


class TestUrlInput:
    """URL input rides in as the file upload's content (file-content convention)."""

    def _post_url(self, client, url_str, *, key="secret"):
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        return client.post(
            "/v1/audio/transcriptions",
            files={"file": ("source.url", url_str.encode(), "text/plain")},
            data={"model": "parascribe"},
            headers=headers,
        )

    def test_url_content_treated_as_audio_when_disabled(self, tmp_path):
        # enable_url_fetch off: the URL text is decoded as audio and fails -> 400.
        with make_client(tmp_path) as client:
            assert self._post_url(client, "https://example.com/a.wav").status_code == 400

    def test_real_audio_not_misdetected_as_url(self, tmp_path, wav):
        # Regression: a genuine upload still transcribes with fetching enabled.
        with make_client(tmp_path, enable_url_fetch=True) as client:
            r = post(client, wav)
            assert r.status_code == 200
            assert r.json()["text"] == "The first part. The second part."

    def test_url_content_transcribes(self, tmp_path, wav, monkeypatch):
        audio_bytes = wav.read_bytes()

        def fake_fetch(url, dest, *, max_bytes, timeout, allowlist):
            assert url == "https://example.com/a.wav"
            dest.write_bytes(audio_bytes)

        monkeypatch.setattr("parascribe.main.fetch_to_file", fake_fetch)
        with make_client(tmp_path, enable_url_fetch=True) as client:
            r = self._post_url(client, "https://example.com/a.wav")
            assert r.status_code == 200
            assert r.json()["text"] == "The first part. The second part."

    def test_fetch_error_is_400(self, tmp_path, monkeypatch):
        from parascribe.fetch import FetchError

        def boom(*a, **k):
            raise FetchError("URL resolves to a disallowed (internal) address")

        monkeypatch.setattr("parascribe.main.fetch_to_file", boom)
        with make_client(tmp_path, enable_url_fetch=True) as client:
            assert self._post_url(client, "https://10.0.0.1/a.wav").status_code == 400

    def test_oversize_fetch_is_413(self, tmp_path, monkeypatch):
        from parascribe.fetch import FetchTooLargeError

        def boom(*a, **k):
            raise FetchTooLargeError("fetched body exceeds max_upload_mb")

        monkeypatch.setattr("parascribe.main.fetch_to_file", boom)
        with make_client(tmp_path, enable_url_fetch=True) as client:
            assert self._post_url(client, "https://example.com/big.wav").status_code == 413


class TestVideoGating:
    def test_video_rejected_when_disabled(self, tmp_path, video):
        with make_client(tmp_path) as client:  # enable_video defaults False
            assert post(client, video).status_code == 400

    def test_video_accepted_when_enabled(self, tmp_path, video):
        with make_client(tmp_path, enable_video=True) as client:
            assert post(client, video).status_code == 200


class TestDiarization:
    def test_requested_but_not_enabled_is_400(self, tmp_path, wav):
        with make_client(tmp_path) as client:  # no diarizer injected
            assert post(client, wav, diarization="true").status_code == 400

    def test_populates_segment_speaker_labels(self, tmp_path, wav):
        with make_client(tmp_path, diarizer=FakeDiarizer()) as client:
            r = post(client, wav, response_format="verbose_json", diarization="true")
            segs = r.json()["segments"]
            assert segs[0]["speaker"] == "SPEAKER_00"
            assert segs[1]["speaker"] == "SPEAKER_01"

    def test_diarized_request_bills_extra_usage(self, tmp_path, wav):
        # diarized=bool(turns) wiring: 8 transcription tokens + 8 * 5 diarization.
        with make_client(tmp_path, diarizer=FakeDiarizer()) as client:
            r = post(client, wav, response_format="verbose_json", diarization="true")
            assert r.json()["usage"]["output_tokens"] == 8 + 8 * 5

    def test_words_get_speaker_with_word_granularity(self, tmp_path, wav):
        with make_client(tmp_path, diarizer=FakeDiarizer()) as client:
            r = post(
                client, wav, response_format="verbose_json", diarization="true",
                **{"timestamp_granularities[]": "word"},
            )
            words = r.json()["words"]
            assert all("speaker" in w for w in words)

    def test_no_speaker_field_on_words_without_diarization(self, tmp_path, wav):
        with make_client(tmp_path, diarizer=FakeDiarizer()) as client:
            r = post(
                client, wav, response_format="verbose_json",
                **{"timestamp_granularities[]": "word"},
            )
            assert all("speaker" not in w for w in r.json()["words"])

    def test_stream_with_diarization_falls_back_to_non_streamed(self, tmp_path, wav):
        with make_client(tmp_path, diarizer=FakeDiarizer()) as client:
            r = post(
                client, wav, response_format="verbose_json",
                diarization="true", stream="true",
            )
            assert not r.headers["content-type"].startswith("text/event-stream")
            assert r.json()["segments"][0]["speaker"] == "SPEAKER_00"


@pytest.fixture
def parascribe_log(caplog):
    """Capture the parascribe logger (no root propagation).

    configure_logging clears the logger's handlers at app startup, so tests must
    call ``attach()`` after entering the TestClient context, then read records
    off the fixture.
    """
    logger = logging.getLogger("parascribe")

    class Capture:
        @property
        def records(self):
            return caplog.records

        @staticmethod
        def attach() -> None:
            logger.addHandler(caplog.handler)

    try:
        yield Capture()
    finally:
        logger.removeHandler(caplog.handler)


class TestForensicCleanliness:
    """Invariant: no transcript text, filenames, or URLs in logs at INFO."""

    def _log_blob(self, caplog) -> str:
        return " ".join(r.getMessage() for r in caplog.records)

    def test_no_content_or_filename_in_logs_at_info(self, tmp_path, wav, parascribe_log):
        with make_client(tmp_path) as client:
            parascribe_log.attach()
            post(client, wav, response_format="verbose_json")
        blob = self._log_blob(parascribe_log)
        assert "done:" in blob  # capture sanity: operational line was logged
        assert "first part" not in blob.lower()  # transcript content
        assert "clip.wav" not in blob  # original filename

    def test_url_never_logged_even_on_fetch_failure(self, tmp_path, parascribe_log, monkeypatch):
        from parascribe.fetch import FetchError

        def boom(*a, **k):
            raise FetchError("could not fetch URL")

        monkeypatch.setattr("parascribe.main.fetch_to_file", boom)
        with make_client(tmp_path, enable_url_fetch=True) as client:
            parascribe_log.attach()
            client.post(
                "/v1/audio/transcriptions",
                files={"file": ("s.url", b"https://secret-host.example.com/a.wav", "text/plain")},
                data={"model": "parascribe"},
                headers={"Authorization": "Bearer secret"},
            )
        assert "secret-host.example.com" not in self._log_blob(parascribe_log)

    def test_debug_logging_gates_transcript_content(self, tmp_path, wav, parascribe_log):
        # Off (default): content stays out even of DEBUG-capable capture.
        with make_client(tmp_path) as client:
            parascribe_log.attach()
            post(client, wav)
        assert "first part" not in self._log_blob(parascribe_log).lower()

    def test_debug_logging_exposes_content_when_enabled(self, tmp_path, wav, parascribe_log):
        with make_client(tmp_path, debug_logging=True) as client:
            parascribe_log.attach()
            post(client, wav)
        assert "first part" in self._log_blob(parascribe_log).lower()


class TestUploadCleanup:
    """Invariant: the upload temp file is deleted on every path."""

    def _workdir_entries(self, tmp_path) -> list:
        return list((tmp_path / "work").iterdir())

    def test_removed_after_success(self, tmp_path, wav):
        with make_client(tmp_path) as client:
            assert post(client, wav).status_code == 200
            assert self._workdir_entries(tmp_path) == []

    def test_removed_after_decode_error(self, tmp_path):
        with make_client(tmp_path) as client:
            r = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("junk.bin", b"not audio at all" * 100, "audio/wav")},
                data={"model": "parascribe"},
                headers={"Authorization": "Bearer secret"},
            )
            assert r.status_code == 400
            assert self._workdir_entries(tmp_path) == []

    def test_removed_after_413(self, tmp_path, wav):
        with make_client(tmp_path, max_upload_mb=0) as client:
            assert post(client, wav).status_code == 413
            assert self._workdir_entries(tmp_path) == []

    def test_removed_after_video_rejection(self, tmp_path, video):
        with make_client(tmp_path) as client:  # enable_video defaults False
            assert post(client, video).status_code == 400
            assert self._workdir_entries(tmp_path) == []


class TestGateSadPaths:
    def test_saturated_gate_returns_503_envelope(self, tmp_path, wav, monkeypatch):
        async def full(self):
            raise QueueFullError

        monkeypatch.setattr("parascribe.main.InferenceGate.acquire", full)
        with make_client(tmp_path) as client:
            r = post(client, wav)
            assert r.status_code == 503
            assert r.json()["error"]["type"] == "server_error"

    def test_gate_released_when_model_load_fails(self, tmp_path, wav):
        settings = Settings(
            execution_provider="cpu", work_dir=tmp_path / "work",
            api_key="secret", models=["good", "bad"],
        )

        def factory(model_id):
            if model_id == "bad":
                raise RuntimeError("model load failed")
            return FakeTranscriber()

        registry = ModelRegistry(settings, factory=factory)
        registry.preload()
        app = create_app(settings=settings, registry=registry)
        with TestClient(app, raise_server_exceptions=False) as client:
            assert post(client, wav, model="bad").status_code == 500
            # The slot came back: a follow-up request must not hang or 503.
            assert post(client, wav, model="good").status_code == 200


class TestInferenceGate:
    async def test_rejects_beyond_capacity(self):
        gate = InferenceGate(capacity=1)
        async with gate:
            with pytest.raises(QueueFullError):
                async with gate:
                    pass

    async def test_serializes_execution(self):
        gate = InferenceGate(capacity=10)
        order: list[tuple[str, int]] = []

        async def worker(n: int) -> None:
            async with gate:
                order.append(("start", n))
                await asyncio.sleep(0.01)
                order.append(("end", n))

        await asyncio.gather(*(worker(n) for n in range(3)))
        # No interleaving: every start is immediately followed by its own end.
        for i in range(0, len(order), 2):
            assert order[i][0] == "start" and order[i + 1][0] == "end"
            assert order[i][1] == order[i + 1][1]


class SlowSecondSegment:
    """Transcriber whose second forward pass blocks until ``unblock`` is set."""

    device = "cpu"
    provider_active = True

    def __init__(self) -> None:
        self.unblock = threading.Event()

    def transcribe(self, audio, *, language=None):
        yield CANNED[0]
        assert self.unblock.wait(timeout=5)
        yield CANNED[1]


class TestStreamingGate:
    """The producer thread owns the gate; the SSE generator only consumes."""

    SETTINGS = {"execution_provider": "cpu", "api_key": "k"}

    def _start(self, transcriber, gate, rid="t"):
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        worker = _start_stream_producer(
            transcriber, gate, None, queue, stop, language_hint=None, rid=rid
        )
        return worker, queue, stop

    async def test_disconnect_holds_gate_until_forward_pass_ends(self):
        gate = InferenceGate(capacity=4)
        await gate.acquire()
        transcriber = SlowSecondSegment()
        worker, queue, stop = self._start(transcriber, gate)
        gen = _stream_events(
            worker, queue, stop,
            settings=Settings(**self.SETTINGS), language_hint=None, duration=6.0,
            response_format="json", include_words=False, rid="t",
        )
        assert (await gen.__anext__()).startswith("data: ")
        await gen.aclose()  # client disconnects mid-stream
        assert stop.is_set()
        # The fake forward pass is still running: the gate must still be held.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gate.acquire(), timeout=0.2)
        transcriber.unblock.set()
        # Released only once the producer thread actually finishes.
        await asyncio.wait_for(gate.acquire(), timeout=2)
        gate.release()

    async def test_gate_released_when_stream_is_never_consumed(self):
        # A client can vanish before Starlette ever iterates the generator; the
        # done-callback must release the gate regardless.
        gate = InferenceGate(capacity=4)
        await gate.acquire()
        worker, _queue, _stop = self._start(FakeTranscriber(), gate)
        await worker
        await asyncio.wait_for(gate.acquire(), timeout=2)
        gate.release()

    async def test_producer_error_released_and_logged_with_rid(self, caplog):
        class Boom:
            def transcribe(self, audio, *, language=None):
                raise RuntimeError("onnxruntime failure")
                yield  # pragma: no cover

        gate = InferenceGate(capacity=4)
        await gate.acquire()
        with caplog.at_level(logging.ERROR, logger="parascribe.main"):
            worker, _queue, _stop = self._start(Boom(), gate, rid="rid1")
            with pytest.raises(RuntimeError):
                await worker
            await asyncio.wait_for(gate.acquire(), timeout=2)
            gate.release()
        errors = [r for r in caplog.records if "stream inference failed" in r.getMessage()]
        assert errors and errors[0].rid == "rid1"
