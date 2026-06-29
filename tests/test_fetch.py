"""URL fetch + SSRF guard tests. The HTTP client and DNS resolution are mocked
at the boundary; no real network calls."""

from __future__ import annotations

import socket
from pathlib import Path

import httpx
import pytest

from parascribe import fetch
from parascribe.fetch import FetchError, FetchTooLargeError, fetch_to_file, looks_like_url

PUBLIC_IP = "93.184.216.34"


class FakeStream:
    def __init__(self, status: int, chunks: tuple[bytes, ...]) -> None:
        self.status_code = status
        self._chunks = chunks

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def iter_bytes(self, _n: int):
        yield from self._chunks


class FakeClient:
    def __init__(self, *, status=200, chunks=(b"audio-bytes",), raise_exc=None, **_kw):
        self._status = status
        self._chunks = chunks
        self._raise_exc = raise_exc

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def stream(self, _method: str, _url: str) -> FakeStream:
        if self._raise_exc is not None:
            raise self._raise_exc
        return FakeStream(self._status, self._chunks)


def patch_resolution(monkeypatch, ip: str) -> None:
    def fake_getaddrinfo(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def patch_client(monkeypatch, **client_kwargs) -> None:
    monkeypatch.setattr(
        fetch.httpx, "Client", lambda *a, **k: FakeClient(**client_kwargs)
    )


def call(tmp_path: Path, url: str, *, allowlist=None, max_bytes=1_000_000) -> Path:
    dest = tmp_path / "fetched"
    fetch_to_file(
        url, dest, max_bytes=max_bytes, timeout=5.0, allowlist=allowlist or []
    )
    return dest


class TestLooksLikeUrl:
    def test_accepts_http_and_https(self):
        assert looks_like_url(b"https://example.com/a.wav") == "https://example.com/a.wav"
        assert looks_like_url(b"http://example.com/a.wav") == "http://example.com/a.wav"

    def test_trims_surrounding_whitespace(self):
        assert looks_like_url(b"  https://example.com/a.wav\n") == "https://example.com/a.wav"

    def test_rejects_other_schemes(self):
        assert looks_like_url(b"ftp://example.com/a.wav") is None
        assert looks_like_url(b"file:///etc/passwd") is None

    def test_rejects_internal_whitespace(self):
        assert looks_like_url(b"https://example.com/a b.wav") is None

    def test_rejects_empty(self):
        assert looks_like_url(b"") is None
        assert looks_like_url(b"   ") is None

    def test_rejects_overlong(self):
        assert looks_like_url(b"https://example.com/" + b"a" * 4000) is None

    def test_rejects_binary_audio(self):
        # A WAV header has sub-0x20 bytes; never mistaken for a URL.
        assert looks_like_url(b"RIFF\x24\x00\x00\x00WAVEfmt ") is None


class TestScheme:
    def test_rejects_non_http_scheme(self, tmp_path):
        with pytest.raises(FetchError, match="scheme"):
            call(tmp_path, "ftp://example.com/a.wav")

    def test_rejects_missing_host(self, tmp_path):
        with pytest.raises(FetchError, match="host"):
            call(tmp_path, "http:///a.wav")


class TestSsrfGuard:
    def test_fetches_public_address(self, tmp_path, monkeypatch):
        patch_resolution(monkeypatch, PUBLIC_IP)
        patch_client(monkeypatch, chunks=(b"abc",))
        dest = call(tmp_path, "https://example.com/a.wav")
        assert dest.read_bytes() == b"abc"

    @pytest.mark.parametrize(
        "ip", ["10.0.0.5", "127.0.0.1", "169.254.169.254", "192.168.1.9", "::1"]
    )
    def test_rejects_internal_address(self, tmp_path, monkeypatch, ip):
        patch_resolution(monkeypatch, ip)
        patch_client(monkeypatch)
        with pytest.raises(FetchError, match="internal"):
            call(tmp_path, "https://evil.example/a.wav")

    def test_unresolvable_host_is_error(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise socket.gaierror("nope")

        monkeypatch.setattr(socket, "getaddrinfo", boom)
        with pytest.raises(FetchError, match="resolve"):
            call(tmp_path, "https://no-such-host.example/a.wav")


class TestAllowlist:
    def test_allowed_host_skips_resolution(self, tmp_path, monkeypatch):
        # getaddrinfo must NOT be consulted for an allowlisted host.
        def boom(*a, **k):
            raise AssertionError("resolution should be skipped for allowlisted hosts")

        monkeypatch.setattr(socket, "getaddrinfo", boom)
        patch_client(monkeypatch, chunks=(b"ok",))
        dest = call(
            tmp_path, "https://Media.Example.com/a.wav", allowlist=["media.example.com"]
        )
        assert dest.read_bytes() == b"ok"

    def test_host_not_in_allowlist_is_error(self, tmp_path, monkeypatch):
        patch_client(monkeypatch)
        with pytest.raises(FetchError, match="allowlist"):
            call(tmp_path, "https://other.example/a.wav", allowlist=["media.example.com"])


class TestResponseHandling:
    def test_too_large_raises(self, tmp_path, monkeypatch):
        patch_resolution(monkeypatch, PUBLIC_IP)
        patch_client(monkeypatch, chunks=(b"x" * 600, b"x" * 600))
        with pytest.raises(FetchTooLargeError):
            call(tmp_path, "https://example.com/big.wav", max_bytes=1000)

    def test_non_200_is_error(self, tmp_path, monkeypatch):
        # Redirects are not followed, so a 3xx surfaces as a non-200 failure.
        patch_resolution(monkeypatch, PUBLIC_IP)
        patch_client(monkeypatch, status=302)
        with pytest.raises(FetchError, match="status 302"):
            call(tmp_path, "https://example.com/a.wav")

    def test_network_error_is_clean(self, tmp_path, monkeypatch):
        patch_resolution(monkeypatch, PUBLIC_IP)
        patch_client(monkeypatch, raise_exc=httpx.ConnectError("refused"))
        # Message is generic: the URL (and any credentials) must not leak.
        with pytest.raises(FetchError, match="^could not fetch URL$"):
            call(tmp_path, "https://example.com/a.wav")
