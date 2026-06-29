"""SSRF-guarded fetch of a remote URL to a local file.

URL input arrives as the *content* of the ``file`` upload: the caller uploads a
single ``http``/``https`` URL string in place of audio bytes. ``looks_like_url``
distinguishes a URL upload from audio; ``fetch_to_file`` retrieves it.

Fetching is gated by ``enable_url_fetch`` and constrained at the boundary:

- only ``http``/``https`` schemes;
- when ``url_fetch_allowlist`` is set, the host must match an entry exactly and
  IP checks are skipped;
- otherwise every resolved address must be public (private / loopback /
  link-local / metadata ranges are rejected);
- redirects are not followed;
- the response is streamed with the same byte cap as a direct upload.

Failures map to HTTP 400 (``FetchError``) or 413 (``FetchTooLargeError``).
Messages exclude the URL, which may carry credentials.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from pathlib import Path
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

_FETCH_CHUNK = 1024 * 1024
# Uploads longer than this are not treated as a URL.
_MAX_URL_LEN = 2048


class FetchError(RuntimeError):
    """The URL could not be fetched / is not allowed (maps to HTTP 400)."""


class FetchTooLargeError(FetchError):
    """The fetched body exceeded the size cap (maps to HTTP 413)."""


def looks_like_url(data: bytes) -> str | None:
    """Return the http(s) URL if ``data`` is exactly one, else None.

    Matches only when the whole content is a single short ASCII http(s) URL with
    no internal whitespace or control bytes.
    """
    if len(data) > _MAX_URL_LEN:
        return None
    text = data.strip()
    # Reject empty content and any internal whitespace/control byte (0x20 is space).
    if not text or any(b <= 0x20 for b in text):
        return None
    try:
        value = text.decode("ascii")
    except UnicodeDecodeError:
        return None
    parsed = urlsplit(value)
    if parsed.scheme in ("http", "https") and parsed.hostname:
        return value
    return None


def _require_public(host: str) -> None:
    """Reject a host whose resolved addresses are not all public.

    Validates every A/AAAA record, so a name resolving to a mix of public and
    internal addresses is rejected. The address is re-resolved when the
    connection is made, so this does not close a DNS-rebinding race on its own.
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchError("could not resolve URL host") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise FetchError("URL resolves to a disallowed (internal) address")


def fetch_to_file(
    url: str,
    dest: Path,
    *,
    max_bytes: int,
    timeout: float,
    allowlist: list[str],
) -> None:
    """Stream ``url`` to ``dest``, enforcing the SSRF guards and the size cap.

    Raises ``FetchError`` for a disallowed/unreachable URL or a non-200 response
    (including redirects, which are not followed), and ``FetchTooLargeError`` if
    the body exceeds ``max_bytes``.
    """
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise FetchError("URL scheme must be http or https")
    host = parsed.hostname
    if not host:
        raise FetchError("URL has no host")

    if allowlist:
        if host.lower() not in {h.lower() for h in allowlist}:
            raise FetchError("URL host is not in the allowlist")
    else:
        _require_public(host)

    size = 0
    try:
        with (
            httpx.Client(timeout=timeout, follow_redirects=False) as client,
            client.stream("GET", url) as resp,
        ):
            if resp.status_code != httpx.codes.OK:
                raise FetchError(f"URL fetch returned status {resp.status_code}")
            with dest.open("wb") as handle:
                for chunk in resp.iter_bytes(_FETCH_CHUNK):
                    size += len(chunk)
                    if size > max_bytes:
                        raise FetchTooLargeError("fetched body exceeds max_upload_mb")
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        # Log the exception type only; its message embeds the URL, which may
        # carry credentials.
        logger.warning("URL fetch failed: %s", type(exc).__name__)
        raise FetchError("could not fetch URL") from exc
