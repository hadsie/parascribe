"""ffmpeg-based decode of arbitrary audio/video input to 16 kHz mono PCM.

onnx-asr reads a PCM wav path or a float32 numpy array; we decode via an ffmpeg
subprocess (explicit args, never shell) straight to a float32 array over a pipe,
avoiding a second temp file. Media parsing is an attack surface: ffmpeg is given
a fixed arg list and a bounded input, and any failure becomes a clean DecodeError
(which the API maps to 400).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from parascribe.asr import SAMPLE_RATE

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)

_FFMPEG = "ffmpeg"
_FFPROBE = "ffprobe"
# Decode runs at 100x+ realtime (pure decompression, no inference), so this
# covers even multi-hour uploads; a subprocess still running after this long is
# hung on pathological input, not working. Transcription itself is not timed.
DEFAULT_TIMEOUT_S = 300.0


class DecodeError(RuntimeError):
    """Input could not be decoded to audio (maps to HTTP 400)."""


def contains_video(source: Path, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> bool:
    """True if ``source`` has a real video stream (ignoring attached cover art).

    Audio files commonly embed cover art as an mjpeg "video" stream with the
    attached_pic disposition; those must NOT count as video so an mp3-with-cover
    is still accepted when video is disabled.
    """
    cmd = [
        _FFPROBE,
        "-v", "error",
        "-select_streams", "v",
        "-show_entries", "stream_disposition=attached_pic",
        "-of", "csv=p=0",
        str(source),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, timeout=timeout_s)
    except FileNotFoundError as exc:
        raise DecodeError("ffprobe executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        logger.warning("ffprobe timed out after %.0fs", timeout_s)
        raise DecodeError("input media could not be probed in time") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip()
        logger.warning("ffprobe failed (rc=%s): %s", exc.returncode, detail[:300])
        raise DecodeError("could not probe input media") from exc
    # One line per video stream: "1" => attached picture, "0" => real video track.
    return any(line.strip() == "0" for line in proc.stdout.decode().splitlines())


def decode_to_pcm(
    source: Path, *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> npt.NDArray[np.float32]:
    """Decode ``source`` (audio or video) to a 16 kHz mono float32 PCM array.

    ffmpeg selects the audio stream automatically, so the same command handles a
    video container's audio track. ``source`` should be a tmpfs path with a
    generated name (never the user's original filename) so logs stay content-free.
    """
    cmd = [
        _FFMPEG,
        "-nostdin",
        "-v", "error",
        "-i", str(source),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, timeout=timeout_s)
    except FileNotFoundError as exc:
        raise DecodeError("ffmpeg executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        logger.warning("ffmpeg decode timed out after %.0fs", timeout_s)
        raise DecodeError("input could not be decoded in time") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip()
        logger.warning("ffmpeg decode failed (rc=%s): %s", exc.returncode, detail[:300])
        raise DecodeError("input could not be decoded as audio or video") from exc

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise DecodeError("input contains no decodable audio stream")
    return audio


def duration_seconds(audio: npt.NDArray[np.float32]) -> float:
    """Decoded audio duration in seconds (full media length, not speech-only)."""
    return float(audio.shape[0]) / SAMPLE_RATE
