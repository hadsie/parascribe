"""onnx-asr model loading, GPU verification, and VAD-chunked transcription.

The library (onnx-asr) owns VAD segmentation and emits absolute segment offsets;
this module is a thin wrapper that pins the execution provider, fails loudly if a
requested GPU is not actually engaged, and yields a normalized ``RawSegment`` per
VAD segment. Token-timestamp offsetting and word grouping live in ``stitch.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import onnx_asr
import onnxruntime as ort

if TYPE_CHECKING:
    import numpy.typing as npt

    from parascribe.config import Settings

    AudioInput = npt.NDArray[np.float32] | str | Path
    ProviderSpec = list[str | tuple[str, dict[str, int]]]

logger = logging.getLogger(__name__)

# onnx-asr's audio sample rate (Parakeet expects 16 kHz mono).
SAMPLE_RATE: Final = 16000


@dataclass(frozen=True)
class RawSegment:
    """One VAD segment as returned by onnx-asr.

    ``start``/``end`` are absolute (original-timeline) seconds. ``timestamps`` are
    per-token and **local to this segment** (they restart at 0.0); offsetting them
    to the global timeline is ``stitch.py``'s job. ``tokens`` are SentencePiece
    subwords parallel to ``timestamps``; ``logprobs`` are per-token (or None).
    """

    start: float
    end: float
    text: str
    tokens: list[str]
    timestamps: list[float]
    logprobs: list[float] | None


class GpuUnavailableError(RuntimeError):
    """Raised when GPU is configured but not actually engaged."""


def build_providers(settings: Settings) -> ProviderSpec:
    """ONNX Runtime providers list for the configured execution provider."""
    if settings.execution_provider == "cuda":
        return [("CUDAExecutionProvider", {"device_id": settings.gpu_device_id})]
    if settings.execution_provider == "coreml":
        return ["CoreMLExecutionProvider"]
    return ["CPUExecutionProvider"]


def build_vad(settings: Settings) -> object:
    """Load the silero VAD with the configured providers.

    The VAD is model-independent, so the registry loads one and shares it across
    every Transcriber rather than reloading it per model.
    """
    return onnx_asr.load_vad("silero", providers=build_providers(settings))


def _iter_sessions(
    obj: object, seen: set[int] | None = None, depth: int = 0
) -> Iterator[ort.InferenceSession]:
    """Walk an object graph yielding every onnxruntime InferenceSession found."""
    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 5:
        return
    seen.add(id(obj))
    if isinstance(obj, ort.InferenceSession):
        yield obj
        return
    namespace = getattr(obj, "__dict__", None)
    if namespace:
        for value in namespace.values():
            yield from _iter_sessions(value, seen, depth + 1)


def active_providers(model: object) -> set[str]:
    """Execution providers actually active on the loaded model's sessions."""
    providers: set[str] = set()
    for session in _iter_sessions(model):
        providers.update(session.get_providers())
    return providers


class Transcriber:
    """Holds the single loaded model + VAD and produces VAD segments.

    Constructed once at startup. ``transcribe`` is NOT internally serialized;
    callers must hold the single-flight lock around it.
    """

    def __init__(
        self, settings: Settings, *, model_id: str | None = None, vad: object | None = None
    ) -> None:
        self.settings = settings
        self.model_id = model_id or settings.model_id
        providers = build_providers(settings)
        if settings.execution_provider == "cuda" and hasattr(ort, "preload_dlls"):
            # Load CUDA/cuDNN from the nvidia-*-cu12 pip wheels (requirements-gpu.txt)
            # so they don't need to be on LD_LIBRARY_PATH.
            ort.preload_dlls()
        logger.info("loading model %s on %s", self.model_id, settings.execution_provider)
        self._model = onnx_asr.load_model(self.model_id, providers=providers)
        self._vad = vad if vad is not None else build_vad(settings)

        vad_options: dict[str, float] = {
            "threshold": settings.vad_threshold,
            "max_speech_duration_s": settings.max_chunk_s,
        }
        if settings.chunk_overlap_s:
            vad_options["speech_pad_ms"] = settings.chunk_overlap_s * 1000.0
        # onnx-asr's VadOptions are loosely typed (int); float thresholds are fine.
        self._adapter = self._model.with_vad(
            self._vad, **vad_options  # type: ignore[arg-type]
        ).with_timestamps()

        self.providers_active = active_providers(self._model)
        self._verify_gpu()
        logger.info("model loaded; active providers: %s", sorted(self.providers_active))

    def _verify_gpu(self) -> None:
        if self.settings.execution_provider != "cuda":
            return
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise GpuUnavailableError(
                "execution_provider=cuda but this onnxruntime build exposes no "
                "CUDAExecutionProvider. Install onnxruntime-gpu (requirements-gpu.txt)."
            )
        if "CUDAExecutionProvider" not in self.providers_active:
            raise GpuUnavailableError(
                "execution_provider=cuda but the model is running on "
                f"{sorted(self.providers_active)}; CUDA failed to initialize "
                "(incompatible onnxruntime-gpu wheel for this GPU?). Refusing to "
                "fall back to CPU."
            )

    @property
    def gpu_active(self) -> bool:
        return "CUDAExecutionProvider" in self.providers_active

    @property
    def provider_active(self) -> bool:
        """Whether the configured execution provider is actually engaged."""
        wanted = {
            "cuda": "CUDAExecutionProvider",
            "coreml": "CoreMLExecutionProvider",
            "cpu": "CPUExecutionProvider",
        }[self.settings.execution_provider]
        return wanted in self.providers_active

    @property
    def device(self) -> str:
        if self.gpu_active:
            return f"cuda:{self.settings.gpu_device_id}"
        return self.settings.execution_provider

    def transcribe(
        self, audio: AudioInput, *, language: str | None = None
    ) -> Iterator[RawSegment]:
        """Yield VAD segments for ``audio`` (a 16 kHz mono float32 array or wav path)."""
        chosen = language or self.settings.default_language
        for seg in self._adapter.recognize(audio, sample_rate=SAMPLE_RATE, language=chosen):
            yield RawSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                tokens=list(seg.tokens or []),
                timestamps=list(seg.timestamps or []),
                logprobs=list(seg.logprobs) if seg.logprobs is not None else None,
            )
