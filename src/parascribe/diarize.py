"""Speaker diarization via pyannote.audio (Phase 1).

pyannote + torch are heavy and optional (requirements-diarization.txt), so they
are imported lazily inside the Diarizer rather than at module import. This module
stays importable without them; the API layer constructs a Diarizer only when
``enable_diarization`` is set, and surfaces DiarizationUnavailableError clearly
if the deps or gated models are missing (never a silent no-speaker fallback).

The model produces opaque speaker labels (SPEAKER_00, ...). Diarization runs the
whole file (clustering is global), so it is not streamable.

NOTE: targets pyannote.audio 4.x (modern huggingface_hub `token=` API). The gated
models require a HuggingFace token + license acceptance for the first download;
validate on GPU hardware.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from parascribe.align import SpeakerTurn
from parascribe.asr import SAMPLE_RATE

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from parascribe.config import Settings

logger = logging.getLogger(__name__)


class DiarizationUnavailableError(RuntimeError):
    """pyannote deps or models are not available (maps to a clear API error)."""


class Diarizer:
    """Loads the pyannote pipeline once and produces speaker turns for audio."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        device = settings.resolved_diarization_device
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizationUnavailableError(
                "diarization requires pyannote.audio (pip install -r "
                "requirements-diarization.txt)"
            ) from exc

        logger.info("loading diarization model %s on %s", settings.diarization_model, device)
        pipeline = Pipeline.from_pretrained(
            settings.diarization_model, token=settings.resolved_hf_token()
        )
        if pipeline is None:
            # pyannote returns None when the model is gated and access/token is missing.
            raise DiarizationUnavailableError(
                f"could not load {settings.diarization_model}: accept its license on "
                "HuggingFace and set hf_token/hf_token_file (or pre-cache the model)"
            )
        pipeline.to(torch.device(device))
        self._pipeline = pipeline
        self._torch = torch
        logger.info("diarization model loaded")

    def diarize(
        self, audio: npt.NDArray[np.float32], *, num_speakers: int | None = None
    ) -> list[SpeakerTurn]:
        """Return speaker turns for a 16 kHz mono float32 array (absolute times)."""
        waveform = self._torch.from_numpy(audio).unsqueeze(0)  # (1, num_samples)
        params: dict[str, int] = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        annotation = self._pipeline(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE}, **params
        )
        return [
            SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
