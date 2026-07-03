"""Runtime configuration via environment variables (PARASCRIBE_* prefix)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ExecutionProvider = Literal["cuda", "cpu", "coreml"]
DiarizationDevice = Literal["cuda", "cpu"]
# What one reported "token" counts. All units are deterministic per input.
UsageUnit = Literal["token", "word", "segment", "char", "file_duration"]
UsageField = Literal["input_tokens", "output_tokens"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PARASCRIBE_",
        env_file=".env",
        extra="ignore",
        # 'model_id' would otherwise collide with pydantic's protected 'model_' namespace.
        protected_namespaces=(),
    )

    # Model / inference
    model_id: str = "istupakov/parakeet-tdt-0.6b-v3-onnx"
    execution_provider: ExecutionProvider = "cuda"
    gpu_device_id: int = 0

    # Multi-model serving (optional). Empty `models` => single mode: model_id is
    # loaded at startup and the request `model` field is ignored for routing. A
    # non-empty allow-list => multi mode: the request model selects from the list,
    # loaded on demand, with up to max_resident_models held in VRAM at once.
    # NoDecode: PARASCRIBE_MODELS is comma-separated, not JSON; the validator below
    # splits it (the default env source would otherwise try to JSON-decode a list).
    models: Annotated[list[str], NoDecode] = []
    max_resident_models: int = 1
    model_ttl_s: float | None = None

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # Auth
    api_key: str | None = None
    api_key_file: Path | None = None

    # Chunking / VAD (mapped to onnx-asr VadOptions)
    max_chunk_s: float = 24.0
    chunk_overlap_s: float = 0.0
    vad_threshold: float = 0.5

    # Limits / IO
    work_dir: Path = Path("/run/parascribe")
    max_upload_mb: int = 2048
    # Kill ffmpeg/ffprobe on input that hangs the decoder (maps to 400). Bounds
    # only the decode subprocess, never transcription/diarization time.
    decode_timeout_s: float = 300.0
    enable_video: bool = False
    # Remote-URL input. When enabled, a file upload whose content is an http(s)
    # URL is fetched server-side. With an empty allowlist only public addresses
    # are reachable; a non-empty allowlist restricts to those exact hosts. See
    # fetch.py.
    enable_url_fetch: bool = False
    url_fetch_allowlist: list[str] = []
    url_fetch_timeout_s: float = 30.0
    # Max requests admitted at once (1 in-flight + the rest queued). Beyond this
    # the server returns 503.
    max_queue: int = 16

    # Diarization (opt-in per request, gated here at the server level)
    enable_diarization: bool = False
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    # None => follow the ASR provider (cuda -> cuda, else cpu).
    diarization_device: DiarizationDevice | None = None
    hf_token: str | None = None
    hf_token_file: Path | None = None

    # Usage / billing (reported as OpenAI 'tokens' usage; see README "Token costs").
    # Each component bills round(count(unit) * multiplier). Audio-input defaults to
    # duration * 10 -> input_tokens (OpenAI parity; multiplier 0 disables it).
    audio_input_usage_unit: UsageUnit = "file_duration"
    audio_input_usage_multiplier: float = 10.0
    audio_input_usage_field: UsageField = "input_tokens"
    transcription_usage_unit: UsageUnit = "token"
    transcription_usage_multiplier: float = 1.0
    # Diarization adds to output_tokens only when it ran; default ~5x transcription.
    diarization_usage_unit: UsageUnit = "token"
    diarization_usage_multiplier: float = 5.0

    # Language / logging
    default_language: str | None = None
    # Operational log verbosity (content-free). debug_logging overrides this to
    # DEBUG and additionally permits transcript content into the logs.
    log_level: str = "INFO"
    debug_logging: bool = False

    @field_validator("models", mode="before")
    @classmethod
    def _split_models(cls, value: object) -> object:
        """Parse PARASCRIBE_MODELS as a comma-separated list (not only JSON)."""
        if isinstance(value, str):
            return [m.strip() for m in value.split(",") if m.strip()]
        return value

    def resolved_api_key(self) -> str | None:
        """The configured bearer token, from api_key or api_key_file (or None).

        Raises if api_key_file is set but unreadable: a typo'd path must fail
        the server loudly, not silently disable authentication.
        """
        return self._read_secret(self.api_key, self.api_key_file, name="api_key_file")

    def resolved_hf_token(self) -> str | None:
        """HuggingFace token for the (one-time, gated) diarization model download."""
        return self._read_secret(self.hf_token, self.hf_token_file, name="hf_token_file")

    @staticmethod
    def _read_secret(value: str | None, path: Path | None, *, name: str) -> str | None:
        if value:
            return value
        if path is None:
            return None
        if not path.exists():
            raise ValueError(f"{name} points to a missing file: {path}")
        secret = path.read_text(encoding="utf-8").strip()
        if not secret:
            raise ValueError(f"{name} points to an empty file: {path}")
        return secret

    @property
    def resolved_diarization_device(self) -> DiarizationDevice:
        """Device for diarization: explicit setting, else follow the ASR provider."""
        if self.diarization_device is not None:
            return self.diarization_device
        return "cuda" if self.execution_provider == "cuda" else "cpu"
