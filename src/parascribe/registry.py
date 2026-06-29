"""Model registry: resolve a request's model to a loaded Transcriber.

Single mode (empty ``models`` allow-list) holds one model, loaded at startup, and
ignores the request model for routing. Multi mode resolves the request model
against the allow-list, lazy-loading on demand and evicting least-recently-used
models when ``max_resident_models`` would be exceeded; an optional idle TTL evicts
models that have not been used recently.

Every mutation (load, evict) must run while the caller holds the single-flight
inference gate, so a model is never loaded or evicted while an inference runs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from parascribe.asr import Transcriber, build_vad

if TYPE_CHECKING:
    from parascribe.config import Settings

logger = logging.getLogger(__name__)

TranscriberFactory = Callable[[str], Transcriber]


class UnknownModelError(Exception):
    """Requested model is not in the configured allow-list (maps to 400)."""

    def __init__(self, requested: str, allowed: set[str]) -> None:
        self.requested = requested
        self.allowed = allowed
        super().__init__(f"Unknown model {requested!r}. Allowed: {sorted(allowed)}")


class ModelRegistry:
    """Loads, caches, and evicts Transcribers keyed by model id.

    Not internally serialized: callers must hold the single-flight gate around
    ``get``/``preload`` so loads and evictions never race an inference.
    """

    def __init__(self, settings: Settings, *, factory: TranscriberFactory | None = None) -> None:
        self.settings = settings
        self._factory = factory or self._default_factory
        self._allowed = set(settings.models) | {settings.model_id}
        self._loaded: dict[str, Transcriber] = {}
        self._last_used: dict[str, float] = {}
        self._shared_vad: object | None = None
        # Captured from the first loaded model so /health can report the device and
        # provider even after the default model has been evicted.
        self.device: str = settings.execution_provider
        self.provider_active = False

    @property
    def multi(self) -> bool:
        """Whether multi-model mode is active (a non-empty allow-list)."""
        return bool(self.settings.models)

    def allowed_ids(self) -> list[str]:
        return sorted(self._allowed)

    def loaded_ids(self) -> list[str]:
        return list(self._loaded)

    def resolve(self, requested: str | None) -> str:
        """Map a request ``model`` value to a canonical model id.

        Single mode always returns the configured model_id. Multi mode returns the
        requested id if it is in the allow-list, the default for an empty request,
        and raises ``UnknownModelError`` otherwise.
        """
        if not self.multi or not requested:
            return self.settings.model_id
        if requested not in self._allowed:
            raise UnknownModelError(requested, self._allowed)
        return requested

    def get(self, model_id: str, *, now: float | None = None) -> Transcriber:
        """Return the loaded Transcriber for ``model_id``, loading it if absent.

        Evicts idle (TTL) and least-recently-used (capacity) models as needed. Must
        be called only while holding the single-flight gate.
        """
        now = time.monotonic() if now is None else now
        self._evict_stale(now=now, keep=model_id)
        if model_id not in self._loaded:
            self._evict_for_capacity(keep=model_id)
            transcriber = self._factory(model_id)
            self._loaded[model_id] = transcriber
            self.device = transcriber.device
            self.provider_active = transcriber.provider_active
        self._last_used[model_id] = now
        return self._loaded[model_id]

    def preload(self) -> Transcriber:
        """Load the default model at startup, running the GPU fail-loud check."""
        return self.get(self.settings.model_id)

    def _default_factory(self, model_id: str) -> Transcriber:
        if self._shared_vad is None:
            self._shared_vad = build_vad(self.settings)
        return Transcriber(self.settings, model_id=model_id, vad=self._shared_vad)

    def _evict_stale(self, *, now: float, keep: str) -> None:
        ttl = self.settings.model_ttl_s
        if ttl is None:
            return
        for mid in [m for m, used in self._last_used.items() if m != keep and now - used > ttl]:
            self._evict(mid)

    def _evict_for_capacity(self, *, keep: str) -> None:
        # About to add `keep`; evict the least-recently-used until there is room.
        cap = max(1, self.settings.max_resident_models)
        while len(self._loaded) >= cap and self._loaded:
            lru = min(self._last_used, key=lambda m: self._last_used[m])
            self._evict(lru)

    def _evict(self, model_id: str) -> None:
        # Dropping all references frees the onnxruntime sessions (and GPU memory).
        # Safe because eviction only runs under the single-flight gate, so the model
        # is never mid-inference.
        self._loaded.pop(model_id, None)
        self._last_used.pop(model_id, None)
        logger.info("evicted model %s", model_id)

    @classmethod
    def seeded(
        cls, settings: Settings, transcriber: Transcriber, *, now: float = 0.0
    ) -> ModelRegistry:
        """Registry pre-populated with one already-built Transcriber.

        Used to inject a (possibly fake) model without a real load, e.g. in tests
        or when a Transcriber is constructed outside the registry.
        """
        reg = cls(settings, factory=lambda _model_id: transcriber)
        reg._loaded[settings.model_id] = transcriber
        reg._last_used[settings.model_id] = now
        reg.device = transcriber.device
        reg.provider_active = transcriber.provider_active
        return reg
