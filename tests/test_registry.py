"""Unit tests for the model registry (resolution, lazy load, LRU/TTL eviction)."""

from __future__ import annotations

import pytest

from parascribe.config import Settings
from parascribe.registry import ModelRegistry, UnknownModelError


class FakeTranscriber:
    device = "cpu"
    provider_active = True

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id


def counting_factory():
    """A factory returning FakeTranscribers, recording the model ids it loaded."""
    loaded: list[str] = []

    def factory(model_id: str) -> FakeTranscriber:
        loaded.append(model_id)
        return FakeTranscriber(model_id)

    return factory, loaded


def settings(**overrides) -> Settings:
    base = {"execution_provider": "cpu", "model_id": "a"}
    return Settings(**{**base, **overrides})


def registry(**overrides) -> tuple[ModelRegistry, list[str]]:
    factory, loaded = counting_factory()
    return ModelRegistry(settings(**overrides), factory=factory), loaded


class TestResolve:
    def test_single_mode_ignores_requested_model(self):
        reg, _ = registry()  # no `models` => single mode
        assert reg.multi is False
        assert reg.resolve("anything-at-all") == "a"

    def test_multi_mode_returns_allowed_model(self):
        reg, _ = registry(models=["a", "b"])
        assert reg.resolve("b") == "b"

    def test_multi_mode_unknown_model_raises(self):
        reg, _ = registry(models=["a", "b"])
        with pytest.raises(UnknownModelError):
            reg.resolve("c")

    def test_multi_mode_empty_request_resolves_to_default(self):
        reg, _ = registry(models=["a", "b"])
        assert reg.resolve(None) == "a"
        assert reg.resolve("") == "a"

    def test_default_model_is_always_allowed(self):
        # model_id not listed in `models` is still implicitly allowed.
        reg, _ = registry(models=["b", "c"], model_id="a")
        assert reg.resolve("a") == "a"


class TestLoading:
    def test_get_loads_once_then_caches(self):
        reg, loaded = registry(models=["a", "b"])
        reg.get("a", now=0)
        reg.get("a", now=1)
        assert loaded == ["a"]  # second get is a cache hit

    def test_preload_loads_default_and_records_provider(self):
        reg, loaded = registry(models=["a", "b"])
        reg.preload()
        assert loaded == ["a"]
        assert reg.provider_active is True
        assert reg.device == "cpu"


class TestCapacityEviction:
    def test_cap_one_swaps(self):
        reg, loaded = registry(models=["a", "b"], max_resident_models=1)
        reg.get("a", now=0)
        reg.get("b", now=1)
        assert reg.loaded_ids() == ["b"]  # a evicted
        assert loaded == ["a", "b"]

    def test_cap_two_keeps_both(self):
        reg, _ = registry(models=["a", "b", "c"], max_resident_models=2)
        reg.get("a", now=0)
        reg.get("b", now=1)
        assert set(reg.loaded_ids()) == {"a", "b"}

    def test_cap_two_evicts_least_recently_used(self):
        reg, _ = registry(models=["a", "b", "c"], max_resident_models=2)
        reg.get("a", now=0)
        reg.get("b", now=1)
        reg.get("a", now=2)  # touch a so b becomes the LRU
        reg.get("c", now=3)  # forces an eviction
        assert set(reg.loaded_ids()) == {"a", "c"}  # b evicted, not a

    def test_reloading_evicted_model_calls_factory_again(self):
        reg, loaded = registry(models=["a", "b"], max_resident_models=1)
        reg.get("a", now=0)
        reg.get("b", now=1)  # evicts a
        reg.get("a", now=2)  # must reload a
        assert loaded == ["a", "b", "a"]


class TestTtlEviction:
    def test_idle_model_evicted_on_next_get(self):
        reg, _ = registry(models=["a", "b"], max_resident_models=5, model_ttl_s=10.0)
        reg.get("a", now=0)
        reg.get("b", now=1)
        reg.get("a", now=20)  # b idle 19s > 10s TTL -> evicted even though a is a hit
        assert reg.loaded_ids() == ["a"]

    def test_model_within_ttl_survives(self):
        reg, _ = registry(models=["a", "b"], max_resident_models=5, model_ttl_s=10.0)
        reg.get("a", now=0)
        reg.get("b", now=1)
        reg.get("a", now=5)  # b idle 4s < 10s TTL
        assert set(reg.loaded_ids()) == {"a", "b"}

    def test_no_ttl_keeps_models_until_capacity(self):
        reg, _ = registry(models=["a", "b"], max_resident_models=5)  # model_ttl_s None
        reg.get("a", now=0)
        reg.get("b", now=1_000_000)
        assert set(reg.loaded_ids()) == {"a", "b"}


class TestSeeded:
    def test_seeded_registry_serves_injected_transcriber(self):
        t = FakeTranscriber("a")
        reg = ModelRegistry.seeded(settings(), t)
        assert reg.get("a") is t
        assert reg.provider_active is True
        assert reg.device == "cpu"
