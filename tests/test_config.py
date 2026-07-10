"""Settings parsing tests."""

from __future__ import annotations

import pytest

from parascribe.config import Settings


class TestModelsParsing:
    def test_comma_separated_string_is_split(self):
        s = Settings(execution_provider="cpu", models="a, b ,c")
        assert s.models == ["a", "b", "c"]

    def test_empty_string_is_empty_list(self):
        s = Settings(execution_provider="cpu", models="")
        assert s.models == []

    def test_list_input_passes_through(self):
        s = Settings(execution_provider="cpu", models=["a", "b"])
        assert s.models == ["a", "b"]

    def test_comma_separated_env_var(self, monkeypatch):
        monkeypatch.setenv("PARASCRIBE_MODELS", "x,y,z")
        monkeypatch.setenv("PARASCRIBE_EXECUTION_PROVIDER", "cpu")
        assert Settings().models == ["x", "y", "z"]


class TestUsageSettings:
    def test_defaults_bill_transcription_by_token(self):
        s = Settings(execution_provider="cpu")
        assert s.transcription_usage_unit == "token"
        assert s.transcription_usage_multiplier == 1.0

    def test_diarization_default_multiplier_is_five(self):
        s = Settings(execution_provider="cpu")
        assert s.diarization_usage_unit == "token"
        assert s.diarization_usage_multiplier == 5.0

    def test_multiplier_accepts_fractions_from_env(self, monkeypatch):
        monkeypatch.setenv("PARASCRIBE_EXECUTION_PROVIDER", "cpu")
        monkeypatch.setenv("PARASCRIBE_TRANSCRIPTION_USAGE_MULTIPLIER", "0.25")
        assert Settings().transcription_usage_multiplier == 0.25

    def test_unit_from_env(self, monkeypatch):
        monkeypatch.setenv("PARASCRIBE_EXECUTION_PROVIDER", "cpu")
        monkeypatch.setenv("PARASCRIBE_DIARIZATION_USAGE_UNIT", "file_duration")
        assert Settings().diarization_usage_unit == "file_duration"

    def test_audio_input_defaults_to_openai_parity(self):
        s = Settings(execution_provider="cpu")
        assert s.audio_input_usage_unit == "file_duration"
        assert s.audio_input_usage_multiplier == 10.0
        assert s.audio_input_usage_field == "input_tokens"

    def test_audio_input_field_from_env(self, monkeypatch):
        monkeypatch.setenv("PARASCRIBE_EXECUTION_PROVIDER", "cpu")
        monkeypatch.setenv("PARASCRIBE_AUDIO_INPUT_USAGE_FIELD", "output_tokens")
        assert Settings().audio_input_usage_field == "output_tokens"


class TestSecretResolution:
    def test_direct_api_key_is_returned(self):
        s = Settings(execution_provider="cpu", api_key="k")
        assert s.resolved_api_key() == "k"

    def test_neither_set_returns_none(self):
        s = Settings(execution_provider="cpu")
        assert s.resolved_api_key() is None

    def test_api_key_file_is_read_and_stripped(self, tmp_path):
        keyfile = tmp_path / "key"
        keyfile.write_text("  sekrit\n")
        s = Settings(execution_provider="cpu", api_key_file=keyfile)
        assert s.resolved_api_key() == "sekrit"

    def test_direct_api_key_wins_over_file(self, tmp_path):
        keyfile = tmp_path / "key"
        keyfile.write_text("from-file")
        s = Settings(execution_provider="cpu", api_key="direct", api_key_file=keyfile)
        assert s.resolved_api_key() == "direct"

    def test_missing_api_key_file_raises(self, tmp_path):
        # A typo'd key-file path must not silently disable auth.
        s = Settings(execution_provider="cpu", api_key_file=tmp_path / "nope")
        with pytest.raises(ValueError, match="missing file"):
            s.resolved_api_key()

    def test_empty_api_key_file_raises(self, tmp_path):
        keyfile = tmp_path / "key"
        keyfile.write_text("  \n")
        s = Settings(execution_provider="cpu", api_key_file=keyfile)
        with pytest.raises(ValueError, match="empty file"):
            s.resolved_api_key()

    def test_missing_hf_token_file_raises(self, tmp_path):
        s = Settings(execution_provider="cpu", hf_token_file=tmp_path / "nope")
        with pytest.raises(ValueError, match="missing file"):
            s.resolved_hf_token()
