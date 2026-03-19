"""Tests for froggy.config — YAML config persistence."""

from __future__ import annotations

from froggy.config import config_path, get_config, load_config, save_config, set_config


class TestConfigPath:
    def test_returns_config_yaml_under_froggy_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path / "custom"))
        assert config_path() == tmp_path / "custom" / "config.yaml"

    def test_respects_froggy_home_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path / "alt"))
        p = config_path()
        assert "alt" in str(p)
        assert p.name == "config.yaml"

    def test_default_path_ends_with_config_yaml(self, monkeypatch):
        monkeypatch.delenv("FROGGY_HOME", raising=False)
        p = config_path()
        assert p.name == "config.yaml"
        assert ".froggy" in str(p)


class TestLoadConfig:
    def test_returns_empty_dict_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path / "nope"))
        assert load_config() == {}

    def test_returns_empty_dict_when_file_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        cfg = tmp_path / "config.yaml"
        cfg.write_text("")
        assert load_config() == {}

    def test_returns_parsed_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        cfg = tmp_path / "config.yaml"
        cfg.write_text("device: mps\nbackend: mlx\n")
        result = load_config()
        assert result == {"device": "mps", "backend": "mlx"}

    def test_returns_empty_dict_for_invalid_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        cfg = tmp_path / "config.yaml"
        cfg.write_text("not a dict: [")
        assert load_config() == {}


class TestSaveConfig:
    def test_writes_yaml_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        data = {"device": "cuda", "limit": 10}
        save_config(data)
        loaded = load_config()
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path, monkeypatch):
        deep = tmp_path / "a" / "b"
        monkeypatch.setenv("FROGGY_HOME", str(deep))
        save_config({"x": 1})
        assert (deep / "config.yaml").exists()


class TestGetConfig:
    def test_returns_value_for_existing_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        save_config({"device": "mps"})
        assert get_config("device") == "mps"

    def test_returns_default_for_missing_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        save_config({"device": "mps"})
        assert get_config("backend", "auto") == "auto"

    def test_returns_none_for_missing_key_no_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        assert get_config("anything") is None


class TestSetConfig:
    def test_persists_key_value(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        set_config("device", "mps")
        assert get_config("device") == "mps"

    def test_preserves_existing_keys(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        set_config("device", "cpu")
        set_config("backend", "mlx")
        assert get_config("device") == "cpu"
        assert get_config("backend") == "mlx"

    def test_creates_file_if_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path / "new"))
        set_config("key", "val")
        assert (tmp_path / "new" / "config.yaml").exists()
        assert get_config("key") == "val"
