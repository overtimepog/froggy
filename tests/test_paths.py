"""Tests for froggy.paths — managed home directory resolution."""

from pathlib import Path

from froggy.paths import ensure_froggy_home, froggy_home, models_dir


class TestFroggyHome:
    """Tests for froggy_home() default and override behavior."""

    def test_default_returns_dot_froggy(self, monkeypatch):
        monkeypatch.delenv("FROGGY_HOME", raising=False)
        result = froggy_home()
        assert result == Path.home() / ".froggy"

    def test_respects_froggy_home_env_var(self, monkeypatch, tmp_path):
        custom = tmp_path / "custom_froggy"
        monkeypatch.setenv("FROGGY_HOME", str(custom))
        assert froggy_home() == custom

    def test_returns_path_object(self, monkeypatch):
        monkeypatch.delenv("FROGGY_HOME", raising=False)
        assert isinstance(froggy_home(), Path)


class TestModelsDir:
    """Tests for models_dir() — always a child of froggy_home()."""

    def test_default_models_dir(self, monkeypatch):
        monkeypatch.delenv("FROGGY_HOME", raising=False)
        assert models_dir() == Path.home() / ".froggy" / "models"

    def test_respects_froggy_home_override(self, monkeypatch, tmp_path):
        custom = tmp_path / "custom_froggy"
        monkeypatch.setenv("FROGGY_HOME", str(custom))
        assert models_dir() == custom / "models"


class TestEnsureFroggyHome:
    """Tests for ensure_froggy_home() — directory creation and idempotency."""

    def test_creates_home_and_models_dirs(self, monkeypatch, tmp_path):
        home = tmp_path / "froggy_test"
        monkeypatch.setenv("FROGGY_HOME", str(home))
        ensure_froggy_home()
        assert home.is_dir()
        assert (home / "models").is_dir()

    def test_idempotent(self, monkeypatch, tmp_path):
        home = tmp_path / "froggy_test"
        monkeypatch.setenv("FROGGY_HOME", str(home))
        ensure_froggy_home()
        ensure_froggy_home()  # second call should not raise
        assert home.is_dir()
        assert (home / "models").is_dir()

    def test_returns_home_path(self, monkeypatch, tmp_path):
        home = tmp_path / "froggy_test"
        monkeypatch.setenv("FROGGY_HOME", str(home))
        result = ensure_froggy_home()
        assert result == home
