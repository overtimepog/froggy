"""Tests for the ``froggy config`` CLI command."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from froggy.cli import cli


class TestConfigHelp:
    def test_config_help_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_config_appears_in_main_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "config" in result.output


class TestConfigShowAll:
    def test_bare_config_shows_all_settings(self):
        runner = CliRunner()
        with patch("froggy.cli.load_config", return_value={"device": "mps", "backend": "mlx"}):
            result = runner.invoke(cli, ["config"])
        assert result.exit_code == 0
        assert "device" in result.output
        assert "mps" in result.output

    def test_bare_config_empty(self):
        runner = CliRunner()
        with patch("froggy.cli.load_config", return_value={}):
            result = runner.invoke(cli, ["config"])
        assert result.exit_code == 0
        assert "No configuration" in result.output or "{}" in result.output


class TestConfigGet:
    def test_get_existing_key(self):
        runner = CliRunner()
        with patch("froggy.cli.get_config", return_value="mps"):
            result = runner.invoke(cli, ["config", "get", "device"])
        assert result.exit_code == 0
        assert "mps" in result.output

    def test_get_missing_key(self):
        runner = CliRunner()
        with patch("froggy.cli.get_config", return_value=None):
            result = runner.invoke(cli, ["config", "get", "missing_key"])
        assert result.exit_code == 0
        assert "not set" in result.output.lower()


class TestConfigSet:
    def test_set_calls_set_config(self):
        runner = CliRunner()
        with patch("froggy.cli.set_config") as mock_set:
            result = runner.invoke(cli, ["config", "set", "device", "mps"])
        assert result.exit_code == 0
        mock_set.assert_called_once_with("device", "mps")

    def test_set_confirms_in_output(self):
        runner = CliRunner()
        with patch("froggy.cli.set_config"):
            result = runner.invoke(cli, ["config", "set", "backend", "mlx"])
        assert result.exit_code == 0
        assert "backend" in result.output
        assert "mlx" in result.output
