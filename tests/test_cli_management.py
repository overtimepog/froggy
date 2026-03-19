"""Tests for model management CLI commands: list, remove, info."""

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from froggy.cli import cli

# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

SAMPLE_MODELS = [
    {
        "name": "org--gpt2",
        "path": "/fake/models/org--gpt2",
        "format": "SafeTensors",
        "size": 548_000_000,
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "modified": 1710000000.0,
    },
    {
        "name": "org--llama-7b",
        "path": "/fake/models/org--llama-7b",
        "format": "GGUF",
        "size": 4_200_000_000,
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "modified": 1710100000.0,
    },
]

SAMPLE_INFO = {
    "name": "org--gpt2",
    "path": "/fake/models/org--gpt2",
    "format": "SafeTensors",
    "size": 548_000_000,
    "model_type": "gpt2",
    "architectures": ["GPT2LMHeadModel"],
    "file_count": 5,
    "has_gguf": False,
    "has_lora": False,
    "modified": 1710000000.0,
}


# ---------------------------------------------------------------------------
# Help text tests
# ---------------------------------------------------------------------------


class TestListHelp:
    def test_list_help_shows_json_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0
        assert "--json" in result.output

    def test_list_help_shows_description(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0
        assert "List downloaded models" in result.output


class TestRemoveHelp:
    def test_remove_help_shows_name_argument(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["remove", "--help"])
        assert result.exit_code == 0
        assert "NAME" in result.output

    def test_remove_help_shows_yes_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["remove", "--help"])
        assert result.exit_code == 0
        assert "--yes" in result.output


class TestInfoHelp:
    def test_info_help_shows_name_argument(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--help"])
        assert result.exit_code == 0
        assert "NAME" in result.output

    def test_info_help_shows_description(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--help"])
        assert result.exit_code == 0
        assert "Show detailed information about a model" in result.output


class TestMainHelpIntegration:
    def test_help_lists_list_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "list" in result.output

    def test_help_lists_remove_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "remove" in result.output

    def test_help_lists_info_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "info" in result.output


# ---------------------------------------------------------------------------
# list command tests
# ---------------------------------------------------------------------------


class TestListCommand:
    @patch("froggy.cli.list_models", return_value=SAMPLE_MODELS)
    @patch("froggy.cli.models_dir", return_value=Path("/fake/models"))
    def test_list_renders_table_with_model_names(self, _mock_dir, _mock_list):
        runner = CliRunner()
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "org--gpt2" in result.output
        assert "org--llama-7b" in result.output

    @patch("froggy.cli.list_models", return_value=SAMPLE_MODELS)
    @patch("froggy.cli.models_dir", return_value=Path("/fake/models"))
    def test_list_table_contains_format_and_size(self, _mock_dir, _mock_list):
        runner = CliRunner()
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "SafeTensors" in result.output
        assert "GGUF" in result.output

    @patch("froggy.cli.list_models", return_value=[])
    @patch("froggy.cli.models_dir", return_value=Path("/fake/models"))
    def test_list_empty_shows_no_models_message(self, _mock_dir, _mock_list):
        runner = CliRunner()
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No models" in result.output

    @patch("froggy.cli.list_models", return_value=SAMPLE_MODELS)
    @patch("froggy.cli.models_dir", return_value=Path("/fake/models"))
    def test_list_json_outputs_valid_json(self, _mock_dir, _mock_list):
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "org--gpt2"


# ---------------------------------------------------------------------------
# remove command tests
# ---------------------------------------------------------------------------


class TestRemoveCommand:
    @patch("froggy.cli.remove_model", return_value=548_000_000)
    @patch("froggy.cli.model_info", return_value=SAMPLE_INFO)
    @patch("froggy.cli.models_dir", return_value=Path("/fake/models"))
    def test_remove_with_yes_flag(self, _mock_dir, _mock_info, _mock_rm):
        runner = CliRunner()
        result = runner.invoke(cli, ["remove", "org--gpt2", "--yes"])
        assert result.exit_code == 0
        assert "Removed" in result.output
        _mock_rm.assert_called_once()

    @patch("froggy.cli.remove_model", return_value=548_000_000)
    @patch("froggy.cli.model_info", return_value=SAMPLE_INFO)
    @patch("froggy.cli.models_dir", return_value=Path("/fake/models"))
    def test_remove_with_confirmation_prompt(self, _mock_dir, _mock_info, _mock_rm):
        runner = CliRunner()
        result = runner.invoke(cli, ["remove", "org--gpt2"], input="y\n")
        assert result.exit_code == 0
        assert "Removed" in result.output
        _mock_rm.assert_called_once()

    @patch("froggy.cli.remove_model", return_value=548_000_000)
    @patch("froggy.cli.model_info", return_value=SAMPLE_INFO)
    @patch("froggy.cli.models_dir", return_value=Path("/fake/models"))
    def test_remove_declined_aborts(self, _mock_dir, _mock_info, _mock_rm):
        runner = CliRunner()
        result = runner.invoke(cli, ["remove", "org--gpt2"], input="n\n")
        assert result.exit_code == 1
        _mock_rm.assert_not_called()

    @patch(
        "froggy.cli.model_info",
        side_effect=ValueError("Model 'nope' not found in /fake/models"),
    )
    @patch("froggy.cli.models_dir", return_value=Path("/fake/models"))
    def test_remove_not_found_error(self, _mock_dir, _mock_info):
        runner = CliRunner()
        result = runner.invoke(cli, ["remove", "nope", "--yes"])
        assert result.exit_code != 0
        assert "nope" in result.output

    @patch("froggy.cli.remove_model", return_value=548_000_000)
    @patch("froggy.cli.model_info", return_value=SAMPLE_INFO)
    @patch("froggy.cli.models_dir", return_value=Path("/fake/models"))
    def test_remove_shows_freed_space(self, _mock_dir, _mock_info, _mock_rm):
        runner = CliRunner()
        result = runner.invoke(cli, ["remove", "org--gpt2", "--yes"])
        assert result.exit_code == 0
        # Should mention freed size in some human-readable form
        assert "MB" in result.output or "GB" in result.output


# ---------------------------------------------------------------------------
# info command tests
# ---------------------------------------------------------------------------


class TestInfoCommand:
    @patch("froggy.cli.model_info", return_value=SAMPLE_INFO)
    @patch("froggy.cli.models_dir", return_value=Path("/fake/models"))
    def test_info_displays_metadata(self, _mock_dir, _mock_info):
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "org--gpt2"])
        assert result.exit_code == 0
        assert "org--gpt2" in result.output
        assert "SafeTensors" in result.output

    @patch("froggy.cli.model_info", return_value=SAMPLE_INFO)
    @patch("froggy.cli.models_dir", return_value=Path("/fake/models"))
    def test_info_shows_size_and_path(self, _mock_dir, _mock_info):
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "org--gpt2"])
        assert result.exit_code == 0
        assert "/fake/models/org--gpt2" in result.output

    @patch(
        "froggy.cli.model_info",
        side_effect=ValueError("Model 'nope' not found in /fake/models"),
    )
    @patch("froggy.cli.models_dir", return_value=Path("/fake/models"))
    def test_info_not_found_error(self, _mock_dir, _mock_info):
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "nope"])
        assert result.exit_code != 0
        assert "nope" in result.output
