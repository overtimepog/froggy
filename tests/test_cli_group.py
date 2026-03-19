"""Tests for the CLI group structure (click group with subcommands)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
from click.testing import CliRunner

from froggy.cli import cli


class TestCliGroupHelp:
    """Verify --help shows the group structure with chat subcommand."""

    def test_help_lists_chat_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "chat" in result.output

    def test_help_shows_group_header(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # Group help includes "Commands:" section
        assert "Commands:" in result.output


class TestChatSubcommandHelp:
    """Verify chat --help shows the expected options."""

    def test_chat_help_shows_models_dir(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--models-dir" in result.output

    def test_chat_help_shows_device(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--device" in result.output

    def test_chat_help_shows_tools_dir(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--tools-dir" in result.output


class TestBareInvocation:
    """Bare `froggy` (no subcommand) should invoke chat."""

    @patch("froggy.cli.discover_models")
    @patch("froggy.cli.discover_ollama_models", return_value=[])
    @patch("froggy.cli.select_model", return_value=None)
    def test_bare_invocation_enters_chat_path(
        self, mock_select, mock_ollama, mock_discover
    ):
        """Bare invocation triggers discover_models, proving chat was invoked."""
        mock_discover.return_value = [MagicMock(label="test", model_type="test")]
        runner = CliRunner()
        runner.invoke(cli, [])
        mock_discover.assert_called_once()


# ---------------------------------------------------------------------------
# Download subcommand tests
# ---------------------------------------------------------------------------


class TestDownloadSubcommandHelp:
    """Verify download command appears in --help and has proper options."""

    def test_help_lists_download_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "download" in result.output

    def test_download_help_shows_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "--pick" in result.output
        assert "SOURCE" in result.output


class TestDownloadCliRunner:
    """CliRunner tests exercising the download command flow."""

    @patch("froggy.cli.download_model")
    def test_download_basic_success(self, mock_dl):
        """Basic download invocation calls download_model and exits 0."""
        mock_dl.return_value = Path("/tmp/test_models/org--model")
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "org/model"])
        assert result.exit_code == 0
        mock_dl.assert_called_once_with("org/model", fmt="auto")

    @patch("froggy.cli.download_model")
    def test_download_with_format_flag(self, mock_dl):
        mock_dl.return_value = Path("/tmp/test_models/org--model")
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "org/model", "--format", "gguf"])
        assert result.exit_code == 0
        mock_dl.assert_called_once_with("org/model", fmt="gguf")

    @patch("froggy.cli.download_model", side_effect=click.ClickException("No compatible"))
    def test_download_error_displays_message(self, _mock_dl):
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "org/model"])
        assert result.exit_code != 0
        assert "No compatible" in result.output

    @patch("froggy.cli.list_variants")
    def test_pick_mode_shows_table(self, mock_variants):
        """--pick mode renders a table of variants."""
        mock_variants.return_value = [
            {"type": "mlx", "repo": "mlx-community/model-4bit", "filename": None, "size": None, "quant": None},
            {"type": "gguf", "repo": "org/model", "filename": "model-Q4_K_M.gguf", "size": 4_000_000_000, "quant": "Q4_K_M"},
            {"type": "safetensors", "repo": "org/model", "filename": None, "size": None, "quant": None},
        ]
        runner = CliRunner()
        # Input "1" to select MLX variant, then mock download_model to succeed
        with patch("froggy.cli.download_model") as mock_dl:
            mock_dl.return_value = Path("/tmp/test")
            result = runner.invoke(cli, ["download", "org/model", "--pick"], input="1\n")

        assert "Available Variants" in result.output or "mlx" in result.output
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Recommend subcommand tests
# ---------------------------------------------------------------------------


class TestRecommendSubcommandHelp:
    """Verify recommend command appears in --help and has proper options."""

    def test_help_lists_recommend_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "recommend" in result.output

    def test_recommend_help_shows_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--help"])
        assert result.exit_code == 0
        assert "--limit" in result.output
        assert "--use-case" in result.output
        assert "--json" in result.output


class TestRecommendCliRunner:
    """CliRunner tests exercising the recommend command flow."""

    @patch("froggy.cli.llmfit_recommend")
    @patch("froggy.cli.ensure_llmfit")
    def test_recommend_renders_table(self, mock_ensure, mock_recommend):
        mock_ensure.return_value = Path("/fake/llmfit")
        mock_recommend.return_value = [
            {
                "name": "llama-3.2-3b",
                "score": 95,
                "best_quant": "Q4_K_M",
                "estimated_tps": 45.2,
                "fit_level": "full",
                "run_mode": "gpu",
                "memory_required_gb": 2.1,
            }
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert "llama-3.2-3b" in result.output
        assert "95" in result.output
        assert "Q4_K_M" in result.output

    @patch("froggy.cli.llmfit_recommend")
    @patch("froggy.cli.ensure_llmfit")
    def test_recommend_json_output(self, mock_ensure, mock_recommend):
        mock_ensure.return_value = Path("/fake/llmfit")
        mock_recommend.return_value = [
            {
                "name": "llama-3.2-3b",
                "score": 95,
                "best_quant": "Q4_K_M",
                "estimated_tps": 45.2,
                "fit_level": "full",
                "run_mode": "gpu",
                "memory_required_gb": 2.1,
            }
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["recommend", "--json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["name"] == "llama-3.2-3b"

    @patch("froggy.cli.ensure_llmfit", return_value=None)
    def test_recommend_llmfit_unavailable(self, _mock_ensure):
        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code != 0
        assert "llmfit" in result.output.lower() or "error" in result.output.lower()

    @patch("froggy.cli.llmfit_recommend")
    @patch("froggy.cli.ensure_llmfit")
    def test_recommend_empty_results(self, mock_ensure, mock_recommend):
        mock_ensure.return_value = Path("/fake/llmfit")
        mock_recommend.return_value = []
        runner = CliRunner()
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert "No recommendations" in result.output

    @patch("froggy.cli.llmfit_recommend")
    @patch("froggy.cli.ensure_llmfit")
    def test_recommend_passes_limit_and_use_case(self, mock_ensure, mock_recommend):
        mock_ensure.return_value = Path("/fake/llmfit")
        mock_recommend.return_value = []
        runner = CliRunner()
        runner.invoke(cli, ["recommend", "--limit", "3", "--use-case", "coding"])
        mock_recommend.assert_called_once_with(
            Path("/fake/llmfit"), limit=3, use_case="coding",
        )
