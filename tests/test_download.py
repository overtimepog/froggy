"""Tests for froggy.download building blocks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest

from froggy.download import (
    ParsedSource,
    detect_platform,
    download_model,
    find_mlx_repo,
    get_system_memory,
    list_gguf_files,
    list_variants,
    parse_source,
    select_gguf_file,
)

# ---------------------------------------------------------------------------
# TestParseSource
# ---------------------------------------------------------------------------


class TestParseSource:
    """parse_source() handles bare repo IDs, full URLs, and URL sub-paths."""

    def test_bare_repo_id(self):
        result = parse_source("org/model")
        assert result == ParsedSource(repo_id="org/model", filename=None, revision=None)

    def test_full_url(self):
        result = parse_source("https://huggingface.co/org/model")
        assert result.repo_id == "org/model"
        assert result.filename is None
        assert result.revision is None

    def test_url_with_blob_file(self):
        result = parse_source(
            "https://huggingface.co/org/model/blob/main/file.gguf"
        )
        assert result.repo_id == "org/model"
        assert result.filename == "file.gguf"
        assert result.revision == "main"

    def test_url_with_tree_revision(self):
        result = parse_source("https://huggingface.co/org/model/tree/v2")
        assert result.repo_id == "org/model"
        assert result.filename is None
        assert result.revision == "v2"

    def test_url_trailing_slash(self):
        result = parse_source("https://huggingface.co/org/model/")
        assert result.repo_id == "org/model"

    def test_url_with_query_params(self):
        result = parse_source("https://huggingface.co/org/model?foo=bar")
        assert result.repo_id == "org/model"

    def test_url_blob_nested_path(self):
        result = parse_source(
            "https://huggingface.co/org/model/blob/main/subdir/file.gguf"
        )
        assert result.repo_id == "org/model"
        assert result.filename == "subdir/file.gguf"
        assert result.revision == "main"

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_source("not-a-repo")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_source("")


# ---------------------------------------------------------------------------
# TestDetectPlatform
# ---------------------------------------------------------------------------


class TestDetectPlatform:
    """detect_platform() returns 'mlx' on macOS ARM64, 'gguf' otherwise."""

    @patch("froggy.download.platform")
    def test_macos_arm64_returns_mlx(self, mock_platform):
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"
        assert detect_platform() == "mlx"

    @patch("froggy.download.platform")
    def test_linux_x86_returns_gguf(self, mock_platform):
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"
        assert detect_platform() == "gguf"

    @patch("froggy.download.platform")
    def test_macos_x86_returns_gguf(self, mock_platform):
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "x86_64"
        assert detect_platform() == "gguf"

    @patch("froggy.download.platform")
    def test_windows_returns_gguf(self, mock_platform):
        mock_platform.system.return_value = "Windows"
        mock_platform.machine.return_value = "AMD64"
        assert detect_platform() == "gguf"


# ---------------------------------------------------------------------------
# TestFindMlxRepo
# ---------------------------------------------------------------------------


class TestFindMlxRepo:
    """find_mlx_repo() searches mlx-community naming patterns."""

    def _make_api(self, found_repos: set[str]):
        """Return a mock HfApi that succeeds for repos in *found_repos*."""
        from huggingface_hub.utils import RepositoryNotFoundError

        api = MagicMock()

        def model_info_side_effect(repo_id, **kwargs):
            if repo_id in found_repos:
                return MagicMock()
            raise RepositoryNotFoundError(f"{repo_id} not found")

        api.model_info.side_effect = model_info_side_effect
        return api

    def test_finds_4bit_variant(self):
        api = self._make_api({"mlx-community/model-4bit"})
        result = find_mlx_repo("org/model", api)
        assert result == "mlx-community/model-4bit"

    def test_finds_8bit_when_no_4bit(self):
        api = self._make_api({"mlx-community/model-8bit"})
        result = find_mlx_repo("org/model", api)
        assert result == "mlx-community/model-8bit"

    def test_returns_none_when_nothing_found(self):
        api = self._make_api(set())
        result = find_mlx_repo("org/model", api)
        assert result is None

    def test_returns_first_match_in_priority_order(self):
        # Both 4bit and 8bit exist — should get 4bit first
        api = self._make_api(
            {"mlx-community/model-4bit", "mlx-community/model-8bit"}
        )
        result = find_mlx_repo("org/model", api)
        assert result == "mlx-community/model-4bit"

    def test_handles_hf_http_error(self):
        """HfHubHTTPError is also caught gracefully."""
        from huggingface_hub.utils import HfHubHTTPError

        api = MagicMock()
        api.model_info.side_effect = HfHubHTTPError("rate limited")
        result = find_mlx_repo("org/model", api)
        assert result is None


# ---------------------------------------------------------------------------
# TestListGgufFiles
# ---------------------------------------------------------------------------


def _make_sibling(filename: str, size: int = 1_000_000):
    """Create a mock sibling object."""
    s = MagicMock()
    s.rfilename = filename
    s.size = size
    return s


class TestListGgufFiles:
    """list_gguf_files() filters .gguf siblings and parses quant tags."""

    def test_filters_gguf_files(self):
        api = MagicMock()
        info = MagicMock()
        info.siblings = [
            _make_sibling("model-Q4_K_M.gguf", 4_000_000_000),
            _make_sibling("config.json", 1000),
            _make_sibling("model-Q8_0.gguf", 8_000_000_000),
        ]
        api.model_info.return_value = info

        result = list_gguf_files("org/model", api)
        assert len(result) == 2
        filenames = {f["filename"] for f in result}
        assert filenames == {"model-Q4_K_M.gguf", "model-Q8_0.gguf"}

    def test_parses_quant_from_filename(self):
        api = MagicMock()
        info = MagicMock()
        info.siblings = [
            _make_sibling("model-Q4_K_M.gguf"),
            _make_sibling("model-Q8_0.gguf"),
            _make_sibling("model-f16.gguf"),
        ]
        api.model_info.return_value = info

        result = list_gguf_files("org/model", api)
        quants = {f["filename"]: f["quant"] for f in result}
        assert quants["model-Q4_K_M.gguf"] == "Q4_K_M"
        assert quants["model-Q8_0.gguf"] == "Q8_0"
        assert quants["model-f16.gguf"] == "f16"

    def test_unknown_quant_when_no_match(self):
        api = MagicMock()
        info = MagicMock()
        info.siblings = [_make_sibling("model.gguf")]
        api.model_info.return_value = info

        result = list_gguf_files("org/model", api)
        assert result[0]["quant"] == "unknown"

    def test_empty_when_no_gguf_files(self):
        api = MagicMock()
        info = MagicMock()
        info.siblings = [_make_sibling("config.json"), _make_sibling("model.bin")]
        api.model_info.return_value = info

        result = list_gguf_files("org/model", api)
        assert result == []

    def test_includes_size(self):
        api = MagicMock()
        info = MagicMock()
        info.siblings = [_make_sibling("model-Q4_K_M.gguf", 4_200_000_000)]
        api.model_info.return_value = info

        result = list_gguf_files("org/model", api)
        assert result[0]["size"] == 4_200_000_000

    def test_iq_quant_parsing(self):
        api = MagicMock()
        info = MagicMock()
        info.siblings = [_make_sibling("model-IQ2_XXS.gguf")]
        api.model_info.return_value = info

        result = list_gguf_files("org/model", api)
        assert result[0]["quant"] == "IQ2_XXS"


# ---------------------------------------------------------------------------
# TestSelectGgufFile
# ---------------------------------------------------------------------------


class TestSelectGgufFile:
    """select_gguf_file() picks highest quality within memory budget."""

    def _files(self):
        return [
            {"filename": "model-Q2_K.gguf", "size": 2_000_000_000, "quant": "Q2_K"},
            {"filename": "model-Q4_K_M.gguf", "size": 4_000_000_000, "quant": "Q4_K_M"},
            {"filename": "model-Q8_0.gguf", "size": 8_000_000_000, "quant": "Q8_0"},
        ]

    def test_picks_highest_quality_no_memory_limit(self):
        result = select_gguf_file(self._files(), max_memory=None)
        assert result is not None
        assert result["quant"] == "Q8_0"

    def test_picks_within_memory_budget(self):
        # 8GB RAM * 0.7 = 5.6GB budget → Q4_K_M (4GB) fits, Q8_0 (8GB) doesn't
        result = select_gguf_file(self._files(), max_memory=8_000_000_000)
        assert result is not None
        assert result["quant"] == "Q4_K_M"

    def test_returns_none_for_empty_list(self):
        result = select_gguf_file([], max_memory=None)
        assert result is None

    def test_returns_none_when_nothing_fits(self):
        result = select_gguf_file(self._files(), max_memory=1_000_000_000)
        assert result is None

    def test_unknown_quant_ranked_lowest(self):
        files = [
            {"filename": "model.gguf", "size": 1_000_000_000, "quant": "unknown"},
            {"filename": "model-Q2_K.gguf", "size": 2_000_000_000, "quant": "Q2_K"},
        ]
        result = select_gguf_file(files, max_memory=None)
        assert result["quant"] == "Q2_K"


# ---------------------------------------------------------------------------
# TestGetSystemMemory
# ---------------------------------------------------------------------------


class TestGetSystemMemory:
    """get_system_memory() returns bytes or None on failure."""

    @patch("froggy.download.os.sysconf")
    def test_returns_memory_bytes(self, mock_sysconf):
        mock_sysconf.side_effect = lambda key: {
            "SC_PAGE_SIZE": 4096,
            "SC_PHYS_PAGES": 4_000_000,
        }[key]
        result = get_system_memory()
        assert result == 4096 * 4_000_000

    @patch("froggy.download.os.sysconf", side_effect=ValueError("unsupported"))
    def test_returns_none_on_error(self, _mock):
        assert get_system_memory() is None

    @patch("froggy.download.os.sysconf", side_effect=OSError("not available"))
    def test_returns_none_on_os_error(self, _mock):
        assert get_system_memory() is None


# ---------------------------------------------------------------------------
# TestDownloadModel
# ---------------------------------------------------------------------------


class TestDownloadModel:
    """download_model() orchestrates the MLX → GGUF → safetensors fallback chain."""

    @patch("froggy.download.ensure_froggy_home")
    @patch("froggy.download.models_dir")
    @patch("froggy.download.snapshot_download")
    @patch("froggy.download.find_mlx_repo", return_value="mlx-community/model-4bit")
    @patch("froggy.download.detect_platform", return_value="mlx")
    def test_successful_mlx_download(
        self, _detect, _find_mlx, mock_snap, mock_mdir, _home
    ):
        mock_mdir.return_value = Path("/tmp/test_models")
        api = MagicMock()
        result = download_model("org/model", api=api)
        mock_snap.assert_called_once()
        assert "mlx-community--model-4bit" in str(result)

    @patch("froggy.download.ensure_froggy_home")
    @patch("froggy.download.models_dir")
    @patch("froggy.download.hf_hub_download")
    @patch("froggy.download.select_gguf_file")
    @patch("froggy.download.list_gguf_files")
    @patch("froggy.download.get_system_memory", return_value=16_000_000_000)
    @patch("froggy.download.find_mlx_repo", return_value=None)
    @patch("froggy.download.detect_platform", return_value="mlx")
    def test_gguf_fallback_when_no_mlx(
        self, _detect, _find_mlx, _mem, mock_list, mock_select, mock_dl, mock_mdir, _home
    ):
        mock_mdir.return_value = Path("/tmp/test_models")
        mock_list.return_value = [
            {"filename": "model-Q4_K_M.gguf", "size": 4_000_000_000, "quant": "Q4_K_M"}
        ]
        mock_select.return_value = mock_list.return_value[0]
        api = MagicMock()
        result = download_model("org/model", api=api)
        mock_dl.assert_called_once()
        assert "org--model" in str(result)

    @patch("froggy.download.ensure_froggy_home")
    @patch("froggy.download.models_dir")
    @patch("froggy.download.snapshot_download")
    @patch("froggy.download.list_gguf_files", return_value=[])
    @patch("froggy.download.find_mlx_repo", return_value=None)
    @patch("froggy.download.detect_platform", return_value="mlx")
    def test_safetensors_fallback(
        self, _detect, _find_mlx, _list_gguf, mock_snap, mock_mdir, _home
    ):
        mock_mdir.return_value = Path("/tmp/test_models")
        api = MagicMock()
        result = download_model("org/model", api=api)
        mock_snap.assert_called_once()
        assert "org--model" in str(result)

    @patch("froggy.download.ensure_froggy_home")
    @patch("froggy.download.models_dir")
    @patch("froggy.download.snapshot_download", side_effect=Exception("fail"))
    @patch("froggy.download.list_gguf_files", return_value=[])
    @patch("froggy.download.find_mlx_repo", return_value=None)
    @patch("froggy.download.detect_platform", return_value="gguf")
    def test_error_when_nothing_available(
        self, _detect, _find_mlx, _list_gguf, _snap, mock_mdir, _home
    ):
        mock_mdir.return_value = Path("/tmp/test_models")
        api = MagicMock()
        with pytest.raises(click.ClickException, match="No compatible"):
            download_model("org/model", api=api)

    @patch("froggy.download.ensure_froggy_home")
    @patch("froggy.download.models_dir")
    @patch("froggy.download.hf_hub_download")
    def test_direct_filename_download(self, mock_dl, mock_mdir, _home):
        mock_mdir.return_value = Path("/tmp/test_models")
        api = MagicMock()
        result = download_model(
            "https://huggingface.co/org/model/blob/main/specific-file.gguf",
            api=api,
        )
        mock_dl.assert_called_once_with(
            "org/model",
            "specific-file.gguf",
            revision="main",
            local_dir=str(Path("/tmp/test_models/org--model")),
            token=None,
        )
        assert "org--model" in str(result)

    @patch("froggy.download.ensure_froggy_home")
    @patch("froggy.download.models_dir")
    @patch("froggy.download.hf_hub_download")
    @patch("froggy.download.select_gguf_file")
    @patch("froggy.download.list_gguf_files")
    @patch("froggy.download.get_system_memory", return_value=16_000_000_000)
    def test_format_gguf_skips_mlx(
        self, _mem, mock_list, mock_select, mock_dl, mock_mdir, _home
    ):
        """--format gguf should NOT call find_mlx_repo."""
        mock_mdir.return_value = Path("/tmp/test_models")
        mock_list.return_value = [
            {"filename": "model-Q8_0.gguf", "size": 8_000_000_000, "quant": "Q8_0"}
        ]
        mock_select.return_value = mock_list.return_value[0]
        api = MagicMock()
        result = download_model("org/model", fmt="gguf", api=api)
        mock_dl.assert_called_once()
        assert "org--model" in str(result)


# ---------------------------------------------------------------------------
# TestListVariants
# ---------------------------------------------------------------------------


class TestListVariants:
    """list_variants() returns all downloadable variants for a repo."""

    def _make_api(self, mlx_repo=None, gguf_files=None):
        from huggingface_hub.utils import RepositoryNotFoundError

        api = MagicMock()

        def model_info_side_effect(repo_id, **kwargs):
            if mlx_repo and repo_id == mlx_repo:
                return MagicMock()
            if repo_id == "org/model" and kwargs.get("files_metadata"):
                info = MagicMock()
                info.siblings = [
                    _make_sibling(f["filename"], f.get("size", 1000))
                    for f in (gguf_files or [])
                ]
                return info
            raise RepositoryNotFoundError(f"{repo_id} not found")

        api.model_info.side_effect = model_info_side_effect
        return api

    def test_includes_mlx_variant(self):
        api = self._make_api(mlx_repo="mlx-community/model-4bit")
        variants = list_variants("org/model", api=api)
        types = [v["type"] for v in variants]
        assert "mlx" in types

    def test_includes_gguf_variants(self):
        api = self._make_api(
            gguf_files=[
                {"filename": "model-Q4_K_M.gguf", "size": 4_000_000_000},
                {"filename": "model-Q8_0.gguf", "size": 8_000_000_000},
            ]
        )
        variants = list_variants("org/model", api=api)
        gguf_variants = [v for v in variants if v["type"] == "gguf"]
        assert len(gguf_variants) == 2

    def test_always_includes_safetensors(self):
        api = self._make_api()
        variants = list_variants("org/model", api=api)
        types = [v["type"] for v in variants]
        assert "safetensors" in types

    def test_no_mlx_no_gguf(self):
        """When no MLX and no GGUF, only safetensors variant."""
        api = self._make_api()
        variants = list_variants("org/model", api=api)
        assert len(variants) == 1
        assert variants[0]["type"] == "safetensors"
