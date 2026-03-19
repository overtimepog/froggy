"""Tests for froggy.llmfit — binary management and system info parsing.

All external calls (urllib, subprocess, shutil.which) are mocked.
No real network or binary execution occurs.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path
from unittest import mock

from froggy.llmfit import (
    _download_llmfit,
    _is_stale,
    _platform_asset_name,
    ensure_llmfit,
    get_memory_budget,
    llmfit_binary_path,
    llmfit_recommend,
    llmfit_system_info,
)

# ---------------------------------------------------------------------------
# TestLlmfitBinaryPath
# ---------------------------------------------------------------------------


class TestLlmfitBinaryPath:
    """llmfit_binary_path() returns the correct managed path."""

    def test_default_path(self):
        """Should be <froggy_home>/bin/llmfit."""
        from froggy.paths import froggy_home

        assert llmfit_binary_path() == froggy_home() / "bin" / "llmfit"

    def test_respects_froggy_home_env(self, tmp_path, monkeypatch):
        """FROGGY_HOME override should propagate to binary path."""
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path / "custom"))
        assert llmfit_binary_path() == tmp_path / "custom" / "bin" / "llmfit"


# ---------------------------------------------------------------------------
# TestPlatformAssetName
# ---------------------------------------------------------------------------


class TestPlatformAssetName:
    """_platform_asset_name() maps platform to GitHub release asset names."""

    @mock.patch("froggy.llmfit.platform")
    def test_macos_arm64(self, mock_platform):
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"
        assert _platform_asset_name() == "aarch64-apple-darwin"

    @mock.patch("froggy.llmfit.platform")
    def test_macos_x86_64(self, mock_platform):
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "x86_64"
        assert _platform_asset_name() == "x86_64-apple-darwin"

    @mock.patch("froggy.llmfit.platform")
    def test_linux_x86_64(self, mock_platform):
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"
        assert _platform_asset_name() == "x86_64-unknown-linux-gnu"

    @mock.patch("froggy.llmfit.platform")
    def test_windows_returns_none(self, mock_platform):
        mock_platform.system.return_value = "Windows"
        mock_platform.machine.return_value = "AMD64"
        assert _platform_asset_name() is None

    @mock.patch("froggy.llmfit.platform")
    def test_linux_arm_returns_none(self, mock_platform):
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "aarch64"
        assert _platform_asset_name() is None


# ---------------------------------------------------------------------------
# TestIsStale
# ---------------------------------------------------------------------------


class TestIsStale:
    """_is_stale() checks file existence and 24h mtime threshold."""

    def test_recent_file_not_stale(self, tmp_path):
        f = tmp_path / "llmfit"
        f.write_text("x")
        assert _is_stale(f) is False

    def test_old_file_is_stale(self, tmp_path):
        f = tmp_path / "llmfit"
        f.write_text("x")
        old_time = time.time() - (25 * 60 * 60)  # 25 hours ago
        os.utime(f, (old_time, old_time))
        assert _is_stale(f) is True

    def test_nonexistent_file_is_stale(self, tmp_path):
        f = tmp_path / "does_not_exist"
        assert _is_stale(f) is True


# ---------------------------------------------------------------------------
# TestDownloadLlmfit
# ---------------------------------------------------------------------------


def _make_tarball_bytes(inner_name: str = "llmfit-v1.0/llmfit") -> bytes:
    """Create a gzipped tarball containing a fake llmfit binary."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"#!/bin/sh\necho fake"
        info = tarfile.TarInfo(name=inner_name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TestDownloadLlmfit:
    """_download_llmfit() fetches from GitHub Releases and extracts."""

    @mock.patch("froggy.llmfit._platform_asset_name", return_value="aarch64-apple-darwin")
    @mock.patch("froggy.llmfit.urllib.request.urlopen")
    def test_successful_download(self, mock_urlopen, _mock_asset, tmp_path):
        dest = tmp_path / "bin" / "llmfit"

        release_json = json.dumps({
            "assets": [
                {
                    "name": "llmfit-aarch64-apple-darwin.tar.gz",
                    "browser_download_url": "https://example.com/llmfit.tar.gz",
                }
            ]
        }).encode()

        tarball = _make_tarball_bytes()

        # First call: release metadata; second call: tarball download
        mock_resp_meta = mock.MagicMock()
        mock_resp_meta.read.return_value = release_json
        mock_resp_meta.__enter__ = mock.Mock(return_value=mock_resp_meta)
        mock_resp_meta.__exit__ = mock.Mock(return_value=False)

        mock_resp_tar = mock.MagicMock()
        mock_resp_tar.read.return_value = tarball
        mock_resp_tar.__enter__ = mock.Mock(return_value=mock_resp_tar)
        mock_resp_tar.__exit__ = mock.Mock(return_value=False)

        mock_urlopen.side_effect = [mock_resp_meta, mock_resp_tar]

        assert _download_llmfit(dest) is True
        assert dest.exists()
        assert os.access(dest, os.X_OK)

    @mock.patch("froggy.llmfit._platform_asset_name", return_value=None)
    def test_unsupported_platform(self, _mock_asset, tmp_path):
        dest = tmp_path / "bin" / "llmfit"
        assert _download_llmfit(dest) is False

    @mock.patch("froggy.llmfit._platform_asset_name", return_value="aarch64-apple-darwin")
    @mock.patch("froggy.llmfit.urllib.request.urlopen", side_effect=OSError("network"))
    def test_network_error(self, _mock_urlopen, _mock_asset, tmp_path):
        dest = tmp_path / "bin" / "llmfit"
        assert _download_llmfit(dest) is False

    @mock.patch("froggy.llmfit._platform_asset_name", return_value="aarch64-apple-darwin")
    @mock.patch("froggy.llmfit.urllib.request.urlopen")
    def test_missing_asset_in_release(self, mock_urlopen, _mock_asset, tmp_path):
        dest = tmp_path / "bin" / "llmfit"

        release_json = json.dumps({
            "assets": [
                {
                    "name": "llmfit-some-other-platform.tar.gz",
                    "browser_download_url": "https://example.com/other.tar.gz",
                }
            ]
        }).encode()

        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = release_json
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)
        mock_urlopen.return_value = mock_resp

        assert _download_llmfit(dest) is False


# ---------------------------------------------------------------------------
# TestEnsureLlmfit
# ---------------------------------------------------------------------------


class TestEnsureLlmfit:
    """ensure_llmfit() orchestrates download with staleness + PATH fallback."""

    @mock.patch("froggy.llmfit._download_llmfit")
    def test_fresh_binary_returned_directly(self, mock_dl, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        binary = bin_dir / "llmfit"
        binary.write_text("x")

        result = ensure_llmfit()
        assert result == binary
        mock_dl.assert_not_called()

    @mock.patch("froggy.llmfit._download_llmfit", return_value=True)
    def test_missing_binary_triggers_download(self, mock_dl, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        result = ensure_llmfit()
        mock_dl.assert_called_once()
        assert result == tmp_path / "bin" / "llmfit"

    @mock.patch("froggy.llmfit._download_llmfit", return_value=True)
    def test_stale_binary_triggers_redownload(self, mock_dl, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        binary = bin_dir / "llmfit"
        binary.write_text("x")
        old_time = time.time() - (25 * 60 * 60)
        os.utime(binary, (old_time, old_time))

        result = ensure_llmfit()
        mock_dl.assert_called_once()
        assert result == binary

    @mock.patch("froggy.llmfit.shutil.which", return_value="/usr/local/bin/llmfit")
    @mock.patch("froggy.llmfit._download_llmfit", return_value=False)
    def test_download_fails_falls_back_to_path(self, _mock_dl, _mock_which, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        result = ensure_llmfit()
        assert result == Path("/usr/local/bin/llmfit")

    @mock.patch("froggy.llmfit.shutil.which", return_value=None)
    @mock.patch("froggy.llmfit._download_llmfit", return_value=False)
    def test_both_fail_returns_none(self, _mock_dl, _mock_which, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        result = ensure_llmfit()
        assert result is None


# ---------------------------------------------------------------------------
# TestLlmfitSystemInfo
# ---------------------------------------------------------------------------


class TestLlmfitSystemInfo:
    """llmfit_system_info() parses subprocess JSON output."""

    def test_valid_json(self):
        system_data = {
            "total_ram_gb": 32.0,
            "gpu_vram_gb": 0.0,
            "unified_memory": True,
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({"system": system_data}),
            stderr="",
        )
        with mock.patch("froggy.llmfit.subprocess.run", return_value=completed):
            result = llmfit_system_info("/fake/llmfit")
        assert result == system_data

    def test_invalid_json(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not json", stderr="",
        )
        with mock.patch("froggy.llmfit.subprocess.run", return_value=completed):
            result = llmfit_system_info("/fake/llmfit")
        assert result is None

    def test_timeout(self):
        with mock.patch(
            "froggy.llmfit.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="llmfit", timeout=30),
        ):
            result = llmfit_system_info("/fake/llmfit")
        assert result is None

    def test_nonzero_exit_code(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error",
        )
        with mock.patch("froggy.llmfit.subprocess.run", return_value=completed):
            result = llmfit_system_info("/fake/llmfit")
        assert result is None

    def test_missing_system_key(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({"other": "data"}),
            stderr="",
        )
        with mock.patch("froggy.llmfit.subprocess.run", return_value=completed):
            result = llmfit_system_info("/fake/llmfit")
        assert result is None

    def test_subprocess_uses_timeout(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({"system": {"total_ram_gb": 16.0}}),
            stderr="",
        )
        with mock.patch("froggy.llmfit.subprocess.run", return_value=completed) as mock_run:
            llmfit_system_info("/fake/llmfit")
        _, kwargs = mock_run.call_args
        assert kwargs["timeout"] == 30


# ---------------------------------------------------------------------------
# TestLlmfitRecommend
# ---------------------------------------------------------------------------

_SAMPLE_RECOMMEND_OUTPUT = json.dumps({
    "models": [
        {
            "name": "llama-3.2-3b",
            "score": 95,
            "best_quant": "Q4_K_M",
            "estimated_tps": 45.2,
            "fit_level": "full",
            "run_mode": "gpu",
            "memory_required_gb": 2.1,
            "category": "small",
        }
    ]
})


class TestLlmfitRecommend:
    """llmfit_recommend() parses JSON model list from subprocess."""

    def test_valid_output_parsed(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=_SAMPLE_RECOMMEND_OUTPUT, stderr="",
        )
        with mock.patch("froggy.llmfit.subprocess.run", return_value=completed):
            result = llmfit_recommend("/fake/llmfit")
        assert len(result) == 1
        assert result[0]["name"] == "llama-3.2-3b"
        assert result[0]["score"] == 95
        assert result[0]["best_quant"] == "Q4_K_M"
        assert result[0]["estimated_tps"] == 45.2
        assert result[0]["fit_level"] == "full"
        assert result[0]["run_mode"] == "gpu"
        assert result[0]["memory_required_gb"] == 2.1
        # category is NOT included in output (not in spec)
        assert "category" not in result[0]

    def test_empty_models_list(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({"models": []}),
            stderr="",
        )
        with mock.patch("froggy.llmfit.subprocess.run", return_value=completed):
            result = llmfit_recommend("/fake/llmfit")
        assert result == []

    def test_json_parse_error_returns_empty(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not json", stderr="",
        )
        with mock.patch("froggy.llmfit.subprocess.run", return_value=completed):
            result = llmfit_recommend("/fake/llmfit")
        assert result == []

    def test_timeout_returns_empty(self):
        with mock.patch(
            "froggy.llmfit.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="llmfit", timeout=30),
        ):
            result = llmfit_recommend("/fake/llmfit")
        assert result == []

    def test_missing_keys_use_defaults(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({"models": [{"name": "partial-model"}]}),
            stderr="",
        )
        with mock.patch("froggy.llmfit.subprocess.run", return_value=completed):
            result = llmfit_recommend("/fake/llmfit")
        assert len(result) == 1
        assert result[0]["name"] == "partial-model"
        assert result[0]["score"] == 0
        assert result[0]["best_quant"] == "unknown"
        assert result[0]["estimated_tps"] == 0.0
        assert result[0]["fit_level"] == "unknown"
        assert result[0]["run_mode"] == "unknown"
        assert result[0]["memory_required_gb"] == 0.0

    def test_nonzero_exit_returns_empty(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error",
        )
        with mock.patch("froggy.llmfit.subprocess.run", return_value=completed):
            result = llmfit_recommend("/fake/llmfit")
        assert result == []

    def test_passes_limit_and_use_case(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({"models": []}),
            stderr="",
        )
        with mock.patch("froggy.llmfit.subprocess.run", return_value=completed) as mock_run:
            llmfit_recommend("/fake/llmfit", limit=3, use_case="coding")
        args = mock_run.call_args[0][0]
        assert "--limit" in args
        assert "3" in args
        assert "--use-case" in args
        assert "coding" in args


# ---------------------------------------------------------------------------
# TestGetMemoryBudget
# ---------------------------------------------------------------------------


class TestGetMemoryBudget:
    """get_memory_budget() uses llmfit when available, falls back to heuristic."""

    @mock.patch("froggy.llmfit.llmfit_system_info")
    @mock.patch("froggy.llmfit.ensure_llmfit")
    def test_unified_memory_mac(self, mock_ensure, mock_sysinfo):
        mock_ensure.return_value = Path("/fake/llmfit")
        mock_sysinfo.return_value = {
            "total_ram_gb": 32.0,
            "gpu_vram_gb": 0.0,
            "unified_memory": True,
        }
        result = get_memory_budget()
        assert result == int(32.0 * 1024 * 1024 * 1024)

    @mock.patch("froggy.llmfit.llmfit_system_info")
    @mock.patch("froggy.llmfit.ensure_llmfit")
    def test_discrete_gpu(self, mock_ensure, mock_sysinfo):
        mock_ensure.return_value = Path("/fake/llmfit")
        mock_sysinfo.return_value = {
            "total_ram_gb": 64.0,
            "gpu_vram_gb": 24.0,
            "unified_memory": False,
        }
        result = get_memory_budget()
        assert result == int(24.0 * 1024 * 1024 * 1024)

    @mock.patch("froggy.llmfit.llmfit_system_info")
    @mock.patch("froggy.llmfit.ensure_llmfit")
    def test_cpu_only(self, mock_ensure, mock_sysinfo):
        mock_ensure.return_value = Path("/fake/llmfit")
        mock_sysinfo.return_value = {
            "total_ram_gb": 16.0,
            "gpu_vram_gb": 0.0,
            "unified_memory": False,
        }
        result = get_memory_budget()
        assert result == int(16.0 * 1024 * 1024 * 1024)

    @mock.patch("froggy.download.get_system_memory", return_value=16_000_000_000)
    @mock.patch("froggy.llmfit.ensure_llmfit", return_value=None)
    def test_llmfit_unavailable_falls_back(self, _mock_ensure, _mock_mem):
        result = get_memory_budget()
        assert result == int(16_000_000_000 * 0.7)

    @mock.patch("froggy.download.get_system_memory", return_value=None)
    @mock.patch("froggy.llmfit.ensure_llmfit", return_value=None)
    def test_both_fail_returns_none(self, _mock_ensure, _mock_mem):
        result = get_memory_budget()
        assert result is None
