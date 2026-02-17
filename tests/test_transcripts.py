"""Tests for lenny.transcripts — transcript discovery and download."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lenny.transcripts import (
    _find_episodes_dir,
    _git_available,
    download_transcripts,
    ensure_transcripts,
    transcript_data_dir,
)


# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------

class TestTranscriptDataDir:
    def test_respects_xdg_data_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        result = transcript_data_dir()
        assert result == tmp_path / "lenny" / "transcripts"

    def test_macos_default(self, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr("sys.platform", "darwin")
        result = transcript_data_dir()
        assert "Library" in str(result)
        assert str(result).endswith("lenny/transcripts")

    def test_linux_default(self, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr("sys.platform", "linux")
        result = transcript_data_dir()
        assert ".local/share/lenny/transcripts" in str(result)


# ---------------------------------------------------------------------------
# Transcript discovery
# ---------------------------------------------------------------------------

class TestFindEpisodesDir:
    def test_env_var_takes_priority(self, monkeypatch, tmp_path):
        ep_dir = tmp_path / "my-transcripts"
        ep_dir.mkdir()
        monkeypatch.setenv("LENNY_TRANSCRIPTS", str(ep_dir))
        result = _find_episodes_dir()
        assert result == str(ep_dir)

    def test_returns_none_when_not_found(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LENNY_TRANSCRIPTS", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "empty_xdg"))
        # Prevent walk-up-from-source from finding the real transcripts
        import lenny.transcripts as mod
        orig_file = mod.__file__
        try:
            mod.__file__ = str(tmp_path / "fake" / "lenny" / "transcripts.py")
            result = _find_episodes_dir()
        finally:
            mod.__file__ = orig_file
        assert result is None

    def test_finds_data_dir_transcripts(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LENNY_TRANSCRIPTS", raising=False)
        monkeypatch.chdir(tmp_path)
        # Create transcripts in the XDG data dir
        xdg = tmp_path / "xdg_data"
        episodes = xdg / "lenny" / "transcripts" / "episodes"
        episodes.mkdir(parents=True)
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg))

        # Prevent walk-up-from-source from finding real transcripts
        import lenny.transcripts as mod
        orig_file = mod.__file__
        try:
            mod.__file__ = str(tmp_path / "fake" / "lenny" / "transcripts.py")
            result = _find_episodes_dir()
        finally:
            mod.__file__ = orig_file
        assert result == str(episodes)


# ---------------------------------------------------------------------------
# Git availability check
# ---------------------------------------------------------------------------

class TestGitAvailable:
    def test_git_available_when_installed(self):
        # This test assumes git is installed (which it should be in dev)
        assert _git_available() is True

    def test_git_unavailable(self, monkeypatch):
        def mock_run(*args, **kwargs):
            raise FileNotFoundError("git not found")
        monkeypatch.setattr(subprocess, "run", mock_run)
        assert _git_available() is False


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

class TestDownload:
    def test_download_via_git_calls_clone(self, monkeypatch, tmp_path):
        dest = tmp_path / "transcripts"

        calls = []

        def mock_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[0] == "git" and cmd[1] == "--version":
                result = MagicMock()
                result.returncode = 0
                return result
            if cmd[0] == "git" and cmd[1] == "clone":
                # Simulate successful git clone
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "episodes").mkdir(exist_ok=True)
                result = MagicMock()
                result.returncode = 0
                return result
            return MagicMock(returncode=0)

        monkeypatch.setattr(subprocess, "run", mock_run)
        assert download_transcripts(dest) is True
        assert (dest / "episodes").is_dir()
        # Verify git clone was called
        clone_calls = [c for c in calls if len(c) > 1 and c[1] == "clone"]
        assert len(clone_calls) == 1

    def test_download_fails_on_bad_returncode(self, monkeypatch, tmp_path):
        dest = tmp_path / "transcripts"

        def mock_run(cmd, **kwargs):
            if cmd[0] == "git" and cmd[1] == "--version":
                return MagicMock(returncode=0)
            result = MagicMock()
            result.returncode = 1
            result.stderr = "fatal: could not connect"
            return result

        monkeypatch.setattr(subprocess, "run", mock_run)
        assert download_transcripts(dest) is False
        # Dest should be cleaned up
        assert not dest.exists()

    def test_download_fails_when_dest_exists(self, tmp_path):
        dest = tmp_path / "transcripts"
        dest.mkdir()
        assert download_transcripts(dest) is False

    def test_download_fails_missing_episodes_dir(self, monkeypatch, tmp_path):
        dest = tmp_path / "transcripts"

        def mock_run(cmd, **kwargs):
            if cmd[0] == "git" and cmd[1] == "--version":
                return MagicMock(returncode=0)
            # Simulate clone that produces wrong structure
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "README.md").touch()
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr(subprocess, "run", mock_run)
        assert download_transcripts(dest) is False

    def test_download_fails_without_git(self, monkeypatch, tmp_path):
        """When git is unavailable and the tarball fallback also fails, return False."""
        dest = tmp_path / "transcripts"

        def mock_run(*args, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(subprocess, "run", mock_run)
        # Also mock the tarball fallback so the test stays offline and deterministic.
        monkeypatch.setattr(
            "lenny.transcripts._download_via_tarball",
            lambda dest, console=None: False,
        )
        assert download_transcripts(dest) is False


# ---------------------------------------------------------------------------
# ensure_transcripts orchestration
# ---------------------------------------------------------------------------

class TestEnsureTranscripts:
    def test_returns_existing_dir(self, monkeypatch, tmp_path):
        ep_dir = tmp_path / "episodes"
        ep_dir.mkdir()
        monkeypatch.setenv("LENNY_TRANSCRIPTS", str(ep_dir))
        console = MagicMock()
        result = ensure_transcripts(console)
        assert result == str(ep_dir)

    def test_non_tty_raises_with_instructions(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LENNY_TRANSCRIPTS", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "empty"))
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: False))
        console = MagicMock()

        # Prevent walk-up-from-source from finding real transcripts
        import lenny.transcripts as mod
        orig_file = mod.__file__
        try:
            mod.__file__ = str(tmp_path / "fake" / "lenny" / "transcripts.py")
            with pytest.raises(FileNotFoundError, match="LENNY_TRANSCRIPTS"):
                ensure_transcripts(console)
        finally:
            mod.__file__ = orig_file

    def test_non_tty_does_not_prompt(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LENNY_TRANSCRIPTS", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "empty"))
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: False))
        console = MagicMock()

        import lenny.transcripts as mod
        orig_file = mod.__file__
        try:
            mod.__file__ = str(tmp_path / "fake" / "lenny" / "transcripts.py")
            with pytest.raises(FileNotFoundError):
                ensure_transcripts(console)
        finally:
            mod.__file__ = orig_file

        # Confirm.ask should never have been called (console is a mock)
        console.input.assert_not_called()
