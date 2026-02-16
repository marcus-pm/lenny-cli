"""Sandbox regression tests.

Exercises every known file-access bypass vector to ensure the REPL
sandbox blocks reads outside the transcript allowlist.

Run:  python -m pytest tests/test_sandbox.py -v
"""

from __future__ import annotations

import os
import textwrap

import pytest

from lenny.engine import (
    _BLOCKED_MODULES,
    _make_restricted_import,
    _make_restricted_open,
)

TRANSCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "transcripts", "episodes")
)
OUTSIDE_FILE = "/etc/hosts"  # always exists on macOS/Linux


@pytest.fixture()
def restricted_open():
    return _make_restricted_open([TRANSCRIPT_DIR])


@pytest.fixture()
def restricted_import(restricted_open):
    return _make_restricted_import(_BLOCKED_MODULES, restricted_open)


# ------------------------------------------------------------------
# Builtins open()
# ------------------------------------------------------------------
class TestRestrictedOpen:
    def test_blocks_read_outside_allowlist(self, restricted_open):
        with pytest.raises(PermissionError, match="outside allowed"):
            restricted_open(OUTSIDE_FILE, "r")

    def test_blocks_write_anywhere(self, restricted_open):
        with pytest.raises(PermissionError, match="Write access"):
            restricted_open("/tmp/evil.txt", "w")

    def test_blocks_path_traversal(self, restricted_open):
        traversal = os.path.join(TRANSCRIPT_DIR, "..", "..", ".env")
        with pytest.raises(PermissionError, match="outside allowed"):
            restricted_open(traversal, "r")

    def test_allows_transcript_read(self, restricted_open):
        first_ep = os.listdir(TRANSCRIPT_DIR)[0]
        path = os.path.join(TRANSCRIPT_DIR, first_ep, "transcript.md")
        with restricted_open(path, "r") as f:
            assert len(f.read(10)) > 0


# ------------------------------------------------------------------
# os.open / os.read / os.system bypass
# ------------------------------------------------------------------
class TestOsBypass:
    def test_os_open_stripped(self, restricted_import):
        safe_os = restricted_import("os")
        assert not hasattr(safe_os, "open")

    def test_os_read_stripped(self, restricted_import):
        safe_os = restricted_import("os")
        assert not hasattr(safe_os, "read")

    def test_os_system_stripped(self, restricted_import):
        safe_os = restricted_import("os")
        assert not hasattr(safe_os, "system")

    def test_os_popen_stripped(self, restricted_import):
        safe_os = restricted_import("os")
        assert not hasattr(safe_os, "popen")

    def test_os_path_still_works(self, restricted_import):
        safe_os = restricted_import("os")
        assert safe_os.path.join("a", "b") == "a/b"

    def test_os_listdir_still_works(self, restricted_import):
        safe_os = restricted_import("os")
        assert len(safe_os.listdir(".")) > 0


# ------------------------------------------------------------------
# io.open / io.FileIO bypass
# ------------------------------------------------------------------
class TestIoBypass:
    def test_io_open_blocked_outside(self, restricted_import):
        safe_io = restricted_import("io")
        with pytest.raises(PermissionError):
            safe_io.open(OUTSIDE_FILE, "r")

    def test_io_fileio_blocked(self, restricted_import):
        safe_io = restricted_import("io")
        with pytest.raises(PermissionError):
            safe_io.FileIO(OUTSIDE_FILE)

    def test_io_stringio_still_works(self, restricted_import):
        safe_io = restricted_import("io")
        buf = safe_io.StringIO("hello")
        assert buf.read() == "hello"

    def test_io_open_allows_transcript(self, restricted_import):
        safe_io = restricted_import("io")
        first_ep = os.listdir(TRANSCRIPT_DIR)[0]
        path = os.path.join(TRANSCRIPT_DIR, first_ep, "transcript.md")
        with safe_io.open(path, "r") as f:
            assert len(f.read(10)) > 0


# ------------------------------------------------------------------
# pathlib.Path.read_text / read_bytes / open bypass
# ------------------------------------------------------------------
class TestPathlibBypass:
    def test_path_read_text_blocked(self, restricted_import):
        safe_pathlib = restricted_import("pathlib")
        with pytest.raises(PermissionError):
            safe_pathlib.Path(OUTSIDE_FILE).read_text()

    def test_path_read_bytes_blocked(self, restricted_import):
        safe_pathlib = restricted_import("pathlib")
        with pytest.raises(PermissionError):
            safe_pathlib.Path(OUTSIDE_FILE).read_bytes()

    def test_path_open_blocked(self, restricted_import):
        safe_pathlib = restricted_import("pathlib")
        with pytest.raises(PermissionError):
            safe_pathlib.Path(OUTSIDE_FILE).open("r")

    def test_path_write_text_blocked(self, restricted_import):
        safe_pathlib = restricted_import("pathlib")
        with pytest.raises(PermissionError):
            safe_pathlib.Path("/tmp/evil.txt").write_text("pwned")

    def test_path_write_bytes_blocked(self, restricted_import):
        safe_pathlib = restricted_import("pathlib")
        with pytest.raises(PermissionError):
            safe_pathlib.Path("/tmp/evil.txt").write_bytes(b"pwned")

    def test_path_joining_still_works(self, restricted_import):
        safe_pathlib = restricted_import("pathlib")
        p = safe_pathlib.Path("a") / "b" / "c"
        assert str(p) == "a/b/c"

    def test_path_read_text_allows_transcript(self, restricted_import):
        safe_pathlib = restricted_import("pathlib")
        first_ep = os.listdir(TRANSCRIPT_DIR)[0]
        path = os.path.join(TRANSCRIPT_DIR, first_ep, "transcript.md")
        content = safe_pathlib.Path(path).read_text()
        assert len(content) > 0


# ------------------------------------------------------------------
# posix / nt low-level bypass
# ------------------------------------------------------------------
class TestPosixBypass:
    def test_posix_import_blocked(self, restricted_import):
        with pytest.raises(ImportError, match="blocked for security"):
            restricted_import("posix")

    def test_nt_import_blocked(self, restricted_import):
        with pytest.raises(ImportError, match="blocked for security"):
            restricted_import("nt")

    def test__io_import_blocked(self, restricted_import):
        with pytest.raises(ImportError, match="blocked for security"):
            restricted_import("_io")


# ------------------------------------------------------------------
# Fully-blocked modules
# ------------------------------------------------------------------
class TestBlockedModules:
    @pytest.mark.parametrize("mod", [
        "subprocess", "socket", "http", "urllib", "requests",
        "ftplib", "smtplib", "ctypes", "importlib",
    ])
    def test_dangerous_module_blocked(self, restricted_import, mod):
        with pytest.raises(ImportError, match="blocked for security"):
            restricted_import(mod)

    @pytest.mark.parametrize("mod", ["re", "json", "math", "collections"])
    def test_safe_module_allowed(self, restricted_import, mod):
        m = restricted_import(mod)
        assert m is not None
