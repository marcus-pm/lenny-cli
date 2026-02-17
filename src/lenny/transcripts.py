"""Transcript discovery and first-run download for Lenny CLI.

Handles locating the transcript corpus across multiple sources
(environment variable, project directory, XDG data directory)
and offers interactive download when transcripts aren't found.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_TRANSCRIPT_REPO = "https://github.com/ChatPRD/lennys-podcast-transcripts.git"

# Pinned to a specific branch for git clone.
_TRANSCRIPT_BRANCH = "main"

# Pinned to commit be8ab89 (2026-02-17).
# Update _TRANSCRIPT_TARBALL_SHA when transcripts corpus is refreshed.
_TRANSCRIPT_TARBALL_SHA = "be8ab89a890a833cbba2c892178f823fff178c65"
_TRANSCRIPT_TARBALL_URL = (
    "https://github.com/ChatPRD/lennys-podcast-transcripts/archive/"
    f"{_TRANSCRIPT_TARBALL_SHA}.tar.gz"
)
# GitHub names the top-level dir inside the archive after the commit SHA.
_TRANSCRIPT_TARBALL_ROOT = f"lennys-podcast-transcripts-{_TRANSCRIPT_TARBALL_SHA}"


# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------

def _default_data_dir() -> Path:
    """Return the platform-appropriate data directory for Lenny.

    macOS:  ~/Library/Application Support/lenny
    Linux:  $XDG_DATA_HOME/lenny  (default ~/.local/share/lenny)
    """
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "lenny"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "lenny"

    return Path.home() / ".local" / "share" / "lenny"


def transcript_data_dir() -> Path:
    """Return the path where downloaded transcripts are stored."""
    return _default_data_dir() / "transcripts"


# ---------------------------------------------------------------------------
# Download — git path
# ---------------------------------------------------------------------------

def _git_available() -> bool:
    """Check if git is available on PATH."""
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def download_transcripts(dest: Path, console: object | None = None) -> bool:
    """Download the transcript corpus into *dest*.

    Prefers ``git clone --depth 1`` for speed and full directory
    structure.  Falls back to a pinned tarball download when git is
    unavailable.  Returns True on success, False on any failure.

    Parameters
    ----------
    dest:
        Target directory.  Must not already exist.
    console:
        Optional Rich Console for status display.
    """
    if dest.exists():
        logger.warning("Download destination already exists: %s", dest)
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    if _git_available():
        return _download_via_git(dest, console)

    logger.info("git not found; falling back to tarball download")
    return _download_via_tarball(dest, console)


def _download_via_git(dest: Path, console: object | None = None) -> bool:
    """Clone the transcript repo using git."""
    cmd = [
        "git", "clone",
        "--depth", "1",
        "--branch", _TRANSCRIPT_BRANCH,
        _TRANSCRIPT_REPO,
        str(dest),
    ]
    logger.debug("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )
        if result.returncode != 0:
            logger.error("git clone failed:\n%s", result.stderr)
            # Clean up partial clone
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            return False

        # Validate expected structure
        episodes_dir = dest / "episodes"
        if not episodes_dir.is_dir():
            logger.error("Downloaded repo missing episodes/ directory")
            shutil.rmtree(dest, ignore_errors=True)
            return False

        return True
    except subprocess.TimeoutExpired:
        logger.error("git clone timed out after 300s")
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        return False
    except Exception as e:
        logger.error("git clone error: %s", e)
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        return False


# ---------------------------------------------------------------------------
# Download — tarball fallback
# ---------------------------------------------------------------------------

def _safe_extract(tf: tarfile.TarFile, extract_root: Path) -> None:
    """Extract *tf* into *extract_root*, raising ValueError on path traversal.

    Uses os.path.commonpath to guard against zip-slip / tar-slip attacks:
    every member's resolved destination must share ``extract_root`` as a
    common path prefix.
    """
    resolved_root = str(extract_root.resolve())
    for member in tf.getmembers():
        member_path = (extract_root / member.name).resolve()
        if os.path.commonpath([resolved_root, str(member_path)]) != resolved_root:
            raise ValueError(
                f"Path traversal detected in tarball member: {member.name!r}"
            )
    tf.extractall(extract_root, filter="data")


def _download_via_tarball(dest: Path, console: object | None = None) -> bool:
    """Download and extract the pinned transcript tarball.

    Downloads to a temp file, validates it is a gzip tarball, extracts
    into a staging temp directory (with path traversal guard), validates
    the expected structure, then moves into *dest* atomically.

    Returns True on success, False on any failure.
    """
    logger.debug("Downloading tarball from %s", _TRANSCRIPT_TARBALL_URL)

    tmp_file: str | None = None
    tmp_dir: str | None = None
    try:
        # 1. Download to a named temp file
        fd, tmp_file = tempfile.mkstemp(suffix=".tar.gz", prefix="lenny-transcripts-")
        os.close(fd)

        with urllib.request.urlopen(_TRANSCRIPT_TARBALL_URL, timeout=300) as response:  # noqa: S310
            with open(tmp_file, "wb") as f:
                shutil.copyfileobj(response, f)

        # 2. Validate it's a real gzip tarball before touching anything
        if not tarfile.is_tarfile(tmp_file):
            logger.error("Downloaded file is not a valid tarball: %s", tmp_file)
            return False

        # 3. Extract into a staging temp directory (not directly into dest)
        tmp_dir = tempfile.mkdtemp(prefix="lenny-extract-")
        with tarfile.open(tmp_file, "r:gz") as tf:
            _safe_extract(tf, Path(tmp_dir))

        # 4. Locate the extracted root directory
        extracted_root = Path(tmp_dir) / _TRANSCRIPT_TARBALL_ROOT
        if not extracted_root.is_dir():
            # GitHub may use a slightly different naming; try the first subdir
            subdirs = [p for p in Path(tmp_dir).iterdir() if p.is_dir()]
            if len(subdirs) == 1:
                extracted_root = subdirs[0]
                logger.debug("Using extracted root: %s", extracted_root)
            else:
                logger.error(
                    "Could not locate extracted root %r in %s (found: %s)",
                    _TRANSCRIPT_TARBALL_ROOT,
                    tmp_dir,
                    [p.name for p in Path(tmp_dir).iterdir()],
                )
                return False

        # 5. Validate expected structure before moving
        if not (extracted_root / "episodes").is_dir():
            logger.error("Extracted tarball missing episodes/ directory")
            return False

        # 6. Move into the final destination atomically
        shutil.move(str(extracted_root), str(dest))

        # 7. Final validation
        if not (dest / "episodes").is_dir():
            logger.error("episodes/ directory missing after move to %s", dest)
            return False

        return True

    except ValueError as e:
        # Path traversal guard triggered
        logger.error("Tarball extraction rejected: %s", e)
        return False
    except urllib.error.URLError as e:
        logger.error("Tarball download failed: %s", e)
        return False
    except Exception as e:
        logger.error("Tarball download/extract error: %s", e)
        return False
    finally:
        # Clean up temp artifacts on both success and failure
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.unlink(tmp_file)
            except OSError:
                pass
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        # If we failed and dest was partially created, remove it
        # (dest is only moved into place on success, but guard anyway)
        # Note: do NOT remove dest on success — that's the final product.


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _find_episodes_dir() -> str | None:
    """Locate the transcripts/episodes directory.

    Search order:
    1. LENNY_TRANSCRIPTS env var (explicit override)
    2. Walk up from this source file looking for transcripts/episodes/
       (works for editable installs and running from the repo)
    3. Walk up from cwd looking for transcripts/episodes/
    4. Previously-downloaded transcripts in XDG data directory
    """
    # 1. Explicit env var
    env_path = os.environ.get("LENNY_TRANSCRIPTS")
    if env_path and os.path.isdir(env_path):
        return env_path

    # 2. Walk up from source file
    current = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = current / "transcripts" / "episodes"
        if candidate.is_dir():
            return str(candidate)
        if current.parent == current:
            break
        current = current.parent

    # 3. Walk up from cwd
    current = Path.cwd()
    for _ in range(10):
        candidate = current / "transcripts" / "episodes"
        if candidate.is_dir():
            return str(candidate)
        if current.parent == current:
            break
        current = current.parent

    # 4. Previously-downloaded transcripts in data directory
    data_episodes = transcript_data_dir() / "episodes"
    if data_episodes.is_dir():
        return str(data_episodes)

    return None


def ensure_transcripts(console: object) -> str:
    """Locate or download transcripts, returning the episodes directory path.

    In non-interactive environments (stdin is not a TTY), raises
    FileNotFoundError with actionable setup instructions instead of
    prompting.

    Parameters
    ----------
    console:
        A Rich Console instance for display output.
    """
    # Try to find existing transcripts
    episodes_dir = _find_episodes_dir()
    if episodes_dir is not None:
        return episodes_dir

    # Not found — handle based on interactivity
    if not sys.stdin.isatty():
        raise FileNotFoundError(
            "Transcript data not found.\n\n"
            "Set the LENNY_TRANSCRIPTS environment variable to the episodes directory:\n"
            "  export LENNY_TRANSCRIPTS=/path/to/transcripts/episodes\n\n"
            "Or clone the transcripts manually:\n"
            f"  git clone {_TRANSCRIPT_REPO} ~/lenny-transcripts\n"
            "  export LENNY_TRANSCRIPTS=~/lenny-transcripts/episodes\n"
        )

    # Interactive: offer to download
    from rich.prompt import Confirm  # late import to avoid circular deps

    console.print()
    console.print(
        "[accent]Transcript data not found.[/accent]\n\n"
        "  Lenny needs podcast transcripts to work.\n"
        "  I can download them for you (~50 MB).\n"
    )

    dest = transcript_data_dir()
    if not Confirm.ask(
        f"  Download transcripts to {dest}?",
        default=True,
        console=console,
    ):
        raise FileNotFoundError(
            "Transcripts are required to run Lenny.\n\n"
            "To set up manually:\n"
            "  export LENNY_TRANSCRIPTS=/path/to/transcripts/episodes\n"
        )

    with console.status("[accent]Downloading transcripts...[/accent]"):
        success = download_transcripts(dest, console)

    if not success:
        raise FileNotFoundError(
            "Failed to download transcripts.\n\n"
            "To set up manually:\n"
            f"  git clone {_TRANSCRIPT_REPO} {dest}\n"
            f"  export LENNY_TRANSCRIPTS={dest / 'episodes'}\n"
        )

    episodes_dir = str(dest / "episodes")
    console.print(f"  [success]✓[/success] Transcripts downloaded to {dest}")
    return episodes_dir
