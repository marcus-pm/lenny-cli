"""Local transcript cache for MCP-fetched content.

Provides a two-tier cache (in-memory + disk) so that the RLM research
path's open() calls work on transcripts fetched from the MCP server.
"""

from __future__ import annotations

import os
from pathlib import Path


def _cache_base_dir() -> Path:
    """Return the base cache directory, respecting XDG_CACHE_HOME."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "lenny" / "transcripts"


class TranscriptCache:
    """Two-tier transcript cache: in-memory dict + disk files.

    Disk layout mirrors the local transcript directory structure so that
    the RLM sandbox's restricted open() can read cached transcripts:
        {cache_dir}/{slug}/transcript.md
    """

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else _cache_base_dir()
        self._memory: dict[str, str] = {}

    @property
    def path(self) -> str:
        """Return the cache directory path as a string (for open() paths)."""
        return str(self.cache_dir)

    def has(self, slug: str) -> bool:
        """Check if a transcript is cached (memory or disk)."""
        if slug in self._memory:
            return True
        return self._transcript_path(slug).is_file()

    def get(self, slug: str) -> str | None:
        """Read a cached transcript, or None if not cached."""
        # Check in-memory first
        if slug in self._memory:
            return self._memory[slug]

        # Check disk
        path = self._transcript_path(slug)
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            self._memory[slug] = content
            return content

        return None

    def put(self, slug: str, content: str) -> None:
        """Cache a transcript to memory and disk."""
        self._memory[slug] = content

        path = self._transcript_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _transcript_path(self, slug: str) -> Path:
        """Return the disk path for a cached transcript."""
        return self.cache_dir / slug / "transcript.md"
