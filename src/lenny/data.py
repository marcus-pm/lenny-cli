"""Transcript loading, parsing, and indexing for Lenny's Podcast episodes."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from lenny.cache import TranscriptCache
    from lenny.mcp_client import MCPClient

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    slug: str
    guest: str
    title: str
    youtube_url: str
    video_id: str
    publish_date: str
    description: str
    duration: str
    duration_seconds: float
    view_count: int
    keywords: list[str]
    file_path: str
    filename: str = ""  # MCP filename (e.g. "podcasts/brian-chesky.md")

    def to_catalog_entry(self) -> dict:
        """Compact representation for the RLM context catalog."""
        return {
            "slug": self.slug,
            "guest": self.guest,
            "title": self.title,
            "youtube_url": self.youtube_url,
            "publish_date": self.publish_date,
            "duration": self.duration,
            "keywords": self.keywords,
        }


@dataclass
class TranscriptIndex:
    episodes: dict[str, Episode] = field(default_factory=dict)
    transcript_dir: str = ""
    _mcp_client: MCPClient | None = field(default=None, repr=False)
    _cache: TranscriptCache | None = field(default=None, repr=False)

    @classmethod
    def load_from_mcp(
        cls,
        mcp_client: MCPClient,
        cache: TranscriptCache,
    ) -> TranscriptIndex:
        """Load episode catalog from the MCP server.

        Fetches episode metadata via list_content (paginated) and builds
        Episode objects.  Full transcript content is NOT fetched here —
        use load_transcript(slug) for on-demand retrieval.
        """
        index = cls(
            transcript_dir=cache.path,
            _mcp_client=mcp_client,
            _cache=cache,
        )

        offset = 0
        page_size = 200
        while True:
            page = mcp_client.list_content(
                content_type="podcast", limit=page_size, offset=offset,
            )
            results = page.get("results", [])
            if not results:
                break

            for entry in results:
                filename = entry.get("filename", "")
                # Derive slug from filename: "podcasts/brian-chesky.md" → "brian-chesky"
                slug = filename.removeprefix("podcasts/").removesuffix(".md")
                if not slug:
                    continue

                episode = Episode(
                    slug=slug,
                    guest=entry.get("guest", slug),
                    title=entry.get("title", ""),
                    youtube_url="",  # populated lazily on read_content
                    video_id="",
                    publish_date=entry.get("date", ""),
                    description=entry.get("description", ""),
                    duration="",
                    duration_seconds=0.0,
                    view_count=0,
                    keywords=entry.get("tags", []),
                    file_path="",  # no local file yet
                    filename=filename,
                )
                index.episodes[slug] = episode

            total = page.get("total", 0)
            offset += page_size
            if offset >= total:
                break

        return index

    def get_catalog(self) -> list[dict]:
        """Get a compact catalog of all episodes for the RLM context."""
        return [ep.to_catalog_entry() for ep in self.episodes.values()]

    def load_transcript(self, slug: str) -> str | None:
        """Load the full transcript text for a given episode slug.

        Resolution order:
        1. Local cache (memory + disk)
        2. MCP server (read_content) — result is cached
        3. Local file (legacy install)
        """
        episode = self.episodes.get(slug)
        if episode is None:
            return None

        # 1. Check local cache (MCP mode)
        if self._cache is not None:
            cached = self._cache.get(slug)
            if cached is not None:
                return _strip_frontmatter(cached)

        # 2. Try MCP server
        if self._mcp_client is not None and episode.filename:
            try:
                content = self._mcp_client.read_content(episode.filename)
                # Cache the raw content (with frontmatter) for open() compat
                if self._cache is not None:
                    self._cache.put(slug, content)
                # Backfill youtube_url and other metadata from frontmatter
                self._backfill_metadata(episode, content)
                return _strip_frontmatter(content)
            except Exception as e:
                logger.debug("MCP read_content failed for %s: %s", slug, e)

        # 3. Fall back to local file
        if episode.file_path and os.path.isfile(episode.file_path):
            with open(episode.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return _strip_frontmatter(content)

        return None

    def _backfill_metadata(self, episode: Episode, content: str) -> None:
        """Populate youtube_url and other fields from transcript frontmatter."""
        meta = _parse_frontmatter_text(content)
        if meta is None:
            return
        if not episode.youtube_url:
            episode.youtube_url = meta.get("youtube_url", "")
        if not episode.video_id:
            episode.video_id = meta.get("video_id", "")
        if not episode.duration:
            episode.duration = meta.get("duration", "")
        if not episode.duration_seconds:
            episode.duration_seconds = float(meta.get("duration_seconds", 0))
        if not episode.view_count:
            episode.view_count = int(meta.get("view_count", 0))

    def get_episode_meta(self, slug: str) -> dict | None:
        """Get full metadata for an episode."""
        episode = self.episodes.get(slug)
        if episode is None:
            return None
        return {
            "slug": episode.slug,
            "guest": episode.guest,
            "title": episode.title,
            "youtube_url": episode.youtube_url,
            "video_id": episode.video_id,
            "publish_date": episode.publish_date,
            "description": episode.description,
            "duration": episode.duration,
            "duration_seconds": episode.duration_seconds,
            "view_count": episode.view_count,
            "keywords": episode.keywords,
        }


def _strip_frontmatter(content: str) -> str:
    """Strip YAML frontmatter from markdown content, return body only."""
    match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
    if match:
        content = content[match.end():]
    return content.strip()


def _parse_frontmatter_text(content: str) -> dict | None:
    """Parse YAML frontmatter from a markdown string."""
    if not content.startswith("---"):
        return None
    end = content.find("\n---", 3)
    if end == -1:
        return None
    frontmatter = content[4:end]
    try:
        return yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        return None
