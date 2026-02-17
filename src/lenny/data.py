"""Transcript loading, parsing, and indexing for Lenny's Podcast episodes."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


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

    @classmethod
    def load(cls, transcript_dir: str | None = None) -> TranscriptIndex:
        """Load all episode metadata from the transcripts directory."""
        if transcript_dir is None:
            transcript_dir = _find_transcript_dir()

        if transcript_dir is None or not os.path.isdir(transcript_dir):
            raise FileNotFoundError(
                "Transcript data not found.\n"
                "Set the LENNY_TRANSCRIPTS environment variable to the episodes directory,\n"
                "or run `lenny` interactively to download transcripts automatically."
            )

        index = cls(transcript_dir=transcript_dir)

        for entry in sorted(os.listdir(transcript_dir)):
            episode_dir = os.path.join(transcript_dir, entry)
            transcript_path = os.path.join(episode_dir, "transcript.md")
            if not os.path.isfile(transcript_path):
                continue

            meta = _parse_frontmatter(transcript_path)
            if meta is None:
                continue

            episode = Episode(
                slug=entry,
                guest=meta.get("guest", entry),
                title=meta.get("title", ""),
                youtube_url=meta.get("youtube_url", ""),
                video_id=meta.get("video_id", ""),
                publish_date=str(meta.get("publish_date", "")),
                description=meta.get("description", ""),
                duration=meta.get("duration", ""),
                duration_seconds=float(meta.get("duration_seconds", 0)),
                view_count=int(meta.get("view_count", 0)),
                keywords=meta.get("keywords", []),
                file_path=transcript_path,
            )
            index.episodes[entry] = episode

        return index

    def get_catalog(self) -> list[dict]:
        """Get a compact catalog of all episodes for the RLM context."""
        return [ep.to_catalog_entry() for ep in self.episodes.values()]

    def load_transcript(self, slug: str) -> str | None:
        """Load the full transcript text for a given episode slug."""
        episode = self.episodes.get(slug)
        if episode is None:
            return None
        with open(episode.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Strip frontmatter, return just the transcript body
        match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
        if match:
            content = content[match.end():]
        return content.strip()

    def search_transcripts(self, keyword: str) -> list[dict]:
        """Search all transcripts for a keyword, return matching slugs with snippets."""
        keyword_lower = keyword.lower()
        results = []
        for slug, episode in self.episodes.items():
            with open(episode.file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Search in content (case-insensitive)
            content_lower = content.lower()
            idx = content_lower.find(keyword_lower)
            if idx == -1:
                # Also check keywords list
                if any(keyword_lower in kw.lower() for kw in episode.keywords):
                    results.append({
                        "slug": slug,
                        "guest": episode.guest,
                        "title": episode.title,
                        "youtube_url": episode.youtube_url,
                        "match_type": "keyword_tag",
                        "snippet": f"Tagged with keyword matching '{keyword}'",
                    })
                continue

            # Extract a snippet around the match
            start = max(0, idx - 100)
            end = min(len(content), idx + len(keyword) + 100)
            snippet = content[start:end].replace("\n", " ").strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."

            results.append({
                "slug": slug,
                "guest": episode.guest,
                "title": episode.title,
                "youtube_url": episode.youtube_url,
                "match_type": "content",
                "snippet": snippet,
            })

        return results

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


def _find_transcript_dir() -> str | None:
    """Locate the transcripts/episodes directory.

    Search order:
    1. LENNY_TRANSCRIPTS env var (explicit override)
    2. Walk up from this source file looking for transcripts/episodes/
       (works for editable installs and running from the repo)
    3. Walk up from cwd looking for transcripts/episodes/
    4. Previously-downloaded transcripts in XDG data directory

    Returns None if no transcript directory is found.
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

    # 4. Check XDG data directory for downloaded transcripts
    try:
        from lenny.transcripts import transcript_data_dir
        data_episodes = transcript_data_dir() / "episodes"
        if data_episodes.is_dir():
            return str(data_episodes)
    except ImportError:
        pass

    return None


def _parse_frontmatter(filepath: str) -> dict | None:
    """Parse YAML frontmatter from a markdown file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

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
