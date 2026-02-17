"""BM25 paragraph-level search index over podcast transcripts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from lenny.data import TranscriptIndex

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1


@dataclass
class ParagraphChunk:
    """A chunk of transcript text with episode metadata."""
    episode_slug: str
    guest: str
    title: str
    youtube_url: str
    text: str
    paragraph_index: int


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------
_MIN_CHUNK_CHARS = 50
_TARGET_CHUNK_CHARS = 800
_OVERLAP_CHARS = 100


def _split_into_chunks(text: str, episode_slug: str, guest: str,
                        title: str, youtube_url: str) -> list[ParagraphChunk]:
    """Split transcript text into overlapping chunks of ~500-1000 chars.

    Strategy:
    1. Split on double newlines to get raw paragraphs.
    2. Merge adjacent paragraphs into chunks targeting ~800 chars.
    3. Add ~100 char overlap between consecutive chunks for context continuity.
    4. Filter out chunks shorter than 50 chars.
    """
    raw_paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not raw_paragraphs:
        return []

    chunks: list[ParagraphChunk] = []
    current_text = ""
    chunk_idx = 0

    for para in raw_paragraphs:
        if not current_text:
            current_text = para
        elif len(current_text) + len(para) + 2 <= _TARGET_CHUNK_CHARS:
            current_text += "\n\n" + para
        else:
            # Emit current chunk
            if len(current_text) >= _MIN_CHUNK_CHARS:
                chunks.append(ParagraphChunk(
                    episode_slug=episode_slug,
                    guest=guest,
                    title=title,
                    youtube_url=youtube_url,
                    text=current_text,
                    paragraph_index=chunk_idx,
                ))
                chunk_idx += 1

            # Start new chunk with overlap from the end of the previous
            if len(current_text) > _OVERLAP_CHARS:
                # Take the last ~100 chars as overlap, breaking at a word boundary
                overlap_start = current_text[-_OVERLAP_CHARS:]
                word_break = overlap_start.find(" ")
                if word_break > 0:
                    overlap_start = overlap_start[word_break + 1:]
                current_text = overlap_start + "\n\n" + para
            else:
                current_text = para

    # Emit final chunk
    if current_text and len(current_text) >= _MIN_CHUNK_CHARS:
        chunks.append(ParagraphChunk(
            episode_slug=episode_slug,
            guest=guest,
            title=title,
            youtube_url=youtube_url,
            text=current_text,
            paragraph_index=chunk_idx,
        ))

    return chunks


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercasing tokenizer."""
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Search index
# ---------------------------------------------------------------------------
def _compute_content_hash(transcript_dir: str) -> str:
    """Compute a fast fingerprint of all transcript files.

    Hashes the sorted list of (filename, mtime_ns, size) tuples.
    Detects file additions, deletions, or modifications without
    reading file contents.
    """
    entries: list[str] = []
    for name in sorted(os.listdir(transcript_dir)):
        transcript_path = os.path.join(transcript_dir, name, "transcript.md")
        try:
            st = os.stat(transcript_path)
            entries.append(f"{name}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            continue
    fingerprint = "\n".join(entries)
    return hashlib.sha256(fingerprint.encode()).hexdigest()


@dataclass
class _CacheMetadata:
    """Stored alongside the index to detect staleness."""
    transcript_dir: str
    transcript_dir_mtime: float
    episode_count: int
    content_hash: str = ""


class TranscriptSearchIndex:
    """BM25 search index over paragraph-level chunks of all transcripts."""

    def __init__(
        self,
        chunks: list[ParagraphChunk],
        bm25: BM25Okapi,
        cache_meta: _CacheMetadata | None = None,
    ):
        self.chunks = chunks
        self.bm25 = bm25
        self._cache_meta = cache_meta

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, index: TranscriptIndex) -> TranscriptSearchIndex:
        """Build a BM25 index from all transcripts in the given index."""
        all_chunks: list[ParagraphChunk] = []

        for slug, episode in index.episodes.items():
            transcript = index.load_transcript(slug)
            if not transcript:
                continue

            episode_chunks = _split_into_chunks(
                text=transcript,
                episode_slug=slug,
                guest=episode.guest,
                title=episode.title,
                youtube_url=episode.youtube_url,
            )
            all_chunks.extend(episode_chunks)

        # Tokenize all chunks for BM25
        tokenized = [_tokenize(chunk.text) for chunk in all_chunks]
        bm25 = BM25Okapi(tokenized)

        cache_meta = _CacheMetadata(
            transcript_dir=index.transcript_dir,
            transcript_dir_mtime=os.path.getmtime(index.transcript_dir),
            episode_count=len(index.episodes),
            content_hash=_compute_content_hash(index.transcript_dir),
        )

        return cls(chunks=all_chunks, bm25=bm25, cache_meta=cache_meta)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 20) -> list[ParagraphChunk]:
        """Search for the most relevant chunks matching the query."""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]

    def search_with_scores(
        self, query: str, top_k: int = 20,
    ) -> list[tuple[ParagraphChunk, float]]:
        """Search and return chunks with their BM25 scores."""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], scores[i]) for i in top_indices if scores[i] > 0]

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------
    def save(self, path: str) -> bool:
        """Save the index to disk as a JSON file (atomic write).

        Writes to a temporary file first, then atomically replaces the
        target path.  Returns True on success, False on any I/O error
        (e.g. permission denied, read-only filesystem).
        """
        try:
            cache_dir = os.path.dirname(path)
            os.makedirs(cache_dir, exist_ok=True)

            payload = {
                "version": _CACHE_VERSION,
                "cache_meta": asdict(self._cache_meta) if self._cache_meta else {},
                "chunks": [asdict(c) for c in self.chunks],
            }

            fd, tmp_path = tempfile.mkstemp(
                dir=cache_dir, suffix=".tmp", prefix=".bm25_",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                os.replace(tmp_path, path)
            except BaseException:
                # Clean up temp file on any failure (including KeyboardInterrupt)
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            return True
        except OSError:
            return False

    @classmethod
    def _load_from_cache(cls, path: str) -> TranscriptSearchIndex | None:
        """Load an index from a JSON cache file, or None on any error.

        Reconstructs ParagraphChunk objects from JSON dicts, re-tokenizes
        the chunk texts, and rebuilds the BM25 index in memory.  This
        avoids pickle deserialization entirely.
        """
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("version") != _CACHE_VERSION:
                logger.debug("Cache version mismatch: %s", data.get("version"))
                return None

            chunks = [ParagraphChunk(**c) for c in data["chunks"]]
            tokenized = [_tokenize(chunk.text) for chunk in chunks]
            bm25 = BM25Okapi(tokenized)
            meta = _CacheMetadata(**data["cache_meta"])

            return cls(chunks=chunks, bm25=bm25, cache_meta=meta)
        except (json.JSONDecodeError, KeyError, TypeError, Exception) as e:
            logger.debug("Cache load failed: %s", e)
            return None

    @classmethod
    def load_or_build(
        cls, index: TranscriptIndex, cache_path: str,
    ) -> TranscriptSearchIndex:
        """Load from disk cache if fresh, otherwise build and cache.

        The cache is considered stale if:
        - The JSON file doesn't exist or is corrupted
        - The transcript directory mtime has changed
        - The episode count has changed
        - The content hash (file mtimes + sizes) has changed

        If the cache path is unwritable (e.g. read-only filesystem),
        the index is built in memory and returned without caching.
        """
        cached = cls._load_from_cache(cache_path)
        if cached is not None and cached._cache_meta is not None:
            current_mtime = os.path.getmtime(index.transcript_dir)
            current_hash = _compute_content_hash(index.transcript_dir)
            meta = cached._cache_meta
            if (
                meta.transcript_dir == index.transcript_dir
                and meta.transcript_dir_mtime == current_mtime
                and meta.episode_count == len(index.episodes)
                and meta.content_hash == current_hash
            ):
                return cached

        # Build fresh and attempt to cache (non-fatal if write fails)
        fresh = cls.build(index)
        fresh.save(cache_path)  # returns False on error — that's fine

        # Clean up legacy pickle cache if present
        legacy_pkl = cache_path.replace(".json", ".pkl")
        if legacy_pkl != cache_path:
            try:
                os.unlink(legacy_pkl)
            except OSError:
                pass

        return fresh
