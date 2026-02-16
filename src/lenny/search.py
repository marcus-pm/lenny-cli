"""BM25 paragraph-level search index over podcast transcripts."""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from lenny.data import TranscriptIndex


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
@dataclass
class _CacheMetadata:
    """Stored alongside the index to detect staleness."""
    transcript_dir: str
    transcript_dir_mtime: float
    episode_count: int


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
        """Save the index to disk as a pickle file.

        Returns True if saved successfully, False on any I/O error
        (e.g. permission denied, read-only filesystem).
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({
                    "chunks": self.chunks,
                    "bm25": self.bm25,
                    "cache_meta": self._cache_meta,
                }, f)
            return True
        except OSError:
            return False

    @classmethod
    def _load_from_cache(cls, path: str) -> TranscriptSearchIndex | None:
        """Load an index from a pickle cache file, or None if not found."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return cls(
                chunks=data["chunks"],
                bm25=data["bm25"],
                cache_meta=data.get("cache_meta"),
            )
        except (pickle.UnpicklingError, KeyError, EOFError, Exception):
            return None

    @classmethod
    def load_or_build(
        cls, index: TranscriptIndex, cache_path: str,
    ) -> TranscriptSearchIndex:
        """Load from disk cache if fresh, otherwise build and cache.

        The cache is considered stale if:
        - The pickle file doesn't exist or is corrupted
        - The transcript directory mtime has changed
        - The episode count has changed

        If the cache path is unwritable (e.g. read-only filesystem),
        the index is built in memory and returned without caching.
        """
        cached = cls._load_from_cache(cache_path)
        if cached is not None and cached._cache_meta is not None:
            current_mtime = os.path.getmtime(index.transcript_dir)
            meta = cached._cache_meta
            if (
                meta.transcript_dir == index.transcript_dir
                and meta.transcript_dir_mtime == current_mtime
                and meta.episode_count == len(index.episodes)
            ):
                return cached

        # Build fresh and attempt to cache (non-fatal if write fails)
        fresh = cls.build(index)
        fresh.save(cache_path)  # returns False on error — that's fine
        return fresh
