"""Tests for lenny.search — JSON cache, content hashing, and BM25 index."""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lenny.search import (
    ParagraphChunk,
    TranscriptSearchIndex,
    _CacheMetadata,
    _compute_content_hash,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(n: int = 5) -> list[ParagraphChunk]:
    """Create synthetic chunks for testing."""
    return [
        ParagraphChunk(
            episode_slug=f"ep-{i}",
            guest=f"Guest {i}",
            title=f"Episode {i} Title",
            youtube_url=f"https://youtube.com/watch?v={i}",
            text=f"This is the transcript content for episode {i}. "
                 f"It discusses topic {i} and mentions concept {i}.",
            paragraph_index=0,
        )
        for i in range(n)
    ]


def _build_index(chunks: list[ParagraphChunk]) -> TranscriptSearchIndex:
    """Build a TranscriptSearchIndex from pre-made chunks."""
    from rank_bm25 import BM25Okapi
    tokenized = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    meta = _CacheMetadata(
        transcript_dir="/tmp/test-transcripts",
        transcript_dir_mtime=1000.0,
        episode_count=len(chunks),
        content_hash="abc123",
    )
    return TranscriptSearchIndex(chunks=chunks, bm25=bm25, cache_meta=meta)


def _make_transcript_dir(tmp_path: Path, count: int = 3) -> str:
    """Create a fake transcript directory structure."""
    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir()
    for i in range(count):
        ep_dir = episodes_dir / f"ep-{i:03d}"
        ep_dir.mkdir()
        (ep_dir / "transcript.md").write_text(
            f"---\ntitle: Episode {i}\n---\nContent for episode {i}."
        )
    return str(episodes_dir)


# ---------------------------------------------------------------------------
# JSON cache round-trip
# ---------------------------------------------------------------------------

class TestSaveAndLoad:
    def test_roundtrip_preserves_chunks(self, tmp_path):
        chunks = _make_chunks(5)
        idx = _build_index(chunks)
        cache_path = str(tmp_path / "cache.json")

        assert idx.save(cache_path) is True
        loaded = TranscriptSearchIndex._load_from_cache(cache_path)

        assert loaded is not None
        assert len(loaded.chunks) == 5
        for orig, loaded_c in zip(chunks, loaded.chunks):
            assert loaded_c.episode_slug == orig.episode_slug
            assert loaded_c.guest == orig.guest
            assert loaded_c.title == orig.title
            assert loaded_c.youtube_url == orig.youtube_url
            assert loaded_c.text == orig.text
            assert loaded_c.paragraph_index == orig.paragraph_index

    def test_roundtrip_search_produces_same_results(self, tmp_path):
        chunks = _make_chunks(5)
        idx = _build_index(chunks)
        cache_path = str(tmp_path / "cache.json")
        idx.save(cache_path)

        loaded = TranscriptSearchIndex._load_from_cache(cache_path)
        assert loaded is not None

        original_results = idx.search("episode 2 topic")
        loaded_results = loaded.search("episode 2 topic")

        assert len(original_results) == len(loaded_results)
        for orig, loaded_r in zip(original_results, loaded_results):
            assert orig.episode_slug == loaded_r.episode_slug

    def test_roundtrip_preserves_metadata(self, tmp_path):
        chunks = _make_chunks(3)
        idx = _build_index(chunks)
        cache_path = str(tmp_path / "cache.json")
        idx.save(cache_path)

        loaded = TranscriptSearchIndex._load_from_cache(cache_path)
        assert loaded is not None
        assert loaded._cache_meta is not None
        assert loaded._cache_meta.transcript_dir == "/tmp/test-transcripts"
        assert loaded._cache_meta.transcript_dir_mtime == 1000.0
        assert loaded._cache_meta.episode_count == 3
        assert loaded._cache_meta.content_hash == "abc123"

    def test_save_creates_directory(self, tmp_path):
        chunks = _make_chunks(2)
        idx = _build_index(chunks)
        cache_path = str(tmp_path / "subdir" / "deep" / "cache.json")

        assert idx.save(cache_path) is True
        assert os.path.exists(cache_path)

    def test_json_file_is_valid_json(self, tmp_path):
        chunks = _make_chunks(2)
        idx = _build_index(chunks)
        cache_path = str(tmp_path / "cache.json")
        idx.save(cache_path)

        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["version"] == 1
        assert "cache_meta" in data
        assert "chunks" in data
        assert len(data["chunks"]) == 2


# ---------------------------------------------------------------------------
# Cache load failure cases
# ---------------------------------------------------------------------------

class TestLoadFailures:
    def test_missing_file_returns_none(self, tmp_path):
        result = TranscriptSearchIndex._load_from_cache(
            str(tmp_path / "nonexistent.json")
        )
        assert result is None

    def test_corrupted_json_returns_none(self, tmp_path):
        cache_path = str(tmp_path / "bad.json")
        with open(cache_path, "w") as f:
            f.write("{not valid json")
        assert TranscriptSearchIndex._load_from_cache(cache_path) is None

    def test_partial_json_returns_none(self, tmp_path):
        """Truncated write (e.g. from a crash) should return None."""
        cache_path = str(tmp_path / "partial.json")
        with open(cache_path, "w") as f:
            f.write('{"version": 1, "cache_meta": {}, "chunks": [{"episode_slug":')
        assert TranscriptSearchIndex._load_from_cache(cache_path) is None

    def test_wrong_version_returns_none(self, tmp_path):
        cache_path = str(tmp_path / "future.json")
        with open(cache_path, "w") as f:
            json.dump({"version": 99, "cache_meta": {}, "chunks": []}, f)
        assert TranscriptSearchIndex._load_from_cache(cache_path) is None

    def test_missing_chunks_key_returns_none(self, tmp_path):
        cache_path = str(tmp_path / "nokey.json")
        with open(cache_path, "w") as f:
            json.dump({"version": 1, "cache_meta": {}}, f)
        assert TranscriptSearchIndex._load_from_cache(cache_path) is None

    def test_empty_file_returns_none(self, tmp_path):
        cache_path = str(tmp_path / "empty.json")
        Path(cache_path).touch()
        assert TranscriptSearchIndex._load_from_cache(cache_path) is None


# ---------------------------------------------------------------------------
# Atomic write safety
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_no_temp_files_after_successful_save(self, tmp_path):
        chunks = _make_chunks(2)
        idx = _build_index(chunks)
        cache_path = str(tmp_path / "cache.json")
        idx.save(cache_path)

        files = os.listdir(tmp_path)
        assert files == ["cache.json"], f"Unexpected files: {files}"

    def test_save_to_readonly_dir_returns_false(self, tmp_path):
        """Save to a non-existent path under /dev/null should fail gracefully."""
        chunks = _make_chunks(2)
        idx = _build_index(chunks)
        # /dev/null/subdir is not a real directory
        result = idx.save("/dev/null/subdir/cache.json")
        assert result is False


# ---------------------------------------------------------------------------
# No pickle regression
# ---------------------------------------------------------------------------

class TestNoPicle:
    def test_pickle_not_imported(self):
        """Ensure pickle is not imported in the search module."""
        import lenny.search as search_mod
        source = inspect.getsource(search_mod)
        assert "import pickle" not in source
        assert "pickle.load" not in source
        assert "pickle.dump" not in source


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_deterministic(self, tmp_path):
        transcript_dir = _make_transcript_dir(tmp_path)
        h1 = _compute_content_hash(transcript_dir)
        h2 = _compute_content_hash(transcript_dir)
        assert h1 == h2
        assert len(h1) == 64  # full SHA-256 hex

    def test_changes_on_file_edit(self, tmp_path):
        transcript_dir = _make_transcript_dir(tmp_path)
        h1 = _compute_content_hash(transcript_dir)

        # Modify a transcript file (change content → changes mtime + size)
        time.sleep(0.05)  # ensure mtime_ns differs
        ep_file = Path(transcript_dir) / "ep-001" / "transcript.md"
        ep_file.write_text("---\ntitle: Modified\n---\nNew content here.")
        h2 = _compute_content_hash(transcript_dir)

        assert h1 != h2

    def test_changes_on_file_addition(self, tmp_path):
        transcript_dir = _make_transcript_dir(tmp_path, count=2)
        h1 = _compute_content_hash(transcript_dir)

        # Add a new episode
        new_ep = Path(transcript_dir) / "ep-999"
        new_ep.mkdir()
        (new_ep / "transcript.md").write_text("---\ntitle: New\n---\nNew ep.")
        h2 = _compute_content_hash(transcript_dir)

        assert h1 != h2

    def test_changes_on_file_deletion(self, tmp_path):
        transcript_dir = _make_transcript_dir(tmp_path, count=3)
        h1 = _compute_content_hash(transcript_dir)

        # Delete an episode's transcript
        (Path(transcript_dir) / "ep-002" / "transcript.md").unlink()
        h2 = _compute_content_hash(transcript_dir)

        assert h1 != h2

    def test_empty_dir(self, tmp_path):
        transcript_dir = tmp_path / "empty"
        transcript_dir.mkdir()
        h = _compute_content_hash(str(transcript_dir))
        assert isinstance(h, str)
        assert len(h) == 64


# ---------------------------------------------------------------------------
# Staleness detection integration
# ---------------------------------------------------------------------------

class TestStalenessDetection:
    def test_cache_hit_when_unchanged(self, tmp_path):
        """Cache should be used when transcripts haven't changed."""
        transcript_dir = _make_transcript_dir(tmp_path)
        # Build a mock TranscriptIndex
        index = MagicMock()
        index.transcript_dir = transcript_dir
        index.episodes = {"ep-000": None, "ep-001": None, "ep-002": None}

        # Create a cache file manually
        chunks = _make_chunks(3)
        meta = _CacheMetadata(
            transcript_dir=transcript_dir,
            transcript_dir_mtime=os.path.getmtime(transcript_dir),
            episode_count=3,
            content_hash=_compute_content_hash(transcript_dir),
        )
        idx = TranscriptSearchIndex(
            chunks=chunks,
            bm25=__import__("rank_bm25").BM25Okapi(
                [_tokenize(c.text) for c in chunks]
            ),
            cache_meta=meta,
        )
        cache_path = str(tmp_path / "cache.json")
        idx.save(cache_path)

        # load_or_build should return cached version (build not called)
        result = TranscriptSearchIndex.load_or_build(index, cache_path)
        index.load_transcript.assert_not_called()  # build() wasn't invoked
        assert len(result.chunks) == 3
