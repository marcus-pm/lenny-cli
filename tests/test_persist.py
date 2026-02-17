"""Tests for lenny.persist — slug generation, file saving, and citation formatting."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from lenny.persist import (
    build_query_slug,
    format_terminal_citations,
    save_response_markdown,
)


# ---------------------------------------------------------------------------
# build_query_slug tests
# ---------------------------------------------------------------------------

class TestBuildQuerySlug:

    def test_basic_query(self):
        slug = build_query_slug("What did Brian Chesky say about founder mode?")
        assert "brian" in slug
        assert "chesky" in slug
        assert "founder" in slug

    def test_lowercase(self):
        slug = build_query_slug("Brian Chesky FOUNDER Mode")
        assert slug == slug.lower()

    def test_removes_stopwords(self):
        slug = build_query_slug("What is the best way to do this?")
        assert "what" not in slug
        assert "the" not in slug
        assert "is" not in slug

    def test_max_5_tokens(self):
        slug = build_query_slug(
            "frameworks guests recommend prioritization product management strategy"
        )
        assert slug.count("-") <= 4  # at most 5 tokens = 4 hyphens

    def test_min_2_tokens_fallback(self):
        """When stopword removal leaves < 2 tokens, fall back to all tokens."""
        slug = build_query_slug("Is it?")
        # "is" and "it" are stopwords, but both >1 char so fallback includes them
        assert slug != "response"

    def test_empty_query_fallback(self):
        slug = build_query_slug("")
        assert slug == "response"

    def test_single_char_tokens_fallback(self):
        slug = build_query_slug("a")
        assert slug == "response"

    def test_punctuation_stripped(self):
        slug = build_query_slug("What's the guest's opinion on AI/ML?")
        assert "'" not in slug
        assert "/" not in slug
        assert "?" not in slug

    def test_only_alphanumeric_and_hyphens(self):
        slug = build_query_slug("How do experts — like Brian — handle growth?")
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in slug)

    def test_no_repeated_hyphens(self):
        slug = build_query_slug("compare --- and --- frameworks")
        assert "--" not in slug

    def test_max_length_48(self):
        slug = build_query_slug(
            "superlongwordthatgoeson " * 10
        )
        assert len(slug) <= 48

    def test_deterministic(self):
        query = "What frameworks do guests recommend?"
        assert build_query_slug(query) == build_query_slug(query)

    def test_hyphen_separated(self):
        slug = build_query_slug("Brian Chesky founder mode")
        parts = slug.split("-")
        assert len(parts) >= 2

    def test_numeric_tokens_preserved(self):
        slug = build_query_slug("Top 10 frameworks for 2025")
        assert "10" in slug
        assert "2025" in slug


# ---------------------------------------------------------------------------
# save_response_markdown tests
# ---------------------------------------------------------------------------

class TestSaveResponseMarkdown:

    def test_creates_file(self, tmp_path: Path):
        now = datetime(2025, 3, 15, 14, 30, 45)
        path = save_response_markdown(
            query="What did Brian Chesky say?",
            answer="He said some things about founder mode.",
            mode="fast",
            cost_summary="  Haiku: 1 calls, 5200 in / 1800 out ($0.0114)\n  Query total: $0.0114 in 4.2s",
            output_dir=tmp_path,
            now=now,
        )
        assert path.exists()
        assert path.suffix == ".md"

    def test_filename_format(self, tmp_path: Path):
        now = datetime(2025, 3, 15, 14, 30, 45)
        path = save_response_markdown(
            query="Brian Chesky founder mode",
            answer="Answer here.",
            mode="fast",
            cost_summary="  Query total: $0.01 in 2.0s",
            output_dir=tmp_path,
            now=now,
        )
        # Filename should start with timestamp
        assert path.name.startswith("20250315-143045-")
        assert path.name.endswith(".md")
        # Slug portion should contain informative tokens
        slug_part = path.stem.replace("20250315-143045-", "")
        assert "brian" in slug_part
        assert "chesky" in slug_part

    def test_file_content_has_metadata(self, tmp_path: Path):
        now = datetime(2025, 3, 15, 14, 30, 45)
        path = save_response_markdown(
            query="Test query here",
            answer="The answer body.",
            mode="research",
            cost_summary="  Query total: $0.15 in 42.3s",
            output_dir=tmp_path,
            now=now,
        )
        content = path.read_text(encoding="utf-8")
        # YAML front matter
        assert content.startswith("---\n")
        assert "timestamp: 2025-03-15 14:30:45" in content
        assert 'query: "Test query here"' in content
        assert "route: research" in content
        assert "$0.15" in content
        # Answer body
        assert "The answer body." in content

    def test_file_content_has_answer(self, tmp_path: Path):
        answer = "## Key Findings\n\nGuests recommend several frameworks."
        path = save_response_markdown(
            query="frameworks",
            answer=answer,
            mode="research",
            cost_summary="  Query total: $0.10 in 30.0s",
            output_dir=tmp_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "## Key Findings" in content
        assert "Guests recommend several frameworks." in content

    def test_utf8_encoding(self, tmp_path: Path):
        path = save_response_markdown(
            query="unicode test",
            answer="Caf\u00e9 \u2014 \u2713 checked",
            mode="fast",
            cost_summary="  Query total: $0.01 in 1.0s",
            output_dir=tmp_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "Caf\u00e9" in content
        assert "\u2713" in content

    def test_collision_suffix(self, tmp_path: Path):
        now = datetime(2025, 1, 1, 12, 0, 0)
        # Create first file
        path1 = save_response_markdown(
            query="same query",
            answer="First answer.",
            mode="fast",
            cost_summary="  cost",
            output_dir=tmp_path,
            now=now,
        )
        # Create second file with same timestamp + query
        path2 = save_response_markdown(
            query="same query",
            answer="Second answer.",
            mode="fast",
            cost_summary="  cost",
            output_dir=tmp_path,
            now=now,
        )
        assert path1 != path2
        assert path1.exists()
        assert path2.exists()
        assert "-1" in path2.stem

    def test_double_collision_suffix(self, tmp_path: Path):
        now = datetime(2025, 1, 1, 12, 0, 0)
        path1 = save_response_markdown(
            query="same query", answer="a", mode="fast",
            cost_summary="c", output_dir=tmp_path, now=now,
        )
        path2 = save_response_markdown(
            query="same query", answer="b", mode="fast",
            cost_summary="c", output_dir=tmp_path, now=now,
        )
        path3 = save_response_markdown(
            query="same query", answer="c", mode="fast",
            cost_summary="c", output_dir=tmp_path, now=now,
        )
        assert "-1" in path2.stem
        assert "-2" in path3.stem

    def test_chronological_sort(self, tmp_path: Path):
        """Files should sort chronologically via alphabetical sort."""
        times = [
            datetime(2025, 1, 1, 9, 0, 0),
            datetime(2025, 1, 1, 10, 0, 0),
            datetime(2025, 1, 2, 8, 0, 0),
        ]
        paths = []
        for i, t in enumerate(times):
            p = save_response_markdown(
                query=f"query number {i}",
                answer="a", mode="fast", cost_summary="c",
                output_dir=tmp_path, now=t,
            )
            paths.append(p)

        sorted_names = sorted(p.name for p in paths)
        assert sorted_names == [p.name for p in paths]

    def test_citations_preserved_in_file(self, tmp_path: Path):
        """Saved file should contain the original Markdown with usable URLs."""
        answer = (
            "**Brian Chesky** in *Founder Mode* "
            "([link](https://youtube.com/watch?v=abc123))"
        )
        path = save_response_markdown(
            query="chesky founder",
            answer=answer,
            mode="fast",
            cost_summary="  cost",
            output_dir=tmp_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "https://youtube.com/watch?v=abc123" in content


# ---------------------------------------------------------------------------
# format_terminal_citations tests
# ---------------------------------------------------------------------------

class TestFormatTerminalCitations:

    def test_link_citation_pattern(self):
        """The ([link](url)) pattern is transformed to a plain URL line."""
        text = (
            "**Brian Chesky** in *Founder Mode* "
            "([link](https://youtube.com/watch?v=abc123))"
        )
        result = format_terminal_citations(text)
        assert "https://youtube.com/watch?v=abc123" in result
        assert "[link]" not in result

    def test_url_on_separate_line(self):
        """The URL should appear as a separate indented line."""
        text = (
            "**Brian Chesky** in *Founder Mode* "
            "([link](https://youtube.com/watch?v=abc123))"
        )
        result = format_terminal_citations(text)
        lines = result.split("\n")
        url_lines = [l for l in lines if "https://youtube.com" in l]
        assert len(url_lines) == 1
        assert url_lines[0].startswith("  ")

    def test_guest_name_preserved(self):
        text = (
            "**Brian Chesky** in *Founder Mode* "
            "([link](https://youtube.com/watch?v=abc123))"
        )
        result = format_terminal_citations(text)
        assert "**Brian Chesky**" in result
        assert "*Founder Mode*" in result

    def test_watch_variant(self):
        """([watch](url)) should also be handled."""
        text = "See the episode ([watch](https://youtube.com/watch?v=xyz))"
        result = format_terminal_citations(text)
        assert "https://youtube.com/watch?v=xyz" in result
        assert "[watch]" not in result

    def test_youtube_variant(self):
        """([YouTube](url)) should also be handled."""
        text = "Episode reference ([YouTube](https://youtube.com/watch?v=xyz))"
        result = format_terminal_citations(text)
        assert "https://youtube.com/watch?v=xyz" in result

    def test_inline_markdown_link(self):
        """General [text](url) links get url shown as plain text."""
        text = "See [this article](https://example.com/article) for details."
        result = format_terminal_citations(text)
        assert "https://example.com/article" in result
        assert "[this article]" not in result

    def test_no_links_unchanged(self):
        text = "This is plain text with no links at all."
        result = format_terminal_citations(text)
        assert result == text

    def test_multiple_citations(self):
        text = (
            "- **Guest A** in *Episode 1* ([link](https://youtube.com/a))\n"
            "- **Guest B** in *Episode 2* ([link](https://youtube.com/b))"
        )
        result = format_terminal_citations(text)
        assert "https://youtube.com/a" in result
        assert "https://youtube.com/b" in result

    def test_mixed_content(self):
        """Lines without links should pass through unchanged."""
        text = (
            "## Summary\n"
            "\n"
            "Several guests discussed this topic.\n"
            "\n"
            "- **Brian Chesky** in *Founder Mode* ([link](https://youtube.com/abc))\n"
            "\n"
            "The key takeaway is..."
        )
        result = format_terminal_citations(text)
        assert "## Summary" in result
        assert "Several guests discussed this topic." in result
        assert "The key takeaway is..." in result
        assert "https://youtube.com/abc" in result

    def test_preserves_multiline_structure(self):
        """Non-citation lines should not be removed or collapsed."""
        text = "Line 1\n\nLine 2\n\nLine 3"
        result = format_terminal_citations(text)
        assert result == text
