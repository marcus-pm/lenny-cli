"""Tests for lenny.style — UI tokens, themes, splash card, and rendering helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from rich.text import Text
from rich.theme import Theme

from lenny.style import (
    THEMES,
    THEME_MINIMAL,
    THEME_WARM,
    LennyTheme,
    build_splash_art,
    build_splash_card,
    detect_color_depth,
    format_cost_compact,
    format_route_badge,
    format_save_confirmation,
    answer_panel_params,
    is_wide_terminal,
    _pick_daily_prompt,
    PROGRESS_LABELS,
)


# ---------------------------------------------------------------------------
# Terminal capability detection
# ---------------------------------------------------------------------------

class TestDetectColorDepth:

    def test_truecolor_from_env(self):
        with patch.dict("os.environ", {"COLORTERM": "truecolor"}, clear=False):
            console = MagicMock()
            assert detect_color_depth(console) == "truecolor"

    def test_24bit_from_env(self):
        with patch.dict("os.environ", {"COLORTERM": "24bit"}, clear=False):
            console = MagicMock()
            assert detect_color_depth(console) == "truecolor"

    def test_256color_from_term(self):
        with patch.dict("os.environ", {"COLORTERM": "", "TERM": "xterm-256color"}, clear=False):
            console = MagicMock()
            assert detect_color_depth(console) == "256"

    def test_falls_back_to_rich(self):
        with patch.dict("os.environ", {"COLORTERM": "", "TERM": "xterm"}, clear=False):
            console = MagicMock()
            console.color_system = "standard"
            assert detect_color_depth(console) == "standard"

    def test_no_color(self):
        with patch.dict("os.environ", {"COLORTERM": "", "TERM": "dumb"}, clear=False):
            console = MagicMock()
            console.color_system = None
            assert detect_color_depth(console) == "none"


class TestIsWideTerminal:

    def test_wide(self):
        console = MagicMock()
        console.width = 120
        assert is_wide_terminal(console) is True

    def test_narrow(self):
        console = MagicMock()
        console.width = 40
        assert is_wide_terminal(console) is False

    def test_exact_threshold(self):
        console = MagicMock()
        console.width = 80
        assert is_wide_terminal(console, threshold=80) is True

    def test_custom_threshold(self):
        console = MagicMock()
        console.width = 50
        assert is_wide_terminal(console, threshold=60) is False


# ---------------------------------------------------------------------------
# Brand mark / splash art
# ---------------------------------------------------------------------------

class TestBuildSplashArt:

    def _make_console(self, width: int) -> Console:
        """Build a Console with a fixed width for testing."""
        return Console(width=width, force_terminal=True, file=MagicMock())

    def test_wide_terminal_returns_text(self):
        console = self._make_console(120)
        result = build_splash_art(THEME_WARM, console)
        assert isinstance(result, Text)
        plain = result.plain
        assert "@@@@" in plain   # campfire art building blocks

    def test_medium_terminal_returns_none(self):
        console = self._make_console(80)
        result = build_splash_art(THEME_WARM, console)
        assert result is None

    def test_narrow_terminal_returns_none(self):
        console = self._make_console(35)
        result = build_splash_art(THEME_WARM, console)
        assert result is None

    def test_minimal_theme_returns_none(self):
        console = self._make_console(120)
        result = build_splash_art(THEME_MINIMAL, console)
        assert result is None

    def test_art_width_boundary_100(self):
        console = self._make_console(100)
        result = build_splash_art(THEME_WARM, console)
        assert result is not None

    def test_art_width_boundary_99(self):
        console = self._make_console(99)
        result = build_splash_art(THEME_WARM, console)
        assert result is None


# ---------------------------------------------------------------------------
# Splash card
# ---------------------------------------------------------------------------

class TestBuildSplashCard:

    def _make_console(self, width: int = 80) -> Console:
        return Console(width=width, force_terminal=True, file=MagicMock())

    def test_contains_episode_count(self):
        console = self._make_console()
        card = build_splash_card(152, "auto", THEME_WARM, console)
        assert "152" in card.plain

    def test_contains_mode_label(self):
        console = self._make_console()
        card = build_splash_card(100, "research", THEME_WARM, console)
        assert "research" in card.plain

    def test_contains_app_name(self):
        console = self._make_console()
        card = build_splash_card(100, "auto", THEME_WARM, console)
        assert "Lenny" in card.plain

    def test_contains_tagline(self):
        console = self._make_console()
        card = build_splash_card(100, "auto", THEME_WARM, console)
        assert "Podcast transcript explorer" in card.plain

    def test_contains_command_bar(self):
        console = self._make_console()
        card = build_splash_card(100, "auto", THEME_WARM, console)
        assert "/help" in card.plain
        assert "/theme" in card.plain

    def test_warm_theme_has_daily_prompt(self):
        console = self._make_console()
        card = build_splash_card(100, "auto", THEME_WARM, console)
        assert "Try:" in card.plain

    def test_minimal_theme_no_daily_prompt(self):
        console = self._make_console()
        card = build_splash_card(100, "auto", THEME_MINIMAL, console)
        assert "Try:" not in card.plain

    def test_narrow_terminal_no_art(self):
        console = self._make_console(35)
        card = build_splash_card(100, "auto", THEME_WARM, console)
        assert "@@@@" not in card.plain  # no campfire art
        assert "Lenny" in card.plain    # still has app name


# ---------------------------------------------------------------------------
# Daily prompt rotation
# ---------------------------------------------------------------------------

class TestPickDailyPrompt:

    def test_deterministic(self):
        """Same call on the same day returns the same prompt."""
        assert _pick_daily_prompt() == _pick_daily_prompt()

    def test_returns_string(self):
        result = _pick_daily_prompt()
        assert isinstance(result, str)
        assert len(result) > 10


# ---------------------------------------------------------------------------
# Route badge
# ---------------------------------------------------------------------------

class TestFormatRouteBadge:

    def test_fast_badge(self):
        badge = format_route_badge("fast", "specific guest lookup", THEME_WARM)
        plain = badge.plain
        assert "FAST" in plain
        assert "specific guest lookup" in plain

    def test_research_badge(self):
        badge = format_route_badge("research", "cross-episode synthesis", THEME_WARM)
        plain = badge.plain
        assert "RESEARCH" in plain
        assert "cross-episode synthesis" in plain

    def test_badge_works_with_minimal_theme(self):
        fast = format_route_badge("fast", "lookup", THEME_MINIMAL)
        assert "FAST" in fast.plain
        research = format_route_badge("research", "analysis", THEME_MINIMAL)
        assert "RESEARCH" in research.plain


# ---------------------------------------------------------------------------
# Answer panel params
# ---------------------------------------------------------------------------

class TestAnswerPanelParams:

    def test_fast_params(self):
        params = answer_panel_params("fast", THEME_WARM)
        assert "border_style" in params
        assert params["border_style"] == THEME_WARM.border_fast
        assert "Lenny" in params["title"]
        assert "fast" in params["title"]

    def test_research_params(self):
        params = answer_panel_params("research", THEME_WARM)
        assert params["border_style"] == THEME_WARM.border_research
        assert "research" in params["title"]


# ---------------------------------------------------------------------------
# Cost and save helpers
# ---------------------------------------------------------------------------

class TestFormatHelpers:

    def test_format_cost_compact(self):
        result = format_cost_compact("  Query total: $0.01 in 2.0s")
        assert isinstance(result, Text)
        assert "$0.01" in result.plain

    def test_format_save_confirmation(self):
        result = format_save_confirmation("20250315-143045-chesky-founder.md")
        assert isinstance(result, Text)
        assert "20250315-143045-chesky-founder.md" in result.plain
        assert "\u2713" in result.plain  # checkmark


# ---------------------------------------------------------------------------
# Theme definitions
# ---------------------------------------------------------------------------

class TestThemes:

    def test_themes_dict_has_warm_and_minimal(self):
        assert "warm" in THEMES
        assert "minimal" in THEMES

    def test_theme_warm_to_rich_theme(self):
        theme = THEME_WARM.to_rich_theme()
        assert isinstance(theme, Theme)

    def test_theme_minimal_to_rich_theme(self):
        theme = THEME_MINIMAL.to_rich_theme()
        assert isinstance(theme, Theme)

    def test_warm_shows_art(self):
        assert THEME_WARM.show_splash_art is True

    def test_minimal_hides_art(self):
        assert THEME_MINIMAL.show_splash_art is False


# ---------------------------------------------------------------------------
# Progress labels completeness
# ---------------------------------------------------------------------------

class TestProgressLabels:

    def test_all_expected_keys_present(self):
        expected = [
            "reading_transcripts", "analyzing_parallel", "analyzing_ai",
            "searching_text", "scanning_catalog", "preparing_answer",
            "thinking", "analyzing_excerpts", "searching_fast",
            "searching_episodes", "searching_transcripts",
        ]
        for key in expected:
            assert key in PROGRESS_LABELS, f"Missing key: {key}"

    def test_analyzing_excerpts_is_formattable(self):
        result = PROGRESS_LABELS["analyzing_excerpts"].format(n=3, s="s")
        assert "3" in result
        assert "excerpts" in result

    def test_searching_episodes_is_formattable(self):
        result = PROGRESS_LABELS["searching_episodes"].format(n=152)
        assert "152" in result
