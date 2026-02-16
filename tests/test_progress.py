"""Tests for lenny.progress — ProgressDisplay status extraction and lifecycle."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from rich.live import Live

from rlm.core.types import (
    CodeBlock,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
    ModelUsageSummary,
    UsageSummary,
)

from lenny.progress import ProgressDisplay, _infer_from_code, _truncate


# ---------------------------------------------------------------------------
# Helpers for building mock iterations
# ---------------------------------------------------------------------------

def _make_iteration(
    *,
    response: str = "",
    stdout: str = "",
    stderr: str = "",
    code: str = "",
    rlm_calls: list | None = None,
    final_answer: str | None = None,
) -> RLMIteration:
    """Build a minimal RLMIteration for testing."""
    result = REPLResult(
        stdout=stdout,
        stderr=stderr,
        locals={},
        execution_time=1.0,
        rlm_calls=rlm_calls or [],
    )
    blocks = [CodeBlock(code=code, result=result)] if code or stdout else []
    return RLMIteration(
        prompt="test",
        response=response,
        code_blocks=blocks,
        final_answer=final_answer,
    )


def _make_rlm_call() -> RLMChatCompletion:
    """Build a minimal sub-LLM call record."""
    return RLMChatCompletion(
        root_model="haiku",
        prompt="p",
        response="r",
        usage_summary=UsageSummary(
            {"m": ModelUsageSummary(1, 100, 50)}
        ),
        execution_time=1.0,
    )


# ---------------------------------------------------------------------------
# _extract_status tests
# ---------------------------------------------------------------------------

class TestExtractStatus:
    """Test ProgressDisplay._extract_status under various iteration shapes."""

    def _make_progress(self) -> ProgressDisplay:
        console = MagicMock()
        return ProgressDisplay(console, initial_status="Searching...")

    def test_final_answer_set(self):
        p = self._make_progress()
        it = _make_iteration(final_answer="Here is the answer")
        assert p._extract_status(it) == "Preparing answer..."

    def test_final_call_in_response(self):
        """FINAL(...) at start of line triggers preparing answer."""
        p = self._make_progress()
        it = _make_iteration(response="Let me wrap up.\nFINAL(my_answer)")
        assert p._extract_status(it) == "Preparing answer..."

    def test_final_var_in_response(self):
        """FINAL_VAR(...) also triggers preparing answer."""
        p = self._make_progress()
        it = _make_iteration(response="Done.\nFINAL_VAR(result)")
        assert p._extract_status(it) == "Preparing answer..."

    def test_incidental_final_word_does_not_trigger(self):
        """The word FINAL in prose should NOT trigger 'Preparing answer'."""
        p = self._make_progress()
        it = _make_iteration(
            response="The FINAL step is to analyze the data.",
            code="x = 1",
        )
        assert p._extract_status(it) != "Preparing answer..."

    def test_stdout_line_preferred(self):
        """A meaningful stdout line is used as status."""
        p = self._make_progress()
        it = _make_iteration(
            code="print('hello')",
            stdout="Found 12 relevant excerpts",
        )
        assert p._extract_status(it) == "Found 12 relevant excerpts"

    def test_stdout_noise_filtered(self):
        """Blank lines, JSON dumps, and tracebacks are skipped."""
        p = self._make_progress()
        it = _make_iteration(
            code="x = 1",
            stdout='  \n{"key": "val"}\n[1,2,3]\nTraceback (most recent call):\nhi',
        )
        # Only "hi" should be too short (2 chars) — but wait, it's < 5 chars
        # So nothing qualifies → falls through to code inference
        assert p._extract_status(it) != '{"key": "val"}'

    def test_stdout_takes_last_meaningful_line(self):
        p = self._make_progress()
        it = _make_iteration(
            code="print stuff",
            stdout="First line of output\nSecond line of output\nThird line wins",
        )
        assert p._extract_status(it) == "Third line wins"

    def test_sub_llm_calls_counted(self):
        """When code blocks have rlm_calls, status reports the count."""
        p = self._make_progress()
        it = _make_iteration(
            code="llm_query_batched(prompts)",
            rlm_calls=[_make_rlm_call(), _make_rlm_call(), _make_rlm_call()],
        )
        assert p._extract_status(it) == "Analyzing 3 excerpts..."

    def test_single_sub_llm_call_no_plural(self):
        p = self._make_progress()
        it = _make_iteration(
            code="llm_query(prompt)",
            rlm_calls=[_make_rlm_call()],
        )
        assert p._extract_status(it) == "Analyzing 1 excerpt..."

    def test_code_inference_transcript_open(self):
        p = self._make_progress()
        it = _make_iteration(
            code='with open(f"{transcript_dir}/{slug}/transcript.md") as f:',
        )
        assert p._extract_status(it) == "Reading transcripts..."

    def test_code_inference_llm_query_batched(self):
        p = self._make_progress()
        it = _make_iteration(code="results = llm_query_batched(prompts)")
        assert p._extract_status(it) == "Analyzing excerpts in parallel..."

    def test_code_inference_llm_query(self):
        p = self._make_progress()
        it = _make_iteration(code="answer = llm_query(prompt)")
        assert p._extract_status(it) == "Analyzing with AI..."

    def test_code_inference_regex_search(self):
        p = self._make_progress()
        it = _make_iteration(code="matches = re.search(r'pattern', text)")
        assert p._extract_status(it) == "Searching transcript text..."

    def test_code_inference_catalog_scan(self):
        p = self._make_progress()
        it = _make_iteration(code='episodes = context["catalog"]')
        assert p._extract_status(it) == "Scanning episode catalog..."

    def test_fallback_thinking(self):
        """Empty iteration with no code blocks falls back to 'Thinking...'"""
        p = self._make_progress()
        it = RLMIteration(prompt="test", response="hmm", code_blocks=[])
        assert p._extract_status(it) == "Thinking..."


# ---------------------------------------------------------------------------
# _best_stdout_line tests
# ---------------------------------------------------------------------------

class TestBestStdoutLine:

    def _make_progress(self) -> ProgressDisplay:
        return ProgressDisplay(MagicMock())

    def test_empty_stdout(self):
        p = self._make_progress()
        it = _make_iteration(code="x = 1", stdout="")
        assert p._best_stdout_line(it) is None

    def test_no_code_blocks(self):
        p = self._make_progress()
        it = RLMIteration(prompt="test", response="", code_blocks=[])
        assert p._best_stdout_line(it) is None

    def test_filters_short_lines(self):
        p = self._make_progress()
        it = _make_iteration(code="x", stdout="hi\nok\nA longer useful line here")
        assert p._best_stdout_line(it) == "A longer useful line here"

    def test_filters_long_lines(self):
        p = self._make_progress()
        long_line = "x" * 121
        it = _make_iteration(code="x", stdout=f"{long_line}\nNormal line here")
        assert p._best_stdout_line(it) == "Normal line here"

    def test_filters_json_dumps(self):
        p = self._make_progress()
        it = _make_iteration(code="x", stdout='{"key": "value"}\nActual status')
        assert p._best_stdout_line(it) == "Actual status"

    def test_filters_list_dumps(self):
        p = self._make_progress()
        it = _make_iteration(code="x", stdout="[1, 2, 3, 4]\nReal output here")
        assert p._best_stdout_line(it) == "Real output here"

    def test_filters_tracebacks(self):
        p = self._make_progress()
        it = _make_iteration(
            code="x",
            stdout='Traceback (most recent call last):\n  File "foo.py"\nGood line',
        )
        assert p._best_stdout_line(it) == "Good line"

    def test_filters_repr_output(self):
        p = self._make_progress()
        it = _make_iteration(code="x", stdout="<module 'os'>\nUseful output")
        assert p._best_stdout_line(it) == "Useful output"

    def test_filters_horizontal_rules(self):
        p = self._make_progress()
        it = _make_iteration(code="x", stdout="----------\nStatus update")
        assert p._best_stdout_line(it) == "Status update"


# ---------------------------------------------------------------------------
# log() lifecycle — completed steps, dedup, capping
# ---------------------------------------------------------------------------

class TestLogLifecycle:

    def _make_progress(self) -> ProgressDisplay:
        p = ProgressDisplay(MagicMock(), initial_status="Starting...")
        p._start_time = time.time()
        return p

    def test_first_log_moves_initial_to_completed(self):
        p = self._make_progress()
        it = _make_iteration(code="x", stdout="Found 5 episodes")
        p.log(it)

        assert p._iteration_count == 1
        assert p._completed_steps == ["Starting"]
        assert p._current_status == "Found 5 episodes"

    def test_duplicate_status_not_pushed(self):
        """If new status == current, don't push a completed step."""
        p = self._make_progress()
        p._current_status = "Reading transcripts..."

        it = _make_iteration(
            code='open(f"{transcript_dir}/slug/transcript.md")',
        )
        # This will infer "Reading transcripts..." — same as current
        p.log(it)
        assert "Reading transcripts" not in p._completed_steps

    def test_completed_steps_cap(self):
        """Internal list is capped to avoid unbounded growth."""
        p = self._make_progress()
        # Push many unique statuses
        for i in range(20):
            it = _make_iteration(
                code="x",
                stdout=f"Status message number {i:02d} here",
            )
            p.log(it)

        # Internal list should be capped (not 20 items)
        assert len(p._completed_steps) <= 8  # _MAX_COMPLETED * 2

    def test_iteration_count_increments(self):
        p = self._make_progress()
        for _ in range(5):
            p.log(_make_iteration(code="x", stdout="Something useful"))
        assert p._iteration_count == 5

    def test_log_metadata_is_noop(self):
        """log_metadata should not raise or change state."""
        p = self._make_progress()
        metadata = MagicMock(spec=RLMMetadata)
        p.log_metadata(metadata)
        assert p._iteration_count == 0


# ---------------------------------------------------------------------------
# Context manager lifecycle
# ---------------------------------------------------------------------------

class TestContextManager:

    def test_enter_resets_state(self):
        console = MagicMock()
        p = ProgressDisplay(console, initial_status="Test...")
        p._iteration_count = 5
        p._completed_steps = ["old step"]

        with patch.object(Live, "__enter__", return_value=MagicMock()):
            with patch.object(Live, "__exit__", return_value=None):
                with p:
                    assert p._iteration_count == 0
                    assert p._completed_steps == []
                    assert p._start_time is not None
                    assert p._live is not None

    def test_exit_clears_live(self):
        console = MagicMock()
        p = ProgressDisplay(console)

        with patch.object(Live, "__enter__", return_value=MagicMock()):
            with patch.object(Live, "__exit__", return_value=None):
                with p:
                    pass

        assert p._live is None


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestTruncate:

    def test_short_string_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        assert _truncate("hello", 5) == "hello"

    def test_long_string_truncated(self):
        result = _truncate("hello world", 8)
        assert len(result) == 8
        assert result.endswith("\u2026")
        assert result == "hello w\u2026"


class TestInferFromCode:

    def test_transcript_open(self):
        assert _infer_from_code('open(f"{td}/transcript.md")') == "Reading transcripts..."

    def test_llm_query_batched(self):
        assert _infer_from_code("llm_query_batched(prompts)") == "Analyzing excerpts in parallel..."

    def test_llm_query(self):
        assert _infer_from_code("answer = llm_query(p)") == "Analyzing with AI..."

    def test_regex_search(self):
        assert _infer_from_code("re.search(r'pat', text)") == "Searching transcript text..."

    def test_regex_findall(self):
        assert _infer_from_code("re.findall(r'pat', text)") == "Searching transcript text..."

    def test_catalog_access(self):
        assert _infer_from_code('eps = context["catalog"]') == "Scanning episode catalog..."

    def test_unknown_code_returns_none(self):
        assert _infer_from_code("x = 1 + 2") is None


# ---------------------------------------------------------------------------
# Elapsed time formatting
# ---------------------------------------------------------------------------

class TestElapsed:

    def test_no_start_time(self):
        p = ProgressDisplay(MagicMock())
        assert p._elapsed() == "0:00"

    def test_seconds_only(self):
        p = ProgressDisplay(MagicMock())
        p._start_time = time.time() - 45
        elapsed = p._elapsed()
        assert elapsed == "0:45"

    def test_minutes_and_seconds(self):
        p = ProgressDisplay(MagicMock())
        p._start_time = time.time() - 155
        elapsed = p._elapsed()
        assert elapsed == "2:35"
