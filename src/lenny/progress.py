"""Live progress display for RLM deep analysis queries.

Shows an animated spinner with human-readable status text that updates
as the RLM works through its iterations.  Completed steps accumulate
above the spinner so the user can see what has been done.

Implements the RLM logger duck-type interface (log, log_metadata,
iteration_count) so it can be assigned to ``RLM.logger``.
"""

from __future__ import annotations

import re
import time

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from rlm.core.types import RLMIteration, RLMMetadata


# Lines to skip when extracting status from REPL stdout
_NOISE_PATTERNS = [
    re.compile(r"^\s*$"),              # blank lines
    re.compile(r"^Traceback", re.I),   # error traces
    re.compile(r"^\s*File \""),        # traceback file lines
    re.compile(r"^\{"),                # JSON / dict dumps
    re.compile(r"^\["),                # list dumps
    re.compile(r"^<"),                 # repr output
    re.compile(r"^-+$"),              # horizontal rules
]

# How many completed steps to keep visible
_MAX_COMPLETED = 4


class ProgressDisplay:
    """RLM logger that drives a Rich Live spinner with updating status.

    Usage::

        progress = ProgressDisplay(console, initial_status="Searching 303 episodes...")
        engine.rlm.logger = progress
        try:
            with progress:
                answer, cost = engine.query(query)
        finally:
            engine.rlm.logger = None
    """

    def __init__(
        self,
        console: Console,
        *,
        initial_status: str = "Thinking...",
    ) -> None:
        self.console = console
        self._iteration_count = 0
        self._start_time: float | None = None
        self._completed_steps: list[str] = []
        self._current_status: str = initial_status
        self._live: Live | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ProgressDisplay:
        self._start_time = time.time()
        self._iteration_count = 0
        self._completed_steps = []
        self._live = Live(
            self._build_renderable(),
            console=self.console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live is not None:
            self._live.__exit__(*args)
            self._live = None

    # ------------------------------------------------------------------
    # RLMLogger interface (duck-typed)
    # ------------------------------------------------------------------

    def log_metadata(self, metadata: RLMMetadata) -> None:  # noqa: ARG002
        """Called once at start. No-op."""

    def log(self, iteration: RLMIteration) -> None:
        """Called after each RLM iteration — update the status display."""
        self._iteration_count += 1

        new_status = self._extract_status(iteration)

        # Move current status to completed (avoid duplicates)
        if self._current_status and self._current_status != new_status:
            # Strip trailing "..." for the completed form
            done_text = self._current_status.rstrip(".")
            self._completed_steps.append(done_text)
            # Cap internal list to avoid unbounded growth
            if len(self._completed_steps) > _MAX_COMPLETED * 2:
                self._completed_steps = self._completed_steps[-_MAX_COMPLETED:]

        self._current_status = new_status

        if self._live is not None:
            self._live.update(self._build_renderable())

    @property
    def iteration_count(self) -> int:
        return self._iteration_count

    # ------------------------------------------------------------------
    # Status extraction
    # ------------------------------------------------------------------

    def _extract_status(self, iteration: RLMIteration) -> str:
        """Derive a single human-readable status line from the iteration."""
        # Check if this is the final answer iteration
        if iteration.final_answer is not None:
            return "Preparing answer..."

        # Check for FINAL() / FINAL_VAR() call in the LLM response
        if iteration.response and re.search(
            r"^\s*FINAL(?:_VAR)?\(", iteration.response, re.MULTILINE,
        ):
            return "Preparing answer..."

        # Try to get a meaningful line from stdout
        stdout_line = self._best_stdout_line(iteration)
        if stdout_line:
            return _truncate(stdout_line, 70)

        # Infer from sub-LLM calls
        for cb in iteration.code_blocks:
            if cb.result.rlm_calls:
                n = len(cb.result.rlm_calls)
                return f"Analyzing {n} excerpt{'s' if n != 1 else ''}..."

        # Infer from code patterns
        for cb in iteration.code_blocks:
            hint = _infer_from_code(cb.code)
            if hint:
                return hint

        # Fallback
        return "Thinking..."

    def _best_stdout_line(self, iteration: RLMIteration) -> str | None:
        """Return the most informative stdout line from the iteration."""
        candidates: list[str] = []
        for cb in iteration.code_blocks:
            if not cb.result.stdout:
                continue
            for line in cb.result.stdout.strip().split("\n"):
                clean = line.strip()
                if len(clean) < 5 or len(clean) > 120:
                    continue
                if any(p.match(clean) for p in _NOISE_PATTERNS):
                    continue
                candidates.append(clean)
        # Take the last meaningful line (most recent output)
        return candidates[-1] if candidates else None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _build_renderable(self) -> RenderableType:
        """Build the multi-line display: completed steps + animated spinner."""
        parts: list[RenderableType] = []

        # Completed steps (dim, with checkmark)
        for step in self._completed_steps[-_MAX_COMPLETED:]:
            parts.append(Text(f"  \u2713 {step}", style="dim"))

        # Current step: spinner + status text + elapsed time
        status_text = Text()
        status_text.append(self._current_status, style="bold")
        status_text.append(f"  {self._elapsed()}", style="dim")

        spinner = Spinner("dots", text=status_text, style="cyan")
        # Wrap with indent
        parts.append(Text("  ", end=""))
        parts.append(spinner)

        return Group(*parts)

    def _elapsed(self) -> str:
        """Format elapsed time as M:SS."""
        if self._start_time is None:
            return "0:00"
        secs = int(time.time() - self._start_time)
        return f"{secs // 60}:{secs % 60:02d}"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"


def _infer_from_code(code: str) -> str | None:
    """Infer a human-readable status from REPL code when stdout is empty."""
    low = code.lower()
    if "open(" in low and "transcript" in low:
        return "Reading transcripts..."
    if "llm_query_batched" in low:
        return "Analyzing excerpts in parallel..."
    if "llm_query" in low:
        return "Analyzing with AI..."
    if "re.search" in low or "re.findall" in low:
        return "Searching transcript text..."
    if "context[" in low and "catalog" in low:
        return "Scanning episode catalog..."
    return None
