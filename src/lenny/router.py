"""Query classifier for routing between RAG and RLM paths.

Three-tier classification:
1. Deterministic guardrails — hard rules for obvious cases (zero latency)
2. LLM judge — cheap Haiku call for ambiguous queries (~1-2s, ~$0.001)
3. Conservative fallback — default to RLM when uncertain
"""

from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)


class QueryMode(Enum):
    """Routing mode for a query."""
    RAG = "rag"
    RLM = "rlm"
    AUTO = "auto"


@dataclass
class RouteDecision:
    """The result of classifying a query."""
    mode: QueryMode
    reason: str


# ---------------------------------------------------------------------------
# Tier 1: Deterministic guardrails
# ---------------------------------------------------------------------------

# Hard RLM — any match forces RLM immediately
_RLM_PATTERNS: list[tuple[str, str]] = [
    # Cross-episode synthesis
    (r"\bacross\s+episodes?\b", "cross-episode synthesis"),
    (r"\bcompare\b", "comparison analysis"),
    (r"\bwhat\s+do\s+(guests?|people|experts?|leaders?)\s+(think|say|believe|recommend)\b", "cross-episode synthesis"),
    (r"\b(guests?|people|experts?|leaders?)\s+(recommend|suggest|advise)\b", "cross-episode synthesis"),
    (r"\bthemes?\b", "thematic analysis"),
    (r"\bpatterns?\b", "pattern analysis"),
    (r"\bdisagree\b", "cross-episode comparison"),
    (r"\bconsensus\b", "consensus analysis"),
    (r"\bcommon(ly)?\b.*\b(advice|theme|pattern|thread|view|opinion)\b", "cross-episode synthesis"),

    # Quantitative / exhaustive
    (r"\bhow\s+many\b", "quantitative analysis"),
    (r"\ball\s+episodes?\b", "exhaustive search"),
    (r"\bevery\s+guest\b", "exhaustive search"),
    (r"\blist\s+all\b", "exhaustive listing"),
    (r"\brank\b.*\b(guests?|episodes?|topics?)\b", "ranking analysis"),
    (r"\bmost\s+common\b", "frequency analysis"),
    (r"\bmost\s+popular\b", "frequency analysis"),

    # Complex analysis
    (r"\bframework\b", "framework analysis"),
    (r"\bsummarize\s+all\b", "exhaustive summary"),
    (r"\btrend\b", "trend analysis"),
    (r"\bevolution\s+of\b", "evolution analysis"),
    (r"\bover\s+time\b", "temporal analysis"),
    (r"\bchanged?\s+(over|through|across)\b", "temporal analysis"),
]

# Hard RAG — specific targeted lookups
_RAG_PATTERNS: list[tuple[str, str]] = [
    (r"\bwhat\s+did\s+\w+(\s+\w+)?\s+(say|mention|talk|discuss|recommend)\b", "specific guest lookup"),
    (r"\bwhich\s+episode\b", "episode lookup"),
    (r"\bfind\s+(the\s+)?quote\b", "quote lookup"),
    (r"\bwho\s+said\b", "quote attribution"),
    (r"\bwhen\s+(was|did)\b", "factual lookup"),
    (r"\bwhat\s+is\b", "definition lookup"),
    (r"\bdefine\b", "definition lookup"),
    (r"\btell\s+me\s+about\s+the\s+episode\b", "episode lookup"),
]

# Multi-entity / plural subjects → forces RLM even when RAG surface patterns match
_MULTI_ENTITY_PATTERNS: list[tuple[str, str]] = [
    (r"\bguests\b", "multi-guest synthesis"),
    (r"\bexperts\b", "multi-expert synthesis"),
    (r"\bleaders\b", "multi-leader synthesis"),
    (r"\bpeople\b", "multi-person synthesis"),
    (r"\b\w+\s+and\s+\w+\s+on\b", "multi-entity comparison"),
    (r"\bboth\b", "comparison analysis"),
    (r"\bdifferent\s+(guests?|people|views?|opinions?|perspectives?)\b", "cross-episode synthesis"),
]


# ---------------------------------------------------------------------------
# Tier 2: LLM judge
# ---------------------------------------------------------------------------
_JUDGE_SYSTEM_PROMPT = """\
You are a query routing classifier for a podcast transcript search application.

The application has two query paths:
- RAG: Fast keyword search + single LLM synthesis. Best for: looking up what a specific person said, finding a particular quote, checking a fact from one episode, simple definitions.
- RLM: Deep multi-step analysis where an LLM writes code to search across all 303 transcripts, extract patterns, and synthesize findings. Best for: comparing perspectives across guests, identifying themes, analyzing trends, any question requiring information from many episodes.

Given a user query, respond with exactly one word: RAG or RLM.

If unsure, say RLM (it's slower but more thorough — better to be safe)."""

_JUDGE_MODEL = "claude-haiku-4-5-20251001"


def _llm_judge(
    query: str,
    client: anthropic.Anthropic,
) -> RouteDecision | None:
    """Ask Haiku to classify an ambiguous query. Returns None on error."""
    try:
        start = time.perf_counter()
        response = client.messages.create(
            model=_JUDGE_MODEL,
            max_tokens=8,
            system=_JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        answer = response.content[0].text.strip().upper()
        elapsed = time.perf_counter() - start
        logger.debug(
            "LLM judge: %r → %s (%.1fs, %d in/%d out tokens)",
            query, answer, elapsed,
            response.usage.input_tokens, response.usage.output_tokens,
        )

        if answer.startswith("RAG"):
            return RouteDecision(QueryMode.RAG, "llm-judge → rag")
        elif answer.startswith("RLM"):
            return RouteDecision(QueryMode.RLM, "llm-judge → rlm")
        else:
            # Unparseable response — fall through to conservative default
            logger.debug("LLM judge returned unparseable: %r", answer)
            return None
    except Exception as e:
        # Network error, rate limit, etc. — non-fatal, fall through
        logger.debug("LLM judge error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------
def classify_query(
    query: str,
    conversation_history: list[dict] | None = None,
    client: anthropic.Anthropic | None = None,
) -> RouteDecision:
    """Classify a query as RAG or RLM using a three-tier approach.

    Tier 1 — Deterministic guardrails (evaluated in order):
      1a. Follow-up to RLM → RLM
      1b. Multi-entity / plural-subject → RLM
      1c. RAG surface patterns (specific lookups) → RAG
      1d. RLM surface patterns (synthesis/analysis) → RLM

    Tier 2 — LLM judge (only if no guardrail matched AND client provided):
      Cheap Haiku call (~1-2s, ~$0.001) to classify ambiguous queries.

    Tier 3 — Conservative fallback:
      Default to RLM (prefer thorough over fast).
    """
    query_lower = query.lower().strip()
    history = conversation_history or []

    # ── Tier 1: Deterministic guardrails ──────────────────────────────

    # 1a. Follow-up detection
    if history and _is_followup(query_lower):
        last_mode = history[-1].get("mode", "rlm")
        if last_mode == "rlm":
            return RouteDecision(QueryMode.RLM, "follow-up to deep analysis")

    # 1b. Multi-entity / plural-subject → RLM
    for pattern, reason in _MULTI_ENTITY_PATTERNS:
        if re.search(pattern, query_lower):
            return RouteDecision(QueryMode.RLM, reason)

    # 1c. RAG signals (specific lookups)
    for pattern, reason in _RAG_PATTERNS:
        if re.search(pattern, query_lower):
            return RouteDecision(QueryMode.RAG, reason)

    # 1d. RLM signals (synthesis/analysis)
    for pattern, reason in _RLM_PATTERNS:
        if re.search(pattern, query_lower):
            return RouteDecision(QueryMode.RLM, reason)

    # ── Tier 2: LLM judge for ambiguous queries ──────────────────────

    if client is not None:
        decision = _llm_judge(query, client)
        if decision is not None:
            return decision

    # ── Tier 3: Conservative fallback ─────────────────────────────────

    return RouteDecision(QueryMode.RLM, "ambiguous → default rlm")


def _is_followup(query_lower: str) -> bool:
    """Detect if a query is likely a follow-up to the previous answer."""
    followup_signals = [
        r"^(and|but|also|what about|how about|tell me more|expand|elaborate|narrow|can you)",
        r"^(more on|go deeper|dig into|specifically|in particular)",
        r"\b(you (just |)mentioned|from (that|those|the previous|your))\b",
        r"^(that|those|these|the same)\b",
    ]
    return any(re.search(p, query_lower) for p in followup_signals)
