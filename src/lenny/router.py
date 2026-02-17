"""Query classifier for routing between fast and research paths.

Three-tier classification:
1. Deterministic guardrails — hard rules for obvious cases (zero latency)
2. LLM judge — cheap Haiku call for ambiguous queries (~1-2s, ~$0.001)
3. Conservative fallback — default to research when uncertain
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
    FAST = "fast"
    RESEARCH = "research"
    AUTO = "auto"


@dataclass
class RouteDecision:
    """The result of classifying a query."""
    mode: QueryMode
    reason: str


# ---------------------------------------------------------------------------
# Tier 1: Deterministic guardrails
# ---------------------------------------------------------------------------

# Hard RESEARCH — any match forces research immediately
_RESEARCH_PATTERNS: list[tuple[str, str]] = [
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

    # Strategy / guide queries (intent-phrase patterns)
    (r"\b(create|build|write|design|develop)\s+(a\s+)?(guide|playbook|handbook)\b", "strategy synthesis"),
    (r"\b(guide|playbook|handbook)\s+(for|to|on)\b", "strategy synthesis"),
    (r"\bstrategy\s+(for|to|on)\b", "strategy synthesis"),
    (r"\b(0|zero)\s*[-\u2013]?\s*to\s*[-\u2013]?\s*(1|one)\b", "strategy synthesis"),
    (r"\bfrom\s+scratch\b", "comprehensive analysis"),
    (r"\bend[- ]to[- ]end\b", "comprehensive analysis"),
]

# Hard FAST — specific targeted lookups
_FAST_PATTERNS: list[tuple[str, str]] = [
    (r"\bwhat\s+did\s+\w+(\s+\w+)?\s+(say|mention|talk|discuss|recommend)\b", "specific guest lookup"),
    (r"\bwhich\s+episode\b", "episode lookup"),
    (r"\bfind\s+(the\s+)?quote\b", "quote lookup"),
    (r"\bwho\s+said\b", "quote attribution"),
    (r"\bwhen\s+(was|did)\b", "factual lookup"),
    (r"\bwhat\s+is\b", "definition lookup"),
    (r"\bdefine\b", "definition lookup"),
    (r"\btell\s+me\s+about\s+the\s+episode\b", "episode lookup"),
]

# Multi-entity / plural subjects \u2192 forces research even when fast surface patterns match
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
- FAST: Fast keyword search + single LLM synthesis. Best for: looking up what a specific person said, finding a particular quote, checking a fact from one episode, simple definitions.
- RESEARCH: Deep multi-step analysis where an LLM writes code to search across all transcripts, extract patterns, and synthesize findings. Best for: comparing perspectives across guests, identifying themes, analyzing trends, any question requiring information from many episodes.

Given a user query, respond with exactly one word: FAST or RESEARCH.

If unsure, say RESEARCH (it's slower but more thorough \u2014 better to be safe)."""

_JUDGE_MODEL = "claude-haiku-4-5-20251001"

_JUDGE_TOKEN_RE = re.compile(r"\b(FAST|RESEARCH)\b")


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
            "LLM judge: %r \u2192 %s (%.1fs, %d in/%d out tokens)",
            query, answer, elapsed,
            response.usage.input_tokens, response.usage.output_tokens,
        )

        match = _JUDGE_TOKEN_RE.search(answer)
        if match is None:
            logger.debug("LLM judge returned unparseable: %r", answer)
            return None

        token = match.group(1)
        if token == "FAST":
            return RouteDecision(QueryMode.FAST, "llm-judge \u2192 fast")
        else:
            return RouteDecision(QueryMode.RESEARCH, "llm-judge \u2192 research")
    except Exception as e:
        # Network error, rate limit, etc. \u2014 non-fatal, fall through
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
    """Classify a query as fast or research using a three-tier approach.

    Tier 1 \u2014 Deterministic guardrails (evaluated in order):
      1a. Follow-up to research \u2192 research
      1b. Multi-entity / plural-subject \u2192 research
      1c. Fast surface patterns (specific lookups) \u2192 fast
      1d. Research surface patterns (synthesis/analysis) \u2192 research

    Tier 2 \u2014 LLM judge (only if no guardrail matched AND client provided):
      Cheap Haiku call (~1-2s, ~$0.001) to classify ambiguous queries.

    Tier 3 \u2014 Conservative fallback:
      Default to research (prefer thorough over fast).
    """
    query_lower = query.lower().strip()
    history = conversation_history or []

    # \u2500\u2500 Tier 1: Deterministic guardrails \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    # 1a. Follow-up detection
    if history and _is_followup(query_lower):
        last_mode = history[-1].get("mode", "research")
        if last_mode in ("research", "rlm"):
            return RouteDecision(QueryMode.RESEARCH, "follow-up to deep analysis")

    # 1b. Multi-entity / plural-subject \u2192 research
    for pattern, reason in _MULTI_ENTITY_PATTERNS:
        if re.search(pattern, query_lower):
            return RouteDecision(QueryMode.RESEARCH, reason)

    # 1c. Fast signals (specific lookups)
    for pattern, reason in _FAST_PATTERNS:
        if re.search(pattern, query_lower):
            return RouteDecision(QueryMode.FAST, reason)

    # 1d. Research signals (synthesis/analysis)
    for pattern, reason in _RESEARCH_PATTERNS:
        if re.search(pattern, query_lower):
            return RouteDecision(QueryMode.RESEARCH, reason)

    # \u2500\u2500 Tier 2: LLM judge for ambiguous queries \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    if client is not None:
        decision = _llm_judge(query, client)
        if decision is not None:
            return decision

    # \u2500\u2500 Tier 3: Conservative fallback \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    return RouteDecision(QueryMode.RESEARCH, "ambiguous \u2192 default research")


def _is_followup(query_lower: str) -> bool:
    """Detect if a query is likely a follow-up to the previous answer."""
    followup_signals = [
        r"^(and|but|also|what about|how about|tell me more|expand|elaborate|narrow|can you)",
        r"^(more on|go deeper|dig into|specifically|in particular)",
        r"\b(you (just |)mentioned|from (that|those|the previous|your))\b",
        r"^(that|those|these|the same)\b",
    ]
    return any(re.search(p, query_lower) for p in followup_signals)
