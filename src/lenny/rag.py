"""RAG engine — fast retrieve-then-generate path using BM25 + single Haiku call."""

from __future__ import annotations

import time
import textwrap
from collections import defaultdict
from typing import TYPE_CHECKING

import anthropic

from lenny.costs import QueryCost, make_query_cost_from_usage

if TYPE_CHECKING:
    from lenny.search import TranscriptSearchIndex

# Model used for RAG synthesis
RAG_MODEL = "claude-haiku-4-5-20251001"

# BM25 score threshold — below this, excerpts are too weak to synthesize
_RELEVANCE_THRESHOLD = 5.0

# Max chunks per episode to avoid one episode dominating the context
_MAX_CHUNKS_PER_EPISODE = 3

# Max total chunks to send to Haiku
_MAX_TOTAL_CHUNKS = 15

INSUFFICIENT_EVIDENCE = (
    "I couldn't find enough relevant information in the podcast transcripts "
    "to answer this question confidently. Try rephrasing your question, or "
    "use `/mode research` for a deeper search that examines transcripts more thoroughly."
)

RAG_SYSTEM_PROMPT = textwrap.dedent("""\
You are a research assistant specialized in Lenny's Podcast transcripts. \
Answer the user's question using ONLY the provided transcript excerpts.

## Rules
- Base your answer solely on the excerpts below. Do not invent or assume information.
- If the excerpts don't contain enough information to answer confidently, say so clearly.
- Cite every episode you reference: **Guest Name** in *Episode Title* ([link](youtube_url))
- When quoting, attribute the quote to the speaker.
- Structure your answer with markdown headers/bullets as appropriate.
- Keep your answer concise but thorough — aim for 2-4 paragraphs.
""")


class RAGEngine:
    """Fast query path: BM25 retrieval + single Haiku synthesis call."""

    def __init__(
        self,
        search_index: TranscriptSearchIndex,
        api_key: str,
        model: str = RAG_MODEL,
    ):
        self.search_index = search_index
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    def query(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> tuple[str, QueryCost]:
        """Run a RAG query: BM25 search → deduplicate → Haiku synthesis.

        Returns (answer_text, query_cost).
        """
        start_time = time.perf_counter()
        history = conversation_history or []

        # 1. Retrieve with scores
        results = self.search_index.search_with_scores(question, top_k=30)

        # 2. Relevance gate — if best score is too low, bail out
        if not results or results[0][1] < _RELEVANCE_THRESHOLD:
            elapsed = time.perf_counter() - start_time
            cost = make_query_cost_from_usage(
                model=self.model,
                input_tokens=0,
                output_tokens=0,
                execution_time=elapsed,
            )
            return INSUFFICIENT_EVIDENCE, cost

        # 3. Deduplicate: keep top N chunks per episode
        per_episode: dict[str, list[tuple]] = defaultdict(list)
        for chunk, score in results:
            if len(per_episode[chunk.episode_slug]) < _MAX_CHUNKS_PER_EPISODE:
                per_episode[chunk.episode_slug].append((chunk, score))

        # Flatten and sort by score, then cap total
        all_kept = []
        for ep_chunks in per_episode.values():
            all_kept.extend(ep_chunks)
        all_kept.sort(key=lambda x: x[1], reverse=True)
        all_kept = all_kept[:_MAX_TOTAL_CHUNKS]

        # 4. Build the user prompt
        excerpts_text = self._format_excerpts(all_kept)
        history_text = self._format_history(history)

        user_message = f"""{excerpts_text}

{history_text}

## Question
{question}"""

        # 5. Call Haiku
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text
        elapsed = time.perf_counter() - start_time

        # 6. Build cost from usage
        cost = make_query_cost_from_usage(
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            execution_time=elapsed,
        )

        return answer, cost

    # ------------------------------------------------------------------
    # Prompt formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_excerpts(chunks_with_scores: list[tuple]) -> str:
        """Format retrieved chunks for the Haiku prompt."""
        lines = ["## Transcript Excerpts\n"]
        for i, (chunk, score) in enumerate(chunks_with_scores, 1):
            lines.append(
                f"### Excerpt {i}: {chunk.guest} — *{chunk.title}*\n"
                f"**Episode:** [{chunk.title}]({chunk.youtube_url})\n\n"
                f"{chunk.text}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        """Format recent RAG-mode conversation history for context."""
        # Only include RAG-mode entries to avoid confusing Haiku with
        # RLM's more detailed answers
        rag_history = [h for h in history if h.get("mode") in ("fast", "rag")]
        recent = rag_history[-5:]
        if not recent:
            return ""

        lines = ["## Conversation History\n"]
        for entry in recent:
            lines.append(f"**Q:** {entry['question']}")
            lines.append(f"**A:** {entry['answer'][:500]}\n")
        return "\n".join(lines)
