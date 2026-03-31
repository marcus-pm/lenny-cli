"""RAG engine — fast retrieve-then-generate path using MCP search + single Haiku call."""

from __future__ import annotations

import logging
import time
import textwrap
from collections import defaultdict
from typing import TYPE_CHECKING

import anthropic

from lenny.costs import QueryCost, make_query_cost_from_usage

if TYPE_CHECKING:
    from lenny.mcp_client import MCPClient

logger = logging.getLogger(__name__)

# Model used for RAG synthesis
RAG_MODEL = "claude-haiku-4-5-20251001"

# Max snippets per episode to avoid one episode dominating the context
_MAX_SNIPPETS_PER_EPISODE = 3

# Max total snippets to send to Haiku
_MAX_TOTAL_SNIPPETS = 15

INSUFFICIENT_EVIDENCE = (
    "I couldn't find enough relevant information in the podcast transcripts "
    "to answer this question confidently. Try rephrasing your question, or "
    "use `/mode research` for a deeper search that examines transcripts more thoroughly."
)

SEARCH_FAILED = (
    "The transcript search encountered an error. Try again, or use "
    "`/mode research` for a deeper analysis."
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
    """Fast query path: MCP search + single Haiku synthesis call."""

    def __init__(
        self,
        api_key: str,
        mcp_client: MCPClient | None = None,
        model: str = RAG_MODEL,
    ):
        self.mcp_client = mcp_client
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=3)

    def query(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> tuple[str, QueryCost]:
        """Run a RAG query: search → deduplicate → Haiku synthesis.

        Returns (answer_text, query_cost).
        """
        start_time = time.perf_counter()
        history = conversation_history or []

        if self.mcp_client is not None:
            excerpts_text = self._search_via_mcp(question)
        else:
            excerpts_text = None

        # Relevance gate: None = search errored, "" = search succeeded but empty
        if excerpts_text is None or excerpts_text == "":
            elapsed = time.perf_counter() - start_time
            cost = make_query_cost_from_usage(
                model=self.model,
                input_tokens=0,
                output_tokens=0,
                execution_time=elapsed,
                calls=0,
            )
            message = SEARCH_FAILED if excerpts_text is None else INSUFFICIENT_EVIDENCE
            return message, cost

        # Build the user prompt
        history_text = self._format_history(history)
        user_message = f"""{excerpts_text}

{history_text}

## Question
{question}"""

        # Call Haiku
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text
        elapsed = time.perf_counter() - start_time

        cost = make_query_cost_from_usage(
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            execution_time=elapsed,
        )

        return answer, cost

    # ------------------------------------------------------------------
    # MCP search path
    # ------------------------------------------------------------------
    def _search_via_mcp(self, question: str) -> str | None:
        """Search using the MCP server's search_content tool."""
        try:
            results = self.mcp_client.search_content(
                query=question, content_type="podcast", limit=20,
            )
        except Exception:
            logger.warning("MCP search failed, retrying with fresh session", exc_info=True)
            try:
                # Session may have expired — force re-init and retry once
                self.mcp_client._session_id = None
                results = self.mcp_client.search_content(
                    query=question, content_type="podcast", limit=20,
                )
            except Exception as e:
                logger.warning("MCP search retry failed: %s", e)
                return None

        entries = results.get("results", [])
        if not entries:
            logger.info("MCP search returned 0 results for query: %s", question)
            return ""  # empty string = search ran but found nothing

        # Flatten snippets from all results, with episode metadata
        all_snippets: list[dict] = []
        for entry in entries:
            # search_content doesn't always include 'guest' — extract from filename
            guest = entry.get("guest", "")
            if not guest:
                # Derive from filename: "podcasts/anneka-gupta.md" → "Anneka Gupta"
                fn = entry.get("filename", "")
                slug = fn.removeprefix("podcasts/").removesuffix(".md")
                guest = slug.replace("-", " ").title() if slug else ""
            title = entry.get("title", "")
            filename = entry.get("filename", "")
            for snippet_obj in entry.get("snippets", []):
                all_snippets.append({
                    "guest": guest,
                    "title": title,
                    "filename": filename,
                    "text": snippet_obj.get("text", ""),
                    "match_count": snippet_obj.get("match_count", 0),
                })
            # If no snippets list, use the top-level snippet
            if not entry.get("snippets") and entry.get("snippet"):
                all_snippets.append({
                    "guest": guest,
                    "title": title,
                    "filename": filename,
                    "text": entry["snippet"],
                    "match_count": entry.get("match_count", 0),
                })

        if not all_snippets:
            return None

        # Deduplicate: max N snippets per episode
        per_episode: dict[str, list[dict]] = defaultdict(list)
        for snip in all_snippets:
            key = snip["filename"]
            if len(per_episode[key]) < _MAX_SNIPPETS_PER_EPISODE:
                per_episode[key].append(snip)

        # Flatten, sort by match_count, cap total
        all_kept = []
        for ep_snips in per_episode.values():
            all_kept.extend(ep_snips)
        all_kept.sort(key=lambda x: x.get("match_count", 0), reverse=True)
        all_kept = all_kept[:_MAX_TOTAL_SNIPPETS]

        return self._format_mcp_excerpts(all_kept)

    @staticmethod
    def _format_mcp_excerpts(snippets: list[dict]) -> str:
        """Format MCP search snippets for the Haiku prompt."""
        lines = ["## Transcript Excerpts\n"]
        for i, snip in enumerate(snippets, 1):
            lines.append(
                f"### Excerpt {i}: {snip['guest']} — *{snip['title']}*\n\n"
                f"{snip['text']}\n"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_history(history: list[dict]) -> str:
        """Format recent RAG-mode conversation history for context."""
        rag_history = [h for h in history if h.get("mode") in ("fast", "rag")]
        recent = rag_history[-5:]
        if not recent:
            return ""

        lines = ["## Conversation History\n"]
        for entry in recent:
            lines.append(f"**Q:** {entry['question']}")
            lines.append(f"**A:** {entry['answer'][:500]}\n")
        return "\n".join(lines)
