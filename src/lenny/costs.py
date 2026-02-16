"""Token usage tracking and cost estimation for Anthropic models."""

from __future__ import annotations

from dataclasses import dataclass, field

# Anthropic pricing per million tokens (as of Feb 2026)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Opus 4.6
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    # Opus 4
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    # Sonnet 4
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    # Haiku 4.5
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
}

# Fallback pricing for unknown models
DEFAULT_PRICING = {"input": 3.0, "output": 15.0}


@dataclass
class QueryCost:
    """Cost breakdown for a single query."""
    model_costs: dict[str, dict] = field(default_factory=dict)
    total_cost: float = 0.0
    execution_time: float = 0.0


@dataclass
class SessionCosts:
    """Cumulative cost tracking across a session."""
    queries: list[QueryCost] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0

    def add_query(self, usage_summary, execution_time: float) -> QueryCost:
        """Record usage from an RLMChatCompletion result."""
        query = QueryCost(execution_time=execution_time)

        for model_name, model_usage in usage_summary.model_usage_summaries.items():
            pricing = MODEL_PRICING.get(model_name, DEFAULT_PRICING)
            input_cost = (model_usage.total_input_tokens / 1_000_000) * pricing["input"]
            output_cost = (model_usage.total_output_tokens / 1_000_000) * pricing["output"]

            query.model_costs[model_name] = {
                "calls": model_usage.total_calls,
                "input_tokens": model_usage.total_input_tokens,
                "output_tokens": model_usage.total_output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost,
            }
            query.total_cost += input_cost + output_cost

            self.total_input_tokens += model_usage.total_input_tokens
            self.total_output_tokens += model_usage.total_output_tokens

        self.total_cost += query.total_cost
        self.queries.append(query)
        return query

    def add_raw_query_cost(self, query_cost: QueryCost) -> None:
        """Register a pre-built QueryCost into session totals.

        Used by the RAG path which builds QueryCost directly from
        Anthropic API usage rather than going through RLM's UsageSummary.
        """
        for model_name, costs in query_cost.model_costs.items():
            self.total_input_tokens += costs["input_tokens"]
            self.total_output_tokens += costs["output_tokens"]
        self.total_cost += query_cost.total_cost
        self.queries.append(query_cost)


def make_query_cost_from_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    execution_time: float,
) -> QueryCost:
    """Create a QueryCost from raw Anthropic API usage (for RAG path).

    Unlike ``SessionCosts.add_query()`` which takes an RLM UsageSummary,
    this accepts raw token counts from a direct Anthropic API call.
    """
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return QueryCost(
        model_costs={model: {
            "calls": 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }},
        total_cost=input_cost + output_cost,
        execution_time=execution_time,
    )


def format_query_cost(query: QueryCost) -> str:
    """Format a single query's cost for display."""
    lines = []
    for model_name, costs in query.model_costs.items():
        short_name = _short_model_name(model_name)
        lines.append(
            f"  {short_name}: {costs['calls']} calls, "
            f"{costs['input_tokens']:,} in / {costs['output_tokens']:,} out tokens "
            f"(${costs['total_cost']:.4f})"
        )
    lines.append(f"  Query total: ${query.total_cost:.4f} in {query.execution_time:.1f}s")
    return "\n".join(lines)


def format_session_cost(session: SessionCosts) -> str:
    """Format cumulative session costs for display."""
    lines = [
        f"Session: {len(session.queries)} queries",
        f"  Total tokens: {session.total_input_tokens:,} in / {session.total_output_tokens:,} out",
        f"  Total cost: ${session.total_cost:.4f}",
    ]
    return "\n".join(lines)


def _short_model_name(model_name: str) -> str:
    """Shorten model names for display."""
    if "opus" in model_name:
        return "Opus"
    if "sonnet" in model_name:
        return "Sonnet"
    if "haiku" in model_name:
        return "Haiku"
    return model_name
