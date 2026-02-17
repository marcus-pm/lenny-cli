"""Router regression tests for hybrid deterministic + LLM-judge routing."""

from __future__ import annotations

from types import SimpleNamespace

from lenny.router import QueryMode, classify_query


class _FakeMessages:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def create(self, **kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(text=self._response_text)],
            usage=SimpleNamespace(input_tokens=12, output_tokens=1),
        )


class _FakeClient:
    def __init__(self, response_text: str):
        self.messages = _FakeMessages(response_text)


# ---------------------------------------------------------------------------
# Deterministic guardrails
# ---------------------------------------------------------------------------

def test_deterministic_research_compare():
    decision = classify_query("compare Ravi and Julie on PM hiring", [])
    assert decision.mode == QueryMode.RESEARCH


def test_deterministic_fast_specific_lookup():
    decision = classify_query("What did Brian Chesky say about founder mode?", [])
    assert decision.mode == QueryMode.FAST


def test_followup_after_research_stays_research():
    history = [{"question": "Q1", "answer": "A1", "mode": "research"}]
    decision = classify_query("can you go deeper?", history)
    assert decision.mode == QueryMode.RESEARCH


def test_followup_with_legacy_rlm_history():
    """History from before rename uses 'rlm' — follow-up should still work."""
    history = [{"question": "Q1", "answer": "A1", "mode": "rlm"}]
    decision = classify_query("tell me more about that", history)
    assert decision.mode == QueryMode.RESEARCH


# ---------------------------------------------------------------------------
# New routing patterns
# ---------------------------------------------------------------------------

def test_deterministic_research_guide():
    decision = classify_query("create a guide for building 0 to 1 products", [])
    assert decision.mode == QueryMode.RESEARCH


def test_deterministic_research_strategy():
    decision = classify_query("what's the best strategy for scaling a team?", [])
    assert decision.mode == QueryMode.RESEARCH


def test_deterministic_research_playbook():
    decision = classify_query("build me a playbook for user onboarding", [])
    assert decision.mode == QueryMode.RESEARCH


# ---------------------------------------------------------------------------
# LLM judge (Tier 2)
# ---------------------------------------------------------------------------

def test_ambiguous_without_client_defaults_research():
    decision = classify_query("product strategy lessons from this year", [])
    assert decision.mode == QueryMode.RESEARCH
    assert "default research" in decision.reason


def test_ambiguous_with_llm_judge_fast():
    client = _FakeClient("FAST")
    decision = classify_query("brian chesky founder mode details", [], client=client)
    assert decision.mode == QueryMode.FAST
    assert "llm-judge" in decision.reason


def test_ambiguous_with_unparseable_llm_judge_defaults_research():
    client = _FakeClient("UNSURE")
    decision = classify_query("brian chesky founder mode details", [], client=client)
    assert decision.mode == QueryMode.RESEARCH
    assert "default research" in decision.reason


def test_judge_noisy_response_still_parses():
    """Judge returns 'RESEARCH.' with trailing punctuation — should still parse."""
    client = _FakeClient("RESEARCH.")
    decision = classify_query("brian chesky founder mode details", [], client=client)
    assert decision.mode == QueryMode.RESEARCH
    assert "llm-judge" in decision.reason
