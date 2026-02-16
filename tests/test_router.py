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


def test_deterministic_rlm_compare():
    decision = classify_query("compare Ravi and Julie on PM hiring", [])
    assert decision.mode == QueryMode.RLM


def test_deterministic_rag_specific_lookup():
    decision = classify_query("What did Brian Chesky say about founder mode?", [])
    assert decision.mode == QueryMode.RAG


def test_followup_after_rlm_stays_rlm():
    history = [{"question": "Q1", "answer": "A1", "mode": "rlm"}]
    decision = classify_query("can you go deeper?", history)
    assert decision.mode == QueryMode.RLM


def test_ambiguous_without_client_defaults_rlm():
    decision = classify_query("product strategy lessons from this year", [])
    assert decision.mode == QueryMode.RLM
    assert "default rlm" in decision.reason


def test_ambiguous_with_llm_judge_rag():
    client = _FakeClient("RAG")
    decision = classify_query("brian chesky founder mode details", [], client=client)
    assert decision.mode == QueryMode.RAG
    assert "llm-judge" in decision.reason


def test_ambiguous_with_unparseable_llm_judge_defaults_rlm():
    client = _FakeClient("UNSURE")
    decision = classify_query("brian chesky founder mode details", [], client=client)
    assert decision.mode == QueryMode.RLM
    assert "default rlm" in decision.reason
