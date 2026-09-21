"""Offline tests for the DeepSeek client wrapper."""

import comprehensive_evaluation
from comprehensive_evaluation import DeepSeekLLM


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_payload_includes_seed_when_set():
    llm = DeepSeekLLM(api_key="k", seed=42)
    payload = llm._build_payload(
        [{"role": "user", "content": "hi"}], 0.7, 100
    )
    assert payload["seed"] == 42
    assert payload["model"] == "deepseek-chat"


def test_payload_omits_seed_by_default():
    llm = DeepSeekLLM(api_key="k")
    payload = llm._build_payload([], 0.7, 100)
    assert "seed" not in payload


def test_usage_stats_track_successful_calls(monkeypatch):
    llm = DeepSeekLLM(api_key="k")

    def fake_post(url, json=None, headers=None, timeout=None):
        return FakeResponse(
            {"choices": [{"message": {"content": "ok"}}]}
        )

    monkeypatch.setattr(comprehensive_evaluation.requests, "post", fake_post)

    assert llm([]) == "ok"
    stats = llm.usage_stats()
    assert stats["calls"] == 1
    assert stats["errors"] == 0
    assert stats["latency_seconds"] >= 0.0


def test_usage_stats_count_errors(monkeypatch):
    llm = DeepSeekLLM(api_key="k")

    def failing_post(url, json=None, headers=None, timeout=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(comprehensive_evaluation.requests, "post", failing_post)

    result = llm([])

    assert "API" in result
    assert llm.usage_stats()["errors"] == 1
