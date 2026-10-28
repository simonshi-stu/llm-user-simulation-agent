"""Offline tests for the DeepSeek client wrapper."""

import pytest

import comprehensive_evaluation
from comprehensive_evaluation import DeepSeekLLM, LLMTransportError


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


def test_payload_can_request_json_mode():
    llm = DeepSeekLLM(api_key="k", json_mode=True)
    payload = llm._build_payload([], 0.4, 100)

    assert payload["response_format"] == {"type": "json_object"}


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
        raise RuntimeError("network down at https://secret.invalid/path")

    monkeypatch.setattr(comprehensive_evaluation.requests, "post", failing_post)

    # A transport failure must be a typed, safe failure, not an "API error"
    # completion that could be parsed or stored as model output.
    with pytest.raises(LLMTransportError) as excinfo:
        llm([])

    assert excinfo.value.code == "transport_error"
    assert "secret.invalid" not in str(excinfo.value)
    assert "network down" not in str(excinfo.value)
    assert llm.usage_stats()["errors"] == 1
