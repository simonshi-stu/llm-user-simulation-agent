"""Offline tests for the experiment-local memory path."""

import sys
import types

from fakes import FakeInteractionTool, FakeLLM, make_review

import comprehensive_evaluation
from comprehensive_evaluation import ExperimentConfig, ExperimentRunner
from improved_agent_with_quality import ImprovedSimulationAgent
from local_memory import LocalMemoryStore


def make_agent(llm, memory_store, user_id="u1", item_id="i1"):
    agent = ImprovedSimulationAgent(
        llm=llm,
        enable_reflection=False,
        use_memory=True,
        memory_store=memory_store,
    )
    agent.task = {"user_id": user_id, "item_id": item_id}
    agent.interaction_tool = FakeInteractionTool(
        user_reviews=[make_review(4, "Historical user review " * 8)],
        item_reviews=[make_review(5, "Reference review " * 8)],
    )
    return agent


def test_memory_is_user_scoped_and_bounded():
    store = LocalMemoryStore(max_entries_per_user=2)

    store.remember("u1", "first")
    store.remember("u1", "second")
    store.remember("u1", "third")
    store.remember("u2", "other")

    assert [entry.text for entry in store.recall("u1")] == ["third", "second"]
    assert [entry.text for entry in store.recall("u2")] == ["other"]


def test_memory_is_shared_by_agents_within_a_run():
    store = LocalMemoryStore()
    first = make_agent(
        FakeLLM(['{"stars": 4.0, "review": "First generated review."}']),
        store,
    )
    first.workflow()

    second_llm = FakeLLM(
        ['{"stars": 3.0, "review": "Second generated review."}']
    )
    second = make_agent(second_llm, store, item_id="i2")
    second.workflow()

    prompt = second_llm.calls[0]["messages"][0]["content"]
    assert "本次实验内该用户此前生成的评论" in prompt
    assert "First generated review." in prompt


def test_memory_disabled_does_not_read_or_write():
    store = LocalMemoryStore()
    agent = ImprovedSimulationAgent(
        llm=FakeLLM(['{"stars": 4.0, "review": "No memory."}']),
        enable_reflection=False,
        use_memory=False,
        memory_store=store,
    )
    agent.task = {"user_id": "u1", "item_id": "i1"}
    agent.interaction_tool = FakeInteractionTool()

    agent.workflow()

    assert store.recall("u1") == []


def test_runner_gives_one_store_to_agents_in_an_experiment(
    monkeypatch, tmp_path
):
    class FixedLLM:
        def __init__(self):
            self.calls = []

        def __call__(self, messages, **_kwargs):
            self.calls.append(messages)
            stars = 4.0 if len(self.calls) == 1 else 3.0
            return f'{{"stars": {stars}, "review": "Generated in this run."}}'

        def usage_stats(self):
            return {"calls": len(self.calls), "errors": 0, "latency_seconds": 0.0}

    class FakeSimulator:
        def __init__(self, **_kwargs):
            self.groundtruth_data = []

        def set_task_and_groundtruth(self, **_kwargs):
            self.groundtruth_data = [{"stars": 4.0}, {"stars": 3.0}]

        def set_agent(self, agent_class):
            self.agent_class = agent_class

        def set_llm(self, llm):
            self.llm = llm

        def run_simulation(self, **_kwargs):
            outputs = []
            for item_id in ("i1", "i2"):
                agent = self.agent_class(llm=self.llm)
                agent.task = {"user_id": "u1", "item_id": item_id}
                agent.interaction_tool = FakeInteractionTool(
                    user_reviews=[make_review(4, "History " * 20)],
                    item_reviews=[make_review(5, "Reference " * 20)],
                )
                outputs.append(agent.workflow())
            return outputs

        def evaluate(self):
            return {}

    fake_framework = types.ModuleType("websocietysimulator")
    fake_framework.Simulator = FakeSimulator
    monkeypatch.setitem(sys.modules, "websocietysimulator", fake_framework)
    created_llms = []

    def build_llm(**_kwargs):
        llm = FixedLLM()
        created_llms.append(llm)
        return llm

    monkeypatch.setattr(comprehensive_evaluation, "DeepSeekLLM", build_llm)

    runner = ExperimentRunner(
        data_dir="Dataset",
        task_set="yelp",
        api_key="test-key",
        num_tasks=2,
        max_workers=1,
        output_dir=str(tmp_path / "results"),
    )
    runner.run_experiment(
        ExperimentConfig(
            name="Local_Memory",
            enable_reflection=False,
            use_memory=True,
            max_reference_reviews=1,
        )
    )

    assert len(created_llms) == 1
    second_prompt = created_llms[0].calls[1][0]["content"]
    assert "本次实验内该用户此前生成的评论" in second_prompt
    assert "Generated in this run." in second_prompt
