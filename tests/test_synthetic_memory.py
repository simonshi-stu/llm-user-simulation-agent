"""Synthetic repeated-user ablation for the experiment-local memory path."""

from fakes import FakeInteractionTool, FakeLLM, make_review

from improved_agent_with_quality import ImprovedSimulationAgent
from local_memory import LocalMemoryStore


MEMORY_MARKER = "本次实验内该用户此前生成的评论"
REPEATED_USER = "synthetic-user-1"
OTHER_USER = "synthetic-user-2"


SYNTHETIC_TASKS = (
    {"user_id": REPEATED_USER, "item_id": "synthetic-item-a"},
    {"user_id": REPEATED_USER, "item_id": "synthetic-item-b"},
    {"user_id": OTHER_USER, "item_id": "synthetic-item-c"},
    {"user_id": REPEATED_USER, "item_id": "synthetic-item-d"},
)


def run_synthetic_ablation(use_memory):
    """Run repeated-user tasks and return prompts, outputs and the store."""
    responses = [
        '{"stars": 4.0, "review": "Synthetic review A."}',
        '{"stars": 3.0, "review": "Synthetic review B."}',
        '{"stars": 5.0, "review": "Other user review."}',
        '{"stars": 2.0, "review": "Synthetic review D."}',
    ]
    llm = FakeLLM(responses)
    store = LocalMemoryStore()
    prompts = []
    outputs = []

    for task in SYNTHETIC_TASKS:
        agent = ImprovedSimulationAgent(
            llm=llm,
            enable_reflection=False,
            use_memory=use_memory,
            memory_store=store,
        )
        agent.task = task
        agent.interaction_tool = FakeInteractionTool(
            user={"user_id": task["user_id"], "name": task["user_id"]},
            item={"item_id": task["item_id"], "name": task["item_id"]},
            user_reviews=[make_review(4, "Synthetic historical review " * 8)],
            item_reviews=[make_review(5, "Synthetic reference review " * 8)],
        )

        outputs.append(agent.workflow())
        prompts.append(llm.calls[-1]["messages"][0]["content"])

    return {
        "outputs": outputs,
        "prompts": prompts,
        "memory_prompt_hits": sum(MEMORY_MARKER in prompt for prompt in prompts),
        "store": store,
    }


def test_repeated_user_memory_is_visible_and_user_scoped():
    """Full receives prior same-user reviews; another user does not."""
    full = run_synthetic_ablation(use_memory=True)
    prompts = full["prompts"]

    assert MEMORY_MARKER not in prompts[0]
    assert MEMORY_MARKER in prompts[1]
    assert "Synthetic review A." in prompts[1]
    assert MEMORY_MARKER not in prompts[2]
    assert "Synthetic review A." not in prompts[2]
    assert MEMORY_MARKER in prompts[3]
    assert "Synthetic review B." in prompts[3]
    assert "Synthetic review A." in prompts[3]

    assert full["memory_prompt_hits"] == 2
    assert [
        entry.text for entry in full["store"].recall(REPEATED_USER)
    ] == [
        "Synthetic review D.",
        "Synthetic review B.",
        "Synthetic review A.",
    ]
    assert [
        entry.text for entry in full["store"].recall(OTHER_USER)
    ] == ["Other user review."]


def test_repeated_user_no_memory_ablation_has_no_memory_effect():
    """No_Memory never injects or persists generated reviews."""
    no_memory = run_synthetic_ablation(use_memory=False)

    assert no_memory["memory_prompt_hits"] == 0
    assert all(MEMORY_MARKER not in prompt for prompt in no_memory["prompts"])
    assert no_memory["store"].recall(REPEATED_USER) == []
    assert no_memory["store"].recall(OTHER_USER) == []
