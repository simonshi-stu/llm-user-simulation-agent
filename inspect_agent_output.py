"""
inspect_agent_output.py - English Version

Simple agent output viewer - displays generated content without comparison
Focuses on viewing what the agent produces for individual tasks
"""

import os
import time
from websocietysimulator import Simulator

# Import DeepSeek LLM
import requests


class DeepSeekEmbeddingModel:
    """DeepSeek Embedding Model"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-embedding"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def _api_embed(self, texts):
        try:
            url = f"{self.base_url}/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {"model": self.model, "input": texts}
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            print(f"❌ Embedding API error: {e}")
            return [[0.0] * 768 for _ in texts]

    def embed_documents(self, texts):
        if not texts:
            return []
        return self._api_embed(texts)

    def embed_query(self, text):
        if not text:
            return [0.0] * 768
        return self._api_embed([text])[0]


class DeepseekLLM:
    """DeepSeek LLM Client"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1",
                 chat_model: str = "deepseek-chat", embedding_model: str = "deepseek-embedding"):
        self.api_key = api_key
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def __call__(self, messages, temperature=0.7, max_tokens=800):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"❌ Chat API error: {e}")
            return "(API error)"

    def get_embedding_model(self):
        return DeepSeekEmbeddingModel(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.embedding_model
        )


# Import your agent (with wrapper support)
try:
    from agent_wrapper import MySimulationAgent
except ImportError:
    from improved_agent_with_quality import ImprovedSimulationAgent as OriginalAgent

    class MySimulationAgent(OriginalAgent):
        """Temporary wrapper"""
        def __init__(self, llm=None, **kwargs):
            if llm is not None:
                super().__init__(llm=llm, **kwargs)
            else:
                from websocietysimulator.llm import LLMBase
                class DummyLLM(LLMBase):
                    def __call__(self, *args, **kwargs):
                        return "dummy"
                super().__init__(llm=DummyLLM(), **kwargs)

        def __call__(self, task_description):
            if isinstance(task_description, dict):
                self.task = task_description
            else:
                self.task = {
                    'user_id': getattr(task_description, 'user_id', None),
                    'item_id': getattr(task_description, 'item_id', None)
                }
            return self.workflow()


def simple_inspect():
    """Simple agent output viewer"""

    print("\n" + "="*70)
    print("📝 Simple Agent Output Viewer")
    print("="*70)

    # Configuration
    DATA_DIR = "Dataset"
    TASK_SET = "yelp"
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

    # Input task index
    try:
        task_idx = int(input("\nEnter task index (0-399): ").strip())
        if task_idx < 0 or task_idx >= 400:
            print("❌ Invalid index, using default: 0")
            task_idx = 0
    except (TypeError, ValueError):
        print("⚠️  Using default task index: 0")
        task_idx = 0

    print(f"\n📌 Inspecting task #{task_idx} output\n")

    # Initialize Simulator
    print("[1/4] Initializing Simulator...")
    try:
        simulator = Simulator(data_dir=DATA_DIR, device="auto", cache=True)
        print("✅ Done\n")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return

    # Load tasks
    print("[2/4] Loading tasks...")
    try:
        simulator.set_task_and_groundtruth(
            task_dir=f"example/track1/{TASK_SET}/tasks",
            groundtruth_dir=f"example/track1/{TASK_SET}/groundtruth"
        )

        # Get task
        if hasattr(simulator, 'task_pool'):
            task = simulator.task_pool[task_idx]
        elif hasattr(simulator, 'tasks'):
            task = simulator.tasks[task_idx]
        else:
            print("❌ Cannot access tasks")
            return

        # Get groundtruth
        if hasattr(simulator, 'groundtruth_data'):
            groundtruth = simulator.groundtruth_data[task_idx]
        elif hasattr(simulator, 'groundtruth_pool'):
            groundtruth = simulator.groundtruth_pool[task_idx]
        else:
            groundtruth = None

        print("✅ Done\n")

    except Exception as e:
        print(f"❌ Failed: {e}")
        return

    # Display task information
    print("="*70)
    print("📋 Task Information")
    print("="*70)

    user_id = getattr(task, 'user_id', task.get('user_id', 'N/A') if hasattr(task, 'get') else 'N/A')
    item_id = getattr(task, 'item_id', task.get('item_id', 'N/A') if hasattr(task, 'get') else 'N/A')

    print(f"\nUser ID: {user_id}")
    print(f"Item ID: {item_id}")

    if groundtruth:
        if isinstance(groundtruth, dict):
            gt_stars = groundtruth.get('stars', 'N/A')
            gt_text = groundtruth.get('text', '')
        else:
            gt_stars = getattr(groundtruth, 'stars', 'N/A')
            gt_text = getattr(groundtruth, 'text', '')

        print("\n🎯 Ground Truth:")
        print(f"   Stars: {gt_stars}")
        if gt_text:
            print(f"   Review: {gt_text[:150]}...")

    # Set agent
    print("\n[3/4] Setting up agent...")
    try:
        simulator.set_agent(MySimulationAgent)
        simulator.set_llm(DeepseekLLM(api_key=API_KEY))
        print("✅ Done\n")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return

    # Run agent
    print("[4/4] Running agent...")
    print("⏳ Generating, please wait (10-30 seconds)...\n")

    try:
        agent = MySimulationAgent(llm=simulator.llm)
        agent.interaction_tool = simulator.interaction_tool

        start_time = time.time()
        output = agent(task)
        elapsed_time = time.time() - start_time

        print(f"✅ Completed in {elapsed_time:.2f} seconds\n")

        # Display output
        print("="*70)
        print("🤖 Agent Output")
        print("="*70)

        if output and isinstance(output, dict):
            # Extract rating and review
            if "output" in output:
                pred_stars = output["output"].get("stars")
                pred_review = output["output"].get("review", "")
            else:
                pred_stars = output.get("stars")
                pred_review = output.get("review", "")

            # Display rating
            print(f"\n⭐ Predicted Rating: {pred_stars}")

            # Calculate error (if groundtruth available)
            if groundtruth:
                if isinstance(groundtruth, dict):
                    actual_stars = groundtruth.get("stars")
                else:
                    actual_stars = getattr(groundtruth, "stars", None)

                if actual_stars is not None and pred_stars is not None:
                    error = abs(float(pred_stars) - float(actual_stars))
                    print(f"🎯 Actual Rating: {actual_stars}")
                    print(f"📊 Error: {error:.2f}")

            # Display review
            print("\n📝 Generated Review:")
            print("-" * 70)
            print(pred_review)
            print("-" * 70)

            print(f"\n📏 Review Length: {len(pred_review)} characters")

        else:
            print("❌ Invalid output format")
            print(output)

        print("\n" + "="*70)
        print("✅ Inspection Complete")
        print("="*70)

    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        import traceback
        traceback.print_exc()

    # Ask to continue
    print("\n💡 Options:")
    print("  1. Inspect another task")
    print("  2. Exit")

    try:
        choice = int(input("\nSelect (1-2): ").strip())
        if choice == 1:
            simple_inspect()
    except (TypeError, ValueError):
        pass

    print("\n👋 Goodbye!")


if __name__ == '__main__':
    simple_inspect()
