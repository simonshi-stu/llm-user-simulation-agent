"""Deterministic fakes for offline agent tests."""


def make_review(stars, text, useful=0, funny=0, cool=0):
    return {
        "stars": stars,
        "text": text,
        "useful": useful,
        "funny": funny,
        "cool": cool,
    }


class FakeLLM:
    """LLM stub that returns queued responses and records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages, temperature=0.7, max_tokens=800):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self.responses:
            raise AssertionError("FakeLLM has no queued responses left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeInteractionTool:
    """Minimal stand-in for the simulator interaction tool."""

    def __init__(self, user=None, item=None, user_reviews=None, item_reviews=None):
        self.user = user if user is not None else {
            "user_id": "u1",
            "name": "Test User",
            "review_count": 2,
        }
        self.item = item if item is not None else {
            "item_id": "i1",
            "name": "Test Cafe",
            "categories": "Coffee, Bakery",
        }
        self.user_reviews = list(user_reviews or [])
        self.item_reviews = list(item_reviews or [])
        self.calls = []

    def get_user(self, user_id):
        self.calls.append(("get_user", user_id))
        return self.user

    def get_item(self, item_id):
        self.calls.append(("get_item", item_id))
        return self.item

    def get_reviews(self, user_id=None, item_id=None):
        self.calls.append(("get_reviews", user_id, item_id))
        if user_id is not None:
            return self.user_reviews
        return self.item_reviews
