"""Offline checks for the public Amazon source adapter."""

from datetime import datetime, timezone

import pytest

from amazon_p0 import normalize_amazon_row
from amazon_full_evaluation import run_full_evaluation


def _epoch_seconds():
    return int(datetime(2023, 9, 12, tzinfo=timezone.utc).timestamp())


def test_normalizer_supports_both_schemas_and_epoch_units():
    seconds = _epoch_seconds()
    rows = [{
        "reviewerID": "legacy-user",
        "asin": "legacy-item",
        "overall": 4,
        "reviewText": "legacy text",
        "unixReviewTime": seconds,
    }]
    for scale in (1_000, 1_000_000, 1_000_000_000):
        rows.append({
            "user_id": "new-user",
            "item_id": "new-item",
            "rating": 3,
            "text": "new text",
            "timestamp": seconds * scale,
        })

    for index, row in enumerate(rows):
        record, reason = normalize_amazon_row(row, index)
        assert reason is None
        assert record["timestamp"] == "2023-09-12"
        assert record["user_id"].startswith("u_")
        assert record["item_id"].startswith("i_")
        assert "reviewerID" not in record
        assert "asin" not in record


def test_invalid_timestamp_is_explicitly_rejected():
    record, reason = normalize_amazon_row(
        {
            "reviewerID": "user",
            "asin": "item",
            "overall": 5,
            "reviewText": "text",
            "unixReviewTime": "not-a-timestamp",
        },
        0,
    )

    assert record is None
    assert reason == "invalid_timestamp"


def test_full_runner_rejects_historical_deepseek_configuration():
    with pytest.raises(ValueError, match="deepseek_not_allowed"):
        run_full_evaluation(
            "private-source",
            "private-output",
            "private-key",
            chat_model="deepseek-chat",
            base_url="https://example.invalid/v1",
        )
