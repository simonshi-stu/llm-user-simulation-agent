"""Offline checks for deterministic label-safe rating baselines."""

import pytest

from rating_baselines import RatingContext, predict_for_context


def test_user_and_item_predictions_use_only_visible_context():
    context = RatingContext(
        task_index=0,
        split="test",
        user_group="U0001",
        user_repeat_stratum="repeated",
        user_stars=(5.0, 4.0),
        item_stars=(1.0, 5.0),
    )

    predictions = predict_for_context(context, global_prior=3.0)

    assert predictions["user"].value == pytest.approx(24.0 / 7.0)
    assert predictions["item"].value == pytest.approx(3.0)
    assert predictions["global"].value == pytest.approx(3.0)


def test_empty_context_is_explicit_cold_start():
    context = RatingContext(
        task_index=1,
        split="test",
        user_group="U0002",
        user_repeat_stratum="first_seen",
        user_stars=(),
        item_stars=(),
    )

    predictions = predict_for_context(context, global_prior=4.25)

    assert all(prediction.value == pytest.approx(4.25) for prediction in predictions.values())
    assert all(prediction.user_cold_start for prediction in predictions.values())
    assert all(prediction.item_cold_start for prediction in predictions.values())
