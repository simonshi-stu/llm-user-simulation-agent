"""Synthetic, no-network tests for the label-safe P0 rating baselines."""

import inspect
import json
import random

import pytest

from rating_baselines import (
    BASELINE_MODES,
    DEFAULT_LAMBDA_ITEM,
    DEFAULT_LAMBDA_USER,
    DEFAULT_WEIGHT_USER,
    BaselineInputError,
    BaselineParams,
    RatingContext,
    RatingPrediction,
    predict_all,
    predict_for_context,
    score_baselines,
    validate_global_prior,
)


def context(
    task_index,
    *,
    user=(),
    item=(),
    split="test",
    group="U0001",
    stratum="first_seen",
):
    return RatingContext(
        task_index=task_index,
        split=split,
        user_group=group,
        user_repeat_stratum=stratum,
        user_stars=tuple(user),
        item_stars=tuple(item),
    )


def test_hand_calculated_four_baselines_and_shrinkage():
    prediction = predict_for_context(
        context(0, user=(5.0, 4.0), item=(1.0, 5.0)), 3.0
    )

    # user mean 4.5 shrunk to prior 3 with lambda 5: (4.5*2 + 5*3) / 7
    assert prediction["user"].value == pytest.approx(24.0 / 7.0)
    # item mean 3.0 with the same shrinkage stays 3.0
    assert prediction["item"].value == pytest.approx(3.0)
    assert prediction["global"].value == pytest.approx(3.0)
    assert prediction["combined"].value == pytest.approx(
        DEFAULT_WEIGHT_USER * (24.0 / 7.0) + (1 - DEFAULT_WEIGHT_USER) * 3.0
    )
    assert prediction["user"].user_cold_start is False
    assert prediction["user"].item_cold_start is False


def test_tiny_single_observation_shrinks_toward_prior():
    prediction = predict_for_context(context(0, user=(5.0,)), 3.0)
    # (5*1 + 5*3) / 6
    assert prediction["user"].value == pytest.approx(20.0 / 6.0)
    assert prediction["item"].value == pytest.approx(3.0)


def test_cold_start_falls_back_to_global_and_is_counted():
    prediction = predict_for_context(context(7, user=(), item=()), 4.25)

    for mode in BASELINE_MODES:
        assert prediction[mode].value == pytest.approx(4.25)
        assert prediction[mode].user_cold_start is True
        assert prediction[mode].item_cold_start is True


def test_partial_cold_start_flags_are_per_context():
    prediction = predict_for_context(context(1, user=(2.0,), item=()), 4.0)
    assert prediction["user"].user_cold_start is False
    assert prediction["user"].item_cold_start is True
    assert prediction["item"].value == pytest.approx(4.0)
    assert prediction["combined"].value == pytest.approx(
        DEFAULT_WEIGHT_USER * ((2.0 + 5.0 * 4.0) / 6.0)
        + (1 - DEFAULT_WEIGHT_USER) * 4.0
    )


def test_invalid_prior_and_context_stars_are_rejected():
    with pytest.raises(BaselineInputError, match="invalid_global_prior"):
        validate_global_prior(0.5)
    with pytest.raises(BaselineInputError, match="invalid_global_prior"):
        validate_global_prior(float("nan"))
    with pytest.raises(BaselineInputError, match="invalid_global_prior"):
        validate_global_prior(True)
    with pytest.raises(BaselineInputError, match="invalid_context_star"):
        predict_for_context(context(0, user=(6.0,)), 3.0)
    with pytest.raises(BaselineInputError, match="invalid_context_star"):
        predict_for_context(context(0, item=(float("inf"),)), 3.0)
    with pytest.raises(BaselineInputError, match="invalid_context_star"):
        predict_for_context(context(0, user=(True,)), 3.0)


def test_invalid_fixed_params_are_rejected():
    with pytest.raises(BaselineInputError, match="invalid_shrinkage_lambda"):
        BaselineParams(lambda_user=-1.0)
    with pytest.raises(BaselineInputError, match="invalid_combination_weight"):
        BaselineParams(weight_user=1.5)


def test_predict_all_keeps_same_task_order_for_every_mode():
    contexts = [context(2, user=(5.0,)), context(0), context(1, item=(4.0,))]
    predictions = predict_all(contexts, 3.5)
    for mode in BASELINE_MODES:
        assert [row.task_index for row in predictions[mode]] == [2, 0, 1]
        assert [row.mode for row in predictions[mode]] == [mode] * 3
        assert all(1.0 <= row.value <= 5.0 for row in predictions[mode])
    with pytest.raises(BaselineInputError, match="duplicate_task_index"):
        predict_all([context(0), context(0)], 3.5)


def test_predictions_accept_no_groundtruth_or_split_tuning():
    for function in (predict_for_context, predict_all):
        parameters = list(inspect.signature(function).parameters)
        assert parameters == ["context", "global_prior", "params"] or (
            parameters == ["contexts", "global_prior", "params"]
        )
    assert "actual" not in RatingContext.__dataclass_fields__
    # split/stratum labels must not change any prediction value.
    left = predict_for_context(
        context(0, user=(5.0,), split="train", stratum="first_seen"), 3.0
    )
    right = predict_for_context(
        context(0, user=(5.0,), split="test", stratum="repeat"), 3.0
    )
    for mode in BASELINE_MODES:
        assert left[mode].value == right[mode].value


def test_default_params_are_the_fixed_documented_values():
    params = BaselineParams()
    assert params.lambda_user == DEFAULT_LAMBDA_USER == 5.0
    assert params.lambda_item == DEFAULT_LAMBDA_ITEM == 5.0
    assert params.weight_user == DEFAULT_WEIGHT_USER == 0.5


def _scored_fixture(start=0, count=6, split="test"):
    """Three user clusters, two tasks each, deterministic labels."""
    contexts = []
    actuals = {}
    splits = {}
    clusters = {}
    strata = {}
    for position in range(count):
        index = start + position
        cluster = f"U{(position // 2) + 1:04d}"
        actual = float((position % 5) + 1)
        contexts.append(
            context(
                index,
                user=(min(5.0, actual + 1.0), max(1.0, actual - 1.0)),
                item=(actual,),
                split=split,
                group=cluster,
                stratum="first_seen" if position % 2 == 0 else "repeat",
            )
        )
        actuals[index] = actual
        splits[index] = split
        clusters[index] = cluster
        strata[index] = "first_seen" if position % 2 == 0 else "repeat"
    predictions = predict_all(contexts, 3.0)
    return predictions, actuals, splits, clusters, strata


def _score(n_boot=200, seed=42, start=0, count=6, split="test"):
    predictions, actuals, splits, clusters, strata = _scored_fixture(
        start=start, count=count, split=split
    )
    return score_baselines(
        predictions,
        actuals,
        split_by_index=splits,
        cluster_by_index=clusters,
        stratum_by_index=strata,
        n_boot=n_boot,
        seed=seed,
    )


def test_score_baselines_reports_point_metrics_and_strata():
    report = _score()
    assert report["modes"] == list(BASELINE_MODES)
    assert report["task_counts"]["total"] == 6
    assert report["task_counts"]["user_clusters"]["total"] == 3
    for mode in BASELINE_MODES:
        test = report["point"][mode]["test"]
        assert test["n"] == 6
        assert test["coverage"] == 1.0
        assert test["rmse"] is not None and test["rmse"] >= 0.0
        assert test["mae"] is not None
        assert test["mean_signed_bias"] is not None
    assert (
        report["point"]["combined"]["test_repeat_strata"]["first_seen"]["n"] == 3
    )


def test_score_requires_matching_groundtruth_and_order():
    predictions, actuals, splits, clusters, strata = _scored_fixture()
    with pytest.raises(BaselineInputError, match="unmatched_groundtruth"):
        score_baselines(
            predictions,
            {key: value for key, value in actuals.items() if key != 0},
            split_by_index=splits,
            cluster_by_index=clusters,
            stratum_by_index=strata,
        )
    broken = {mode: list(rows) for mode, rows in predictions.items()}
    broken["user"] = list(reversed(broken["user"]))
    with pytest.raises(BaselineInputError, match="mode_task_order_mismatch"):
        score_baselines(
            broken,
            actuals,
            split_by_index=splits,
            cluster_by_index=clusters,
            stratum_by_index=strata,
        )
    bad_actuals = dict(actuals)
    bad_actuals[0] = 5.5
    with pytest.raises(BaselineInputError, match="invalid_groundtruth_star"):
        score_baselines(
            predictions,
            bad_actuals,
            split_by_index=splits,
            cluster_by_index=clusters,
            stratum_by_index=strata,
        )


def test_cluster_bootstrap_is_deterministic_and_paired():
    first = _score(seed=42)
    second = _score(seed=42)
    assert json.dumps(first["bootstrap_test"], sort_keys=True) == json.dumps(
        second["bootstrap_test"], sort_keys=True
    )
    bootstrap = first["bootstrap_test"]
    assert bootstrap["status"] == "computed"
    assert bootstrap["cluster_count"] == 3
    assert bootstrap["resampling_unit"] == "user_cluster_with_replacement"
    for mode in BASELINE_MODES:
        for metric in ("rmse", "mae", "mean_signed_bias"):
            interval = bootstrap["intervals"][mode][metric]
            assert interval is not None and interval[0] <= interval[1]
    assert "global" not in bootstrap["paired_vs_global"]
    for mode in ("user", "item", "combined"):
        paired = bootstrap["paired_vs_global"][mode]
        point = (
            bootstrap["point"][mode]["rmse"]
            - bootstrap["point"]["global"]["rmse"]
        )
        assert paired["rmse_difference"]["point"] == pytest.approx(point)


def test_bootstrap_draws_whole_users_once_per_iteration(monkeypatch):
    calls = {"count": 0}
    original = random.Random.choices

    def counting_choices(self, population, *, k):
        calls["count"] += 1
        return original(self, population, k=k)

    monkeypatch.setattr(random.Random, "choices", counting_choices)
    report = _score(n_boot=200, seed=42)
    # One shared cluster draw per iteration for all four baselines, with
    # replacement allowed and no forced re-draw of degenerate samples.
    assert calls["count"] == report["bootstrap_test"]["n_boot"] == 200


def test_bootstrap_is_not_fabricated_for_one_cluster_or_empty_test():
    single_cluster = _score(count=2)
    bootstrap = single_cluster["bootstrap_test"]
    assert bootstrap["status"] == "insufficient_user_clusters"
    assert bootstrap["intervals"] is None

    empty_test = _score(count=2, split="train")
    bootstrap = empty_test["bootstrap_test"]
    assert bootstrap["status"] == "empty_test"
    assert bootstrap["intervals"] is None


def test_item_that_always_falls_back_has_zero_paired_difference():
    contexts = [
        context(
            index,
            user=(3.0,),
            item=(),
            group=f"U{(index // 2) + 1:04d}",
            stratum="first_seen" if index % 2 == 0 else "repeat",
        )
        for index in range(4)
    ]
    actuals = {index: float((index % 3) + 1) for index in range(4)}
    splits = {index: "test" for index in range(4)}
    clusters = {index: f"U{(index // 2) + 1:04d}" for index in range(4)}
    strata = {
        index: "first_seen" if index % 2 == 0 else "repeat"
        for index in range(4)
    }
    report = score_baselines(
        predict_all(contexts, 3.0),
        actuals,
        split_by_index=splits,
        cluster_by_index=clusters,
        stratum_by_index=strata,
        n_boot=200,
        seed=42,
    )
    paired = report["bootstrap_test"]["paired_vs_global"]["item"]
    assert paired["rmse_difference"] == {"point": 0.0, "ci95": [0.0, 0.0]}
    assert paired["mae_difference"] == {"point": 0.0, "ci95": [0.0, 0.0]}


def test_prediction_rows_are_frozen_dataclasses():
    prediction = RatingPrediction(0, "global", 3.0, False, False)
    with pytest.raises((AttributeError, TypeError)):
        prediction.value = 4.0
