"""Deterministic, label-safe rating baselines for the offline P0 protocol.

Prediction functions in this module consume only provider-visible rating
contexts and an already verified global prior. They never read groundtruth
ratings, task labels, review text, or source files; scoring is a separate step
that callers must run only after every prediction has been computed.

The four baselines are:

- ``global``:   the verified frozen-history-prefix prior.
- ``user``:     the target user's visible history mean shrunk to the prior.
- ``item``:     other users' visible item reference mean shrunk to the prior.
- ``combined``: a fixed convex combination of the shrunk user and item values.

Shrinkage follows ``(mean * n + lambda * prior) / (n + lambda)``. A context
with no visible user history (or no visible item references) falls back to the
global prior and is counted explicitly as a cold start.

User-cluster bootstrap intervals resample whole users with replacement; every
baseline shares the same draw, so paired differences against ``global`` are
computed on identical resamples. Intervals are never fabricated for an empty
test split or a single user cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence

BASELINE_MODES: tuple[str, ...] = ("global", "user", "item", "combined")
DEFAULT_LAMBDA_USER = 5.0
DEFAULT_LAMBDA_ITEM = 5.0
DEFAULT_WEIGHT_USER = 0.5
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_BOOTSTRAP_SAMPLES = 2000
STAR_MIN = 1.0
STAR_MAX = 5.0
SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")
STRATUM_NAMES: tuple[str, ...] = ("first_seen", "repeat")


class BaselineInputError(ValueError):
    """Safe, categorized input failure that never carries data values."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class RatingContext:
    """Provider-visible context for one task; never contains groundtruth.

    ``user_stars``/``item_stars`` are the already time-filtered, non-target
    rating histories the strict temporal provider exposed to the agent.
    """

    task_index: int
    split: str
    user_group: str
    user_repeat_stratum: str
    user_stars: tuple[float, ...]
    item_stars: tuple[float, ...]


@dataclass(frozen=True)
class BaselineParams:
    """Fixed baseline parameters; defaults are never tuned on any split."""

    lambda_user: float = DEFAULT_LAMBDA_USER
    lambda_item: float = DEFAULT_LAMBDA_ITEM
    weight_user: float = DEFAULT_WEIGHT_USER

    def __post_init__(self) -> None:
        for name in ("lambda_user", "lambda_item"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise BaselineInputError("invalid_shrinkage_lambda")
        weight = self.weight_user
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or not 0.0 <= float(weight) <= 1.0
        ):
            raise BaselineInputError("invalid_combination_weight")


@dataclass(frozen=True)
class RatingPrediction:
    task_index: int
    mode: str
    value: float
    user_cold_start: bool
    item_cold_start: bool


def validate_global_prior(value: object) -> float:
    """Accept only a finite prior inside the allowed star range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineInputError("invalid_global_prior")
    prior = float(value)
    if not math.isfinite(prior) or not STAR_MIN <= prior <= STAR_MAX:
        raise BaselineInputError("invalid_global_prior")
    return prior


def _validated_stars(values: Sequence[float]) -> list[float]:
    stars: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BaselineInputError("invalid_context_star")
        number = float(value)
        if not math.isfinite(number) or not STAR_MIN <= number <= STAR_MAX:
            raise BaselineInputError("invalid_context_star")
        stars.append(number)
    return stars


def _shrunk_mean(mean: float, count: int, prior: float, lambda_: float) -> float:
    return (mean * count + lambda_ * prior) / (count + lambda_)


def _clamp_star(value: float) -> float:
    return min(STAR_MAX, max(STAR_MIN, value))


def predict_for_context(
    context: RatingContext,
    global_prior: float,
    params: BaselineParams | None = None,
) -> dict[str, RatingPrediction]:
    """Predict all four baselines from one safe context and verified prior."""
    prior = validate_global_prior(global_prior)
    resolved = params if params is not None else BaselineParams()
    if not isinstance(context, RatingContext):
        raise BaselineInputError("invalid_rating_context")
    user_stars = _validated_stars(context.user_stars)
    item_stars = _validated_stars(context.item_stars)
    user_cold = not user_stars
    item_cold = not item_stars
    user_value = (
        prior
        if user_cold
        else _shrunk_mean(
            sum(user_stars) / len(user_stars),
            len(user_stars),
            prior,
            resolved.lambda_user,
        )
    )
    item_value = (
        prior
        if item_cold
        else _shrunk_mean(
            sum(item_stars) / len(item_stars),
            len(item_stars),
            prior,
            resolved.lambda_item,
        )
    )
    combined_value = (
        resolved.weight_user * user_value
        + (1.0 - resolved.weight_user) * item_value
    )
    values = {
        "global": prior,
        "user": user_value,
        "item": item_value,
        "combined": combined_value,
    }
    return {
        mode: RatingPrediction(
            task_index=context.task_index,
            mode=mode,
            value=_clamp_star(value),
            user_cold_start=user_cold,
            item_cold_start=item_cold,
        )
        for mode, value in values.items()
    }


def predict_all(
    contexts: Sequence[RatingContext],
    global_prior: float,
    params: BaselineParams | None = None,
) -> dict[str, tuple[RatingPrediction, ...]]:
    """Predict every baseline for every context in the given order.

    The signature intentionally accepts no groundtruth or split information:
    every mode sees the same tasks in the same order, and no parameter is
    tuned on any split.
    """
    prior = validate_global_prior(global_prior)
    resolved = params if params is not None else BaselineParams()
    predictions: dict[str, list[RatingPrediction]] = {
        mode: [] for mode in BASELINE_MODES
    }
    seen: set[int] = set()
    for context in contexts:
        if not isinstance(context, RatingContext):
            raise BaselineInputError("invalid_rating_context")
        if context.task_index in seen:
            raise BaselineInputError("duplicate_task_index")
        seen.add(context.task_index)
        for mode, prediction in predict_for_context(
            context, prior, resolved
        ).items():
            predictions[mode].append(prediction)
    return {mode: tuple(rows) for mode, rows in predictions.items()}


def _error_metrics(
    rows: Sequence[dict], mode: str
) -> dict[str, float | int | None]:
    if not rows:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "mean_signed_bias": None,
        }
    errors = [row["values"][mode] - row["actual"] for row in rows]
    return {
        "n": len(rows),
        "rmse": math.sqrt(
            sum(error * error for error in errors) / len(errors)
        ),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "mean_signed_bias": sum(errors) / len(errors),
    }


def _metrics_for_mode(rows: Sequence[dict], mode: str) -> dict:
    metrics = dict(_error_metrics(rows, mode))
    seen = [row for row in rows if mode in row["values"]]
    metrics["coverage"] = (
        len(seen) / len(rows) if rows else None
    )
    if mode in ("user", "item", "combined"):
        metrics["cold_start"] = {
            "user_fallback": sum(
                row["cold_user"] for row in rows if mode in row["values"]
            ),
            "item_fallback": sum(
                row["cold_item"] for row in rows if mode in row["values"]
            ),
            "both_fallback": sum(
                row["cold_user"] and row["cold_item"]
                for row in rows if mode in row["values"]
            ),
        }
    return metrics


def _percentile_interval(values: Sequence[float]) -> list[float] | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return [ordered[0], ordered[0]]
    return [
        ordered[int(0.025 * (len(ordered) - 1))],
        ordered[int(0.975 * (len(ordered) - 1))],
    ]


def _bootstrap_test(
    rows: Sequence[dict],
    *,
    modes: Sequence[str],
    n_boot: int,
    seed: int,
) -> dict:
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        return {
            "status": "empty_test",
            "intervals": None,
            "detail": "the test split has no valid tasks; no interval is fabricated",
        }
    clusters = sorted({row["cluster"] for row in test_rows})
    if len(clusters) < 2:
        return {
            "status": "insufficient_user_clusters",
            "intervals": None,
            "task_count": len(test_rows),
            "cluster_count": len(clusters),
            "detail": (
                "a single user cluster cannot support a user-cluster "
                "interval; no interval is fabricated"
            ),
        }

    groups: dict[str, list[dict]] = {cluster: [] for cluster in clusters}
    for row in test_rows:
        groups[row["cluster"]].append(row)
    iterations = max(200, int(n_boot))
    rng = random.Random(seed)
    samples: dict[str, dict[str, list[float]]] = {
        mode: {"rmse": [], "mae": [], "mean_signed_bias": []}
        for mode in modes
    }
    deltas: dict[str, dict[str, list[float]]] = {
        mode: {"rmse": [], "mae": []}
        for mode in modes if mode != "global"
    }
    for _ in range(iterations):
        # Resample whole users with replacement; duplicates are allowed and
        # the same draw is shared by every baseline in this iteration.
        sampled: list[dict] = []
        for cluster in rng.choices(clusters, k=len(clusters)):
            sampled.extend(groups[cluster])
        per_mode = {mode: _error_metrics(sampled, mode) for mode in modes}
        for mode in modes:
            for key in ("rmse", "mae", "mean_signed_bias"):
                samples[mode][key].append(per_mode[mode][key])
            if mode != "global":
                deltas[mode]["rmse"].append(
                    per_mode[mode]["rmse"] - per_mode["global"]["rmse"]
                )
                deltas[mode]["mae"].append(
                    per_mode[mode]["mae"] - per_mode["global"]["mae"]
                )

    point = {mode: _error_metrics(test_rows, mode) for mode in modes}
    return {
        "status": "computed",
        "seed": seed,
        "n_boot": iterations,
        "task_count": len(test_rows),
        "cluster_count": len(clusters),
        "resampling_unit": "user_cluster_with_replacement",
        "pairing": (
            "one cluster draw per iteration shared by every baseline; "
            "paired differences use identical resamples"
        ),
        "point": point,
        "intervals": {
            mode: {
                key: _percentile_interval(values)
                for key, values in samples[mode].items()
            }
            for mode in modes
        },
        "paired_vs_global": {
            mode: {
                "rmse_difference": {
                    "point": point[mode]["rmse"] - point["global"]["rmse"],
                    "ci95": _percentile_interval(deltas[mode]["rmse"]),
                },
                "mae_difference": {
                    "point": point[mode]["mae"] - point["global"]["mae"],
                    "ci95": _percentile_interval(deltas[mode]["mae"]),
                },
            }
            for mode in modes
            if mode != "global"
        },
    }


def score_baselines(
    predictions_by_mode: Mapping[str, Sequence[RatingPrediction]],
    actual_by_index: Mapping[int, float],
    *,
    split_by_index: Mapping[int, str],
    cluster_by_index: Mapping[int, str],
    stratum_by_index: Mapping[int, str],
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    """Score already-computed predictions after the caller read groundtruth.

    This function is the only place that receives actual ratings. It validates
    that every baseline saw the same tasks in the same order and that every
    prediction has a matching groundtruth value.
    """
    if not isinstance(predictions_by_mode, Mapping) or set(
        predictions_by_mode
    ) != set(BASELINE_MODES):
        raise BaselineInputError("missing_baseline_modes")
    orders = {
        mode: tuple(row.task_index for row in predictions_by_mode[mode])
        for mode in BASELINE_MODES
    }
    if len(set(orders.values())) != 1:
        raise BaselineInputError("mode_task_order_mismatch")
    indexes = orders["global"]
    if len(set(indexes)) != len(indexes):
        raise BaselineInputError("duplicate_task_index")
    if set(indexes) != set(int(key) for key in actual_by_index):
        raise BaselineInputError("unmatched_groundtruth")

    rows: list[dict] = []
    for position, index in enumerate(indexes):
        actual_raw = actual_by_index[index]
        if isinstance(actual_raw, bool) or not isinstance(
            actual_raw, (int, float)
        ):
            raise BaselineInputError("invalid_groundtruth_star")
        actual = float(actual_raw)
        if not math.isfinite(actual) or not STAR_MIN <= actual <= STAR_MAX:
            raise BaselineInputError("invalid_groundtruth_star")
        split = split_by_index.get(index)
        if split not in SPLIT_NAMES:
            raise BaselineInputError("invalid_split")
        cluster = cluster_by_index.get(index)
        if not isinstance(cluster, str) or not cluster:
            raise BaselineInputError("invalid_user_cluster")
        stratum = stratum_by_index.get(index)
        if stratum not in STRATUM_NAMES:
            raise BaselineInputError("invalid_user_stratum")
        values: dict[str, float] = {}
        cold_user = False
        cold_item = False
        for mode in BASELINE_MODES:
            prediction = predictions_by_mode[mode][position]
            if not isinstance(prediction, RatingPrediction):
                raise BaselineInputError("invalid_prediction_row")
            if prediction.task_index != index or prediction.mode != mode:
                raise BaselineInputError("mode_task_order_mismatch")
            if (
                not math.isfinite(prediction.value)
                or not STAR_MIN <= prediction.value <= STAR_MAX
            ):
                raise BaselineInputError("invalid_prediction_value")
            values[mode] = prediction.value
            cold_user = prediction.user_cold_start
            cold_item = prediction.item_cold_start
        rows.append({
            "index": index,
            "split": split,
            "cluster": cluster,
            "stratum": stratum,
            "actual": actual,
            "values": values,
            "cold_user": cold_user,
            "cold_item": cold_item,
        })

    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in SPLIT_NAMES
    }
    cluster_counts = {
        split: len({
            row["cluster"] for row in rows if row["split"] == split
        })
        for split in SPLIT_NAMES
    }
    point: dict[str, dict] = {}
    for mode in BASELINE_MODES:
        mode_stats: dict[str, object] = {}
        for split in (*SPLIT_NAMES, "all"):
            subset = [
                row for row in rows
                if split == "all" or row["split"] == split
            ]
            mode_stats[split] = _metrics_for_mode(subset, mode)
        strata = {
            stratum: _metrics_for_mode(
                [row for row in rows
                 if row["split"] == "test" and row["stratum"] == stratum],
                mode,
            )
            for stratum in STRATUM_NAMES
        }
        mode_stats["test_repeat_strata"] = strata
        point[mode] = mode_stats

    return {
        "schema_version": 1,
        "modes": list(BASELINE_MODES),
        "task_counts": {
            "total": len(rows),
            "by_split": split_counts,
            "user_clusters": {
                "total": len({row["cluster"] for row in rows}),
                "by_split": cluster_counts,
            },
            "repeat_target_tasks": sum(
                row["stratum"] == "repeat" for row in rows
            ),
            "first_seen_target_tasks": sum(
                row["stratum"] == "first_seen" for row in rows
            ),
        },
        "point": point,
        "bootstrap_test": _bootstrap_test(
            rows, modes=BASELINE_MODES, n_boot=n_boot, seed=seed
        ),
    }
