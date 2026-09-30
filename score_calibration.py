"""Score calibration for star-rating predictions.

The calibrator learns a mapping from raw predicted stars to observed stars
on a validation split. It is dependency-free and deterministic so it can be
unit-tested offline.
"""

import math
import random

from improved_agent_with_quality import normalize_stars


class ScoreCalibrator:
    """Fit raw predictions onto observed ratings.

    Methods:
        "bias":   corrected = raw + mean(actual - predicted)
        "linear": corrected = intercept + slope * raw (least squares)
    """

    VALID_METHODS = ("bias", "linear")

    def __init__(self, method: str = "linear"):
        if method not in self.VALID_METHODS:
            raise ValueError(f"unknown calibration method: {method}")
        self.method = method
        self.bias = 0.0
        self.slope = 1.0
        self.intercept = 0.0
        self.fitted = False

    def fit(self, predictions, actuals):
        if len(predictions) != len(actuals):
            raise ValueError(
                "predictions and actuals must have the same length"
            )
        if not predictions:
            raise ValueError("cannot fit on empty data")

        predicted = [float(value) for value in predictions]
        actual = [float(value) for value in actuals]
        n = len(predicted)

        if self.method == "bias":
            self.bias = sum(
                a - p for p, a in zip(predicted, actual)
            ) / n
        else:
            mean_x = sum(predicted) / n
            mean_y = sum(actual) / n
            variance = sum((x - mean_x) ** 2 for x in predicted) / n
            if variance == 0.0:
                self.slope = 0.0
                self.intercept = mean_y
            else:
                covariance = sum(
                    (x - mean_x) * (y - mean_y)
                    for x, y in zip(predicted, actual)
                ) / n
                self.slope = covariance / variance
                self.intercept = mean_y - self.slope * mean_x
        self.fitted = True
        return self

    def calibrate(self, stars: float) -> float:
        if not self.fitted:
            raise RuntimeError("calibrator must be fitted before use")
        if self.method == "bias":
            corrected = float(stars) + self.bias
        else:
            corrected = self.intercept + self.slope * float(stars)
        return normalize_stars(corrected)

    def calibrate_many(self, values):
        return [self.calibrate(value) for value in values]

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "bias": self.bias,
            "slope": self.slope,
            "intercept": self.intercept,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ScoreCalibrator":
        calibrator = cls(method=payload.get("method", "linear"))
        calibrator.bias = float(payload.get("bias", 0.0))
        calibrator.slope = float(payload.get("slope", 1.0))
        calibrator.intercept = float(payload.get("intercept", 0.0))
        calibrator.fitted = bool(payload.get("fitted", False))
        return calibrator


def split_pairs(predictions, actuals, validation_ratio: float = 0.5,
                seed: int = 0):
    """Deterministically split paired data into (train, validation)."""
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have the same length")
    indices = list(range(len(predictions)))
    random.Random(seed).shuffle(indices)
    cut = int(len(indices) * (1.0 - validation_ratio))
    train_idx = indices[:cut]
    valid_idx = indices[cut:]
    train = (
        [predictions[i] for i in train_idx],
        [actuals[i] for i in train_idx],
    )
    validation = (
        [predictions[i] for i in valid_idx],
        [actuals[i] for i in valid_idx],
    )
    return train, validation


def rmse(predictions, actuals) -> float:
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have the same length")
    if not predictions:
        raise ValueError("cannot compute RMSE on empty data")
    total = sum(
        (float(p) - float(a)) ** 2 for p, a in zip(predictions, actuals)
    )
    return (total / len(predictions)) ** 0.5


def evaluate_temporal_calibration(records, split_by_index, method="bias"):
    """Fit on the declared validation/calibration split; evaluate test once.

    The train split is reserved for model/prompt choices. The calibration method
    must be selected before calling this helper; it is fitted only on the
    validation split and the test labels are used only for final reporting.
    """
    if method not in ScoreCalibrator.VALID_METHODS:
        raise ValueError(f"unknown calibration method: {method}")
    if not isinstance(split_by_index, dict):
        raise ValueError("split_by_index must map task indexes to split names")
    allowed_splits = {"train", "validation", "test"}

    valid_records = []
    seen_indexes = set()
    for position, record in enumerate(records):
        index = record.get("index", position)
        try:
            index = int(index)
            predicted = float(record["predicted"])
            actual = float(record["actual"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("records need numeric index, predicted, actual") from exc
        if not math.isfinite(predicted) or not math.isfinite(actual):
            raise ValueError("predicted and actual values must be finite")
        if index in seen_indexes:
            raise ValueError("task indexes must be unique")
        seen_indexes.add(index)
        if index not in split_by_index:
            raise ValueError(f"task index {index} has no declared split")
        split = split_by_index[index]
        if split not in allowed_splits:
            raise ValueError(f"unknown temporal split for task index {index}")
        valid_records.append((index, predicted, actual, split))

    calibration = [row for row in valid_records if row[3] == "validation"]
    test = [row for row in valid_records if row[3] == "test"]
    train_count = sum(row[3] == "train" for row in valid_records)
    if len(calibration) < 2:
        raise ValueError("at least two validation rows are required to calibrate")
    if not test:
        raise ValueError("at least one untouched test row is required")

    calibrator = ScoreCalibrator(method=method).fit(
        [row[1] for row in calibration],
        [row[2] for row in calibration],
    )
    raw_predictions = [row[1] for row in test]
    actuals = [row[2] for row in test]
    calibrated_predictions = calibrator.calibrate_many(raw_predictions)

    def summarize(predictions):
        errors = [float(prediction) - actual for prediction, actual in zip(
            predictions, actuals
        )]
        return {
            "rmse": rmse(predictions, actuals),
            "mae": sum(abs(error) for error in errors) / len(errors),
            "mean_signed_bias": sum(errors) / len(errors),
        }

    return {
        "evaluation_protocol": "explicit_temporal_validation_fit_test_only",
        "calibration_method": method,
        "train_rows_not_used_for_calibration": train_count,
        "validation_rows_used_to_fit_calibrator": len(calibration),
        "test_rows_used_only_for_final_evaluation": len(test),
        "raw_test": summarize(raw_predictions),
        "calibrated_test": summarize(calibrated_predictions),
        "calibration_parameters": calibrator.to_dict(),
    }
