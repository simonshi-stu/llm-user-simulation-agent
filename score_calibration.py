"""Score calibration for star-rating predictions.

The calibrator learns a mapping from raw predicted stars to observed stars
on a validation split. It is dependency-free and deterministic so it can be
unit-tested offline.
"""

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
