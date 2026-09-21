"""Offline tests for the score calibration module."""

import pytest

from score_calibration import (
    ScoreCalibrator,
    rmse,
    split_pairs,
)


class TestScoreCalibrator:
    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError):
            ScoreCalibrator(method="magic")

    def test_calibrate_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            ScoreCalibrator().calibrate(3.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ScoreCalibrator().fit([1.0, 2.0], [1.0])

    def test_empty_fit_raises(self):
        with pytest.raises(ValueError):
            ScoreCalibrator().fit([], [])

    def test_bias_method_corrects_systematic_shift(self):
        calibrator = ScoreCalibrator(method="bias")
        calibrator.fit([3, 3, 3, 3], [4, 4, 4, 4])
        assert calibrator.calibrate(3) == 4.0

    def test_linear_method_recovers_identity(self):
        calibrator = ScoreCalibrator(method="linear")
        calibrator.fit([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert calibrator.calibrate(3) == 3.0
        assert calibrator.calibrate(5) == 5.0

    def test_linear_method_learns_offset(self):
        calibrator = ScoreCalibrator(method="linear")
        calibrator.fit([1, 2, 3, 4], [2, 3, 4, 5])
        assert calibrator.calibrate(2) == 3.0

    def test_constant_predictions_fall_back_to_mean(self):
        calibrator = ScoreCalibrator(method="linear")
        calibrator.fit([3, 3, 3], [2, 4, 3])
        assert calibrator.slope == 0.0
        assert calibrator.calibrate(3) == 3.0

    def test_output_is_snapped_and_clamped(self):
        calibrator = ScoreCalibrator(method="bias")
        calibrator.fit([1, 1, 1], [5, 5, 5])
        assert calibrator.calibrate(1) == 5.0
        assert calibrator.calibrate(5) == 5.0

    def test_serialization_roundtrip(self):
        calibrator = ScoreCalibrator(method="linear")
        calibrator.fit([1, 2, 3], [2, 3, 4])
        restored = ScoreCalibrator.from_dict(calibrator.to_dict())
        assert restored.calibrate(2) == calibrator.calibrate(2)


class TestSplitAndRmse:
    def test_split_is_deterministic_for_a_seed(self):
        predictions = list(range(10))
        actuals = list(range(10))
        first = split_pairs(predictions, actuals, validation_ratio=0.5, seed=7)
        second = split_pairs(predictions, actuals, validation_ratio=0.5, seed=7)
        assert first == second

    def test_split_sizes(self):
        train, validation = split_pairs(
            list(range(10)), list(range(10)), validation_ratio=0.4, seed=0
        )
        assert len(train[0]) == 6
        assert len(validation[0]) == 4

    def test_rmse_known_value(self):
        assert rmse([1, 2], [2, 2]) == 0.5 ** 0.5

    def test_calibration_reduces_rmse_on_shifted_data(self):
        predictions = [2, 3, 4, 3, 2, 4, 3, 2]
        actuals = [3, 4, 5, 4, 3, 5, 4, 3]
        train, validation = split_pairs(
            predictions, actuals, validation_ratio=0.5, seed=1
        )
        calibrator = ScoreCalibrator(method="bias").fit(*train)

        raw_rmse = rmse(*validation)
        calibrated_rmse = rmse(calibrator.calibrate_many(validation[0]), validation[1])

        assert calibrated_rmse < raw_rmse
