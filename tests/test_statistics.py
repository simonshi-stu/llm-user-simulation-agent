"""Offline tests for per-task records and paired bootstrap testing."""

import pytest

from comprehensive_evaluation import (
    build_per_task_records,
    extract_prediction,
    paired_bootstrap_test,
)


def make_record(error):
    return {"error": error, "squared_error": error ** 2}


class TestPerTaskRecords:
    def test_extract_prediction_handles_nested_output(self):
        assert extract_prediction({"output": {"stars": 4.0}}) == 4.0
        assert extract_prediction({"stars": 3}) == 3.0
        assert extract_prediction({"output": {"review": "x"}}) is None
        assert extract_prediction(None) is None

    def test_records_pair_predictions_and_groundtruth(self):
        records = build_per_task_records(
            [{"stars": 4.0}, {"stars": 2.0}],
            [{"stars": 3.0}, {"stars": 2.0}],
        )
        assert records[0]["error"] == 1.0
        assert records[0]["squared_error"] == 1.0
        assert records[1]["error"] == 0.0

    def test_missing_prediction_is_marked(self):
        records = build_per_task_records([None], [{"stars": 3.0}])
        assert records[0]["predicted"] is None
        assert records[0]["error"] is None
        assert records[0]["squared_error"] is None


class TestPairedBootstrap:
    def test_identical_records_show_no_effect(self):
        records = [make_record(1.0) for _ in range(10)]
        result = paired_bootstrap_test(records, records, n_boot=500, seed=0)
        assert result["mean_difference"] == 0.0
        assert result["p_value"] == 1.0
        assert result["ci95_low"] <= 0.0 <= result["ci95_high"]

    def test_clear_improvement_is_significant(self):
        better = [make_record(0.2) for _ in range(30)]
        worse = [make_record(1.0) for _ in range(30)]
        result = paired_bootstrap_test(better, worse, n_boot=2000, seed=0)
        assert result["mean_difference"] == pytest.approx(-0.8)
        assert result["p_value"] < 0.05

    def test_noisy_small_effect_is_not_significant(self):
        better = [make_record(0.9 if i % 2 else 1.1) for i in range(40)]
        worse = [make_record(1.0) for _ in range(40)]
        result = paired_bootstrap_test(better, worse, n_boot=2000, seed=0)
        assert result["p_value"] > 0.05

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap_test(
                [make_record(1.0)],
                [make_record(1.0), make_record(2.0)],
            )

    def test_empty_records_raise(self):
        with pytest.raises(ValueError):
            paired_bootstrap_test([], [])

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap_test(
                [{"error": 1.0}], [{"error": 1.0}], metric="missing"
            )
