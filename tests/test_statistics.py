"""Offline tests for metrics, per-task records and paired bootstrap testing."""

import sys
import types

import pytest

from comprehensive_evaluation import (
    build_per_task_records,
    calculate_additional_metrics,
    calculate_calibrated_metrics,
    ExperimentConfig,
    ExperimentRunner,
    extract_prediction,
    paired_bootstrap_test,
)


def make_record(error):
    return {"error": error, "squared_error": error ** 2}


class TestPerTaskRecords:
    def test_additional_metrics_include_rmse_and_mae(self):
        metrics = calculate_additional_metrics(
            [{"stars": 4.0}, {"stars": 2.0}],
            [{"stars": 3.0}, {"stars": 1.0}],
        )

        assert metrics["rmse"] == pytest.approx(1.0)
        assert metrics["mae"] == pytest.approx(1.0)
        assert metrics["num_valid_predictions"] == 2

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

    def test_calibration_reports_held_out_metrics(self):
        records = build_per_task_records(
            [{"stars": 3.0}, {"stars": 3.0}, {"stars": 4.0}, {"stars": 4.0}],
            [{"stars": 4.0}, {"stars": 4.0}, {"stars": 5.0}, {"stars": 5.0}],
        )

        metrics = calculate_calibrated_metrics(
            records, method="bias", validation_ratio=0.5, seed=0
        )

        assert metrics["calibration_method"] == "bias"
        assert metrics["calibration_train_size"] == 2
        assert metrics["calibration_validation_size"] == 2
        assert "calibrated_rmse" in metrics

    def test_framework_evaluation_failure_keeps_project_metrics(
        self, monkeypatch, tmp_path
    ):
        class FailingSimulator:
            def __init__(self, **_kwargs):
                self.groundtruth_data = []

            def set_task_and_groundtruth(self, **_kwargs):
                self.groundtruth_data = [{"stars": 3.0}]

            def set_agent(self, _agent_class):
                pass

            def set_llm(self, _llm):
                pass

            def run_simulation(self, **_kwargs):
                return [{"output": {"stars": 4.0}}]

            def evaluate(self):
                raise ZeroDivisionError("division by zero")

        fake_framework = types.ModuleType("websocietysimulator")
        fake_framework.Simulator = FailingSimulator
        monkeypatch.setitem(sys.modules, "websocietysimulator", fake_framework)

        runner = ExperimentRunner(
            data_dir="Dataset",
            task_set="yelp",
            api_key="test-key",
            num_tasks=1,
            max_workers=1,
            output_dir=str(tmp_path / "results"),
        )
        result = runner.run_experiment(
            ExperimentConfig(
                name="Deterministic",
                enable_reflection=False,
                use_memory=False,
                max_reference_reviews=0,
                agent_kind="deterministic",
            )
        )

        assert result["rmse"] == pytest.approx(1.0)
        assert result["mae"] == pytest.approx(1.0)
        assert "error" not in result
        assert result["evaluation_warning"] == "division by zero"


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


class TestResultsAnalyzer:
    def test_statistical_analysis_handles_empty_results(self):
        from comprehensive_evaluation import ResultsAnalyzer

        analyzer = ResultsAnalyzer({})

        analysis = analyzer.generate_statistical_analysis()

        assert "没有可用" in analysis

    def test_statistical_analysis_lists_failed_configs(self):
        from comprehensive_evaluation import ResultsAnalyzer

        analyzer = ResultsAnalyzer({"Baseline": {"error": "boom"}})

        analysis = analyzer.generate_statistical_analysis()

        assert "Baseline" in analysis

    def test_full_report_survives_all_failed_results(self, tmp_path):
        from comprehensive_evaluation import ResultsAnalyzer

        analyzer = ResultsAnalyzer({"Baseline": {"error": "boom"}})

        report_path = analyzer.save_full_report(
            filename=str(tmp_path / "report.md")
        )

        assert (tmp_path / "report.md").exists()
        assert report_path == str(tmp_path / "report.md")
