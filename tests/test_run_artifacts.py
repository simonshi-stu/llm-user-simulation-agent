"""Offline tests for run metadata and per-task persistence."""

import json
from pathlib import Path

import pytest

from comprehensive_evaluation import (
    ExperimentConfig,
    ExperimentRunner,
    build_per_task_records,
)


def make_runner(tmp_path, **overrides):
    kwargs = dict(
        data_dir="Dataset",
        task_set="yelp",
        api_key="test-key",
        num_tasks=5,
        max_workers=2,
        output_dir=str(tmp_path / "results"),
        chat_model="deepseek-chat",
        base_url="https://example.invalid",
        seed=42,
    )
    kwargs.update(overrides)
    return ExperimentRunner(**kwargs)


def test_run_id_and_run_dir(tmp_path):
    runner = make_runner(tmp_path)
    assert runner.run_id
    assert runner.run_dir.startswith(str(tmp_path / "results"))


def test_build_run_metadata_is_complete(tmp_path):
    runner = make_runner(tmp_path, seed=7)
    configs = [
        ExperimentConfig("A", False, False, 3),
        ExperimentConfig("B", True, True, 5),
    ]
    metadata = runner.build_run_metadata(configs)
    assert metadata["seed"] == 7
    assert metadata["task_set"] == "yelp"
    assert metadata["num_tasks"] == 5
    assert metadata["experiments"] == ["A", "B"]
    assert metadata["run_id"] == runner.run_id


def test_save_per_task_records_writes_into_run_dir(tmp_path):
    runner = make_runner(tmp_path)
    records = build_per_task_records([{"stars": 4.0}], [{"stars": 3.0}])

    runner._save_per_task_records("Baseline", records)

    path = Path(runner.run_dir) / "per_task_Baseline.json"
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[0]["error"] == 1.0


def test_extract_groundtruths_prefers_known_attributes(tmp_path):
    runner = make_runner(tmp_path)

    class FakeSimulator:
        groundtruth_pool = [{"stars": index} for index in range(10)]

    groundtruths = runner._extract_groundtruths(FakeSimulator())
    assert len(groundtruths) == 5


def test_compare_uses_paired_records(tmp_path):
    runner = make_runner(tmp_path)
    runner.per_task_records["A"] = [{"error": 0.1}] * 20
    runner.per_task_records["B"] = [{"error": 1.0}] * 20

    result = runner.compare("A", "B", metric="error")

    assert result["mean_difference"] == pytest.approx(-0.9)
    assert result["p_value"] < 0.05
