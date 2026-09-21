"""Offline tests for the evaluation CLI.

These tests never import websocietysimulator and never call the LLM API.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from comprehensive_evaluation import build_parser, default_experiments, main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.data_dir == "Dataset"
    assert args.task_set == "yelp"
    assert args.num_tasks == 10
    assert args.max_workers == 5
    assert args.output_dir == "results"
    assert args.seed is None
    assert args.dry_run is False


def test_default_experiment_names():
    names = [config.name for config in default_experiments()]
    assert names == [
        "Deterministic",
        "Baseline",
        "No_Context",
        "Full",
        "No_Reflection",
        "No_Memory",
        "Fewer_References",
    ]


def test_dry_run_succeeds_without_api_key(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    output_dir = tmp_path / "results"

    exit_code = main([
        "--dry-run",
        "--task-set", "amazon",
        "--num-tasks", "3",
        "--experiment", "Baseline", "No_Memory",
        "--output-dir", str(output_dir),
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "DRY RUN" in captured.out
    assert "amazon" in captured.out
    assert "Baseline" in captured.out
    assert "No_Memory" in captured.out
    assert "No_Reflection" not in captured.out
    assert not output_dir.exists()


def test_unknown_experiment_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        main(["--dry-run", "--experiment", "No_Such_Config"])
    assert excinfo.value.code == 2


def test_missing_api_key_returns_error(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    output_dir = tmp_path / "results"

    exit_code = main(["--num-tasks", "1", "--output-dir", str(output_dir)])

    assert exit_code == 2
    assert "no API key" in capsys.readouterr().out
    assert not output_dir.exists()


def test_cli_dry_run_runs_as_script():
    result = subprocess.run(
        [
            sys.executable,
            "comprehensive_evaluation.py",
            "--dry-run",
            "--num-tasks", "2",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
