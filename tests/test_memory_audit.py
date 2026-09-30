"""Synthetic, no-network tests for privacy-safe audit and temporal manifests."""

import json
import gzip
from pathlib import Path
import subprocess
import sys

import memory_audit
import pytest
from memory_audit import (
    TemporalManifestInputError,
    _load_temporal_review_index,
    audit_dataset,
    audit_run_artifacts,
    build_temporal_manifest,
    iter_records,
    rating_diagnostics,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def build_fixture(root: Path):
    task_dir = root / "tasks"
    truth_dir = root / "groundtruth"
    task_rows = [
        {"type": "user_behavior_simulation", "user_id": "private-user-1",
         "item_id": "private-item-1"},
        {"type": "user_behavior_simulation", "user_id": "private-user-1",
         "item_id": "private-item-2"},
        {"type": "user_behavior_simulation", "user_id": "private-user-1",
         "item_id": "private-item-3"},
    ]
    truth_rows = [
        {"stars": 1.0, "review": "private target review one",
         "timestamp": "2024-01-02T00:00:00Z", "review_id": "target-1"},
        {"stars": 4.0, "review": "private target review two",
         "timestamp": "2024-01-04T00:00:00Z", "review_id": "target-2"},
        {"stars": 5.0, "review": "private target review three",
         "timestamp": "2024-01-06T00:00:00Z", "review_id": "target-3"},
    ]
    for index, (task, truth) in enumerate(zip(task_rows, truth_rows)):
        write_json(task_dir / f"task_{index}.json", task)
        write_json(truth_dir / f"groundtruth_{index}.json", truth)

    review_data = root / "processed" / "reviews.json"
    write_json(review_data, [
        {"review_id": "old-history", "user_id": "private-user-1",
         "item_id": "private-item-9", "date": "2024-01-01T00:00:00Z",
         "stars": 5.0, "text": "private old history"},
        {"review_id": "target-1", "user_id": "private-user-1",
         "item_id": "private-item-1", "date": "2024-01-02T00:00:00Z",
         "stars": 1.0, "text": "private target review one"},
        {"review_id": "item-reference", "user_id": "other-user",
         "item_id": "private-item-1", "date": "2024-01-01T12:00:00Z",
         "stars": 4.0, "text": "private item reference"},
        {"review_id": "future-item", "user_id": "future-user",
         "item_id": "private-item-1", "date": "2024-01-03T00:00:00Z",
         "stars": 5.0, "text": "private future item review"},
        {"review_id": "target-2", "user_id": "private-user-1",
         "item_id": "private-item-2", "date": "2024-01-04T00:00:00Z",
         "stars": 4.0, "text": "private target review two"},
        {"review_id": "target-3", "user_id": "private-user-1",
         "item_id": "private-item-3", "date": "2024-01-06T00:00:00Z",
         "stars": 5.0, "text": "private target review three"},
        {"review_id": "future-user", "user_id": "private-user-1",
         "item_id": "private-item-8", "date": "2024-01-07T00:00:00Z",
         "stars": 2.0, "text": "private future user review"},
    ])
    return task_dir, truth_dir, review_data


def test_dataset_audit_counts_repeat_opportunities_and_hides_text(tmp_path):
    task_dir, truth_dir, review_data = build_fixture(tmp_path)

    result = audit_dataset(task_dir, truth_dir, [review_data])
    rendered = json.dumps(result)

    assert result["task_sequence"]["tasks_after_first_user_occurrence"] == 2
    assert result["task_sequence"]["sequential_potential_recall_tasks"] == 2
    assert result["task_sequence"]["sequential_recall_tasks_with_memory_limit_8"] == 2
    assert result["leakage_evidence"]["target_text_found_in_same_user_source_rows"] == 3
    assert result["leakage_evidence"]["target_text_found_in_target_item_source_rows"] == 3
    assert result["leakage_evidence"]["tasks_with_same_user_reviews_at_or_after_target_time"] == 3
    assert "private-user-1" not in rendered
    assert "private target review one" not in rendered
    assert str(task_dir) not in rendered


def test_manifest_excludes_heldout_targets_and_future_reviews(tmp_path):
    task_dir, truth_dir, review_data = build_fixture(tmp_path)

    manifest = build_temporal_manifest(
        task_dir, truth_dir, [review_data],
        train_ratio=0.34, validation_ratio=0.33,
    )

    assert manifest["eligible_task_count"] == 3
    assert manifest["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    rows = manifest["tasks"]
    assert [row["task_index"] for row in rows] == [0, 1, 2]
    assert [row["user_repeat_stratum"] for row in rows] == [
        "first_seen", "repeat", "repeat"
    ]
    assert rows[0]["visible_history_row_indexes"] == [0]
    assert rows[0]["target_item_reference_row_indexes"] == [2]
    assert all(row["target_text_absent_from_context"] for row in rows)
    assert all("timestamp" not in row and "stars" not in row for row in rows)
    rendered = json.dumps(manifest)
    assert "private-user-1" not in rendered
    assert "private target review one" not in rendered


def test_manifest_cli_pairing_conflict_exits_nonzero_without_private_details(
    tmp_path,
):
    task_dir = tmp_path / "tasks"
    truth_dir = tmp_path / "groundtruth"
    write_json(task_dir / "0.json", {
        "user_id": "private-task-user", "item_id": "private-item",
        "timestamp": "2024-01-01T00:00:00Z",
    })
    write_json(truth_dir / "0.json", {
        "user_id": "private-label-user", "item_id": "private-item",
        "timestamp": "2024-01-01T00:00:00Z", "stars": 4,
    })
    reviews = tmp_path / "reviews.json"
    write_json(reviews, [])

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "memory_audit.py"),
            "manifest",
            "--task-dir", str(task_dir),
            "--groundtruth-dir", str(truth_dir),
            "--review-data", str(reviews),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "task_groundtruth_pairing_unreliable" in completed.stderr
    assert "private-task-user" not in completed.stderr
    assert "private-label-user" not in completed.stderr
    assert str(tmp_path) not in completed.stderr


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("task_directory_missing", "task_or_groundtruth_source_missing"),
        ("review_source_missing", "review_source_missing"),
        ("multi_record_alignment", "task_groundtruth_pairing_unreliable"),
    ],
)
def test_manifest_cli_fails_closed_on_missing_or_unreliable_sources(
    tmp_path, failure, expected_code,
):
    task_dir = tmp_path / "tasks"
    truth_dir = tmp_path / "groundtruth"
    review_file = tmp_path / "reviews.json"
    if failure == "task_directory_missing":
        write_json(truth_dir / "0.json", {"stars": 4})
        write_json(review_file, [])
    elif failure == "review_source_missing":
        write_json(task_dir / "0.json", {"user_id": "u", "item_id": "i"})
        write_json(truth_dir / "0.json", {"stars": 4})
    else:
        write_json(task_dir / "batch.json", [
            {"user_id": "u", "item_id": "i0"},
            {"user_id": "u", "item_id": "i1"},
        ])
        write_json(truth_dir / "batch.json", [{"stars": 4}, {"stars": 5}])
        write_json(review_file, [])
    review_argument = (
        tmp_path / "missing-reviews.json"
        if failure == "review_source_missing" else review_file
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "memory_audit.py"),
            "manifest",
            "--task-dir", str(task_dir),
            "--groundtruth-dir", str(truth_dir),
            "--review-data", str(review_argument),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert expected_code in completed.stderr
    assert str(tmp_path) not in completed.stderr


def test_zero_eligible_manifest_is_audit_only_not_a_provider_input(
    tmp_path, capsys,
):
    task_dir = tmp_path / "tasks"
    truth_dir = tmp_path / "groundtruth"
    write_json(task_dir / "0.json", {"user_id": "u", "item_id": "i"})
    write_json(truth_dir / "0.json", {
        "review_id": "missing-target", "timestamp": "2024-01-02T00:00:00Z",
        "stars": 4,
    })
    reviews = tmp_path / "reviews.json"
    write_json(reviews, [{
        "review_id": "old", "user_id": "u", "item_id": "old-item",
        "date": "2024-01-01T00:00:00Z", "stars": 4, "text": "old review",
    }])

    result = memory_audit.main([
        "manifest", "--task-dir", str(task_dir), "--groundtruth-dir",
        str(truth_dir), "--review-data", str(reviews),
    ])
    rendered = capsys.readouterr().out
    audit = json.loads(rendered)

    assert result == 0
    assert audit["eligible_task_count"] == 0
    assert audit["manifest_audit_status"] == "audit_only_no_eligible_tasks"
    assert audit["runtime_eligible"] is False
    assert audit["runtime_rejection_reasons"] == ["no_eligible_tasks"]


def test_temporal_source_byte_limit_is_checked_before_parsing(tmp_path, monkeypatch):
    source = tmp_path / "reviews.json"
    source.write_text("not parsed", encoding="utf-8")
    monkeypatch.setattr(memory_audit, "TEMPORAL_MAX_REVIEW_SOURCE_BYTES", 2)

    def parser_must_not_run(_source):
        raise AssertionError("oversized source was parsed before preflight")

    monkeypatch.setattr(memory_audit, "iter_records", parser_must_not_run)
    with pytest.raises(TemporalManifestInputError) as caught:
        _load_temporal_review_index(
            [source], relevant_users=set(), relevant_items=set()
        )
    assert caught.value.code == "review_source_byte_limit_exceeded"


def test_compressed_temporal_source_is_bounded_by_decoded_size_before_parsing(
    tmp_path, monkeypatch,
):
    source = tmp_path / "reviews.json.gz"
    with gzip.open(source, "wb") as handle:
        handle.write(b" " * 4096)
    assert source.stat().st_size < 1024
    monkeypatch.setattr(memory_audit, "TEMPORAL_MAX_REVIEW_SOURCE_BYTES", 1024)

    def parser_must_not_run(_source):
        raise AssertionError("oversized decoded source was parsed")

    monkeypatch.setattr(memory_audit, "iter_records", parser_must_not_run)
    with pytest.raises(TemporalManifestInputError) as caught:
        _load_temporal_review_index(
            [source], relevant_users=set(), relevant_items=set()
        )
    assert caught.value.code == "review_source_decoded_byte_limit_exceeded"


def test_temporal_source_record_limit_stops_bounded_metadata_retention(
    tmp_path, monkeypatch,
):
    source = tmp_path / "reviews.json"
    write_json(source, [
        {"user_id": "u", "item_id": "i", "text": "one"},
        {"user_id": "u", "item_id": "i", "text": "two"},
    ])
    monkeypatch.setattr(memory_audit, "TEMPORAL_MAX_REVIEW_ROWS", 1)

    with pytest.raises(TemporalManifestInputError) as caught:
        _load_temporal_review_index(
            [source], relevant_users={"u"}, relevant_items={"i"}
        )
    assert caught.value.code == "review_source_record_limit_exceeded"


def test_missing_time_marks_rows_ineligible_instead_of_guessing(tmp_path):
    task_dir = tmp_path / "tasks"
    truth_dir = tmp_path / "groundtruth"
    write_json(task_dir / "task_0.json", {"user_id": "u", "item_id": "i"})
    write_json(truth_dir / "groundtruth_0.json", {"stars": 4, "review": "x"})
    review_file = tmp_path / "reviews.json"
    write_json(review_file, [{"user_id": "u", "item_id": "i", "text": "history"}])

    result = build_temporal_manifest(task_dir, truth_dir, [review_file])

    assert result["eligible_task_count"] == 0
    assert result["ineligible_task_count"] == 1
    assert result["tasks"] == []


def test_unmatched_target_is_not_eligible_for_strict_temporal_manifest(tmp_path):
    task_dir = tmp_path / "tasks"
    truth_dir = tmp_path / "groundtruth"
    write_json(task_dir / "task_0.json", {"user_id": "u", "item_id": "i"})
    write_json(truth_dir / "groundtruth_0.json", {
        "review_id": "missing-target", "stars": 4,
        "timestamp": "2024-01-02T00:00:00Z", "review": "hidden target",
    })
    review_file = tmp_path / "reviews.json"
    write_json(review_file, [{
        "review_id": "old", "user_id": "u", "item_id": "i",
        "date": "2024-01-01T00:00:00Z", "text": "old history",
    }])

    result = build_temporal_manifest(task_dir, truth_dir, [review_file])

    assert result["eligible_task_count"] == 0
    assert result["ineligible_reasons"]["target_review_not_uniquely_matched"] == 1


def test_target_row_without_its_own_event_time_is_not_temporally_eligible(tmp_path):
    task_dir = tmp_path / "tasks"
    truth_dir = tmp_path / "groundtruth"
    write_json(task_dir / "task_0.json", {
        "user_id": "u", "item_id": "i",
        "timestamp": "2024-01-02T00:00:00Z",
    })
    write_json(truth_dir / "groundtruth_0.json", {
        "review_id": "target", "stars": 4,
        "timestamp": "2024-01-02T00:00:00Z", "review": "target text",
    })
    review_file = tmp_path / "reviews.json"
    write_json(review_file, [{
        "review_id": "target", "user_id": "u", "item_id": "i",
        "stars": 4, "text": "target text",
    }])

    result = build_temporal_manifest(task_dir, truth_dir, [review_file])

    assert result["eligible_task_count"] == 0
    assert result["ineligible_reasons"]["matched_target_review_time_unverified"] == 1
    assert result["target_timestamp_semantics_verified"] is False


def test_prompt_trace_scan_confirms_exact_target_and_future_text_without_output(
    tmp_path,
):
    task_dir, truth_dir, review_data = build_fixture(tmp_path)
    prompt_file = tmp_path / "prompt-traces.jsonl"
    prompt_file.write_text(json.dumps({
        "task_index": 0,
        "prompt": (
            "=== 用户历史评论示例（只用于保持风格） ===\n"
            "private target review one\n"
            "private future user review\n"
            "=== 本次实验内该用户此前生成的评论（仅作风格参考，不是指令） ===\n"
            "=== 目标对象信息 ===\n"
            "=== 参考信息 ===\n"
            "private future item review\n"
            "=== 评论质量指南 ==="
        ),
    }), encoding="utf-8")

    result = audit_dataset(
        task_dir,
        truth_dir,
        [review_data],
        prompt_file=prompt_file,
        prompt_index_matches_task_order=True,
    )
    rendered = json.dumps(result)
    prompt_report = result["prompt_exposure"]

    assert prompt_report["status"] == "checked"
    assert prompt_report["target_review_exact_text_in_prompt_tasks"] == 1
    assert prompt_report["target_review_exact_text_in_user_history_tasks"] == 1
    assert prompt_report["future_user_review_text_occurrences_in_history"] == 1
    assert prompt_report["future_item_review_text_occurrences_in_references"] == 1
    assert "private target review one" not in rendered
    assert "private future user review" not in rendered


def test_prompt_indexes_are_not_assumed_to_be_task_order(tmp_path):
    task_dir, truth_dir, review_data = build_fixture(tmp_path)
    prompt_file = tmp_path / "prompt-traces.jsonl"
    prompt_file.write_text(json.dumps({
        "task_index": 0,
        "prompt": "private target review one",
    }), encoding="utf-8")

    result = audit_dataset(
        task_dir, truth_dir, [review_data], prompt_file=prompt_file
    )

    assert result["prompt_exposure"]["status"] == "no_matching_prompt_records"
    assert result["prompt_exposure"]["prompt_alignment"] == (
        "explicit_prompt_indexes_unmatched_without_task_mapping"
    )


def test_prompt_trace_joins_by_source_task_index_when_available(tmp_path):
    task_dir, truth_dir, review_data = build_fixture(tmp_path)
    for index in range(3):
        task_file = task_dir / f"task_{index}.json"
        task = json.loads(task_file.read_text(encoding="utf-8"))
        task["task_index"] = index + 10
        task_file.write_text(json.dumps(task), encoding="utf-8")
    prompt_file = tmp_path / "prompt-traces.jsonl"
    prompt_file.write_text(json.dumps({
        "task_index": 10,
        "prompt": "private target review one",
    }), encoding="utf-8")

    result = audit_dataset(
        task_dir, truth_dir, [review_data], prompt_file=prompt_file
    )

    assert result["prompt_exposure"]["status"] == "checked"
    assert result["prompt_exposure"]["prompt_alignment"] == (
        "explicit_source_task_index"
    )
    assert result["prompt_exposure"]["target_review_exact_text_in_prompt_tasks"] == 1


def test_json_array_and_json_lines_yield_records_incrementally(tmp_path):
    array_file = tmp_path / "rows.json"
    array_file.write_text(
        '[{"user_id":"u1","item_id":"i1"},'
        '{"user_id":"u2","item_id":"i2"}]', encoding="utf-8"
    )
    lines_file = tmp_path / "rows.jsonl"
    lines_file.write_text(
        '{"user_id":"u3","item_id":"i3"}\n'
        '{"user_id":"u4","item_id":"i4"}\n', encoding="utf-8"
    )

    assert len(list(iter_records(array_file))) == 2
    assert len(list(iter_records(lines_file))) == 2


def test_run_audit_separates_recall_from_prompt_and_hides_legacy_ids():
    report = audit_run_artifacts(
        full_diagnostics=[{
            "task_index": 1,
            "execution_order": 1,
            "user_id": "private-user-1",
            "memory_candidate_count": 2,
            "memory_recalled_count": 1,
            "memory_recalled_sequences": [1],
            "memory_recalled_origin_orders": [0],
            "memory_prompt_entry_count": 1,
            "memory_prompt_sequences": [1],
            "memory_sequence": 3,
            "user_repeat_stratum": "repeat",
        }],
        full_records=[
            {"index": 0, "predicted": 3.0, "actual": 4.0, "error": 1.0},
            {"index": 1, "predicted": 4.0, "actual": 4.0, "error": 0.0},
        ],
        no_memory_records=[
            {"index": 0, "predicted": 3.0, "actual": 4.0, "error": 1.0},
            {"index": 1, "predicted": 2.0, "actual": 4.0, "error": 2.0},
        ],
        no_memory_diagnostics=[{
            "task_index": 1,
            "memory_candidate_count": 0,
            "memory_recalled_count": 0,
            "memory_prompt_entry_count": 0,
        }],
        source_task_index_matches_record_index=True,
        paired_task_indexes_confirmed=True,
        n_boot=300,
    )
    rendered = json.dumps(report)

    assert report["memory_cases"][0]["candidate_count"] == 2
    assert report["memory_cases"][0]["recalled_count"] == 1
    assert report["memory_cases"][0]["prompt_entry_count"] == 1
    assert report["memory_cases"][0]["full_absolute_error"] == 0.0
    assert report["memory_cases"][0]["no_memory_absolute_error"] == 2.0
    assert report["memory_cases"][0]["no_memory_candidate_count"] == 0
    assert report["memory_cases"][0]["no_memory_prompt_entry_count"] == 0
    assert report["sequence_order_check"]["all_recalled_sequences_precede_current_write"]
    assert report["sequence_order_check"]["checked_memory_origin_orders"] == 1
    assert report["sequence_order_check"]["future_or_same_order_generated_memory_reads"] == 0
    assert report["paired_error_bootstrap"]["n_paired"] == 2
    assert "private-user-1" not in rendered


def test_rating_diagnostics_group_bias_by_truth_and_prediction():
    report = audit_run_artifacts(
        full_diagnostics=[],
        full_records=[
            {"index": 0, "predicted": 5.0, "actual": 1.0, "error": 4.0},
            {"index": 1, "predicted": 4.0, "actual": 4.0, "error": 0.0},
        ],
    )

    summary = report["rating_diagnostics_full"]
    assert summary["mean_signed_bias"] == 2.0
    assert summary["by_true_rating"]["1.0"]["mean_signed_bias"] == 4.0
    assert summary["prediction_counts_1_to_5"]["5"] == 1
    assert summary["mean_signed_bias_ci95"] is not None


def test_rating_bias_intervals_use_user_clusters_when_groups_are_complete():
    result = rating_diagnostics(
        [
            {"index": 0, "predicted": 4.0, "actual": 3.0},
            {"index": 1, "predicted": 5.0, "actual": 4.0},
            {"index": 2, "predicted": 2.0, "actual": 3.0},
        ],
        user_groups={0: "U1", 1: "U1", 2: "U2"},
        user_repeat_strata={0: "first_seen", 1: "repeat", 2: "first_seen"},
        n_boot=300,
    )

    assert result["uncertainty_resampling_unit"] == "user_cluster"
    assert result["by_user_repeat_stratum"]["first_seen"]["n"] == 2
    assert result["by_user_repeat_stratum"]["repeat"]["n"] == 1


def test_rating_diagnostics_never_infers_first_seen_from_score_index():
    rows = [
        {"index": 0, "predicted": 4.0, "actual": 3.0},
        {"index": 1, "predicted": 5.0, "actual": 4.0},
    ]
    groups = {0: "U1", 1: "U1"}

    without_order = rating_diagnostics(rows, user_groups=groups, n_boot=200)
    assert "by_user_repeat_stratum" not in without_order

    verified = rating_diagnostics(
        rows, user_groups=groups,
        user_repeat_strata={0: "repeat", 1: "first_seen"}, n_boot=200,
    )
    assert verified["by_user_repeat_stratum"]["first_seen"]["n"] == 1
    assert verified["by_user_repeat_stratum"]["repeat"]["n"] == 1


def test_run_audit_flags_memory_written_by_a_later_execution():
    report = audit_run_artifacts(
        full_diagnostics=[{
            "execution_order": 0,
            "memory_recalled_count": 1,
            "memory_recalled_sequences": [1],
            "memory_recalled_origin_orders": [1],
            "memory_sequence": 2,
            "memory_prompt_entry_count": 1,
        }],
        full_records=[],
    )

    assert report["sequence_order_check"][
        "future_or_same_order_generated_memory_reads"
    ] == 1


def test_run_audit_does_not_pair_indexes_without_explicit_confirmation():
    report = audit_run_artifacts(
        full_diagnostics=[{
            "task_index": 1,
            "execution_order": 1,
            "memory_recalled_count": 1,
            "memory_prompt_entry_count": 1,
        }],
        full_records=[{"index": 1, "predicted": 4.0, "actual": 3.0, "error": 1.0}],
        no_memory_records=[{"index": 1, "predicted": 3.0, "actual": 3.0, "error": 0.0}],
    )

    assert report["memory_cases"][0]["record_index"] is None
    assert report["memory_cases"][0]["full_prediction"] is None
    assert report["paired_error_bootstrap"]["status"] == "not_computed"


def test_run_audit_task_bootstrap_resamples_tasks_without_user_mapping():
    report = audit_run_artifacts(
        full_diagnostics=[],
        full_records=[
            {"index": 0, "actual": 3.0, "error": 0.0},
            {"index": 1, "actual": 3.0, "error": 2.0},
        ],
        no_memory_records=[
            {"index": 0, "actual": 3.0, "error": 1.0},
            {"index": 1, "actual": 3.0, "error": 1.0},
        ],
        paired_task_indexes_confirmed=True,
        n_boot=300,
    )

    paired = report["paired_error_bootstrap"]
    assert paired["resampling_unit"] == "task"
    assert paired["mean_difference"] == 0.0
    assert paired["ci95_low"] < 0.0 < paired["ci95_high"]
