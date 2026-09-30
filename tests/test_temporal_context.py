"""Synthetic temporal execution contract; no Simulator, API, or real data."""

import json
from pathlib import Path

import pytest

from memory_audit import _load_task_pairs, build_temporal_manifest, main as audit_main
from temporal_context import (
    TEMPORAL_ABLATION_CONTRACTS,
    TemporalAblationMode,
    TemporalContextProvider,
    TemporalManifestError,
    run_temporal_ablation_suite,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def review(review_id, user_id, item_id, date, stars, text):
    return {
        "review_id": review_id,
        "user_id": user_id,
        "item_id": item_id,
        "date": f"2024-01-{date:02d}T00:00:00Z",
        "stars": stars,
        "text": text,
        "useful": 1,
        "funny": 0,
        "cool": 0,
    }


def make_temporal_fixture(root: Path):
    task_dir = root / "tasks"
    truth_dir = root / "groundtruth"
    tasks = [
        {"user_id": "user-A", "item_id": "item-X", "task_index": 0},
        # A second task for the same user at the exact same event time proves
        # that source-order tie-breaking does not make peer output causal memory.
        {"user_id": "user-A", "item_id": "item-X", "task_index": 1},
        {"user_id": "user-A", "item_id": "item-Y", "task_index": 2},
        {"user_id": "user-B", "item_id": "item-Y", "task_index": 3},
        {"user_id": "user-C", "item_id": "item-Z", "task_index": 4},
        {"user_id": "user-C", "item_id": "item-W", "task_index": 5},
    ]
    truths = [
        {"stars": 1, "review_id": "target-A0", "timestamp": "2024-01-02T00:00:00Z",
         "review": "GROUNDTRUTH_LABEL_A0"},
        {"stars": 5, "review_id": "target-B0", "timestamp": "2024-01-02T00:00:00Z",
         "review": "GROUNDTRUTH_LABEL_B0"},
        {"stars": 2, "review_id": "target-A1", "timestamp": "2024-01-04T00:00:00Z",
         "review": "GROUNDTRUTH_LABEL_A1"},
        {"stars": 4, "review_id": "target-B1", "timestamp": "2024-01-05T00:00:00Z",
         "review": "GROUNDTRUTH_LABEL_B1"},
        {"stars": 3, "review_id": "ambiguous-target", "timestamp": "2024-01-07T00:00:00Z",
         "review": "GROUNDTRUTH_AMBIGUOUS"},
        {"stars": 4, "review_id": "target-C0", "timestamp": "2024-01-08T00:00:00Z",
         "review": "GROUNDTRUTH_LABEL_C0"},
    ]
    for index, (task, truth) in enumerate(zip(tasks, truths)):
        # Matching stems make the source task/label pairing explicit.
        write_json(task_dir / f"{index}.json", task)
        write_json(truth_dir / f"{index}.json", truth)

    reviews = [
        review("history-A", "user-A", "old-item-A", 1, 5, "HISTORY_A_OLD " * 8),
        review("history-B", "user-B", "old-item-B", 1, 2, "HISTORY_B_OLD " * 8),
        review("reference-X-old", "user-C", "item-X", 1, 4, "REFERENCE_X_OLD " * 8),
        # Same timestamp as tasks 0/1: must not be visible as an item reference.
        review("reference-X-same-time", "user-C", "item-X", 2, 4,
               "REFERENCE_X_SAME_TIME " * 8),
        review("target-A0", "user-A", "item-X", 2, 1, "TARGET_SOURCE_A0 " * 8),
        review("target-B0", "user-A", "item-X", 2, 5, "TARGET_SOURCE_B0 " * 8),
        # Real event before A's second task, but after A's first task.
        review("future-A0", "user-A", "item-Z", 3, 3, "A_REAL_HISTORY_AFTER_A0 " * 8),
        review("target-A1", "user-A", "item-Y", 4, 2, "TARGET_SOURCE_A1 " * 8),
        review("reference-Y-old", "user-C", "item-Y", 3, 4, "REFERENCE_Y_OLD " * 8),
        review("target-B1", "user-B", "item-Y", 5, 4, "TARGET_SOURCE_B1 " * 8),
        review("future-B1", "user-B", "item-Z", 6, 3, "B_REAL_FUTURE " * 8),
        # Duplicate target ID makes task 4 ambiguous. Both rows predate task 5
        # and must remain withheld rather than becoming C's trusted history.
        review("ambiguous-target", "user-C", "item-Z", 6, 3,
               "AMBIGUOUS_SOURCE_EARLY " * 8),
        review("ambiguous-target", "user-C", "item-Z", 7, 3,
               "AMBIGUOUS_SOURCE_ON_TARGET_TIME " * 8),
        review("target-C0", "user-C", "item-W", 8, 4, "TARGET_SOURCE_C0 " * 8),
        review("reference-X-unknown-owner", None, "item-X", 1, 4,
               "REFERENCE_X_UNKNOWN_OWNER " * 8),
    ]
    review_file = root / "reviews.json"
    write_json(review_file, reviews)
    manifest = build_temporal_manifest(
        task_dir,
        truth_dir,
        [review_file],
        train_ratio=0.6,
        validation_ratio=0.2,
        # This synthetic fixture defines date as the actual event date.
        timestamp_semantics_caller_asserted=True,
    )
    return task_dir, truth_dir, review_file, manifest


class RecordingFakeLLM:
    def __init__(self, task_index):
        self.task_index = task_index
        self.prompts = []

    def __call__(self, messages, **_kwargs):
        self.prompts.append(messages[0]["content"])
        return json.dumps({
            "stars": 3,
            "review": f"SYNTHETIC_GENERATED_REVIEW_{self.task_index}",
        })


def test_provider_reconstructs_strict_context_and_excludes_every_target(tmp_path):
    task_dir, truth_dir, review_file, manifest = make_temporal_fixture(tmp_path)
    provider = TemporalContextProvider(
        task_dir, truth_dir, [review_file], manifest
    )

    contexts = provider.contexts_in_execution_order()
    assert [context.task_index for context in contexts] == [0, 1, 2, 3, 5]
    assert [context.execution_order for context in contexts] == list(range(5))
    assert [
        (context.task_index, context.source_task_index) for context in contexts
    ] == [(0, "0"), (1, "1"), (2, "2"), (3, "3"), (5, "5")]
    assert contexts[0].split == contexts[1].split
    assert [context.user_repeat_stratum for context in contexts] == [
        "first_seen", "first_seen", "repeat", "first_seen", "first_seen"
    ]

    by_index = {context.task_index: context for context in contexts}
    assert [row["_temporal_source_row_index"] for row in by_index[0].history] == [0]
    assert [row["_temporal_source_row_index"] for row in by_index[0].item_references] == [2]
    assert 14 not in {
        row["_temporal_source_row_index"] for row in by_index[0].item_references
    }
    assert not {"review_id", "user_id", "item_id"}.intersection(
        by_index[0].history[0]
    )
    assert [row["_temporal_source_row_index"] for row in by_index[2].history] == [0, 6]
    assert 4 not in [row["_temporal_source_row_index"] for row in by_index[2].history]
    assert [row["_temporal_source_row_index"] for row in by_index[5].history] == [2, 3, 8]
    assert not {11, 12} & {
        row["_temporal_source_row_index"] for row in by_index[5].history
    }
    assert 3 not in {
        row["_temporal_source_row_index"] for row in by_index[0].item_references
    }

    all_target_texts = {
        f"TARGET_SOURCE_{name} " for name in ("A0", "B0", "A1", "B1", "C0")
    }
    for context in contexts:
        target_time = context.target_timestamp
        for row in (*context.history, *context.item_references):
            assert row["_temporal_event_time"] < target_time
            assert not any(text in row["text"] for text in all_target_texts)
    assert manifest["ineligible_reasons"]["target_review_not_uniquely_matched"] == 1
    assert manifest["target_timestamp_semantics_verified"] is False
    assert manifest["target_timestamp_semantics_caller_asserted"] is True
    manifest_text = json.dumps(manifest)
    for sensitive_value in (
        "user-A", "item-X", "target-A0", "TARGET_SOURCE_A0", "GROUNDTRUTH_LABEL_A0"
    ):
        assert sensitive_value not in manifest_text
    assert provider.split_by_task_index == {
        row["task_index"]: row["split"] for row in manifest["tasks"]
    }


def test_cli_manifest_can_be_consumed_by_provider(tmp_path, capsys):
    task_dir, truth_dir, review_file, _ = make_temporal_fixture(tmp_path)
    manifest_file = tmp_path / "temporal_manifest.json"
    assert audit_main([
        "manifest", "--task-dir", str(task_dir),
        "--groundtruth-dir", str(truth_dir),
        "--review-data", str(review_file),
        "--confirm-target-timestamp-semantics",
        "--output", str(manifest_file),
    ]) == 0
    capsys.readouterr()

    provider = TemporalContextProvider(
        task_dir, truth_dir, [review_file], manifest_file
    )
    assert len(provider.contexts_in_execution_order()) == 5


def test_provider_fails_closed_on_unconfirmed_or_stale_manifest(tmp_path):
    task_dir, truth_dir, review_file, _ = make_temporal_fixture(tmp_path)
    unconfirmed = build_temporal_manifest(task_dir, truth_dir, [review_file])
    with pytest.raises(
        TemporalManifestError, match="lack the required caller assertion"
    ):
        TemporalContextProvider(task_dir, truth_dir, [review_file], unconfirmed)

    _, _, _, confirmed = make_temporal_fixture(tmp_path / "other")
    stale = json.loads(json.dumps(confirmed))
    stale["tasks"][0]["visible_history_row_indexes"] = []
    with pytest.raises(TemporalManifestError, match="do not match current sources"):
        TemporalContextProvider(task_dir, truth_dir, [review_file], stale)


def test_provider_detects_changed_source_context_fingerprint(tmp_path):
    task_dir, truth_dir, review_file, manifest = make_temporal_fixture(tmp_path)
    reviews = json.loads(review_file.read_text(encoding="utf-8"))
    reviews[0]["text"] = "changed authorized history source text"
    write_json(review_file, reviews)

    with pytest.raises(TemporalManifestError, match="source fingerprints"):
        TemporalContextProvider(task_dir, truth_dir, [review_file], manifest)


def test_provider_rejects_zero_eligible_audit_manifest(tmp_path):
    task_dir = tmp_path / "tasks"
    truth_dir = tmp_path / "groundtruth"
    write_json(task_dir / "0.json", {"user_id": "u", "item_id": "i"})
    write_json(truth_dir / "0.json", {
        "review_id": "missing", "timestamp": "2024-01-02T00:00:00Z",
        "stars": 4,
    })
    review_file = tmp_path / "reviews.json"
    write_json(review_file, [
        review("old", "u", "old-item", 1, 4, "OLD " * 8),
    ])
    manifest = build_temporal_manifest(
        task_dir, truth_dir, [review_file],
        timestamp_semantics_caller_asserted=True,
    )

    assert manifest["task_count"] == 1
    assert manifest["eligible_task_count"] == 0
    with pytest.raises(TemporalManifestError, match="audit-only"):
        TemporalContextProvider(task_dir, truth_dir, [review_file], manifest)


def test_provider_rejects_zero_tasks_and_empty_review_source(tmp_path):
    task_dir = tmp_path / "empty-tasks"
    truth_dir = tmp_path / "empty-truth"
    write_json(task_dir / "batch.json", [])
    write_json(truth_dir / "batch.json", [])
    review_file = tmp_path / "nonempty-reviews.json"
    write_json(review_file, [{"text": "source row"}])
    zero_task_manifest = build_temporal_manifest(
        task_dir, truth_dir, [review_file],
        timestamp_semantics_caller_asserted=True,
    )
    with pytest.raises(TemporalManifestError, match="no paired tasks"):
        TemporalContextProvider(
            task_dir, truth_dir, [review_file], zero_task_manifest
        )

    valid_tasks = tmp_path / "tasks"
    valid_truth = tmp_path / "truth"
    write_json(valid_tasks / "0.json", {
        "user_id": "u", "item_id": "i",
        "timestamp": "2024-01-02T00:00:00Z",
    })
    write_json(valid_truth / "0.json", {
        "review_id": "target", "stars": 4,
        "timestamp": "2024-01-02T00:00:00Z",
    })
    empty_reviews = tmp_path / "empty-reviews.json"
    write_json(empty_reviews, [])
    empty_manifest = build_temporal_manifest(
        valid_tasks, valid_truth, [empty_reviews],
        timestamp_semantics_caller_asserted=True,
    )
    with pytest.raises(TemporalManifestError, match="contains no records"):
        TemporalContextProvider(
            valid_tasks, valid_truth, [empty_reviews], empty_manifest
        )


def test_provider_source_fingerprint_covers_irrelevant_records(tmp_path):
    task_dir, truth_dir, review_file, manifest = make_temporal_fixture(tmp_path)
    reviews = json.loads(review_file.read_text(encoding="utf-8"))
    reviews.append(review(
        "unrelated", "unrelated-user", "unrelated-item", 1, 3,
        "UNRELATED ORIGINAL CONTENT " * 8,
    ))
    write_json(review_file, reviews)

    with pytest.raises(TemporalManifestError, match="source fingerprints"):
        TemporalContextProvider(task_dir, truth_dir, [review_file], manifest)


def test_multifile_review_indexes_are_natural_order_and_stable(tmp_path):
    task_dir = tmp_path / "tasks"
    truth_dir = tmp_path / "groundtruth"
    write_json(task_dir / "0.json", {
        "task_index": 0, "user_id": "u", "item_id": "i",
        "timestamp": "2024-01-03T00:00:00Z",
    })
    write_json(truth_dir / "0.json", {
        "task_index": 0, "review_id": "target", "stars": 4,
        "timestamp": "2024-01-03T00:00:00Z",
    })
    review_dir = tmp_path / "reviews"
    write_json(review_dir / "part_2.json", [
        review("old", "u", "old-item", 1, 5, "OLD MULTIFILE HISTORY " * 8),
    ])
    write_json(review_dir / "part_10.json", [
        review("target", "u", "i", 3, 4, "TARGET MULTIFILE " * 8),
    ])
    manifest = build_temporal_manifest(
        task_dir, truth_dir, [review_dir],
        timestamp_semantics_caller_asserted=True,
    )
    provider = TemporalContextProvider(
        task_dir, truth_dir, [review_dir], manifest
    )

    assert manifest["source_review_rows"] == 2
    assert manifest["tasks"][0]["visible_history_row_indexes"] == [0]
    assert [
        row["_temporal_source_row_index"]
        for row in provider.context_for_task(0).history
    ] == [0]


def test_matched_target_source_time_is_held_out_when_task_time_is_missing(tmp_path):
    task_dir = tmp_path / "tasks"
    truth_dir = tmp_path / "groundtruth"
    write_json(task_dir / "0.json", {
        "task_index": 0, "user_id": "u", "item_id": "i",
    })
    write_json(truth_dir / "0.json", {
        "task_index": 0, "review_id": "target-0", "stars": 3,
    })
    write_json(task_dir / "1.json", {
        "task_index": 1, "user_id": "u", "item_id": "i",
    })
    write_json(truth_dir / "1.json", {
        "task_index": 1, "review_id": "target-1", "stars": 4,
        "timestamp": "2024-01-04T00:00:00Z",
    })
    review_file = tmp_path / "reviews.json"
    write_json(review_file, [
        review("old-history", "u", "old-item", 1, 4, "OLD_HISTORY " * 8),
        review("target-0", "u", "i", 2, 3, "TARGET_ZERO " * 8),
        review("same-user-event", "u", "other-item", 2, 2,
               "SAME_USER_EVENT_TIME " * 8),
        review("same-item-event", "other-user", "i", 2, 5,
               "SAME_ITEM_EVENT_TIME " * 8),
        review("target-1", "u", "i", 4, 4, "TARGET_ONE " * 8),
    ])
    manifest = build_temporal_manifest(
        task_dir, truth_dir, [review_file], timestamp_semantics_caller_asserted=True
    )
    provider = TemporalContextProvider(
        task_dir, truth_dir, [review_file], manifest
    )

    assert manifest["eligible_task_count"] == 1
    later = provider.context_for_task(1)
    assert [row["_temporal_source_row_index"] for row in later.history] == [0]
    assert later.item_references == ()


def test_unmatched_target_with_missing_identity_fails_closed_by_context_channel(
    tmp_path,
):
    task_dir = tmp_path / "tasks"
    truth_dir = tmp_path / "groundtruth"
    write_json(task_dir / "0.json", {"task_index": 0, "user_id": "u1"})
    write_json(truth_dir / "0.json", {
        "task_index": 0, "review_id": "unmatched-item-target",
        "timestamp": "2024-01-02T00:00:00Z", "stars": 3,
    })
    write_json(task_dir / "1.json", {"task_index": 1, "item_id": "i1"})
    write_json(truth_dir / "1.json", {
        "task_index": 1, "review_id": "unmatched-user-target",
        "timestamp": "2024-01-03T00:00:00Z", "stars": 4,
    })
    write_json(task_dir / "2.json", {
        "task_index": 2, "user_id": "u2", "item_id": "i1",
    })
    write_json(truth_dir / "2.json", {
        "task_index": 2, "review_id": "target-2", "stars": 4,
        "timestamp": "2024-01-05T00:00:00Z",
    })
    review_file = tmp_path / "reviews.json"
    write_json(review_file, [
        review("u2-history", "u2", "old-item", 1, 4, "U2_HISTORY " * 8),
        review("item-reference", "u3", "i1", 1, 5, "I1_REFERENCE " * 8),
        review("target-2", "u2", "i1", 5, 4, "TARGET_TWO " * 8),
    ])

    manifest = build_temporal_manifest(
        task_dir, truth_dir, [review_file], timestamp_semantics_caller_asserted=True
    )
    provider = TemporalContextProvider(
        task_dir, truth_dir, [review_file], manifest
    )

    suppression = manifest["review_context_suppression"]
    assert suppression["suppress_all_user_history"] is True
    assert suppression["suppress_all_item_references"] is True
    context = provider.context_for_task(2)
    assert context.history == ()
    assert context.item_references == ()


def test_all_modes_share_task_context_and_trace_actual_prompt_inclusion(tmp_path):
    task_dir, truth_dir, review_file, manifest = make_temporal_fixture(tmp_path)
    provider = TemporalContextProvider(task_dir, truth_dir, [review_file], manifest)
    llms = {}

    def llm_factory(task_index, mode):
        llm = RecordingFakeLLM(task_index)
        llms[(mode, task_index)] = llm
        return llm

    suite = run_temporal_ablation_suite(provider, llm_factory)
    expected_indexes = [0, 1, 2, 3, 5]
    signatures = {
        mode: [
            (row["task_index"], row["execution_order"], row["context_sha256"])
            for row in records
        ]
        for mode, records in suite.items()
    }
    assert all(signature == signatures["No_Memory"] for signature in signatures.values())
    assert [row["task_index"] for row in suite["No_Memory"]] == expected_indexes

    for task_index in expected_indexes:
        no_memory = next(
            row for row in suite["No_Memory"] if row["task_index"] == task_index
        )
        trusted = next(
            row for row in suite["Trusted_History_Stats"]
            if row["task_index"] == task_index
        )
        assert no_memory["visible_history_source_row_indexes"] == (
            trusted["visible_history_source_row_indexes"]
        )
        assert no_memory["visible_item_reference_source_row_indexes"] == (
            trusted["visible_item_reference_source_row_indexes"]
        )
        assert no_memory["history_source_sha256"] == trusted["history_source_sha256"]
        assert no_memory["agent_diagnostics"]["memory_recalled_count"] == 0
        assert trusted["agent_diagnostics"]["memory_recalled_count"] == 0
        assert no_memory["agent_diagnostics"][
            "user_history_profile_source_row_indexes"
        ] == no_memory["visible_history_source_row_indexes"]
        assert trusted["trusted_history_stats_included"] is True
        assert no_memory["trusted_history_stats_included"] is False

        prompt = llms[("No_Memory", task_index)].prompts[0]
        assert "GROUNDTRUTH_" not in prompt
        assert "TARGET_SOURCE_" not in prompt
        assert set(no_memory["agent_diagnostics"]["user_history_prompt_row_indexes"]).issubset(
            no_memory["visible_history_source_row_indexes"]
        )
        assert set(no_memory["agent_diagnostics"]["item_reference_prompt_row_indexes"]).issubset(
            no_memory["visible_item_reference_source_row_indexes"]
        )

    repeated_index = 2
    same_time_generated = next(
        row for row in suite["Generated_Review_Memory"]
        if row["task_index"] == 1
    )
    same_time_combined = next(
        row for row in suite["Combined"] if row["task_index"] == 1
    )
    assert same_time_generated["agent_diagnostics"]["memory_recalled_count"] == 0
    assert same_time_combined["agent_diagnostics"]["memory_recalled_count"] == 0
    assert "SYNTHETIC_GENERATED_REVIEW_0" not in llms[
        ("Generated_Review_Memory", 1)
    ].prompts[0]
    assert "SYNTHETIC_GENERATED_REVIEW_0" not in llms[("Combined", 1)].prompts[0]

    generated_record = next(
        row for row in suite["Generated_Review_Memory"]
        if row["task_index"] == repeated_index
    )
    combined_record = next(
        row for row in suite["Combined"] if row["task_index"] == repeated_index
    )
    assert generated_record["agent_diagnostics"]["memory_recalled_count"] == 2
    assert generated_record["agent_diagnostics"]["memory_prompt_entry_count"] == 2
    assert generated_record["agent_diagnostics"]["memory_recalled_origin_orders"] == [1, 0]
    assert combined_record["agent_diagnostics"]["memory_prompt_entry_count"] == 2
    repeated_no_memory = next(
        row for row in suite["No_Memory"] if row["task_index"] == repeated_index
    )
    assert repeated_no_memory["agent_diagnostics"][
        "user_history_prompt_row_indexes"
    ] == [6, 0]
    assert "SYNTHETIC_GENERATED_REVIEW_0" in llms[("Generated_Review_Memory", 2)].prompts[0]
    assert "SYNTHETIC_GENERATED_REVIEW_1" in llms[("Generated_Review_Memory", 2)].prompts[0]
    assert "SYNTHETIC_GENERATED_REVIEW_0" in llms[("Combined", 2)].prompts[0]
    assert "SYNTHETIC_GENERATED_REVIEW_1" in llms[("Combined", 2)].prompts[0]
    assert "目标时点前可见真实历史统计" in llms[("Trusted_History_Stats", 2)].prompts[0]
    assert "SYNTHETIC_GENERATED_REVIEW_0" not in llms[("Trusted_History_Stats", 2)].prompts[0]

    stats = generated_record["trusted_history_stats_source_row_indexes"]
    assert stats == []
    trusted_repeat = next(
        row for row in suite["Trusted_History_Stats"] if row["task_index"] == 2
    )
    assert trusted_repeat["trusted_history_stats_source_row_indexes"] == [0, 6]
    assert trusted_repeat["trusted_history_stats_source_review_count"] == 2
    assert trusted_repeat["trusted_history_stats_source_event_time_max"].startswith(
        "2024-01-03"
    )


def test_four_ablation_contracts_are_explicit():
    assert {
        mode.value: (
            contract.generated_review_memory,
            contract.trusted_history_stats,
        )
        for mode, contract in TEMPORAL_ABLATION_CONTRACTS.items()
    } == {
        "No_Memory": (False, False),
        "Generated_Review_Memory": (True, False),
        "Trusted_History_Stats": (False, True),
        "Combined": (True, True),
    }
    assert TemporalAblationMode.NO_MEMORY.value == "No_Memory"


def test_multi_record_task_groundtruth_pairing_needs_explicit_matching_indexes(
    tmp_path,
):
    task_dir = tmp_path / "tasks"
    truth_dir = tmp_path / "groundtruth"
    write_json(task_dir / "batch.json", [
        {"task_index": 20, "user_id": "u", "item_id": "i20"},
        {"task_index": 10, "user_id": "u", "item_id": "i10"},
    ])
    write_json(truth_dir / "batch.json", [
        {"task_index": 10, "stars": 1},
        {"task_index": 20, "stars": 5},
    ])

    pairs, diagnostics = _load_task_pairs(task_dir, truth_dir)

    assert diagnostics["warnings"] == []
    assert diagnostics["record_alignment"] == "explicit_task_index"
    assert [truth["stars"] for _, truth in pairs] == [5, 1]

    write_json(truth_dir / "batch.json", [
        {"stars": 1},
        {"stars": 5},
    ])
    pairs, diagnostics = _load_task_pairs(task_dir, truth_dir)
    assert pairs == []
    assert diagnostics["warnings"] == [
        "multi_record_pairing_requires_matching_unique_task_indexes"
    ]


def test_task_groundtruth_user_or_item_conflict_is_rejected(tmp_path):
    task_dir = tmp_path / "tasks"
    truth_dir = tmp_path / "groundtruth"
    write_json(task_dir / "one.json", {
        "task_index": 7, "user_id": "user-A", "item_id": "item-A",
    })
    write_json(truth_dir / "one.json", {
        "task_index": 7, "user_id": "user-B", "item_id": "item-A",
        "stars": 4,
    })

    pairs, diagnostics = _load_task_pairs(task_dir, truth_dir)

    assert pairs == []
    assert diagnostics["warnings"] == [
        "paired_task_groundtruth_identity_mismatch"
    ]


def test_task_groundtruth_event_time_conflict_is_rejected(tmp_path):
    task_dir = tmp_path / "tasks"
    truth_dir = tmp_path / "groundtruth"
    write_json(task_dir / "one.json", {
        "task_index": 7, "user_id": "u", "item_id": "i",
        "timestamp": "2024-01-02T00:00:00Z",
    })
    write_json(truth_dir / "one.json", {
        "task_index": 7, "user_id": "u", "item_id": "i",
        "timestamp": "2024-01-03T00:00:00Z", "stars": 4,
    })

    pairs, diagnostics = _load_task_pairs(task_dir, truth_dir)

    assert pairs == []
    assert diagnostics["warnings"] == [
        "paired_task_groundtruth_identity_mismatch"
    ]
