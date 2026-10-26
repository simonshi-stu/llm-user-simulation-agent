"""Synthetic, offline tests for the Amazon P0 adapter, prepare, and runner.

No real review data, identifiers, or network access is used here. The
fixture is a hand-built 36-row Amazon-shaped JSONL file whose day/user/item
layout yields deterministic selection and split counts.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

import amazon_p0
from amazon_p0 import (
    AmazonP0Error,
    download_source,
    iter_source_records,
    load_source_records,
    main,
    normalize_amazon_row,
    prepare_dataset,
    run_baselines,
)
from temporal_context import TemporalContextProvider, TemporalManifestError

ROOT = Path(__file__).resolve().parents[1]

RAW_USERS = tuple(f"RAW-USER-{index:02d}" for index in range(1, 10))
RAW_ITEMS = tuple(f"RAW-ITEM-{name}" for name in (
    "A1", "B1", "X1", "C1", "D1", "E1", "F1", "G1", "H1", "I1", "J1", "K1",
    "L1", "M1", "N1", "O1", "Y1", "Z1", "W1", "V1", "U1", "T1", "S1", "R1",
    "Q1", "P1", "N2", "M2", "L2", "K2", "J2", "I2", "H2",
))
RAW_MARKERS = (*RAW_USERS, *RAW_ITEMS, "RAW-REVIEWER-NAME", "RAW-SUMMARY",
               "SYNTHETIC p0 review body")


def epoch(day_offset: int) -> int:
    base = datetime(2014, 7, 1, tzinfo=timezone.utc)
    return int((base + timedelta(days=day_offset)).timestamp())


def raw_row(user, item, stars, day, text):
    return {
        "reviewerID": user,
        "asin": item,
        "reviewerName": "RAW-REVIEWER-NAME",
        "helpful": [3, 7],
        "reviewText": text,
        "overall": stars,
        "summary": "RAW-SUMMARY",
        "unixReviewTime": epoch(day),
        "image": ["http://raw-image.invalid/1.jpg"],
    }


def _target_plan():
    def user(number):
        return RAW_USERS[number - 1]

    def item(name):
        return f"RAW-ITEM-{name}"

    return [
        (1, user(1), item("A1"), 5), (1, user(2), item("B1"), 4),
        (1, user(3), item("X1"), 3),
        (2, user(4), item("C1"), 2), (2, user(5), item("X1"), 5),
        (2, user(6), item("D1"), 1),
        (3, user(7), item("E1"), 4), (3, user(8), item("F1"), 2),
        (3, user(1), item("B1"), 4),
        (4, user(2), item("G1"), 3), (4, user(9), item("H1"), 5),
        (4, user(3), item("I1"), 2),
        (5, user(4), item("J1"), 4), (5, user(5), item("K1"), 3),
        (5, user(6), item("L1"), 5),
        (6, user(7), item("M1"), 2), (6, user(8), item("N1"), 4),
        (6, user(9), item("O1"), 3),
        (7, user(1), item("X1"), 4), (7, user(2), item("Y1"), 5),
        (7, user(3), item("Z1"), 2),
        (8, user(1), item("W1"), 3), (8, user(2), item("V1"), 4),
        (8, user(4), item("U1"), 5),
        (9, user(1), item("T1"), 2), (9, user(3), item("S1"), 3),
        (9, user(5), item("R1"), 4),
        (10, user(2), item("Q1"), 1), (10, user(4), item("P1"), 5),
        (10, user(6), item("N2"), 3),
        (11, user(1), item("M2"), 4), (11, user(5), item("L2"), 2),
        (11, user(7), item("K2"), 3),
        (12, user(2), item("J2"), 5), (12, user(3), item("I2"), 1),
        (12, user(8), item("H2"), 4),
    ]


def build_synthetic_rows(*, copy_stars=5):
    rows = []
    for position, (day, user, item, stars) in enumerate(_target_plan()):
        text = f"SYNTHETIC p0 review body {position:02d}"
        rows.append(raw_row(user, item, stars, day, text))
    # A frozen-history row duplicates the (7, RAW-USER-01, RAW-ITEM-X1)
    # target's text: every context must withhold this matchable copy, and
    # the global prior mask must exclude it too.
    rows[4]["reviewText"] = rows[18]["reviewText"]
    rows[4]["overall"] = copy_stars
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_gz_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_source(tmp_path: Path, *, copy_stars=5) -> Path:
    source = tmp_path / "raw" / "p0_synthetic.jsonl.gz"
    write_gz_jsonl(source, build_synthetic_rows(copy_stars=copy_stars))
    return source


def prepare_fixture(tmp_path: Path, *, name="work"):
    source = make_source(tmp_path)
    work = tmp_path / name
    summary = prepare_dataset(
        source,
        work,
        history_ratio=0.5,
        max_tasks=50,
        seed=42,
        timestamp_semantics_confirmed=True,
    )
    return source, work, summary


def read_jsonl(path: Path):
    return amazon_p0._read_jsonl(path)


# ---------------------------------------------------------------------------
# Field adapter and strict parsing.
# ---------------------------------------------------------------------------


def test_adapter_maps_fields_pseudonymizes_and_drops_metadata():
    raw = raw_row("RAW-USER-99", "RAW-ITEM-99", 4, 3, "a plain synthetic body")
    record, reason = normalize_amazon_row(raw, 5)

    assert reason is None
    assert set(record) == set(amazon_p0.CANONICAL_FIELDS)
    assert record["timestamp"] == "2014-07-04"
    assert record["stars"] == 4.0
    assert record["source_row"] == 5
    assert record["user_id"].startswith("u_") and record["item_id"].startswith("i_")
    assert "RAW-USER-99" not in json.dumps(record)
    assert "RAW-ITEM-99" not in json.dumps(record)
    assert "RAW-REVIEWER-NAME" not in json.dumps(record)
    assert "RAW-SUMMARY" not in json.dumps(record)
    assert "helpful" not in record and "image" not in record

    repeated, _ = normalize_amazon_row(raw, 5)
    assert repeated == record  # stable pseudonyms and review id
    other_row, _ = normalize_amazon_row(raw, 6)
    assert other_row["review_id"] != record["review_id"]


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda row: row.update(reviewerID=None), "missing_user_id"),
        (lambda row: row.update(reviewerID=7), "missing_user_id"),
        (lambda row: row.update(reviewerID="  "), "missing_user_id"),
        (lambda row: row.update(asin=None), "missing_item_id"),
        (lambda row: row.update(asin=[]), "missing_item_id"),
        (lambda row: row.update(overall=None), "missing_stars"),
        (lambda row: row.update(overall=True), "invalid_stars"),
        (lambda row: row.update(overall="4"), "invalid_stars"),
        (lambda row: row.update(overall=0), "invalid_stars"),
        (lambda row: row.update(overall=6), "invalid_stars"),
        (lambda row: row.update(reviewText=None), "missing_review_text"),
        (lambda row: row.update(reviewText=12), "invalid_review_text"),
        (lambda row: row.update(unixReviewTime=None), "missing_timestamp"),
        (lambda row: row.update(unixReviewTime=True), "invalid_timestamp"),
        (lambda row: row.update(unixReviewTime=-1), "invalid_timestamp"),
        (lambda row: row.update(unixReviewTime="1400000000"),
         "invalid_timestamp"),
        (lambda row: row.update(unixReviewTime=1.5), "invalid_timestamp"),
        (lambda row: row.update(unixReviewTime=2 ** 62), "invalid_timestamp"),
    ],
)
def test_adapter_rejects_invalid_key_fields_with_categories(mutate, reason):
    raw = raw_row("RAW-USER-01", "RAW-ITEM-A1", 4, 1, "body")
    mutate(raw)
    record, found = normalize_amazon_row(raw, 0)
    assert record is None
    assert found == reason


def test_adapter_rejects_non_object_rows():
    record, reason = normalize_amazon_row(["not", "a", "row"], 0)
    assert record is None
    assert reason == "not_an_object"


def test_source_parser_is_strict_about_malformed_json(tmp_path):
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"reviewerID": "x"}\n{"reviewerID": \n', encoding="utf-8")
    with pytest.raises(AmazonP0Error, match="source_parse_failed"):
        list(iter_source_records(broken))

    array = tmp_path / "broken.json"
    array.write_text('[{"reviewerID": "x"},', encoding="utf-8")
    with pytest.raises(AmazonP0Error, match="source_parse_failed"):
        list(iter_source_records(array))


@pytest.mark.parametrize("payload", [
    '[{"a": 1}] NOT_JSON',
    '[{"a": 1}] {"b": 2}',
    '[{"a": 1}][{"b": 2}]',
    '[] garbage',
])
def test_source_parser_rejects_trailing_json_documents(tmp_path, payload):
    path = tmp_path / "trailing.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(AmazonP0Error, match="source_parse_failed"):
        list(iter_source_records(path))


def test_source_parser_accepts_empty_and_nested_arrays(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    assert list(iter_source_records(empty)) == []

    nested = tmp_path / "nested.json"
    nested.write_text('[[1, 2], {"a": 1}]', encoding="utf-8")
    assert list(iter_source_records(nested)) == [[1, 2], {"a": 1}]


def test_source_parser_handles_invalid_utf8_safely(tmp_path):
    array = tmp_path / "bad_utf8.json"
    array.write_bytes(b'[{"a": "\xff\xfe"}]')
    with pytest.raises(AmazonP0Error, match="source_parse_failed"):
        list(iter_source_records(array))

    jsonl = tmp_path / "bad_utf8.jsonl"
    jsonl.write_bytes(b'{"reviewerID": "x", "reviewText": "\xff"}\n')
    with pytest.raises(AmazonP0Error, match="source_parse_failed"):
        list(iter_source_records(jsonl))


def test_source_parser_rejects_oversized_lines_and_records_early(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(amazon_p0, "MAX_RAW_RECORD_BYTES", 64)
    long_line = tmp_path / "long.jsonl"
    long_line.write_text(json.dumps({"reviewerID": "x" * 200}) + "\n")
    with pytest.raises(AmazonP0Error, match="source_line_limit_exceeded"):
        list(iter_source_records(long_line))

    big_record = tmp_path / "big.json"
    big_record.write_text(json.dumps([{"reviewerID": "x" * 200}]), encoding="utf-8")
    with pytest.raises(AmazonP0Error, match="source_record_limit_exceeded"):
        list(iter_source_records(big_record))


def test_array_record_cap_counts_bytes_and_spares_small_records(
    tmp_path, monkeypatch,
):
    # Many small records must not trip the cap just because they share one
    # read chunk; the cap applies per record, measured in UTF-8 bytes.
    monkeypatch.setattr(amazon_p0, "MAX_RAW_RECORD_BYTES", 64)
    many_small = tmp_path / "many.json"
    payload = [{"a": index} for index in range(50)]
    many_small.write_text(json.dumps(payload), encoding="utf-8")
    assert list(iter_source_records(many_small)) == payload

    multibyte = tmp_path / "multibyte.json"
    text = "好" * 30  # 90 UTF-8 bytes but only 30 characters
    multibyte.write_text(json.dumps([{"a": text}]), encoding="utf-8")
    with pytest.raises(AmazonP0Error, match="source_record_limit_exceeded"):
        list(iter_source_records(multibyte))

    complete_but_large = tmp_path / "large.json"
    complete_but_large.write_text(
        json.dumps([{"a": "x" * 100}]), encoding="utf-8"
    )
    with pytest.raises(AmazonP0Error, match="source_record_limit_exceeded"):
        list(iter_source_records(complete_but_large))


def test_source_parser_supports_json_arrays(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
    assert list(iter_source_records(path)) == [{"a": 1}, {"a": 2}]


def test_shared_source_gates_apply(tmp_path, monkeypatch):
    rows = [
        raw_row(f"RAW-USER-{index}", "RAW-ITEM-A1", 4, 1, "body")
        for index in range(6)
    ]
    source = tmp_path / "rows.jsonl"
    write_jsonl(source, rows)

    monkeypatch.setattr(amazon_p0, "TEMPORAL_MAX_REVIEW_ROWS", 5)
    with pytest.raises(AmazonP0Error, match="source_row_limit_exceeded"):
        load_source_records(source)

    monkeypatch.setattr(amazon_p0, "TEMPORAL_MAX_REVIEW_SOURCE_BYTES", 100)
    with pytest.raises(AmazonP0Error, match="source_byte_limit_exceeded"):
        amazon_p0._preflight_source(source)
    with pytest.raises(AmazonP0Error, match="source_byte_limit_exceeded"):
        prepare_dataset(
            source, tmp_path / "byte_limited",
            history_ratio=0.5, max_tasks=50, seed=42,
            timestamp_semantics_confirmed=True,
        )
    assert not (tmp_path / "byte_limited").exists()

    compressed = tmp_path / "rows.jsonl.gz"
    write_gz_jsonl(compressed, rows)
    decoded = len(gzip.open(compressed, "rb").read())
    limit = compressed.stat().st_size + 1
    assert compressed.stat().st_size < limit < decoded
    monkeypatch.setattr(amazon_p0, "TEMPORAL_MAX_REVIEW_SOURCE_BYTES", limit)
    with pytest.raises(
        AmazonP0Error, match="source_decoded_byte_limit_exceeded"
    ):
        amazon_p0._preflight_source(compressed)


def test_duplicate_rows_are_audited_and_kept_with_unique_ids(tmp_path):
    template = raw_row("RAW-USER-01", "RAW-ITEM-A1", 4, 4, "same body")
    source = tmp_path / "dupes.jsonl"
    write_jsonl(source, [dict(template), dict(template), raw_row(
        "RAW-USER-02", "RAW-ITEM-A2", 3, 4, "other body",
    )])

    records, audit = load_source_records(source)
    assert audit["parsed_rows"] == 3
    assert audit["valid_rows"] == 3
    assert audit["duplicate_groups"] == 1
    assert audit["duplicate_rows"] == 1
    assert len({record["review_id"] for record in records}) == 3
    assert [record["source_row"] for record in records] == [0, 1, 2]

    again, again_audit = load_source_records(source)
    assert [record["review_id"] for record in again] == [
        record["review_id"] for record in records
    ]
    assert again_audit == audit


# ---------------------------------------------------------------------------
# Prepare: frozen history, selection, manifest, policy.
# ---------------------------------------------------------------------------


def test_prepare_requires_confirmation_and_refuses_overwrite(tmp_path):
    source = make_source(tmp_path)
    work = tmp_path / "work"

    with pytest.raises(AmazonP0Error, match="confirmation_required"):
        prepare_dataset(source, work)
    assert not work.exists()

    summary = prepare_dataset(
        source, work, history_ratio=0.5, max_tasks=50, seed=42,
        timestamp_semantics_confirmed=True,
    )
    assert summary["status"] == "prepared"
    with pytest.raises(AmazonP0Error, match="output_dir_exists"):
        prepare_dataset(
            source, work, history_ratio=0.5, max_tasks=50, seed=42,
            timestamp_semantics_confirmed=True,
        )
    with pytest.raises(AmazonP0Error, match="source_missing"):
        prepare_dataset(
            tmp_path / "absent.jsonl", tmp_path / "other",
            timestamp_semantics_confirmed=True,
        )


def test_prepare_cli_output_is_consumed_by_the_strict_provider(
    tmp_path, capsys,
):
    source = make_source(tmp_path)
    work = tmp_path / "work"
    assert main([
        "prepare", "--source", str(source), "--work-dir", str(work),
        "--history-ratio", "0.5", "--max-tasks", "50", "--seed", "42",
        "--confirm-target-timestamp-semantics",
    ]) == 0
    stdout = capsys.readouterr().out
    assert "RAW-USER" not in stdout and "review body" not in stdout

    provider = TemporalContextProvider(
        work / "tasks", work / "groundtruth",
        [work / "runtime_reviews.jsonl"], work / "manifest.json",
    )  # raises if the CLI artifacts cannot be strictly revalidated
    contexts = provider.contexts_in_execution_order()
    assert len(contexts) == 15
    assert [context.execution_order for context in contexts] == list(range(15))
    assert len({context.task_index for context in contexts}) == 15


def test_selection_never_gives_any_user_only_one_task_and_keeps_dates_distinct():
    rows = []
    for user_index in range(9):
        start_day = 10 + (user_index % 3)
        for offset in range(3):
            rows.append({
                "user_id": f"u_{user_index:03d}",
                "item_id": f"i_{offset}",
                "timestamp": f"2014-07-{start_day + offset:02d}",
                "source_row": user_index * 3 + offset,
                "stars": 4.0,
            })
    for budget in (7, 8, 15, 26):
        selected = amazon_p0.select_targets(rows, max_tasks=budget, seed=42)
        counts = Counter(row["user_id"] for row in selected)
        assert len(selected) <= budget
        assert len(selected) >= budget - 1
        assert counts and min(counts.values()) >= 2
        assert len({row["timestamp"] for row in selected}) >= 3
        per_user_dates = {
            user_id: {
                row["timestamp"] for row in selected
                if row["user_id"] == user_id
            }
            for user_id in counts
        }
        assert all(len(dates) >= 2 for dates in per_user_dates.values())
        per_user_ordered = {
            user_id: [
                row["timestamp"] for row in selected
                if row["user_id"] == user_id
            ]
            for user_id in counts
        }
        assert all(
            dates == sorted(dates) for dates in per_user_ordered.values()
        )
        assert selected == sorted(
            selected, key=lambda row: (row["timestamp"], row["source_row"])
        )
    assert amazon_p0._spread_indexes(5, 4) == [0, 1, 3, 4]
    assert amazon_p0._spread_indexes(3, 2) == [0, 2]
    assert amazon_p0._spread_indexes(2, 2) == [0, 1]

    # Budgets that cannot satisfy the selection contract fail safely.
    with pytest.raises(AmazonP0Error, match="insufficient_selected_tasks"):
        amazon_p0.select_targets(rows, max_tasks=3, seed=42)
    with pytest.raises(AmazonP0Error, match="insufficient_timestamp_groups"):
        amazon_p0.select_targets(rows, max_tasks=4, seed=42)


def test_selection_rejects_users_with_only_same_day_repeats():
    same_day_only = [
        {"user_id": "u_a", "item_id": "i1", "timestamp": "2014-07-10",
         "source_row": 0},
        {"user_id": "u_a", "item_id": "i2", "timestamp": "2014-07-10",
         "source_row": 1},
        {"user_id": "u_b", "item_id": "i3", "timestamp": "2014-07-10",
         "source_row": 2},
        {"user_id": "u_b", "item_id": "i4", "timestamp": "2014-07-10",
         "source_row": 3},
    ]
    with pytest.raises(
        AmazonP0Error, match="insufficient_repeated_user_candidates"
    ):
        amazon_p0.select_targets(same_day_only, max_tasks=4, seed=42)

    mixed = [
        {"user_id": "u_a", "item_id": "i1", "timestamp": "2014-07-10",
         "source_row": 0},
        {"user_id": "u_a", "item_id": "i2", "timestamp": "2014-07-10",
         "source_row": 1},
        {"user_id": "u_a", "item_id": "i3", "timestamp": "2014-07-11",
         "source_row": 2},
        {"user_id": "u_b", "item_id": "i4", "timestamp": "2014-07-10",
         "source_row": 3},
        {"user_id": "u_b", "item_id": "i5", "timestamp": "2014-07-10",
         "source_row": 4},
    ]
    # u_a has two distinct dates; u_b only same-day repeats, so the final
    # selection cannot satisfy two users with two time-distinct tasks each.
    with pytest.raises(AmazonP0Error, match="insufficient_selected_users"):
        amazon_p0.select_targets(mixed, max_tasks=4, seed=42)


def test_prepare_frozen_history_selection_and_split_contract(tmp_path):
    _, work, summary = prepare_fixture(tmp_path)
    policy = json.loads((work / "p0_policy.json").read_text(encoding="utf-8"))
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    tasks = read_jsonl(work / "tasks" / "p0.jsonl")
    runtime = read_jsonl(work / "runtime_reviews.jsonl")
    normalized = read_jsonl(work / "normalized_reviews.jsonl")

    assert summary["parsed_rows"] == 36
    assert summary["rejected_rows"] == 0
    assert summary["selected_tasks"] == 15
    assert summary["split_counts"] == {"train": 9, "validation": 2, "test": 4}

    cut_day = policy["protocol"]["cut_day"]
    assert cut_day == "2014-07-08"
    assert policy["protocol"]["history_rows"] == 18
    assert policy["protocol"]["prior_excluded_rows"] == 1
    assert policy["protocol"]["global_prior_rows"] == 17
    # Hand-calculated 17-row prior: 56/17 after dropping the target-text copy.
    assert summary["global_prior_rows"] == 17
    assert summary["prior_excluded_rows"] == 1
    assert policy["protocol"]["global_prior"] == pytest.approx(56.0 / 17.0)
    assert policy["protocol"]["frozen_history"] is True
    assert policy["source"]["license_status"] == "not_explicitly_verified"
    assert policy["source"]["parsed_rows"] == 36
    assert policy["source"]["non_midnight_timestamp_rows"] == 0
    assert {"reviewerName", "summary", "helpful"}.issubset(
        set(policy["fields"]["dropped_fields"])
    )
    assert "not a partition of the raw" in policy["protocol"][
        "observed_history_note"
    ]

    # The cut never splits a same-day group.
    assert all(row["timestamp"] < cut_day for row in normalized if row["source_row"] < 18)
    history_days = {row["timestamp"] for row in normalized if row["timestamp"] < cut_day}
    target_days = {row["timestamp"] for row in normalized if row["timestamp"] >= cut_day}
    assert history_days.isdisjoint(target_days)
    assert max(history_days) < cut_day <= min(target_days)

    # Runtime source composition is exactly frozen history + selected targets.
    target_ids = {row["target_review_id"] for row in tasks}
    assert len(runtime) == 18 + 15
    for row in runtime:
        assert row["timestamp"] < cut_day or row["review_id"] in target_ids
    for row in normalized:
        if row["timestamp"] >= cut_day and row["review_id"] not in target_ids:
            assert row["source_row"] >= 18

    # Every selected target is a repeated user's time-distinct task.
    per_user = {}
    for row in tasks:
        per_user.setdefault(row["user_id"], set()).add(row["target_date"])
    assert all(len(dates) >= 2 for dates in per_user.values())
    assert len(per_user) == 5

    # Manifest match/split contract; same-day targets never span splits.
    assert manifest["eligible_task_count"] == 15
    assert manifest["target_review_row_match_counts"] == {
        "unique": 15, "ambiguous": 0, "unmatched": 0,
    }
    assert manifest["split_counts"] == {"train": 9, "validation": 2, "test": 4}
    assert all(row["target_text_absent_from_context"] for row in manifest["tasks"])
    task_dates = {row["task_index"]: row["target_date"] for row in tasks}
    splits_by_date = {}
    for row in manifest["tasks"]:
        splits_by_date.setdefault(task_dates[row["task_index"]], set()).add(
            row["split"]
        )
    assert all(len(splits) == 1 for splits in splits_by_date.values())
    assert len({row["user_group"] for row in manifest["tasks"]}) == 5
    assert any(row["user_repeat_stratum"] == "repeat" for row in manifest["tasks"])


def test_global_prior_excludes_target_text_copies_from_predictions(tmp_path):
    _, work, _ = prepare_fixture(tmp_path)
    policy = json.loads((work / "p0_policy.json").read_text(encoding="utf-8"))
    safe_prior = 56.0 / 17.0
    assert policy["protocol"]["history_rows"] == 18
    assert policy["protocol"]["prior_excluded_rows"] == 1
    assert policy["protocol"]["global_prior_rows"] == 17
    assert policy["protocol"]["global_prior"] == pytest.approx(safe_prior)

    out = tmp_path / "out"
    run_baselines(work, out, n_boot=200, seed=42)
    predictions = read_jsonl(out / "predictions.jsonl")
    global_rows = [
        row for row in predictions if row["mode"] == "global"
    ]
    assert len(global_rows) == 15
    assert all(
        row["prediction"] == pytest.approx(safe_prior)
        for row in global_rows
    )


def test_excluded_copy_stars_do_not_change_any_prediction(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    work_a = left_root / "work"
    work_b = right_root / "work"
    prepare_dataset(
        make_source(left_root, copy_stars=5), work_a,
        history_ratio=0.5, max_tasks=50, seed=42,
        timestamp_semantics_confirmed=True,
    )
    prepare_dataset(
        make_source(right_root, copy_stars=1), work_b,
        history_ratio=0.5, max_tasks=50, seed=42,
        timestamp_semantics_confirmed=True,
    )
    policy_a = json.loads(
        (work_a / "p0_policy.json").read_text(encoding="utf-8")
    )
    policy_b = json.loads(
        (work_b / "p0_policy.json").read_text(encoding="utf-8")
    )
    assert policy_a["protocol"]["global_prior"] == pytest.approx(56.0 / 17.0)
    assert policy_b["protocol"]["global_prior"] == pytest.approx(56.0 / 17.0)

    run_baselines(work_a, left_root / "out", n_boot=200, seed=42)
    run_baselines(work_b, right_root / "out", n_boot=200, seed=42)

    def prediction_key(path):
        return [
            (row["task_index"], row["mode"], row["prediction"])
            for row in read_jsonl(path)
        ]

    assert prediction_key(
        left_root / "out" / "predictions.jsonl"
    ) == prediction_key(right_root / "out" / "predictions.jsonl")


def test_prepare_rejects_empty_eligible_prior_history(tmp_path):
    def row(user, item, stars, day, text):
        return raw_row(user, item, stars, day, text)

    source = tmp_path / "prior.jsonl"
    write_jsonl(source, [
        row("RAW-USER-01", "RAW-ITEM-A1", 5, 1, "SHARED_ALPHA"),
        row("RAW-USER-02", "RAW-ITEM-B1", 4, 1, "SHARED_BETA"),
        row("RAW-USER-01", "RAW-ITEM-C1", 3, 2, "SHARED_ALPHA"),
        row("RAW-USER-02", "RAW-ITEM-D1", 2, 2, "SHARED_BETA"),
        row("RAW-USER-01", "RAW-ITEM-E1", 5, 3, "E3 body"),
        row("RAW-USER-02", "RAW-ITEM-F1", 1, 3, "F3 body"),
        row("RAW-USER-01", "RAW-ITEM-G1", 4, 4, "G4 body"),
        row("RAW-USER-02", "RAW-ITEM-H1", 3, 4, "H4 body"),
    ])
    work = tmp_path / "work"
    with pytest.raises(AmazonP0Error, match="no_eligible_prior_history"):
        prepare_dataset(
            source, work, history_ratio=0.25, max_tasks=50, seed=42,
            timestamp_semantics_confirmed=True,
        )
    assert not work.exists()


def test_oversized_numeric_literals_fail_safely(tmp_path, capsys):
    huge = 10 ** 400
    source = tmp_path / "huge.jsonl"
    write_jsonl(source, [
        raw_row("RAW-USER-01", "RAW-ITEM-A1", 5, 1, "ok body"),
        {
            "reviewerID": "RAW-USER-02", "asin": "RAW-ITEM-A2",
            "reviewText": "body", "overall": huge,
            "unixReviewTime": epoch(2),
        },
        {
            "reviewerID": "RAW-USER-03", "asin": "RAW-ITEM-A3",
            "reviewText": "body", "overall": 4, "unixReviewTime": huge,
        },
    ])
    records, audit = load_source_records(source)
    assert len(records) == 1
    assert audit["rejected_by_reason"] == {
        "invalid_stars": 1, "invalid_timestamp": 1,
    }
    record, reason = normalize_amazon_row({
        "reviewerID": "raw", "asin": "raw-item", "reviewText": "body",
        "overall": huge, "unixReviewTime": 1,
    }, 0)
    assert record is None and reason == "invalid_stars"

    work = tmp_path / "work"
    assert main([
        "prepare", "--source", str(source), "--work-dir", str(work),
        "--history-ratio", "0.5", "--max-tasks", "50", "--seed", "42",
        "--confirm-target-timestamp-semantics",
    ]) == 2
    stderr = capsys.readouterr().err
    assert "p0 failed: insufficient_day_groups" in stderr
    assert "Traceback" not in stderr
    assert not work.exists()


def test_provider_contexts_withhold_every_target_copy_and_future_row(tmp_path):
    _, work, _ = prepare_fixture(tmp_path)
    tasks = read_jsonl(work / "tasks" / "p0.jsonl")
    groundtruth = read_jsonl(work / "groundtruth" / "p0.jsonl")
    runtime = read_jsonl(work / "runtime_reviews.jsonl")
    target_ids = {row["target_review_id"] for row in tasks}
    target_indexes = {
        position for position, row in enumerate(runtime)
        if row["review_id"] in target_ids
    }
    target_texts = {row["text"] for row in groundtruth}

    provider = TemporalContextProvider(
        work / "tasks", work / "groundtruth",
        [work / "runtime_reviews.jsonl"], work / "manifest.json",
    )
    contexts = provider.contexts_in_execution_order()
    assert len(contexts) == 15
    for context in contexts:
        visible = list(context.history) + list(context.item_references)
        visible_indexes = {
            row["_temporal_source_row_index"] for row in visible
        }
        assert visible_indexes.isdisjoint(target_indexes)
        for row in visible:
            assert row["_temporal_event_time"] < context.target_timestamp
            assert row["text"] not in target_texts

    # The duplicate copy of the (7, user-1, X1) target text is a history row
    # for another user, yet it must not appear in that target's item refs.
    x1_target = next(
        context for context in contexts
        if context.task_index == 0
    )
    assert [
        row["_temporal_source_row_index"]
        for row in x1_target.item_references
    ] == [2]
    assert len(x1_target.item_references) == 1


def test_provider_rejects_changed_manifest_or_review_source(tmp_path):
    _, work, _ = prepare_fixture(tmp_path)

    def provider(root):
        return TemporalContextProvider(
            root / "tasks", root / "groundtruth",
            [root / "runtime_reviews.jsonl"], root / "manifest.json",
        )

    provider(work)  # baseline sanity: unchanged sources pass
    changed_manifest = tmp_path / "changed_manifest"
    shutil.copytree(work, changed_manifest)
    manifest = json.loads(
        (changed_manifest / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["split_policy"]["train_ratio"] = 0.7
    (changed_manifest / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(TemporalManifestError):
        provider(changed_manifest)

    changed_source = tmp_path / "changed_source"
    shutil.copytree(work, changed_source)
    lines = (changed_source / "runtime_reviews.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    payload = json.loads(lines[0])
    payload["text"] = "mutated text"
    lines[0] = json.dumps(payload)
    (changed_source / "runtime_reviews.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    with pytest.raises(TemporalManifestError):
        provider(changed_source)


def test_prepare_failure_exits_nonzero_safely_without_partial_dir(
    tmp_path, capsys,
):
    sparse = tmp_path / "sparse.jsonl"
    write_jsonl(sparse, [
        raw_row("RAW-USER-07", "RAW-ITEM-A1", 4, 1, "private body one"),
        raw_row("RAW-USER-08", "RAW-ITEM-A2", 3, 1, "private body two"),
        raw_row("RAW-USER-01", "RAW-ITEM-A3", 4, 2, "private body three"),
        raw_row("RAW-USER-02", "RAW-ITEM-A4", 3, 2, "private body four"),
        raw_row("RAW-USER-01", "RAW-ITEM-A5", 5, 3, "private body five"),
        raw_row("RAW-USER-03", "RAW-ITEM-A6", 2, 3, "private body six"),
    ])
    work = tmp_path / "work"
    assert main([
        "prepare", "--source", str(sparse), "--work-dir", str(work),
        "--history-ratio", "0.5", "--max-tasks", "50", "--seed", "42",
        "--confirm-target-timestamp-semantics",
    ]) == 2
    stderr = capsys.readouterr().err
    assert "p0 failed: insufficient_selected_users" in stderr
    assert str(tmp_path) not in stderr
    assert "RAW-USER" not in stderr and "private body" not in stderr
    assert not work.exists()


def test_prepare_cli_requires_confirmation_and_reports_parse_category(
    tmp_path, capsys,
):
    source = make_source(tmp_path)
    work = tmp_path / "work"
    assert main([
        "prepare", "--source", str(source), "--work-dir", str(work),
    ]) == 2
    assert "confirmation_required" in capsys.readouterr().err
    assert not work.exists()

    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    assert main([
        "prepare", "--source", str(broken), "--work-dir", str(work),
        "--confirm-target-timestamp-semantics",
    ]) == 2
    stderr = capsys.readouterr().err
    assert "p0 failed: source_parse_failed" in stderr
    assert str(broken) not in stderr
    assert not work.exists()


# ---------------------------------------------------------------------------
# Run: baselines, report privacy, determinism, failure handling.
# ---------------------------------------------------------------------------


def test_run_cli_writes_reproducible_redacted_report(tmp_path, capsys):
    _, work, _ = prepare_fixture(tmp_path)
    first = tmp_path / "report_a"
    second = tmp_path / "report_b"
    for output in (first, second):
        assert main([
            "run", "--prepared-dir", str(work), "--output-dir", str(output),
            "--n-boot", "200", "--seed", "42",
        ]) == 0

    report = json.loads((first / "report.json").read_text(encoding="utf-8"))
    assert report["task_counts"]["total"] == 15
    assert report["task_counts"]["by_split"] == {
        "train": 9, "validation": 2, "test": 4,
    }
    assert report["task_counts"]["user_clusters"]["by_split"]["test"] == 4
    assert report["bootstrap_test"]["status"] == "computed"
    assert report["bootstrap_test"]["cluster_count"] == 4
    assert report["bootstrap_test"]["resampling_unit"] == (
        "user_cluster_with_replacement"
    )
    assert "not a partition of the raw source file" in report["protocol"][
        "split_scope_note"
    ]
    assert "not Memory or LLM results" in report["interpretation"]
    assert (first / "report.md").exists()
    assert (first / "predictions.jsonl").exists()
    assert (first / "scoring.jsonl").exists()

    # Same inputs and seed reproduce byte-identical aggregate reports.
    assert (first / "report.json").read_bytes() == (
        second / "report.json"
    ).read_bytes()

    # Aggregate outputs carry no raw identifiers, text, or local paths.
    rendered = json.dumps(report) + (first / "report.md").read_text(
        encoding="utf-8"
    )
    for marker in (*RAW_MARKERS, str(tmp_path)):
        assert marker not in rendered
    for row in read_jsonl(work / "normalized_reviews.jsonl")[:5]:
        assert row["user_id"] not in rendered
        assert row["item_id"] not in rendered
        assert row["review_id"] not in rendered


def test_run_is_idempotent_for_predictions_under_label_perturbation(
    tmp_path,
):
    _, work, _ = prepare_fixture(tmp_path)
    baseline_out = tmp_path / "baseline"
    run_baselines(work, baseline_out, n_boot=200, seed=42)

    perturbed = tmp_path / "perturbed"
    shutil.copytree(work, perturbed)
    truth_path = perturbed / "groundtruth" / "p0.jsonl"
    truth_rows = read_jsonl(truth_path)
    for row in truth_rows:
        row["stars"] = 3.0
    write_jsonl(truth_path, truth_rows)
    perturbed_out = tmp_path / "perturbed_out"
    run_baselines(perturbed, perturbed_out, n_boot=200, seed=42)

    def prediction_key(path):
        return [
            (row["task_index"], row["mode"], row["prediction"])
            for row in read_jsonl(path)
        ]

    assert prediction_key(baseline_out / "predictions.jsonl") == prediction_key(
        perturbed_out / "predictions.jsonl"
    )
    baseline_scoring = read_jsonl(baseline_out / "scoring.jsonl")
    perturbed_scoring = read_jsonl(perturbed_out / "scoring.jsonl")
    assert any(
        left["actual"] != right["actual"]
        for left, right in zip(baseline_scoring, perturbed_scoring)
    )


def test_run_fails_closed_with_safe_category_and_cleans_output(
    tmp_path, capsys,
):
    _, work, _ = prepare_fixture(tmp_path)

    stale = tmp_path / "stale"
    shutil.copytree(work, stale)
    lines = (stale / "runtime_reviews.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    payload = json.loads(lines[0])
    payload["text"] = "mutated private body"
    lines[0] = json.dumps(payload)
    (stale / "runtime_reviews.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    output = tmp_path / "stale_out"
    assert main([
        "run", "--prepared-dir", str(stale), "--output-dir", str(output),
        "--n-boot", "200", "--seed", "42",
    ]) == 2
    stderr = capsys.readouterr().err
    assert "p0 failed: runtime_manifest_rejected" in stderr
    assert str(tmp_path) not in stderr
    assert "mutated private body" not in stderr
    assert not output.exists()

    tampered = tmp_path / "tampered"
    shutil.copytree(work, tampered)
    policy = json.loads(
        (tampered / "p0_policy.json").read_text(encoding="utf-8")
    )
    policy["protocol"]["global_prior"] = 1.0
    (tampered / "p0_policy.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    with pytest.raises(AmazonP0Error, match="global_prior_mismatch"):
        run_baselines(
            tampered, tmp_path / "tampered_out", n_boot=200, seed=42
        )
    assert not (tmp_path / "tampered_out").exists()

    missing = tmp_path / "missing_out"
    with pytest.raises(AmazonP0Error, match="prepared_dir_missing"):
        run_baselines(tmp_path / "absent", missing, n_boot=200, seed=42)


def test_run_rejects_tampered_policy_parameters_and_protocol(
    tmp_path, capsys,
):
    _, work, _ = prepare_fixture(tmp_path)

    def tampered_copy(name, mutate):
        target = tmp_path / name
        shutil.copytree(work, target)
        policy = json.loads(
            (target / "p0_policy.json").read_text(encoding="utf-8")
        )
        mutate(policy)
        (target / "p0_policy.json").write_text(
            json.dumps(policy), encoding="utf-8"
        )
        return target

    def run(source, output):
        return run_baselines(source, output, n_boot=200, seed=42)

    lambda_case = tampered_copy(
        "lambda_case",
        lambda policy: policy["baselines"].update(lambda_user=999),
    )
    with pytest.raises(AmazonP0Error, match="policy_parameter_mismatch"):
        run(lambda_case, tmp_path / "lambda_out")
    assert not (tmp_path / "lambda_out").exists()

    weight_case = tampered_copy(
        "weight_case",
        lambda policy: policy["baselines"].update(weight_user=0.9),
    )
    with pytest.raises(AmazonP0Error, match="policy_parameter_mismatch"):
        run(weight_case, tmp_path / "weight_out")

    max_tasks_case = tampered_copy(
        "max_tasks_case",
        lambda policy: policy["protocol"].update(max_tasks=6),
    )
    with pytest.raises(AmazonP0Error, match="policy_artifact_mismatch"):
        run(max_tasks_case, tmp_path / "max_tasks_out")

    # A budget-constrained selection makes the recorded seed observable.
    seed_root = tmp_path / "seed_root"
    seed_work = seed_root / "work"
    prepare_dataset(
        make_source(seed_root), seed_work,
        history_ratio=0.5, max_tasks=8, seed=42,
        timestamp_semantics_confirmed=True,
    )
    seed_policy = json.loads(
        (seed_work / "p0_policy.json").read_text(encoding="utf-8")
    )
    records = read_jsonl(seed_work / "normalized_reviews.jsonl")
    _, candidates, _ = amazon_p0.split_frozen_history(records, 0.5)
    base_ids = [
        record["review_id"]
        for record in amazon_p0.select_targets(
            candidates, max_tasks=8, seed=42
        )
    ]
    differing_seed = next(
        candidate
        for candidate in range(1, 2000)
        if [
            record["review_id"]
            for record in amazon_p0.select_targets(
                candidates, max_tasks=8, seed=candidate
            )
        ] != base_ids
    )
    seed_policy["protocol"]["seed"] = differing_seed
    (seed_work / "p0_policy.json").write_text(
        json.dumps(seed_policy), encoding="utf-8"
    )
    with pytest.raises(AmazonP0Error, match="policy_artifact_mismatch"):
        run(seed_work, tmp_path / "seed_out")

    frozen_case = tampered_copy(
        "frozen_case",
        lambda policy: policy["protocol"].update(frozen_history=False),
    )
    with pytest.raises(AmazonP0Error, match="policy_protocol_mismatch"):
        run(frozen_case, tmp_path / "frozen_out")

    history_case = tampered_copy(
        "history_case",
        lambda policy: policy["protocol"].update(history_rows=99),
    )
    with pytest.raises(AmazonP0Error, match="global_prior_source_mismatch"):
        run(history_case, tmp_path / "history_out")

    split_case = tampered_copy(
        "split_case",
        lambda policy: policy["protocol"]["manifest_split_counts"].update(
            test=1
        ),
    )
    with pytest.raises(AmazonP0Error, match="policy_artifact_mismatch"):
        run(split_case, tmp_path / "split_out")
    assert not (tmp_path / "split_out").exists()
    capsys.readouterr()


def test_run_rebuilds_report_without_policy_free_text(tmp_path, capsys):
    _, work, _ = prepare_fixture(tmp_path)
    tampered = tmp_path / "notes_case"
    shutil.copytree(work, tampered)
    policy = json.loads(
        (tampered / "p0_policy.json").read_text(encoding="utf-8")
    )
    marker = "SYNTHETIC_PRIVATE_MARKER"
    policy["protocol"]["observed_history_note"] = marker
    policy["protocol"]["runtime_source_composition"] = marker
    policy["protocol"]["target_visibility"] = marker
    policy["protocol"]["global_prior_source"] = marker
    policy["protocol"]["selection_policy"] = marker
    policy["source"]["license_note"] = marker
    (tampered / "p0_policy.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    output = tmp_path / "notes_out"
    assert main([
        "run", "--prepared-dir", str(tampered), "--output-dir", str(output),
        "--n-boot", "200", "--seed", "42",
    ]) == 0
    stdout = capsys.readouterr().out
    report_text = (output / "report.json").read_text(encoding="utf-8")
    markdown_text = (output / "report.md").read_text(encoding="utf-8")
    assert marker not in stdout + report_text + markdown_text
    report = json.loads(report_text)
    assert report["protocol"]["observed_history_note"] == (
        amazon_p0.OBSERVED_HISTORY_NOTE
    )
    assert report["protocol"]["runtime_source_composition"] == (
        amazon_p0.RUNTIME_SOURCE_COMPOSITION
    )
    assert report["protocol"]["target_visibility"] == (
        amazon_p0.TARGET_VISIBILITY
    )
    assert report["protocol"]["global_prior_source"] == (
        amazon_p0.GLOBAL_PRIOR_SOURCE
    )
    assert report["protocol"]["baseline_params"]["lambda_user"] == 5.0


def test_artifact_gate_runs_before_parsing_or_provider(tmp_path, monkeypatch):
    _, work, _ = prepare_fixture(tmp_path)
    prepared = tmp_path / "oversized"
    shutil.copytree(work, prepared)
    tracked = [
        prepared / name for name in (
            "runtime_reviews.jsonl",
            "normalized_reviews.jsonl",
            "tasks/p0.jsonl",
            "groundtruth/p0.jsonl",
        )
    ]
    limit = max(path.stat().st_size for path in tracked)
    runtime = prepared / "runtime_reviews.jsonl"
    runtime.write_bytes(
        runtime.read_bytes() + b"\n" + b"x" * (limit + 100)
    )
    monkeypatch.setattr(
        amazon_p0, "TEMPORAL_MAX_REVIEW_SOURCE_BYTES", limit
    )

    calls = {"jsonl_reads": 0, "provider": 0}
    original_read_jsonl = amazon_p0._read_jsonl

    def spy_read_jsonl(*args, **kwargs):
        calls["jsonl_reads"] += 1
        return original_read_jsonl(*args, **kwargs)

    def spy_provider(*args, **kwargs):
        calls["provider"] += 1
        raise AssertionError("provider constructed before the artifact gate")

    monkeypatch.setattr(amazon_p0, "_read_jsonl", spy_read_jsonl)
    monkeypatch.setattr(
        amazon_p0, "TemporalContextProvider", spy_provider
    )
    output = tmp_path / "oversized_out"
    with pytest.raises(AmazonP0Error, match="artifact_limit_exceeded"):
        run_baselines(prepared, output, n_boot=200, seed=42)
    assert calls == {"jsonl_reads": 0, "provider": 0}
    assert not output.exists()


def test_artifact_gate_counts_rows_before_parsing(tmp_path, monkeypatch):
    _, work, _ = prepare_fixture(tmp_path)
    monkeypatch.setattr(amazon_p0, "TEMPORAL_MAX_REVIEW_ROWS", 5)
    calls = {"jsonl_reads": 0}
    original_read_jsonl = amazon_p0._read_jsonl

    def spy_read_jsonl(*args, **kwargs):
        calls["jsonl_reads"] += 1
        return original_read_jsonl(*args, **kwargs)

    monkeypatch.setattr(amazon_p0, "_read_jsonl", spy_read_jsonl)
    output = tmp_path / "row_out"
    with pytest.raises(AmazonP0Error, match="artifact_row_limit_exceeded"):
        run_baselines(work, output, n_boot=200, seed=42)
    assert calls["jsonl_reads"] == 0
    assert not output.exists()


# ---------------------------------------------------------------------------
# Download gates (stubbed transport; never the real network in tests).
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, payload: bytes, url: str):
        self.payload = payload
        self.url = url
        self.offset = 0

    def read(self, size):
        chunk = self.payload[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


GZIP_PAYLOAD = gzip.compress(b'{"reviewerID": "x"}\n')
PINNED = amazon_p0.SOURCE_FILE_URL


def fake_opener(payload: bytes, url: str):
    def opener(request, timeout=None):
        return FakeResponse(payload, url)

    return opener


def test_download_requires_pinned_url_bounded_limit_and_fresh_output(
    tmp_path,
):
    with pytest.raises(AmazonP0Error, match="download_source_not_allowed"):
        download_source(
            "http://snap.stanford.edu/x.json.gz", tmp_path / "a.json.gz",
            opener=fake_opener(GZIP_PAYLOAD, "http://snap.stanford.edu/x.json.gz"),
        )
    other_official = (
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
        "reviews_Books_5.json.gz"
    )
    with pytest.raises(AmazonP0Error, match="download_source_not_allowed"):
        download_source(
            other_official, tmp_path / "b.json.gz",
            opener=fake_opener(GZIP_PAYLOAD, other_official),
        )
    with pytest.raises(AmazonP0Error, match="invalid_download_limit"):
        download_source(
            PINNED, tmp_path / "c.json.gz", max_bytes=0,
            opener=fake_opener(GZIP_PAYLOAD, PINNED),
        )
    with pytest.raises(AmazonP0Error, match="invalid_download_limit"):
        download_source(
            PINNED, tmp_path / "d.json.gz",
            max_bytes=amazon_p0.MAX_DOWNLOAD_BYTES + 1,
            opener=fake_opener(GZIP_PAYLOAD, PINNED),
        )
    existing = tmp_path / "exists.json.gz"
    existing.write_bytes(b"keep me")
    with pytest.raises(AmazonP0Error, match="output_file_exists"):
        download_source(
            PINNED, existing, opener=fake_opener(GZIP_PAYLOAD, PINNED),
        )
    assert existing.read_bytes() == b"keep me"
    part = tmp_path / "staged.json.gz"
    (tmp_path / "staged.json.gz.part").write_bytes(b"stale")
    with pytest.raises(AmazonP0Error, match="output_part_file_exists"):
        download_source(
            PINNED, part, opener=fake_opener(GZIP_PAYLOAD, PINNED),
        )
    assert not part.exists()


def test_download_enforces_size_redirect_and_gzip_integrity(tmp_path):
    with pytest.raises(AmazonP0Error, match="download_size_limit_exceeded"):
        download_source(
            PINNED, tmp_path / "big.json.gz", max_bytes=10,
            opener=fake_opener(GZIP_PAYLOAD, PINNED),
        )
    assert not (tmp_path / "big.json.gz").exists()
    assert not list(tmp_path.glob("*.part"))

    with pytest.raises(AmazonP0Error, match="download_invalid_gzip"):
        download_source(
            PINNED, tmp_path / "not.gz", max_bytes=1024,
            opener=fake_opener(b"\x1f\x8bnot-a-valid-gzip", PINNED),
        )
    assert not list(tmp_path.glob("*.part"))

    with pytest.raises(AmazonP0Error, match="download_invalid_gzip"):
        download_source(
            PINNED, tmp_path / "truncated.json.gz", max_bytes=1024,
            opener=fake_opener(GZIP_PAYLOAD[:-5], PINNED),
        )
    assert not (tmp_path / "truncated.json.gz").exists()
    assert not list(tmp_path.glob("*.part"))

    with pytest.raises(AmazonP0Error, match="download_redirect_not_allowed"):
        download_source(
            PINNED, tmp_path / "redirected.json.gz", max_bytes=1024,
            opener=fake_opener(
                GZIP_PAYLOAD, "http://snap.stanford.edu/downgraded.gz"
            ),
        )
    assert not list(tmp_path.glob("*.part"))


def test_download_enforces_decompression_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(
        amazon_p0, "TEMPORAL_MAX_REVIEW_SOURCE_BYTES",
        len(GZIP_PAYLOAD) + 1,
    )
    bomb = gzip.compress(b"x" * (len(GZIP_PAYLOAD) + 100))
    destination = tmp_path / "bomb.json.gz"
    with pytest.raises(
        AmazonP0Error, match="download_decompression_limit_exceeded"
    ):
        download_source(
            PINNED, destination, max_bytes=1024,
            opener=fake_opener(bomb, PINNED),
        )
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_stub_writes_exact_bytes_and_hash(tmp_path):
    info = download_source(
        PINNED, tmp_path / "ok.json.gz", max_bytes=1024,
        opener=fake_opener(GZIP_PAYLOAD, PINNED),
    )
    assert info["bytes"] == len(GZIP_PAYLOAD)
    assert info["sha256"] == hashlib.sha256(GZIP_PAYLOAD).hexdigest()
    assert info["decompressed_bytes"] == len(b'{"reviewerID": "x"}\n')
    assert (tmp_path / "ok.json.gz").read_bytes() == GZIP_PAYLOAD
    assert not list(tmp_path.glob("*.part"))


# ---------------------------------------------------------------------------
# Offline guarantees and repository hygiene.
# ---------------------------------------------------------------------------


def test_normal_pipeline_attempts_no_network(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("the P0 tests must never reach the network")

    monkeypatch.setattr(amazon_p0.urllib.request, "urlopen", forbidden)
    source = make_source(tmp_path)
    work = tmp_path / "work"
    assert main([
        "prepare", "--source", str(source), "--work-dir", str(work),
        "--history-ratio", "0.5", "--max-tasks", "50", "--seed", "42",
        "--confirm-target-timestamp-semantics",
    ]) == 0
    assert main([
        "run", "--prepared-dir", str(work),
        "--output-dir", str(tmp_path / "out"),
        "--n-boot", "200", "--seed", "42",
    ]) == 0
    capsys.readouterr()
    assert "websocietysimulator" not in sys.modules
    assert not hasattr(amazon_p0, "ImprovedSimulationAgent")


def test_local_asset_roots_are_gitignored():
    lines = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(
            encoding="utf-8"
        ).splitlines()
    }
    assert "Dataset/" in lines
    assert "results/" in lines
