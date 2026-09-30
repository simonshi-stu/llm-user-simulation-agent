#!/usr/bin/env python3
"""Privacy-conscious, no-LLM audit tools for user-memory evaluations.

The CLI prints aggregate counts and numeric diagnostics only. It never prints
task/user/item identifiers, review text, or prompts. Generated manifests contain
source row indexes and hashes, not source records or raw identifiers.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


USER_FIELDS = ("user_id", "reviewer_id", "uid")
ITEM_FIELDS = ("item_id", "business_id", "product_id", "book_id", "iid")
REVIEW_ID_FIELDS = ("target_review_id", "review_id", "reviewId")
DATE_FIELDS = (
    "target_timestamp", "target_time", "target_date", "timestamp",
    "created_at", "date", "review_time", "time",
)
TEXT_FIELDS = ("review_text", "text", "content", "review")
STAR_FIELDS = ("stars", "rating", "score")
ROOT_ROW_FIELDS = frozenset(
    (*USER_FIELDS, *ITEM_FIELDS, *REVIEW_ID_FIELDS, *DATE_FIELDS,
     *TEXT_FIELDS, "stars", "rating", "score", "type", "task_index",
     "task_idx", "index", "prompt", "prompt_text", "messages")
)


class _JSONStream:
    """Incremental JSON decoder for JSON arrays and JSON-lines files."""

    def __init__(self, handle, chunk_size: int = 64 * 1024):
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _fill(self) -> None:
        if self.eof:
            return
        self.buffer = self.buffer[self.position:]
        self.position = 0
        chunk = self.handle.read(self.chunk_size)
        if not chunk:
            self.eof = True
            return
        self.buffer += chunk

    def _skip_space(self) -> None:
        while True:
            while (self.position < len(self.buffer)
                   and self.buffer[self.position].isspace()):
                self.position += 1
            if self.position < len(self.buffer) or self.eof:
                return
            self._fill()

    def peek(self) -> str:
        self._skip_space()
        if self.position >= len(self.buffer):
            return ""
        return self.buffer[self.position]

    def pop(self) -> str:
        value = self.peek()
        if value:
            self.position += 1
        return value

    def decode_value(self) -> Any:
        while True:
            self._skip_space()
            if self.position >= len(self.buffer):
                raise ValueError("unexpected end of JSON input")
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self._fill()
                continue
            self.position = end
            return value

    def iter_array(self) -> Iterator[Any]:
        if self.pop() != "[":
            raise ValueError("expected JSON array")
        if self.peek() == "]":
            self.pop()
            return
        while True:
            yield self.decode_value()
            separator = self.pop()
            if separator == "]":
                return
            if separator != ",":
                raise ValueError("expected ',' or ']' in JSON array")

    def iter_root_values(self) -> Iterator[Any]:
        first = self.peek()
        if first == "[":
            yield from self.iter_array()
            return
        if first != "{":
            yield self.decode_value()
            return

        self.pop()  # opening object
        if self.peek() == "}":
            self.pop()
            return

        key = self.decode_value()
        if not isinstance(key, str) or self.pop() != ":":
            raise ValueError("invalid top-level JSON object")

        # A JSON object whose first field identifies a record is a single row.
        # Otherwise stream root mapping/container values one at a time.
        if key in ROOT_ROW_FIELDS:
            row = {key: self.decode_value()}
            while True:
                separator = self.pop()
                if separator == "}":
                    break
                if separator != ",":
                    raise ValueError("expected ',' or '}' in JSON object")
                name = self.decode_value()
                if not isinstance(name, str) or self.pop() != ":":
                    raise ValueError("invalid JSON object field")
                row[name] = self.decode_value()
            yield row
            return

        while True:
            if self.peek() == "[":
                yield from self.iter_array()
            else:
                yield self.decode_value()
            separator = self.pop()
            if separator == "}":
                return
            if separator != ",":
                raise ValueError("expected ',' or '}' in JSON object")
            key = self.decode_value()
            if not isinstance(key, str) or self.pop() != ":":
                raise ValueError("invalid JSON object field")


def _natural_key(path: Path):
    normalized = str(path).replace("\\", "/").casefold()
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", normalized)]


def _open_json(path: Path):
    if path.name.casefold().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _flatten_records(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if _looks_like_record(value):
            yield value
            return
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from _flatten_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _flatten_records(child)


def _looks_like_record(value: dict[str, Any]) -> bool:
    if any(field in value for field in (*USER_FIELDS, *ITEM_FIELDS)):
        return True
    if any(field in value for field in STAR_FIELDS):
        return True
    if any(field in value for field in ("task_index", "task_idx", "index")):
        return True
    if any(field in value for field in ("prompt", "prompt_text", "messages")):
        return True
    return "type" in value and any(
        key in value for key in ("task", "target", "input")
    )


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield record-shaped objects from JSON arrays, JSONL, or JSON objects.

    Arrays are decoded one row at a time, so a large review file does not need
    to fit in memory. JSONL is also streamed line by line. A single JSON object
    (or one nested record) is held only for the duration of that row.
    """
    source = Path(path)
    lower = source.name.casefold()
    if lower.endswith((".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")):
        with _open_json(source) as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield from _flatten_records(json.loads(line))
        return

    with _open_json(source) as handle:
        stream = _JSONStream(handle)
        for value in stream.iter_root_values():
            yield from _flatten_records(value)


def _files(path: str | Path) -> list[Path]:
    value = Path(path)
    if value.is_file():
        return [value]
    if not value.is_dir():
        return []
    return sorted(
        (candidate for candidate in value.rglob("*")
         if candidate.is_file()
         and candidate.name.casefold().endswith(
             (".json", ".jsonl", ".ndjson", ".json.gz", ".jsonl.gz",
              ".ndjson.gz")
         )),
        key=lambda candidate: _natural_key(
            Path(candidate.relative_to(value).as_posix())
        ),
    )


def _read_directory_records(path: str | Path) -> tuple[list[dict], int]:
    paths = _files(path)
    records: list[dict] = []
    for source in paths:
        records.extend(iter_records(source))
    return records, len(paths)


def _field(value: Any, names: Iterable[str]) -> Any:
    wanted = tuple(names)
    if isinstance(value, dict):
        for name in wanted:
            if name in value and value[name] is not None:
                return value[name]
        for child in value.values():
            found = _field(child, wanted)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _field(child, wanted)
            if found is not None:
                return found
    return None


def _field_name(value: Any, names: Iterable[str]) -> str | None:
    wanted = tuple(names)
    if isinstance(value, dict):
        for name in wanted:
            if name in value and value[name] is not None:
                return name
        for child in value.values():
            found = _field_name(child, wanted)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _field_name(child, wanted)
            if found is not None:
                return found
    return None


def _as_id(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    normalized = str(value).strip()
    return normalized or None


def _review_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    found = _field(value, TEXT_FIELDS)
    if isinstance(found, str):
        return found.strip() or None
    return None


def _text_digest(text: str | None) -> str | None:
    if not text:
        return None
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _parse_time(value: Any) -> datetime | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if abs(number) > 1e12:
            number /= 1000.0
        try:
            return datetime.fromtimestamp(number, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if re.fullmatch(r"\d{10,13}(?:\.\d+)?", candidate):
        try:
            return _parse_time(float(candidate))
        except ValueError:
            return None
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _extract_task(record: dict[str, Any], index: int) -> dict[str, Any]:
    raw_time = _field(record, DATE_FIELDS)
    review_id = _as_id(_field(record, REVIEW_ID_FIELDS))
    return {
        "index": index,
        "user_id": _as_id(_field(record, USER_FIELDS)),
        "item_id": _as_id(_field(record, ITEM_FIELDS)),
        "target_review_id": review_id,
        "source_task_index": _as_id(
            _field(record, ("task_index", "task_idx", "index", "idx"))
        ),
        "timestamp": _parse_time(raw_time),
        "timestamp_source_field": _field_name(record, DATE_FIELDS),
        "target_text_digest": _text_digest(_review_text(
            _field(record, TEXT_FIELDS)
        )),
        "target_stars": _field(record, STAR_FIELDS),
    }


def _load_task_pairs(
    task_dir: str | Path,
    groundtruth_dir: str | Path,
) -> tuple[list[tuple[dict, dict]], dict[str, Any]]:
    task_files = _files(task_dir)
    truth_files = _files(groundtruth_dir)
    diagnostics = {
        "task_files": len(task_files),
        "groundtruth_files": len(truth_files),
        "alignment": "natural_filename_order",
        "warnings": [],
    }
    if not task_files or not truth_files:
        diagnostics["warnings"].append("task_or_groundtruth_files_missing")
        return [], diagnostics
    if len(task_files) != len(truth_files):
        diagnostics["warnings"].append("task_groundtruth_file_count_mismatch")
        return [], diagnostics

    pairs: list[tuple[dict, dict]] = []
    for task_file, truth_file in zip(task_files, truth_files):
        if task_file.stem != truth_file.stem:
            diagnostics["alignment"] = "natural_filename_order_assumed"
            if "filename_stems_differ" not in diagnostics["warnings"]:
                diagnostics["warnings"].append("filename_stems_differ")
        task_rows = list(iter_records(task_file))
        truth_rows = list(iter_records(truth_file))
        if len(task_rows) != len(truth_rows):
            diagnostics["warnings"].append("paired_file_record_count_mismatch")
            return [], diagnostics
        pairs.extend(zip(task_rows, truth_rows))
    diagnostics["task_count"] = len(pairs)
    return pairs, diagnostics


def _review_row(record: dict[str, Any], row_index: int) -> dict[str, Any]:
    user_id = _as_id(_field(record, USER_FIELDS))
    item_id = _as_id(_field(record, ITEM_FIELDS))
    raw_id = _as_id(_field(record, ("review_id", "reviewId", "id")))
    timestamp = _parse_time(_field(record, DATE_FIELDS))
    text_digest = _text_digest(_review_text(_field(record, TEXT_FIELDS)))
    stars = _field(record, STAR_FIELDS)
    if raw_id is not None:
        key_payload = f"id\0{raw_id}"
    else:
        key_payload = "\0".join((
            user_id or "", item_id or "", timestamp.isoformat() if timestamp else "",
            str(stars or ""), text_digest or "", str(row_index),
        ))
    return {
        "row_index": row_index,
        "row_key": hashlib.sha256(key_payload.encode("utf-8")).hexdigest(),
        "raw_id": raw_id,
        "user_id": user_id,
        "item_id": item_id,
        "timestamp": timestamp,
        "text_digest": text_digest,
        "stars": stars,
    }


def _load_reviews(
    paths: Iterable[str | Path],
    *,
    relevant_users: set[str] | None = None,
    relevant_items: set[str] | None = None,
) -> tuple[list[dict], int, int]:
    rows: list[dict] = []
    file_count = 0
    row_count = 0
    for value in paths:
        for path in _files(value):
            file_count += 1
            for record in iter_records(path):
                user_id = _as_id(_field(record, USER_FIELDS))
                item_id = _as_id(_field(record, ITEM_FIELDS))
                if relevant_users is not None or relevant_items is not None:
                    if (
                        user_id not in (relevant_users or set())
                        and item_id not in (relevant_items or set())
                    ):
                        row_count += 1
                        continue
                rows.append(_review_row(record, row_count))
                row_count += 1
    return rows, file_count, row_count


def _count_user_sequence(tasks: list[dict]) -> dict[str, Any]:
    seen: set[str] = set()
    seen_counts: Counter[str] = Counter()
    memory_counts: Counter[str] = Counter()
    repeat_tasks = 0
    repeated_user_tasks = 0
    potential_recall_tasks = 0
    predicted_recall_tasks = 0
    max_user_tasks = 0

    for task in tasks:
        user_id = task.get("user_id")
        if not user_id:
            continue
        if user_id in seen:
            repeat_tasks += 1
            repeated_user_tasks += 1
            potential_recall_tasks += 1
        else:
            seen.add(user_id)
        seen_counts[user_id] += 1
        available = memory_counts[user_id]
        if available:
            predicted_recall_tasks += 1
        memory_counts[user_id] = min(8, available + 1)
        max_user_tasks = max(max_user_tasks, seen_counts[user_id])

    unique_users = len(seen)
    return {
        "tasks_with_user_id": sum(bool(task.get("user_id")) for task in tasks),
        "distinct_users": unique_users,
        "users_with_repeated_tasks": sum(count > 1 for count in seen_counts.values()),
        "tasks_after_first_user_occurrence": repeat_tasks,
        "repeat_user_task_rate": (
            repeat_tasks / sum(bool(task.get("user_id")) for task in tasks)
            if any(task.get("user_id") for task in tasks) else None
        ),
        "sequential_potential_recall_tasks": potential_recall_tasks,
        "sequential_recall_tasks_with_memory_limit_8": predicted_recall_tasks,
        "maximum_tasks_for_one_user": max_user_tasks,
        "recall_order_assumption": (
            "natural source task order; serial execution; generated memory only"
        ),
    }


def _matching_review_rows(task: dict, rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if (task.get("user_id") is None or row["user_id"] == task["user_id"])
        and (task.get("item_id") is None or row["item_id"] == task["item_id"])
    ]


def _leakage_evidence(
    pairs: list[tuple[dict, dict]], rows: list[dict],
    rows_scanned: int | None = None,
) -> dict:
    tasks = [
        _review_identity(task, truth, rows, index)
        for index, (task, truth) in enumerate(pairs)
    ]
    exact_target_in_user_rows = 0
    exact_target_in_item_rows = 0
    future_user_task_count = 0
    future_item_task_count = 0
    future_user_rows = 0
    future_item_rows = 0
    dated_task_count = 0
    identifiable_target_texts = 0
    target_text_in_task_payloads = 0

    for index, (task_record, _) in enumerate(pairs):
        task = tasks[index]
        task_text_digest = _text_digest(
            _review_text(_field(task_record, TEXT_FIELDS))
        )
        if (
            task.get("target_text_digest")
            and task_text_digest == task["target_text_digest"]
        ):
            target_text_in_task_payloads += 1
        if task["target_text_digest"]:
            identifiable_target_texts += 1
        relevant = _matching_review_rows(task, rows)
        user_rows = [row for row in relevant
                     if task["user_id"] is not None
                     and row["user_id"] == task["user_id"]]
        item_rows = [row for row in relevant
                     if task["item_id"] is not None
                     and row["item_id"] == task["item_id"]]
        if task["target_text_digest"]:
            if any(row["text_digest"] == task["target_text_digest"]
                   for row in user_rows):
                exact_target_in_user_rows += 1
            if any(row["text_digest"] == task["target_text_digest"]
                   for row in item_rows):
                exact_target_in_item_rows += 1
        if task["timestamp"] is None:
            continue
        dated_task_count += 1
        future_users = [row for row in user_rows
                        if row["timestamp"] and row["timestamp"] >= task["timestamp"]]
        future_items = [row for row in item_rows
                        if row["timestamp"] and row["timestamp"] >= task["timestamp"]]
        future_user_rows += len(future_users)
        future_item_rows += len(future_items)
        future_user_task_count += bool(future_users)
        future_item_task_count += bool(future_items)

    return {
        "review_rows_scanned": rows_scanned if rows_scanned is not None else len(rows),
        "relevant_review_rows_indexed": len(rows),
        "tasks_with_trusted_target_timestamp": dated_task_count,
        "tasks_with_target_review_text_for_exact_match": identifiable_target_texts,
        "target_text_found_in_task_payloads": target_text_in_task_payloads,
        "target_text_found_in_same_user_source_rows": exact_target_in_user_rows,
        "target_text_found_in_target_item_source_rows": exact_target_in_item_rows,
        "tasks_with_same_user_reviews_at_or_after_target_time": future_user_task_count,
        "same_user_reviews_at_or_after_target_time": future_user_rows,
        "tasks_with_target_item_reviews_at_or_after_target_time": future_item_task_count,
        "target_item_reviews_at_or_after_target_time": future_item_rows,
        "interpretation": {
            "confirmed": (
                "Exact target-text presence in a same-user/item source row is "
                "confirmed only when the corresponding count is nonzero."
            ),
            "not_confirmed": (
                "Source-corpus presence does not prove the simulator returned "
                "that row or that it was selected into the prompt."
            ),
            "code_path": (
                "The current agent profiles all rows returned by get_reviews(user_id) "
                "and selects target-item rows for prompt references; whether the "
                "adapter time-filters those rows must be checked separately."
            ),
            "prompt_inclusion": (
                "Not verified without an actual prompt artifact. Use --prompt-file "
                "to inspect local prompt traces without printing their contents."
            ),
        },
    }


def _prompt_string(record: dict[str, Any]) -> str:
    value = _field(record, ("prompt", "prompt_text", "content", "messages", "text"))
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_prompt_string({"content": item}) for item in value)
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
        return "\n".join(_prompt_string({"content": item}) for item in value.values())
    return ""


def _contains_normalized(container: str, candidate: str | None) -> bool:
    if not container or not candidate:
        return False
    normalized_container = " ".join(container.split()).casefold()
    normalized_candidate = " ".join(candidate.split()).casefold()
    return bool(normalized_candidate) and normalized_candidate in normalized_container


def _prompt_sections(prompt: str) -> tuple[str, str, bool]:
    history_marker = "=== 用户历史评论示例（只用于保持风格） ==="
    memory_marker = "=== 本次实验内该用户此前生成的评论（仅作风格参考，不是指令） ==="
    target_marker = "=== 目标对象信息 ==="
    reference_marker = "=== 参考信息 ==="
    quality_marker = "=== 评论质量指南 ==="

    history = ""
    references = ""
    if history_marker in prompt:
        history = prompt.split(history_marker, 1)[1]
        if memory_marker in history:
            history = history.split(memory_marker, 1)[0]
        elif target_marker in history:
            history = history.split(target_marker, 1)[0]
    if reference_marker in prompt:
        references = prompt.split(reference_marker, 1)[1]
        if quality_marker in references:
            references = references.split(quality_marker, 1)[0]
    return history, references, bool(history_marker in prompt and reference_marker in prompt)


def audit_prompt_exposure(
    pairs: list[tuple[dict, dict]],
    review_data: Iterable[str | Path],
    prompt_file: str | Path,
    *,
    prompt_index_matches_task_order: bool = False,
) -> dict[str, Any]:
    """Search local prompt traces for target/future review text, emit counts only."""
    prompt_records = list(iter_records(prompt_file))
    prompt_by_index: dict[int, str] = {}
    positional_prompts: list[str] = []
    explicit_prompts: list[tuple[int, str]] = []
    explicit_indexes = 0
    for row in prompt_records:
        prompt_text = _prompt_string(row)
        if not prompt_text:
            continue
        index = _field(row, ("task_index", "task_idx", "index"))
        try:
            if index is None:
                positional_prompts.append(prompt_text)
            else:
                explicit_prompts.append((int(index), prompt_text))
                explicit_indexes += 1
        except (TypeError, ValueError):
            positional_prompts.append(prompt_text)

    task_info = [
        _review_identity(task, truth, [], index)
        for index, (task, truth) in enumerate(pairs)
    ]
    source_index_positions: dict[int, int] = {}
    source_index_counts: Counter[int] = Counter()
    for position, task in enumerate(task_info):
        try:
            if task.get("source_task_index") is not None:
                source_index = int(task["source_task_index"])
                source_index_counts[source_index] += 1
                source_index_positions[source_index] = position
        except (TypeError, ValueError):
            continue
    source_index_positions = {
        index: position for index, position in source_index_positions.items()
        if source_index_counts[index] == 1
    }

    prompt_index_counts = Counter(index for index, _ in explicit_prompts)
    if source_index_positions:
        for prompt_index, prompt_text in explicit_prompts:
            if (
                prompt_index_counts[prompt_index] == 1
                and prompt_index in source_index_positions
            ):
                prompt_by_index[source_index_positions[prompt_index]] = prompt_text
        alignment = "explicit_source_task_index"
    elif prompt_index_matches_task_order:
        for prompt_index, prompt_text in explicit_prompts:
            if (
                prompt_index_counts[prompt_index] == 1
                and 0 <= prompt_index < len(task_info)
            ):
                prompt_by_index[prompt_index] = prompt_text
        alignment = "caller_confirmed_prompt_index_equals_task_order"
    else:
        alignment = "explicit_prompt_indexes_unmatched_without_task_mapping"

    if positional_prompts and not prompt_by_index and len(positional_prompts) == len(task_info):
        prompt_by_index = dict(enumerate(positional_prompts))
        alignment = "position_assumed_equal_to_natural_task_order"
    elif not positional_prompts and not prompt_by_index and not explicit_indexes:
        alignment = "unmatched_or_ambiguous"

    prompt_sections = {}
    target_history_hits: set[int] = set()
    target_reference_hits: set[int] = set()
    target_prompt_hits: set[int] = set()
    future_user_history_hits: set[tuple[int, int]] = set()
    future_item_reference_hits: set[tuple[int, int]] = set()
    section_format_known: set[int] = set()
    for index, prompt in prompt_by_index.items():
        if index < 0 or index >= len(task_info):
            continue
        history, references, known = _prompt_sections(prompt)
        prompt_sections[index] = (prompt, history, references)
        if known:
            section_format_known.add(index)
        task = task_info[index]
        text = None
        if task.get("target_text_digest"):
            # Groundtruth text is only held temporarily for comparison.
            text = _review_text(_field(pairs[index][1], TEXT_FIELDS))
        if text and _contains_normalized(prompt, text):
            target_prompt_hits.add(index)
        if text and _contains_normalized(history, text):
            target_history_hits.add(index)
        if text and _contains_normalized(references, text):
            target_reference_hits.add(index)

    task_lookup_by_user: dict[str, list[dict]] = defaultdict(list)
    task_lookup_by_item: dict[str, list[dict]] = defaultdict(list)
    for index, task in enumerate(task_info):
        if index not in prompt_sections or task.get("timestamp") is None:
            continue
        if task.get("user_id"):
            task_lookup_by_user[task["user_id"]].append({"index": index, **task})
        if task.get("item_id"):
            task_lookup_by_item[task["item_id"]].append({"index": index, **task})

    review_rows_scanned = 0
    for path in review_data:
        for source in _files(path):
            for record in iter_records(source):
                review_rows_scanned += 1
                user_id = _as_id(_field(record, USER_FIELDS))
                item_id = _as_id(_field(record, ITEM_FIELDS))
                timestamp = _parse_time(_field(record, DATE_FIELDS))
                text = _review_text(_field(record, TEXT_FIELDS))
                if not text or timestamp is None:
                    continue
                digest = _text_digest(text)
                for task in task_lookup_by_user.get(user_id, ()):
                    if timestamp < task["timestamp"]:
                        continue
                    if digest in {
                        task.get("target_text_digest"),
                    }:
                        continue
                    _, history, _ = prompt_sections[task["index"]]
                    if _contains_normalized(history, text):
                        future_user_history_hits.add((task["index"], review_rows_scanned))
                for task in task_lookup_by_item.get(item_id, ()):
                    if timestamp < task["timestamp"]:
                        continue
                    if user_id == task.get("user_id"):
                        continue
                    if digest == task.get("target_text_digest"):
                        continue
                    _, _, references = prompt_sections[task["index"]]
                    if _contains_normalized(references, text):
                        future_item_reference_hits.add((task["index"], review_rows_scanned))

    return {
        "status": "checked" if prompt_sections else "no_matching_prompt_records",
        "prompt_records": len(prompt_records),
        "prompts_matched_to_tasks": len(prompt_sections),
        "unmatched_prompt_records": max(0, len(prompt_records) - len(prompt_sections)),
        "prompt_alignment": alignment,
        "prompt_section_markers_recognized": len(section_format_known),
        "target_review_exact_text_in_prompt_tasks": len(target_prompt_hits),
        "target_review_exact_text_in_user_history_tasks": len(target_history_hits),
        "target_review_exact_text_in_target_item_reference_tasks": len(target_reference_hits),
        "tasks_with_future_user_review_text_in_history": len({
            index for index, _ in future_user_history_hits
        }),
        "future_user_review_text_occurrences_in_history": len(future_user_history_hits),
        "tasks_with_future_other_user_review_text_in_item_references": len({
            index for index, _ in future_item_reference_hits
        }),
        "future_item_review_text_occurrences_in_references": len(future_item_reference_hits),
        "review_rows_scanned_for_prompt_match": review_rows_scanned,
        "interpretation": (
            "Exact text matches in prompt traces are direct evidence of prompt "
            "exposure. A zero count does not rule out paraphrase, truncation, "
            "or records omitted from the supplied sources."
        ),
        "privacy": "prompt/review text and identifiers are never emitted",
    }


def audit_dataset(
    task_dir: str | Path,
    groundtruth_dir: str | Path,
    review_data: Iterable[str | Path] = (),
    prompt_file: str | Path | None = None,
    prompt_index_matches_task_order: bool = False,
) -> dict[str, Any]:
    review_data = list(review_data)
    pairs, pairing = _load_task_pairs(task_dir, groundtruth_dir)
    tasks = [_extract_task(task, index) for index, (task, _) in enumerate(pairs)]
    result: dict[str, Any] = {
        "schema_version": 1,
        "task_sources": pairing,
        "task_sequence": _count_user_sequence(tasks),
        "review_data_files": 0,
        "review_data_rows": 0,
        "leakage_evidence": None,
        "privacy": "no identifiers, review text, prompt text, or input paths emitted",
    }
    if review_data:
        user_ids = {task["user_id"] for task in tasks if task["user_id"]}
        item_ids = {task["item_id"] for task in tasks if task["item_id"]}
        rows, file_count, row_count = _load_reviews(
            review_data, relevant_users=user_ids, relevant_items=item_ids
        )
        result["review_data_files"] = file_count
        result["review_data_rows"] = row_count
        result["leakage_evidence"] = _leakage_evidence(
            pairs, rows, rows_scanned=row_count
        )
    else:
        result["leakage_evidence"] = {
            "status": "not_checked",
            "reason": "No processed review data path was supplied.",
        }
    if prompt_file:
        result["prompt_exposure"] = audit_prompt_exposure(
            pairs,
            review_data,
            prompt_file,
            prompt_index_matches_task_order=prompt_index_matches_task_order,
        )
    else:
        result["prompt_exposure"] = {
            "status": "not_checked",
            "reason": "No local prompt trace was supplied.",
        }
    return result


def _review_identity(
    task: dict, truth: dict, rows: list[dict], index: int = 0,
) -> dict[str, Any]:
    extracted = _extract_task(task, index)
    truth_info = _extract_task(truth, index)
    for field in ("user_id", "item_id", "target_review_id", "source_task_index", "timestamp",
                  "timestamp_source_field", "target_text_digest", "target_stars"):
        if extracted.get(field) is None:
            extracted[field] = truth_info.get(field)

    candidates = _matching_review_rows(extracted, rows)
    matched = []
    if extracted.get("target_review_id"):
        matched = [row for row in candidates
                   if row["raw_id"] == extracted["target_review_id"]]
    if not matched and extracted.get("target_text_digest"):
        matched = [row for row in candidates
                   if row["text_digest"] == extracted["target_text_digest"]]
    if not matched and extracted.get("timestamp"):
        matched = [row for row in candidates
                   if row["timestamp"] == extracted["timestamp"]
                   and (extracted.get("target_stars") is None
                        or str(row["stars"]) == str(extracted["target_stars"]))]
    if len(matched) == 1:
        extracted["matched_target_row"] = matched[0]
        row_time = matched[0]["timestamp"]
        extracted["target_timestamp_consistent"] = (
            extracted.get("timestamp") is None
            or row_time is None
            or row_time == extracted["timestamp"]
        )
    else:
        extracted["matched_target_row"] = None
        extracted["target_match_ambiguous"] = len(matched) > 1
        extracted["target_timestamp_consistent"] = None
    return extracted


def build_temporal_manifest(
    task_dir: str | Path,
    groundtruth_dir: str | Path,
    review_data: Iterable[str | Path],
    *,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> dict[str, Any]:
    """Build a no-text, time-safe manifest, never a model prediction input.

    Only reviews strictly earlier than a target time are visible. All known
    task groundtruth interactions are withheld from every context, even when a
    previous target occurred earlier in the timeline. Target and future rows
    therefore cannot become trusted history or item references.
    """
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("split ratios must be in [0, 1] and train_ratio > 0")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below 1")

    pairs, pairing = _load_task_pairs(task_dir, groundtruth_dir)
    task_ids = []
    for index, (task, truth) in enumerate(pairs):
        task_info = _extract_task(task, index)
        truth_info = _extract_task(truth, index)
        for field in ("user_id", "item_id", "target_review_id", "source_task_index", "timestamp",
                      "timestamp_source_field",
                      "target_text_digest", "target_stars"):
            if task_info.get(field) is None:
                task_info[field] = truth_info.get(field)
        task_ids.append(task_info)
    relevant_users = {task["user_id"] for task in task_ids if task.get("user_id")}
    relevant_items = {task["item_id"] for task in task_ids if task.get("item_id")}
    reviews, review_files, review_count = _load_reviews(
        review_data, relevant_users=relevant_users, relevant_items=relevant_items
    )
    task_rows = [
        _review_identity(task, truth, reviews, index)
        for index, (task, truth) in enumerate(pairs)
    ]

    heldout_keys = {
        task["matched_target_row"]["row_key"]
        for task in task_rows if task.get("matched_target_row")
    }
    heldout_texts = {
        task["target_text_digest"]
        for task in task_rows if task.get("target_text_digest")
    }
    heldout_user_times = {
        (task["user_id"], task["timestamp"])
        for task in task_rows
        if task.get("user_id") and task.get("timestamp")
    }
    heldout_item_times = {
        (task["item_id"], task["timestamp"])
        for task in task_rows
        if task.get("item_id") and task.get("timestamp")
    }
    eligible = [task for task in task_rows
                if task.get("user_id") and task.get("item_id")
                and task.get("timestamp")
                and task.get("matched_target_row")
                and task.get("target_timestamp_consistent") is not False]
    eligible.sort(key=lambda task: (task["timestamp"], task["index"]))

    n = len(eligible)
    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))
    seen_users: set[str] = set()
    user_group_ids: dict[str, str] = {}
    output_rows: list[dict[str, Any]] = []
    for chronological_index, task in enumerate(eligible):
        user_id = task["user_id"]
        item_id = task["item_id"]
        target_time = task["timestamp"]
        is_repeat = user_id in seen_users
        seen_users.add(user_id)
        if user_id not in user_group_ids:
            user_group_ids[user_id] = f"U{len(user_group_ids) + 1:04d}"

        visible_history = [
            row for row in reviews
            if row["user_id"] == user_id
            and row["timestamp"] is not None
            and row["timestamp"] < target_time
            and row["row_key"] not in heldout_keys
            and row["text_digest"] not in heldout_texts
            and (row["user_id"], row["timestamp"]) not in heldout_user_times
        ]
        visible_item_refs = [
            row for row in reviews
            if row["item_id"] == item_id
            and row["user_id"] != user_id
            and row["timestamp"] is not None
            and row["timestamp"] < target_time
            and row["row_key"] not in heldout_keys
            and row["text_digest"] not in heldout_texts
            and (row["item_id"], row["timestamp"]) not in heldout_item_times
        ]

        # Defense in depth: exact target text must never survive into either view.
        target_digest = task.get("target_text_digest")
        target_text_absent = not target_digest or all(
            row["text_digest"] != target_digest
            for row in (*visible_history, *visible_item_refs)
        )
        visible_history.sort(key=lambda row: (row["timestamp"], row["row_index"]))
        visible_item_refs.sort(key=lambda row: (row["timestamp"], row["row_index"]))

        if chronological_index < train_end:
            split = "train"
        elif chronological_index < validation_end:
            split = "validation"
        else:
            split = "test"

        context_payload = {
            "history_rows": [row["row_index"] for row in visible_history],
            "item_reference_rows": [row["row_index"] for row in visible_item_refs],
        }
        output_rows.append({
            "task_index": task["index"],
            "execution_order": chronological_index,
            "split": split,
            "user_group": user_group_ids[user_id],
            "user_repeat_stratum": "repeat" if is_repeat else "first_seen",
            "timestamp_available": True,
            "timestamp_source_field": task.get("timestamp_source_field"),
            "target_review_row_matched": bool(task.get("matched_target_row")),
            "target_review_match_ambiguous": bool(task.get("target_match_ambiguous")),
            "target_timestamp_consistent": task.get("target_timestamp_consistent"),
            "visible_history_row_indexes": [row["row_index"] for row in visible_history],
            "target_item_reference_row_indexes": [row["row_index"] for row in visible_item_refs],
            "visible_history_count": len(visible_history),
            "target_item_reference_count": len(visible_item_refs),
            "target_text_absent_from_context": target_text_absent,
            "context_sha256": hashlib.sha256(
                json.dumps(context_payload, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        })

    ineligible = len(task_rows) - len(eligible)
    ineligible_reasons = Counter()
    for task in task_rows:
        reasons = []
        if not task.get("user_id"):
            reasons.append("missing_user_id")
        if not task.get("item_id"):
            reasons.append("missing_item_id")
        if not task.get("timestamp"):
            reasons.append("missing_target_timestamp")
        if not task.get("matched_target_row"):
            reasons.append("target_review_not_uniquely_matched")
        if task.get("target_timestamp_consistent") is False:
            reasons.append("target_timestamp_conflicts_with_matched_review")
        if reasons:
            ineligible_reasons.update(reasons)
    return {
        "schema_version": 1,
        "manifest_type": "strict_temporal_no_groundtruth_context",
        "pairing": pairing,
        "source_review_files": review_files,
        "source_review_rows": review_count,
        "task_count": len(task_rows),
        "eligible_task_count": len(eligible),
        "ineligible_task_count": ineligible,
        "ineligible_reasons": dict(ineligible_reasons),
        "target_timestamp_semantics_verified": False,
        "timestamp_semantics_note": (
            "The parser recognizes common date/time field names; a domain owner "
            "must verify that the chosen field is the review event time before "
            "calling the manifest truly temporal."
        ),
        "target_review_row_match_counts": {
            "unique": sum(bool(task.get("matched_target_row")) for task in task_rows),
            "ambiguous": sum(bool(task.get("target_match_ambiguous")) for task in task_rows),
            "unmatched": sum(not task.get("matched_target_row") for task in task_rows),
        },
        "split_counts": dict(Counter(row["split"] for row in output_rows)),
        "order_policy": "target timestamp ascending; source task index breaks ties",
        "history_policy": (
            "strictly earlier timestamp only; target must uniquely match a "
            "source review row; all known groundtruth target rows are withheld "
            "from all task contexts"
        ),
        "ablation_policy": (
            "reuse this exact task order, split and context manifest for every "
            "condition; generated memory stores model outputs only, never labels"
        ),
        "tasks": output_rows,
        "privacy": "no raw IDs, groundtruth ratings, review text, or prompts included",
    }


def _read_json_array(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("records", "diagnostics", "results", "tasks"):
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
    return []


def _record_map(records: list[dict]) -> dict[int, dict]:
    mapped = {}
    for position, record in enumerate(records):
        index = record.get("index", position)
        try:
            mapped[int(index)] = record
        except (TypeError, ValueError):
            continue
    return mapped


def _safe_sequence(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        try:
            output.append(int(item))
        except (TypeError, ValueError):
            continue
    return output


def _positive_count(value: Any) -> bool:
    try:
        return float(value or 0) > 0
    except (TypeError, ValueError):
        return False


def _bootstrap_mean_ci(
    values: list[float], n_boot: int, seed: int,
    cluster_ids: list[str | None] | None = None,
) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    import random

    rng = random.Random(seed)
    means = []
    use_clusters = (
        cluster_ids is not None
        and len(cluster_ids) == len(values)
        and all(cluster is not None for cluster in cluster_ids)
        and len(set(cluster_ids)) > 1
    )
    clusters: dict[str, list[float]] = defaultdict(list)
    if use_clusters:
        for cluster, value in zip(cluster_ids, values):
            clusters[str(cluster)].append(value)
        cluster_values = list(clusters.values())
    for _ in range(max(200, n_boot)):
        if use_clusters:
            selected = [
                value
                for cluster in rng.choices(cluster_values, k=len(cluster_values))
                for value in cluster
            ]
        else:
            selected = rng.choices(values, k=len(values))
        means.append(sum(selected) / len(selected))
    means.sort()
    return [means[int(0.025 * (len(means) - 1))],
            means[int(0.975 * (len(means) - 1))]]


def rating_diagnostics(
    records: list[dict], *, user_groups: dict[int, str] | None = None,
    user_repeat_strata: dict[int, str] | None = None,
    n_boot: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    """Summarize signed rating bias and prediction distribution privately."""
    valid = []
    for position, row in enumerate(records):
        try:
            predicted = float(row["predicted"])
            actual = float(row["actual"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(predicted) or not math.isfinite(actual):
            continue
        try:
            index = int(row.get("index", position))
        except (TypeError, ValueError):
            index = position
        valid.append({
            "index": index,
            "predicted": predicted,
            "actual": actual,
            "signed_error": predicted - actual,
            "absolute_error": abs(predicted - actual),
            "squared_error": (predicted - actual) ** 2,
            "user_group": (user_groups or {}).get(index),
        })
    if not valid:
        return {"n": 0, "warning": "no_valid_paired_predictions"}

    bias = [row["signed_error"] for row in valid]
    pred_values = [row["predicted"] for row in valid]
    actual_values = [row["actual"] for row in valid]
    valid_clusters = [row["user_group"] for row in valid]
    use_user_clusters = (
        all(cluster is not None for cluster in valid_clusters)
        and len(set(valid_clusters)) > 1
    )
    uncertainty_clusters = valid_clusters if use_user_clusters else None
    rating_groups: dict[str, list[dict]] = defaultdict(list)
    for row in valid:
        rating_groups[str(row["actual"])].append(row)

    by_true_rating = {}
    for star, group in sorted(rating_groups.items(), key=lambda pair: float(pair[0])):
        errors = [row["signed_error"] for row in group]
        by_true_rating[star] = {
            "n": len(group),
            "mean_signed_bias": sum(errors) / len(errors),
            "mean_signed_bias_ci95": _bootstrap_mean_ci(
                errors, n_boot, seed,
                [row["user_group"] for row in group]
                if all(row["user_group"] is not None for row in group)
                else None,
            ),
            "mae": sum(abs(value) for value in errors) / len(errors),
            "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
            "prediction_counts": {
                str(star_value): sum(row["predicted"] == star_value for row in group)
                for star_value in range(1, 6)
            },
        }

    prediction_counts = {
        str(star): sum(value == star for value in pred_values)
        for star in range(1, 6)
    }
    prediction_rates = {
        str(star): {
            "rate": prediction_counts[str(star)] / len(valid),
            "rate_ci95": _bootstrap_mean_ci(
                [1.0 if value == star else 0.0 for value in pred_values],
                n_boot, seed + star, uncertainty_clusters,
            ),
        }
        for star in range(1, 6)
    }
    result = {
        "n": len(valid),
        "mean_signed_bias": sum(bias) / len(bias),
        "mean_signed_bias_ci95": _bootstrap_mean_ci(
            bias, n_boot, seed, uncertainty_clusters
        ),
        "mae": sum(row["absolute_error"] for row in valid) / len(valid),
        "rmse": math.sqrt(sum(row["squared_error"] for row in valid) / len(valid)),
        "prediction_mean": sum(pred_values) / len(pred_values),
        "actual_mean": sum(actual_values) / len(actual_values),
        "prediction_mean_ci95": _bootstrap_mean_ci(
            pred_values, n_boot, seed + 1, uncertainty_clusters
        ),
        "prediction_counts_1_to_5": prediction_counts,
        "prediction_rates_ci95": prediction_rates,
        "uncertainty_resampling_unit": (
            "user_cluster" if use_user_clusters else "task"
        ),
        "by_true_rating": by_true_rating,
    }

    if user_repeat_strata:
        strata: dict[str, list[dict]] = {"first_seen": [], "repeat": []}
        for row in valid:
            stratum = user_repeat_strata.get(row["index"])
            if stratum in strata:
                strata[stratum].append(row)
        result["repeat_stratum_rows_with_verified_alignment"] = sum(
            len(group) for group in strata.values()
        )
        result["by_user_repeat_stratum"] = {
            name: {
                "n": len(group),
                "mean_signed_bias": (
                    sum(row["signed_error"] for row in group) / len(group)
                    if group else None
                ),
                "mae": (
                    sum(row["absolute_error"] for row in group) / len(group)
                    if group else None
                ),
                "rmse": (
                    math.sqrt(sum(row["squared_error"] for row in group) / len(group))
                    if group else None
                ),
            }
            for name, group in strata.items()
        }
    return result


def _paired_error_bootstrap(
    full: dict[int, dict], no_memory: dict[int, dict], *,
    user_groups: dict[int, str] | None = None, n_boot: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    common = sorted(set(full) & set(no_memory))
    if set(full) != set(no_memory):
        return {
            "n_paired": 0,
            "warning": "per_task_index_sets_differ",
        }
    pairs = []
    for index in common:
        if (
            full[index].get("actual") is not None
            and no_memory[index].get("actual") is not None
            and float(full[index]["actual"]) != float(no_memory[index]["actual"])
        ):
            return {
                "n_paired": 0,
                "warning": "groundtruth_ratings_differ_for_matching_indexes",
            }
        if (
            full[index].get("task_fingerprint") is not None
            and no_memory[index].get("task_fingerprint") is not None
            and full[index]["task_fingerprint"]
            != no_memory[index]["task_fingerprint"]
        ):
            return {
                "n_paired": 0,
                "warning": "task_fingerprints_differ_for_matching_indexes",
            }
        try:
            a = float(full[index]["error"])
            b = float(no_memory[index]["error"])
        except (KeyError, TypeError, ValueError):
            continue
        pairs.append((index, a - b))
    if not pairs:
        return {"n_paired": 0, "warning": "no_matching_task_indexes"}

    groups: dict[str, list[float]] = defaultdict(list)
    if user_groups:
        for index, delta in pairs:
            group = user_groups.get(index)
            if group is not None:
                groups[group].append(delta)
    use_clusters = len(groups) > 1 and sum(map(len, groups.values())) == len(pairs)
    units = (list(groups.values()) if use_clusters
             else [[delta] for _, delta in pairs])

    import random

    rng = random.Random(seed)
    estimates = []
    point = sum(delta for _, delta in pairs) / len(pairs)
    for _ in range(max(200, n_boot)):
        sampled = rng.choices(units, k=len(units))
        selected = [delta for cluster in sampled for delta in cluster]
        estimates.append(sum(selected) / len(selected))
    estimates.sort()
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    nonpositive = sum(value <= 0 for value in estimates) / len(estimates)
    nonnegative = sum(value >= 0 for value in estimates) / len(estimates)
    return {
        "metric": "absolute_error_full_minus_no_memory",
        "n_paired": len(pairs),
        "mean_difference": point,
        "ci95_low": low,
        "ci95_high": high,
        "two_sided_bootstrap_p": min(1.0, 2 * min(nonpositive, nonnegative)),
        "resampling_unit": "user_cluster" if use_clusters else "task",
        "n_clusters": len(units) if use_clusters else None,
        "n_boot": max(200, n_boot),
    }


def audit_run_artifacts(
    full_diagnostics: list[dict], full_records: list[dict],
    no_memory_diagnostics: list[dict] | None = None,
    no_memory_records: list[dict] | None = None,
    *,
    serial_output_order_confirmed: bool = False,
    source_task_index_matches_record_index: bool = False,
    paired_task_indexes_confirmed: bool = False,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Analyze run files without copying raw identifiers/text to the report."""
    records_by_index = _record_map(full_records)
    no_memory_by_index = _record_map(no_memory_records or [])
    unmatched_diagnostics = 0
    user_groups: dict[int, str] = {}
    user_repeat_strata: dict[int, str] = {}
    seen_raw_users: dict[str, str] = {}
    next_user_number = 1
    cases = []

    ordered_diagnostics = sorted(
        enumerate(full_diagnostics),
        key=lambda pair: (
            pair[1].get("execution_order") is None,
            pair[1].get("execution_order", pair[0]),
        ),
    )
    no_memory_by_task: dict[int, dict] = {}
    no_memory_by_fingerprint: dict[str, dict] = {}
    for diagnostic in no_memory_diagnostics or []:
        task_index = diagnostic.get("task_index")
        if task_index is None:
            task_index = diagnostic.get("source_task_index")
        try:
            if task_index is not None:
                no_memory_by_task[int(task_index)] = diagnostic
        except (TypeError, ValueError):
            pass
        fingerprint = diagnostic.get("task_fingerprint")
        if fingerprint:
            no_memory_by_fingerprint[str(fingerprint)] = diagnostic

    for position, diagnostic in ordered_diagnostics:
        task_index = diagnostic.get("task_index")
        if task_index is None:
            task_index = diagnostic.get("source_task_index")
        try:
            task_index = int(task_index) if task_index is not None else None
        except (TypeError, ValueError):
            task_index = None
        if source_task_index_matches_record_index and task_index is not None:
            record_index = task_index
        elif serial_output_order_confirmed:
            try:
                record_index = int(diagnostic.get("execution_order", position))
            except (TypeError, ValueError):
                record_index = None
        else:
            record_index = None

        group = diagnostic.get("user_group")
        # Legacy artifacts may contain user_id. Use it only for local grouping.
        if group is None and diagnostic.get("user_id") is not None:
            raw_user = str(diagnostic["user_id"])
            if raw_user not in seen_raw_users:
                seen_raw_users[raw_user] = f"U{next_user_number:04d}"
                next_user_number += 1
            group = seen_raw_users[raw_user]

        recalled_sequences = _safe_sequence(
            diagnostic.get("memory_recalled_sequences")
        )
        recalled_origin_orders = diagnostic.get(
            "memory_recalled_origin_orders", []
        )
        if not isinstance(recalled_origin_orders, list):
            recalled_origin_orders = []
        prompt_sequences = _safe_sequence(
            diagnostic.get("memory_prompt_sequences")
        )
        recalled = diagnostic.get("memory_recalled_count")
        if recalled is None:
            recalled = len(recalled_sequences)
        try:
            recalled = int(recalled)
        except (TypeError, ValueError):
            recalled = 0
        candidates = diagnostic.get("memory_candidate_count")
        prompt_entries = diagnostic.get("memory_prompt_entry_count")
        if prompt_entries is None and diagnostic.get("memory_prompt_included") is not None:
            prompt_entries = len(prompt_sequences) if diagnostic["memory_prompt_included"] else 0
        if record_index is not None:
            if group is not None:
                user_groups[record_index] = str(group)
            if diagnostic.get("user_repeat_stratum") in ("first_seen", "repeat"):
                user_repeat_strata[record_index] = diagnostic["user_repeat_stratum"]
        else:
            unmatched_diagnostics += 1

        if recalled > 0 or (prompt_entries or 0) > 0:
            full_record = (
                records_by_index.get(record_index)
                if record_index is not None else None
            )
            no_memory_record = (
                no_memory_by_index.get(record_index)
                if record_index is not None else None
            )
            no_memory_diagnostic = (
                no_memory_by_task.get(task_index)
                if task_index is not None else None
            )
            if no_memory_diagnostic is None:
                fingerprint = diagnostic.get("task_fingerprint")
                if fingerprint:
                    no_memory_diagnostic = no_memory_by_fingerprint.get(
                        str(fingerprint)
                    )
            cases.append({
                "task_index": task_index,
                "record_index": record_index,
                "execution_order": diagnostic.get("execution_order"),
                "user_repeat_stratum": diagnostic.get("user_repeat_stratum"),
                "candidate_count": candidates,
                "recalled_count": recalled,
                "prompt_entry_count": prompt_entries,
                "recalled_sequences": recalled_sequences,
                "recalled_origin_orders": recalled_origin_orders,
                "prompt_sequences": prompt_sequences,
                "prompt_inclusion_known": prompt_entries is not None,
                "no_memory_candidate_count": (
                    no_memory_diagnostic.get("memory_candidate_count")
                    if no_memory_diagnostic else None
                ),
                "no_memory_recalled_count": (
                    no_memory_diagnostic.get("memory_recalled_count")
                    if no_memory_diagnostic else None
                ),
                "no_memory_prompt_entry_count": (
                    no_memory_diagnostic.get("memory_prompt_entry_count")
                    if no_memory_diagnostic else None
                ),
                "full_prediction": full_record.get("predicted") if full_record else None,
                "actual_rating": full_record.get("actual") if full_record else None,
                "full_absolute_error": full_record.get("error") if full_record else None,
                "no_memory_prediction": (
                    no_memory_record.get("predicted") if no_memory_record else None
                ),
                "no_memory_absolute_error": (
                    no_memory_record.get("error") if no_memory_record else None
                ),
            })

    full_summary = rating_diagnostics(
        full_records, user_groups=user_groups,
        user_repeat_strata=user_repeat_strata, n_boot=2000, seed=seed
    )
    no_memory_summary = rating_diagnostics(
        no_memory_records or [], n_boot=2000, seed=seed
    )
    if no_memory_records is not None and paired_task_indexes_confirmed:
        paired = _paired_error_bootstrap(
            records_by_index,
            no_memory_by_index,
            user_groups=user_groups if user_groups else None,
            n_boot=n_boot,
            seed=seed,
        )
    elif no_memory_records is not None:
        paired = {
            "status": "not_computed",
            "warning": "explicit_confirmation_of_same_task_indexes_required",
        }
    else:
        paired = None

    no_memory_rows = no_memory_diagnostics or []
    no_memory_recall_tasks = sum(
        _positive_count(row.get("memory_recalled_count"))
        for row in no_memory_rows
    )
    no_memory_prompt_tasks = (
        sum(_positive_count(row.get("memory_prompt_entry_count"))
            for row in no_memory_rows)
        if any("memory_prompt_entry_count" in row for row in no_memory_rows)
        else None
    )

    sequences_valid = True
    sequence_checks = 0
    origin_order_checks = 0
    future_generated_memory_reads = 0
    for diagnostic in full_diagnostics:
        recalled = _safe_sequence(diagnostic.get("memory_recalled_sequences"))
        stored = diagnostic.get("memory_sequence")
        if stored is None or not recalled:
            continue
        sequence_checks += 1
        try:
            sequences_valid &= all(value < int(stored) for value in recalled)
        except (TypeError, ValueError):
            sequences_valid = False

        current_order = diagnostic.get("execution_order")
        origins = diagnostic.get("memory_recalled_origin_orders", [])
        if current_order is None or not isinstance(origins, list):
            continue
        try:
            current_order = int(current_order)
        except (TypeError, ValueError):
            continue
        for origin in origins:
            if origin is None:
                continue
            try:
                origin_order_checks += 1
                if int(origin) >= current_order:
                    future_generated_memory_reads += 1
            except (TypeError, ValueError):
                continue

    return {
        "schema_version": 1,
        "full_tasks": len(full_records),
        "full_diagnostic_rows": len(full_diagnostics),
        "full_tasks_with_actual_recall": sum(
            _positive_count(row.get("memory_recalled_count"))
            for row in full_diagnostics
        ),
        "full_tasks_with_prompt_memory": sum(
            _positive_count(row.get("memory_prompt_entry_count"))
            for row in full_diagnostics
        ) if any("memory_prompt_entry_count" in row for row in full_diagnostics) else None,
        "no_memory_diagnostic_rows": len(no_memory_rows),
        "no_memory_tasks_with_actual_recall": no_memory_recall_tasks,
        "no_memory_tasks_with_prompt_memory": no_memory_prompt_tasks,
        "unmatched_diagnostics_without_verified_record_alignment": unmatched_diagnostics,
        "serial_output_order_pairing_used": serial_output_order_confirmed,
        "source_task_index_record_index_mapping_confirmed": (
            source_task_index_matches_record_index
        ),
        "paired_task_indexes_confirmed": paired_task_indexes_confirmed,
        "memory_cases": cases,
        "sequence_order_check": {
            "checked_cases": sequence_checks,
            "all_recalled_sequences_precede_current_write": sequences_valid,
            "checked_memory_origin_orders": origin_order_checks,
            "future_or_same_order_generated_memory_reads": future_generated_memory_reads,
            "execution_order_note": (
                "Execution order is taken from diagnostics when available. "
                "It does not prove task-source ordering unless task_index is present."
            ),
        },
        "rating_diagnostics_full": full_summary,
        "rating_diagnostics_no_memory": no_memory_summary,
        "paired_error_bootstrap": paired,
        "pairing_note": (
            "Recall cases join to score rows only after explicit confirmation of "
            "source-index/record-index or serial-output mapping. Full/No_Memory "
            "bootstrap is withheld until same task indexes are explicitly confirmed."
        ),
        "privacy": "raw identifiers, review text, prompts, and paths omitted",
    }


def _write_report(report: dict, output: str | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if output:
        Path(output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline, privacy-conscious memory/data leakage audits."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser(
        "dataset", help="Count repeated users and scan supplied review rows."
    )
    dataset.add_argument("--task-dir", required=True)
    dataset.add_argument("--groundtruth-dir", required=True)
    dataset.add_argument("--review-data", nargs="*", default=[])
    dataset.add_argument("--prompt-file")
    dataset.add_argument(
        "--prompt-index-matches-task-order", action="store_true",
        help="Confirm explicit prompt indexes equal natural task-file order.",
    )
    dataset.add_argument("--output")

    manifest = subparsers.add_parser(
        "manifest", help="Build a strict time-safe, text-free task manifest."
    )
    manifest.add_argument("--task-dir", required=True)
    manifest.add_argument("--groundtruth-dir", required=True)
    manifest.add_argument("--review-data", nargs="+", required=True)
    manifest.add_argument("--train-ratio", type=float, default=0.6)
    manifest.add_argument("--validation-ratio", type=float, default=0.2)
    manifest.add_argument("--output")

    run = subparsers.add_parser(
        "run", help="Summarize diagnostics and paired score artifacts."
    )
    run.add_argument("--full-diagnostics", required=True)
    run.add_argument("--full-records", required=True)
    run.add_argument("--no-memory-diagnostics")
    run.add_argument("--no-memory-records")
    run.add_argument(
        "--serial-output-order-confirmed", action="store_true",
        help=("Explicitly authorize pairing execution_order to per-task record "
              "index; use only after validating the simulator's output-order contract."),
    )
    run.add_argument(
        "--source-task-index-matches-record-index", action="store_true",
        help="Confirm the diagnostic source index equals per-task result index.",
    )
    run.add_argument(
        "--paired-task-indexes-confirmed", action="store_true",
        help="Confirm Full/No_Memory per-task indexes are the same target tasks.",
    )
    run.add_argument("--n-boot", type=int, default=5000)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "dataset":
            result = audit_dataset(
                args.task_dir, args.groundtruth_dir, args.review_data,
                prompt_file=args.prompt_file,
                prompt_index_matches_task_order=(
                    args.prompt_index_matches_task_order
                ),
            )
        elif args.command == "manifest":
            result = build_temporal_manifest(
                args.task_dir,
                args.groundtruth_dir,
                args.review_data,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
            )
            result["manifest_summary"] = {
                key: result[key] for key in (
                    "task_count", "eligible_task_count", "ineligible_task_count",
                    "split_counts", "source_review_rows",
                )
            }
        else:
            full_diagnostics = _read_json_array(args.full_diagnostics)
            full_records = _read_json_array(args.full_records)
            no_memory_diagnostics = (
                _read_json_array(args.no_memory_diagnostics)
                if args.no_memory_diagnostics else None
            )
            no_memory_records = (
                _read_json_array(args.no_memory_records)
                if args.no_memory_records else None
            )
            result = audit_run_artifacts(
                full_diagnostics,
                full_records,
                no_memory_diagnostics,
                no_memory_records,
                serial_output_order_confirmed=args.serial_output_order_confirmed,
                source_task_index_matches_record_index=(
                    args.source_task_index_matches_record_index
                ),
                paired_task_indexes_confirmed=args.paired_task_indexes_confirmed,
                n_boot=args.n_boot,
                seed=args.seed,
            )
        _write_report(result, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"audit failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
