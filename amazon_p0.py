#!/usr/bin/env python3
"""Offline P0 pipeline for the public Amazon 2014 Musical Instruments 5-core.

This module builds a field-traceable, time-safe, label-isolated offline task
set, scores four deterministic rating baselines, and writes a redacted
aggregate report. It never calls an LLM or the Simulator framework.

The only network operation is the explicit ``download`` subcommand; every
other command is fully offline and refuses to overwrite existing output.

Protocol summary
----------------

- Source rows map either the legacy Amazon 5-core fields or the newer review
  mirror fields: ``reviewerID/user_id -> user_id`` (a local stable
  pseudonym), ``asin -> item_id`` (local pseudonym), ``overall/rating ->
  stars``, ``reviewText/text -> text`` and ``unixReviewTime/timestamp -> UTC
  calendar day``.
  ``reviewerName``, ``summary``, ``helpful``, vote/image/style fields and all
  unknown dynamic metadata are dropped; the full normalized review text stays
  in the gitignored local work directory only.
- Valid rows are split at a day boundary close to the configured history
  ratio: the earlier days form the **frozen observed history prefix**, and the
  later days form the target candidate region. Same-day groups are never
  split.
- Targets are selected from repeated users in the candidate region (never one
  task per user), time-ordered, bounded by ``--max-tasks``.
- The runtime review source contains exactly the frozen history rows plus the
  selected target rows. The existing strict temporal manifest/provider
  therefore withholds every selected target row (from every split) from every
  context; observed history is frozen and never rolls forward.
- The canonical time-filtered manifest only partitions the selected target
  cohort (development/validation/test); that is not a partition of the raw
  file. Protocol metadata is stored separately in ``p0_policy.json``.

Safety gates (shared with ``memory_audit``): at most 64 MiB on disk including
decompressed content, at most 100,000 parsed rows, strict JSON only (no
``eval``), and a bounded per-record size for early rejection of pathological
lines.

License boundary: public availability of the dataset does not imply an open
license. The official page was not found to state one explicitly; this
repository records ``license_status = "not_explicitly_verified"`` and treats
the data as research use pending confirmation. No legal terms are accepted
automatically.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import urllib.request
import zlib
from typing import Any, Iterator

from memory_audit import (
    TEMPORAL_MAX_REVIEW_ROWS,
    TEMPORAL_MAX_REVIEW_SOURCE_BYTES,
    TemporalManifestInputError,
    _text_digest,
    build_temporal_manifest,
)
from rating_baselines import (
    BASELINE_MODES,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_LAMBDA_ITEM,
    DEFAULT_LAMBDA_USER,
    DEFAULT_WEIGHT_USER,
    BaselineInputError,
    RatingContext,
    predict_all,
    score_baselines,
)
from temporal_context import TemporalContextProvider, TemporalManifestError

ADAPTER_VERSION = "amazon-mi-p0-v1"
SOURCE_PAGE_URL = "https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html"
SOURCE_FILE_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    "reviews_Musical_Instruments_5.json.gz"
)
DATASET_CITATION = (
    "J. McAuley, C. Targett, J. Shi, A. van den Hengel. "
    "Image-Based Recommendations on Styles and Substitutes. SIGIR 2015. "
    "Amazon 2014 5-core review dataset (Musical Instruments)."
)
LICENSE_STATUS = "not_explicitly_verified"
LICENSE_NOTE = (
    "The official dataset page was not found to state an explicit data "
    "license, and public download availability does not imply an open "
    "license. The Kaggle mirrors' CC0/utility-script MIT terms do not cover "
    "the original reviews. Research use pending confirmation; no legal terms "
    "are accepted automatically."
)
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
MAX_RAW_RECORD_BYTES = 1024 * 1024
MAX_POLICY_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
PER_USER_TARGET_CAP = 4
HISTORY_RATIO_DEFAULT = 0.6
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
MAX_TASKS_DEFAULT = 400
SEED_DEFAULT = 42
ADAPTER_NAMESPACE = "amazon-mi-p0\0"

RUNTIME_SOURCE_COMPOSITION = (
    "frozen history prefix + selected target source rows"
)
TARGET_VISIBILITY = (
    "every selected target row is withheld from every context in every "
    "split, including earlier targets"
)
OBSERVED_HISTORY_NOTE = (
    "Observed history is the frozen prefix and never rolls forward with "
    "after-cut non-target rows. The manifest train/validation/test partition "
    "applies to the selected target cohort only; it is not a partition of "
    "the raw file and all three splits see the same frozen history."
)
GLOBAL_PRIOR_SOURCE = (
    "eligible frozen history prefix rows only: target review ids and "
    "target-text copies are excluded, and no target labels are used"
)
SPLIT_SCOPE_NOTE = (
    "The development/validation/test split partitions the selected target "
    "cohort only (a conditional sample of repeated-user targets after the "
    "frozen cut day). It is not a partition of the raw source file, and "
    "every split sees the same frozen observed history."
)

CANONICAL_FIELDS: tuple[str, ...] = (
    "review_id", "user_id", "item_id", "stars", "timestamp", "text",
    "source_row",
)
FIELD_MAPPING = {
    "reviewerID/user_id": "user_id (local pseudonym)",
    "asin": "item_id (local pseudonym)",
    "overall/rating": "stars",
    "reviewText/text": "text",
    "unixReviewTime/timestamp": "timestamp (UTC calendar day, YYYY-MM-DD)",
}
DROPPED_FIELDS = (
    "reviewerName", "summary", "helpful", "image", "images", "vote",
    "votes", "style", "verifiedPurchase", "reviewTime",
)
ARTIFACT_NAMES = (
    "manifest.json",
    "p0_policy.json",
    "tasks",
    "groundtruth",
    "runtime_reviews.jsonl",
    "normalized_reviews.jsonl",
)


class AmazonP0Error(ValueError):
    """Safe, categorized failure; the message is a stable category code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pseudonym(prefix: str, raw_value: str) -> str:
    payload = (ADAPTER_NAMESPACE + raw_value).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()[:16]


def _review_id(
    user_id: str, item_id: str, day: str, stars: float, text_digest: str | None,
    source_row: int,
) -> str:
    payload = "\0".join((
        user_id, item_id, day, format(stars, ".10g"), text_digest or "",
        str(source_row),
    ))
    return "r_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Explicit download (the only network operation; never enabled by default).
# ---------------------------------------------------------------------------


def _validate_downloaded_gzip(path: Path) -> int:
    """Fully decompress the staged archive; invalid/truncated input fails."""
    decoded = 0
    try:
        with gzip.open(path, "rb") as handle:
            while chunk := handle.read(64 * 1024):
                decoded += len(chunk)
                if decoded > TEMPORAL_MAX_REVIEW_SOURCE_BYTES:
                    raise AmazonP0Error(
                        "download_decompression_limit_exceeded"
                    )
    except AmazonP0Error:
        raise
    except (OSError, EOFError, zlib.error) as exc:
        raise AmazonP0Error("download_invalid_gzip") from exc
    return decoded


def download_source(
    url: str,
    output: str | Path,
    *,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    opener=None,
) -> dict[str, Any]:
    """Download only the pinned official archive explicitly requested.

    The URL must equal the one pinned official HTTPS file, ``max_bytes`` is a
    positive integer bounded by the adapter limit, the archive is streamed to
    an exclusively created ``.part`` sibling, fully decompressed to verify
    gzip integrity and the decompression cap, and only then published with a
    fresh-name check that never overwrites an existing file. All failures
    surface as safe category codes without paths.
    """
    if url != SOURCE_FILE_URL:
        raise AmazonP0Error("download_source_not_allowed")
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or not 1 <= max_bytes <= MAX_DOWNLOAD_BYTES
    ):
        raise AmazonP0Error("invalid_download_limit")
    destination = Path(output)
    if destination.exists():
        raise AmazonP0Error("output_file_exists")
    part = destination.with_name(destination.name + ".part")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AmazonP0Error("output_directory_unavailable") from exc
    try:
        part_handle = part.open("xb")
    except FileExistsError as exc:
        raise AmazonP0Error("output_part_file_exists") from exc
    except OSError as exc:
        raise AmazonP0Error("output_directory_unavailable") from exc

    request = urllib.request.Request(
        url, headers={"User-Agent": "amazon-p0-adapter/1.0"}
    )
    open_url = opener if opener is not None else urllib.request.urlopen
    total = 0
    digest = hashlib.sha256()
    try:
        with part_handle:
            with open_url(request, timeout=60) as response:
                final = response.geturl()
                if final != SOURCE_FILE_URL:
                    raise AmazonP0Error("download_redirect_not_allowed")
                while chunk := response.read(64 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise AmazonP0Error("download_size_limit_exceeded")
                    digest.update(chunk)
                    part_handle.write(chunk)
        if total == 0:
            raise AmazonP0Error("download_empty_response")
        decoded = _validate_downloaded_gzip(part)
        if decoded == 0:
            raise AmazonP0Error("download_empty_response")
        if destination.exists():
            raise AmazonP0Error("output_file_exists")
        os.replace(part, destination)
    except AmazonP0Error:
        part.unlink(missing_ok=True)
        raise
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise AmazonP0Error("download_network_failed") from exc
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise AmazonP0Error("download_network_failed") from exc
    return {
        "bytes": total,
        "sha256": digest.hexdigest(),
        "decompressed_bytes": decoded,
    }


# ---------------------------------------------------------------------------
# Strict, bounded source parsing.
# ---------------------------------------------------------------------------


def _open_binary(path: Path):
    if path.name.casefold().endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def _preflight_source(path: Path) -> tuple[int, int]:
    """Enforce the shared 64 MiB on-disk/decompressed gates before parsing."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AmazonP0Error("source_unavailable") from exc
    if size > TEMPORAL_MAX_REVIEW_SOURCE_BYTES:
        raise AmazonP0Error("source_byte_limit_exceeded")
    if not path.name.casefold().endswith(".gz"):
        return size, size
    decoded = 0
    try:
        with gzip.open(path, "rb") as handle:
            while chunk := handle.read(64 * 1024):
                decoded += len(chunk)
                if decoded > TEMPORAL_MAX_REVIEW_SOURCE_BYTES:
                    raise AmazonP0Error("source_decoded_byte_limit_exceeded")
    except AmazonP0Error:
        raise
    except (OSError, EOFError) as exc:
        raise AmazonP0Error("source_parse_failed") from exc
    return size, decoded


def _decode_json_object(payload: bytes) -> Any:
    if payload.startswith(b"\xef\xbb\xbf"):
        payload = payload[3:]
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AmazonP0Error("source_parse_failed") from exc


def _iter_jsonl_rows(handle) -> Iterator[Any]:
    """Yield one decoded JSON value per line with an early line-size gate."""
    buffer = b""
    while True:
        chunk = handle.read(64 * 1024)
        if not chunk:
            break
        buffer += chunk
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                if len(buffer) > MAX_RAW_RECORD_BYTES:
                    raise AmazonP0Error("source_line_limit_exceeded")
                break
            line, buffer = buffer[:newline], buffer[newline + 1:]
            if len(line) > MAX_RAW_RECORD_BYTES:
                raise AmazonP0Error("source_line_limit_exceeded")
            if line.strip():
                yield _decode_json_object(line.strip())
    tail = buffer.strip()
    if tail:
        if len(tail) > MAX_RAW_RECORD_BYTES:
            raise AmazonP0Error("source_line_limit_exceeded")
        yield _decode_json_object(tail)


class _BoundedJSONArrayReader:
    """Incremental JSON array decoder with a byte-accurate record cap.

    The cap applies to the unconsumed part of one in-flight record, never to
    a chunk that merely contains many small complete records.
    """

    def __init__(self, handle):
        self.handle = handle
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _compact(self) -> None:
        if self.position:
            self.buffer = self.buffer[self.position:]
            self.position = 0

    def _fill(self) -> None:
        self._compact()
        try:
            chunk = self.handle.read(64 * 1024)
        except UnicodeDecodeError as exc:
            raise AmazonP0Error("source_parse_failed") from exc
        if not chunk:
            self.eof = True
            return
        self.buffer += chunk

    def _skip_space(self) -> None:
        while True:
            while (
                self.position < len(self.buffer)
                and self.buffer[self.position].isspace()
            ):
                self.position += 1
            if self.position < len(self.buffer) or self.eof:
                return
            self._fill()

    def _unconsumed_bytes(self) -> int:
        return len(self.buffer[self.position:].encode("utf-8"))

    def _record_bytes(self, start: int, end: int) -> int:
        return len(self.buffer[start:end].encode("utf-8"))

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
                raise AmazonP0Error("source_parse_failed")
            start = self.position
            try:
                value, end = self.decoder.raw_decode(
                    self.buffer, start
                )
            except json.JSONDecodeError:
                if self.eof:
                    raise AmazonP0Error("source_parse_failed")
                if self._unconsumed_bytes() > MAX_RAW_RECORD_BYTES:
                    raise AmazonP0Error("source_record_limit_exceeded")
                self._fill()
                continue
            if self._record_bytes(start, end) > MAX_RAW_RECORD_BYTES:
                raise AmazonP0Error("source_record_limit_exceeded")
            self.position = end
            return value


def _iter_json_array_rows(handle) -> Iterator[Any]:
    reader = _BoundedJSONArrayReader(handle)
    if reader.pop() != "[":
        raise AmazonP0Error("source_parse_failed")
    if reader.peek() == "]":
        reader.pop()
        if reader.peek() != "":
            raise AmazonP0Error("source_parse_failed")
        return
    while True:
        yield reader.decode_value()
        separator = reader.pop()
        if separator == "]":
            if reader.peek() != "":
                raise AmazonP0Error("source_parse_failed")
            return
        if separator != ",":
            raise AmazonP0Error("source_parse_failed")


def iter_source_records(path: str | Path) -> Iterator[Any]:
    """Strictly stream raw Amazon records from JSONL or JSON-array files."""
    source = Path(path)
    if not source.is_file():
        raise AmazonP0Error("source_unavailable")
    name = source.name.casefold()
    jsonl = name.endswith(
        (".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")
    )
    if jsonl:
        with _open_binary(source) as handle:
            yield from _iter_jsonl_rows(handle)
        return
    if not name.endswith((".json", ".json.gz")):
        raise AmazonP0Error("source_format_not_supported")
    try:
        with _open_binary(source) as binary:
            head = binary.read(4096)
    except OSError as exc:
        raise AmazonP0Error("source_unavailable") from exc
    if not head.lstrip().startswith(b"["):
        with _open_binary(source) as handle:
            yield from _iter_jsonl_rows(handle)
        return
    with _open_binary(source) as binary:
        wrapper = io.TextIOWrapper(binary, encoding="utf-8", newline="")
        yield from _iter_json_array_rows(wrapper)


# ---------------------------------------------------------------------------
# Field adapter: mapping, validation, rejection audit.
# ---------------------------------------------------------------------------


def _finite_float(value: Any) -> float | None:
    """Convert a JSON number to a finite float, never raising on huge ints."""
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            number = float(value)
        elif isinstance(value, str) and value.strip():
            number = float(value.strip())
        else:
            return None
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _first_present(raw: dict[str, Any], *keys: str) -> Any:
    """Return the first supplied field that is present and not ``None``."""
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return None


def _timestamp_seconds(value: Any) -> float | None:
    """Normalize second, millisecond, microsecond, or nanosecond epochs."""
    number = _finite_float(value)
    if number is None or number < 0.0:
        return None
    if number >= 1e18:
        number /= 1e9
    elif number >= 1e15:
        number /= 1e6
    elif number >= 1e12:
        number /= 1e3
    elif number != int(number):
        return None
    return number


def normalize_amazon_row(
    raw: Any, source_row: int
) -> tuple[dict[str, Any] | None, str | None]:
    """Map one raw Amazon row; return ``(record, None)`` or ``(None, reason)``.

    Invalid key fields are rejected with an explicit reason instead of
    silently becoming a normal task. The normalized record keeps only the
    canonical allowlist; identity fields are locally pseudonymized.
    """
    if not isinstance(raw, dict):
        return None, "not_an_object"
    reviewer_id = _first_present(raw, "reviewerID", "user_id", "reviewer_id")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        return None, "missing_user_id"
    asin = _first_present(raw, "asin", "item_id")
    if not isinstance(asin, str) or not asin.strip():
        return None, "missing_item_id"
    overall = _first_present(raw, "overall", "rating", "stars")
    if overall is None:
        return None, "missing_stars"
    stars = _finite_float(overall)
    if stars is None or not 1.0 <= stars <= 5.0:
        return None, "invalid_stars"
    text = _first_present(raw, "reviewText", "text", "review_text")
    if text is None:
        return None, "missing_review_text"
    if not isinstance(text, str):
        return None, "invalid_review_text"
    unix_time = _first_present(raw, "unixReviewTime", "timestamp")
    if unix_time is None:
        return None, "missing_timestamp"
    number = _timestamp_seconds(unix_time)
    if number is None:
        return None, "invalid_timestamp"
    try:
        moment = datetime.fromtimestamp(number, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None, "invalid_timestamp"

    user_id = _pseudonym("u_", reviewer_id.strip())
    item_id = _pseudonym("i_", asin.strip())
    day = moment.date().isoformat()
    record = {
        "review_id": _review_id(
            user_id, item_id, day, stars, _text_digest(text), source_row
        ),
        "user_id": user_id,
        "item_id": item_id,
        "stars": stars,
        "timestamp": day,
        "text": text,
        "source_row": source_row,
    }
    return record, None


def _duplicate_audit(records: list[dict]) -> dict[str, int]:
    key_counts = Counter(
        (
            record["user_id"],
            record["item_id"],
            record["timestamp"],
            record["stars"],
            _text_digest(record["text"]),
        )
        for record in records
    )
    return {
        "duplicate_groups": sum(count > 1 for count in key_counts.values()),
        "duplicate_rows": sum(
            count - 1 for count in key_counts.values() if count > 1
        ),
    }


def load_source_records(
    path: str | Path,
) -> tuple[list[dict], dict[str, Any]]:
    """Stream, validate and audit the whole source within the shared gates."""
    source = Path(path)
    parsed_rows = 0
    non_midnight = 0
    rejected: Counter[str] = Counter()
    records: list[dict] = []
    for source_row, raw in enumerate(iter_source_records(source)):
        if source_row >= TEMPORAL_MAX_REVIEW_ROWS:
            raise AmazonP0Error("source_row_limit_exceeded")
        parsed_rows += 1
        if isinstance(raw, dict):
            number = _finite_float(raw.get("unixReviewTime"))
            if (
                number is not None
                and number >= 0.0
                and number == int(number)
                and int(number) % 86400 != 0
            ):
                non_midnight += 1
        record, reason = normalize_amazon_row(raw, source_row)
        if reason is not None:
            rejected[reason] += 1
            continue
        records.append(record)
    if parsed_rows == 0:
        raise AmazonP0Error("source_no_records")
    if not records:
        raise AmazonP0Error("source_no_valid_records")
    audit: dict[str, Any] = {
        "parsed_rows": parsed_rows,
        "valid_rows": len(records),
        "rejected_rows": sum(rejected.values()),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "non_midnight_timestamp_rows": non_midnight,
    }
    audit.update(_duplicate_audit(records))
    return records, audit


# ---------------------------------------------------------------------------
# Frozen history prefix and repeated-user target selection.
# ---------------------------------------------------------------------------


def split_frozen_history(
    records: list[dict], history_ratio: float,
) -> tuple[list[dict], list[dict], str]:
    """Cut at a day boundary near the ratio; same-day groups stay intact."""
    if not 0.0 < history_ratio < 1.0:
        raise AmazonP0Error("invalid_history_ratio")
    ordered = sorted(
        records, key=lambda row: (row["timestamp"], row["source_row"])
    )
    day_counts = Counter(row["timestamp"] for row in ordered)
    days = sorted(day_counts)
    if len(days) < 2:
        raise AmazonP0Error("insufficient_day_groups")
    total = len(ordered)
    threshold = total * history_ratio
    cumulative = 0
    best: tuple[float, int, int] | None = None
    for position in range(1, len(days)):
        cumulative += day_counts[days[position - 1]]
        distance = abs(cumulative - threshold)
        if best is None or distance < best[0]:
            best = (distance, position, cumulative)
    assert best is not None
    _, history_day_count, history_rows = best
    if history_rows <= 0:
        raise AmazonP0Error("empty_history_prefix")
    if history_rows >= total:
        raise AmazonP0Error("empty_target_region")
    cut_day = days[history_day_count]
    history = [row for row in ordered if row["timestamp"] < cut_day]
    targets = [row for row in ordered if row["timestamp"] >= cut_day]
    return history, targets, cut_day


def _spread_indexes(row_count: int, take: int) -> list[int]:
    """Evenly spaced user-row indexes, endpoints included when take > 1."""
    if take <= 1 or row_count <= 1:
        return [0] * max(1, take)
    return sorted({
        round(position * (row_count - 1) / (take - 1))
        for position in range(take)
    })


def eligible_history_stars(
    rows,
    *,
    cut_day: str,
    target_ids: set[str],
    target_digests: set[str],
) -> dict[str, Any]:
    """Shared frozen-prefix mask for the global prior.

    A prefix row is eligible only when its day is strictly before the cut and
    it is neither a selected target review id nor a matchable copy of any
    selected target text (same normalized digest rule as the strict provider).
    Target stars never enter the prior; the caller passes only already
    validated rows.
    """
    prefix_total = 0
    excluded = 0
    stars: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            raise AmazonP0Error("artifact_malformed")
        timestamp = row.get("timestamp")
        if not isinstance(timestamp, str):
            raise AmazonP0Error("artifact_malformed")
        if timestamp >= cut_day:
            continue
        prefix_total += 1
        review_id = row.get("review_id")
        text = row.get("text")
        digest = _text_digest(text) if isinstance(text, str) else None
        if (
            (isinstance(review_id, str) and review_id in target_ids)
            or (digest is not None and digest in target_digests)
        ):
            excluded += 1
            continue
        value = _finite_float(row.get("stars"))
        if value is None or not 1.0 <= value <= 5.0:
            raise AmazonP0Error("artifact_malformed")
        stars.append(value)
    return {
        "prior": sum(stars) / len(stars) if stars else None,
        "prefix_total_rows": prefix_total,
        "excluded_rows": excluded,
        "prior_rows": len(stars),
    }


def _user_task_sequence(
    rows: list[dict], *, per_user_cap: int,
) -> list[dict]:
    """One time-ordered row per distinct day, evenly spread across dates."""
    ordered = sorted(
        rows, key=lambda row: (row["timestamp"], row["source_row"])
    )
    by_date: dict[str, dict] = {}
    for row in ordered:
        timestamp = row.get("timestamp")
        if not isinstance(timestamp, str):
            raise AmazonP0Error("artifact_malformed")
        by_date.setdefault(timestamp, row)
    dates = sorted(by_date)
    take = min(per_user_cap, len(dates))
    return [
        by_date[dates[index]]
        for index in _spread_indexes(len(dates), take)
    ]


def select_targets(
    target_candidates: list[dict],
    *,
    max_tasks: int = MAX_TASKS_DEFAULT,
    seed: int = SEED_DEFAULT,
    per_user_cap: int = PER_USER_TARGET_CAP,
) -> list[dict]:
    """Select time-ordered multi-task targets from repeated users only.

    A user is a candidate only with at least two distinct UTC target dates,
    and every selected row of a user is on a distinct date. The shuffled user
    pool is capped at ``max_tasks // 2`` so every selected user receives at
    least two tasks; budgets below four tasks cannot satisfy two users with
    two time-distinct tasks each and are rejected. The final selection is
    sorted chronologically.
    """
    if max_tasks < 2:
        raise AmazonP0Error("invalid_max_tasks")
    if max_tasks < 4:
        raise AmazonP0Error("insufficient_selected_tasks")
    by_user: dict[str, list[dict]] = defaultdict(list)
    for row in target_candidates:
        by_user[row["user_id"]].append(row)
    repeated = {
        user_id: sequence
        for user_id, rows in by_user.items()
        if len(sequence := _user_task_sequence(
            rows, per_user_cap=per_user_cap
        )) >= 2
    }
    if not repeated:
        raise AmazonP0Error("insufficient_repeated_user_candidates")
    users = sorted(repeated)
    random.Random(seed).shuffle(users)
    pool = users[: max(2, max_tasks // 2)]
    cursor = {user_id: 0 for user_id in pool}
    selected: list[dict] = []
    while len(selected) < max_tasks:
        progressed = False
        for user_id in pool:
            if len(selected) >= max_tasks:
                break
            sequence = repeated[user_id]
            if cursor[user_id] < len(sequence):
                selected.append(sequence[cursor[user_id]])
                cursor[user_id] += 1
                progressed = True
        if not progressed:
            break
    selected.sort(key=lambda row: (row["timestamp"], row["source_row"]))
    per_user_dates: dict[str, list[str]] = defaultdict(list)
    for row in selected:
        per_user_dates[row["user_id"]].append(row["timestamp"])
    if len(per_user_dates) < 2:
        raise AmazonP0Error("insufficient_selected_users")
    for dates in per_user_dates.values():
        if len(dates) < 2 or len(set(dates)) < 2:
            raise AmazonP0Error("insufficient_repeated_user_candidates")
    if len(selected) < 4:
        raise AmazonP0Error("insufficient_selected_tasks")
    if len({row["timestamp"] for row in selected}) < 3:
        raise AmazonP0Error("insufficient_timestamp_groups")
    return selected


# ---------------------------------------------------------------------------
# Prepare: write the ignored local task/groundtruth/manifest/policy assets.
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def _parse_jsonl_object(line: bytes) -> dict:
    try:
        value = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AmazonP0Error("artifact_malformed") from exc
    if not isinstance(value, dict):
        raise AmazonP0Error("artifact_malformed")
    return value


def _read_jsonl(
    path: Path,
    *,
    max_rows: int = TEMPORAL_MAX_REVIEW_ROWS,
    max_line_bytes: int = MAX_RAW_RECORD_BYTES,
) -> list[dict]:
    """Bounded JSONL reader for local artifacts (never unbounded)."""
    rows: list[dict] = []
    row_count = 0
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise AmazonP0Error("artifact_malformed") from exc
    with handle:
        buffer = b""
        while True:
            chunk = handle.read(64 * 1024)
            if not chunk:
                break
            buffer += chunk
            while True:
                newline = buffer.find(b"\n")
                if newline < 0:
                    if len(buffer) > max_line_bytes:
                        raise AmazonP0Error("artifact_limit_exceeded")
                    break
                line, buffer = buffer[:newline], buffer[newline + 1:]
                if len(line) > max_line_bytes:
                    raise AmazonP0Error("artifact_limit_exceeded")
                if line.strip():
                    row_count += 1
                    if row_count > max_rows:
                        raise AmazonP0Error("artifact_limit_exceeded")
                    rows.append(_parse_jsonl_object(line))
        tail = buffer.strip()
        if tail:
            if len(tail) > max_line_bytes:
                raise AmazonP0Error("artifact_limit_exceeded")
            row_count += 1
            if row_count > max_rows:
                raise AmazonP0Error("artifact_limit_exceeded")
            rows.append(_parse_jsonl_object(tail))
    return rows


def _read_json(path: Path, *, max_bytes: int = MAX_POLICY_BYTES) -> dict:
    """Bounded JSON object reader for small, untrusted local artifacts."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AmazonP0Error("artifact_malformed") from exc
    if size > max_bytes:
        raise AmazonP0Error("artifact_limit_exceeded")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AmazonP0Error("artifact_malformed") from exc
    if not isinstance(value, dict):
        raise AmazonP0Error("artifact_malformed")
    return value


def _check_artifact_file(
    path: Path,
    *,
    max_bytes: int,
    max_rows: int | None = None,
) -> None:
    """Size/row gates that must run before any JSON parsing of an artifact."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AmazonP0Error("artifact_malformed") from exc
    if size > max_bytes:
        raise AmazonP0Error("artifact_limit_exceeded")
    if max_rows is None:
        return
    rows = 0
    last_byte = b"\n"
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                rows += chunk.count(b"\n")
                last_byte = chunk[-1:]
    except OSError as exc:
        raise AmazonP0Error("artifact_malformed") from exc
    if size and last_byte != b"\n":
        rows += 1
    if rows > max_rows:
        raise AmazonP0Error("artifact_row_limit_exceeded")


def _artifact_gate(prepared: Path) -> None:
    """Bound every run-time artifact before any parsing or provider read."""
    _check_artifact_file(
        prepared / "p0_policy.json", max_bytes=MAX_POLICY_BYTES
    )
    _check_artifact_file(
        prepared / "manifest.json", max_bytes=MAX_MANIFEST_BYTES
    )
    for name in (
        "runtime_reviews.jsonl",
        "normalized_reviews.jsonl",
        "tasks/p0.jsonl",
        "groundtruth/p0.jsonl",
    ):
        _check_artifact_file(
            prepared / name,
            max_bytes=TEMPORAL_MAX_REVIEW_SOURCE_BYTES,
            max_rows=TEMPORAL_MAX_REVIEW_ROWS,
        )


def prepare_dataset(
    source: str | Path,
    work_dir: str | Path,
    *,
    history_ratio: float = HISTORY_RATIO_DEFAULT,
    max_tasks: int = MAX_TASKS_DEFAULT,
    seed: int = SEED_DEFAULT,
    timestamp_semantics_confirmed: bool = False,
) -> dict[str, Any]:
    """Build one self-contained P0 work directory; refuse to overwrite."""
    if timestamp_semantics_confirmed is not True:
        raise AmazonP0Error("confirmation_required")
    source_path = Path(source)
    if not source_path.is_file():
        raise AmazonP0Error("source_missing")
    work = Path(work_dir)
    if work.exists():
        raise AmazonP0Error("output_dir_exists")
    try:
        work.mkdir(parents=True)
    except OSError as exc:
        raise AmazonP0Error("output_directory_unavailable") from exc
    try:
        compressed_bytes, decompressed_bytes = _preflight_source(source_path)
        records, audit = load_source_records(source_path)
        history, target_candidates, cut_day = split_frozen_history(
            records, history_ratio
        )
        selected = select_targets(
            target_candidates, max_tasks=max_tasks, seed=seed
        )
        target_ids = {record["review_id"] for record in selected}
        target_digests = {
            digest for record in selected
            if (digest := _text_digest(record["text"])) is not None
        }
        prior_info = eligible_history_stars(
            history,
            cut_day=cut_day,
            target_ids=target_ids,
            target_digests=target_digests,
        )
        global_prior = prior_info["prior"]
        if global_prior is None:
            raise AmazonP0Error("no_eligible_prior_history")

        tasks_dir = work / "tasks"
        truth_dir = work / "groundtruth"
        task_rows = []
        truth_rows = []
        for task_index, record in enumerate(selected):
            identity = {
                "task_index": task_index,
                "user_id": record["user_id"],
                "item_id": record["item_id"],
                "target_date": record["timestamp"],
                "target_review_id": record["review_id"],
            }
            task_rows.append(identity)
            truth_rows.append({
                **identity,
                "stars": record["stars"],
                # The withheld target review text stays in this local ignored
                # groundtruth directory only. It lets the strict provider
                # withhold duplicate/matching target texts from every context.
                "text": record["text"],
            })
        _write_jsonl(tasks_dir / "p0.jsonl", task_rows)
        _write_jsonl(truth_dir / "p0.jsonl", truth_rows)
        _write_jsonl(
            work / "runtime_reviews.jsonl", [*history, *selected]
        )
        _write_jsonl(work / "normalized_reviews.jsonl", records)

        manifest = build_temporal_manifest(
            tasks_dir,
            truth_dir,
            [work / "runtime_reviews.jsonl"],
            train_ratio=TRAIN_RATIO,
            validation_ratio=VALIDATION_RATIO,
            timestamp_semantics_caller_asserted=True,
        )
        match_counts = manifest["target_review_row_match_counts"]
        if (
            match_counts.get("unique") != len(selected)
            or match_counts.get("ambiguous")
            or match_counts.get("unmatched")
        ):
            raise AmazonP0Error("target_not_uniquely_matched")
        if not manifest.get("runtime_eligible"):
            raise AmazonP0Error("manifest_not_runtime_eligible")
        split_counts = dict(manifest.get("split_counts") or {})
        for split in ("train", "validation", "test"):
            if split_counts.get(split, 0) < 1:
                raise AmazonP0Error(f"{split}_split_empty")
        manifest_rows = manifest["tasks"]
        if len({row["user_group"] for row in manifest_rows}) < 2:
            raise AmazonP0Error("single_user_cluster")
        if not any(
            row["user_repeat_stratum"] == "repeat" for row in manifest_rows
        ):
            raise AmazonP0Error("no_repeat_tasks")
        if sum(row["visible_history_count"] for row in manifest_rows) < 1:
            raise AmazonP0Error("no_visible_history")

        (work / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        user_ids = {row["user_id"] for row in records}
        item_ids = {row["item_id"] for row in records}
        rows_per_user = Counter(row["user_id"] for row in records)
        rows_per_item = Counter(row["item_id"] for row in records)
        policy = {
            "schema_version": 1,
            "adapter": "amazon_p0",
            "adapter_version": ADAPTER_VERSION,
            "source": {
                "file_name": source_path.name,
                "source_url": SOURCE_FILE_URL,
                "page_url": SOURCE_PAGE_URL,
                "citation": DATASET_CITATION,
                "sha256": _sha256_file(source_path),
                "compressed_bytes": compressed_bytes,
                "decompressed_bytes": decompressed_bytes,
                **audit,
                "distinct_users": len(user_ids),
                "distinct_items": len(item_ids),
                "min_rows_per_user": min(rows_per_user.values()),
                "min_rows_per_item": min(rows_per_item.values()),
                "license_status": LICENSE_STATUS,
                "license_note": LICENSE_NOTE,
            },
            "fields": {
                "mapping": FIELD_MAPPING,
                "canonical_fields": list(CANONICAL_FIELDS),
                "dropped_fields": list(DROPPED_FIELDS),
                "time_granularity": (
                    "UTC calendar day; same-day reviews are never treated "
                    "as causally ordered"
                ),
                "timestamp_semantics": (
                    "caller_asserted_amazon_2014_unix_review_time; code "
                    "verifies only epoch-integer conversion and day grouping"
                ),
            },
            "protocol": {
                "history_ratio": history_ratio,
                "cut_day": cut_day,
                "history_rows": len(history),
                "target_region_rows": len(target_candidates),
                "selected_tasks": len(selected),
                "selected_distinct_users": len(
                    {row["user_id"] for row in selected}
                ),
                "max_tasks": max_tasks,
                "seed": seed,
                "per_user_target_cap": PER_USER_TARGET_CAP,
                "selection_policy": (
                    "target-region users with at least two distinct target "
                    "dates only; seed-shuffled user pool bounded at "
                    "max_tasks//2 so every selected user gets at least two "
                    "time-distinct tasks; never one task per user"
                ),
                "frozen_history": True,
                "runtime_source_composition": (
                    "frozen history prefix + selected target source rows"
                ),
                "target_visibility": (
                    "every selected target row is withheld from every "
                    "context in every split, including earlier targets"
                ),
                "observed_history_note": (
                    "Observed history is the frozen prefix and never rolls "
                    "forward with after-cut non-target rows. The manifest "
                    "train/validation/test partition applies to the selected "
                    "target cohort only; it is not a partition of the raw "
                    "file and all three splits see the same frozen history."
                ),
                "split_ratios": {
                    "train": TRAIN_RATIO,
                    "validation": VALIDATION_RATIO,
                    "same_day_groups": "indivisible",
                },
                "manifest_split_counts": split_counts,
                "global_prior": global_prior,
                "global_prior_rows": prior_info["prior_rows"],
                "prior_excluded_rows": prior_info["excluded_rows"],
                "global_prior_source": GLOBAL_PRIOR_SOURCE,
            },
            "baselines": {
                "modes": list(BASELINE_MODES),
                "lambda_user": DEFAULT_LAMBDA_USER,
                "lambda_item": DEFAULT_LAMBDA_ITEM,
                "weight_user": DEFAULT_WEIGHT_USER,
                "fallback": "global prior (counted as cold start)",
                "output_range": "continuous 1.0-5.0",
                "tuning": "fixed defaults; no test/validation tuning",
            },
            "bootstrap": {
                "seed": DEFAULT_BOOTSTRAP_SEED,
                "n_boot": DEFAULT_BOOTSTRAP_SAMPLES,
                "unit": "user cluster, resampled with replacement",
            },
            "privacy": (
                "aggregate counts only in CLI output; identifiers are local "
                "pseudonyms and raw review text stays in gitignored local "
                "files"
            ),
        }
        (work / "p0_policy.json").write_text(
            json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "prepared",
            "parsed_rows": audit["parsed_rows"],
            "valid_rows": audit["valid_rows"],
            "rejected_rows": audit["rejected_rows"],
            "duplicate_rows": audit["duplicate_rows"],
            "distinct_users": len(user_ids),
            "distinct_items": len(item_ids),
            "cut_day": cut_day,
            "history_rows": len(history),
            "prior_excluded_rows": prior_info["excluded_rows"],
            "global_prior_rows": prior_info["prior_rows"],
            "selected_tasks": len(selected),
            "split_counts": split_counts,
        }
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Run: consume the strict provider, then score four baselines.
# ---------------------------------------------------------------------------


def rating_context_from_provider(context) -> RatingContext:
    """Convert one provider context into the label-safe baseline context."""
    user_stars = tuple(
        float(row["stars"]) for row in context.history if "stars" in row
    )
    item_stars = tuple(
        float(row["stars"])
        for row in context.item_references
        if "stars" in row
    )
    return RatingContext(
        task_index=context.task_index,
        split=context.split,
        user_group=context.user_group,
        user_repeat_stratum=context.user_repeat_stratum,
        user_stars=user_stars,
        item_stars=item_stars,
    )


def _validated_policy(policy: dict) -> dict[str, Any]:
    """Validate policy shape and the fixed, actually-implemented parameters.

    Descriptive free text is never returned: the report rebuilds every
    narrative field from module constants, so tampering with note fields
    cannot leak into outputs.
    """
    if (
        policy.get("schema_version") != 1
        or policy.get("adapter") != "amazon_p0"
        or policy.get("adapter_version") != ADAPTER_VERSION
    ):
        raise AmazonP0Error("policy_version_mismatch")
    baselines = policy.get("baselines")
    if not isinstance(baselines, dict):
        raise AmazonP0Error("policy_parameter_mismatch")
    if baselines.get("modes") != list(BASELINE_MODES):
        raise AmazonP0Error("policy_parameter_mismatch")
    for key, expected in (
        ("lambda_user", DEFAULT_LAMBDA_USER),
        ("lambda_item", DEFAULT_LAMBDA_ITEM),
        ("weight_user", DEFAULT_WEIGHT_USER),
    ):
        value = baselines.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) != float(expected)
        ):
            raise AmazonP0Error("policy_parameter_mismatch")
    protocol = policy.get("protocol")
    if not isinstance(protocol, dict):
        raise AmazonP0Error("policy_protocol_mismatch")
    cut_day = protocol.get("cut_day")
    if not isinstance(cut_day, str):
        raise AmazonP0Error("policy_protocol_mismatch")
    try:
        parsed_day = datetime.fromisoformat(cut_day)
    except ValueError as exc:
        raise AmazonP0Error("policy_protocol_mismatch") from exc
    if parsed_day.date().isoformat() != cut_day:
        raise AmazonP0Error("policy_protocol_mismatch")
    if protocol.get("frozen_history") is not True:
        raise AmazonP0Error("policy_protocol_mismatch")
    history_ratio = protocol.get("history_ratio")
    if (
        isinstance(history_ratio, bool)
        or not isinstance(history_ratio, (int, float))
        or not 0.0 < float(history_ratio) < 1.0
    ):
        raise AmazonP0Error("policy_protocol_mismatch")
    if protocol.get("per_user_target_cap") != PER_USER_TARGET_CAP:
        raise AmazonP0Error("policy_protocol_mismatch")
    numbers: dict[str, int] = {}
    for key in (
        "history_rows",
        "target_region_rows",
        "selected_tasks",
        "max_tasks",
        "seed",
        "prior_excluded_rows",
        "global_prior_rows",
        "selected_distinct_users",
    ):
        value = protocol.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AmazonP0Error("policy_protocol_mismatch")
        numbers[key] = value
    if numbers["selected_tasks"] < 4 or numbers["history_rows"] < 1:
        raise AmazonP0Error("policy_protocol_mismatch")
    prior = protocol.get("global_prior")
    if (
        isinstance(prior, bool)
        or not isinstance(prior, (int, float))
        or not math.isfinite(float(prior))
        or not 1.0 <= float(prior) <= 5.0
    ):
        raise AmazonP0Error("policy_protocol_mismatch")
    split_counts = protocol.get("manifest_split_counts")
    if not isinstance(split_counts, dict):
        raise AmazonP0Error("policy_protocol_mismatch")
    validated_splits: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        value = split_counts.get(split)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AmazonP0Error("policy_protocol_mismatch")
        validated_splits[split] = value
    return {
        "cut_day": cut_day,
        "history_ratio": float(history_ratio),
        "history_rows": numbers["history_rows"],
        "target_region_rows": numbers["target_region_rows"],
        "selected_tasks": numbers["selected_tasks"],
        "max_tasks": numbers["max_tasks"],
        "seed": numbers["seed"],
        "prior_excluded_rows": numbers["prior_excluded_rows"],
        "global_prior_rows": numbers["global_prior_rows"],
        "selected_distinct_users": numbers["selected_distinct_users"],
        "global_prior": float(prior),
        "manifest_split_counts": validated_splits,
    }


def _reconstruct_selection(
    prepared: Path, validated: dict[str, Any],
) -> tuple[str, list[str], int]:
    """Rebuild the frozen cut and target selection from the normalized source.

    This verifies that the recorded cut day, ratios, seed, task budget and
    selected review ids match the current normalized artifact instead of
    trusting the policy file.
    """
    records = _read_jsonl(prepared / "normalized_reviews.jsonl")
    _, target_candidates, cut_day = split_frozen_history(
        records, validated["history_ratio"]
    )
    selected = select_targets(
        target_candidates,
        max_tasks=validated["max_tasks"],
        seed=validated["seed"],
    )
    return cut_day, [record["review_id"] for record in selected], len(
        target_candidates
    )


def _target_text_digests(prepared: Path) -> set[str]:
    """Collect withheld target text digests; target stars are never touched."""
    digests: set[str] = set()
    for row in _read_jsonl(prepared / "groundtruth" / "p0.jsonl"):
        text = row.get("text")
        if isinstance(text, str):
            digest = _text_digest(text)
            if digest is not None:
                digests.add(digest)
    return digests


def _verified_prior(
    prepared: Path,
    validated: dict[str, Any],
    target_ids: set[str],
    target_digests: set[str],
) -> dict[str, Any]:
    """Recompute the eligible-frozen-prefix prior and verify the policy."""
    prior_info = eligible_history_stars(
        _read_jsonl(prepared / "runtime_reviews.jsonl"),
        cut_day=validated["cut_day"],
        target_ids=target_ids,
        target_digests=target_digests,
    )
    if prior_info["prior"] is None:
        raise AmazonP0Error("no_eligible_prior_history")
    if (
        prior_info["prefix_total_rows"] != validated["history_rows"]
        or prior_info["excluded_rows"] != validated["prior_excluded_rows"]
        or prior_info["prior_rows"] != validated["global_prior_rows"]
    ):
        raise AmazonP0Error("global_prior_source_mismatch")
    if abs(prior_info["prior"] - validated["global_prior"]) > 1e-12:
        raise AmazonP0Error("global_prior_mismatch")
    return prior_info


def _validated_groundtruth(prepared: Path) -> dict[int, float]:
    """Read actual stars only after all predictions were computed."""
    actuals: dict[int, float] = {}
    for row in _read_jsonl(prepared / "groundtruth" / "p0.jsonl"):
        task_index = row.get("task_index")
        stars = row.get("stars")
        if (
            not isinstance(task_index, int)
            or isinstance(stars, bool)
            or not isinstance(stars, (int, float))
        ):
            raise AmazonP0Error("artifact_malformed")
        value = float(stars)
        if not math.isfinite(value) or not 1.0 <= value <= 5.0:
            raise AmazonP0Error("artifact_malformed")
        if task_index in actuals:
            raise AmazonP0Error("artifact_malformed")
        actuals[task_index] = value
    return actuals


def build_report(
    validated: dict[str, Any],
    prior_info: dict[str, Any],
    scored: dict,
    contexts,
    *,
    prepared: Path,
) -> dict[str, Any]:
    """Aggregate report built from validated numbers and fixed constants only."""
    tasks_with_history = sum(
        bool(context.history) for context in contexts
    )
    tasks_with_references = sum(
        bool(context.item_references) for context in contexts
    )
    total_history_rows = sum(len(context.history) for context in contexts)
    total_reference_rows = sum(
        len(context.item_references) for context in contexts
    )
    return {
        "schema_version": 1,
        "adapter_version": ADAPTER_VERSION,
        "protocol": {
            "cut_day": validated["cut_day"],
            "history_rows": validated["history_rows"],
            "prior_excluded_rows": validated["prior_excluded_rows"],
            "global_prior_rows": validated["global_prior_rows"],
            "target_region_rows": validated["target_region_rows"],
            "selected_tasks": validated["selected_tasks"],
            "selected_distinct_users": validated["selected_distinct_users"],
            "seed": validated["seed"],
            "history_ratio": validated["history_ratio"],
            "max_tasks": validated["max_tasks"],
            "frozen_history": True,
            "runtime_source_composition": RUNTIME_SOURCE_COMPOSITION,
            "target_visibility": TARGET_VISIBILITY,
            "observed_history_note": OBSERVED_HISTORY_NOTE,
            "split_scope_note": SPLIT_SCOPE_NOTE,
            "global_prior": validated["global_prior"],
            "global_prior_source": GLOBAL_PRIOR_SOURCE,
            "baseline_params": {
                "modes": list(BASELINE_MODES),
                "lambda_user": DEFAULT_LAMBDA_USER,
                "lambda_item": DEFAULT_LAMBDA_ITEM,
                "weight_user": DEFAULT_WEIGHT_USER,
                "fallback": "global prior (counted as cold start)",
                "output_range": "continuous 1.0-5.0",
                "tuning": "fixed defaults; no test/validation tuning",
            },
            "manifest_sha256": _sha256_file(prepared / "manifest.json"),
            "runtime_reviews_sha256": _sha256_file(
                prepared / "runtime_reviews.jsonl"
            ),
        },
        "task_counts": {
            **scored["task_counts"],
            "tasks_with_visible_history": tasks_with_history,
            "tasks_with_item_references": tasks_with_references,
            "total_visible_history_rows": total_history_rows,
            "total_item_reference_rows": total_reference_rows,
        },
        "baselines": scored["point"],
        "bootstrap_test": scored["bootstrap_test"],
        "interpretation": (
            "These are deterministic offline baselines computed from the "
            "frozen non-target history prefix and time-filtered provider "
            "contexts. They measure whether a static rating policy beats the "
            "global mean on this conditional target sample; they are not "
            "Memory or LLM results and say nothing about generated reviews."
        ),
        "privacy": (
            "aggregate report only: no identifiers, review text, prompts or "
            "paths"
        ),
    }


def render_markdown_report(report: dict) -> str:
    protocol = report["protocol"]
    lines = [
        "# Offline P0 rating baseline report",
        "",
        "Aggregate, redacted output. No identifiers, review text, or paths.",
        "",
        "## Protocol",
        "",
        f"- Cut day (first target day): {protocol.get('cut_day')}",
        f"- Frozen history rows: {protocol.get('history_rows')}",
        f"- Target region rows: {protocol.get('target_region_rows')}",
        f"- Selected tasks: {protocol.get('selected_tasks')}",
        f"- Selection seed: {protocol.get('seed')}",
        f"- Frozen history (non-rolling): {protocol.get('frozen_history')}",
        f"- Runtime source: {protocol.get('runtime_source_composition')}",
        f"- Target visibility: {protocol.get('target_visibility')}",
        f"- Global prior: {protocol.get('global_prior')} "
        f"({protocol.get('global_prior_rows')} eligible rows; "
        f"{protocol.get('prior_excluded_rows')} target-copy rows excluded)",
        f"- Split scope: {protocol.get('split_scope_note')}",
        "",
        "## Task counts",
        "",
    ]
    counts = report["task_counts"]
    lines.append(
        f"- Total: {counts['total']} | splits: {counts['by_split']}"
    )
    lines.append(
        f"- User clusters: {counts['user_clusters']['total']} "
        f"(test: {counts['user_clusters']['by_split']['test']})"
    )
    lines.append(
        f"- Tasks with visible history: "
        f"{counts['tasks_with_visible_history']} | item references: "
        f"{counts['tasks_with_item_references']}"
    )
    lines.extend([
        "",
        "## Baselines (test split)",
        "",
        "| mode | n | RMSE | MAE | mean signed bias |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for mode in BASELINE_MODES:
        test = report["baselines"][mode]["test"]

        def show(value, digits=4):
            return "n/a" if value is None else f"{value:.{digits}f}"

        lines.append(
            f"| {mode} | {test['n']} | {show(test['rmse'])} | "
            f"{show(test['mae'])} | {show(test['mean_signed_bias'])} |"
        )
    bootstrap = report["bootstrap_test"]
    lines.extend(["", "## User-cluster bootstrap (test)", ""])
    if bootstrap.get("status") != "computed":
        lines.append(
            f"- Not computed: {bootstrap.get('status')} "
            f"({bootstrap.get('detail', '')})"
        )
    else:
        lines.append(
            f"- {bootstrap['n_boot']} resamples, seed {bootstrap['seed']}, "
            f"{bootstrap['cluster_count']} user clusters, unit "
            f"{bootstrap['resampling_unit']}"
        )
        lines.extend([
            "",
            "| mode | RMSE CI95 | MAE CI95 | RMSE diff vs global CI95 |",
            "| --- | --- | --- | --- |",
        ])

        def interval(value):
            if not value:
                return "n/a"
            return f"[{value[0]:.4f}, {value[1]:.4f}]"

        for mode in BASELINE_MODES:
            intervals = bootstrap["intervals"][mode]
            paired = bootstrap["paired_vs_global"].get(mode)
            lines.append(
                f"| {mode} | {interval(intervals['rmse'])} | "
                f"{interval(intervals['mae'])} | "
                f"{interval(paired['rmse_difference']['ci95']) if paired else 'reference'} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
    ])
    return "\n".join(lines)


def run_baselines(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Consume the strict provider, predict, then score; write new outputs."""
    prepared = Path(prepared_dir)
    if not prepared.is_dir():
        raise AmazonP0Error("prepared_dir_missing")
    for name in ARTIFACT_NAMES:
        if not (prepared / name).exists():
            raise AmazonP0Error("prepared_artifact_missing")
    output = Path(output_dir)
    if output.exists():
        raise AmazonP0Error("output_dir_exists")
    try:
        output.mkdir(parents=True)
    except OSError as exc:
        raise AmazonP0Error("output_directory_unavailable") from exc
    try:
        _artifact_gate(prepared)
        policy = _read_json(
            prepared / "p0_policy.json", max_bytes=MAX_POLICY_BYTES
        )
        validated = _validated_policy(policy)
        task_rows = _read_jsonl(prepared / "tasks" / "p0.jsonl")
        selected_tasks = validated["selected_tasks"]
        if len(task_rows) != selected_tasks:
            raise AmazonP0Error("policy_artifact_mismatch")
        target_ids: set[str] = set()
        for row in task_rows:
            review_id = row.get("target_review_id")
            if not isinstance(review_id, str) or not review_id:
                raise AmazonP0Error("artifact_malformed")
            target_ids.add(review_id)
        if len(target_ids) != selected_tasks:
            raise AmazonP0Error("artifact_malformed")

        # Rebuild the cut and selection from the normalized source: the
        # recorded ratio/seed/budget/cut and the task list must match.
        reconstructed_cut, reconstructed_ids, region_rows = (
            _reconstruct_selection(prepared, validated)
        )
        if reconstructed_cut != validated["cut_day"]:
            raise AmazonP0Error("policy_artifact_mismatch")
        if region_rows != validated["target_region_rows"]:
            raise AmazonP0Error("policy_artifact_mismatch")
        if reconstructed_ids != [
            row["target_review_id"] for row in task_rows
        ]:
            raise AmazonP0Error("policy_artifact_mismatch")
        if (
            len(set(reconstructed_ids)) != selected_tasks
            or len({
                row["user_id"] for row in task_rows
            }) != validated["selected_distinct_users"]
        ):
            raise AmazonP0Error("policy_artifact_mismatch")

        # The prior is recomputed from the frozen prefix with the same
        # target-id and target-text-copy mask; target stars are not involved.
        target_digests = _target_text_digests(prepared)
        prior_info = _verified_prior(
            prepared, validated, target_ids, target_digests
        )
        prior = prior_info["prior"]

        provider = TemporalContextProvider(
            prepared / "tasks",
            prepared / "groundtruth",
            [prepared / "runtime_reviews.jsonl"],
            prepared / "manifest.json",
        )
        contexts = provider.contexts_in_execution_order()
        if len(contexts) != selected_tasks:
            raise AmazonP0Error("runtime_eligibility_changed")
        computed_split_counts = {
            split: sum(context.split == split for context in contexts)
            for split in ("train", "validation", "test")
        }
        if computed_split_counts != validated["manifest_split_counts"]:
            raise AmazonP0Error("policy_artifact_mismatch")
        rating_contexts = [
            rating_context_from_provider(context) for context in contexts
        ]

        # Prediction functions receive no groundtruth ratings. The scorer
        # below is the only place that reads actual stars, after predictions.
        predictions = predict_all(rating_contexts, prior)
        actuals = _validated_groundtruth(prepared)
        expected_indexes = {context.task_index for context in contexts}
        if set(actuals) != expected_indexes:
            raise AmazonP0Error("groundtruth_index_mismatch")
        split_by_index = {
            context.task_index: context.split for context in rating_contexts
        }
        cluster_by_index = {
            context.task_index: context.user_group
            for context in rating_contexts
        }
        stratum_by_index = {
            context.task_index: context.user_repeat_stratum
            for context in rating_contexts
        }
        scored = score_baselines(
            predictions,
            actuals,
            split_by_index=split_by_index,
            cluster_by_index=cluster_by_index,
            stratum_by_index=stratum_by_index,
            n_boot=n_boot,
            seed=seed,
        )
        report = build_report(
            validated, prior_info, scored, contexts, prepared=prepared
        )
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "report.md").write_text(
            render_markdown_report(report), encoding="utf-8"
        )
        prediction_rows = []
        scoring_rows = []
        for mode in BASELINE_MODES:
            for prediction in predictions[mode]:
                prediction_rows.append({
                    "task_index": prediction.task_index,
                    "split": split_by_index[prediction.task_index],
                    "mode": mode,
                    "prediction": prediction.value,
                    "user_cold_start": prediction.user_cold_start,
                    "item_cold_start": prediction.item_cold_start,
                })
                actual = actuals[prediction.task_index]
                scoring_rows.append({
                    "task_index": prediction.task_index,
                    "mode": mode,
                    "actual": actual,
                    "signed_error": prediction.value - actual,
                    "absolute_error": abs(prediction.value - actual),
                })
        _write_jsonl(output / "predictions.jsonl", prediction_rows)
        _write_jsonl(output / "scoring.jsonl", scoring_rows)
        return report
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline, no-LLM P0 pipeline for the public Amazon 2014 Musical "
            "Instruments 5-core dataset."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download",
        help="Explicitly fetch the pinned 2.46 MB official archive.",
    )
    download.add_argument(
        "--url", default=SOURCE_FILE_URL,
        help="Only the exact pinned official HTTPS archive is accepted.",
    )
    download.add_argument("--output", required=True)
    download.add_argument(
        "--max-bytes", type=int, default=MAX_DOWNLOAD_BYTES
    )

    prepare = subparsers.add_parser(
        "prepare",
        help="Build probe-free tasks, groundtruth, manifest and policy.",
    )
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--work-dir", required=True)
    prepare.add_argument(
        "--history-ratio", type=float, default=HISTORY_RATIO_DEFAULT
    )
    prepare.add_argument("--max-tasks", type=int, default=MAX_TASKS_DEFAULT)
    prepare.add_argument("--seed", type=int, default=SEED_DEFAULT)
    prepare.add_argument(
        "--confirm-target-timestamp-semantics", action="store_true",
        help=(
            "Caller assertion that unixReviewTime is the review event time. "
            "Required; the manifest still records that code did not verify "
            "domain semantics."
        ),
    )

    run = subparsers.add_parser(
        "run",
        help="Revalidate the strict provider and score four baselines.",
    )
    run.add_argument("--prepared-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument(
        "--n-boot", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    run.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            info = download_source(
                args.url, args.output, max_bytes=args.max_bytes
            )
            print(
                f"download ok: bytes={info['bytes']} "
                f"sha256={info['sha256']}"
            )
        elif args.command == "prepare":
            summary = prepare_dataset(
                args.source,
                args.work_dir,
                history_ratio=args.history_ratio,
                max_tasks=args.max_tasks,
                seed=args.seed,
                timestamp_semantics_confirmed=(
                    args.confirm_target_timestamp_semantics
                ),
            )
            print(json.dumps(summary, sort_keys=True))
        else:
            report = run_baselines(
                args.prepared_dir,
                args.output_dir,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            print(
                "p0 run complete: "
                f"tasks={report['task_counts']['total']} "
                f"test={report['task_counts']['by_split']['test']}"
            )
    except TemporalManifestError:
        print("p0 failed: runtime_manifest_rejected", file=sys.stderr)
        return 2
    except TemporalManifestInputError as exc:
        print(f"p0 failed: {exc.code}", file=sys.stderr)
        return 2
    except AmazonP0Error as exc:
        print(f"p0 failed: {exc.code}", file=sys.stderr)
        return 2
    except BaselineInputError as exc:
        print(f"p0 failed: {exc.code}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError, ArithmeticError,
            json.JSONDecodeError):
        print("p0 failed: internal_error", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
