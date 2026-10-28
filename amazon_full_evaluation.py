#!/usr/bin/env python3
"""Full raw-record Amazon evaluation for Agent A.

This runner is deliberately separate from the historical 400-task framework
runner.  It reads every record in the Amazon Musical Instruments raw source,
assigns an explicit outcome to every raw row, and evaluates valid rows in
chronological order with a task-local, time-safe interaction tool.

The raw source, target labels, generated reviews and per-record results are
runtime artifacts only.  They are expected to live in the user's Drive/Colab
workspace and must not be copied into the repository.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

from amazon_p0 import (
    _pseudonym,
    _text_digest,
    iter_source_records,
    normalize_amazon_row,
)


RUNNER_VERSION = "amazon-raw-full-temporal-v2"
FULL_EVALUATION_MODEL_POLICY = (
    "The full raw evaluation requires an explicitly supplied, approved current "
    "model configuration; the historical DeepSeek defaults are disabled."
)
MAX_ITEM_DESCRIPTION_CHARS = 800
PROGRESS_EVERY = 25

POSITIVE_TERMS = {
    "good", "great", "excellent", "amazing", "awesome", "love", "loved",
    "like", "liked", "best", "perfect", "recommend", "recommended",
    "happy", "helpful", "quality", "easy", "works", "worked", "nice",
}
NEGATIVE_TERMS = {
    "bad", "poor", "terrible", "awful", "worst", "hate", "hated", "dislike",
    "disappointed", "disappointing", "broken", "broke", "cheap", "difficult",
    "hard", "waste", "return", "returned", "problem", "problems", "failure",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    """Yield raw rows from the supported JSON source or its CSV mirror."""
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    yield from iter_source_records(path)


def locate_amazon_source(raw_root: str | Path) -> Path:
    root = Path(raw_root)
    candidates = (
        root / "reviews_Musical_Instruments_5.json.gz",
        root / "reviews_Musical_Instruments_5.json",
        root / "reviews_Musical_Instruments_5.jsonl",
        root / "reviews_Musical_Instruments_5.jsonl.gz",
        root / "Musical_Instruments.jsonl",
        root / "Musical_Instruments.json",
        root / "Musical_Instruments.json.gz",
        root / "Musical_Instruments.csv",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    # Permit a Drive staging directory with a conventional review filename,
    # while never silently selecting a metadata file or an unrelated dataset.
    fallback = sorted(
        candidate for candidate in root.iterdir()
        if candidate.is_file()
        and candidate.name.casefold().startswith("reviews_musical_instruments_5")
        and candidate.suffix.casefold() in {".json", ".gz", ".jsonl"}
    )
    if fallback:
        return fallback[0]
    raise FileNotFoundError("amazon_musical_instruments_raw_source_missing")


def load_amazon_records(source: Path) -> tuple[list[dict], list[dict], dict]:
    """Normalize every raw row and retain a redacted status for rejects."""
    valid: list[dict] = []
    rejected: list[dict] = []
    parsed_rows = 0
    for raw_index, raw in enumerate(_jsonl_records(source)):
        parsed_rows += 1
        record, reason = normalize_amazon_row(raw, raw_index)
        if reason is not None:
            rejected.append({
                "raw_index": raw_index,
                "status": "not_evaluated",
                "status_reason": reason,
                "reason": reason,
            })
            continue
        record = dict(record)
        record["raw_index"] = raw_index
        record["text_digest"] = _text_digest(record.get("text"))
        valid.append(record)
    if parsed_rows == 0:
        raise ValueError("amazon_raw_source_empty")
    if not valid:
        raise ValueError("amazon_raw_source_has_no_valid_rows")
    return valid, rejected, {
        "parsed_rows": parsed_rows,
        "valid_rows": len(valid),
        "rejected_rows": len(rejected),
        "rejected_by_reason": dict(sorted(
            Counter(row["reason"] for row in rejected).items()
        )),
    }


def _text_value(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_text_value(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_text_value(item) for item in value.values())
    return str(value or "").strip()


def load_item_metadata(raw_root: str | Path) -> dict[str, dict]:
    """Load non-label item facts without exposing target review data."""
    root = Path(raw_root)
    source = next(
        (
            candidate for candidate in (
                root / "meta_Musical_Instruments.jsonl.gz",
                root / "meta_Musical_Instruments.json.gz",
                root / "meta_Musical_Instruments.jsonl",
                root / "meta_Musical_Instruments.json",
            )
            if candidate.is_file() and candidate.stat().st_size > 0
        ),
        None,
    )
    if source is None:
        return {}

    metadata: dict[str, dict] = {}
    for raw in _jsonl_records(source):
        if not isinstance(raw, dict):
            continue
        asin = raw.get("asin") or raw.get("item_id") or raw.get("id")
        if not isinstance(asin, str) or not asin.strip():
            continue
        category = raw.get("category") or raw.get("categories")
        description = _text_value(raw.get("description"))
        item = {
            "item_id": _pseudonym("i_", asin.strip()),
            "title": _text_value(raw.get("title"))[:300],
            "brand": _text_value(raw.get("brand"))[:120],
            "category": _text_value(category)[:300],
            "description": description[:MAX_ITEM_DESCRIPTION_CHARS],
            "price": _text_value(raw.get("price"))[:80],
        }
        metadata[item["item_id"]] = {
            key: value for key, value in item.items()
            if value
        }
    return metadata


def _decorate(row: dict, *, section: str) -> dict:
    """Make a prompt-safe copy with provenance fields used by diagnostics."""
    value = {
        "review_id": row.get("review_id"),
        "user_id": row.get("user_id"),
        "item_id": row.get("item_id"),
        "stars": float(row.get("stars", 3.0)),
        "text": str(row.get("text") or ""),
        "_temporal_source_row_index": int(row.get("raw_index", row.get("source_row", 0))),
        "_temporal_event_time": row.get("timestamp"),
        "useful": 0,
        "funny": 0,
        "cool": 0,
    }
    value["_context_section"] = section
    return value


class TemporalAmazonInteractionTool:
    """Expose only rows strictly before the current target calendar day."""

    def __init__(self, records: list[dict], item_metadata: dict[str, dict]):
        self.records = records
        self.item_metadata = item_metadata
        self.by_user: dict[str, list[dict]] = defaultdict(list)
        self.by_item: dict[str, list[dict]] = defaultdict(list)
        self.ordered_timestamps = sorted(
            (str(row["timestamp"]), int(row["raw_index"]), row)
            for row in records
        )
        self.by_timestamp: dict[str, list[dict]] = defaultdict(list)
        for row in records:
            self.by_user[row["user_id"]].append(row)
            self.by_item[row["item_id"]].append(row)
            self.by_timestamp[row["timestamp"]].append(row)
        for rows in (*self.by_user.values(), *self.by_item.values()):
            rows.sort(key=lambda row: (row["timestamp"], row["raw_index"]))
        self.current_target: dict | None = None
        self.current_history: list[dict] = []
        self.current_references: list[dict] = []

    def set_target(self, target: dict) -> tuple[list[dict], list[dict]]:
        self.current_target = target
        target_date = target["timestamp"]
        target_digest = target.get("text_digest")
        user_id = target["user_id"]
        item_id = target["item_id"]

        self.current_history = [
            _decorate(row, section="history")
            for row in self.by_user.get(user_id, [])
            if row["timestamp"] < target_date
            and row.get("text_digest") != target_digest
        ]
        self.current_references = [
            _decorate(row, section="reference")
            for row in self.by_item.get(item_id, [])
            if row["timestamp"] < target_date
            and row["user_id"] != user_id
            and row.get("text_digest") != target_digest
        ]
        return self.current_history, self.current_references

    def get_user(self, user_id: str) -> dict:
        if self.current_target is None or user_id != self.current_target["user_id"]:
            return {"user_id": user_id, "review_count": 0}
        stars = [float(row["stars"]) for row in self.current_history]
        return {
            "user_id": user_id,
            "review_count": len(self.current_history),
            "history_mean_stars": (
                round(sum(stars) / len(stars), 3) if stars else None
            ),
        }

    def get_item(self, item_id: str) -> dict:
        return self.item_metadata.get(item_id, {"item_id": item_id})

    def get_reviews(self, user_id: str | None = None,
                    item_id: str | None = None) -> list[dict]:
        if user_id is not None:
            if self.current_target and user_id == self.current_target["user_id"]:
                return list(self.current_history)
            return []
        if item_id is not None:
            if self.current_target and item_id == self.current_target["item_id"]:
                return list(self.current_references)
            return []
        return []


def _global_prior_by_date(records: list[dict]) -> dict[str, float | None]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[row["timestamp"]].append(float(row["stars"]))
    result: dict[str, float | None] = {}
    total = 0.0
    count = 0
    for date in sorted(grouped):
        result[date] = total / count if count else None
        total += sum(grouped[date])
        count += len(grouped[date])
    return result


def _mean_stars(rows: Iterable[dict]) -> float | None:
    values = [float(row["stars"]) for row in rows]
    return sum(values) / len(values) if values else None


def _snap(value: float | None, default: float = 3.0) -> float:
    if value is None or not math.isfinite(float(value)):
        value = default
    return float(min(5.0, max(1.0, math.floor(float(value) + 0.5))))


def target_baselines(target: dict, tool: TemporalAmazonInteractionTool,
                     global_priors: dict[str, float | None]) -> dict:
    history, references = tool.set_target(target)
    global_mean = global_priors.get(target["timestamp"])
    return {
        "user_mean": _mean_stars(history),
        "item_mean": _mean_stars(references),
        "global_mean": global_mean,
        "user_mean_rounded": _snap(_mean_stars(history), _snap(global_mean)),
        "item_mean_rounded": _snap(_mean_stars(references), _snap(global_mean)),
        "global_mean_rounded": _snap(global_mean),
        "history_count": len(history),
        "item_reference_count": len(references),
        "history_mean_length": (
            sum(len(row.get("text", "")) for row in history) / len(history)
            if history else None
        ),
        "history_tone_score": _tone_score(
            " ".join(row.get("text", "") for row in history)
        ) if history else None,
        "history_texts": [row.get("text", "") for row in history],
    }


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in text.split() if token.strip()]


def _tone_score(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    positive = sum(token.strip(".,!?;:()[]{}\"'") in POSITIVE_TERMS for token in tokens)
    negative = sum(token.strip(".,!?;:()[]{}\"'") in NEGATIVE_TERMS for token in tokens)
    return (positive - negative) / max(1, len(tokens))


def _tone_label(score: float) -> str:
    if score > 0.02:
        return "positive"
    if score < -0.02:
        return "negative"
    return "neutral"


def _bucket(stars: float) -> str:
    if stars >= 4.0:
        return "high"
    if stars <= 2.0:
        return "low"
    return "neutral"


def _pearson(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    denominator = math.sqrt(
        sum(value * value for value in centered_a)
        * sum(value * value for value in centered_b)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(centered_a, centered_b)) / denominator


def _style_features(text: str) -> dict:
    clean = (text or "").strip()
    words = _tokens(clean)
    sentences = [part for part in clean.replace("!", ".").replace("?", ".").split(".") if part.strip()]
    return {
        "length": len(clean),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "tone_score": _tone_score(clean),
        "exclamation_count": clean.count("!"),
        "question_count": clean.count("?"),
    }


def _metric_rows(rows: list[dict], prediction_key: str) -> dict:
    usable = [
        row for row in rows
        if row.get(prediction_key) is not None
        and row.get("actual_stars") is not None
        and row.get("status") in {"personalized", "cold_start"}
    ]
    if not usable:
        return {"n": 0, "mae": None, "rmse": None, "accuracy_±0.5": None,
                "accuracy_exact": None, "trend_accuracy": None,
                "pearson": None}
    errors = [float(row[prediction_key]) - float(row["actual_stars"]) for row in usable]
    abs_errors = [abs(value) for value in errors]
    actual = [float(row["actual_stars"]) for row in usable]
    predicted = [float(row[prediction_key]) for row in usable]
    pearson = _pearson(predicted, actual)
    return {
        "n": len(usable),
        "mae": sum(abs_errors) / len(abs_errors),
        "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "accuracy_±0.5": sum(value <= 0.5 for value in abs_errors) / len(abs_errors),
        "accuracy_exact": sum(value == 0 for value in abs_errors) / len(abs_errors),
        "trend_accuracy": sum(
            _bucket(actual_value) == _bucket(predicted_value)
            for actual_value, predicted_value in zip(actual, predicted)
        ) / len(usable),
        "pearson": pearson,
    }


def compute_sentiment(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("review") and row.get("status") in {"personalized", "cold_start"}:
            grouped[_bucket(float(row["actual_stars"]))].append(row)
    result = {}
    for name in ("high", "low", "neutral"):
        group = grouped.get(name, [])
        if not group:
            result[name] = {"n": 0}
            continue
        tone_matches = []
        bucket_matches = []
        tone_gaps = []
        length_ratios = []
        copy_ratios = []
        for row in group:
            generated = _style_features(row["review"])
            history_score = row.get("history_tone_score")
            if history_score is not None:
                tone_gaps.append(abs(generated["tone_score"] - float(history_score)))
                tone_matches.append(
                    _tone_label(generated["tone_score"]) == _tone_label(float(history_score))
                )
            expected = "positive" if name == "high" else "negative" if name == "low" else "neutral"
            bucket_matches.append(_tone_label(generated["tone_score"]) == expected)
            if row.get("history_mean_length"):
                length_ratios.append(
                    generated["length"] / max(1.0, float(row["history_mean_length"]))
                )
            copy_ratios.append(float(row.get("history_copy_ratio", 0.0)))
        result[name] = {
            "n": len(group),
            "generated_positive_rate": sum(
                _tone_label(_style_features(row["review"])["tone_score"]) == "positive"
                for row in group
            ) / len(group),
            "generated_negative_rate": sum(
                _tone_label(_style_features(row["review"])["tone_score"]) == "negative"
                for row in group
            ) / len(group),
            "tone_match_to_history_rate": (
                sum(tone_matches) / len(tone_matches) if tone_matches else None
            ),
            "tone_abs_gap_mean": (
                sum(tone_gaps) / len(tone_gaps) if tone_gaps else None
            ),
            "rating_bucket_tone_match_rate": sum(bucket_matches) / len(bucket_matches),
            "length_ratio_to_history_mean": (
                sum(length_ratios) / len(length_ratios) if length_ratios else None
            ),
            "history_copy_ratio_mean": sum(copy_ratios) / len(copy_ratios),
        }
    return result


def _history_bucket(history_count: int) -> str:
    if history_count == 0:
        return "0"
    if history_count <= 2:
        return "1-2"
    if history_count <= 5:
        return "3-5"
    return "6+"


def _grouped_rating_metrics(rows: list[dict], group_key) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(group_key(row))].append(row)
    return {
        name: _metric_rows(group, "predicted_stars")
        for name, group in sorted(groups.items())
    }


def _cost_summary(usage: dict) -> dict:
    """Calculate cost only when token usage and caller-supplied rates exist."""
    input_rate_text = os.environ.get("DEEPSEEK_INPUT_USD_PER_MILLION", "")
    output_rate_text = os.environ.get("DEEPSEEK_OUTPUT_USD_PER_MILLION", "")
    try:
        input_rate = float(input_rate_text)
        output_rate = float(output_rate_text)
        if input_rate < 0 or output_rate < 0:
            raise ValueError
    except (TypeError, ValueError):
        input_rate = output_rate = None

    if not usage.get("token_usage_available"):
        return {
            "cost_usd": None,
            "cost_status": "token_usage_unavailable",
        }
    if input_rate is None or output_rate is None:
        return {
            "cost_usd": None,
            "cost_status": "pricing_rates_not_configured",
        }
    return {
        "cost_usd": (
            usage["prompt_tokens"] * input_rate
            + usage["completion_tokens"] * output_rate
        ) / 1_000_000,
        "cost_status": "calculated_from_configured_rates",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _existing_run_matches(candidate: Path, source: Path, raw_audit: dict) -> bool:
    metadata_path = candidate / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_metadata = metadata.get("source", {})
        return (
            metadata.get("runner_version") == RUNNER_VERSION
            and source_metadata.get("sha256") == _sha256_file(source)
            and source_metadata.get("parsed_rows") == raw_audit["parsed_rows"]
        )
    except (OSError, TypeError, ValueError):
        return False


def _prepare_run(output_root: Path, source: Path, raw_audit: dict,
                 records: list[dict], rejected: list[dict], metadata: dict) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_root.glob("amazon_raw_full_temporal_v2_*"))
    for candidate in reversed(existing):
        if (
            (candidate / "plan.jsonl").is_file()
            and not (candidate / "summary.json").is_file()
            and _existing_run_matches(candidate, source, raw_audit)
        ):
            return candidate

    run_dir = output_root / (
        "amazon_raw_full_temporal_v2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir.mkdir(parents=False, exist_ok=False)
    plan = sorted(records, key=lambda row: (row["timestamp"], row["raw_index"]))
    with (run_dir / "plan.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for target_index, row in enumerate(plan):
            handle.write(json.dumps({
                "target_index": target_index,
                "raw_index": row["raw_index"],
                "user_id": row["user_id"],
                "item_id": row["item_id"],
                "target_date": row["timestamp"],
                "actual_stars": row["stars"],
                "target_text_digest": row.get("text_digest"),
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (run_dir / "status.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        **metadata,
        "runner_version": RUNNER_VERSION,
        "source": {
            "path_name": source.name,
            "sha256": _sha256_file(source),
            **raw_audit,
        },
        "protocol": {
            "target_order": "UTC calendar date then raw source row",
            "visible_history": "same-user rows strictly before target UTC calendar date",
            "item_references": "other-user same-item rows strictly before target UTC calendar date",
            "target_review_text": "withheld; only digest retained in private artifacts",
            "target_rating": "withheld from Agent A; used only after prediction for scoring",
            "cold_start": "valid row with zero eligible prior same-user history",
            "memory": "disabled to prevent same-day generated-output leakage",
            "experiments": "reflection-enabled Agent A, one sequential task stream",
        },
        "counts": {
            "raw_rows": raw_audit["parsed_rows"],
            "valid_rows": raw_audit["valid_rows"],
            "not_evaluated_rows": raw_audit["rejected_rows"],
            "planned_valid_targets": len(records),
        },
    }
    _write_json(run_dir / "metadata.json", metadata)
    return run_dir


def _render_report(summary: dict, output_path: Path) -> None:
    lines = [
        "# Amazon raw dataset full evaluation",
        "",
        f"- Runner: `{summary['runner_version']}`",
        f"- Source rows: **{summary['coverage']['raw_rows']}**",
        f"- Valid raw rows: **{summary['coverage']['valid_rows']}**",
        f"- Explicitly not evaluated: **{summary['coverage']['not_evaluated_rows']}**",
        f"- Completed model outcomes: **{summary['coverage']['completed_model_rows']}**",
        f"- Failed model rows: **{summary['coverage']['failed_model_rows']}**",
        f"- Wall time (seconds): **{summary['runtime']['elapsed_seconds']:.1f}**",
        f"- LLM calls: **{summary['runtime']['llm_calls']}**",
        f"- Cost (USD): **{summary['runtime']['cost_usd']}** ({summary['runtime']['cost_status']})",
        "",
        "## Coverage and outcome status",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for key, value in summary["coverage"]["status_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## Rating performance", "", "| cohort | n | MAE | RMSE | within ±0.5 | trend | Pearson |", "|---|---:|---:|---:|---:|---:|---:|"]
    for cohort in ("all_completed", "personalized", "cold_start"):
        metric = summary["rating"]["model"].get(cohort, {})
        lines.append(
            f"| {cohort} | {metric.get('n', 0)} | {metric.get('mae')} | "
            f"{metric.get('rmse')} | {metric.get('accuracy_±0.5')} | "
            f"{metric.get('trend_accuracy')} | {metric.get('pearson')} |"
        )
    lines += ["", "### Rating and history strata", "", "| stratum | n | MAE | RMSE | within ±0.5 | trend |", "|---|---:|---:|---:|---:|---:|"]
    for group_name, groups in (
        ("actual_rating", summary["rating"]["model"].get("actual_groups", {})),
        ("history_length", summary["rating"]["model"].get("history_groups", {})),
    ):
        for group, metric in groups.items():
            lines.append(
                f"| {group_name}:{group} | {metric.get('n', 0)} | {metric.get('mae')} | "
                f"{metric.get('rmse')} | {metric.get('accuracy_±0.5')} | "
                f"{metric.get('trend_accuracy')} |"
            )
    lines += ["", "## Baseline comparison (same completed rows)", "", "| baseline | n | MAE | RMSE | within ±0.5 |", "|---|---:|---:|---:|---:|"]
    for name, metric in summary["rating"]["baselines"].items():
        lines.append(
            f"| {name} | {metric.get('n', 0)} | {metric.get('mae')} | {metric.get('rmse')} | {metric.get('accuracy_±0.5')} |"
        )
    lines += ["", "## Comment tone/style proxy", "", "| actual-rating group | n | tone match to history | rating-tone match | length ratio | history-copy ratio |", "|---|---:|---:|---:|---:|---:|"]
    for group, metric in summary["sentiment"].items():
        lines.append(
            f"| {group} | {metric.get('n', 0)} | {metric.get('tone_match_to_history_rate')} | "
            f"{metric.get('rating_bucket_tone_match_rate')} | {metric.get('length_ratio_to_history_mean')} | "
            f"{metric.get('history_copy_ratio_mean')} |"
        )
    lines += ["", "## Historical 400-task comparison", "", "The historical values below are an explicitly labelled sample, not a full-dataset result.", "", "| run | n | MAE | RMSE | within ±0.5 | Pearson |", "|---|---:|---:|---:|---:|---:|"]
    historical = summary.get("historical_400")
    if historical:
        lines.append(
            f"| historical sample | {historical['n']} | {historical['mae']} | {historical['rmse']} | "
            f"{historical['accuracy_±0.5']} | {historical['pearson']} |"
        )
    lines += ["", "## Interpretation boundary", "", "This report measures predictive error and deterministic tone/style proxies. It does not prove that a generated review is written by a real person, and it does not treat the historical 400-task sample as full-dataset evidence.", ""]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_full_evaluation(raw_root: str | Path, output_root: str | Path,
                        api_key: str, *, chat_model: str,
                        base_url: str) -> dict:
    if not chat_model or not base_url:
        raise ValueError("full_evaluation_model_config_required")
    if chat_model.casefold().startswith("deepseek"):
        raise ValueError("deepseek_not_allowed_for_full_evaluation")
    # Keep source-only dry runs independent of the LLM/framework dependency
    # stack. These imports are needed only after the private run is prepared.
    from comprehensive_evaluation import DeepSeekLLM, extract_prediction
    from improved_agent_with_quality import (
        ImprovedSimulationAgent,
        reference_overlap_ratio,
    )

    source = locate_amazon_source(raw_root)
    records, rejected, raw_audit = load_amazon_records(source)
    item_metadata = load_item_metadata(raw_root)
    metadata = {
        "chat_model": chat_model,
        "base_url": base_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    run_dir = _prepare_run(Path(output_root), source, raw_audit, records, rejected, metadata)
    _write_json(run_dir / "item_metadata_manifest.json", {
        "item_count": len(item_metadata),
        "metadata_source": "meta_Musical_Instruments if present",
    })

    plan = _read_jsonl(run_dir / "plan.jsonl")
    existing = _read_jsonl(run_dir / "status.jsonl")
    existing_raw_indexes = [
        int(row["raw_index"]) for row in existing if "raw_index" in row
    ]
    if len(existing_raw_indexes) != len(set(existing_raw_indexes)):
        raise RuntimeError("duplicate_raw_status_rows")
    completed_targets = {
        int(row["target_index"])
        for row in existing
        if "target_index" in row
    }
    records_by_raw = {int(row["raw_index"]): row for row in records}
    ordered_records = [records_by_raw[int(row["raw_index"])] for row in plan]

    tool = TemporalAmazonInteractionTool(records, item_metadata)
    global_priors = _global_prior_by_date(records)
    llm = DeepSeekLLM(
        api_key=api_key,
        base_url=base_url,
        chat_model=chat_model,
        seed=42,
        json_mode=True,
    )
    agent = ImprovedSimulationAgent(
        llm=llm,
        enable_reflection=True,
        use_memory=False,
        max_reference_reviews=5,
        include_context=True,
    )
    started = time.time()
    processed_now = 0
    for plan_row, target in zip(plan, ordered_records):
        target_index = int(plan_row["target_index"])
        if target_index in completed_targets:
            continue
        baseline = target_baselines(target, tool, global_priors)
        tool.set_target(target)
        task = {
            "task_index": target_index,
            "user_id": target["user_id"],
            "item_id": target["item_id"],
        }
        agent.task = task
        agent.interaction_tool = tool
        call_start = llm.usage_stats()["calls"]
        row_started = time.time()
        try:
            output = agent.workflow()
            diagnostics = dict(agent.last_diagnostics or {})
            predicted = extract_prediction(output)
            quality_status = diagnostics.get("quality_status")
            status = "personalized" if baseline["history_count"] else "cold_start"
            personalization_mode = status
            status_reason = (
                "eligible_prior_user_history"
                if personalization_mode == "personalized"
                else "no_eligible_prior_user_history"
            )
            if quality_status != "success":
                predicted = None
                status = "failed"
                status_reason = diagnostics.get("reason") or "quality_validation_failed"
            if predicted is not None and not 1.0 <= float(predicted) <= 5.0:
                predicted = None
                status = "failed"
                status_reason = "prediction_out_of_range"
            if predicted is None and status != "failed":
                status = "failed"
                status_reason = "missing_prediction"
            review = str(output.get("review", "")) if predicted is not None else ""
            history_copy_ratio = reference_overlap_ratio(
                review, baseline["history_texts"], ngram_size=8
            ) if review else 0.0
            result_row = {
                "target_index": target_index,
                "raw_index": target["raw_index"],
                "user_id": target["user_id"],
                "item_id": target["item_id"],
                "target_date": target["timestamp"],
                "actual_stars": target["stars"],
                "status": status,
                "status_reason": status_reason,
                "personalization_mode": personalization_mode,
                "predicted_stars": predicted,
                "review": review,
                "quality_status": diagnostics.get("quality_status"),
                "quality_reason": diagnostics.get("reason"),
                "failure_stage": diagnostics.get("failure_stage"),
                "history_count": baseline["history_count"],
                "item_reference_count": baseline["item_reference_count"],
                "history_mean_length": baseline["history_mean_length"],
                "history_tone_score": baseline["history_tone_score"],
                "history_copy_ratio": history_copy_ratio,
                "baseline_user_mean": baseline["user_mean"],
                "baseline_item_mean": baseline["item_mean"],
                "baseline_global_mean": baseline["global_mean"],
                "baseline_user_mean_rounded": baseline["user_mean_rounded"],
                "baseline_item_mean_rounded": baseline["item_mean_rounded"],
                "baseline_global_mean_rounded": baseline["global_mean_rounded"],
                "llm_calls": llm.usage_stats()["calls"] - call_start,
                "elapsed_seconds": time.time() - row_started,
                "diagnostics": diagnostics,
            }
        except Exception:
            result_row = {
                "target_index": target_index,
                "raw_index": target["raw_index"],
                "user_id": target["user_id"],
                "item_id": target["item_id"],
                "target_date": target["timestamp"],
                "actual_stars": target["stars"],
                "status": "failed",
                "status_reason": "runner_exception",
                "personalization_mode": (
                    "personalized" if baseline["history_count"] else "cold_start"
                ),
                "predicted_stars": None,
                "review": "",
                "failure_reason": "runner_exception",
                "history_count": baseline["history_count"],
                "item_reference_count": baseline["item_reference_count"],
                "history_mean_length": baseline["history_mean_length"],
                "history_tone_score": baseline["history_tone_score"],
                "history_copy_ratio": 0.0,
                "baseline_user_mean": baseline["user_mean"],
                "baseline_item_mean": baseline["item_mean"],
                "baseline_global_mean": baseline["global_mean"],
                "baseline_user_mean_rounded": baseline["user_mean_rounded"],
                "baseline_item_mean_rounded": baseline["item_mean_rounded"],
                "baseline_global_mean_rounded": baseline["global_mean_rounded"],
                "llm_calls": llm.usage_stats()["calls"] - call_start,
                "elapsed_seconds": time.time() - row_started,
            }
        _append_jsonl(run_dir / "status.jsonl", result_row)
        processed_now += 1
        if processed_now == 1 or processed_now % PROGRESS_EVERY == 0:
            print(
                "PROGRESS",
                processed_now,
                "/",
                len(plan),
                "completed; llm_calls=",
                llm.usage_stats()["calls"],
                "elapsed_seconds=",
                round(time.time() - started, 1),
            )

    all_rows = _read_jsonl(run_dir / "status.jsonl")
    all_rows = sorted(all_rows, key=lambda row: int(row["raw_index"]))
    # The full run must have one and only one status for every raw source row.
    raw_indexes = [int(row["raw_index"]) for row in all_rows]
    expected_raw_indexes = set(range(raw_audit["parsed_rows"]))
    if (
        len(all_rows) != raw_audit["parsed_rows"]
        or len(raw_indexes) != len(set(raw_indexes))
        or set(raw_indexes) != expected_raw_indexes
    ):
        raise RuntimeError("full_raw_coverage_incomplete")
    for row in all_rows:
        status = row.get("status")
        if status not in {"personalized", "cold_start", "failed", "not_evaluated"}:
            raise RuntimeError("invalid_raw_status")
        if status in {"failed", "not_evaluated"} and not row.get("status_reason"):
            raise RuntimeError("status_reason_missing")
    completed_rows = [row for row in all_rows if row.get("status") in {"personalized", "cold_start"}]
    failed_rows = [row for row in all_rows if row.get("status") == "failed"]
    status_counts = Counter(row.get("status", "unknown") for row in all_rows)

    model_rows = completed_rows
    def baseline_rows(key: str) -> list[dict]:
        return [
            {**row, "baseline_prediction": row.get(key)}
            for row in model_rows if row.get(key) is not None
        ]

    model_all = _metric_rows(model_rows, "predicted_stars")
    model_personalized = _metric_rows(
        [row for row in model_rows if row.get("status") == "personalized"],
        "predicted_stars",
    )
    model_cold = _metric_rows(
        [row for row in model_rows if row.get("status") == "cold_start"],
        "predicted_stars",
    )
    baselines = {}
    for name, key in (
        ("user_mean", "baseline_user_mean"),
        ("item_mean", "baseline_item_mean"),
        ("global_mean", "baseline_global_mean"),
        ("user_mean_rounded", "baseline_user_mean_rounded"),
        ("item_mean_rounded", "baseline_item_mean_rounded"),
        ("global_mean_rounded", "baseline_global_mean_rounded"),
    ):
        rows_for_baseline = baseline_rows(key)
        baselines[name] = _metric_rows(rows_for_baseline, "baseline_prediction")

    usage = llm.usage_stats()
    summary = {
        "runner_version": RUNNER_VERSION,
        "run_dir": str(run_dir),
        "coverage": {
            "raw_rows": raw_audit["parsed_rows"],
            "valid_rows": raw_audit["valid_rows"],
            "not_evaluated_rows": raw_audit["rejected_rows"],
            "completed_model_rows": len(completed_rows),
            "failed_model_rows": len(failed_rows),
            "personalized_rows": status_counts.get("personalized", 0),
            "cold_start_rows": status_counts.get("cold_start", 0),
            "pending_model_rows": 0,
            "status_counts": dict(sorted(status_counts.items())),
            "coverage_over_raw": len(completed_rows) / raw_audit["parsed_rows"],
            "coverage_over_valid": len(completed_rows) / raw_audit["valid_rows"],
            "failure_rate_over_valid": (
                status_counts.get("failed", 0) / raw_audit["valid_rows"]
            ),
            "coverage_complete": True,
        },
        "rating": {
            "model": {
                "all_completed": model_all,
                "personalized": model_personalized,
                "cold_start": model_cold,
                "actual_groups": _grouped_rating_metrics(
                    model_rows, lambda row: _bucket(float(row["actual_stars"]))
                ),
                "history_groups": _grouped_rating_metrics(
                    model_rows, lambda row: _history_bucket(int(row["history_count"]))
                ),
            },
            "baselines": baselines,
        },
        "sentiment": compute_sentiment(model_rows),
        "runtime": {
            "elapsed_seconds": time.time() - started,
            **usage,
            "llm_calls": usage.get("calls", 0),
            "llm_errors": usage.get("errors", 0),
            **_cost_summary(usage),
        },
        "historical_400": {
            "n": 400,
            "mae": 0.58625,
            "rmse": 0.8605957239029253,
            "accuracy_±0.5": 0.485,
            "pearson": 0.5678260043292725,
            "label": "historical sample only",
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _render_report(summary, run_dir / "report.md")
    print("SUMMARY_PATH", run_dir / "summary.json")
    print(json.dumps(summary["coverage"], ensure_ascii=False, sort_keys=True))
    print(json.dumps(summary["rating"]["model"], ensure_ascii=False, sort_keys=True))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run every valid Amazon raw review through Agent A.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--chat-model",
        default=None,
        help="Explicit approved current model name; required for a model run.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Explicit private model endpoint; required for a model run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source parsing and print counts without calling the model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dry_run:
        source = locate_amazon_source(args.raw_root)
        records, _rejected, audit = load_amazon_records(source)
        print(json.dumps({
            "runner_version": RUNNER_VERSION,
            "source_name": source.name,
            "parsed_rows": audit["parsed_rows"],
            "valid_rows": audit["valid_rows"],
            "rejected_rows": audit["rejected_rows"],
            "rejected_by_reason": audit["rejected_by_reason"],
            "target_rows": len(records),
            "first_target_date": min(row["timestamp"] for row in records),
            "last_target_date": max(row["timestamp"] for row in records),
        }, ensure_ascii=False, sort_keys=True))
        return 0
    if not args.output_root:
        print("ERROR: --output-root is required unless --dry-run is used")
        return 2
    if not args.chat_model or not args.base_url:
        print("ERROR: --chat-model and --base-url are required for a model run")
        return 2
    if args.chat_model.casefold().startswith("deepseek"):
        print("ERROR: DeepSeek is not approved for the full evaluation")
        return 2
    api_key = args.api_key or os.environ.get("LLM_API_KEY", "")
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: LLM_API_KEY or OPENAI_API_KEY is missing")
        return 2
    run_full_evaluation(
        args.raw_root,
        args.output_root,
        api_key,
        chat_model=args.chat_model,
        base_url=args.base_url,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
