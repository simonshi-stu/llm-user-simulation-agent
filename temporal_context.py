"""Framework-independent temporal context provider and offline ablation runner.

This module deliberately does not integrate with ``websocietysimulator``. It
rebuilds a manifest from the current local source files, checks the caller's
timestamp-semantics assertion, reconstructs the exact review rows, and exposes
only strictly eligible user history/item references to a single-task adapter.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from local_memory import LocalMemoryStore
from memory_audit import (
    DATE_FIELDS,
    STAR_FIELDS,
    TEXT_FIELDS,
    _as_id,
    _extract_task,
    _field,
    _load_task_pairs,
    _parse_time,
    _build_temporal_manifest_with_state,
    _files,
    load_selected_temporal_review_records,
    _review_text,
    _text_digest,
)


class TemporalManifestError(ValueError):
    """The provided manifest or current source files cannot be trusted."""


class TemporalAblationMode(str, Enum):
    NO_MEMORY = "No_Memory"
    GENERATED_REVIEW_MEMORY = "Generated_Review_Memory"
    TRUSTED_HISTORY_STATS = "Trusted_History_Stats"
    COMBINED = "Combined"


@dataclass(frozen=True)
class TemporalAblationContract:
    mode: TemporalAblationMode
    generated_review_memory: bool
    trusted_history_stats: bool


TEMPORAL_ABLATION_CONTRACTS = {
    TemporalAblationMode.NO_MEMORY: TemporalAblationContract(
        TemporalAblationMode.NO_MEMORY, False, False
    ),
    TemporalAblationMode.GENERATED_REVIEW_MEMORY: TemporalAblationContract(
        TemporalAblationMode.GENERATED_REVIEW_MEMORY, True, False
    ),
    TemporalAblationMode.TRUSTED_HISTORY_STATS: TemporalAblationContract(
        TemporalAblationMode.TRUSTED_HISTORY_STATS, False, True
    ),
    TemporalAblationMode.COMBINED: TemporalAblationContract(
        TemporalAblationMode.COMBINED, True, True
    ),
}


def _as_mode(value: TemporalAblationMode | str) -> TemporalAblationMode:
    if isinstance(value, TemporalAblationMode):
        return value
    try:
        return TemporalAblationMode(value)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in TemporalAblationMode)
        raise ValueError(f"unknown temporal ablation mode; choose {choices}") from exc


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _numeric_star(row: dict) -> float | None:
    value = _field(row, STAR_FIELDS)
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 1.0 <= number <= 5.0:
        return None
    return number


def summarize_trusted_history(history: Iterable[dict]) -> dict[str, Any]:
    """Compute prompt-safe stats from the already time-filtered real history."""
    rows = list(history)
    ratings = [star for row in rows if (star := _numeric_star(row)) is not None]
    counts = Counter(
        str(int(star)) if star.is_integer() else format(star, "g")
        for star in ratings
    )
    event_times = [
        parsed for row in rows
        if (parsed := _parse_time(
            row.get("_temporal_event_time") or _field(row, DATE_FIELDS)
        )) is not None
    ]
    row_indexes = [
        int(row["_temporal_source_row_index"])
        for row in rows if row.get("_temporal_source_row_index") is not None
    ]
    text_digests = [
        _text_digest(_review_text(_field(row, TEXT_FIELDS))) or ""
        for row in rows
    ]
    text_lengths = [
        len(_review_text(_field(row, TEXT_FIELDS)) or "") for row in rows
    ]
    provenance = {
        "source_row_indexes": row_indexes,
        "source_text_digests": text_digests,
        "source_ratings": [
            _numeric_star(row) for row in rows
        ],
        "source_text_lengths": text_lengths,
        "source_event_times": sorted(value.isoformat() for value in event_times),
    }
    return {
        "source_review_count": len(rows),
        "rated_source_review_count": len(ratings),
        "mean_stars": sum(ratings) / len(ratings) if ratings else None,
        "star_distribution": dict(sorted(counts.items())),
        "mean_review_length": (
            sum(text_lengths) / len(rows) if rows else None
        ),
        "source_event_time_min": (
            min(event_times).isoformat() if event_times else None
        ),
        "source_event_time_max": (
            max(event_times).isoformat() if event_times else None
        ),
        "source_row_indexes": row_indexes,
        "source_sha256": _sha256_json(provenance),
    }


@dataclass(frozen=True)
class TemporalTaskContext:
    task_index: int
    source_task_index: str | None
    execution_order: int
    split: str
    user_group: str
    user_repeat_stratum: str
    target_timestamp: str
    context_sha256: str
    manifest_context_sha256: str
    history: tuple[dict, ...]
    item_references: tuple[dict, ...]
    trusted_history_stats: dict[str, Any]
    _user_id: str = field(repr=False)
    _item_id: str = field(repr=False)

    @property
    def task_payload(self) -> dict[str, Any]:
        """The Agent-visible task intentionally contains no target label/time."""
        return {
            "user_id": self._user_id,
            "item_id": self._item_id,
            "task_index": self.task_index,
        }


class TemporalContextProvider:
    """Reconstruct strict runtime contexts from local source data.

    A schema-2 manifest with an explicit caller assertion is mandatory. The
    manifest is then regenerated against the current sources and compared in
    full. Context rows are loaded from those sources, rechecked, and returned;
    a manifest of row numbers/hashes alone is never used as model context.
    """

    def __init__(
        self,
        task_dir: str | Path,
        groundtruth_dir: str | Path,
        review_data: Iterable[str | Path],
        manifest: dict | str | Path,
    ):
        self.task_dir = Path(task_dir)
        self.groundtruth_dir = Path(groundtruth_dir)
        if isinstance(review_data, (str, Path)):
            review_data = (review_data,)
        self.review_data = tuple(Path(path) for path in review_data)
        self.manifest = self._read_manifest(manifest)
        self._validate_manifest_header()

        if (
            not self.task_dir.is_dir() or not _files(self.task_dir)
            or not self.groundtruth_dir.is_dir() or not _files(self.groundtruth_dir)
        ):
            raise TemporalManifestError(
                "task and groundtruth sources must both contain task files"
            )
        if not self.review_data or any(
            not path.exists() or not _files(path) for path in self.review_data
        ):
            raise TemporalManifestError(
                "review source is missing or contains no supported files"
            )
        _, pairing = _load_task_pairs(self.task_dir, self.groundtruth_dir)
        if pairing.get("warnings") or pairing.get("alignment") != "natural_filename_order":
            raise TemporalManifestError(
                "task/groundtruth source pairing is not reliable"
            )

        policy = self.manifest.get("split_policy") or {}
        try:
            train_ratio = float(policy["train_ratio"])
            validation_ratio = float(policy["validation_ratio"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TemporalManifestError("manifest has no valid split policy") from exc

        try:
            expected, source_state = _build_temporal_manifest_with_state(
                self.task_dir,
                self.groundtruth_dir,
                self.review_data,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                timestamp_semantics_caller_asserted=True,
            )
        except ValueError as exc:
            raise TemporalManifestError(
                "current temporal source could not be safely loaded"
            ) from exc
        if expected.get("pairing", {}).get("warnings"):
            raise TemporalManifestError("source task pairing has warnings")
        if self.manifest != expected:
            raise TemporalManifestError(
                "manifest, source fingerprints, task order, or context rows "
                "do not match current sources"
            )

        if expected.get("task_count", 0) <= 0:
            raise TemporalManifestError("no paired tasks are available for execution")
        if expected.get("source_review_rows", 0) <= 0:
            raise TemporalManifestError("review audit source contains no records")
        if expected.get("eligible_task_count", 0) <= 0:
            raise TemporalManifestError(
                "no eligible temporal task contexts; manifest is audit-only"
            )

        self._contexts = self._reconstruct_contexts(expected, source_state)
        self._contexts_by_index = {context.task_index: context for context in self._contexts}
        self.manifest_sha256 = _sha256_json({
            "schema_version": self.manifest["schema_version"],
            "tasks": self.manifest["tasks"],
            "split_policy": self.manifest["split_policy"],
        })

    @staticmethod
    def _read_manifest(manifest: dict | str | Path) -> dict:
        if isinstance(manifest, dict):
            return deepcopy(manifest)
        try:
            with Path(manifest).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise TemporalManifestError("cannot read temporal manifest") from exc
        if not isinstance(payload, dict):
            raise TemporalManifestError("temporal manifest must be a JSON object")
        return payload

    def _validate_manifest_header(self) -> None:
        if self.manifest.get("schema_version") != 2:
            raise TemporalManifestError("only temporal manifest schema 2 is supported")
        if self.manifest.get("manifest_type") != "strict_temporal_no_groundtruth_context":
            raise TemporalManifestError("manifest type is not a strict temporal manifest")
        if self.manifest.get("target_timestamp_semantics_verified") is not False:
            raise TemporalManifestError(
                "manifest must not claim that code verified event-time semantics"
            )
        if self.manifest.get("target_timestamp_semantics_caller_asserted") is not True:
            raise TemporalManifestError(
                "target event-time semantics lack the required caller assertion"
            )
        if self.manifest.get("timestamp_semantics_confirmation") != "caller_asserted":
            raise TemporalManifestError(
                "manifest lacks the explicit timestamp-semantics assertion"
            )
        if self.manifest.get("privacy") != (
            "no raw IDs, groundtruth ratings, review text, or prompts included"
        ):
            raise TemporalManifestError("manifest privacy contract is missing")

    def _reconstruct_contexts(self, expected: dict, source_state: dict):
        wanted_indexes = {
            int(row_index)
            for task in expected["tasks"]
            for row_index in (
                *task["visible_history_row_indexes"],
                *task["target_item_reference_row_indexes"],
            )
        }
        try:
            source_rows, row_count, file_count, source_bytes, source_sha256 = (
                load_selected_temporal_review_records(
                    self.review_data, wanted_indexes
                )
            )
        except ValueError as exc:
            raise TemporalManifestError(
                "review source changed or exceeded offline input limits"
            ) from exc
        if (
            row_count != expected["source_review_rows"]
            or file_count != expected["source_review_files"]
            or source_bytes != expected["source_review_bytes"]
            or source_sha256 != expected["source_review_sha256"]
        ):
            raise TemporalManifestError(
                "review source fingerprint changed during context reconstruction"
            )
        review_index = source_state["review_index"]
        metadata_rows = {
            row["row_index"]: row for row in review_index["rows"]
        }
        identities = source_state["task_rows"]
        heldout_keys = source_state["heldout_keys"]
        heldout_review_ids = source_state["heldout_review_ids"]
        heldout_texts = source_state["heldout_texts"]
        heldout_user_times = source_state["heldout_user_times"]
        heldout_item_times = source_state["heldout_item_times"]
        pairs = source_state["pairs"]

        contexts = []
        for manifest_row in expected["tasks"]:
            task_index = int(manifest_row["task_index"])
            if task_index < 0 or task_index >= len(pairs):
                raise TemporalManifestError("manifest task index is out of range")
            identity = identities[task_index]
            target_time = identity.get("timestamp")
            user_id = identity.get("user_id")
            item_id = identity.get("item_id")
            if not user_id or not item_id or target_time is None:
                raise TemporalManifestError("eligible manifest task lost its identity/time")

            expected_history_indexes = manifest_row["visible_history_row_indexes"]
            expected_reference_indexes = manifest_row["target_item_reference_row_indexes"]
            history = self._load_context_rows(
                expected_history_indexes,
                source_rows,
                metadata_rows,
                user_id=user_id,
                item_id=None,
                target_time=target_time,
                heldout_keys=heldout_keys,
                heldout_review_ids=heldout_review_ids,
                heldout_texts=heldout_texts,
                heldout_user_times=heldout_user_times,
                heldout_item_times=heldout_item_times,
                target_user_id=user_id,
                context_kind="history",
            )
            references = self._load_context_rows(
                expected_reference_indexes,
                source_rows,
                metadata_rows,
                user_id=None,
                item_id=item_id,
                target_time=target_time,
                heldout_keys=heldout_keys,
                heldout_review_ids=heldout_review_ids,
                heldout_texts=heldout_texts,
                heldout_user_times=heldout_user_times,
                heldout_item_times=heldout_item_times,
                target_user_id=user_id,
                context_kind="item_reference",
            )

            user_row = pairs[task_index][0]
            truth_row = pairs[task_index][1]
            source_task = _extract_task(user_row, task_index)
            source_truth = _extract_task(truth_row, task_index)
            resolved_user = _as_id(source_task.get("user_id") or source_truth.get("user_id"))
            resolved_item = _as_id(source_task.get("item_id") or source_truth.get("item_id"))
            if resolved_user != user_id or resolved_item != item_id:
                raise TemporalManifestError("task identity changed during context reconstruction")

            runtime_context_digest = _sha256_json({
                "target_timestamp": target_time.isoformat(),
                "manifest_context_sha256": manifest_row["context_sha256"],
                "history": [self._row_fingerprint(row) for row in history],
                "item_references": [self._row_fingerprint(row) for row in references],
            })
            stats = summarize_trusted_history(history)
            contexts.append(TemporalTaskContext(
                task_index=task_index,
                source_task_index=identity.get("source_task_index"),
                execution_order=int(manifest_row["execution_order"]),
                split=manifest_row["split"],
                user_group=manifest_row["user_group"],
                user_repeat_stratum=manifest_row["user_repeat_stratum"],
                target_timestamp=target_time.isoformat(),
                context_sha256=runtime_context_digest,
                manifest_context_sha256=manifest_row["context_sha256"],
                history=tuple(history),
                item_references=tuple(references),
                trusted_history_stats=stats,
                _user_id=user_id,
                _item_id=item_id,
            ))

        contexts.sort(key=lambda context: context.execution_order)
        if [context.execution_order for context in contexts] != list(range(len(contexts))):
            raise TemporalManifestError("manifest execution order is not contiguous")
        return tuple(contexts)

    @staticmethod
    def _row_fingerprint(row: dict) -> dict[str, Any]:
        timestamp = _parse_time(
            row.get("_temporal_event_time") or _field(row, DATE_FIELDS)
        )
        return {
            "source_row_index": row.get("_temporal_source_row_index"),
            "event_time": timestamp.isoformat() if timestamp else None,
            "text_sha256": _text_digest(_review_text(_field(row, TEXT_FIELDS))),
            "stars": _field(row, STAR_FIELDS),
            "useful": row.get("useful"),
            "funny": row.get("funny"),
            "cool": row.get("cool"),
        }

    @staticmethod
    def _load_context_rows(
        indexes: list[int],
        source_rows: dict[int, dict],
        metadata_rows: dict[int, dict],
        *,
        user_id: str | None,
        item_id: str | None,
        target_time: datetime,
        heldout_keys: set[str],
        heldout_review_ids: set[str],
        heldout_texts: set[str],
        heldout_user_times: set[tuple[str, datetime]],
        heldout_item_times: set[tuple[str, datetime]],
        target_user_id: str,
        context_kind: str,
    ) -> list[dict]:
        if len(indexes) != len(set(indexes)):
            raise TemporalManifestError("context contains duplicate source indexes")
        output = []
        for row_index in indexes:
            if row_index not in source_rows or row_index not in metadata_rows:
                raise TemporalManifestError("context source row index is unavailable")
            raw = source_rows[row_index]
            meta = metadata_rows[row_index]
            row_time = meta["timestamp"]
            if row_time is None or row_time >= target_time:
                raise TemporalManifestError("context includes same-time/future row")
            if meta["row_key"] in heldout_keys:
                raise TemporalManifestError("context includes a known target source row")
            if meta["raw_id"] in heldout_review_ids:
                raise TemporalManifestError("context includes a known target review ID")
            if meta["text_digest"] in heldout_texts:
                raise TemporalManifestError("context includes a known target text")
            if context_kind == "history" and (
                meta["user_id"], row_time
            ) in heldout_user_times:
                raise TemporalManifestError(
                    "history includes a held-out target user/time"
                )
            if context_kind == "item_reference" and (
                meta["item_id"], row_time
            ) in heldout_item_times:
                raise TemporalManifestError("context includes a held-out target event time")
            if context_kind == "history" and meta["user_id"] != user_id:
                raise TemporalManifestError("history row belongs to another user")
            if context_kind == "item_reference" and (
                meta["item_id"] != item_id
                or meta["user_id"] is None
                or meta["user_id"] == target_user_id
            ):
                raise TemporalManifestError("item reference is not an external user's item review")

            text = _review_text(_field(raw, TEXT_FIELDS))
            item = {"text": text or ""}
            star_value = _field(raw, STAR_FIELDS)
            if star_value is None and context_kind == "history":
                raise TemporalManifestError(
                    "visible user-history row lacks a rating required by the Agent"
                )
            if star_value is not None:
                try:
                    parsed_stars = float(star_value)
                except (TypeError, ValueError):
                    raise TemporalManifestError(
                        "visible source review has a nonnumeric star value"
                    )
                if not math.isfinite(parsed_stars):
                    raise TemporalManifestError(
                        "visible source review has a nonfinite star value"
                    )
                if not 1.0 <= parsed_stars <= 5.0:
                    raise TemporalManifestError(
                        "visible source review rating is outside the 1-5 range"
                    )
                item["stars"] = parsed_stars
            for field_name in ("useful", "funny", "cool"):
                value = _field(raw, (field_name,))
                try:
                    parsed_value = float(value or 0)
                except (TypeError, ValueError):
                    parsed_value = 0.0
                item[field_name] = (
                    parsed_value
                    if math.isfinite(parsed_value) and parsed_value >= 0
                    else 0.0
                )
            item["_temporal_source_row_index"] = row_index
            item["_temporal_event_time"] = row_time.isoformat()
            output.append(item)
        return output

    def contexts_in_execution_order(self) -> tuple[TemporalTaskContext, ...]:
        return self._contexts

    def context_for_task(self, task_index: int) -> TemporalTaskContext:
        try:
            return self._contexts_by_index[int(task_index)]
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"task index {task_index} is not eligible") from exc

    @property
    def split_by_task_index(self) -> dict[int, str]:
        return {context.task_index: context.split for context in self._contexts}


class TemporalInteractionTool:
    """A minimal tool adapter exposing only a provider-approved task context."""

    def __init__(self, context: TemporalTaskContext):
        self.context = context

    def get_user(self, user_id: str) -> dict[str, str]:
        if str(user_id) != self.context._user_id:
            raise ValueError("requested user is outside the current task context")
        # No precomputed review-count/average fields can bypass the provider.
        return {"profile": "user"}

    def get_item(self, item_id: str) -> dict[str, str]:
        if str(item_id) != self.context._item_id:
            raise ValueError("requested item is outside the current task context")
        return {"name": "target item", "categories": ""}

    def get_reviews(
        self, user_id: str | None = None, item_id: str | None = None
    ) -> list[dict]:
        if user_id is not None:
            if str(user_id) != self.context._user_id:
                raise ValueError("requested user reviews are outside the current context")
            return deepcopy(list(self.context.history))
        if item_id is not None:
            if str(item_id) != self.context._item_id:
                raise ValueError("requested item reviews are outside the current context")
            return deepcopy(list(self.context.item_references))
        raise ValueError("review query must specify the current user or item")


def run_temporal_ablation(
    provider: TemporalContextProvider,
    mode: TemporalAblationMode | str,
    llm_factory: Callable[[int, str], Any],
    *,
    enable_reflection: bool = False,
) -> list[dict[str, Any]]:
    """Execute one explicit condition serially using provider-built contexts.

    There is no default LLM factory and no Simulator import here. The factory
    receives only a task index and mode (never raw context); only the Agent's
    intended prompt reaches its returned callable. This loop—not a worker-count
    setting—enforces the declared order. Use a fake unless an online run is
    separately authorized.
    """
    selected_mode = _as_mode(mode)
    contract = TEMPORAL_ABLATION_CONTRACTS[selected_mode]
    generated_entries_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    from improved_agent_with_quality import ImprovedSimulationAgent

    records = []
    for context in provider.contexts_in_execution_order():
        llm = llm_factory(context.task_index, selected_mode.value)
        task_memory_store = None
        if contract.generated_review_memory:
            task_memory_store = LocalMemoryStore()
            target_time = datetime.fromisoformat(context.target_timestamp)
            prior_entries = [
                entry for entry in generated_entries_by_user[context._user_id]
                if entry["event_time"] < target_time
            ]
            for entry in sorted(
                prior_entries,
                key=lambda item: (item["event_time"], item["execution_order"]),
            ):
                task_memory_store.remember(
                    context._user_id,
                    entry["text"],
                    stars=entry["stars"],
                    item_id=entry["item_id"],
                    origin_task_index=entry["task_index"],
                    origin_execution_order=entry["execution_order"],
                )
        agent = ImprovedSimulationAgent(
            llm=llm,
            enable_reflection=enable_reflection,
            use_memory=contract.generated_review_memory,
            max_reference_reviews=5,
            memory_store=task_memory_store,
            enable_framework_memory=False,
            trusted_history_stats=(
                context.trusted_history_stats
                if contract.trusted_history_stats else None
            ),
        )
        agent.task = context.task_payload
        agent.interaction_tool = TemporalInteractionTool(context)
        agent._audit_task_index = context.task_index
        agent._audit_execution_order = context.execution_order
        prediction = agent.workflow()
        agent_diagnostics = dict(agent.last_diagnostics)
        agent_diagnostics["user_history_profile_source_row_indexes"] = [
            row["_temporal_source_row_index"] for row in context.history
        ]
        if (
            contract.generated_review_memory
            and isinstance(prediction, dict)
            and str(prediction.get("review", "")).strip()
        ):
            generated_entries_by_user[context._user_id].append({
                "event_time": datetime.fromisoformat(context.target_timestamp),
                "execution_order": context.execution_order,
                "task_index": context.task_index,
                "text": str(prediction["review"]),
                "stars": prediction.get("stars"),
                "item_id": context._item_id,
            })
        stats = context.trusted_history_stats
        records.append({
            "task_index": context.task_index,
            "source_task_index": context.source_task_index,
            "execution_order": context.execution_order,
            "split": context.split,
            "user_repeat_stratum": context.user_repeat_stratum,
            "mode": selected_mode.value,
            "prediction": prediction,
            "context_sha256": context.context_sha256,
            "manifest_context_sha256": context.manifest_context_sha256,
            "visible_history_source_row_indexes": [
                row["_temporal_source_row_index"] for row in context.history
            ],
            "visible_item_reference_source_row_indexes": [
                row["_temporal_source_row_index"]
                for row in context.item_references
            ],
            "history_source_review_count": stats["source_review_count"],
            "history_source_event_time_max": stats["source_event_time_max"],
            "history_source_sha256": stats["source_sha256"],
            "trusted_history_stats_included": contract.trusted_history_stats,
            "trusted_history_stats_source_review_count": (
                stats["source_review_count"] if contract.trusted_history_stats else 0
            ),
            "trusted_history_stats_source_event_time_max": (
                stats["source_event_time_max"]
                if contract.trusted_history_stats else None
            ),
            "trusted_history_stats_source_row_indexes": (
                stats["source_row_indexes"]
                if contract.trusted_history_stats else []
            ),
            "agent_diagnostics": agent_diagnostics,
        })
    return records


def run_temporal_ablation_suite(
    provider: TemporalContextProvider,
    llm_factory: Callable[[int, str], Any],
    modes: Iterable[TemporalAblationMode | str] = tuple(TemporalAblationMode),
) -> dict[str, list[dict[str, Any]]]:
    """Run conditions on the same manifest order/context, raising on drift."""
    suite = {}
    baseline_signature = None
    for value in modes:
        mode = _as_mode(value)
        records = run_temporal_ablation(provider, mode, llm_factory)
        signature = [
            (
                row["task_index"], row["execution_order"],
                row["context_sha256"], row["manifest_context_sha256"],
            )
            for row in records
        ]
        if baseline_signature is None:
            baseline_signature = signature
        elif signature != baseline_signature:
            raise TemporalManifestError(
                "ablation conditions did not share identical task/context order"
            )
        suite[mode.value] = records
    return suite
