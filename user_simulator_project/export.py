"""Stage-1 export: P1 outcomes -> strict ``SimulationRunV1`` artifact.

This module is the user-simulation side of the shared result contract. It runs
the accepted P1 reliable workflow offline with a deterministic fake callable
(no network client, no API key, no environment reads) and writes one versioned
artifact into a new, gitignored ``results/`` or ``Dataset/`` directory:

- ``synthetic``: the accepted synthetic provider fixture (the same fixture the
  P1 smoke scenarios use) with source kind ``fake_on_synthetic``;
- ``prepared``: a local, already-verified P0 prepared directory loaded through
  :func:`reliable_agent.load_verified_prepared_dataset`, which reuses every P0
  gate (artifact size/row gates before parsing, policy validation, selection
  reconstruction, prior re-verification, strict provider revalidation and
  split-count checks) and keeps the frozen-history, UTC-day memory and
  target-copy exclusion rules untouched. Source kind ``fake_on_real_context``.

Honest notes:

- P1 prediction functions never receive groundtruth ratings, and groundtruth
  rows are parsed by the shared P0/P1 machinery only for identity/time/text
  verification and target exclusion; they are never fed into predictions, the
  statistical prior or any statistic. No scoring happens in this stage; any
  future scoring belongs to the evaluation project's scorer.
- Both modes use a deterministic fake callable. The exported artifact proves
  shape, gates and status flow only; it cannot measure real-LLM quality,
  accuracy or memory effects, and no accuracy metric is computed here.
- The shared contract has a closed field set without groundtruth/raw
  prompt/key/path fields, but ``review`` is generated free text that can
  contain arbitrary words: no content-level sanitization or complete
  anonymization is claimed. ``user_group`` is a deterministic pseudonym kept
  for pairing, not an irreversibility guarantee. Review text is written only
  into the local gitignored artifact.

Output rules: the output directory must not exist, it must live under the
repository's ignored ``results/`` or ``Dataset/`` trees anchored to the
module's repository root (never guessed from the process working directory),
and a failed run removes only this run's own known artifacts and the directory
when it is empty; unknown or late-created files are preserved. The artifact is
published by writing a ``.part`` file, fully revalidating exactly those bytes,
and then a hard-link publish (atomic, no-overwrite) under the final name; when
hard links are unavailable the publish fails closed with
``atomic_publish_unavailable`` rather than ever writing the final file
non-atomically, and a crash cannot leave a half-written file under the final
name.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from p1_smoke import ScriptedFakeLLM, build_synthetic_provider, grounded_response
import amazon_p0
from reliable_agent import (
    MODEL_SOURCE_FAKE,
    BudgetLedger,
    BudgetSpec,
    ReliableMemoryStore,
    ReliableTaskRecord,
    RenderConfig,
    load_verified_prepared_dataset,
    run_reliable_suite,
)
from temporal_context import TemporalManifestError, TemporalTaskContext

from . import simulation_run_v1 as contract

DEFAULT_MAX_TASKS = 3
MAX_TASKS_CAP = 12
FIXED_SEED = 42
ARTIFACT_FILENAME = "simulation_run.v1.json"
EXPORT_TOOL = "user_simulator_project.stage1"
SYNTHETIC_GLOBAL_PRIOR = 3.5

DEFAULT_RENDER_CONFIG = RenderConfig()

# Production output root is anchored to this module's repository, never to the
# process working directory: only the repository's gitignored ``results/`` and
# ``Dataset/`` trees are accepted.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAFE_OUTPUT_DIRS = ("results", "Dataset")


class ExportError(ValueError):
    """Safe, categorized export failure (stable code only)."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ExportResult:
    """Safe summary of one export; never contains review text or paths."""

    mode: str
    source_kind: str
    record_count: int
    status_counts: Mapping[str, int]
    prediction_success_count: int
    artifact_sha256: str
    schema_version: str
    schema_hash: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def anonymize_user_group(raw_group: str) -> str:
    """Deterministic anonymous group label; equal raw groups stay equal."""
    if not isinstance(raw_group, str) or not raw_group:
        raise ExportError("invalid_user_group")
    digest = hashlib.sha256(
        b"simulation_run.v1-usergroup\0" + raw_group.encode("utf-8")
    ).hexdigest()
    return "ug_" + digest[:16]


def _validate_params(max_tasks: Any, seed: Any) -> int:
    if (
        isinstance(max_tasks, bool)
        or not isinstance(max_tasks, int)
        or not 1 <= max_tasks <= MAX_TASKS_CAP
    ):
        raise ExportError("invalid_max_tasks")
    if seed != FIXED_SEED:
        raise ExportError("unsupported_seed")
    return max_tasks


def _ignored_output_root(output_dir: Path) -> bool:
    """Only gitignored ``results/`` or ``Dataset/`` trees inside the module's
    own repository are acceptable; the process working directory is ignored.
    """
    resolved = output_dir.resolve()
    for name in _SAFE_OUTPUT_DIRS:
        candidate = (_REPO_ROOT / name).resolve()
        try:
            resolved.relative_to(candidate)
            return True
        except ValueError:
            continue
    return False


def _repair_state(llm_diagnostics: Mapping[str, Any]) -> str:
    if llm_diagnostics.get("repair_attempted"):
        return "attempted"
    if llm_diagnostics.get("repair_reserve_failure"):
        return "reserve_failed"
    if llm_diagnostics.get("repair_requested"):
        return "requested"
    return "not_requested"


def _attempt_count(llm_diagnostics: Mapping[str, Any]) -> int:
    return int(bool(llm_diagnostics.get("draft_attempted"))) + int(
        bool(llm_diagnostics.get("repair_attempted"))
    )


def _bounded_memory_count(value: Any) -> int:
    if type(value) is int and 0 <= value <= contract.MAX_MEMORY_COUNT:
        return value
    raise ExportError("invalid_memory_diagnostic")


def _write_state(value: Any) -> str:
    if isinstance(value, str) and value in contract.WRITE_STATES:
        return value
    raise ExportError("invalid_write_state")


def document_record(
    record: ReliableTaskRecord, context: TemporalTaskContext
) -> dict[str, Any]:
    """Adapt one P1 ``ReliableTaskRecord`` to a contract record.

    Mapping only: no P0 output fields are changed, and bounded diagnostics are
    selected explicitly (attempts/repair/prompt bytes/memory counts). The raw
    user group is anonymized; task-scoped ``ev_`` refs pass through.
    """
    result = record.result
    diagnostics = result.diagnostics if isinstance(result.diagnostics, Mapping) else {}
    llm = diagnostics.get("llm")
    llm = llm if isinstance(llm, Mapping) else {}
    rendered = diagnostics.get("rendered")
    rendered = rendered if isinstance(rendered, Mapping) else {}
    memory = record.memory if isinstance(record.memory, Mapping) else {}

    prompt_bytes = rendered.get("prompt_bytes")
    if type(prompt_bytes) is not int or prompt_bytes < 0:
        prompt_bytes = 0

    stars = result.stars if type(result.stars) in (int, float) else None
    if (
        result.status == "success"
        and type(stars) is float
        and stars.is_integer()
    ):
        # P1 stores an already validated on-grid 1..5 rating as float; the
        # shared contract records success ratings as JSON integers. A
        # non-integral value is deliberately left untouched so the strict
        # reader rejects it and the export fails closed.
        stars = int(stars)
    review = result.review if type(result.review) is str and result.review else None
    refs = [ref for ref in result.evidence_refs if isinstance(ref, str)]

    return {
        "task_index": record.task_index,
        "execution_order": record.execution_order,
        "split": record.split,
        "context_sha256": context.context_sha256,
        "user_group": anonymize_user_group(record.user_group),
        "status": result.status,
        "reason": result.reason,
        "prediction_origin": result.prediction_origin,
        "stars": stars,
        "review": review,
        "evidence_refs": refs,
        "diagnostics": {
            "llm_attempts": _attempt_count(llm),
            "repair": _repair_state(llm),
            "prompt_bytes": prompt_bytes,
            "memory": {
                "candidates": _bounded_memory_count(memory.get("candidates", 0)),
                "time_eligible": _bounded_memory_count(
                    memory.get("time_eligible", 0)
                ),
                "selected": _bounded_memory_count(memory.get("selected", 0)),
                "prompt_included": _bounded_memory_count(
                    memory.get("prompt_included", 0)
                ),
                "filtered_injection": _bounded_memory_count(
                    memory.get("filtered_injection", 0)
                ),
                "write": _write_state(memory.get("write", "not_attempted")),
            },
        },
    }


def _budget_spec(task_count: int) -> BudgetSpec:
    return BudgetSpec(
        max_llm_calls_per_task=2,
        max_llm_calls_run=max(4, 2 * task_count),
        max_prompt_bytes_run=400_000,
        max_output_tokens_run=max(4, 2 * task_count) * 700,
    )


def _config_fingerprint(
    *,
    mode: str,
    seed: int,
    max_tasks: int,
    task_count: int,
) -> str:
    config = {
        "exporter": EXPORT_TOOL,
        "contract_schema_version": contract.SCHEMA_VERSION,
        "model_source": MODEL_SOURCE_FAKE,
        "mode": mode,
        "seed": seed,
        "max_tasks_requested": max_tasks,
        "task_count_executed": task_count,
        "execution": "serial_max_workers_1",
        "render_config": {
            field: getattr(DEFAULT_RENDER_CONFIG, field)
            for field in DEFAULT_RENDER_CONFIG.__dataclass_fields__
        },
        "budget": {
            "max_llm_calls_per_task": 2,
            "max_llm_calls_run": max(4, 2 * task_count),
            "max_prompt_bytes_run": 400_000,
            "max_output_tokens_run": max(4, 2 * task_count) * 700,
        },
        "baseline_params": "fixed_p1_defaults",
        "memory_recall_rule": "strictly_earlier_utc_day_only",
    }
    return _sha256_bytes(_canonical_json(config).encode("utf-8"))


def _publish_artifact(output: Path, text: str) -> Path:
    """Write ``.part`` -> revalidate those bytes -> atomic no-overwrite publish.

    Ownership rules:

    - the ``.part`` file is opened with an exclusive create; a pre-existing
      part file is never touched or deleted (``part_file_exists`` refusal),
      and only a part this call actually created is ever cleaned up;
    - the final name only appears through ``os.link`` from this call's own
      part, which is atomic and refuses an existing destination
      (``output_file_exists``);
    - when hard links are unavailable the publish fails closed with
      ``atomic_publish_unavailable`` instead of writing the final file
      non-atomically, so a crash can never leave a half-written final file.
    """
    final = output / ARTIFACT_FILENAME
    part = output / (ARTIFACT_FILENAME + ".part")
    part_owned = False
    try:
        with part.open("x", encoding="utf-8", newline="\n") as handle:
            part_owned = True
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        # Never delete a part file this call did not create.
        raise ExportError("part_file_exists") from exc
    except OSError as exc:
        if part_owned:
            part.unlink(missing_ok=True)
        raise ExportError("artifact_write_failed") from exc
    try:
        payload = part.read_bytes()
        validation = contract.validate_payload(payload)
        if not validation.valid:
            raise ExportError("artifact_contract_violation")
        try:
            os.link(part, final)
        except FileExistsError as exc:
            raise ExportError("output_file_exists") from exc
        except (OSError, NotImplementedError) as exc:
            raise ExportError("atomic_publish_unavailable") from exc
    except OSError as exc:
        raise ExportError("artifact_write_failed") from exc
    finally:
        try:
            part.unlink(missing_ok=True)
        except OSError:
            pass
    return final


def _cleanup_own_output(
    output: Path, owned_names: set[str], own_child_dirs: set[str]
) -> None:
    """Remove only this run's own known artifacts and an empty directory.

    Unknown or late-created files (for example a foreign ``.part`` or a final
    artifact that appeared during the run) are preserved: the directory is
    removed only when it is empty after this run's own artifacts and its own
    synthetic fixture directory are gone. This is never a recursive discard.
    """
    for name in sorted(owned_names):
        try:
            (output / name).unlink(missing_ok=True)
        except OSError:
            pass
    for name in sorted(own_child_dirs):
        candidate = output / name
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
    try:
        output.rmdir()
    except OSError:
        pass


def _execute_and_write(
    output_dir: str | Path,
    *,
    mode: str,
    source_kind: str,
    max_tasks: int,
    seed: int,
    prepare: Callable[[Path], tuple[Sequence[TemporalTaskContext], float, str, str]],
    own_child_dirs: set[str] | None = None,
) -> ExportResult:
    output = Path(output_dir)
    if output.exists():
        raise ExportError("output_dir_exists")
    if not _ignored_output_root(output):
        raise ExportError("output_dir_not_ignored")
    created = False
    completed = False
    owned_names: set[str] = set()
    try:
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ExportError("output_dir_exists") from exc
        except OSError as exc:
            raise ExportError("output_directory_unavailable") from exc
        created = True

        contexts, global_prior, dataset_fp, manifest_fp = prepare(output)
        if len(contexts) < 1:
            raise ExportError("no_tasks")
        task_count = len(contexts)
        config_fp = _config_fingerprint(
            mode=mode, seed=seed, max_tasks=max_tasks, task_count=task_count
        )
        run_id = "stage1-" + _sha256_bytes(
            f"{mode}|seed:{seed}|tasks:{task_count}|{dataset_fp}".encode("utf-8")
        )[:12]
        budget = BudgetLedger(_budget_spec(task_count))
        store = ReliableMemoryStore(run_id)
        fake = ScriptedFakeLLM([grounded_response(4)] * (2 * task_count))
        run = run_reliable_suite(
            list(contexts),
            llm=fake,
            budget=budget,
            global_prior=global_prior,
            memory_store=store,
            run_id=run_id,
            model_source=MODEL_SOURCE_FAKE,
        )
        context_by_index = {context.task_index: context for context in contexts}
        records = [
            document_record(record, context_by_index[record.task_index])
            for record in run.records
        ]
        text = contract.build_document(
            source_kind=source_kind,
            dataset_fingerprint_sha256=dataset_fp,
            manifest_fingerprint_sha256=manifest_fp,
            config_fingerprint_sha256=config_fp,
            records=records,
        )
        published = _publish_artifact(output, text)
        owned_names.add(published.name)
        data = text.encode("utf-8")
        status_counts = {status: 0 for status in contract.STATUSES}
        for record in records:
            status_counts[record["status"]] += 1
        completed = True
        return ExportResult(
            mode=mode,
            source_kind=source_kind,
            record_count=len(records),
            status_counts=status_counts,
            prediction_success_count=status_counts["success"],
            artifact_sha256=_sha256_bytes(data),
            schema_version=contract.SCHEMA_VERSION,
            schema_hash=contract.schema_hash(),
        )
    except ExportError:
        raise
    except contract.ContractError as exc:
        raise ExportError("artifact_contract_violation") from exc
    except (amazon_p0.AmazonP0Error, TemporalManifestError, FileNotFoundError):
        # Verified-gate and file errors stay categorized; the CLI maps them
        # to their own safe codes. The fresh output dir is still cleaned up.
        raise
    except Exception as exc:
        raise ExportError("export_failed") from exc
    finally:
        if created and not completed:
            _cleanup_own_output(output, owned_names, own_child_dirs or set())


def export_synthetic(
    output_dir: str | Path,
    *,
    max_tasks: int = DEFAULT_MAX_TASKS,
    seed: int = FIXED_SEED,
) -> ExportResult:
    """Export the accepted synthetic fixture with a deterministic fake."""
    _validate_params(max_tasks, seed)
    own_child_dirs: set[str] = set()

    def prepare(output: Path):
        fixture = Path(
            tempfile.mkdtemp(prefix="stage1-synthetic-", dir=str(output))
        )
        own_child_dirs.add(fixture.name)
        try:
            provider = build_synthetic_provider(fixture)
            available = provider.contexts_in_execution_order()
            if max_tasks > len(available):
                raise ExportError("max_tasks_exceeds_available")
            contexts = tuple(available[:max_tasks])
            dataset_fp = _sha256_file(fixture / "reviews.json")
            manifest_fp = _sha256_bytes(
                _canonical_json(provider.manifest).encode("utf-8")
            )
            return contexts, SYNTHETIC_GLOBAL_PRIOR, dataset_fp, manifest_fp
        finally:
            shutil.rmtree(fixture, ignore_errors=True)

    return _execute_and_write(
        output_dir,
        mode="synthetic",
        source_kind="fake_on_synthetic",
        max_tasks=max_tasks,
        seed=seed,
        prepare=prepare,
        own_child_dirs=own_child_dirs,
    )


def export_prepared(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    max_tasks: int = DEFAULT_MAX_TASKS,
    seed: int = FIXED_SEED,
) -> ExportResult:
    """Export a local verified prepared directory through every P0 gate."""
    _validate_params(max_tasks, seed)

    def prepare(output: Path):
        dataset = load_verified_prepared_dataset(prepared_dir)
        available = dataset.contexts
        if max_tasks > len(available):
            raise ExportError("max_tasks_exceeds_selected")
        contexts = tuple(available[:max_tasks])
        dataset_fp = _sha256_file(dataset.prepared_dir / "normalized_reviews.jsonl")
        manifest_fp = _sha256_file(dataset.prepared_dir / "manifest.json")
        return contexts, dataset.global_prior, dataset_fp, manifest_fp

    return _execute_and_write(
        output_dir,
        mode="prepared",
        source_kind="fake_on_real_context",
        max_tasks=max_tasks,
        seed=seed,
        prepare=prepare,
    )
