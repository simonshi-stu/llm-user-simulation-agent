"""Mirror of the canonical ``SimulationRunV1`` result contract (version 1).

Canonical source: https://github.com/simonshi-stu/amazon-instruments-evaluation-agent
at ``amazon_eval_agent/contracts/simulation_run_v1.py``. This mirror is
maintained by the user-simulation project so that export-side validation and
the evaluation-side reader agree byte for byte; ``schema_hash()`` must equal
the canonical value (a cross-repo test pins this). Do not edit the shared core
independently of the canonical file: changes require a contract version bump
and a re-mirrored copy.

Export-side additions in this mirror (not part of the canonical reader):
``build_document`` assembles a canonical artifact text and validates it
through the same strict reader before it can be written.

Safety semantics implemented here (shared core):

- Resource gates run before any JSON parsing: string/bytes/bytearray inputs are
  length-gated before any encoding or copy, artifacts above 1 MiB are rejected
  by byte preflight, parsing rejects duplicate JSON keys, non-finite numbers
  and deeply nested documents (parse-time ``RecursionError`` maps to
  ``json_too_deep``), and documents above 500 records are rejected before any
  record-level validation runs.
- The schema is closed: unknown and missing fields, wrong types, booleans used
  as numbers, out-of-range or huge integers, malformed fingerprints, malformed
  evidence refs and status-inconsistent combinations are rejected with stable
  safe error codes. Field values are never echoed into error output.
- Status matrix: ``success`` is an integer 1..5 with a non-empty (after
  stripping) bounded review and 1..16 unique evidence refs; ``degraded`` is a
  continuous 1..5 rating from a labeled statistical baseline with ``review``
  null; ``refused`` and ``failed`` carry no rating and no review.
  ``prediction_origin`` must match the status.
- Diagnostic invariants prove **internal record consistency only**, never that
  an external model call or fingerprint is real: a first-pass success has
  exactly one draft attempt and no repair request; a repaired success has
  exactly two attempts with ``repair=attempted``; ``refused`` has zero
  attempts; ``failed`` with ``llm_transport_error``/``invalid_llm_return``
  requires at least one attempt (``internal_error`` may legitimately have
  zero); ``degraded`` follows a labeled statistical cycle (draft attempt plus
  failed-or-sent repair, never a model success); the memory funnel keeps
  ``candidates >= time_eligible >= selected >= prompt_included`` and
  ``filtered_injection + prompt_included <= selected``; evidence refs are
  unique so one row cannot be counted twice; a success review must be UTF-8
  encodable after parsing (a JSON-escaped lone surrogate is rejected).
- ``source.kind`` is a caller declaration. A validated artifact is **not**
  evidence that a real model ran: reading this module never turns a
  ``real_model_declared`` declaration into a verified measurement, and no
  accuracy metric exists here.
- ``user_group`` is a deterministic pseudonym label kept stable for pairing;
  it is not an irreversibility or anonymity guarantee. ``review`` is generated
  free text and may contain arbitrary words: the closed schema excludes
  groundtruth/raw prompt/key/path *fields*, but it performs no content-level
  sanitization and makes no complete-anonymization claim.

Safe error codes: artifact_type_invalid, artifact_too_large,
artifact_encoding_error, json_syntax_error, json_duplicate_field,
json_nonfinite_number, json_too_deep, top_level_not_object, unknown_field,
missing_field, invalid_type, schema_version_mismatch, simulated_not_true,
invalid_enum, invalid_sha256, invalid_user_group, invalid_integer,
value_out_of_bounds, invalid_stars, invalid_review, invalid_evidence_ref,
duplicate_evidence_ref, status_reason_mismatch, status_origin_mismatch,
diagnostic_attempts_mismatch, memory_funnel_mismatch, write_status_mismatch,
record_count_mismatch, too_many_records, empty_records,
duplicate_task_identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "simulation_run.v1"
CONTRACT_VERSION = 1
CANONICAL_REPO = "https://github.com/simonshi-stu/amazon-instruments-evaluation-agent"
CANONICAL_PATH = "amazon_eval_agent/contracts/simulation_run_v1.py"
MIRROR_NOTE = (
    "mirrored copy maintained by the user-simulation project; schema_hash() "
    "must equal the canonical value"
)

# Resource gates and bounded fields (fixed for contract version 1).
MAX_ARTIFACT_BYTES = 1024 * 1024
MAX_RECORDS = 500
MAX_ARTIFACT_DEPTH = 10
MAX_REVIEW_CHARS = 8000
MAX_EVIDENCE_REFS = 16
MAX_BOUNDED_INT = 2**31 - 1
MAX_PROMPT_BYTES_FIELD = 65536
MAX_MEMORY_COUNT = 64
MAX_LLM_ATTEMPTS_PER_TASK = 2
MAX_ERRORS_REPORTED = 32

# Fixed enums: no caller free text is accepted anywhere in these fields.
SOURCE_KINDS = ("fake_on_synthetic", "fake_on_real_context", "real_model_declared")
FAKE_SOURCE_KINDS = ("fake_on_synthetic", "fake_on_real_context")
PRODUCERS = ("llm-user-simulation-agent",)
MODEL_SOURCES = ("offline_injected_fake_callable",)
STATUSES = ("success", "degraded", "refused", "failed")
SPLITS = ("train", "validation", "test")
ORIGINS = ("llm", "statistical_baseline", "none")
REPAIR_STATES = ("not_requested", "requested", "attempted", "reserve_failed")
WRITE_STATES = ("not_attempted", "skipped", "written", "rejected")

REASONS_BY_STATUS: Mapping[str, tuple[str, ...]] = {
    "success": ("validated_model_output", "validated_model_output_repaired"),
    "degraded": ("statistical_baseline_fallback",),
    "refused": (
        "insufficient_evidence",
        "no_rendered_evidence",
        "budget_exhausted",
        "deadline_exceeded",
        "prompt_budget_exceeded",
        "output_token_limit_exceeded",
    ),
    "failed": ("llm_transport_error", "invalid_llm_return", "internal_error"),
}
ORIGIN_BY_STATUS: Mapping[str, str] = {
    "success": "llm",
    "degraded": "statistical_baseline",
    "refused": "none",
    "failed": "none",
}
WRITE_STATES_BY_STATUS: Mapping[str, tuple[str, ...]] = {
    "success": ("written", "rejected", "not_attempted"),
    "degraded": ("skipped", "not_attempted"),
    "refused": ("skipped", "not_attempted"),
    "failed": ("skipped", "not_attempted"),
}

# Internal-consistency invariants (self-consistency only, never external
# truth). They are part of the schema description so both mirrors agree.
DIAGNOSTIC_INVARIANTS: tuple[str, ...] = (
    "success_first_pass_attempts_exact",
    "success_repaired_attempts_and_repair_exact",
    "repair_state_attempts_consistent",
    "refused_zero_attempts",
    "failed_call_error_requires_attempt",
    "degraded_statistical_cycle",
    "memory_funnel_ordering",
    "memory_filtered_plus_included_bound",
    "evidence_refs_unique",
    "review_utf8_encodable",
)

TOP_LEVEL_FIELDS = ("schema_version", "source", "simulated", "record_count", "records")
SOURCE_FIELDS = (
    "producer",
    "kind",
    "model_source",
    "dataset_fingerprint_sha256",
    "manifest_fingerprint_sha256",
    "config_fingerprint_sha256",
)
RECORD_FIELDS = (
    "task_index",
    "execution_order",
    "split",
    "context_sha256",
    "user_group",
    "status",
    "reason",
    "prediction_origin",
    "stars",
    "review",
    "evidence_refs",
    "diagnostics",
)
DIAGNOSTIC_FIELDS = ("llm_attempts", "repair", "prompt_bytes", "memory")
MEMORY_FIELDS = (
    "candidates",
    "time_eligible",
    "selected",
    "prompt_included",
    "filtered_injection",
    "write",
)

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_USER_GROUP_RE = re.compile(r"\Aug_[0-9a-f]{16}\Z")
_EVIDENCE_ID_RE = re.compile(r"\Aev_[0-9a-f]{16}\Z")


def _schema_spec() -> dict[str, Any]:
    """Deterministic description of the shared schema (hash input)."""
    return {
        "contract_version": CONTRACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "top_level_fields": list(TOP_LEVEL_FIELDS),
        "source_fields": list(SOURCE_FIELDS),
        "record_fields": list(RECORD_FIELDS),
        "diagnostic_fields": list(DIAGNOSTIC_FIELDS),
        "memory_fields": list(MEMORY_FIELDS),
        "source_kinds": list(SOURCE_KINDS),
        "fake_source_kinds": list(FAKE_SOURCE_KINDS),
        "producers": list(PRODUCERS),
        "model_sources": list(MODEL_SOURCES),
        "statuses": list(STATUSES),
        "splits": list(SPLITS),
        "origins": list(ORIGINS),
        "repair_states": list(REPAIR_STATES),
        "write_states": list(WRITE_STATES),
        "reasons_by_status": {
            key: list(value) for key, value in sorted(REASONS_BY_STATUS.items())
        },
        "origin_by_status": dict(sorted(ORIGIN_BY_STATUS.items())),
        "write_states_by_status": {
            key: list(value) for key, value in sorted(WRITE_STATES_BY_STATUS.items())
        },
        "diagnostic_invariants": list(DIAGNOSTIC_INVARIANTS),
        "limits": {
            "max_artifact_bytes": MAX_ARTIFACT_BYTES,
            "max_records": MAX_RECORDS,
            "max_artifact_depth": MAX_ARTIFACT_DEPTH,
            "max_review_chars": MAX_REVIEW_CHARS,
            "max_evidence_refs": MAX_EVIDENCE_REFS,
            "max_bounded_int": MAX_BOUNDED_INT,
            "max_prompt_bytes_field": MAX_PROMPT_BYTES_FIELD,
            "max_memory_count": MAX_MEMORY_COUNT,
            "max_llm_attempts_per_task": MAX_LLM_ATTEMPTS_PER_TASK,
        },
    }


def schema_hash() -> str:
    """Stable hash of the shared schema; the mirror must report the same value."""
    payload = json.dumps(
        _schema_spec(), sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


class ContractError(ValueError):
    """Safe, categorized contract failure; carries stable codes only."""

    def __init__(self, code: str, codes: tuple[str, ...] = ()):
        self.code = code
        self.codes = tuple(codes) if codes else (code,)
        super().__init__(code)


@dataclass(frozen=True)
class SimulationRunV1:
    """A validated ``simulation_run.v1`` artifact (shape-checked values only)."""

    schema_version: str
    source_kind: str
    producer: str
    model_source: str
    dataset_fingerprint_sha256: str
    manifest_fingerprint_sha256: str
    config_fingerprint_sha256: str
    record_count: int
    status_counts: Mapping[str, int]
    prediction_success_count: int
    all_predictions_succeeded: bool
    records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ArtifactValidation:
    """Outcome of one ``validate_payload`` call (never raises for bad input)."""

    valid: bool
    errors: tuple[str, ...]
    document: SimulationRunV1 | None


class _ErrorCollector:
    """Bounded, de-duplicated set of safe error codes."""

    def __init__(self) -> None:
        self._errors: list[str] = []

    def add(self, code: str) -> None:
        if code not in self._errors and len(self._errors) < MAX_ERRORS_REPORTED:
            self._errors.append(code)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(self._errors)


class _DuplicateJSONField(ValueError):
    pass


class _NonFiniteJSONConstant(ValueError):
    pass


def _pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise _DuplicateJSONField("duplicate_json_field")
        seen.add(key)
    return dict(pairs)


def _reject_constant(_name: str) -> Any:
    raise _NonFiniteJSONConstant("json_nonfinite_number")


def _json_loads(text: str) -> Any:
    """Single strict JSON entry point (tests spy on this before parsing)."""
    return json.loads(
        text,
        object_pairs_hook=_pairs_hook,
        parse_constant=_reject_constant,
    )


def _structure_gate(value: Any) -> tuple[bool, bool]:
    """Iterative depth + non-finite walk; returns ``(depth_ok, nonfinite)``."""
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_ARTIFACT_DEPTH:
            return False, False
        if type(node) is float and not math.isfinite(node):
            return True, True
        if isinstance(node, dict):
            for child in node.values():
                stack.append((child, depth + 1))
        elif isinstance(node, list):
            for child in node:
                stack.append((child, depth + 1))
    return True, False


def _invalid(code: str) -> ArtifactValidation:
    return ArtifactValidation(valid=False, errors=(code,), document=None)


def _bounded_int(value: Any) -> bool:
    return type(value) is int and 0 <= value <= MAX_BOUNDED_INT


def _star_ok(value: Any) -> bool:
    """Continuous 1..5 star check with an integer range gate before any cast.

    Huge integers are rejected by comparison, never by ``float(value)``, so
    hostile values cannot raise ``OverflowError``.
    """
    if type(value) is int:
        return 1 <= value <= 5
    if type(value) is float:
        return math.isfinite(value) and 1.0 <= value <= 5.0
    return False


def _sha256_ok(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _check_exact_keys(
    obj: Mapping[str, Any], expected: tuple[str, ...], errors: _ErrorCollector
) -> None:
    expected_set = set(expected)
    for key in obj:
        if key not in expected_set:
            errors.add("unknown_field")
    for key in expected:
        if key not in obj:
            errors.add("missing_field")


def _validate_source(source: Any, errors: _ErrorCollector) -> None:
    if not isinstance(source, dict):
        errors.add("invalid_type")
        return
    _check_exact_keys(source, SOURCE_FIELDS, errors)
    if source.get("producer") not in PRODUCERS:
        errors.add("invalid_enum")
    if source.get("kind") not in SOURCE_KINDS:
        errors.add("invalid_enum")
    if source.get("model_source") not in MODEL_SOURCES:
        errors.add("invalid_enum")
    for field in (
        "dataset_fingerprint_sha256",
        "manifest_fingerprint_sha256",
        "config_fingerprint_sha256",
    ):
        if not _sha256_ok(source.get(field)):
            errors.add("invalid_sha256")


def _validate_record(record: Mapping[str, Any], errors: _ErrorCollector) -> None:
    _check_exact_keys(record, RECORD_FIELDS, errors)
    for field in ("task_index", "execution_order"):
        if not _bounded_int(record.get(field)):
            errors.add("invalid_integer")
    if record.get("split") not in SPLITS:
        errors.add("invalid_enum")
    if not _sha256_ok(record.get("context_sha256")):
        errors.add("invalid_sha256")
    user_group = record.get("user_group")
    if not (
        isinstance(user_group, str) and _USER_GROUP_RE.fullmatch(user_group) is not None
    ):
        errors.add("invalid_user_group")

    status = record.get("status")
    if status not in STATUSES:
        errors.add("invalid_enum")
        return
    reason = record.get("reason")
    if reason not in REASONS_BY_STATUS[status]:
        errors.add("status_reason_mismatch")
    if record.get("prediction_origin") != ORIGIN_BY_STATUS[status]:
        errors.add("status_origin_mismatch")

    stars = record.get("stars")
    if status == "success":
        stars_ok = type(stars) is int and 1 <= stars <= 5
    elif status == "degraded":
        stars_ok = _star_ok(stars)
    else:
        stars_ok = stars is None
    if not stars_ok:
        errors.add("invalid_stars")

    review = record.get("review")
    if status == "success":
        review_ok = (
            isinstance(review, str)
            and bool(review.strip())
            and len(review) <= MAX_REVIEW_CHARS
        )
        if review_ok:
            try:
                review.encode("utf-8")
            except UnicodeEncodeError:
                # A JSON-escaped lone surrogate parses into an unencodable
                # string; reject it here so no consumer can trip over it.
                review_ok = False
    else:
        review_ok = review is None
    if not review_ok:
        errors.add("invalid_review")

    refs = record.get("evidence_refs")
    if not isinstance(refs, list):
        errors.add("invalid_evidence_ref")
    else:
        if status == "success":
            count_ok = 1 <= len(refs) <= MAX_EVIDENCE_REFS
        elif status == "degraded":
            count_ok = 0 <= len(refs) <= MAX_EVIDENCE_REFS
        else:
            count_ok = len(refs) == 0
        if not count_ok:
            errors.add("invalid_evidence_ref")
        seen_refs: set[str] = set()
        for ref in refs:
            if not (
                isinstance(ref, str) and _EVIDENCE_ID_RE.fullmatch(ref) is not None
            ):
                errors.add("invalid_evidence_ref")
                continue
            if ref in seen_refs:
                errors.add("duplicate_evidence_ref")
            seen_refs.add(ref)

    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, dict):
        errors.add("invalid_type")
        return
    _check_exact_keys(diagnostics, DIAGNOSTIC_FIELDS, errors)
    attempts = diagnostics.get("llm_attempts")
    attempts_ok = _bounded_int(attempts)
    if not attempts_ok:
        errors.add("invalid_integer")
    elif attempts > MAX_LLM_ATTEMPTS_PER_TASK:
        errors.add("value_out_of_bounds")
        attempts_ok = False
    repair = diagnostics.get("repair")
    repair_ok = repair in REPAIR_STATES
    if not repair_ok:
        errors.add("invalid_enum")
    if attempts_ok and repair_ok:
        _validate_attempt_repair_invariants(
            status, reason, attempts, repair, errors
        )
    prompt_bytes = diagnostics.get("prompt_bytes")
    if not _bounded_int(prompt_bytes):
        errors.add("invalid_integer")
    elif prompt_bytes > MAX_PROMPT_BYTES_FIELD:
        errors.add("value_out_of_bounds")

    memory = diagnostics.get("memory")
    if not isinstance(memory, dict):
        errors.add("invalid_type")
        return
    _check_exact_keys(memory, MEMORY_FIELDS, errors)
    counts: dict[str, int | None] = {}
    for field in MEMORY_FIELDS[:-1]:
        value = memory.get(field)
        if not _bounded_int(value):
            errors.add("invalid_integer")
            counts[field] = None
        elif value > MAX_MEMORY_COUNT:
            errors.add("value_out_of_bounds")
            counts[field] = None
        else:
            counts[field] = value
    if all(value is not None for value in counts.values()):
        candidates = counts["candidates"]
        time_eligible = counts["time_eligible"]
        selected = counts["selected"]
        prompt_included = counts["prompt_included"]
        filtered = counts["filtered_injection"]
        if not (candidates >= time_eligible >= selected >= prompt_included):
            errors.add("memory_funnel_mismatch")
        if filtered + prompt_included > selected:
            errors.add("memory_funnel_mismatch")
    write = memory.get("write")
    if write not in WRITE_STATES:
        errors.add("invalid_enum")
    elif write not in WRITE_STATES_BY_STATUS[status]:
        errors.add("write_status_mismatch")


def _validate_attempt_repair_invariants(
    status: str,
    reason: Any,
    attempts: int,
    repair: str,
    errors: _ErrorCollector,
) -> None:
    """Internal attempt/repair consistency (self-consistency only).

    These invariants prove the diagnostic counters describe a state the
    workflow could actually reach; they are never evidence that an external
    model call happened.
    """
    if repair == "attempted" and attempts != 2:
        errors.add("diagnostic_attempts_mismatch")
    elif repair in ("requested", "reserve_failed") and attempts != 1:
        errors.add("diagnostic_attempts_mismatch")
    elif repair == "not_requested" and attempts > 1:
        errors.add("diagnostic_attempts_mismatch")

    if status == "success":
        if reason == "validated_model_output" and not (
            attempts == 1 and repair == "not_requested"
        ):
            errors.add("diagnostic_attempts_mismatch")
        elif reason == "validated_model_output_repaired" and not (
            attempts == 2 and repair == "attempted"
        ):
            errors.add("diagnostic_attempts_mismatch")
    elif status == "refused":
        if attempts != 0 or repair != "not_requested":
            errors.add("diagnostic_attempts_mismatch")
    elif status == "failed":
        # A call-level failure (transport / invalid return) means an attempt
        # was made; internal_error may legitimately occur before any call.
        if (
            reason in ("llm_transport_error", "invalid_llm_return")
            and attempts < 1
        ):
            errors.add("diagnostic_attempts_mismatch")
    elif status == "degraded":
        cycle_ok = (attempts == 1 and repair == "reserve_failed") or (
            attempts == 2 and repair == "attempted"
        )
        if not cycle_ok:
            errors.add("diagnostic_attempts_mismatch")


def validate_payload(payload: Any) -> ArtifactValidation:
    """Validate raw artifact bytes/text; never raises for hostile input.

    Order matters: input length is gated before any encoding or copy, byte
    size is checked before decoding, strict JSON parsing maps hostile
    constructs (duplicate keys, non-finite numbers, parse-time recursion) to
    safe codes, the depth walk is iterative, and documents above 500 records
    are rejected **before** any record-level validation runs. A failed
    validation never produces a ``document`` (no downstream scoring can be
    requested from an invalid artifact).
    """
    if isinstance(payload, str):
        if len(payload) > MAX_ARTIFACT_BYTES:
            # UTF-8 encodes every character to at least one byte, so an
            # over-limit string is rejected before any encoding copy.
            return _invalid("artifact_too_large")
        try:
            data = payload.encode("utf-8")
        except UnicodeEncodeError:
            return _invalid("artifact_encoding_error")
    elif isinstance(payload, (bytes, bytearray)):
        if len(payload) > MAX_ARTIFACT_BYTES:
            return _invalid("artifact_too_large")
        data = bytes(payload)
    else:
        return _invalid("artifact_type_invalid")
    if len(data) > MAX_ARTIFACT_BYTES:
        return _invalid("artifact_too_large")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return _invalid("artifact_encoding_error")
    try:
        document = _json_loads(text)
    except _DuplicateJSONField:
        return _invalid("json_duplicate_field")
    except _NonFiniteJSONConstant:
        return _invalid("json_nonfinite_number")
    except RecursionError:
        return _invalid("json_too_deep")
    except json.JSONDecodeError:
        return _invalid("json_syntax_error")
    except (ValueError, OverflowError):
        return _invalid("json_syntax_error")
    except Exception:  # noqa: BLE001 - hostile input must never raise out
        return _invalid("json_syntax_error")

    depth_ok, nonfinite = _structure_gate(document)
    if not depth_ok:
        return _invalid("json_too_deep")
    if nonfinite:
        return _invalid("json_nonfinite_number")

    if not isinstance(document, dict):
        return _invalid("top_level_not_object")

    errors = _ErrorCollector()
    _check_exact_keys(document, TOP_LEVEL_FIELDS, errors)
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.add("schema_version_mismatch")
    if document.get("simulated") is not True:
        errors.add("simulated_not_true")
    _validate_source(document.get("source"), errors)

    records = document.get("records")
    if not isinstance(records, list):
        errors.add("invalid_type")
    elif len(records) > MAX_RECORDS:
        # Early gate: the record-count limit rejects the artifact before any
        # record-level validation runs (no per-record loop past the cap).
        errors.add("too_many_records")
    else:
        record_count = document.get("record_count")
        if not _bounded_int(record_count):
            errors.add("invalid_type")
        elif record_count != len(records):
            errors.add("record_count_mismatch")
        if len(records) == 0:
            errors.add("empty_records")
        else:
            seen_indices: set[int] = set()
            seen_orders: set[int] = set()
            for record in records:
                if not isinstance(record, dict):
                    errors.add("invalid_type")
                    continue
                _validate_record(record, errors)
                index = record.get("task_index")
                if _bounded_int(index):
                    if index in seen_indices:
                        errors.add("duplicate_task_identity")
                    seen_indices.add(index)
                order = record.get("execution_order")
                if _bounded_int(order):
                    if order in seen_orders:
                        errors.add("duplicate_task_identity")
                    seen_orders.add(order)

    if errors.errors:
        return ArtifactValidation(valid=False, errors=errors.errors, document=None)

    source = document["source"]
    status_counts = {status: 0 for status in STATUSES}
    for record in records:
        status_counts[record["status"]] += 1
    success_count = status_counts["success"]
    validated = SimulationRunV1(
        schema_version=SCHEMA_VERSION,
        source_kind=source["kind"],
        producer=source["producer"],
        model_source=source["model_source"],
        dataset_fingerprint_sha256=source["dataset_fingerprint_sha256"],
        manifest_fingerprint_sha256=source["manifest_fingerprint_sha256"],
        config_fingerprint_sha256=source["config_fingerprint_sha256"],
        record_count=len(records),
        status_counts=status_counts,
        prediction_success_count=success_count,
        all_predictions_succeeded=success_count == len(records),
        records=tuple(records),
    )
    return ArtifactValidation(valid=True, errors=(), document=validated)


# ---------------------------------------------------------------------------
# Export-side additions (mirror only): build canonical artifact text.
# ---------------------------------------------------------------------------


def build_document(
    *,
    source_kind: str,
    dataset_fingerprint_sha256: str,
    manifest_fingerprint_sha256: str,
    config_fingerprint_sha256: str,
    records: Sequence[Mapping[str, Any]],
    producer: str = PRODUCERS[0],
    model_source: str = MODEL_SOURCES[0],
    simulated: bool = True,
) -> str:
    """Build and self-validate the canonical artifact text.

    The document is assembled only from plain bounded values, serialized
    canonically, and passed through the same strict reader used by the
    evaluation project. Any contract violation raises :class:`ContractError`
    with safe codes, so a caller can never write an artifact the reader would
    reject.
    """
    document = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "producer": producer,
            "kind": source_kind,
            "model_source": model_source,
            "dataset_fingerprint_sha256": dataset_fingerprint_sha256,
            "manifest_fingerprint_sha256": manifest_fingerprint_sha256,
            "config_fingerprint_sha256": config_fingerprint_sha256,
        },
        "simulated": simulated,
        "record_count": len(records),
        "records": [dict(record) for record in records],
    }
    text = (
        json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )
    validation = validate_payload(text)
    if not validation.valid:
        raise ContractError("artifact_contract_violation", validation.errors)
    return text
