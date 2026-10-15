"""P3 local portfolio demo app: loopback-only HTTP + synthetic fake-only runs.

Honest scope
------------
This is a **local portfolio demo**, not a production service. It serves a
small static Chinese UI and a two-route JSON API over stdlib
``ThreadingHTTPServer`` bound to a loopback address only (127.0.0.1 or ::1).
Every run is offline and fake-only: a hand-written synthetic fixture is
revalidated by the existing P0/P1 provider, and the injected fake callable is
deterministic. No API key, no network client, no real model call, no dataset
asset. The demo never persists artifacts; the browser-side "download report"
button serializes already-received sanitized data locally.

Security boundaries (honest framing)
------------------------------------
- Host header and (when present) Origin are restricted to loopback names; the
  API never sends CORS headers, so cross-site pages can neither read
  ``/api/meta`` nor the run responses.
- State-changing requests (POST /api/run) require a server-issued random CSRF
  token in a custom header (``X-Demo-CSRF``). This is a browser
  same-origin initialization mechanism, **not** a sandbox against other local
  processes; a local process can fetch the token directly. Requests from a
  CLI may omit Origin but must supply the explicit token obtained from
  ``GET /api/meta``.
- Request bodies are strictly bounded (≤16 KiB), require an exact
  ``application/json`` content type and an integer Content-Length; chunked
  transfer, duplicate JSON keys, non-finite numbers, deep nesting and unknown
  fields are rejected with safe category codes.
- Responses contain only safe aggregates, fixed enums and synthetic shapes:
  no tracebacks, no filesystem paths, no prompts, no groundtruth text, no
  secrets, and no reflection of raw client input.
- Inflight work and per-window execution requests are bounded. This is a
  demo, not a claim of complete production-grade security.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import secrets
import socket
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from demo_fixtures import (
    DEMO_APP_SCHEMA_VERSION,
    DEMO_GLOBAL_PRIOR,
    DEMO_MODE_ALL_ID,
    DEMO_MODE_IDS,
    DEMO_MODE_LABELS_ZH,
    DEMO_MODES,
    DEMO_SCENARIOS,
    GT_MARKER,
    SCENARIO_BY_ID,
    STATUS_LABELS_ZH,
    DemoFixtureError,
    DemoScenario,
    FixtureTaskInfo,
    build_demo_provider,
    mode_contract,
    script_for,
)
from reliable_agent import (
    EVIDENCE_ID_PATTERN,
    EVIDENCE_KINDS,
    HONESTY_NOTES,
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_REFUSED,
    STATUS_SUCCESS,
    BudgetLedger,
    ReliableMemoryStore,
    run_reliable_suite,
)
from temporal_context import (
    TEMPORAL_ABLATION_CONTRACTS,
    TemporalAblationMode,
)

LOGGER = logging.getLogger("p3_demo")

ALLOWED_BIND_HOSTS = ("127.0.0.1", "::1")
ALLOWED_LOOPBACK_HOSTNAMES = ("127.0.0.1", "localhost", "::1")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8290

MAX_BODY_BYTES = 16 * 1024
MAX_JSON_DEPTH = 8
MAX_INFLIGHT_DEFAULT = 4
INFLIGHT_WAIT_DEFAULT = 0.5
RATE_LIMIT_DEFAULT = (60, 60.0)
MAX_CONNECTIONS_DEFAULT = 16
MAX_CONNECTIONS_HARD_CAP = 32
CONNECTION_WAIT_DEFAULT = 0.5

# Run-cumulative budget counters used to derive per-task usage deltas.
USAGE_COUNTER_KEYS = (
    "llm_attempts", "llm_successes", "llm_failures", "tool_calls",
    "prompt_bytes_charged", "output_token_budget_charged",
)

WEB_DIR = Path(__file__).resolve().parent / "web_demo"
STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
POST_ROUTES = ("/api/run",)
GET_ROUTES = ("/", "/app.js", "/styles.css", "/favicon.svg", "/api/meta")

ROUTE_LABELS: dict[str, str] = {
    "/": "index",
    "/app.js": "static_app_js",
    "/styles.css": "static_styles_css",
    "/favicon.svg": "static_favicon_svg",
    "/api/meta": "api_meta",
    "/api/run": "api_run",
}

CSP_PAGE = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
CSP_DATA = "default-src 'none'; base-uri 'none'; frame-ancestors 'none'"

SECURITY_HEADERS_BASE = (
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()"),
)

P2_PANEL = {
    "status_zh": "评测集成待验收",
    "real_llm_zh": "真实 LLM 实测未运行",
    "accuracy_zh": "未实测",
    "note_zh": (
        "P2 评测模块仍在独立开发中；本 Demo 不导入 P2 接口，不展示任何未经"
        "验收的评测数字（无 RMSE 等指标）。"
    ),
}

SAFETY_NOTES_ZH = (
    "仅绑定回环地址 127.0.0.1 / ::1；拒绝外部绑定地址与外部 Host/Origin。",
    "POST 仅接受固定枚举 scenario/mode；不接受 user/item/as_of/path/model/URL/代码/上传/密钥。",
    "请求体 ≤16 KiB，必须是 application/json + 整数 Content-Length；拒绝 chunked/重复字段/未知字段/超深 JSON。",
    "响应只有安全聚合与合成 shape；无 traceback、文件路径、prompt 原文、GT 或密钥。",
    "服务器不保存任何产物；下载报告由浏览器用已返回的脱敏数据在本地生成。",
    "CSRF token 是同源初始化机制，不对抗本机其他进程；无 Origin 的 CLI 请求需显式携带 server-issued token。",
    "并发与执行频率有硬上限（inflight 信号量 + 窗口限流）；不宣称生产级完整安全。",
)

PRIVACY_NOTE_ZH = (
    "本演练仅使用手写合成 fixture 与本地 fake callable；响应为脱敏聚合与合成 shape，"
    "不含真实用户数据、真实评论、GT、prompt 原文或文件路径。"
)


# ---------------------------------------------------------------------------
# Safe error type and input validation helpers.
# ---------------------------------------------------------------------------


class DemoAppError(ValueError):
    """Safe, categorized demo failure carrying an HTTP status and code only."""

    def __init__(self, http_status: int, code: str, *, allow: str | None = None):
        self.http_status = int(http_status)
        self.code = str(code)
        self.allow = allow
        super().__init__(self.code)


class _DuplicateJSONKey(ValueError):
    pass


class _NonFiniteJSON(ValueError):
    pass


def _reject_json_constant(name: str) -> Any:
    raise _NonFiniteJSON(name)


def _reject_duplicate_json_keys(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _DuplicateJSONKey(keys)
    return dict(pairs)


def _check_json_depth(value: Any, limit: int) -> None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > limit:
            raise DemoAppError(400, "json_too_deep")
        if isinstance(node, dict):
            for child in node.values():
                stack.append((child, depth + 1))
        elif isinstance(node, list):
            for child in node:
                stack.append((child, depth + 1))


def parse_json_body(raw: bytes) -> dict[str, Any]:
    """Strict single-object JSON parse; every failure is a safe 400 category."""
    if len(raw) > MAX_BODY_BYTES:
        raise DemoAppError(413, "body_too_large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DemoAppError(400, "invalid_encoding") from exc
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except _DuplicateJSONKey as exc:
        raise DemoAppError(400, "duplicate_json_field") from exc
    except _NonFiniteJSON as exc:
        raise DemoAppError(400, "nonfinite_number") from exc
    except RecursionError as exc:
        raise DemoAppError(400, "json_too_deep") from exc
    except (TypeError, ValueError) as exc:
        raise DemoAppError(400, "invalid_json") from exc
    if not isinstance(value, dict):
        raise DemoAppError(400, "not_json_object")
    _check_json_depth(value, MAX_JSON_DEPTH)
    return value


def _scenario_for(scenario_id: Any) -> DemoScenario:
    if isinstance(scenario_id, bool) or not isinstance(scenario_id, str):
        raise DemoAppError(400, "invalid_scenario_type")
    scenario = SCENARIO_BY_ID.get(scenario_id)
    if scenario is None:
        raise DemoAppError(400, "invalid_scenario_value")
    return scenario


def _modes_for(mode_id: Any) -> tuple[TemporalAblationMode, ...]:
    if isinstance(mode_id, bool) or not isinstance(mode_id, str):
        raise DemoAppError(400, "invalid_mode_type")
    if mode_id == DEMO_MODE_ALL_ID:
        return DEMO_MODES
    if mode_id not in DEMO_MODE_IDS:
        raise DemoAppError(400, "invalid_mode_value")
    return (TemporalAblationMode(mode_id),)


def validate_run_request(payload: Mapping[str, Any]) -> tuple[str, tuple[TemporalAblationMode, ...]]:
    """Accept exactly ``{"scenario": enum, "mode": enum}`` and nothing else."""
    unknown = set(payload) - {"scenario", "mode"}
    if unknown:
        raise DemoAppError(400, "unknown_field")
    missing = {"scenario", "mode"} - set(payload)
    if missing:
        raise DemoAppError(400, "missing_field")
    scenario = _scenario_for(payload.get("scenario"))
    modes = _modes_for(payload.get("mode"))
    return scenario.scenario_id, modes


def _validate_port(port: Any) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise DemoAppError(400, "invalid_port")
    return port


# ---------------------------------------------------------------------------
# Bounded concurrency primitives.
# ---------------------------------------------------------------------------


class WindowRateLimiter:
    """Sliding-window limiter with a hard event cap (thread-safe).

    Both bounds are clamped to demo-safe ceilings (max_events ≤ 600,
    window ≤ 3600s); infinite or absurd configurations are rejected.
    """

    MAX_EVENTS_HARD_CAP = 600
    MAX_WINDOW_SECONDS = 3600.0

    def __init__(self, max_events: int, window_seconds: float, *, clock: Callable[[], float] | None = None):
        if (
            isinstance(max_events, bool)
            or not isinstance(max_events, int)
            or not 1 <= max_events <= self.MAX_EVENTS_HARD_CAP
        ):
            raise DemoAppError(400, "invalid_rate_limit")
        if (
            isinstance(window_seconds, bool)
            or not isinstance(window_seconds, (int, float))
            or not 0.0 < float(window_seconds) <= self.MAX_WINDOW_SECONDS
        ):
            raise DemoAppError(400, "invalid_rate_limit")
        self.max_events = int(max_events)
        self.window_seconds = float(window_seconds)
        self._clock = clock if clock is not None else time.monotonic
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = self._clock()
        with self._lock:
            while self._events and now - self._events[0] >= self.window_seconds:
                self._events.popleft()
            if len(self._events) >= self.max_events:
                return False
            self._events.append(now)
            return True


# ---------------------------------------------------------------------------
# Deterministic prompt-recording fake callable (no network, no model).
# ---------------------------------------------------------------------------


class ScriptedDemoLLM:
    """Deterministic fake callable; prompts are recorded in memory only."""

    def __init__(self, items: Sequence[Any]):
        self._items = list(items)
        self.prompts: list[str] = []

    @property
    def remaining(self) -> int:
        return len(self._items)

    def __call__(self, messages, **_kwargs):
        prompt = ""
        if isinstance(messages, list) and messages and isinstance(messages[0], Mapping):
            candidate = messages[0].get("content")
            if isinstance(candidate, str):
                prompt = candidate
        self.prompts.append(prompt)
        if not self._items:
            raise AssertionError("demo script exhausted (contract violation)")
        value = self._items.pop(0)
        if isinstance(value, Exception):
            raise value
        if callable(value):
            return value(prompt)
        return value


# ---------------------------------------------------------------------------
# Safe projection helpers (fixed caps; never echo arbitrary structures).
# ---------------------------------------------------------------------------


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _safe_int_list(value: Any, limit: int = 16) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    output: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        output.append(int(item))
        if len(output) >= limit:
            break
    return output


def _safe_code(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if not re.fullmatch(r"[a-z0-9_]{1,64}", value):
        return None
    return value


def _safe_stars(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not (1.0 <= number <= 5.0):
        return None
    return number


def _safe_kind(value: Any) -> str | None:
    if isinstance(value, str) and value in EVIDENCE_KINDS:
        return value
    return None


def _cap_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def _extract_prompt_rows(prompt_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract evidence/memory JSON lines that actually entered the prompt."""
    evidence: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    for line in prompt_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{") or len(stripped) > 16000:
            continue
        try:
            obj = json.loads(stripped)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        evidence_id = obj.get("evidence_id")
        if isinstance(evidence_id, str) and EVIDENCE_ID_PATTERN.fullmatch(evidence_id):
            if len(evidence) < 16:
                evidence.append(obj)
            continue
        sequence = obj.get("memory_sequence")
        if isinstance(sequence, int) and not isinstance(sequence, bool):
            if len(memory) < 8:
                memory.append(obj)
    return evidence, memory


def _evidence_row_projection(obj: Mapping[str, Any]) -> dict[str, Any]:
    text = _cap_text(obj.get("text"), 260)
    return {
        "evidence_id": str(obj.get("evidence_id")),
        "kind": _safe_kind(obj.get("kind")),
        "event_time": _cap_text(obj.get("event_time"), 40),
        "stars": _safe_stars(obj.get("stars")),
        "text": text,
        "text_capped_for_ui": isinstance(obj.get("text"), str) and text != obj.get("text"),
    }


def _budget_snapshot_projection(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        return {}
    output: dict[str, Any] = {
        key: _safe_int(snapshot.get(key))
        for key in (
            "llm_attempts", "llm_successes", "llm_failures", "tool_calls",
            "prompt_bytes_charged", "output_token_budget_charged",
            "prompt_tokens_estimate",
        )
    }
    actual = snapshot.get("actual_token_usage")
    if isinstance(actual, Mapping):
        output["actual_token_usage"] = {
            "samples": _safe_int(actual.get("samples")),
            "prompt_tokens": _safe_int(actual.get("prompt_tokens")),
            "completion_tokens": _safe_int(actual.get("completion_tokens")),
        }
    else:
        output["actual_token_usage"] = "unknown"
    output["estimated_cost_usd"] = "unknown"
    return output


def _budget_spec_projection(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        return {}
    output: dict[str, Any] = {
        key: _safe_int(spec.get(key))
        for key in (
            "max_llm_calls_per_task", "max_llm_calls_run",
            "max_tool_calls_per_task", "max_tool_calls_run",
            "max_prompt_bytes_per_call", "max_prompt_bytes_run",
            "max_output_tokens_per_call", "max_output_tokens_run",
        )
    }
    deadline = spec.get("deadline_seconds")
    output["deadline_seconds"] = (
        float(deadline)
        if isinstance(deadline, (int, float)) and not isinstance(deadline, bool)
        else None
    )
    return output


def _expected_projection(expected_status: str, expected_reason: str, actual_status: str, actual_reason: str) -> dict[str, Any]:
    matches = actual_status == expected_status and actual_reason == expected_reason
    if expected_status == STATUS_SUCCESS:
        note = "预期合同：严格校验通过的成功结果。"
    else:
        label = STATUS_LABELS_ZH.get(expected_status, expected_status)
        note = f"预期合同：该任务按设计为「{label}」，不是成功预测，UI 不会计为 success。"
    return {
        "status": expected_status,
        "reason": expected_reason,
        "label_zh": STATUS_LABELS_ZH.get(expected_status, expected_status),
        "matches": matches,
        "note_zh": note,
    }


def _task_projection(
    record: Any,
    info: FixtureTaskInfo,
    context: Any,
    expected: Any,
    prompt_by_sha: Mapping[str, str],
    memory_entries: Mapping[int, Any],
    ordinal: int,
    usage_delta: Mapping[str, int],
) -> dict[str, Any]:
    result = record.result
    diagnostics = result.diagnostics if isinstance(result.diagnostics, Mapping) else {}
    rendered_diag = diagnostics.get("rendered") if isinstance(diagnostics, Mapping) else {}
    rendered_diag = rendered_diag if isinstance(rendered_diag, Mapping) else {}
    prompt_sha = rendered_diag.get("prompt_sha256")
    prompt = prompt_by_sha.get(prompt_sha) if isinstance(prompt_sha, str) else None
    evidence_raw, _memory_raw = _extract_prompt_rows(prompt) if prompt else ([], [])
    rendered_evidence = [_evidence_row_projection(obj) for obj in evidence_raw]
    rendered_ids = {row["evidence_id"] for row in rendered_evidence}

    cited_refs = [ref for ref in result.evidence_refs if isinstance(ref, str)]
    by_id_rows = {row["evidence_id"]: row for row in rendered_evidence}
    provenance = diagnostics.get("cited_evidence")
    provenance = provenance if isinstance(provenance, (list, tuple)) else []
    prov_by_id: dict[str, Mapping[str, Any]] = {}
    for row in provenance:
        if isinstance(row, Mapping) and isinstance(row.get("evidence_id"), str):
            prov_by_id[row["evidence_id"]] = row
    cited_registry: list[dict[str, Any]] = []
    for ref in cited_refs:
        prov = prov_by_id.get(ref) or {}
        ui_row = by_id_rows.get(ref) or {}
        text_digest = prov.get("text_sha256")
        cited_registry.append({
            "evidence_id": ref,
            "kind": _safe_kind(prov.get("kind") or ui_row.get("kind")),
            "event_time": _cap_text(prov.get("event_time") or ui_row.get("event_time"), 40),
            "stars": _safe_stars(prov.get("stars", ui_row.get("stars"))),
            "text_sha256_short": (
                text_digest[:16] if isinstance(text_digest, str) else None
            ),
            "text_chars_total": _safe_int(prov.get("text_chars_total")),
            "text_truncated": (
                bool(prov.get("text_truncated"))
                if isinstance(prov.get("text_truncated"), bool)
                else None
            ),
        })

    memory_diag = record.memory if isinstance(record.memory, Mapping) else {}
    selected_sequences = _safe_int_list(memory_diag.get("selected_sequences"))
    included_sequences = _safe_int_list(memory_diag.get("prompt_included_sequences"))
    included_set = set(included_sequences)
    selected_entries: list[dict[str, Any]] = []
    for sequence in selected_sequences:
        entry = memory_entries.get(sequence)
        if entry is None:
            continue
        selected_entries.append({
            "sequence": sequence,
            "stars": _safe_stars(entry.stars),
            "event_time": _cap_text(entry.event_time, 40),
            "text": _cap_text(entry.text, 240),
            "prompt_included": sequence in included_set,
        })

    evidence_counts_source = diagnostics.get("evidence")
    evidence_counts_source = (
        evidence_counts_source if isinstance(evidence_counts_source, Mapping) else {}
    )
    evidence_counts = {
        key: _safe_int(evidence_counts_source.get(key))
        for key in (
            "candidate_rows", "tool_returned_rows", "registered_rows",
            "filtered_empty_rows", "filtered_injection_rows",
            "skipped_by_tool_cap_rows", "tool_text_truncated_rows",
            "tool_payload_bytes", "cited_rows",
        )
    }

    llm_source = diagnostics.get("llm")
    llm_source = llm_source if isinstance(llm_source, Mapping) else {}
    llm_projection = {
        "draft_attempted": bool(llm_source.get("draft_attempted")),
        "repair_requested": bool(llm_source.get("repair_requested")),
        "repair_attempted": bool(llm_source.get("repair_attempted")),
        "repair_reserve_failure": _safe_code(llm_source.get("repair_reserve_failure")),
        "validation_reason_first": _safe_code(llm_source.get("validation_reason_first")),
        "validation_reason_after_repair": _safe_code(llm_source.get("validation_reason_after_repair")),
    }
    failure = diagnostics.get("failure")
    if isinstance(failure, Mapping):
        llm_projection["failure"] = {
            "stage": _safe_code(failure.get("stage")),
            "category": _safe_code(failure.get("category")),
        }

    return {
        "task_ordinal": ordinal + 1,
        "user_label_zh": info.user_label_zh,
        "item_label_zh": info.item_label_zh,
        "target_day": context.target_timestamp[:10] if isinstance(context.target_timestamp, str) else None,
        "actual": {
            "status": result.status,
            "status_label_zh": STATUS_LABELS_ZH.get(result.status, result.status),
            "reason": _safe_code(result.reason),
            "origin": _safe_code(result.prediction_origin),
            "stars": _safe_stars(result.stars),
            "review": _cap_text(result.review, 400) if result.review else None,
            "evidence_refs": cited_refs[:16],
            "simulated": bool(result.simulated),
        },
        "expected": _expected_projection(
            expected.status, expected.reason, result.status, result.reason,
        ),
        "memory": {
            "candidates": _safe_int(memory_diag.get("candidates")),
            "time_eligible": _safe_int(memory_diag.get("time_eligible")),
            "selected": _safe_int(memory_diag.get("selected")),
            "prompt_included": _safe_int(memory_diag.get("prompt_included")),
            "filtered_injection": _safe_int(memory_diag.get("filtered_injection")),
            "write": _safe_code(memory_diag.get("write")),
            "write_reason": _safe_code(memory_diag.get("write_reason")),
            "selected_entries": selected_entries,
            "prompt_included_sequences": included_sequences,
        },
        "budget_after_task": _budget_snapshot_projection(record.budget_after_task),
        "task_usage_delta": {key: int(usage_delta.get(key, 0)) for key in USAGE_COUNTER_KEYS},
        "task_usage_note_zh": (
            "task_usage_delta = 本任务与上一任务累计快照的差值（首任务以 0 为基线）；"
            "budget_after_task 是截至本任务的 run 累计值"
        ),
        "llm": llm_projection,
        "render_summary": {
            "prompt_sha256_short": (
                prompt_sha[:16] if isinstance(prompt_sha, str) and len(prompt_sha) >= 16 else None
            ),
            "prompt_bytes": _safe_int(rendered_diag.get("prompt_bytes")),
            "history_rendered": _safe_int(rendered_diag.get("history_rendered")),
            "reference_rendered": _safe_int(rendered_diag.get("reference_rendered")),
            "history_truncated": _safe_int(rendered_diag.get("history_truncated")),
            "reference_truncated": _safe_int(rendered_diag.get("reference_truncated")),
            "memory_selected": _safe_int(rendered_diag.get("memory_selected")),
            "memory_prompt_included": _safe_int_list(rendered_diag.get("memory_prompt_included")),
            "memory_filtered_injection": _safe_int(rendered_diag.get("memory_filtered_injection")),
            "memory_skipped_by_budget": _safe_int(rendered_diag.get("memory_skipped_by_budget")),
            "stats_rendered": bool(rendered_diag.get("stats_rendered")),
            "stats_included_by_mode": bool(rendered_diag.get("stats_included_by_mode")),
            "prior_rendered": bool(rendered_diag.get("prior_rendered")),
        },
        "evidence_counts": evidence_counts,
        "rendered_evidence": rendered_evidence,
        "rendered_evidence_ids": sorted(rendered_ids),
        "cited_registry": cited_registry,
    }


def _run_one_mode(
    scenario: DemoScenario,
    mode: TemporalAblationMode,
    fixture: Any,
    ordered_contexts: Sequence[Any],
    scope_sha: str,
    run_tag: str,
) -> dict[str, Any]:
    contract = TEMPORAL_ABLATION_CONTRACTS[mode]
    run_id = f"demo-{run_tag}-{mode.value}"
    fake = ScriptedDemoLLM(script_for(scenario))
    budget = BudgetLedger(scenario.budget_spec)
    store = ReliableMemoryStore(run_id) if contract.generated_review_memory else None
    run = run_reliable_suite(
        ordered_contexts,
        llm=fake,
        budget=budget,
        global_prior=DEMO_GLOBAL_PRIOR,
        config=scenario.config,
        memory_store=store,
        run_id=run_id,
        include_trusted_stats=contract.trusted_history_stats,
    )

    prompt_by_sha = {
        hashlib.sha256(prompt.encode("utf-8")).hexdigest(): prompt
        for prompt in fake.prompts
    }
    memory_entries_by_user: dict[str, dict[int, Any]] = {}
    if store is not None:
        for task_index in scenario.task_indexes:
            info = fixture.task_info(task_index)
            if info.user_id in memory_entries_by_user:
                continue
            try:
                recall = store.recall_before(info.user_id, "9999-12-31T23:59:59+00:00", limit=64)
            except Exception:  # noqa: BLE001 - display lookup only, never fatal
                memory_entries_by_user[info.user_id] = {}
                continue
            memory_entries_by_user[info.user_id] = {
                entry.sequence: entry for entry in recall.selected
            }

    expected_by_task = {
        task_index: expectation
        for task_index, expectation in zip(scenario.task_indexes, scenario.expected)
    }
    contexts_by_index = {context.task_index: context for context in ordered_contexts}
    previous_usage: dict[str, int] = {key: 0 for key in USAGE_COUNTER_KEYS}
    tasks: list[dict[str, Any]] = []
    for ordinal, record in enumerate(run.records):
        info = fixture.task_info(record.task_index)
        snapshot = record.budget_after_task if isinstance(record.budget_after_task, Mapping) else {}
        usage_delta: dict[str, int] = {}
        for key in USAGE_COUNTER_KEYS:
            current = _safe_int(snapshot.get(key)) or 0
            usage_delta[key] = max(0, current - previous_usage[key])
            previous_usage[key] = current
        tasks.append(_task_projection(
            record,
            info,
            contexts_by_index[record.task_index],
            expected_by_task[record.task_index],
            prompt_by_sha,
            memory_entries_by_user.get(info.user_id, {}),
            ordinal,
            usage_delta,
        ))

    status_counts = run.summary.get("status_counts") or {}
    success_count = _safe_int(status_counts.get(STATUS_SUCCESS)) or 0
    store_writes = store.counts()["writes"] if store is not None else 0
    expected_writes = success_count if store is not None else 0
    consumed = len(script_for(scenario)) - fake.remaining
    refs_subset = all(
        set(task["actual"]["evidence_refs"]) <= set(task["rendered_evidence_ids"])
        for task in tasks
    )
    budget_total = _budget_snapshot_projection(run.summary.get("budget"))
    delta_sums = {
        key: sum(int((task.get("task_usage_delta") or {}).get(key) or 0) for task in tasks)
        for key in USAGE_COUNTER_KEYS
    }
    delta_matches_total = all(
        delta_sums[key] == (_safe_int(budget_total.get(key)) or 0)
        for key in USAGE_COUNTER_KEYS
    )
    checks = {
        "expected_contract_all_tasks": all(task["expected"]["matches"] for task in tasks),
        "llm_attempts_match_expected": budget.llm_attempts == scenario.expected_llm_attempts,
        "script_consumed_matches_attempts": consumed == budget.llm_attempts,
        "writes_match_successes": store_writes == expected_writes,
        "refs_subset_of_rendered": refs_subset,
        "task_usage_delta_sums_to_total": delta_matches_total,
        "all_results_simulated": all(record.result.simulated is True for record in run.records),
        "task_count_matches_plan": len(run.records) == len(scenario.task_indexes),
    }
    return {
        "mode": {
            "id": mode.value,
            "label_zh": DEMO_MODE_LABELS_ZH[mode],
            "toggles": mode_contract(mode),
            "contract_source": "temporal_context.TEMPORAL_ABLATION_CONTRACTS",
        },
        "simulated": True,
        "context_scope_sha256": scope_sha,
        "run_hash": run.summary.get("records_sha256"),
        "run_id_sha256_16": run.summary.get("run_id_sha256"),
        "model_source": run.summary.get("model_source"),
        "status_counts": {status: _safe_int(status_counts.get(status)) or 0 for status in (
            STATUS_SUCCESS, STATUS_DEGRADED, STATUS_REFUSED, STATUS_FAILED,
        )},
        "contract_passed": all(checks.values()),
        "mode_checks": checks,
        "tasks": tasks,
        "budget_total": budget_total,
        "budget_spec": _budget_spec_projection((run.summary.get("budget") or {}).get("spec")),
        "memory_store": store.counts() if store is not None else None,
    }


def _scenario_projection(scenario: DemoScenario) -> dict[str, Any]:
    return {
        "id": scenario.scenario_id,
        "label_zh": scenario.label_zh,
        "description_zh": scenario.description_zh,
        "task_count": len(scenario.task_indexes),
        "expected_statuses": [expectation.status for expectation in scenario.expected],
        "notes_zh": scenario.notes_zh,
        "budget_spec": _budget_spec_projection({
            "max_llm_calls_per_task": scenario.budget_spec.max_llm_calls_per_task,
            "max_llm_calls_run": scenario.budget_spec.max_llm_calls_run,
            "max_tool_calls_per_task": scenario.budget_spec.max_tool_calls_per_task,
            "max_tool_calls_run": scenario.budget_spec.max_tool_calls_run,
            "max_prompt_bytes_per_call": scenario.budget_spec.max_prompt_bytes_per_call,
            "max_prompt_bytes_run": scenario.budget_spec.max_prompt_bytes_run,
            "max_output_tokens_per_call": scenario.budget_spec.max_output_tokens_per_call,
            "max_output_tokens_run": scenario.budget_spec.max_output_tokens_run,
            "deadline_seconds": scenario.budget_spec.deadline_seconds,
        }),
    }


def _build_comparison(mode_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for entry in mode_results:
        tasks = entry.get("tasks") or []
        memory_prompt_included = sum(
            _safe_int((task.get("memory") or {}).get("prompt_included")) or 0
            for task in tasks
        )
        stats_rendered_tasks = sum(
            1 for task in tasks
            if (task.get("render_summary") or {}).get("stats_rendered")
        )
        rows.append({
            "mode_id": (entry.get("mode") or {}).get("id"),
            "label_zh": (entry.get("mode") or {}).get("label_zh"),
            "generated_memory": (entry.get("mode") or {}).get("toggles", {}).get("generated_memory"),
            "trusted_stats": (entry.get("mode") or {}).get("toggles", {}).get("trusted_stats"),
            "status_counts": entry.get("status_counts"),
            "llm_attempts": (entry.get("budget_total") or {}).get("llm_attempts"),
            "memory_prompt_included_total": memory_prompt_included,
            "stats_rendered_tasks": stats_rendered_tasks,
            "contract_passed": bool(entry.get("contract_passed")),
        })
    scopes = {entry.get("context_scope_sha256") for entry in mode_results}
    return {
        "same_provider_contexts": len(scopes) == 1 and None not in scopes,
        "note_zh": (
            "四种模式复用完全相同的 provider 上下文（真实历史/参考逐字节一致），"
            "仅切换 generated memory 与 trusted stats；合同直接来自 "
            "temporal_context.TEMPORAL_ABLATION_CONTRACTS。"
        ),
        "interpretation_zh": (
            "本表只展示合同与配置差异；数字均为合成 fake 运行计数，"
            "不构成准确率或记忆提升率结论（真实 LLM 实测未运行）。"
        ),
        "rows": rows,
    }


def _assert_no_groundtruth_marker(response: Mapping[str, Any]) -> None:
    try:
        text = json.dumps(response, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise DemoAppError(500, "internal_error") from exc
    if GT_MARKER in text or "GROUNDTRUTH" in text:
        raise DemoAppError(500, "internal_privacy_violation")


def execute_demo(
    scenario_id: str,
    mode_id: str,
    *,
    temp_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one fixed scenario over one or all four fixed modes.

    A brand-new synthetic fixture (own temporary directory), fake callable,
    budget ledger and (for memory modes) memory store are built per request;
    nothing persists across calls.
    """
    scenario = _scenario_for(scenario_id)
    modes = _modes_for(mode_id)
    session_id = "demosession-" + secrets.token_hex(6)
    temp_dir = str(temp_root) if temp_root is not None else None
    with tempfile.TemporaryDirectory(prefix="p3-demo-fixture-", dir=temp_dir) as temp:
        fixture = build_demo_provider(Path(temp) / "fixture")
        contexts_by_index = {
            context.task_index: context
            for context in fixture.contexts_in_execution_order()
        }
        ordered_contexts = tuple(contexts_by_index[i] for i in scenario.task_indexes)
        scope_sha = hashlib.sha256(
            json.dumps([context.context_sha256 for context in ordered_contexts]).encode("utf-8")
        ).hexdigest()
        mode_results = [
            _run_one_mode(
                scenario, mode, fixture, ordered_contexts, scope_sha,
                run_tag=f"{session_id}-{scenario.scenario_id}",
            )
            for mode in modes
        ]
    response = {
        "app": "p3-local-portfolio-demo",
        "schema_version": DEMO_APP_SCHEMA_VERSION,
        "simulated": True,
        "synthetic_session_id": session_id,
        "scenario": _scenario_projection(scenario),
        "modes_run": [mode.value for mode in modes],
        "all_modes_id": DEMO_MODE_ALL_ID,
        "context_scope_sha256": scope_sha,
        "mode_results": mode_results,
        "comparison": _build_comparison(mode_results),
        "p2_panel": dict(P2_PANEL),
        "safety_zh": list(SAFETY_NOTES_ZH),
        "honesty": list(HONESTY_NOTES),
        "privacy_zh": PRIVACY_NOTE_ZH,
    }
    _assert_no_groundtruth_marker(response)
    return response


# ---------------------------------------------------------------------------
# Offline contract check (all scenarios x all modes).
# ---------------------------------------------------------------------------


def run_contract_check(*, temp_root: str | Path | None = None) -> dict[str, Any]:
    """Run every fixed scenario in every mode and verify the fixed contracts."""
    temp_dir = str(temp_root) if temp_root is not None else None
    reports: list[dict[str, Any]] = []
    statuses_seen: set[str] = set()
    all_passed = True
    for scenario in DEMO_SCENARIOS:
        with tempfile.TemporaryDirectory(prefix="p3-demo-check-", dir=temp_dir) as temp:
            fixture = build_demo_provider(Path(temp) / "fixture")
            contexts_by_index = {
                context.task_index: context
                for context in fixture.contexts_in_execution_order()
            }
            ordered_contexts = tuple(contexts_by_index[i] for i in scenario.task_indexes)
            scope_sha = hashlib.sha256(
                json.dumps([context.context_sha256 for context in ordered_contexts]).encode("utf-8")
            ).hexdigest()
            for mode in DEMO_MODES:
                result = _run_one_mode(
                    scenario, mode, fixture, ordered_contexts, scope_sha,
                    run_tag=f"check-{scenario.scenario_id}",
                )
                passed = bool(result["contract_passed"])
                all_passed = all_passed and passed
                for task in result["tasks"]:
                    statuses_seen.add(str((task.get("actual") or {}).get("status")))
                reports.append({
                    "scenario_id": scenario.scenario_id,
                    "mode_id": mode.value,
                    "contract_passed": passed,
                    "status_counts": result["status_counts"],
                    "mode_checks": result["mode_checks"],
                })
    required_statuses = {STATUS_SUCCESS, STATUS_DEGRADED, STATUS_REFUSED, STATUS_FAILED}
    checks = {
        "all_scenario_mode_contracts_passed": all_passed,
        "all_four_statuses_seen": required_statuses <= statuses_seen,
        "scenario_count_full": len(reports) == len(DEMO_SCENARIOS) * len(DEMO_MODES),
    }
    return {
        "contract_passed": all(checks.values()),
        "checks": checks,
        "statuses_seen": sorted(statuses_seen),
        "scenario_reports": reports,
    }


# ---------------------------------------------------------------------------
# HTTP server (loopback-only).
# ---------------------------------------------------------------------------


@dataclass
class DemoServerState:
    csrf_token: str
    runner: Callable[[str, str], dict[str, Any]]
    limiter: WindowRateLimiter
    inflight: threading.BoundedSemaphore
    inflight_wait: float
    max_inflight: int
    connections: threading.BoundedSemaphore
    connection_wait: float
    max_connections: int


def _split_netloc(value: Any) -> tuple[str, int | None] | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > 255 or "@" in candidate:
        return None
    try:
        parsed = urlsplit("//" + candidate)
    except ValueError:
        return None
    try:
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if not host:
        return None
    return host, port


def _is_allowed_host_header(value: Any, server_port: int) -> bool:
    parsed = _split_netloc(value)
    if parsed is None:
        return False
    host, header_port = parsed
    if host not in ALLOWED_LOOPBACK_HOSTNAMES:
        return False
    if header_port is not None and header_port != server_port:
        return False
    return True


def _is_allowed_origin(value: Any, server_port: int) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate or len(candidate) > 255 or "@" in candidate or candidate == "null":
        return False
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return False
    if parsed.scheme != "http":
        return False
    try:
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    if host not in ALLOWED_LOOPBACK_HOSTNAMES:
        return False
    return port == server_port


class DemoRequestHandler(BaseHTTPRequestHandler):
    server_version = "p3-demo"
    sys_version = ""
    timeout = 15
    protocol_version = "HTTP/1.1"

    # -- logging: never raw request lines/paths/headers ------------------

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        return

    def _safe_route_label(self, method: str) -> str:
        try:
            path = urlsplit(self.path or "/").path
        except ValueError:
            return "unmatched"
        if method == "GET" and path in GET_ROUTES:
            return ROUTE_LABELS.get(path, "unmatched")
        if method == "POST" and path in POST_ROUTES:
            return ROUTE_LABELS.get(path, "unmatched")
        return "unmatched"

    # -- response helpers ------------------------------------------------

    def _send_bytes(self, status, body: bytes, content_type: str, *, csp: str, extra_headers=()):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", csp)
        for name, value in SECURITY_HEADERS_BASE:
            self.send_header(name, value)
        for name, value in extra_headers:
            self.send_header(name, value)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if body:
            self.wfile.write(body)

    def _send_json(self, status, payload, extra_headers=(), *, csp: str = CSP_DATA):
        try:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            status = 500
            body = b'{"error":"internal_error","simulated":true}'
        self._send_bytes(status, body, "application/json; charset=utf-8", csp=csp, extra_headers=extra_headers)

    def _send_error_json(self, status, code, extra_headers=()):
        self._send_json(status, {"error": code, "simulated": True}, extra_headers)

    def send_error(self, code, message=None, explain=None):
        mapping = {
            400: "bad_request", 404: "not_found", 405: "method_not_allowed",
            413: "body_too_large", 415: "unsupported_media_type",
            429: "rate_limited", 500: "internal_error", 501: "method_not_allowed",
            503: "server_busy",
        }
        mapped = mapping.get(int(code), "bad_request")
        try:
            self._send_error_json(int(code) if int(code) in mapping else 400, mapped)
        except Exception:  # noqa: BLE001 - last-resort close
            self.close_connection = True

    # -- request entry points --------------------------------------------

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._unsupported_method()

    do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = do_PUT

    def _unsupported_method(self):
        self._send_error_json(405, "method_not_allowed", (("Allow", "GET, POST"),))

    def _dispatch(self, method: str):
        LOGGER.info("request method=%s route=%s", method, self._safe_route_label(method))
        acquired = False
        try:
            server_port = int(self.server.server_address[1])
            if not _is_allowed_host_header(self.headers.get("Host"), server_port):
                self._send_error_json(403, "forbidden_host")
                return
            state: DemoServerState = self.server.state
            acquired = state.inflight.acquire(timeout=state.inflight_wait)
            if not acquired:
                self._send_error_json(503, "server_busy", (("Retry-After", "1"),))
                return
            if method == "GET":
                self._handle_get()
            else:
                self._handle_post()
        except DemoAppError as exc:
            headers = (("Allow", exc.allow),) if exc.allow else ()
            self._send_error_json(exc.http_status, exc.code, headers)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:  # noqa: BLE001 - safe category only, no details
            self._send_error_json(500, "internal_error")
        finally:
            if acquired:
                self.server.state.inflight.release()

    # -- GET routes ------------------------------------------------------

    def _handle_get(self):
        path = urlsplit(self.path or "/").path
        if path in POST_ROUTES:
            raise DemoAppError(405, "method_not_allowed", allow="POST")
        if path in STATIC_FILES:
            filename, content_type = STATIC_FILES[path]
            body = self._read_static(filename)
            csp = CSP_PAGE if content_type.startswith("text/html") else CSP_DATA
            self._send_bytes(200, body, content_type, csp=csp)
            return
        if path == "/api/meta":
            self._send_json(200, self._build_meta())
            return
        raise DemoAppError(404, "not_found")

    def _read_static(self, filename: str) -> bytes:
        # Exact allowlist only: no path joins from request input.
        target = WEB_DIR / filename
        try:
            return target.read_bytes()
        except OSError as exc:
            raise DemoAppError(500, "static_unavailable") from exc

    def _build_meta(self) -> dict[str, Any]:
        state: DemoServerState = self.server.state
        return {
            "app": "p3-local-portfolio-demo",
            "schema_version": DEMO_APP_SCHEMA_VERSION,
            "simulated": True,
            "csrf_token": state.csrf_token,
            "all_modes_id": DEMO_MODE_ALL_ID,
            "scenarios": [
                {
                    "id": scenario.scenario_id,
                    "label_zh": scenario.label_zh,
                    "description_zh": scenario.description_zh,
                    "task_count": len(scenario.task_indexes),
                    "expected_statuses": [e.status for e in scenario.expected],
                }
                for scenario in DEMO_SCENARIOS
            ],
            "modes": [
                {
                    "id": mode.value,
                    "label_zh": DEMO_MODE_LABELS_ZH[mode],
                    **mode_contract(mode),
                }
                for mode in DEMO_MODES
            ] + [{
                "id": DEMO_MODE_ALL_ID,
                "label_zh": "对比全部四模式",
                "generated_memory": None,
                "trusted_stats": None,
            }],
            "status_labels_zh": dict(STATUS_LABELS_ZH),
            "limits": {
                "max_body_bytes": MAX_BODY_BYTES,
                "max_inflight": state.max_inflight,
                "max_connections": state.max_connections,
                "rate_limit": {
                    "max_requests": state.limiter.max_events,
                    "window_seconds": state.limiter.window_seconds,
                },
            },
            "safety_zh": list(SAFETY_NOTES_ZH),
            "honesty": list(HONESTY_NOTES),
            "p2_panel": dict(P2_PANEL),
            "privacy_zh": PRIVACY_NOTE_ZH,
        }

    # -- POST routes -----------------------------------------------------

    def _handle_post(self):
        path = urlsplit(self.path or "/").path
        if path in GET_ROUTES:
            raise DemoAppError(405, "method_not_allowed", allow="GET")
        if path not in POST_ROUTES:
            raise DemoAppError(404, "not_found")
        state: DemoServerState = self.server.state
        self._require_csrf(state)
        if not state.limiter.allow():
            raise DemoAppError(429, "rate_limited")
        if self.headers.get("Transfer-Encoding") is not None:
            raise DemoAppError(400, "unsupported_transfer_encoding")
        content_length = self._content_length()
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise DemoAppError(415, "unsupported_media_type")
        raw = self.rfile.read(content_length)
        if len(raw) != content_length:
            raise DemoAppError(400, "truncated_body")
        payload = parse_json_body(raw)
        scenario_id, _modes = validate_run_request(payload)
        session_id = "demosession-" + secrets.token_hex(6)
        try:
            result = state.runner(scenario_id, payload["mode"])
        except DemoAppError:
            raise
        except DemoFixtureError as exc:
            raise DemoAppError(400, exc.code) from exc
        if not isinstance(result, Mapping):
            raise DemoAppError(500, "internal_error")
        body = dict(result)
        body.setdefault("synthetic_session_id", session_id)
        self._send_json(200, body)

    def _require_csrf(self, state: DemoServerState) -> None:
        origin = self.headers.get("Origin")
        if origin is not None:
            server_port = int(self.server.server_address[1])
            if not _is_allowed_origin(origin, server_port):
                raise DemoAppError(403, "forbidden_origin")
        token = self.headers.get("X-Demo-CSRF")
        if not isinstance(token, str) or not secrets.compare_digest(token, state.csrf_token):
            raise DemoAppError(403, "invalid_csrf")

    def _content_length(self) -> int:
        values = self.headers.get_all("Content-Length") or []
        if len(values) != 1:
            raise DemoAppError(400, "invalid_content_length")
        raw = values[0].strip()
        if not re.fullmatch(r"[0-9]{1,9}", raw):
            raise DemoAppError(400, "invalid_content_length")
        length = int(raw)
        if length > MAX_BODY_BYTES:
            raise DemoAppError(413, "body_too_large")
        return length



class DemoHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a real connection-level pre-thread bound.

    ``ThreadingMixIn`` would spawn one thread per accepted connection, so the
    handler-level inflight semaphore alone cannot stop many empty/slow-header
    connections from exhausting threads. ``process_request`` therefore
    acquires a bounded connection permit *before* any thread is created; over
    the cap the raw socket receives a safe 503 (or is closed) and no thread is
    started. ``handle_error`` never prints tracebacks/paths/request data.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, state, address_family):
        self.address_family = address_family
        self.state = state
        super().__init__(server_address, handler_class)

    def process_request(self, request, client_address):
        state: DemoServerState = self.state
        if not state.connections.acquire(timeout=state.connection_wait):
            self._reject_connection_capacity(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            state.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.state.connections.release()

    def _reject_connection_capacity(self, request) -> None:
        # Raw safe category only; no handler, no thread, no request content.
        body = b'{"error":"server_busy","simulated":true}'
        response = (
            b"HTTP/1.1 503 Service Unavailable\r\n"
            b"Content-Type: application/json; charset=utf-8\r\n"
            b"Connection: close\r\n"
            b"Retry-After: 1\r\n"
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body
        )
        try:
            request.settimeout(2.0)
            request.sendall(response)
        except OSError:
            pass
        finally:
            try:
                request.close()
            except OSError:
                pass

    def handle_error(self, request, client_address):
        # The stdlib default prints a traceback to stderr (exception type,
        # message, path). Emit only a fixed safe category via the logger.
        LOGGER.warning("connection_error category=handler_failure")


def _assert_port_available(host: str, port: int) -> None:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            probe.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        probe.bind((host, port))
    except OSError as exc:
        raise DemoAppError(400, "port_in_use_or_unavailable") from exc
    finally:
        probe.close()


def make_server(
    host: str = DEFAULT_HOST,
    port: int = 0,
    *,
    runner: Callable[[str, str], dict[str, Any]] | None = None,
    csrf_token: str | None = None,
    rate_limit: tuple[int, float] = RATE_LIMIT_DEFAULT,
    max_inflight: int = MAX_INFLIGHT_DEFAULT,
    inflight_wait: float = INFLIGHT_WAIT_DEFAULT,
    max_connections: int = MAX_CONNECTIONS_DEFAULT,
    connection_wait: float = CONNECTION_WAIT_DEFAULT,
) -> Any:
    """Create a loopback-only demo server; port 0 picks an ephemeral port."""
    if host not in ALLOWED_BIND_HOSTS:
        raise DemoAppError(400, "host_not_allowed")
    _validate_port(port)
    if port != 0:
        _assert_port_available(host, port)
    if isinstance(max_inflight, bool) or not isinstance(max_inflight, int) or not 1 <= max_inflight <= 64:
        raise DemoAppError(400, "invalid_inflight_limit")
    if (
        isinstance(inflight_wait, bool)
        or not isinstance(inflight_wait, (int, float))
        or not 0.0 <= float(inflight_wait) <= 30.0
    ):
        raise DemoAppError(400, "invalid_inflight_wait")
    if (
        isinstance(max_connections, bool)
        or not isinstance(max_connections, int)
        or not 1 <= max_connections <= MAX_CONNECTIONS_HARD_CAP
    ):
        raise DemoAppError(400, "invalid_connection_limit")
    if (
        isinstance(connection_wait, bool)
        or not isinstance(connection_wait, (int, float))
        or not 0.0 <= float(connection_wait) <= 30.0
    ):
        raise DemoAppError(400, "invalid_connection_wait")
    token = csrf_token if isinstance(csrf_token, str) and len(csrf_token) >= 16 else secrets.token_hex(24)
    state = DemoServerState(
        csrf_token=token,
        runner=runner if runner is not None else execute_demo,
        limiter=WindowRateLimiter(*rate_limit),
        inflight=threading.BoundedSemaphore(max_inflight),
        inflight_wait=float(inflight_wait),
        max_inflight=max_inflight,
        connections=threading.BoundedSemaphore(max_connections),
        connection_wait=float(connection_wait),
        max_connections=max_connections,
    )
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        return DemoHTTPServer((host, port), DemoRequestHandler, state, family)
    except OSError as exc:
        raise DemoAppError(400, "port_in_use_or_unavailable") from exc


# ---------------------------------------------------------------------------
# HTTP self check for the CLI (no browser needed; server never blocks).
# ---------------------------------------------------------------------------


def _host_header_for(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _origin_for(host: str, port: int) -> str:
    return f"http://{_host_header_for(host, port)}"


def _http_request(
    host: str, port: int, method: str, path: str, *,
    body: bytes | None = None, headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=30)
    try:
        merged = {"Host": _host_header_for(host, port)}
        if headers:
            merged.update(headers)
        connection.request(method, path, body=body, headers=merged)
        response = connection.getresponse()
        return {
            "status": response.status,
            "headers": {name.lower(): value for name, value in response.getheaders()},
            "body": response.read(),
        }
    finally:
        connection.close()


def _http_json_response(host: str, port: int, method: str, path: str, *, body=None, headers=None) -> dict[str, Any]:
    response = _http_request(host, port, method, path, body=body, headers=headers)
    try:
        parsed = json.loads(response["body"].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        parsed = None
    return {"status": response["status"], "headers": response["headers"], "json": parsed}


def run_http_self_check(host: str = DEFAULT_HOST, port: int = 0) -> dict[str, Any]:
    """Start the demo server, hit it over loopback HTTP, then shut it down."""
    server = make_server(host, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    checks: dict[str, Any] = {}
    try:
        actual_port = int(server.server_address[1])
        meta = _http_json_response(host, actual_port, "GET", "/api/meta")
        token = None
        if isinstance(meta["json"], Mapping):
            token = meta["json"].get("csrf_token")
        checks["meta_ok"] = (
            meta["status"] == 200
            and isinstance(token, str)
            and isinstance(meta["json"], Mapping)
            and meta["json"].get("simulated") is True
        )
        index = _http_request(host, actual_port, "GET", "/")
        checks["index_ok"] = index["status"] == 200 and b"SIMULATED" in index["body"]
        checks["index_csp"] = "content-security-policy" in index["headers"]
        checks["no_cors_header"] = "access-control-allow-origin" not in meta["headers"]
        headers = {
            "Content-Type": "application/json",
            "Origin": _origin_for(host, actual_port),
            "X-Demo-CSRF": token if isinstance(token, str) else "",
        }
        run_response = _http_json_response(
            host, actual_port, "POST", "/api/run",
            body=json.dumps({"scenario": "cross_date_success", "mode": DEMO_MODE_ALL_ID}).encode("utf-8"),
            headers=headers,
        )
        run_json = run_response["json"] if isinstance(run_response["json"], Mapping) else {}
        mode_results = run_json.get("mode_results")
        checks["run_ok"] = (
            run_response["status"] == 200
            and run_json.get("simulated") is True
            and isinstance(mode_results, list)
            and len(mode_results) == len(DEMO_MODES)
            and all(entry.get("contract_passed") is True for entry in mode_results if isinstance(entry, Mapping))
        )
        bad_response = _http_json_response(
            host, actual_port, "POST", "/api/run",
            body=b'{"scenario":"not_a_scenario","mode":"No_Memory"}',
            headers=headers,
        )
        bad_json = bad_response["json"] if isinstance(bad_response["json"], Mapping) else {}
        checks["reject_ok"] = bad_response["status"] == 400 and isinstance(bad_json.get("error"), str)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
    checks["contract_passed"] = all(bool(value) for key, value in checks.items() if key != "contract_passed")
    return checks


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P3 local portfolio demo (loopback-only, synthetic fixture, "
            "offline fake callables; no API key, no network, no real model)."
        )
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="loopback bind host: 127.0.0.1 or ::1")
    parser.add_argument(
        "--port", type=int, default=None,
        help="port (serve default 8290; --check default: ephemeral 0)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="run the offline contract check plus a loopback HTTP self-check, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        if args.host not in ALLOWED_BIND_HOSTS:
            print("demo failed: host_not_allowed", file=sys.stderr)
            return 2
        if args.check:
            check_port = _validate_port(0 if args.port is None else args.port)
            contract = run_contract_check()
            if not contract["contract_passed"]:
                print("demo check failed: contract_check_failed", file=sys.stderr)
                return 2
            http_checks = run_http_self_check(args.host, check_port)
            if not http_checks.get("contract_passed"):
                print("demo check failed: http_self_check_failed", file=sys.stderr)
                return 2
            print(
                "demo check ok: "
                f"scenarios={len(DEMO_SCENARIOS)} modes={len(DEMO_MODES)} "
                f"statuses={','.join(contract['statuses_seen'])} http=ok"
            )
            return 0
        serve_port = _validate_port(DEFAULT_PORT if args.port is None else args.port)
        server = make_server(args.host, serve_port)
        actual_port = int(server.server_address[1])
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
        print(
            "P3 demo (simulated, offline fake-only) at "
            f"{_origin_for(args.host, actual_port)}/ — Ctrl+C to stop"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except DemoAppError as exc:
        print(f"demo failed: {exc.code}", file=sys.stderr)
        return 2
    except OSError:
        print("demo failed: port_in_use_or_unavailable", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    except Exception:  # noqa: BLE001 - safe category only
        print("demo failed: internal_error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
